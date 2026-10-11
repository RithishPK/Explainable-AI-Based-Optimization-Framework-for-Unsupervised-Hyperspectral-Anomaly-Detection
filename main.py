"""
Phase 2 - Adaptive Lambda Optimization Framework
=================================================

Extends the original Phase 1 `main.py` with an Adaptive Lambda
Optimization Framework while preserving every other stage of the pipeline
untouched:

    1. Dataset loading      -> unchanged (dataset.coast.get_data)
    2. OPBS band selection  -> unchanged (select_bands.OPBS)
    3. Z-score normalization -> unchanged (utils.ZScoreNorm)
    4. MSNet model          -> unchanged (model.MSNet)
    5. Separation Training   -> extended to optionally use AdaptiveLambda
    6. ROC evaluation        -> unchanged (metric.roc_auc)

Backward compatibility
-----------------------
Set `adaptive_lambda_enabled = False` below to reproduce the exact Phase 1
behaviour: `lmda` stays fixed at `lambda_base` for the whole run,
`separation_training` is called with `adaptive_lambda=None`, and no
lambda-related computation ever runs. See CHANGES.md for a full
explanation of every modification.
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch

from dataset import coast
from model import MSNet
import select_bands
import utils
import metric
from SeT import TotalLoss, Mask, separation_training
from adaptive_lambda import AdaptiveLambda, ResidualLRScheduler, VALID_STRATEGIES


# ======================================================================
# Settings (Phase 1 settings are unchanged; Phase 2 settings are new)
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- v14: determinism. Without this, 'seeds' are decorative and the
# mean +/- std reported in the paper is not reproducible.
seed = 0
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Phase 1 settings (unchanged) ------------------------------------
lmda = 1e-3            # kept for backward compatibility; used as lambda_base
num_bs = 64
num_layers = 3
lr = 1e-3
epochs = 150
output_iter = 5        # FIXED A PRIORI. The reported detection map is the one
                       # captured at this iteration, never max(history). Selecting
                       # by peak AUC requires ground truth and is model-selection
                       # leakage. main.py itself is clean on this point; the Phase 1
                       # REPORT was not - it quoted 0.9975 (the peak) while
                       # output_iter=5 gives 0.9972. Report history[output_iter-1].
max_iter = 10
data_norm = True
Net = MSNet
net_kwargs = dict()
net_kwargs['num_layers'] = num_layers

# --- Phase 2 settings: Adaptive Lambda Optimization Framework --------
adaptive_lambda_enabled = True   # False (or strategy='fixed') => exact Phase 1 behaviour
strategy = 'xai_alignment_bg'     # v14: the project's primary XAI contribution.
                                  # v13 defaulted to 'residual', which silently
                                  # contradicted the stated contribution in every
                                  # default run. Do NOT use 'xai_alignment' (the
                                  # v13 signal) - measured circular and saturating,
                                  # see adaptive_lambda.alignment_diagnostics.
lambda_base = lmda               # starting / reference lambda value
lambda_min = 1e-5                # hard lower clip
lambda_max = 1e-1                # hard upper clip
alpha = 0.5                      # sensitivity of the adaptive update
ema_beta = 0.9                   # smoothing factor, 'ema_residual' strategy only
ig_steps = 20                    # IG interpolation steps, 'xai_alignment' strategy only
                                  # (runs once per outer iteration, at each mask-update
                                  # boundary - not every epoch, see SeT/train.py)

# --- Phase 2 settings: early stopping (optional, off by default) ------
early_stop_patience = None       # e.g. 3: stop once AUC is stable over the last 3 iters
early_stop_tol = None            # e.g. 0.001: "stable" threshold. Both None = disabled,
                                  # runs the full max_iter every time (Phase 1 behaviour).
                                  # Leave disabled if this dataset has no ground truth -
                                  # AUC is meaningless there.

# --- Phase 2 settings: complementary optimization (see CHANGES.md #6) -
# Disabled by default so the default run is unaffected by it.
use_residual_lr_scheduler = False
lr_min_scale = 0.2

if strategy not in VALID_STRATEGIES:
    raise ValueError('strategy must be one of %s' % (VALID_STRATEGIES,))

# ======================================================================
# Load data (unchanged)
# ======================================================================
dataset = coast
data, gt = dataset.get_data()
rows, cols, bands = data.shape
net_kwargs['shape'] = (rows, cols, num_bs)
print('Detecting on %s...' % dataset.name)

# ======================================================================
# Output directory + logging (Phase 2: every lambda update is logged)
# ======================================================================
result_path = os.path.join('results', MSNet(**net_kwargs).name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(result_path, 'train.log'), mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('main')
logger.info(
    'adaptive_lambda_enabled=%s strategy=%s lambda_base=%s lambda_min=%s '
    'lambda_max=%s alpha=%s ema_beta=%s',
    adaptive_lambda_enabled, strategy, lambda_base, lambda_min, lambda_max,
    alpha, ema_beta,
)

# ======================================================================
# Preprocessing: OPBS band selection + Z-score normalization (unchanged)
# ======================================================================
band_idx = select_bands.OPBS(data, num_bs)
data_bs = data[:, :, band_idx]
if data_norm:
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

# ======================================================================
# Model (unchanged - MSNet architecture is never touched)
# ======================================================================
model = Net(**net_kwargs).to(device).float()

# ======================================================================
# Loss (Phase 1 constructor signature unchanged; lambda mutation happens
# via TotalLoss.set_lmda(), see SeT/loss.py)
# ======================================================================
loss = TotalLoss(lambda_base, device)

# ======================================================================
# Adaptive Lambda controller (Phase 2) - None reproduces Phase 1 exactly
# ======================================================================
adaptive_lambda = None
if adaptive_lambda_enabled:
    adaptive_lambda = AdaptiveLambda(
        strategy=strategy,
        lambda_base=lambda_base,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        alpha=alpha,
        ema_beta=ema_beta,
        logger_=logging.getLogger('adaptive_lambda'),
    )

# ======================================================================
# Mask (unchanged)
# ======================================================================
mask = Mask((rows, cols), device)

# ======================================================================
# Optimizer (unchanged)
# ======================================================================
optimizer = Adam(model.parameters(), lr=lr)

# --- Complementary optimization: residual-driven LR scaling (opt-in) --
lr_scheduler = None
if use_residual_lr_scheduler and adaptive_lambda is not None:
    lr_scheduler = ResidualLRScheduler(optimizer, base_lr=lr, min_scale=lr_min_scale)

    # `AdaptiveLambda.step` is already called once per epoch inside
    # `SeT/train.py`; we piggy-back the LR update onto the same signal by
    # wrapping its `update` method so no changes to `SeT/train.py` are
    # required to support this optional feature.
    _orig_update = adaptive_lambda.update

    def _update_and_scale_lr(*args, **kwargs):
        result = _orig_update(*args, **kwargs)
        lr_scheduler.step(adaptive_lambda)
        return result

    adaptive_lambda.update = _update_and_scale_lr

# ======================================================================
# Separation Training (Phase 2: adaptive_lambda threaded through)
# ======================================================================
x_bs = torch.from_numpy(data_bs).to(device).float()
pr_dm, history, extras = separation_training(
    x=x_bs,
    gt=gt,
    model=model,
    loss=loss,
    mask=mask,
    optimizer=optimizer,
    epochs=epochs,
    output_iter=output_iter,
    max_iter=max_iter,
    verbose=True,
    adaptive_lambda=adaptive_lambda,
    ig_steps=ig_steps,
    early_stop_patience=early_stop_patience,
    early_stop_tol=early_stop_tol,
)

loss_history = extras['loss_history']
lambda_history = extras['lambda_history']
auc_per_epoch = extras['auc_per_epoch']
total_epochs = len(loss_history)

# ======================================================================
# Save the detection result (unchanged Phase 1 plots)
# ======================================================================
rx_dm = utils.rx(data)
fpr, tpr, rx_auc = metric.roc_auc(rx_dm, gt)
plt.plot(fpr, tpr, label='RX: %.4f' % rx_auc)

fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)
plt.plot(fpr, tpr, label='%s+SeT: %.4f' % (model.name, pr_auc),
         c='black', alpha=0.7)

plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(result_path, '%s_roc.pdf' % dataset.name))
plt.clf()
plt.close()

iters = [(_ + 1) * epochs for _ in range(max_iter)]
plt.xticks(iters)
plt.plot(iters, history)
plt.scatter([output_iter * epochs], [history[output_iter - 1]],
            marker='o', edgecolors='black', facecolors='white', label='Stop',
            zorder=10)
plt.grid(alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig(os.path.join(result_path, '%s_auc_history.pdf' % dataset.name))
plt.clf()
plt.close()

# ======================================================================
# Phase 2 plots: lambda vs epoch, loss vs epoch, AUC vs epoch
# (saved automatically, every run)
# ======================================================================
epoch_axis = np.arange(1, total_epochs + 1)

plt.plot(epoch_axis, loss_history, color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Total loss')
plt.title('Reconstruction loss vs epoch')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(result_path, '%s_loss_history.pdf' % dataset.name))
plt.clf()
plt.close()

plt.plot(epoch_axis, auc_per_epoch, color='tab:green')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC vs epoch')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(result_path, '%s_auc_vs_epoch.pdf' % dataset.name))
plt.clf()
plt.close()

if adaptive_lambda_enabled and len(lambda_history) > 0:
    plt.plot(epoch_axis, lambda_history, color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('Lambda')
    plt.title('Adaptive lambda (%s) vs epoch' % strategy)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(result_path, '%s_lambda_history.pdf' % dataset.name))
    plt.clf()
    plt.close()

# ======================================================================
# Phase 2: single consolidated CSV export (epoch, loss, lambda, auc) -
# every logged value, ready for further analysis or for loading into
# the Streamlit app's "Downloads" tab format.
# ======================================================================
csv_path = os.path.join(result_path, '%s_phase2_history.csv' % dataset.name)
with open(csv_path, 'w') as f:
    f.write('epoch,loss,lambda,auc\n')
    for e in range(total_epochs):
        lam = lambda_history[e] if e < len(lambda_history) else ''
        auc_e = auc_per_epoch[e] if e < len(auc_per_epoch) else ''
        f.write('%d,%s,%s,%s\n' % (e + 1, loss_history[e], lam, auc_e))

logger.info('Final lambda: %s', adaptive_lambda.get() if adaptive_lambda else lambda_base)
logger.info('Final AUC (SeT): %.4f | RX baseline AUC: %.4f', pr_auc, rx_auc)
logger.info('AUC trace: %s', ['%.4f' % a for a in history])
logger.info('REPORTABLE AUC (output_iter=%d): %.4f', output_iter, history[output_iter - 1])
logger.info('Peak AUC %.4f at iter %d -- DO NOT REPORT THIS NUMBER (leakage).',
            max(history), int(np.argmax(history)) + 1)

print('Complete.')
print('Results are saved in %s/.' % result_path)
