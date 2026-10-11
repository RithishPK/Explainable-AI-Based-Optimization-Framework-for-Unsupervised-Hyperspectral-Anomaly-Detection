"""
Runs the unmodified Phase 1/2 pipeline (OPBS -> Z-score -> MSNet -> SeT,
with optional adaptive lambda) across multiple uploaded datasets:

    - abu_airport   (has GT, standard 'data'/'map' convention)
    - aviris        (has GT, standard 'data'/'map' convention)
    - sandiego      (NO ground truth in the uploaded file - detection
                      map is produced and saved, but no AUC/ROC is
                      computed or reported, since there's nothing to
                      score against. Do not report an AUC for this
                      dataset unless you source a version with GT.)
    - indian_pines  (4 explicit anomaly-class definitions, ALL run and
                      ALL reported - see dataset/indian_pines/indian_pines.py
                      for why a single definition is not silently chosen)

Nothing about select_bands.OPBS, utils.ZScoreNorm, model.MSNet, or
SeT.separation_training is touched - this only adds a loop over datasets
around the exact same calls main.py makes.

Usage:
    python run_multi_dataset.py

Output:
    results/<model>/multi_dataset_summary.csv - one row per
    (dataset, variant) with AUC (or 'no_gt') and the RX baseline AUC for
    comparison. Per-dataset ROC/detection-map plots are also saved to
    results/<model>/ as usual.
"""

import os
import csv
import logging

import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch

from model import MSNet
import select_bands
import utils
import metric
from SeT import TotalLoss, Mask, separation_training
from adaptive_lambda import AdaptiveLambda, VALID_STRATEGIES

from dataset.abu_airport import abu_airport
from dataset.aviris import aviris
from dataset.sandiego import sandiego
from dataset.indian_pines import indian_pines
from dataset.indian_pines.indian_pines import ANOMALY_DEFINITIONS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- shared hyperparameters (same defaults as main.py) -----------------
lmda = 1e-3
num_bs = 64
num_layers = 3
lr = 1e-3
epochs = 150
output_iter = 5
max_iter = 10
data_norm = True

adaptive_lambda_enabled = True
strategy = 'residual'
lambda_min = 1e-5
lambda_max = 1e-1
alpha = 0.5
ema_beta = 0.9

result_path = os.path.join('results', MSNet(num_layers=num_layers, shape=(1, 1, num_bs)).name)
os.makedirs(result_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(result_path, 'multi_dataset.log'), mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('multi_dataset')

summary_rows = []


def run_one(dataset_name, data, gt):
    """Runs the unmodified pipeline once on a single (data, gt) pair.
    gt may be None, in which case AUC is skipped entirely."""
    rows, cols, bands = data.shape
    net_kwargs = dict(num_layers=num_layers, shape=(rows, cols, num_bs))

    band_idx = select_bands.OPBS(data, num_bs)
    data_bs = data[:, :, band_idx]
    if data_norm:
        data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

    model = MSNet(**net_kwargs).to(device).float()
    loss = TotalLoss(lmda, device)

    adaptive_lambda = None
    if adaptive_lambda_enabled:
        adaptive_lambda = AdaptiveLambda(
            strategy=strategy, lambda_base=lmda, lambda_min=lambda_min,
            lambda_max=lambda_max, alpha=alpha, ema_beta=ema_beta,
            logger_=logging.getLogger('adaptive_lambda.%s' % dataset_name),
        )

    mask = Mask((rows, cols), device)
    optimizer = Adam(model.parameters(), lr=lr)

    x_bs = torch.from_numpy(data_bs).to(device).float()
    pr_dm, history, extras = separation_training(
        x=x_bs, gt=gt, model=model, loss=loss, mask=mask,
        optimizer=optimizer, epochs=epochs, output_iter=output_iter,
        max_iter=max_iter, verbose=True, adaptive_lambda=adaptive_lambda,
    )

    if gt is None:
        logger.info('%s: no ground truth available, AUC not computed.', dataset_name)
        summary_rows.append({'dataset': dataset_name, 'auc_set': 'no_gt',
                              'rx_auc': 'no_gt', 'note': 'no ground truth in source file'})
        # still save the raw detection map for visual inspection
        plt.imshow(pr_dm, cmap='gray')
        plt.title('%s detection map (no GT to score against)' % dataset_name)
        plt.savefig(os.path.join(result_path, '%s_detection_map.pdf' % dataset_name))
        plt.clf()
        plt.close()
        return

    rx_dm = utils.rx(data)
    _, _, rx_auc = metric.roc_auc(rx_dm, gt)
    fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)

    plt.plot(fpr, tpr, label='RX: %.4f' % rx_auc)
    plt.plot(fpr, tpr, label='%s+SeT: %.4f' % (model.name, pr_auc), c='black', alpha=0.7)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(result_path, '%s_roc.pdf' % dataset_name))
    plt.clf()
    plt.close()

    logger.info('%s: SeT AUC=%.4f | RX AUC=%.4f', dataset_name, pr_auc, rx_auc)
    summary_rows.append({'dataset': dataset_name, 'auc_set': '%.4f' % pr_auc,
                          'rx_auc': '%.4f' % rx_auc, 'note': ''})


# ======================================================================
# 1-2. Datasets with real ground truth, standard convention
# ======================================================================
for mod in (abu_airport, aviris):
    print('Running %s...' % mod.name)
    data, gt = mod.get_data()
    run_one(mod.name, data, gt)

# ======================================================================
# 3. San Diego - NO ground truth, AUC intentionally skipped
# ======================================================================
print('Running %s (no ground truth in file)...' % sandiego.name)
data, gt = sandiego.get_data()
run_one(sandiego.name, data, gt)

# ======================================================================
# 4. Indian Pines - ALL anomaly definitions run and reported, none
#    chosen based on which scores best.
# ======================================================================
for anomaly_def in ANOMALY_DEFINITIONS:
    variant_name = 'indian_pines_%s' % anomaly_def
    print('Running %s...' % variant_name)
    data, gt = indian_pines.get_data(anomaly_def=anomaly_def)
    run_one(variant_name, data, gt)

# ======================================================================
# Consolidated summary - every result, no selection
# ======================================================================
summary_path = os.path.join(result_path, 'multi_dataset_summary.csv')
with open(summary_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'auc_set', 'rx_auc', 'note'])
    writer.writeheader()
    writer.writerows(summary_rows)

print('Complete. Summary written to %s' % summary_path)
for row in summary_rows:
    print(row)
