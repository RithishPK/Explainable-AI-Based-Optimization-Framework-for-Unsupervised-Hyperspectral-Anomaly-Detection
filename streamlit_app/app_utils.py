"""
streamlit_app/app_utils.py
===========================

Reusable, UI-independent helper functions for the Phase 2 Streamlit
dashboard (`streamlit_app/app.py`). Keeping these separate from `app.py`
means:

    * The core research pipeline (OPBS -> Z-score -> MSNet -> SeT ->
      Adaptive Lambda -> ROC/AUC) can be unit-tested without Streamlit.
    * `app.py` stays a thin UI layer.

Nothing here modifies the project's original modules; it only imports
and orchestrates them, exactly like `main.py` does.
"""

from __future__ import annotations

import io
import os
import sys
import csv
import base64
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make the project root (one level up from streamlit_app/) importable so
# we can reuse the existing modules unmodified.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.optim import Adam

import select_bands
import utils
import metric
from model import MSNet
from SeT import TotalLoss, Mask, separation_training
from adaptive_lambda import AdaptiveLambda, VALID_STRATEGIES


# ----------------------------------------------------------------------
# Configuration container
# ----------------------------------------------------------------------

@dataclass
class RunConfig:
    num_bs: int = 64
    num_layers: int = 3
    lr: float = 1e-3
    epochs: int = 50
    output_iter: int = 3
    max_iter: int = 5
    data_norm: bool = True

    adaptive_lambda_enabled: bool = True
    strategy: str = 'residual'
    lambda_base: float = 1e-3
    lambda_min: float = 1e-5
    lambda_max: float = 1e-1
    alpha: float = 0.5
    ema_beta: float = 0.9
    ig_steps: int = 20  # 'xai_alignment' strategy only - IG steps at each
                         # mask-update boundary. Was previously hardcoded
                         # via SeT.train.separation_training's default and
                         # not reachable from the UI at all.

    seed: Optional[int] = None  # None = no seeding (previous behaviour,
                                 # every run non-reproducible)

    early_stop_patience: Optional[int] = None  # None = disabled, runs
                                                # the full max_iter every
                                                # time (previous behaviour)
    early_stop_tol: Optional[float] = None

    enable_multiscale_gating: bool = False  # EXPERIMENTAL - requires
                                             # strategy='xai_alignment'.
                                             # See adaptive_lambda.multiscale_gate_signal.
    gating_alpha: float = 0.5
    gating_min: float = 0.5
    gating_max: float = 1.5


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------

def load_bundled_coast():
    """Fall back to the dataset that ships with the repository."""
    from dataset import coast
    data, gt = coast.get_data()
    return data, gt, 'coast'


def _load_mat_arrays(file_obj):
    """Returns {key: ndarray} for all non-dunder variables in a .mat file.

    MATLAB switches to the v7.3 format (HDF5-based) for files that are
    large or were saved with the -v7.3 flag - scipy.io.loadmat cannot
    read this format at all and raises an opaque error, not a helpful
    one. With the upload limit raised to 1GB, this becomes a realistic
    case (e.g. WHU-Hi-HanChuan-scale scenes) rather than a hypothetical
    one, so it's handled explicitly here via h5py instead of failing
    with a scipy stack trace that gives the user no idea what went wrong.
    """
    try:
        mat = sio.loadmat(file_obj)
        return {k: v for k, v in mat.items() if not k.startswith('__')}
    except (NotImplementedError, ValueError) as e:
        if 'v7.3' not in str(e) and 'HDF5' not in str(e):
            raise
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "This .mat file is in MATLAB v7.3 (HDF5) format, which scipy "
                "can't read. Install h5py to support it: "
                "python -m pip install h5py"
            ) from e

        file_obj.seek(0)
        arrays = {}
        with h5py.File(file_obj, 'r') as f:
            for key in f.keys():
                arr = f[key][()]
                # v7.3 stores arrays transposed relative to scipy's
                # convention (MATLAB is column-major, h5py reads it
                # row-major) - transpose back so shapes match what the
                # rest of the pipeline (and scipy-loaded files) expect.
                if hasattr(arr, 'ndim') and arr.ndim >= 2:
                    arr = arr.T
                arrays[key] = arr
        return arrays


def load_uploaded_mat(uploaded_files) -> (np.ndarray, Optional[np.ndarray], Optional[dict], str):
    """
    Parse 1 or 2 user-uploaded `.mat` files into a hyperspectral cube and
    (optionally) a ground-truth mask.

    Detection is shape-based, not key-name-based, since real datasets use
    inconsistent variable names ('data', 'Sandiego', 'indian_pines', ...):
        * A 3-D array (rows, cols, bands) is treated as the hyperspectral
          cube. Exactly one is required, across however many files were
          uploaded.
        * A 2-D array (rows, cols) is treated as a ground-truth mask.
          Zero is fine (no GT provided). If more than one 2-D array is
          found, the largest (by pixel count matching the cube) is used.

    Returns
    -------
    (data, gt, raw_multiclass, name)
        data : (rows, cols, bands) float ndarray
        gt   : (rows, cols) bool ndarray, or None if no mask found, or
               None if the mask is multi-class (see raw_multiclass)
        raw_multiclass : dict {'array': ndarray, 'classes': {class_id: pixel_count}}
               if a 2-D mask was found but has more than 2 unique values
               (e.g. Indian Pines' 16-class land-cover map) - the caller
               must let the user explicitly choose which class(es) count
               as "anomaly" rather than guessing. None otherwise.
        name : str, derived from the cube file's name
    """
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    cube = None
    cube_name = None
    mask_candidates = []  # list of (array, source_filename)

    for f in uploaded_files:
        arrays = _load_mat_arrays(f)
        for key, arr in arrays.items():
            if arr.ndim == 3:
                if cube is not None:
                    raise ValueError(
                        "More than one 3-D (rows x cols x bands) array found "
                        "across the uploaded file(s) - only one hyperspectral "
                        "cube is supported per run."
                    )
                cube = arr.astype(float)
                cube_name = getattr(f, 'name', 'uploaded').rsplit('.', 1)[0]
            elif arr.ndim == 2:
                mask_candidates.append(arr)

    if cube is None:
        raise ValueError(
            "No 3-D (rows x cols x bands) hyperspectral cube variable was "
            "found in the uploaded file(s). Check that you uploaded the "
            "data cube, not only a ground-truth/label file."
        )

    rows, cols, _ = cube.shape
    gt = None
    raw_multiclass = None

    matching_masks = [m for m in mask_candidates if m.shape == (rows, cols)]
    if matching_masks:
        raw_gt = matching_masks[0]
        unique_vals = np.unique(raw_gt)
        # treat as binary if there are at most 2 distinct values, however
        # they're encoded (0/1, 0/255, True/False, etc.)
        if len(unique_vals) <= 2:
            gt = (raw_gt != unique_vals.min()).astype(bool) if len(unique_vals) == 2 \
                else np.zeros((rows, cols), dtype=bool)
        else:
            classes, counts = np.unique(raw_gt, return_counts=True)
            raw_multiclass = {
                'array': raw_gt,
                'classes': dict(zip(classes.tolist(), counts.tolist())),
            }

    return cube, gt, raw_multiclass, cube_name


def binarize_multiclass_gt(raw_multiclass: dict, selected_classes: list) -> np.ndarray:
    """Turns a multi-class GT array + explicit user-selected class list into
    a binary anomaly mask. Never called automatically - the caller (UI)
    must have the user pick selected_classes first."""
    arr = raw_multiclass['array']
    return np.isin(arr, selected_classes)


# ----------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------

def run_pipeline(
    data: np.ndarray,
    gt: np.ndarray,
    cfg: RunConfig,
    epoch_callback=None,
    iter_callback=None,
):
    """
    Run the full Phase 1 + Phase 2 pipeline (band selection -> Z-score
    normalization -> MSNet -> Separation Training w/ optional Adaptive
    Lambda -> RX baseline -> ROC/AUC) and return everything the dashboard
    needs to render.

    `epoch_callback(epoch, loss_val, lmda_val)` and
    `iter_callback(iter_idx, auc)` are forwarded verbatim to
    `SeT.separation_training` for live UI updates; both are optional.

    Ground truth handling
    ----------------------
    If `gt` is None (user uploaded a cube with no 'map' variable), a
    dummy all-False ground truth is used purely so the *unmodified*
    `separation_training` / `metric.roc_auc` code can run without being
    touched. In that case AUC/ROC values are not meaningful and the
    caller should tell the user so (the app does).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    rows, cols, bands = data.shape
    gt_missing = gt is None
    if gt_missing:
        gt = np.zeros((rows, cols), dtype=bool)

    # --- Preprocessing: OPBS band selection + Z-score norm (unchanged) --
    band_idx = select_bands.OPBS(data, cfg.num_bs)
    data_bs = data[:, :, band_idx]

    # Pseudocolor composite for display, computed from the FULL raw cube
    # (not the OPBS-selected/normalized subset) - a scene-level preview
    # independent of preprocessing settings, in the spirit of the MSNet
    # paper's Fig. 4. Shown for visual contrast against the detection map.
    pseudocolor_img, pseudocolor_bands = make_pseudocolor(data)

    if cfg.data_norm:
        data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

    # --- Model / loss / mask / optimizer (unchanged) -------------------
    net_kwargs = dict(num_layers=cfg.num_layers, shape=(rows, cols, cfg.num_bs))
    model = MSNet(**net_kwargs).to(device).float()
    loss = TotalLoss(cfg.lambda_base, device)
    mask = Mask((rows, cols), device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # --- Adaptive Lambda controller (Phase 2) --------------------------
    adaptive_lambda = None
    if cfg.adaptive_lambda_enabled:
        adaptive_lambda = AdaptiveLambda(
            strategy=cfg.strategy,
            lambda_base=cfg.lambda_base,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
            alpha=cfg.alpha,
            ema_beta=cfg.ema_beta,
            enable_multiscale_gating=cfg.enable_multiscale_gating,
            num_layers=cfg.num_layers,
            gating_alpha=cfg.gating_alpha,
            gating_min=cfg.gating_min,
            gating_max=cfg.gating_max,
        )

    # --- Separation training --------------------------------------------
    # Early stopping is meaningless (and actively misleading) when there's
    # no real ground truth - AUC would just be computed against an
    # all-False dummy mask, so "AUC stability" doesn't mean anything.
    # Force it off in that case regardless of what the UI/config asked for.
    _es_patience = None if gt_missing else cfg.early_stop_patience
    _es_tol = None if gt_missing else cfg.early_stop_tol

    x_bs = torch.from_numpy(data_bs).to(device).float()
    pr_dm, history, extras = separation_training(
        x=x_bs,
        gt=gt,
        model=model,
        loss=loss,
        mask=mask,
        optimizer=optimizer,
        epochs=cfg.epochs,
        output_iter=cfg.output_iter,
        max_iter=cfg.max_iter,
        verbose=False,
        adaptive_lambda=adaptive_lambda,
        callback=epoch_callback,
        iter_callback=iter_callback,
        ig_steps=cfg.ig_steps,
        early_stop_patience=_es_patience,
        early_stop_tol=_es_tol,
    )

    # --- RX baseline + ROC/AUC (unchanged) -------------------------------
    rx_dm = utils.rx(data)

    results = {
        'model': model,
        'gt_missing': gt_missing,
        'gt': gt,
        'pr_dm': pr_dm,
        'rx_dm': rx_dm,
        'auc_history': history,
        'loss_history': extras['loss_history'],
        'lambda_history': extras['lambda_history'],
        'auc_per_epoch': extras['auc_per_epoch'],
        'band_idx': band_idx,
        'x_bs': x_bs,
        'rows': rows,
        'cols': cols,
        'epochs_per_iter': cfg.epochs,
        'max_iter': cfg.max_iter,
        'strategy': cfg.strategy,
        'early_stopped_at': extras.get('early_stopped_at'),
        'pseudocolor_img': pseudocolor_img,
        'pseudocolor_bands': pseudocolor_bands,
        'final_layer_gates': getattr(adaptive_lambda, 'layer_gates', None) if adaptive_lambda is not None else None,
    }

    if not gt_missing:
        fpr_pr, tpr_pr, auc_pr = metric.roc_auc(pr_dm, gt)
        fpr_rx, tpr_rx, auc_rx = metric.roc_auc(rx_dm, gt)
        results.update({
            'fpr_pr': fpr_pr, 'tpr_pr': tpr_pr, 'auc_pr': auc_pr,
            'fpr_rx': fpr_rx, 'tpr_rx': tpr_rx, 'auc_rx': auc_rx,
        })

    return results


# ----------------------------------------------------------------------
# Plotting helpers (return matplotlib Figures - the app renders them
# with st.pyplot)
# ----------------------------------------------------------------------

def fig_roc(results):
    fig, ax = plt.subplots()
    if 'fpr_pr' in results:
        ax.plot(results['fpr_rx'], results['tpr_rx'], label='RX: %.4f' % results['auc_rx'])
        ax.plot(results['fpr_pr'], results['tpr_pr'], c='black', alpha=0.7,
                 label='%s+SeT: %.4f' % (results['model'].name, results['auc_pr']))
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No ground truth uploaded - ROC unavailable',
                ha='center', va='center')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(alpha=0.3)
    return fig


def fig_lambda_history(results, strategy):
    fig, ax = plt.subplots()
    ax.plot(results['lambda_history'], color='tab:orange')

    # Mark mask-update iteration boundaries - this is where SeT's
    # anomaly/background assignment discontinuously shifts (mask.update),
    # which is where lambda instability was originally observed for the
    # 'residual' strategy, and where 'xai_alignment' deliberately
    # recalibrates (so its curve is flat within a block, then jumps at
    # each line - that is the intended behaviour, not a rendering bug).
    epochs_per_iter = results.get('epochs_per_iter')
    max_iter = results.get('max_iter')
    if epochs_per_iter and max_iter:
        for i in range(1, max_iter):
            ax.axvline(i * epochs_per_iter, color='grey', alpha=0.35, linestyle='--', linewidth=0.8)
        if max_iter > 1:
            ax.axvline(epochs_per_iter, color='grey', alpha=0.35, linestyle='--',
                       linewidth=0.8, label='mask-update boundary')
            ax.legend(fontsize=8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lambda')
    ax.set_title('Adaptive Lambda (%s) vs Epoch' % strategy)
    ax.grid(alpha=0.3)
    return fig


def fig_loss_history(results):
    fig, ax = plt.subplots()
    ax.plot(results['loss_history'], color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Total Loss vs Epoch')
    ax.grid(alpha=0.3)
    return fig


def fig_auc_history(results):
    fig, ax = plt.subplots()
    ax.plot(results['auc_per_epoch'], color='tab:green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Epoch')
    ax.grid(alpha=0.3)
    return fig


def fig_detection_map(dm, title, vmax_percentile=99):
    """
    Percentile-clipped color scale, not raw min-max. A single extreme
    outlier pixel (e.g. an image-border artifact, or a bright man-made
    structure the underlying detector isn't specifically tuned to
    ignore) can sit far above the genuine anomaly score range and, on a
    linear min-max scale, stretch the colorbar so far that real (more
    moderate) anomaly signal becomes visually indistinguishable from
    background - the detector can be working reasonably while the plot
    makes it look like it isn't. Clipping vmax to the 99th percentile
    doesn't change any underlying score or result, only what's visible.
    """
    fig, ax = plt.subplots()
    vmax = np.percentile(dm, vmax_percentile)
    im = ax.imshow(dm, cmap='jet', vmin=dm.min(), vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def fig_ground_truth(gt):
    fig, ax = plt.subplots()
    ax.imshow(gt, cmap='gray')
    ax.set_title('Ground Truth')
    ax.axis('off')
    return fig


def make_pseudocolor(data, percentile_clip=(2, 98)):
    """
    3-band pseudocolor composite from the full (pre-OPBS) hyperspectral
    cube, in the spirit of the MSNet paper's Fig. 4 (upper row) - not a
    true-color image (no wavelength metadata is available in these .mat
    files to pick literal RGB wavelengths), but three bands spread across
    the spectrum mapped to R/G/B, each independently contrast-stretched.

    A plain min-max stretch (what the single-band raw display used
    before) often looks flat/blurry because a handful of outlier pixels
    (common in raw radiance/DN data) dominate the range and crush
    everything else toward one end. Percentile clipping fixes that by
    ignoring the extreme tails per channel before stretching to [0, 1].

    Returns
    -------
    (rgb, (r_idx, g_idx, b_idx)) : (rows, cols, 3) float array in [0, 1],
        and the band indices used, for captioning.
    """
    rows, cols, bands = data.shape
    r_idx = min(bands - 1, int(bands * 0.8))
    g_idx = min(bands - 1, int(bands * 0.5))
    b_idx = min(bands - 1, int(bands * 0.2))

    rgb = np.stack(
        [data[:, :, r_idx], data[:, :, g_idx], data[:, :, b_idx]], axis=-1
    ).astype(float)

    stretched = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], percentile_clip)
        stretched[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-9), 0, 1)

    return stretched, (r_idx, g_idx, b_idx)


def fig_pseudocolor(rgb, band_indices):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    # bilinear interpolation smooths the display without inventing data -
    # a 100x100 (or similar) source image WILL show individual pixels at
    # any real zoom level (that's the actual data resolution, not fixable
    # by rendering), but matplotlib's default interpolation for imshow
    # can render as hard-edged blocks depending on backend/scaling, which
    # looks worse than necessary for what's meant to be a qualitative
    # preview image, not a precision data readout like the detection maps.
    ax.imshow(rgb, interpolation='bilinear')
    ax.set_title(f'Pseudocolor Composite (R=band {band_indices[0]}, '
                 f'G=band {band_indices[1]}, B=band {band_indices[2]})')
    ax.axis('off')
    return fig


# ----------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------

def metrics_csv_bytes(results) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['epoch', 'loss', 'lambda', 'auc'])
    n = len(results['loss_history'])
    for e in range(n):
        writer.writerow([
            e + 1,
            results['loss_history'][e],
            results['lambda_history'][e] if e < len(results['lambda_history']) else '',
            results['auc_per_epoch'][e] if e < len(results['auc_per_epoch']) else '',
        ])
    return buf.getvalue().encode('utf-8')


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.read()


def model_state_bytes(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return buf.read()
