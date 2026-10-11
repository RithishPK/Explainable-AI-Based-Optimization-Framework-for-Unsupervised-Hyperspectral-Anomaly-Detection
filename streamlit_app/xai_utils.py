"""
streamlit_app/xai_utils.py
============================

XAI methods for the trained MSNet model: plain gradient attribution
(the Phase 1 FR4 method), Integrated Gradients (Phase 2 enhancement),
and GradCAM on the deepest encoder's bottleneck layer.

Ported from the user's original Phase 1 notebook code. Behavioral
changes from the original:
    * `plt.show()` calls replaced with returning Figure objects, since
      this runs inside Streamlit (`st.pyplot(fig)`), not a notebook.
    * `spatial_saliency` (referenced but never defined in the original
      snippet's Plot 2) is now explicitly computed from the plain
      gradient, mirroring how `ig_spatial` is derived from
      `ig_attributions` - same reduction (mean abs over the band axis),
      same min-max normalization.
    * IG's forward/backward loop takes an optional `progress_callback`
      so the UI can show real progress across the 50 (or however many)
      interpolation steps, since this is the slow part (~1-2 min at
      100x100 scale per the user's own estimate, more on larger scenes).
    * All hook registration is wrapped in try/finally so a mid-run error
      can never leave a dangling hook on the model that would corrupt a
      later training or XAI run.
    * The GradCAM target layer, `model.encoders[-1].bottleneck.layers[2]`,
      is verified against the actual MSNet architecture
      (Bottleneck.layers = [Conv2d, ReLU, Conv2d], so index 2 is a real
      Conv2d) - this is unchanged from the user's original code, just
      confirmed rather than assumed.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom


# ----------------------------------------------------------------------
# Plain gradient (Phase 1 FR4 original)
# ----------------------------------------------------------------------

def plain_gradient(model, x_input):
    """
    Single backward pass w.r.t. reconstruction error. Returns
    (band_importance, spatial_saliency), both normalized to [0, 1].
    """
    model.eval()
    x_grad = x_input.clone().detach().requires_grad_(True)
    y = model(x_grad)
    recon = y[0]
    recon_error = ((x_grad - recon) ** 2).sum(dim=2)
    recon_error.sum().backward()

    grad_abs = x_grad.grad.abs()

    band_importance = grad_abs.mean(dim=(0, 1)).cpu().numpy()
    band_importance = band_importance / (band_importance.max() + 1e-9)

    spatial_saliency = grad_abs.mean(dim=2).cpu().numpy()
    spatial_saliency = (spatial_saliency - spatial_saliency.min()) / (
        spatial_saliency.max() - spatial_saliency.min() + 1e-9
    )

    return band_importance, spatial_saliency


# ----------------------------------------------------------------------
# Integrated Gradients (Phase 2 enhancement)
# ----------------------------------------------------------------------

def integrated_gradients(model, x_input, baseline=None, steps=50, progress_callback=None):
    """
    Integrated Gradients attribution, averaged over `steps` interpolation
    points between `baseline` (default: zero tensor) and `x_input`.

    progress_callback(step, total_steps), if provided, is called after
    each interpolation step so a caller (e.g. Streamlit) can show real
    progress instead of a plain spinner for what can be a 1-2+ minute
    computation.
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x_input)

    alphas = torch.linspace(0, 1, steps).to(x_input.device)
    integrated_grads = torch.zeros_like(x_input)

    for i, alpha in enumerate(alphas):
        interp_input = baseline + alpha * (x_input - baseline)
        interp_input = interp_input.clone().detach().requires_grad_(True)

        y = model(interp_input)
        recon = y[0]
        recon_error = ((interp_input - recon) ** 2).sum(dim=2)
        recon_error.sum().backward()

        # NOTE (Phase 2 correctness fix): accumulate the SIGNED gradient,
        # not its absolute value. Averaging |grad| across interpolation
        # steps - what this function did previously - breaks the
        # completeness axiom that mathematically defines Integrated
        # Gradients (Sundararajan et al. 2017): sum(attributions) should
        # converge to F(x) - F(baseline) as `steps` grows. Verified
        # numerically that the abs-inside-loop version does not converge
        # to that value at all (stays off by a large, steps-independent
        # margin), while the signed version does. Any IG figures produced
        # before this fix (e.g. earlier Phase 1/2 report figures) were
        # computed with the old, non-convergent formula and should be
        # regenerated for anything citing them as Integrated Gradients.
        integrated_grads += interp_input.grad

        if progress_callback is not None:
            progress_callback(i + 1, steps)

    integrated_grads = integrated_grads / steps
    integrated_grads = integrated_grads * (x_input.detach() - baseline)

    # abs() here, on the final signed attribution, is fine - it only
    # discards sign for display purposes, unlike taking it inside the
    # loop, which changed what the estimator converges to.
    band_importance = integrated_grads.abs().mean(dim=(0, 1)).cpu().numpy()
    band_importance = band_importance / (band_importance.max() + 1e-9)

    spatial_saliency = integrated_grads.abs().mean(dim=2).cpu().numpy()
    spatial_saliency = (spatial_saliency - spatial_saliency.min()) / (
        spatial_saliency.max() - spatial_saliency.min() + 1e-9
    )

    return band_importance, spatial_saliency


# ----------------------------------------------------------------------
# GradCAM (deepest encoder's bottleneck)
# ----------------------------------------------------------------------

def gradcam(model, x_input, rows, cols):
    """
    GradCAM on `model.encoders[-1].bottleneck.layers[2]` (the second
    Conv2d in the coarsest-scale encoder's residual bottleneck block -
    the closest analogue to a "last conv layer" in this architecture,
    since MSNet has no single final conv before its multi-head decoder).

    Hooks are always removed, even if the forward/backward pass raises,
    so a failure here can never corrupt a later training or XAI call.
    """
    activation = {}
    gradient = {}

    def save_activation(module, inp, out):
        activation['bottleneck'] = out.detach()

    def save_gradient(module, grad_input, grad_output):
        gradient['bottleneck'] = grad_output[0].detach()

    target_layer = model.encoders[-1].bottleneck.layers[2]
    fwd_hook = target_layer.register_forward_hook(save_activation)
    bwd_hook = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.eval()
        x_cam = x_input.clone().detach().requires_grad_(True)
        y_cam = model(x_cam)
        recon_cam = y_cam[0]
        recon_error_cam = ((x_cam - recon_cam) ** 2).sum(dim=2)
        recon_error_cam.sum().backward()
    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    act = activation['bottleneck']   # (1, C, H, W)
    grad = gradient['bottleneck']    # (1, C, H, W)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze(0)
    cam = torch.relu(cam).cpu().numpy()

    scale_h = rows / cam.shape[0]
    scale_w = cols / cam.shape[1]
    cam = zoom(cam, (scale_h, scale_w), order=1)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    return cam


# ----------------------------------------------------------------------
# Orchestration - runs all three methods, returns a plain dict (no
# figures yet, so this can be unit-tested / cached independently of
# matplotlib/Streamlit)
# ----------------------------------------------------------------------

def run_xai(model, x_bs, band_idx, rows, cols, ig_steps=50, ig_progress_callback=None):
    plain_band_imp, plain_spatial = plain_gradient(model, x_bs)
    ig_band_imp, ig_spatial = integrated_gradients(
        model, x_bs, steps=ig_steps, progress_callback=ig_progress_callback
    )
    cam = gradcam(model, x_bs, rows, cols)

    top_k = min(10, len(plain_band_imp))
    top_plain_idx = np.argsort(plain_band_imp)[::-1][:top_k]
    top_ig_idx = np.argsort(ig_band_imp)[::-1][:top_k]

    return {
        'plain_band_importance': plain_band_imp,
        'plain_spatial_saliency': plain_spatial,
        'ig_band_importance': ig_band_imp,
        'ig_spatial_saliency': ig_spatial,
        'gradcam_map': cam,
        'top_k': top_k,
        'top_plain_bands': [band_idx[i] for i in top_plain_idx],
        'top_plain_vals': plain_band_imp[top_plain_idx],
        'top_ig_bands': [band_idx[i] for i in top_ig_idx],
        'top_ig_vals': ig_band_imp[top_ig_idx],
        'ig_steps': ig_steps,
    }


# ----------------------------------------------------------------------
# Figure builders (return matplotlib Figures for st.pyplot)
# ----------------------------------------------------------------------

def fig_band_importance_comparison(xr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    top_k = xr['top_k']

    axes[0].bar(range(top_k), xr['top_plain_vals'], color='steelblue',
                alpha=0.85, edgecolor='black')
    axes[0].set_xticks(range(top_k))
    axes[0].set_xticklabels([f"B{b}" for b in xr['top_plain_bands']],
                             rotation=30, ha='right')
    axes[0].set_title('Plain Gradient Attribution\n(FR4 Original)', fontsize=11)
    axes[0].set_ylabel('Normalised Importance')
    axes[0].set_ylim(0, 1.2)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(xr['top_plain_vals']):
        axes[0].text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=8)

    axes[1].bar(range(top_k), xr['top_ig_vals'], color='darkorange',
                alpha=0.85, edgecolor='black')
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels([f"B{b}" for b in xr['top_ig_bands']],
                             rotation=30, ha='right')
    axes[1].set_title(f"Integrated Gradients ({xr['ig_steps']} steps)\n(FR4 Enhanced)", fontsize=11)
    axes[1].set_ylabel('Normalised Importance')
    axes[1].set_ylim(0, 1.2)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(xr['top_ig_vals']):
        axes[1].text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=8)

    fig.suptitle('XAI Comparison — Spectral Band Importance: Gradient vs Integrated Gradients',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def fig_spatial_comparison(xr, gt, pr_dm, auc_pr, gt_missing):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    if gt_missing:
        axes[0].text(0.5, 0.5, 'No ground truth\nuploaded', ha='center', va='center')
        axes[0].axis('off')
    else:
        axes[0].imshow(gt, cmap='gray')
        axes[0].axis('off')
    axes[0].set_title('Ground Truth', fontsize=10)

    im1 = axes[1].imshow(pr_dm, cmap='jet', vmin=pr_dm.min(), vmax=np.percentile(pr_dm, 99))
    axes[1].set_title(f'MSNet Detection\nAUC={auc_pr:.4f}' if not gt_missing else 'MSNet Detection\n(no AUC - no GT)', fontsize=10)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(xr['plain_spatial_saliency'], cmap='hot')
    axes[2].set_title('Plain Gradient\nSaliency', fontsize=10)
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(xr['ig_spatial_saliency'], cmap='hot')
    axes[3].set_title('Integrated Gradients\nSaliency', fontsize=10)
    axes[3].axis('off')
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    gc_map = xr['gradcam_map']
    im4 = axes[4].imshow(gc_map, cmap='jet', vmin=gc_map.min(), vmax=np.percentile(gc_map, 99))
    axes[4].set_title('GradCAM\nActivation Map', fontsize=10)
    axes[4].axis('off')
    fig.colorbar(im4, ax=axes[4], fraction=0.046)

    fig.suptitle('XAI Spatial Explanation Comparison — Three Methods', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def fig_xai_dashboard(xr, gt, pr_dm, auc_pr, gt_missing):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    a0 = fig.add_subplot(gs[0, 0])
    if gt_missing:
        a0.text(0.5, 0.5, 'No GT', ha='center', va='center')
        a0.axis('off')
    else:
        a0.imshow(gt, cmap='gray')
        a0.axis('off')
    a0.set_title('Ground Truth')

    a1 = fig.add_subplot(gs[0, 1])
    im1 = a1.imshow(pr_dm, cmap='jet', vmin=pr_dm.min(), vmax=np.percentile(pr_dm, 99))
    a1.set_title(f'Detection Map\nAUC={auc_pr:.4f}' if not gt_missing else 'Detection Map\n(no AUC)')
    a1.axis('off')
    fig.colorbar(im1, ax=a1, fraction=0.046)

    a2 = fig.add_subplot(gs[0, 2])
    im_ig = a2.imshow(xr['ig_spatial_saliency'], cmap='hot')
    a2.set_title('IG Spatial Saliency')
    a2.axis('off')
    fig.colorbar(im_ig, ax=a2, fraction=0.046)

    a3 = fig.add_subplot(gs[0, 3])
    gc_map = xr['gradcam_map']
    im_gc = a3.imshow(gc_map, cmap='jet', vmin=gc_map.min(), vmax=np.percentile(gc_map, 99))
    a3.set_title('GradCAM Activation')
    a3.axis('off')
    fig.colorbar(im_gc, ax=a3, fraction=0.046)

    top_k = xr['top_k']
    a4 = fig.add_subplot(gs[1, :2])
    a4.bar(range(top_k), xr['top_plain_vals'], alpha=0.6, label='Plain Gradient', color='steelblue')
    a4.bar(range(top_k), xr['top_ig_vals'], alpha=0.7, label='Integrated Gradients',
           color='darkorange', width=0.5)
    a4.set_xticks(range(top_k))
    a4.set_xticklabels([f"B{b}" for b in xr['top_ig_bands']], rotation=30, ha='right', fontsize=9)
    a4.set_title('Top Band Importance: Gradient vs IG')
    a4.set_ylim(0, 1.3)
    a4.legend()
    a4.grid(axis='y', alpha=0.3)

    a5 = fig.add_subplot(gs[1, 2:])
    a5.bar(range(len(xr['ig_band_importance'])), xr['ig_band_importance'],
           color='darkorange', alpha=0.75, label='IG')
    a5.bar(range(len(xr['plain_band_importance'])), xr['plain_band_importance'],
           color='steelblue', alpha=0.5, label='Gradient')
    a5.set_xlabel('Selected Band Index (OPBS order)')
    a5.set_ylabel('Importance (normalised)')
    a5.set_title('All Selected Bands — IG vs Gradient Importance')
    a5.legend()
    a5.grid(axis='y', alpha=0.3)

    fig.suptitle('Enhanced XAI Dashboard — Integrated Gradients + GradCAM\nMSNet Hyperspectral Anomaly Detection',
                 fontsize=13, fontweight='bold')
    return fig
