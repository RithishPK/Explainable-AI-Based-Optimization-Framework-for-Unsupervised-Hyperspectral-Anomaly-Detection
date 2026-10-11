"""
xai_core.py
===========

Minimal, dependency-light Integrated Gradients used as a *control signal*
during training (SeT/train.py -> adaptive_lambda.py), as opposed to
streamlit_app/xai_utils.py's `integrated_gradients`, which is the
full post-hoc explainer used for the XAI dashboard (band importance +
matplotlib figures).

Why this is a separate module instead of importing xai_utils:
    * xai_utils.py pulls in matplotlib/scipy (figure-building deps) that
      have no business being imported by the training loop / CLI path
      (main.py), which must keep running headless with no plotting deps.
    * Training-time IG only ever needs the spatial saliency map (H, W) -
      band importance is not used by any lambda strategy - so this
      version skips that reduction and returns one tensor, not a tuple.
    * v14 correction: this docstring previously said "|grad| accumulation",
      contradicting the code below, which correctly accumulates the SIGNED
      gradient. The code was right; the docstring was stale. Verified by
      completeness check: signed accumulation converges to F(x) - F(baseline),
      abs-inside-loop plateaus at a steps-independent error.

    * Default `steps` is lower (20 vs 50) because this runs ~`max_iter`
      times per training run (once per outer SeT iteration), not once
      on-demand in a dashboard, so wall-clock cost compounds.

Both implementations are intentionally kept numerically equivalent in
their core loop (baseline=zero, SIGNED grad accumulation, multiply by
(x - baseline), abs().mean(dim=2), min-max normalize) so a result
computed here and one computed in the dashboard for the same model/input
agree - this matters if the paper reports IG-alignment values alongside
dashboard IG figures for the same run.
"""

from __future__ import annotations

import torch


def ig_spatial_saliency(model, x_input, steps: int = 20, baseline=None,
                        target_weight=None) -> torch.Tensor:
    """
    Integrated Gradients spatial saliency map, min-max normalized to [0, 1].

    Parameters
    ----------
    model : the MSNet model (or any model with the same (rows, cols, bands)
        -> tuple-of-reconstructions calling convention).
    x_input : torch.Tensor (rows, cols, bands)
    steps : number of interpolation steps between baseline and x_input.
        Kept small (default 20) since this runs at every mask-update
        boundary during training, not just once for a report figure.
    baseline : torch.Tensor, same shape as x_input. Defaults to zeros,
        matching xai_utils.integrated_gradients.
    target_weight : torch.Tensor (rows, cols) or None. v14 addition.

        When None (default) the attributed target is the TOTAL
        reconstruction error, sum_p sum_b (x - y)^2. That is exactly the
        detection statistic `SeT.detect.detect` returns and therefore
        exactly the quantity `SeT.mask.Mask.update` normalizes into the
        soft mask. Attributing it and then comparing the result against
        the mask compares the residual field with itself - measured
        corr(IG, mask) ~= 0.98 on Coast. See
        adaptive_lambda.alignment_diagnostics.

        Passing (1 - mask) restricts the target to BACKGROUND
        reconstruction error only. The question then becomes "which input
        pixels drive the failure to reconstruct the background", which is
        NOT the same field as "where is the error". Attribution mass
        landing on masked pixels then carries real information: those
        pixels are contaminating background modelling elsewhere, which is
        the identity-mapping failure the whole method exists to prevent.

        This is the structural decorrelation. It cannot be achieved by
        post-processing a saliency map computed from the unweighted
        target - any function of (IG, mask) is then a function of the
        same residual field.

    Returns
    -------
    torch.Tensor (rows, cols), values in [0, 1], detached (no grad).
    """
    was_training = model.training
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x_input)

    alphas = torch.linspace(0, 1, steps).to(x_input.device)
    integrated_grads = torch.zeros_like(x_input)

    try:
        for alpha in alphas:
            interp_input = baseline + alpha * (x_input - baseline)
            interp_input = interp_input.clone().detach().requires_grad_(True)

            y = model(interp_input)
            recon = y[0]
            recon_error = ((interp_input - recon) ** 2).sum(dim=2)
            if target_weight is not None:
                recon_error = recon_error * target_weight
            recon_error.sum().backward()

            # Accumulate the SIGNED gradient. Taking abs() here (before
            # averaging across interpolation steps) breaks the
            # completeness axiom that defines Integrated Gradients
            # (Sundararajan et al. 2017): sum(attributions) should equal
            # F(x) - F(baseline). Verified numerically - averaging |grad|
            # instead of grad does not converge to that value as `steps`
            # increases, it converges to something else. Signed
            # accumulation does converge (see xai_core verification notes).
            integrated_grads += interp_input.grad
    finally:
        if was_training:
            model.train()

    integrated_grads = integrated_grads / steps
    integrated_grads = integrated_grads * (x_input.detach() - baseline)

    # Signed per-element attribution IG_i satisfies (approximately, for
    # finite `steps`) sum_i IG_i = F(x) - F(baseline). Taking abs() is
    # fine *here*, on the final signed attribution, for producing an
    # unsigned visual saliency map - unlike taking it inside the loop,
    # this does not change what the attribution converges to, it just
    # discards sign information after the fact.
    spatial = integrated_grads.abs().mean(dim=2)
    spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-9)

    return spatial.detach()
