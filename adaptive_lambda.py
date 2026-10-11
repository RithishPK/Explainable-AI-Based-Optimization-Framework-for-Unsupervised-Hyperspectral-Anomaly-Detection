"""
Phase 2 - Adaptive Lambda Optimization Framework
=================================================

This module introduces an *external* controller for the anomaly-suppression
weight (`lmda`) used inside `SeT.loss.SeTLoss` / `SeT.loss.TotalLoss`.

Design goals (see project requirements):
    * MSNet architecture is never touched.
    * Dataset loading / OPBS band selection are never touched.
    * When `strategy='fixed'` (or the caller never invokes `.step()` /
      `.update()`), the produced lambda is always exactly `lambda_base`,
      i.e. numerically identical to the Phase 1 behaviour.
    * Lambda is always clipped to `[lambda_min, lambda_max]`.
    * All signal computations reuse tensors that the training loop already
      has in memory (the decoder output `y`, the current `mask`, and the
      populated `.grad` fields after `loss.backward()`). No extra forward
      passes are introduced, so the added overhead is a handful of scalar
      reductions per epoch - negligible compared to a MSNet forward/backward
      pass.

Supported strategies
---------------------
    fixed              Lambda never changes (Phase 1 behaviour).
    residual           Lambda grows/shrinks with the *instantaneous* ratio
                        of anomaly-suppression residual to background
                        reconstruction residual.
    ema_residual       Same signal as 'residual', but smoothed with an
                        exponential moving average (`ema_beta`) to avoid
                        epoch-to-epoch jitter.
    gradient_norm      Lambda is modulated by the overall gradient norm of
                        the model: large/unstable gradients relax lambda,
                        small gradients (near convergence) let lambda grow.
    spectral_entropy   Lambda is modulated by the spectral (per-band energy)
                        entropy of the current reconstruction: a more
                        disordered spectral energy distribution raises
                        lambda so the suppression term is emphasised more.

Every update is recorded in `self.history` (a list of dicts) so that the
full lambda trajectory - together with whatever signals produced it - can
be logged, plotted, or exported to CSV.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger('adaptive_lambda')
if not logger.handlers:
    # Library-friendly default: don't print anything unless the host
    # application configures logging (main.py does this for Phase 2).
    logger.addHandler(logging.NullHandler())


VALID_STRATEGIES = (
    'fixed',
    'residual',
    'ema_residual',
    'gradient_norm',
    'spectral_entropy',
    'xai_alignment',        # v13 signal - RETAINED FOR COMPARISON ONLY, see below
    'xai_alignment_bg',     # v14 corrected signal (background-targeted IG)
)

# The v13 'xai_alignment' signal was measured on Coast and found to be
# circular (corr(IG, mask) ~= 0.98) and saturating (alignment > 0.98 from
# iteration 2, never negative). It is kept in VALID_STRATEGIES so the
# failure can be reproduced and reported as an ablation row, NOT because
# it should be used. Use 'xai_alignment_bg'. See adaptive_lambda.
# alignment_diagnostics for the measured numbers. Use 'xai_alignment_bg'.


# ----------------------------------------------------------------------
# Signal extraction helpers
# ----------------------------------------------------------------------
# These helpers only *read* tensors that already exist in the training
# loop (no extra forward pass, no architecture change). They are kept as
# free functions so they can be unit-tested independently of the class.

def residual_signals(x, y, mask, eps: float = 1e-8):
    """
    Compute the background-reconstruction residual and the
    anomaly-suppression residual from the current batch, reusing the same
    quantities the loss itself already computes.

    Parameters
    ----------
    x : torch.Tensor (rows, cols, bands)
        Input (band-selected, normalized) hyperspectral cube.
    y : torch.Tensor (rows, cols, bands)
        Decoder reconstruction (`decoder_outputs[0]` in the original code).
    mask : SeT.mask.Mask
        Current anomaly mask.

    Returns
    -------
    (background_residual, anomaly_residual) : (float, float)
    """
    import torch

    with torch.no_grad():
        diff = x - y
        anm_mask = mask.mask
        bg_mask = 1 - anm_mask

        num_anm = anm_mask.sum().item()
        num_bg = bg_mask.sum().item()

        anomaly_residual = ((anm_mask * diff) ** 2).sum().item() / (num_anm + eps)
        background_residual = ((bg_mask * diff) ** 2).sum().item() / (num_bg + eps)

    return background_residual, anomaly_residual


def gradient_norm(model, eps: float = 1e-8):
    """
    Global L2 norm of all currently-populated parameter gradients.
    Must be called after `loss.backward()` and before the gradients are
    cleared by the next `optimizer.zero_grad()`.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def xai_alignment_signal(spatial_ig, mask, eps: float = 1e-8):
    """
    Signed alignment between an Integrated-Gradients spatial saliency map
    and the current (soft, continuous-valued) SeT mask.

    Unlike `residual_signals` (which mirrors the loss the network is being
    optimized to reduce, and is therefore monotonic across training),
    this signal answers a different question: does the model's own
    explanation agree with what the current mask thinks is anomalous?

    `mask.mask` is a soft mask in [0, 1] (see SeT.mask.Mask.update, which
    assigns it via MinMaxNorm of the detection map) - not binary - so we
    use it directly as a weighting, the same pattern `residual_signals`
    already uses for bg_mask/anm_mask, rather than thresholding it.

    Returns
    -------
    alignment : float, roughly in [-1, 1].
        > 0: IG saliency concentrates inside the masked (putative-anomaly)
             region more than outside it - the explanation corroborates
             the mask, so it is safe to push suppression harder.
        < 0: IG saliency concentrates outside the masked region (e.g. the
             mask has picked up striping/noise the way GradCAM was found
             to on this architecture) - the explanation disagrees with
             the mask, so suppression should ease off rather than trust
             a mask that may be wrong.
        This is what makes the signal usable for control, unlike the
        residual ratio: it is not constrained to only increase.
    """
    import torch

    with torch.no_grad():
        w = mask.mask.squeeze(-1) if mask.mask.dim() == spatial_ig.dim() + 1 else mask.mask
        inv_w = 1.0 - w

        w_sum = w.sum()
        inv_w_sum = inv_w.sum()

        inside = (w * spatial_ig).sum() / (w_sum + eps)
        outside = (inv_w * spatial_ig).sum() / (inv_w_sum + eps)

        alignment = (inside - outside) / (inside + outside + eps)

    return float(alignment.item())


def alignment_diagnostics(spatial_ig, mask, dm, eps: float = 1e-8):
    """
    Diagnostics for the circularity concern in `xai_alignment_signal`.

    MEASURED ON COAST (v13, seed 0, 5 outer iterations):

        iter   AUC     align    corr(ig,mask)   inside   outside
           1  0.9829   0.3892       0.6762      0.0789   0.0347
           2  0.9964   0.9740       0.9807      0.2829   0.0037
           3  0.9977   0.9818       0.9833      0.3869   0.0036
           4  0.9981   0.9851       0.9830      0.4162   0.0031
           5  0.9980   0.9860       0.9825      0.4494   0.0032

    Two facts follow, and they invalidate the v13 claim:

    1. `alignment` was NEVER negative and saturated above 0.98 from
       iteration 2 onward. It is therefore NOT bidirectional in practice,
       which was the entire stated advantage over the 'residual' strategy.
       Lambda pinned to lambda_base * (1 + alpha) = 1.493e-3 and stayed
       there, making the run equivalent to 'fixed' at a slightly larger
       lambda for 8 of 10 iterations.

    2. corr(IG, mask) ~= 0.98. This is structural, not incidental:

           mask       = MinMaxNorm(dm),  dm = sum_b (x - y)^2
           IG target  = sum_p sum_b (x - y)^2     <- the SAME quantity
           d/dx of that target contains the factor 2(x - f(x)) = residual

       So the IG saliency map is, to first order, a re-scaled copy of the
       detection map the mask is built from. Comparing IG mass inside the
       mask against IG mass outside it is comparing the residual against
       itself. `inside` exceeds `outside` by construction.

    Returns a dict; `corr_ig_mask` near 1.0 means the raw signal is
    circular and `alignment_orth` should be used instead.
    """
    import torch

    with torch.no_grad():
        w = mask.mask.squeeze(-1) if mask.mask.dim() == 3 else mask.mask
        ig_f = spatial_ig.flatten().float()
        w_f = w.flatten().float()

        def _corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            return float((a * b).sum() / (a.norm() * b.norm() + eps))

        out = {
            'corr_ig_mask': _corr(ig_f, w_f),
            'ig_mean_inside': float((w * spatial_ig).sum() / (w.sum() + eps)),
            'ig_mean_outside': float(((1 - w) * spatial_ig).sum() / ((1 - w).sum() + eps)),
        }
        if dm is not None:
            out['corr_ig_dm'] = _corr(ig_f, dm.flatten().float())

    return out


def xai_alignment_bg_signal(spatial_ig_bg, mask, eps: float = 1e-8):
    """
    v14 corrected alignment signal, used with a BACKGROUND-TARGETED IG map
    (see xai_core.ig_spatial_saliency's `target_weight` argument).

    WHY THE OBVIOUS FIXES DO NOT WORK
    ---------------------------------
    The first v14 attempt orthogonalized the saliency map against the mask
    and then measured the mask-weighted inside/outside difference. That is
    identically zero for any input:

        (m . ig_orth).sum() = (m_c . ig_orth).sum() + mean(m) * ig_orth.sum()
                            =        0              +        0
                            = 0

    because ig_orth is constructed to have zero covariance with m_c AND
    zero mean. Verified numerically before shipping. More generally: while
    the IG target is the total reconstruction error, IG and the mask are
    both monotone functions of one residual field, so NO function of the
    pair can recover independent information. The target has to change.

    WHAT THIS SIGNAL MEASURES
    -------------------------
    With target_weight = (1 - mask), IG attributes background-only
    reconstruction error. Attribution mass on a masked pixel then means
    that pixel is degrading the background reconstruction elsewhere -
    i.e. it is leaking into the background model. That is the identity-
    mapping failure mode, and it is not readable off the residual map.

        alignment > 0  masked pixels are driving background error
                       -> leakage is happening -> increase suppression
        alignment < 0  masked pixels are NOT driving background error;
                       the mask is likely tracking texture/shadow the
                       model does not blame them for
                       -> ease off, the mask may be wrong

    STATUS
    ------
    The circularity of the v13 signal is MEASURED and settled (Coast,
    corr = 0.98, alignment never negative). This replacement is
    principled but its behaviour on real data is NOT yet established
    across datasets and seeds. Treat it as a hypothesis to be tested by
    the ablation, not as a result. Report it honestly either way.
    """
    import torch

    with torch.no_grad():
        w = mask.mask.squeeze(-1) if mask.mask.dim() == spatial_ig_bg.dim() + 1 else mask.mask
        ig = spatial_ig_bg.float()
        inv = 1.0 - w

        inside = (w * ig).sum() / (w.sum() + eps)
        outside = (inv * ig).sum() / (inv.sum() + eps)

        # Normalized by the spread, not by (inside + outside): the latter
        # saturates at +1 whenever inside >> outside, which is precisely
        # what pinned the v13 signal at 0.98 from iteration 2 onward.
        alignment = torch.tanh((inside - outside) / (ig.std() + eps))

    return float(alignment.item())


def multiscale_gate_signal(spatial_ig, mask, num_layers, alpha_g, g_min, g_max, eps=1e-8):
    """
    Per-layer gate values for MSNet's decoder-side skip-connection
    accumulation (`decoding_sum` in MSNet.forward), reusing the SAME
    Integrated Gradients saliency map already computed for the
    xai_alignment scalar lambda signal in the same boundary_update() call
    - this adds no extra IG cost on top of what xai_alignment already pays.

    Rationale: MSNet's encoder stack already downsamples the scene to
    `num_layers` different scales (down_rate = 2**l, l=0..num_layers-1 -
    see MSNet.__init__). Each decoder layer's output is a reconstruction
    at that scale, fused into the final background estimate via
    unweighted summation (Eq. 8 of Liu et al., 2024). This function asks,
    per scale: does the IG attribution's anomaly-relevant structure
    survive downsampling to this resolution, or does it wash into
    uniform background noise? A scale where it survives is judged
    trustworthy and kept near full contribution; a scale where it washes
    out is judged unreliable at that resolution and gated down.

    Downsampling uses bilinear interpolation to (rows // 2**l, cols //
    2**l) - the identical convention already used by SeT/loss.py's
    MSRLoss for its own multiscale downsampling, not a separately
    invented scaling scheme.

    Only meaningful in combination with strategy='xai_alignment', since
    it depends on the same spatial_ig computed there - see
    AdaptiveLambda.boundary_update.

    Returns a length-`num_layers` list of floats.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        rows, cols = spatial_ig.shape
        mask_2d = mask.mask.squeeze(-1) if mask.mask.dim() == 3 else mask.mask

        gates = []
        scale = 1
        for _l in range(num_layers):
            _rows = max(1, rows // scale)
            _cols = max(1, cols // scale)

            ig_down = F.interpolate(
                spatial_ig.unsqueeze(0).unsqueeze(0), size=(_rows, _cols), mode='bilinear'
            ).squeeze(0).squeeze(0)
            mask_down = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0), size=(_rows, _cols), mode='bilinear'
            ).squeeze(0).squeeze(0)

            w = mask_down
            inv_w = 1.0 - w
            inside = (w * ig_down).sum() / (w.sum() + eps)
            outside = (inv_w * ig_down).sum() / (inv_w.sum() + eps)
            a_l = (inside - outside) / (inside + outside + eps)

            g_l = float(np.clip(1.0 + alpha_g * float(a_l.item()), g_min, g_max))
            gates.append(g_l)
            scale *= 2

    return gates


def spectral_entropy(y, eps: float = 1e-12):
    """
    Shannon entropy of the normalized per-band energy distribution of the
    current reconstruction `y`. Bounded in [0, log(num_bands)].
    """
    import torch

    with torch.no_grad():
        band_energy = (y ** 2).sum(dim=(0, 1))
        probs = band_energy / (band_energy.sum() + eps)
        probs = probs.clamp(min=eps)
        entropy = -(probs * probs.log()).sum().item()

    return entropy


# ----------------------------------------------------------------------
# AdaptiveLambda controller
# ----------------------------------------------------------------------

class AdaptiveLambda:
    """
    Stateful controller that computes an updated `lmda` value every time
    `.step()` / `.update()` is called, following one of `VALID_STRATEGIES`.

    Typical usage inside the training loop (see `SeT/train.py`)::

        adaptive_lambda = AdaptiveLambda(strategy='residual', lambda_base=1e-3, ...)
        ...
        loss_val.backward()
        new_lmda = adaptive_lambda.step(epoch=e, x=x, y=y, mask=mask, model=model)
        criterion.set_lmda(new_lmda)
        optimizer.step()
    """

    def __init__(
        self,
        strategy: str = 'fixed',
        lambda_base: float = 1e-3,
        lambda_min: float = 1e-5,
        lambda_max: float = 1e-1,
        alpha: float = 0.5,
        ema_beta: float = 0.9,
        eps: float = 1e-8,
        logger_: Optional[logging.Logger] = None,
        enable_multiscale_gating: bool = False,
        num_layers: int = 3,
        gating_alpha: float = 0.5,
        gating_min: float = 0.5,
        gating_max: float = 1.5,
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                "Unknown strategy '%s'. Choose one of %s." % (strategy, VALID_STRATEGIES)
            )
        if lambda_min > lambda_max:
            raise ValueError('lambda_min must be <= lambda_max')
        if gating_min > gating_max:
            raise ValueError('gating_min must be <= gating_max')
        if enable_multiscale_gating and strategy not in ('xai_alignment', 'xai_alignment_bg'):
            # Gating reuses the IG saliency map computed inside
            # boundary_update() for strategy='xai_alignment' - it isn't
            # computed at all for any other strategy, so there is nothing
            # to gate from. Fail loudly here rather than silently doing
            # nothing, which is what would otherwise happen.
            raise ValueError(
                "enable_multiscale_gating requires an xai_alignment* strategy "
                "(gating reuses the IG saliency map computed there; no "
                "other strategy computes it)."
            )

        self.strategy = strategy
        self.lambda_base = float(lambda_base)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.alpha = float(alpha)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.logger = logger_ or logger

        self.enable_multiscale_gating = bool(enable_multiscale_gating)
        self.num_layers = int(num_layers)
        self.gating_alpha = float(gating_alpha)
        self.gating_min = float(gating_min)
        self.gating_max = float(gating_max)
        # None until the first boundary_update() with gating enabled -
        # SeT/train.py only assigns this onto model.layer_gates when it
        # is not None, so a run that never reaches a boundary (or has
        # gating disabled) leaves the model's own default (None = paper's
        # original unweighted summation) untouched.
        self.layer_gates = None

        # Current (clipped) lambda value.
        self.lmda = self._clip(self.lambda_base)

        # Running state used by the EMA-residual strategy only.
        self._ema_residual = None
        self._last_diagnostics = {}

        # Full update history: list[dict] with epoch/iteration/lambda/signals.
        self.history = []

    # -- utilities --------------------------------------------------
    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.lambda_min, self.lambda_max))

    def reset(self):
        """Reset the controller to its initial state (useful between runs)."""
        self.lmda = self._clip(self.lambda_base)
        self._ema_residual = None
        self.history = []

    def get(self) -> float:
        return self.lmda

    def lambda_history(self):
        """Flat list of lambda values in update order (for plotting)."""
        return [h['lambda'] for h in self.history]

    def state_dict(self):
        return {
            'strategy': self.strategy,
            'lambda_base': self.lambda_base,
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            'alpha': self.alpha,
            'ema_beta': self.ema_beta,
            'lmda': self.lmda,
            'history': list(self.history),
        }

    # -- strategy implementations ------------------------------------
    def _residual_based(self, background_residual, anomaly_residual):
        ratio = anomaly_residual / (background_residual + self.eps)
        return self.lambda_base * (1.0 + self.alpha * ratio)

    def _ema_residual_based(self, background_residual, anomaly_residual):
        ratio = anomaly_residual / (background_residual + self.eps)
        if self._ema_residual is None:
            self._ema_residual = ratio
        else:
            self._ema_residual = (
                self.ema_beta * self._ema_residual + (1.0 - self.ema_beta) * ratio
            )
        return self.lambda_base * (1.0 + self.alpha * self._ema_residual)

    def _gradient_norm_based(self, grad_norm_value, reference_norm=None):
        reference_norm = 1.0 if reference_norm is None else reference_norm
        scale = reference_norm / (grad_norm_value + self.eps)
        return self.lambda_base * (1.0 + self.alpha * (scale - 1.0))

    def _spectral_entropy_based(self, spectral_entropy_value, max_entropy=None):
        if max_entropy is not None and max_entropy > 0:
            norm_entropy = spectral_entropy_value / (max_entropy + self.eps)
        else:
            norm_entropy = spectral_entropy_value
        return self.lambda_base * (1.0 + self.alpha * norm_entropy)

    def _xai_alignment_based(self, alignment):
        # alignment in ~[-1, 1]; unlike the residual/entropy strategies
        # this can legitimately push lambda below lambda_base, not just
        # above it - that's the point (see xai_alignment_signal docstring).
        return self.lambda_base * (1.0 + self.alpha * alignment)

    # -- public API ----------------------------------------------------
    def update(self, epoch=None, iteration=None, **signals) -> float:
        """
        Compute the new (clipped) lambda from pre-computed scalar signals
        and record it in `self.history`. Missing signals are treated as a
        no-op (lambda is left unchanged) so a training loop never crashes
        because a particular signal could not be computed.
        """
        old_lambda = self.lmda

        try:
            if self.strategy == 'fixed':
                new_lambda = self.lambda_base
            elif self.strategy == 'residual':
                new_lambda = self._residual_based(
                    signals['background_residual'], signals['anomaly_residual']
                )
            elif self.strategy == 'ema_residual':
                new_lambda = self._ema_residual_based(
                    signals['background_residual'], signals['anomaly_residual']
                )
            elif self.strategy == 'gradient_norm':
                new_lambda = self._gradient_norm_based(
                    signals['grad_norm'], signals.get('reference_norm')
                )
            elif self.strategy == 'spectral_entropy':
                new_lambda = self._spectral_entropy_based(
                    signals['spectral_entropy'], signals.get('max_entropy')
                )
            elif self.strategy in ('xai_alignment', 'xai_alignment_bg'):
                # Per-epoch calls (from train_model's step()) intentionally
                # carry no 'alignment' signal - IG is only computed once
                # per outer iteration via boundary_update(), not every
                # epoch (it's ~20x the cost of one forward/backward pass).
                # So a bare per-epoch step() for this strategy is a no-op
                # by design: lambda only moves at mask-update boundaries.
                if 'alignment' in signals:
                    new_lambda = self._xai_alignment_based(signals['alignment'])
                else:
                    new_lambda = old_lambda
            else:  # pragma: no cover - guarded in __init__
                new_lambda = old_lambda
        except KeyError as missing:
            self.logger.debug('Missing signal %s for strategy %s; lambda unchanged.',
                               missing, self.strategy)
            new_lambda = old_lambda

        self.lmda = self._clip(new_lambda)

        record = {
            'epoch': epoch,
            'iteration': iteration,
            'lambda': self.lmda,
            'raw_lambda': float(new_lambda),
            'strategy': self.strategy,
        }
        for k, v in signals.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v
        self.history.append(record)

        self.logger.info(
            'epoch=%s iter=%s strategy=%s lambda=%.6e',
            epoch, iteration, self.strategy, self.lmda,
        )

        return self.lmda

    def compute_signals(self, x=None, y=None, mask=None, model=None):
        """
        Compute only the signals required by the configured strategy,
        directly from tensors already available in the training loop.
        Strategies that don't need a given tensor never touch it, so
        overhead stays minimal.
        """
        signals = {}

        if self.strategy in ('residual', 'ema_residual'):
            bg_res, anm_res = residual_signals(x, y, mask, self.eps)
            signals['background_residual'] = bg_res
            signals['anomaly_residual'] = anm_res

        elif self.strategy == 'gradient_norm':
            signals['grad_norm'] = gradient_norm(model, self.eps)

        elif self.strategy == 'spectral_entropy':
            signals['spectral_entropy'] = spectral_entropy(y)

        return signals

    def step(self, epoch=None, iteration=None, x=None, y=None, mask=None, model=None) -> float:
        """
        Convenience method combining `compute_signals` + `update` in one
        call - this is what the training loop invokes every epoch.
        """
        signals = self.compute_signals(x=x, y=y, mask=mask, model=model)
        return self.update(epoch=epoch, iteration=iteration, **signals)

    def boundary_update(self, iteration, x, model, mask, ig_steps: int = 20) -> float:
        """
        Only meaningful for strategy='xai_alignment'. Called once per
        outer SeT iteration, immediately after `mask.update(dm)` in
        `SeT/train.py::separation_training` - i.e. exactly at the
        boundary where the reported loss/lambda instability was observed,
        using the *freshly updated* mask that will govern the next block
        of epochs.

        Computes IG spatial saliency (via xai_core.ig_spatial_saliency,
        ~ig_steps forward/backward passes - this is the expensive part,
        which is why it happens once per outer iteration and not every
        epoch), derives the IG-mask alignment signal, and recalibrates
        `self.lmda` from it. No-op (returns current lambda unchanged) for
        any other strategy, so wiring this call into the training loop is
        always safe regardless of which strategy is configured.
        """
        if self.strategy not in ('xai_alignment', 'xai_alignment_bg'):
            return self.lmda

        import xai_core

        with_grad_input = x.detach()
        # v14: background-targeted IG for the corrected strategy. This is
        # the structural fix - it changes what IG attributes, which is the
        # only thing that actually decorrelates it from the mask.
        tw = None
        if self.strategy == 'xai_alignment_bg':
            _m = mask.mask.squeeze(-1) if mask.mask.dim() == 3 else mask.mask
            tw = (1.0 - _m).detach()

        spatial_ig = xai_core.ig_spatial_saliency(
            model, with_grad_input, steps=ig_steps, target_weight=tw)

        if self.strategy == 'xai_alignment_bg':
            alignment = xai_alignment_bg_signal(spatial_ig, mask, self.eps)
        else:
            alignment = xai_alignment_signal(spatial_ig, mask, self.eps)

        # Recorded every boundary so the circularity claim is auditable
        # from any run's history, not only from a separate probe script.
        self._last_diagnostics = alignment_diagnostics(
            spatial_ig, mask, dm=None, eps=self.eps)

        if self.enable_multiscale_gating:
            self.layer_gates = multiscale_gate_signal(
                spatial_ig, mask, self.num_layers,
                self.gating_alpha, self.gating_min, self.gating_max, self.eps,
            )

        return self.update(epoch=None, iteration=iteration, alignment=alignment,
                           **self._last_diagnostics)


# ----------------------------------------------------------------------
# Complementary optimization (suggested, opt-in, disabled by default)
# ----------------------------------------------------------------------

class ResidualLRScheduler:
    """
    Residual-driven learning-rate scaling.

    This is the "one additional optimization that complements adaptive
    lambda without changing the MSNet architecture" requested alongside
    the Adaptive Lambda Framework. It reuses the *same*
    `background_residual` / `anomaly_residual` ratio that already drives
    the 'residual' and 'ema_residual' lambda strategies, so it costs no
    extra computation - only an optimizer learning-rate update.

    Rationale: when the anomaly-suppression residual is still large
    relative to the background-reconstruction residual, background/
    anomaly separation is not yet reliable, so a smaller learning rate
    avoids the model overfitting to a noisy separation mask; as the
    ratio drops (separation is working well) the learning rate is
    restored toward its base value. This is entirely decoupled from
    lambda itself - it only *reads* the ratio, it never feeds back into
    `AdaptiveLambda`'s output - and it never touches MSNet or the
    optimizer's parameter groups' structure, only their `lr` field.

    Usage (opt-in, see `main.py`'s `use_residual_lr_scheduler` flag)::

        lr_scheduler = ResidualLRScheduler(optimizer, base_lr=lr)
        ...
        # after adaptive_lambda.step(...) has computed signals this epoch:
        lr_scheduler.step(adaptive_lambda)
    """

    def __init__(self, optimizer, base_lr: float, min_scale: float = 0.2, sensitivity: float = 1.0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_scale = float(min_scale)
        self.sensitivity = float(sensitivity)

    def step(self, adaptive_lambda: 'AdaptiveLambda') -> float:
        """
        Read the most recent residual signal recorded by
        `adaptive_lambda` (if any) and rescale the optimizer's learning
        rate accordingly. No-op (keeps `base_lr`) if no residual signal
        is available yet, e.g. a non-residual strategy is in use.
        """
        scale = 1.0
        if adaptive_lambda.history:
            last = adaptive_lambda.history[-1]
            bg = last.get('background_residual')
            anm = last.get('anomaly_residual')
            if bg is not None and anm is not None:
                ratio = anm / (bg + adaptive_lambda.eps)
                scale = 1.0 / (1.0 + self.sensitivity * ratio)
                scale = max(self.min_scale, min(1.0, scale))

        new_lr = self.base_lr * scale
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr
