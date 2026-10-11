"""
unsupervised_selection.py — how to tune WITHOUT touching test labels.

THE PROBLEM
-----------
Hyperspectral anomaly detection is unsupervised: there is no validation
split. The ground-truth map is the test set and the only set. So every
hyperparameter chosen by looking at AUC - alpha, lambda_max, ig_steps,
num_layers, epochs, num_bs - is chosen on test labels. That is the same
error as checkpoint-scanning, relocated from the epoch loop to the config
block, where it leaves no trace in the code and no reviewer can catch it.

"We tuned until AUC went up" is not a protocol. It is leakage with extra
steps.

THE FIX
-------
Select on a criterion computable from the data and the model alone, with
the ground truth never loaded. Fix the configuration. THEN evaluate once
against ground truth and report that number, whatever it is.

Three label-free signals, combined below:

1. MASK STABILITY. Separation training is supposed to converge on a
   consistent anomaly/background assignment. If the soft mask keeps
   churning between outer iterations, separation is not working -
   regardless of what AUC says. Measured as mean correlation between
   consecutive masks.

2. SCORE BIMODALITY. A working detector produces a heavy background mode
   plus a sparse high-score tail. A failing one produces a single diffuse
   blob. Measured by excess kurtosis of the score map, which is high when
   a sparse tail exists and low when the distribution is unimodal-diffuse.

3. BACKGROUND RECONSTRUCTION QUALITY. Restricted to the low-score
   pixels only (the model's own estimate of background, no labels), the
   reconstruction error should be small. Rising background error means
   the model is degrading, which is the identity-mapping regime.

USAGE
-----
    sel = UnsupervisedSelector()
    for iteration in ...:
        sel.observe(mask=mask, score_map=dm)
    score = sel.criterion()      # higher is better, no ground truth used

Sweep configurations, keep the argmax of `criterion()`, then evaluate
once with metrics_extended.evaluate(). Report the AUC you get.

HONESTY REQUIREMENT
-------------------
This criterion is a PROXY. It is not guaranteed to correlate with AUC,
and you must check whether it does - `rank_correlation_report` below does
that check. If it correlates poorly, say so in the paper; a proxy that
does not track the target is itself a reportable finding, and it is far
better than silently tuning on test labels.
"""

from __future__ import annotations

import numpy as np


def _np(t):
    if hasattr(t, 'detach'):
        t = t.detach().cpu()
    if hasattr(t, 'numpy'):
        t = t.numpy()
    return np.asarray(t, dtype=np.float64)


def mask_stability(masks) -> float:
    """
    Mean Pearson correlation between consecutive soft masks.
    1.0 = fully converged, 0.0 = churning. No labels used.
    """
    if len(masks) < 2:
        return 0.0
    cors = []
    for a, b in zip(masks[:-1], masks[1:]):
        a, b = _np(a).ravel(), _np(b).ravel()
        a, b = a - a.mean(), b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        cors.append(float((a * b).sum() / (den + 1e-12)))
    return float(np.mean(cors))


def score_bimodality(score_map) -> float:
    """
    Excess kurtosis of the normalized score map. High = sparse
    high-score tail over a tight background mode (what a working
    detector produces). Low = diffuse unimodal blob. No labels used.
    """
    s = _np(score_map).ravel()
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    mu, sd = s.mean(), s.std() + 1e-12
    return float(((s - mu) ** 4).mean() / sd ** 4 - 3.0)


def background_reconstruction_error(score_map, percentile: float = 80.0) -> float:
    """
    Mean score over the lowest-scoring `percentile`% of pixels - the
    model's OWN estimate of background, so no labels are involved.
    Lower is better. Rising values indicate the identity-mapping regime.
    """
    s = _np(score_map).ravel()
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    cut = np.percentile(s, percentile)
    bg = s[s <= cut]
    return float(bg.mean()) if bg.size else float('nan')


class UnsupervisedSelector:
    """Accumulates label-free signals across outer iterations."""

    def __init__(self, w_stability=1.0, w_bimodality=1.0, w_bg=1.0):
        self.masks = []
        self.scores = []
        self.w = (float(w_stability), float(w_bimodality), float(w_bg))

    def observe(self, mask=None, score_map=None):
        if mask is not None:
            m = mask.mask if hasattr(mask, 'mask') else mask
            self.masks.append(_np(m).squeeze())
        if score_map is not None:
            self.scores.append(_np(score_map))

    def components(self) -> dict:
        if not self.scores:
            raise RuntimeError("no observations recorded")
        return {
            'mask_stability': mask_stability(self.masks),
            'score_bimodality': score_bimodality(self.scores[-1]),
            'bg_error': background_reconstruction_error(self.scores[-1]),
        }

    def criterion(self) -> float:
        """
        Single scalar, higher is better, ground truth never touched.

        Bimodality is log-compressed because kurtosis is unbounded and
        would otherwise dominate the other two terms entirely.
        """
        c = self.components()
        ws, wb, wg = self.w
        return float(
            ws * c['mask_stability']
            + wb * np.log1p(max(c['score_bimodality'], 0.0))
            - wg * c['bg_error']
        )


def rank_correlation_report(criterion_values, auc_values):
    """
    Does the proxy actually track AUC? Spearman rank correlation across
    the configurations you swept.

    Run this ONCE, after the sweep, purely as a methodological check -
    and report the result honestly. |rho| < 0.3 means the proxy is not
    tracking, which you must state rather than quietly fall back to
    tuning on AUC.
    """
    from scipy.stats import spearmanr
    rho, p = spearmanr(criterion_values, auc_values)
    verdict = ('proxy tracks AUC' if abs(rho) > 0.6 else
               'weak - report as a limitation' if abs(rho) > 0.3 else
               'proxy does NOT track AUC - report this as a negative result')
    return {'spearman_rho': float(rho), 'p_value': float(p), 'verdict': verdict}
