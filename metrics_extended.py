"""
metrics_extended.py — the standard HAD evaluation suite.

WHY THIS EXISTS
---------------
`metric.roc_auc` returns AUC(Pd,Pf) only, which is SATURATED on Coast:

    67 anomaly pixels, 9933 background pixels, 665,511 ranking pairs
    AUC 0.9972  ->  ~28 background pixels rank above the anomalies
    AUC 0.9990  ->  ~10 background pixels rank above the anomalies

Moving 0.998 -> 0.999 means reordering about ten pixels out of ten
thousand. That is inside seed noise. Reporting only AUC(Pd,Pf) on a
saturated benchmark leaves you with one number that cannot move for
honest reasons, which is exactly the pressure that produces leakage.

The metrics below (Chang, IEEE TGRS 2021, "Comprehensive Analysis of
Receiver Operating Characteristic (ROC) Curves for Hyperspectral Anomaly
Detection") are the ones the HAD literature actually reports, and several
have two or three orders of magnitude of dynamic range.

    AUC(Pd,Pf)   conventional ROC AUC.              higher better, SATURATES
    AUC(Pd,tau)  detection vs threshold.            higher better
    AUC(Pf,tau)  false alarm vs threshold.          LOWER better, NOT saturated
    AUC_TD       Pd,Pf + Pd,tau. target detectability.        higher better
    AUC_BS       Pd,Pf - Pf,tau. background suppressibility.  higher better
    AUC_TDBS     Pd,tau - Pf,tau.                             higher better
    AUC_ODP      Pd,Pf + Pd,tau - Pf,tau. overall.            higher better
    AUC_SNPR     Pd,tau / Pf,tau. signal-to-noise prob ratio. higher better,
                 unbounded above - this is the discriminative one

Plus a separability measure from the background/anomaly score
distributions, which is what MSNet's own box plots show.

IMPORTANT
---------
These are EVALUATION metrics. They all use ground truth. Do not tune
hyperparameters against any of them - see `unsupervised_selection.py` for
a criterion you can legitimately tune against.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def _flatten(dm, gt):
    dm = np.asarray(dm, dtype=np.float64).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    if dm.shape != gt.shape:
        raise ValueError(f"shape mismatch: dm {dm.shape} vs gt {gt.shape}")
    return dm, gt


def _normalize(dm):
    lo, hi = dm.min(), dm.max()
    return (dm - lo) / (hi - lo + 1e-12)


def auc_pd_pf(dm, gt):
    """Conventional ROC AUC. Saturates on easy scenes."""
    dm, gt = _flatten(dm, gt)
    fpr, tpr, _ = skm.roc_curve(gt, dm)
    return float(skm.auc(fpr, tpr))


def auc_vs_tau(dm, gt, n_tau: int = 1000):
    """
    AUC(Pd,tau) and AUC(Pf,tau): area under detection / false-alarm rate
    plotted against a normalized threshold sweep over [0, 1].

    AUC(Pf,tau) is the one to watch. It is LOWER-is-better and it does not
    saturate, so it separates methods that AUC(Pd,Pf) reports as identical.
    """
    dm, gt = _flatten(dm, gt)
    d = _normalize(dm)

    taus = np.linspace(0.0, 1.0, n_tau)
    anom, bg = d[gt], d[~gt]

    if anom.size == 0:
        raise ValueError("ground truth contains no anomaly pixels")

    pd = np.array([(anom >= t).mean() for t in taus])
    pf = np.array([(bg >= t).mean() for t in taus])

    return float(np.trapezoid(pd, taus)), float(np.trapezoid(pf, taus))


def separability(dm, gt):
    """
    Background/anomaly separability from the score distributions - the
    quantity MSNet's box-plot figures visualise.

    Returns mean(anomaly) - mean(background), normalized by the pooled
    standard deviation. This is an effect size, so it keeps rising after
    AUC(Pd,Pf) has pinned at ~1.0.
    """
    dm, gt = _flatten(dm, gt)
    d = _normalize(dm)
    a, b = d[gt], d[~gt]
    pooled = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
    return float((a.mean() - b.mean()) / pooled)


def evaluate(dm, gt, n_tau: int = 1000):
    """
    Full suite. Returns a dict ready for a results table.

    Use this everywhere instead of bare `metric.roc_auc`.
    """
    a_pd_pf = auc_pd_pf(dm, gt)
    a_pd_tau, a_pf_tau = auc_vs_tau(dm, gt, n_tau)

    return {
        'AUC_Pd_Pf':  a_pd_pf,
        'AUC_Pd_tau': a_pd_tau,
        'AUC_Pf_tau': a_pf_tau,
        'AUC_TD':     a_pd_pf + a_pd_tau,
        'AUC_BS':     a_pd_pf - a_pf_tau,
        'AUC_TDBS':   a_pd_tau - a_pf_tau,
        'AUC_ODP':    a_pd_pf + a_pd_tau - a_pf_tau,
        'AUC_SNPR':   a_pd_tau / (a_pf_tau + 1e-12),
        'separability': separability(dm, gt),
    }


def to_latex_row(name, m, fmt='%.4f'):
    order = ['AUC_Pd_Pf', 'AUC_Pd_tau', 'AUC_Pf_tau', 'AUC_BS',
             'AUC_ODP', 'AUC_SNPR', 'separability']
    cells = []
    for k in order:
        v = m[k]
        cells.append(('%.2f' % v) if k in ('AUC_SNPR', 'separability') else fmt % v)
    return f"{name} & " + " & ".join(cells) + r" \\"


def print_table(results: dict):
    """results: {method_name: metrics_dict}"""
    keys = ['AUC_Pd_Pf', 'AUC_Pd_tau', 'AUC_Pf_tau', 'AUC_BS',
            'AUC_ODP', 'AUC_SNPR', 'separability']
    w = max(len(k) for k in results) + 2
    print(' ' * w + ''.join(f'{k:>13}' for k in keys))
    print('-' * (w + 13 * len(keys)))
    for name, m in results.items():
        row = f'{name:<{w}}'
        for k in keys:
            row += f'{m[k]:>13.4f}' if m[k] < 1000 else f'{m[k]:>13.1f}'
        print(row)
    print()
    print('AUC_Pf_tau is LOWER-is-better. All others higher-is-better.')
