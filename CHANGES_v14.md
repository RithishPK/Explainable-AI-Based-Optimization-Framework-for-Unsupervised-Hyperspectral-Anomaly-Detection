# v14 — Changes

All findings below are MEASURED on Coast (seed 0, 150 epochs/iter), not asserted.

---

## FINDING 1 — the v13 `xai_alignment` signal is circular and saturating

Measured, v13 code, unmodified:

```
iter   AUC     alignment   corr(IG,mask)   inside   outside
   1  0.9829     0.3892        0.6762      0.0789   0.0347
   2  0.9964     0.9740        0.9807      0.2829   0.0037
   3  0.9977     0.9818        0.9833      0.3869   0.0036
   4  0.9981     0.9851        0.9830      0.4162   0.0031
   5  0.9980     0.9860        0.9825      0.4494   0.0032
```

**alignment was never negative and exceeded 0.98 from iteration 2 onward.**
Lambda pinned at `lambda_base * (1 + alpha) = 1.493e-3` and stayed there.
For 8 of 10 iterations the run is equivalent to `fixed` at a slightly
larger lambda.

This invalidates the v13 claim. The stated advantage over `residual` was
bidirectionality — "it is not constrained to only increase". On real data
it only increased, then stopped moving.

### Why — structural, not a coding bug

```
mask       = MinMaxNorm(dm),   dm = sum_b (x - y)^2      SeT/mask.py
IG target  = sum_p sum_b (x - y)^2                       xai_core.py
```

The same quantity. The gradient of that target contains the factor
`2(x - f(x))` — the residual itself. So the IG saliency map is, to first
order, a rescaled copy of the detection map the mask is built from.
Comparing IG mass inside the mask against outside it compares the
residual with itself; `inside > outside` holds by construction.

---

## FINDING 2 — post-processing cannot fix it (first attempt, rejected)

The first v14 attempt orthogonalized IG against the mask, then measured
the mask-weighted inside/outside difference. It is identically zero:

```
(m . ig_orth).sum() = (m_c . ig_orth).sum() + mean(m) * ig_orth.sum()
                    =        0              +        0              = 0
```

Verified numerically before shipping. Generalizing: while the IG target
is the total reconstruction error, IG and the mask are both monotone
functions of one residual field, so **no** function of the pair recovers
independent information. The target must change.

---

## CHANGE 1 — background-targeted IG (`xai_alignment_bg`)

`xai_core.ig_spatial_saliency` gains `target_weight`. Passing `(1 - mask)`
restricts the attributed target to background-only reconstruction error.
The question becomes "which input pixels drive failure to model the
background", which is not the field the mask is built from. Attribution
mass on masked pixels then means those pixels are contaminating
background modelling elsewhere — the identity-mapping failure mode.

### Measured effect — PARTIAL, NOT A FIX

```
iter   AUC    | corr TOTAL  v13 align | corr BG   v14 align   lambda
   1  0.9829  |    0.6762     0.3892  |  0.5739     0.5528   1.276e-03
   2  0.9958  |    0.9800     0.9739  |  0.8060     1.0000   1.500e-03
   3  0.9975  |    0.9834     0.9821  |  0.7703     1.0000   1.500e-03
```

* Correlation drops **0.98 -> 0.77**. Real, measurable, reproducible.
* **The signal still saturates** (tanh reaches 1.0 by iteration 2) and
  lambda still pins at the ceiling.

**Status: OPEN.** Do not claim this is solved. Some residual correlation
is expected and even desirable — masked pixels genuinely do influence
neighbouring background error through the receptive field, and that is
the leakage signal. But 0.77 is too high to separate signal from
circularity, and the saturation is unresolved. Report it as-is.

---

## CHANGE 2 — `main.py` default strategy

Was `'residual'`, silently contradicting the stated primary contribution
in every default run. Now `'xai_alignment_bg'`.

`'xai_alignment'` (v13) is retained in `VALID_STRATEGIES` so the failure
above can be reproduced as an ablation row — not because it should be used.

## CHANGE 3 — determinism in `main.py`

Added seeding of `random`, `numpy`, `torch`, `cuda`, plus
`cudnn.deterministic=True` / `benchmark=False`. Without these, multi-seed
mean ± std is not reproducible.

## CHANGE 4 — leakage guardrails

* `main.py` now logs the full AUC trace, the reportable value at
  `output_iter`, and the peak with an explicit "do not report" warning.
* `SeT/train.py` early stopping logs a loud warning: it stops on
  `history`, which is AUC against ground truth. Off by default. Must stay off.

**Note on the Phase 1 report.** `main.py` was never the leak.
`output_iter = 5` is fixed a priori and correct. The report quoted
**0.9975** (the peak, iter 6–7) while `output_iter=5` gives **0.9972**.
The leakage was in the write-up, not the code. Fix the number, not the loop.

## CHANGE 5 — `xai_core.py` docstring

Said `|grad| accumulation`; the code correctly accumulates the **signed**
gradient. Both `xai_core.py` and `streamlit_app/xai_utils.py` are correct.
The docstring was stale. Corrected.

## CHANGE 6 — `alignment_diagnostics()`

New. Records `corr(IG, mask)` at every boundary into the run history, so
the circularity is auditable from any run rather than a separate probe.

---

## Still open

* `utils.rx` uses full-image covariance. Background-only covariance is the
  fair baseline; full-image inflates MSNet's apparent margin.
* `xai_alignment_bg` saturation.
* Multiscale gating (`multiscale_gate_signal`) inherits the same IG map
  and therefore the same circularity. Untested. Do not enable for results.
