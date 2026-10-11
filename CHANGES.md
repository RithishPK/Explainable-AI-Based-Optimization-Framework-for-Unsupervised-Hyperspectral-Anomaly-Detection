# Phase 2 - Adaptive Lambda Optimization Framework

This document explains every change made on top of the original Phase 1
implementation, why it was made, and why it preserves backward
compatibility. Nothing in `dataset/`, `select_bands.py`, `model/`
(MSNet), `SeT/mask.py`, `SeT/detect.py`, `utils.py`, or `metric.py` was
touched.

## Summary of changed / added files

| File | Status | What changed |
|---|---|---|
| `adaptive_lambda.py` | **new** | `AdaptiveLambda` controller + signal helpers |
| `SeT/loss.py` | modified | `TotalLoss.set_lmda()` / `.get_lmda()` added |
| `SeT/train.py` | modified | per-epoch adaptive-lambda update + histories + optional UI callbacks |
| `main.py` | modified | Phase 2 config block, logging, 3 new plots, CSV export |
| `streamlit_app/app.py` | **new** | interactive dashboard |
| `streamlit_app/app_utils.py` | **new** | UI-independent pipeline/plot/export helpers |

---

## 1. `adaptive_lambda.py` (new file)

Implements the `AdaptiveLambda` class with five selectable strategies:

* **`fixed`** - returns `lambda_base` unconditionally. This is what makes
  Phase 2 backward-compatible: with this strategy (or with
  `adaptive_lambda_enabled=False` in `main.py`, which never constructs the
  controller at all) `lmda` behaves exactly like Phase 1.
* **`residual`** - `new_lambda = lambda_base * (1 + alpha * anomaly_residual / background_residual)`.
  Both residuals are computed from tensors the loss already produces
  (`x - y`, split by the current anomaly mask), so this adds only a
  couple of scalar reductions per epoch - no extra forward pass.
* **`ema_residual`** - identical signal, smoothed with an exponential
  moving average (`ema_beta`) to damp epoch-to-epoch noise from a single
  batch.
* **`gradient_norm`** - scales `lambda_base` inversely with the global
  L2 norm of the model's gradients (computed right after
  `loss.backward()`, before `optimizer.step()` clears them next epoch).
  Large/unstable gradients relax lambda; small gradients (near
  convergence) let it grow.
* **`spectral_entropy`** - uses the Shannon entropy of the per-band
  energy distribution of the current reconstruction `y` as a proxy for
  scene complexity; higher entropy raises lambda.

Design choices:

* **Clipping.** Every strategy's raw output is passed through
  `np.clip(value, lambda_min, lambda_max)` before being stored in
  `self.lmda`. There is no code path that can set `self.lmda` outside
  this range.
* **Missing-signal safety.** If a signal a strategy needs isn't supplied
  (e.g. `KeyError`), `update()` falls back to the current lambda value
  (a no-op) instead of raising, so a transient computation issue never
  crashes a multi-hour training run.
* **No architecture coupling.** The three signal-extraction helpers
  (`residual_signals`, `gradient_norm`, `spectral_entropy`) only *read*
  tensors that `SeT/train.py` already has in scope (`x`, `y`, `mask`,
  `model.parameters()`), and only for the strategy actually selected -
  `compute_signals()` skips all others. MSNet itself is never imported
  or touched by this module.
* **History.** Every call to `update()`/`step()` appends a dict
  (`epoch`, `iteration`, `lambda`, `raw_lambda`, `strategy`, plus
  whatever signals were used) to `self.history`. `lambda_history()`
  extracts just the lambda trajectory for plotting.

## 2. `SeT/loss.py`

`SeTLoss.lmda` was already a plain Python `float` attribute (not an
`nn.Parameter`/buffer), so it can be reassigned freely between
optimization steps without touching `forward()` at all - `forward()`
simply re-reads `self.lmda` every call. The only change is adding two
explicit methods to `TotalLoss`:

```python
def set_lmda(self, lmda: float) -> None:
    self.set_loss.lmda = float(lmda)

def get_lmda(self) -> float:
    return self.set_loss.lmda
```

This gives `SeT/train.py` and `adaptive_lambda.py` a stable public API
instead of reaching into `loss.set_loss.lmda` directly. `forward()` is
byte-for-byte unchanged, so when these methods are never called (Phase 1
code path), behaviour is identical to before.

## 3. `SeT/train.py`

### `train_model`

Two additions, both optional and both defaulting to a no-op:

* `adaptive_lambda=None` - when not `None`, right after
  `loss.backward()` (so gradients are populated - required by the
  `gradient_norm` strategy) and before `optimizer.step()`, it calls
  `adaptive_lambda.step(epoch=..., x=x, y=y, mask=..., model=model)` and
  pushes the resulting value into the loss via `criterion.set_lmda(...)`.
  The **next** epoch's forward/backward pass then uses the updated
  lambda. When `adaptive_lambda is None`, this whole block is skipped,
  so the function is byte-for-byte equivalent to Phase 1.
* `callback=None` - an optional `callback(epoch, loss_val, lmda_val)`
  hook fired once per epoch, used by the Streamlit app for live charts.
  Not used by `main.py`'s CLI path, so it costs a single `is None` check
  per epoch there.

`train_model` now also returns `loss_history` (list of per-epoch loss
values) instead of `None`, needed for the "reconstruction loss vs
epochs" plot.

### `separation_training`

* Accepts and forwards `adaptive_lambda`, `callback`, and `iter_callback`
  (the latter fires once per outer iteration with the freshly computed
  AUC, for the Streamlit live AUC chart).
* Aggregates `loss_history`, `lambda_history`, and `auc_per_epoch`
  (the outer-iteration AUC broadcast across every epoch in that
  iteration, so it can share an x-axis with loss/lambda) across all
  `max_iter` outer iterations.
* **Return signature change:** `separation_training` now returns
  `(output_dm, history, extras)` instead of `(output_dm, history)`,
  where `extras` is a dict holding the three histories above. This is
  the one place where the *shape* of the return value changed; the
  *values* of `output_dm` and `history` are numerically identical to
  Phase 1 whenever `adaptive_lambda=None`. `main.py` and the Streamlit
  app were both updated to unpack three values. If you have other code
  calling `separation_training` and want the old two-value signature
  back, wrap it: `dm, hist, _ = separation_training(...)`.

## 4. `main.py`

* Added a clearly-separated Phase 2 configuration block:
  `adaptive_lambda_enabled`, `strategy`, `lambda_base`, `lambda_min`,
  `lambda_max`, `alpha`, `ema_beta` (plus `use_residual_lr_scheduler` /
  `lr_min_scale`, see item 15 below).
* Constructs an `AdaptiveLambda` only if `adaptive_lambda_enabled` is
  `True`; otherwise passes `adaptive_lambda=None` to
  `separation_training`, which is exactly the Phase 1 code path.
* Added `logging` configuration writing every lambda update (one line
  per epoch, from `adaptive_lambda.py`'s internal logger) and every
  outer-iteration AUC to `results/<model>/train.log`, mirrored to the
  console.
* Kept the original ROC and "AUC vs outer-iteration" plots completely
  unchanged, and added three new ones, saved automatically:
  `..._lambda_history.pdf`, `..._loss_history.pdf`, `..._auc_vs_epoch.pdf`.
* Added a single consolidated `<dataset>_phase2_history.csv` (columns:
  `epoch, loss, lambda, auc`) with one row per epoch across the whole
  run, for downstream analysis - the same column layout the Streamlit
  app's "Downloads" tab exports.

## 5. Streamlit application (`streamlit_app/`)

* `app_utils.py` holds a `RunConfig` dataclass, `load_bundled_coast()` /
  `load_uploaded_mat()` for dataset input, `run_pipeline()` (calls the
  *same* `select_bands.OPBS`, `utils.ZScoreNorm`, `MSNet`, `TotalLoss`,
  `Mask`, `separation_training` as `main.py` - no logic is duplicated),
  and a set of `fig_*` / export helpers.
* `app.py` is the UI: sidebar for every hyperparameter (band count,
  layers, LR, epochs, outer iterations, output iteration, adaptive
  on/off toggle, strategy selector, lambda min/max sliders, alpha,
  ema_beta), a "Run Detection" button, and four tabs:
  1. **Live Training** - progress bar + live lambda/loss line charts
     (redrawn periodically to stay responsive) + live per-iteration AUC.
  2. **Detection Maps** - MSNet+SeT map, RX baseline map, ground truth
     (if the uploaded `.mat` contains a `map` variable).
  3. **ROC & AUC** - ROC curve, lambda/loss/AUC-vs-epoch plots, final
     AUC metrics.
  4. **Downloads** - metrics CSV, trained model weights (`.pt`), and PNG
     versions of every plot.
* If the uploaded dataset has no ground truth, ROC/AUC panels show a
  clear "unavailable" message instead of a misleading number; detection
  maps are still shown.

## 6. Additional optimization suggestion (that complements Adaptive
   Lambda without touching MSNet)

**Residual-driven learning-rate scaling.** The same
`background_residual` / `anomaly_residual` signal that drives the
`residual` and `ema_residual` lambda strategies is a reasonable proxy
for how well background/anomaly separation is currently working. A
`ResidualLRScheduler` class, added to `adaptive_lambda.py` right next to
`AdaptiveLambda` (disabled by default via `use_residual_lr_scheduler =
False` in `main.py`), reads that same signal from
`adaptive_lambda.history[-1]` after every `AdaptiveLambda.update()` call
and scales the optimizer's learning rate down when the ratio is large
(separation still poor - avoid overfitting a noisy mask) and back up as
it improves, clipped to `[lr_min_scale * lr, lr]`. It is entirely
decoupled from lambda itself (it only *reads* the signal, never affects
`AdaptiveLambda`'s output), reuses a quantity that's already computed
(no extra overhead beyond a couple of scalar ops per epoch), and never
touches MSNet. `main.py` wires it in by wrapping
`adaptive_lambda.update` rather than editing `SeT/train.py` again, so
the feature is fully opt-in and additive.

## Preserved-by-construction guarantees

* **Dataset loading / OPBS / Z-score norm**: `main.py` and
  `app_utils.run_pipeline` call `select_bands.OPBS` and
  `utils.ZScoreNorm` exactly as Phase 1 did - files `select_bands.py`
  and `utils.py` are unmodified.
* **MSNet**: `model/MSNet/MSNet.py` is unmodified; `net_kwargs` passed to
  it are unchanged.
* **Fixed-lambda equivalence**: with `adaptive_lambda_enabled=False`
  (or `strategy='fixed'`), `criterion.set_lmda()` is either never called
  or always called with `lambda_base`, so the loss computation Phase 1
  performed is reproduced exactly.
* **Clipping**: `AdaptiveLambda._clip()` is the single choke point every
  strategy's output passes through; there is no way to bypass it.
