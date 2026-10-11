# Novelty Tests — `xai_alignment`

Six tests. Each has a concrete pass criterion and a run cost.
A contribution that fails T3 or T5 is not a contribution; stop and reframe.

Current status of each is marked UNRUN / PASS / FAIL. Update as you go.

---

## T1 — Deletion test
*If this contribution vanished from the literature, what becomes impossible?*

**Major novelty:** a problem class becomes unsolvable, or a family of methods
doesn't exist.
**Incremental:** a competent group re-derives it in a week.

**Assessment: INCREMENTAL.** Without `xai_alignment`, MSNet+SeT+ still runs;
λ is set by hand or by the residual heuristic. Nothing becomes impossible.
This is not a defect — most published work is incremental. It is a defect only
if you claim otherwise in the title or abstract.

Status: **assessed, no run needed.**

---

## T2 — Independence test
*Does the mechanism transfer off this backbone?*

`xai_alignment` is defined over MSNet's soft mask M and SeT+'s λ. Remove SeT+
and there is no λ to control. The mechanism is **instance-level, not
mechanism-level.**

To upgrade: state the mechanism abstractly — *any* reconstruction detector with
(a) an error-derived pseudo-mask and (b) a scalar trading reconstruction against
suppression can use an attribution-locality signal to set that scalar. Then
name one other method it would apply to (BiGSeT is the obvious one; both are
identified in Cui et al. as using mask-modified reconstruction).

You do not have to *implement* it elsewhere. Stating transferability precisely
costs nothing and is the difference between "we tuned MSNet" and "we propose a
control signal, demonstrated on MSNet."

Status: **reframing required, no run needed.**

---

## T3 — Counterfactual-baseline test  ← HIGHEST RISK, RUN FIRST
*Does a signal-free baseline with the same freedom capture most of the gain?*

`residual` can only ramp λ upward. `xai_alignment` can move it both ways. So
the comparison `xai_alignment` vs `residual` confounds TWO things:

  (a) λ should be bidirectional
  (b) IG is the right signal for deciding the direction

A blind cosine schedule has (a) and not (b). If it matches `xai_alignment`,
your contribution is (a) — "λ should come back down" — which is a scheduling
observation, not an XAI contribution, and the entire IG apparatus is decoration.

**Critical fairness condition:** the cosine must be RANGE-MATCHED to the λ
trajectory `xai_alignment` actually produces. Calibrate amplitude and midpoint
from the observed `lambda_history` of the xai runs. An arbitrary cosine is a
strawman and a reviewer will say so.

**PASS:** `xai_alignment` beats range-matched `cosine` on ≥4 of 5 paired seeds,
with mean paired delta > 0.
**FAIL:** cosine matches or wins. → The contribution is bidirectional λ
scheduling. Rewrite the claim honestly; it is still worth reporting.

Cost: 5 runs on Coast.
Status: **UNRUN. Do this before the 45-run sweep.**

---

## T4 — Attribution-necessity test
*Is IG doing work, or would any saliency map do?*

Swap IG for plain gradient magnitude (1 backward pass instead of 20–50) and
recompute the alignment score identically.

**PASS:** IG wins meaningfully → the completeness axiom matters, and your
corrected-IG fix (the `abs()`-inside-the-loop bug) has measurable consequence.
That is a real, publishable methodological finding.
**FAIL:** plain gradients match → the contribution is "use a saliency map,"
IG's cost is unjustified, and you should say so. Also still publishable, as a
negative result, and it pairs naturally with the GradCAM stripe finding.

Either outcome is useful. This is the cheapest high-information test you have.

Cost: 5 runs on Coast.
Status: **UNRUN.**

---

## T5 — Circularity test  ← THE ONE A REVIEWER WILL ASK
*Is the alignment score measuring the model's agreement with itself?*

The concern, stated sharply:

  M is derived from the pixelwise reconstruction error.
  IG attributes the reconstruction error.
  alignment = f(M, S_IG).

If S is just a smoothed image of the same error field M is built from, then
alignment is high whenever the model is internally consistent, and it carries
no information about whether the model is *correct*. The loop closes on itself.

**The defence — and it is a real one, so state it explicitly in the paper:**

M is *where* the error is. IG is *what caused* it. These differ because of the
receptive field. Error at pixel p can be caused by p itself, or by p's
neighbours.

  - Anomaly → error is **self-caused**. The pixel's own spectrum is what the
    network cannot reconstruct. IG mass concentrates ON p.
  - Background reconstruction failure (edges, texture, shadow) → error is
    **context-caused**. IG mass spreads across the neighbourhood.

So the alignment score is a **locality-of-attribution** measure, and that is
genuinely not derivable from M alone. This is a much stronger framing than
"XAI agrees with the mask," it explains *why* the mechanism should work rather
than just reporting that it does, and it makes a falsifiable prediction.

**Two things you must now do:**

1. **Verify your implementation actually measures this.** If your alignment
   score is Σ(M·S)/ΣM with S already spatially smoothed or pooled to M's
   resolution, you may have averaged away the exact locality signal the defence
   depends on. Check the spatial support of S before aggregation.

2. **Run the shuffled control.** Apply a fixed seeded spatial permutation to S
   before computing alignment. This preserves S's marginal value distribution
   and destroys its spatial correspondence with M.

**PASS:** shuffled-S runs collapse toward `fixed`/`residual` performance.
**FAIL:** shuffled-S performs like real S → alignment is responding to the
magnitude distribution, not spatial correspondence. The mechanism is circular
and the contribution does not survive.

This test is non-negotiable. Without it you have no answer to the reviewer.

Cost: 5 runs on Coast.
Status: **UNRUN.**

---

## T6 — Consistency test
*Does the effect hold across datasets with consistent sign?*

**PASS:** paired delta positive on ≥2 of 3 datasets, sign-consistent on ≥4 of 5
seeds within each.
**FAIL:** works on Coast only → report as dataset-specific, do not generalise.

Cost: the 45-run sweep (35 new after Coast).
Status: **UNRUN.**

---

## Verdict template

Fill this in and give it to Dr. Sudha before writing the results chapter.

| Test | Result | Consequence |
|------|--------|-------------|
| T1 Deletion | Incremental | Title must not claim a framework-level advance |
| T2 Independence | Backbone-bound | State transferability abstractly in §Method |
| T3 Cosine baseline | ? | If FAIL → claim becomes "bidirectional λ scheduling" |
| T4 IG vs gradient | ? | If FAIL → report as negative result, drop IG cost claim |
| T5 Shuffled saliency | ? | If FAIL → contribution does not survive |
| T6 Cross-dataset | ? | If FAIL → scope claim to Coast |

**Tier assignment after all six:**

- T3 ✓ T4 ✓ T5 ✓ T6 ✓ → solid incremental contribution, defensible at TGRS
  workshop level. Not major novelty. Do not claim it is.
- T3 ✓ T5 ✓, T4 ✗ → contribution is the closed loop, not IG specifically.
  Retitle accordingly.
- T5 ✗ → stop. Reframe around the leakage-free protocol and the IMP finding,
  which are real and already yours.

---

## Sequencing correction

Earlier advice was "ablation table first, nothing else." That was wrong in one
respect: T3, T4 and T5 are 15 runs on Coast that can invalidate the primary
contribution. Running the 45-cell sweep before them risks spending an evening
of compute defending a mechanism that a 25-run gate would have killed.

**Phase A (Coast only, 5 arms × 5 seeds = 25 runs):**
  fixed, residual, xai_alignment, cosine, xai_shuffled
  → decision gate on T3 and T5

**Phase B (only if Phase A passes):**
  3 datasets × 3 arms × 5 seeds, reusing the 15 Coast cells = 30 new runs
  → T6 and the main results table
