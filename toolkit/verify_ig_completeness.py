"""
verify_ig_completeness.py — prove your IG implementation is correct.

Your v12 notes record a completeness-axiom violation: abs() applied inside the
interpolation loop, before averaging. That is provably non-convergent — no
number of steps fixes it. This script decides, empirically, whether the fix is
actually present in the code you are running now.

THE AXIOM
---------
For a model F, input x, and baseline x':

    sum_i IG_i(x)  ==  F(x) - F(x')

A correct implementation converges to this as steps -> infinity. An
implementation that takes abs() inside the loop converges to something
strictly larger and never closes the gap. You will see it immediately in the
convergence table: correct IG halves its error roughly every doubling of
steps; broken IG plateaus.

USAGE
-----
Fill in the ADAPTER section with your own model and IG function, then:

    python verify_ig_completeness.py

Run the self-test first to confirm the harness itself works:

    python verify_ig_completeness.py --selftest
"""

import argparse
import sys

import numpy as np

STEP_COUNTS = [8, 16, 32, 64, 128, 256]
CONVERGENCE_TOL = 0.02      # 2% relative error at the highest step count
PLATEAU_RATIO = 0.6         # error must shrink by at least this factor


# ============================================================================
#  ADAPTER — edit this to point at your model and your IG function.
# ============================================================================

def get_model_and_input():
    """
    Return (forward_fn, x, baseline).

    forward_fn(x) -> scalar float. This must be the SAME scalar your IG
    attributes. If your IG targets total reconstruction error, forward_fn
    must return total reconstruction error, not the reconstruction itself.
    """
    # ---- EDIT FROM HERE ----------------------------------------------------
    import torch
    from model.MSNet.MSNet import MSNet        # check this path
    from utils import load_dataset

    data, _ = load_dataset("coast")
    x = torch.tensor(data[:, :, :64], dtype=torch.float32).permute(2, 0, 1)[None]

    model = MSNet(in_channels=64)              # check the signature
    model.eval()

    def forward_fn(inp):
        with torch.no_grad():
            recon = model(inp)
            return float(((inp - recon) ** 2).sum())

    baseline = torch.zeros_like(x)
    return forward_fn, x, baseline
    # ---- TO HERE -----------------------------------------------------------


def compute_ig(forward_fn, x, baseline, steps):
    """
    Call YOUR IG implementation with the given step count.
    Must return an array the same shape as x.
    """
    # ---- EDIT FROM HERE ----------------------------------------------------
    from xai_core import integrated_gradients   # check this name

    return integrated_gradients(
        forward_fn=forward_fn, x=x, baseline=baseline, steps=steps)
    # ---- TO HERE -----------------------------------------------------------


# ============================================================================
#  Verifier — no edits needed below.
# ============================================================================

def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def run_check(forward_fn, x, baseline, ig_fn, step_counts):
    target = float(forward_fn(x)) - float(forward_fn(baseline))
    print(f"\nF(x) - F(baseline) = {target:.6f}   (this is what IG must sum to)\n")

    if abs(target) < 1e-9:
        print("!! Target difference is ~0. Choose a baseline that actually")
        print("   changes the output, or the test is vacuous.")
        return None

    print(f"{'steps':>7} {'sum(IG)':>16} {'abs error':>14} {'rel error':>12}")
    print("-" * 54)

    rel_errors = []
    for s in step_counts:
        attributions = _to_numpy(ig_fn(forward_fn, x, baseline, s))
        total = float(attributions.sum())
        err = abs(total - target)
        rel = err / (abs(target) + 1e-12)
        rel_errors.append(rel)
        print(f"{s:>7} {total:>16.6f} {err:>14.6f} {rel:>11.2%}")

    return rel_errors


def verdict(rel_errors, step_counts):
    print("\n" + "=" * 62)
    if rel_errors is None:
        print("INCONCLUSIVE — see message above.")
        print("=" * 62)
        return 2

    final = rel_errors[-1]
    first = rel_errors[0]
    shrink = final / (first + 1e-12)

    print(f"Final relative error at {step_counts[-1]} steps : {final:.2%}")
    print(f"Shrink factor from {step_counts[0]} to {step_counts[-1]} steps: "
          f"{shrink:.3f}")
    print("-" * 62)

    if final <= CONVERGENCE_TOL:
        print("PASS — IG satisfies completeness. The abs()-in-loop fix is")
        print("present in the code you are actually running.")
        print("\nYou can now report the corrected-vs-flawed divergence as a")
        print("methodological finding: run this again against the old")
        print("implementation and quote both curves.")
        print("=" * 62)
        return 0

    if shrink > PLATEAU_RATIO:
        print("FAIL — error is PLATEAUING, not converging. This is the")
        print("signature of abs() (or any non-linear op) applied inside the")
        print("interpolation loop before averaging. More steps will not help.")
        print("\nFix: accumulate SIGNED gradients across the path, average,")
        print("multiply by (x - baseline), and take abs() only afterwards")
        print("if you need magnitudes for visualisation.")
        print("=" * 62)
        return 1

    print("MARGINAL — converging but slowly. Either the step count is too")
    print("low for this model, or the path integral is mis-scaled. Re-run")
    print("with STEP_COUNTS extended to 512/1024 before concluding.")
    print("=" * 62)
    return 1


def selftest():
    """Harness check on a closed-form function where IG is exactly known."""
    print("SELF-TEST: f(x) = sum(w * x^2), baseline 0.")
    rng = np.random.default_rng(0)
    w = rng.normal(size=12)

    def f(v):
        return float((w * np.asarray(v, dtype=float) ** 2).sum())

    x = rng.normal(size=12)
    baseline = np.zeros(12)

    def correct_ig(fn, xx, bb, steps):
        # signed gradients, averaged, then scaled — the right way
        alphas = (np.arange(steps) + 0.5) / steps
        grads = np.zeros_like(np.asarray(xx, dtype=float))
        for a in alphas:
            point = bb + a * (np.asarray(xx, dtype=float) - bb)
            grads += 2.0 * w * point
        return (grads / steps) * (np.asarray(xx, dtype=float) - bb)

    def broken_ig(fn, xx, bb, steps):
        # abs() INSIDE the loop — the documented v12 bug
        alphas = (np.arange(steps) + 0.5) / steps
        grads = np.zeros_like(np.asarray(xx, dtype=float))
        for a in alphas:
            point = bb + a * (np.asarray(xx, dtype=float) - bb)
            grads += np.abs(2.0 * w * point)
        return (grads / steps) * (np.asarray(xx, dtype=float) - bb)

    print("\n--- correct implementation (expect PASS) ---")
    e = run_check(f, x, baseline, correct_ig, STEP_COUNTS)
    rc_good = verdict(e, STEP_COUNTS)

    print("\n--- abs()-in-loop implementation (expect FAIL) ---")
    e = run_check(f, x, baseline, broken_ig, STEP_COUNTS)
    rc_bad = verdict(e, STEP_COUNTS)

    ok = (rc_good == 0 and rc_bad == 1)
    print("\nHARNESS " + ("OK — it distinguishes the two." if ok
                          else "PROBLEM — it failed to distinguish them."))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="verify the harness on a known closed-form function")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    forward_fn, x, baseline = get_model_and_input()
    errs = run_check(forward_fn, x, baseline, compute_ig, STEP_COUNTS)
    return verdict(errs, STEP_COUNTS)


if __name__ == "__main__":
    sys.exit(main())
