"""
lambda_controls.py — control arms for the novelty tests in NOVELTY_TESTS.md.

These are the experiments that decide whether `xai_alignment` is a contribution.
Wire them as additional values of your existing `lambda_strategy` switch.

Three controls:

  CosineLambda   (T3) — bidirectional lambda with NO signal. Range-matched to
                        the trajectory xai_alignment actually produced.
  ShuffledSaliency (T5) — real IG values, destroyed spatial correspondence.
  UniformSaliency  (T5b) — constant saliency map; alignment reduces to a
                        normalisation constant.

None of these touch ground truth. All are deterministic given a seed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# T3 — range-matched cosine schedule
# ---------------------------------------------------------------------------

def calibrate_cosine(lambda_histories):
    """
    Derive cosine parameters from observed xai_alignment lambda trajectories.

    MUST be called on the lambda_history arrays saved by your xai_alignment
    runs. An arbitrary cosine is a strawman baseline and a reviewer will
    dismiss the comparison.

    Args:
        lambda_histories: list of per-run lambda trajectories (list of floats),
                          one per xai_alignment seed.

    Returns:
        dict with midpoint and amplitude matched to the observed range.
    """
    if not lambda_histories:
        raise ValueError("No lambda histories supplied. Run xai_alignment first.")

    flat = np.concatenate([np.asarray(h, dtype=float) for h in lambda_histories])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("lambda_history contained no finite values.")

    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-8:
        raise ValueError(
            f"xai_alignment lambda was effectively constant ({lo:.6f}). "
            "It is not exercising bidirectional control, which undercuts the "
            "primary claim. Investigate before running controls."
        )

    return {
        "midpoint":  (hi + lo) / 2.0,
        "amplitude": (hi - lo) / 2.0,
        "observed_min": lo,
        "observed_max": hi,
    }


class CosineLambda:
    """
    Blind bidirectional lambda. Same range and same number of direction
    changes as xai_alignment, but zero information content.

    If this matches xai_alignment, the contribution is "lambda should be
    bidirectional", not "IG decides the direction".
    """

    def __init__(self, midpoint, amplitude, n_iterations, n_cycles=1.0):
        self.midpoint = float(midpoint)
        self.amplitude = float(amplitude)
        self.n_iterations = int(n_iterations)
        self.n_cycles = float(n_cycles)
        self.history = []

    def step(self, iteration, **_ignored):
        """Called at each mask-update boundary. Ignores all model state."""
        phase = 2.0 * np.pi * self.n_cycles * iteration / max(self.n_iterations, 1)
        lam = self.midpoint + self.amplitude * np.cos(phase)
        lam = float(max(lam, 0.0))
        self.history.append(lam)
        return lam


# ---------------------------------------------------------------------------
# T5 — saliency controls
# ---------------------------------------------------------------------------

class ShuffledSaliency:
    """
    Applies a fixed seeded spatial permutation to the IG saliency map before
    the alignment score is computed.

    Preserves: the marginal distribution of saliency values, the mean, the
    variance, the compute cost.
    Destroys: spatial correspondence with the soft mask M.

    If alignment-driven lambda still works under this, the alignment score is
    responding to saliency MAGNITUDE, not to spatial agreement, and the
    locality argument in NOVELTY_TESTS.md T5 does not hold.
    """

    def __init__(self, seed, shape):
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(int(np.prod(shape)))
        self.shape = shape

    def __call__(self, saliency):
        s = np.asarray(saliency)
        if s.shape != self.shape:
            raise ValueError(f"Expected saliency shape {self.shape}, got {s.shape}")
        return s.flatten()[self.perm].reshape(self.shape)


class UniformSaliency:
    """
    Replaces the saliency map with a constant equal to its mean.

    Under this, any alignment score of the form sum(M*S)/sum(M) collapses to
    that constant. If performance is unchanged, the alignment score was never
    carrying spatial information at all — a stronger failure than T5.
    """

    def __call__(self, saliency):
        s = np.asarray(saliency)
        return np.full_like(s, float(s.mean()))


# ---------------------------------------------------------------------------
# Diagnostic — run this BEFORE the controls
# ---------------------------------------------------------------------------

def diagnose_alignment_signal(mask, saliency, saliency_shuffled):
    """
    Cheap sanity check on a single trained checkpoint. Costs no training.

    If real and shuffled alignment scores are already indistinguishable here,
    T5 will fail and you can skip the 5 training runs.

    Returns a dict; inspect `separation` — if it is near zero, stop.
    """
    m = np.asarray(mask, dtype=float)
    denom = m.sum() + 1e-12

    a_real = float((m * np.asarray(saliency, dtype=float)).sum() / denom)
    a_shuf = float((m * np.asarray(saliency_shuffled, dtype=float)).sum() / denom)

    s = np.asarray(saliency, dtype=float)
    scale = float(s.mean()) + 1e-12

    return {
        "alignment_real": a_real,
        "alignment_shuffled": a_shuf,
        "separation": (a_real - a_shuf) / scale,
        "verdict": (
            "signal present" if abs(a_real - a_shuf) / scale > 0.10
            else "NO SEPARATION — T5 will likely fail, investigate before training"
        ),
    }


if __name__ == "__main__":
    # Smoke test on synthetic data with a planted anomaly.
    rng = np.random.default_rng(0)
    shape = (100, 100)

    mask = rng.random(shape) * 0.1
    mask[40:45, 40:45] = 0.9          # anomaly cluster

    sal = rng.random(shape) * 0.1
    sal[40:45, 40:45] = 0.9           # attribution agrees -> should separate

    shuffler = ShuffledSaliency(seed=0, shape=shape)
    print(diagnose_alignment_signal(mask, sal, shuffler(sal)))

    cos = CosineLambda(midpoint=1.0, amplitude=0.4, n_iterations=10)
    print([round(cos.step(i), 3) for i in range(10)])
