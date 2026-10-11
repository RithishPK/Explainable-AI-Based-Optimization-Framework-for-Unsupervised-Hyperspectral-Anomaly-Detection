"""
run_ablation.py — two-phase ablation runner with novelty-test control arms.

CHANGE FROM v1: added Phase A. The 45-cell sweep is now gated behind a 25-run
Coast-only block containing the two control arms that can invalidate the
primary contribution. Running the full sweep first risks defending a mechanism
a cheaper test would have killed.

  Phase A — Coast, 5 arms x 5 seeds = 25 runs
            arms: fixed, residual, xai_alignment, cosine, xai_shuffled
            gate: T3 (cosine) and T5 (shuffled) from NOVELTY_TESTS.md

  Phase B — 3 datasets x 3 arms x 5 seeds, reusing Coast = 30 new runs
            only runs if Phase A passes

DESIGN CONSTRAINTS (unchanged, do not relax):
  1. Fixed epoch budget. Reported AUC is the FINAL-iteration AUC. No checkpoint
     is ever selected by looking at ground truth.
  2. The AUC trace is recorded for the convergence figure only. Never argmax it.
  3. Resumable: one JSON per cell.
  4. Paired seeds across arms, so per-seed deltas are valid evidence.

Usage:
    python run_ablation.py --dry-run
    python run_ablation.py --phase A
    python run_ablation.py --gate            # evaluate T3/T5, prints verdict
    python run_ablation.py --phase B
    python run_ablation.py --aggregate-only
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from itertools import product

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG — fixed a priori. Do not tune after seeing results.
# ----------------------------------------------------------------------------

PRIMARY_DATASET = "coast"
DATASETS   = ["coast", "gulfport", "pavia"]
MAIN_ARMS  = ["fixed", "residual", "xai_alignment"]
CONTROL_ARMS = ["cosine", "xai_shuffled"]
SEEDS      = [0, 1, 2, 3, 4]

ITERATIONS      = 10
EPOCHS_PER_ITER = 150
LAMBDA_INIT     = 1.0
N_BANDS         = 64

RESULTS_DIR = "ablation_results"
DEVICE      = "cuda"

# Gate thresholds — set before running, not after.
T3_MIN_WINS = 4      # xai must beat cosine on >= 4 of 5 paired seeds
T5_MIN_DELTA = 0.001  # xai must beat shuffled by this much on average

# ============================================================================
#  ADAPTER — THE ONLY SECTION YOU EDIT.
# ============================================================================

def run_single(dataset, strategy, seed, iterations, epochs_per_iter,
               lambda_init, n_bands, device):
    """
    Train one model end-to-end, return final-iteration metrics.

    Must return {"auc_pd_pf": float, "auc_trace": [float, ...]}.
    auc_pd_pf MUST be trace[-1], never max(trace).

    For control arms, see lambda_controls.py:
      "cosine"        -> CosineLambda, calibrated via calibrate_cosine() on the
                         lambda_history of your completed xai_alignment runs
      "xai_shuffled"  -> xai_alignment with ShuffledSaliency applied to the IG
                         map before the alignment score is computed
    """
    # ---- EDIT FROM HERE -----------------------------------------------------
    from main import train_msnet
    from utils import load_dataset

    data, gt = load_dataset(dataset)

    kwargs = dict(
        data=data, n_bands=n_bands,
        iterations=iterations, epochs_per_iter=epochs_per_iter,
        lambda_init=lambda_init, device=device,
        select_best_by_auc=False,     # MUST stay False
        ground_truth=gt,              # eval only, never touches the loss
    )

    if strategy == "cosine":
        from lambda_controls import CosineLambda, calibrate_cosine
        hist = _load_xai_lambda_histories(dataset)
        params = calibrate_cosine(hist)
        kwargs["lambda_strategy"] = "external"
        kwargs["lambda_controller"] = CosineLambda(
            midpoint=params["midpoint"], amplitude=params["amplitude"],
            n_iterations=iterations)

    elif strategy == "xai_shuffled":
        from lambda_controls import ShuffledSaliency
        kwargs["lambda_strategy"] = "xai_alignment"
        kwargs["saliency_transform"] = ShuffledSaliency(
            seed=seed, shape=data.shape[:2])

    else:
        kwargs["lambda_strategy"] = strategy

    out = train_msnet(**kwargs)
    trace = [float(a) for a in out["auc_history"]]

    return {
        "auc_pd_pf":    trace[-1],
        "auc_trace":    trace,
        "auc_pf_tau":   float(out.get("auc_pf_tau", float("nan"))),
        "separability": float(out.get("separability", float("nan"))),
        "final_lambda": float(out.get("final_lambda", float("nan"))),
        "lambda_trace": [float(x) for x in out.get("lambda_history", [])],
    }
    # ---- TO HERE ------------------------------------------------------------


# ============================================================================
#  Nothing below needs editing.
# ============================================================================

def _load_xai_lambda_histories(dataset):
    """Collect lambda trajectories from completed xai_alignment runs."""
    hist = []
    for sd in SEEDS:
        p = cell_path(dataset, "xai_alignment", sd)
        if os.path.exists(p):
            with open(p) as f:
                lt = json.load(f).get("lambda_trace", [])
                if lt:
                    hist.append(lt)
    if not hist:
        raise RuntimeError(
            "Cannot calibrate the cosine control: no completed xai_alignment "
            f"runs on {dataset}. Run those arms first — an uncalibrated cosine "
            "is a strawman baseline.")
    return hist


def set_all_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def cell_path(dataset, strategy, seed):
    return os.path.join(RESULTS_DIR, f"{dataset}__{strategy}__seed{seed}.json")


def execute_cell(dataset, strategy, seed):
    path = cell_path(dataset, strategy, seed)
    if os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)}")
        return "skipped"

    print(f"  [run ] {dataset} / {strategy} / seed {seed} ...", end="", flush=True)
    set_all_seeds(seed)
    t0 = time.time()
    try:
        metrics = run_single(
            dataset=dataset, strategy=strategy, seed=seed,
            iterations=ITERATIONS, epochs_per_iter=EPOCHS_PER_ITER,
            lambda_init=LAMBDA_INIT, n_bands=N_BANDS, device=DEVICE)
    except Exception:
        print(" FAILED")
        traceback.print_exc()
        return "failed"

    elapsed = time.time() - t0

    trace = metrics.get("auc_trace", [])
    if trace and abs(metrics["auc_pd_pf"] - trace[-1]) > 1e-9:
        raise RuntimeError(
            "auc_pd_pf != final trace value. Ground-truth model selection "
            "detected. Abort.")

    record = {
        "dataset": dataset, "strategy": strategy, "seed": seed,
        "iterations": ITERATIONS, "epochs_per_iter": EPOCHS_PER_ITER,
        "total_epochs": ITERATIONS * EPOCHS_PER_ITER,
        "lambda_init": LAMBDA_INIT, "n_bands": N_BANDS, "device": DEVICE,
        "selection": "final_iteration_fixed_budget",
        "train_seconds": round(elapsed, 2),
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        **metrics,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f" AUC={metrics['auc_pd_pf']:.4f}  ({elapsed/60:.1f} min)")
    return "ok"


def load_all():
    rows = []
    if not os.path.isdir(RESULTS_DIR):
        return rows
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                rows.append(json.load(f))
    return rows


def _paired(rows, ds, arm_a, arm_b):
    a = {r["seed"]: r["auc_pd_pf"] for r in rows
         if r["dataset"] == ds and r["strategy"] == arm_a}
    b = {r["seed"]: r["auc_pd_pf"] for r in rows
         if r["dataset"] == ds and r["strategy"] == arm_b}
    shared = sorted(set(a) & set(b))
    return shared, np.array([a[s] - b[s] for s in shared])


def gate(rows):
    """Evaluate T3 and T5. Prints a verdict and returns True if both pass."""
    ds = PRIMARY_DATASET
    print("\n" + "=" * 70)
    print("NOVELTY GATE — Phase A results")
    print("=" * 70)

    passed = True

    for label, arm, crit in [
        ("T3  xai_alignment vs cosine     (is IG doing more than scheduling?)",
         "cosine", "wins"),
        ("T5  xai_alignment vs shuffled   (is alignment spatial or circular?)",
         "xai_shuffled", "delta"),
    ]:
        shared, d = _paired(rows, ds, "xai_alignment", arm)
        print(f"\n{label}")
        if len(shared) == 0:
            print("   INCOMPLETE — control arm not yet run.")
            passed = False
            continue
        wins = int((d > 0).sum())
        print(f"   deltas = {np.array2string(d, precision=4)}")
        print(f"   mean   = {d.mean():+.4f}   wins = {wins}/{len(d)}")

        if crit == "wins":
            ok = wins >= T3_MIN_WINS and d.mean() > 0
            print(f"   -> {'PASS' if ok else 'FAIL'} "
                  f"(need >={T3_MIN_WINS}/5 wins and positive mean)")
            if not ok:
                print("      Consequence: the contribution is bidirectional "
                      "lambda scheduling, not IG. Rewrite the claim.")
        else:
            ok = d.mean() > T5_MIN_DELTA
            print(f"   -> {'PASS' if ok else 'FAIL'} "
                  f"(need mean delta > {T5_MIN_DELTA})")
            if not ok:
                print("      Consequence: alignment responds to saliency "
                      "magnitude, not spatial correspondence. Circular. Stop.")
        passed = passed and ok

    print("\n" + "=" * 70)
    print("GATE: " + ("PASS — proceed to Phase B" if passed
                      else "FAIL — do NOT run Phase B. Reframe first."))
    print("=" * 70)
    return passed


def aggregate(rows):
    if not rows:
        print("No results found.")
        return

    arms = MAIN_ARMS + CONTROL_ARMS
    print("\n" + "=" * 88)
    print("MEAN +/- STD  AUC(Pd,Pf)   [final iteration, fixed budget]")
    print("=" * 88)
    print(f"{'dataset':<11}" + "".join(f"{a:>16}" for a in arms))
    print("-" * 88)

    summary = {}
    for ds in DATASETS:
        line = f"{ds:<11}"
        for st in arms:
            vals = [r["auc_pd_pf"] for r in rows
                    if r["dataset"] == ds and r["strategy"] == st]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                summary[(ds, st)] = (m, s, len(vals))
                line += f"{m:>9.4f}±{s:.4f}"
            else:
                summary[(ds, st)] = (float("nan"), float("nan"), 0)
                line += f"{'—':>16}"
        print(line)

    print("\n" + "=" * 88)
    print("PAIRED PER-SEED DELTAS (same seed = same init and same OPBS bands)")
    print("=" * 88)
    for ds in DATASETS:
        for ref in ["residual", "cosine", "xai_shuffled"]:
            shared, d = _paired(rows, ds, "xai_alignment", ref)
            if len(shared) == 0:
                continue
            print(f"{ds:<11} xai vs {ref:<14} "
                  f"mean={d.mean():+.4f}  wins={int((d > 0).sum())}/{len(d)}")

    # LaTeX — main arms only; controls belong in the text, not the headline table
    tex = [
        r"\begin{table}[!t]",
        r"\caption{Ablation over adaptive $\lambda$ strategy. AUC$(P_d,P_f)$,",
        r"mean $\pm$ std over %d seeds under an identical fixed budget of %d"
        % (len(SEEDS), ITERATIONS * EPOCHS_PER_ITER),
        r"epochs. The final-iteration model is reported; no ground-truth-based",
        r"checkpoint selection is performed.}",
        r"\label{tab:ablation_lambda}", r"\centering",
        r"\begin{tabular}{l" + "c" * len(MAIN_ARMS) + "}", r"\toprule",
        "Dataset & " + " & ".join(a.replace("_", r"\_") for a in MAIN_ARMS) + r" \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        vals = [summary[(ds, st)] for st in MAIN_ARMS]
        finite = [v[0] for v in vals if not np.isnan(v[0])]
        best = max(finite) if finite else None
        cells = []
        for m, s, n in vals:
            if n == 0:
                cells.append(r"\TODO{}")
            elif best is not None and abs(m - best) < 1e-12:
                cells.append(rf"$\mathbf{{{m:.4f} \pm {s:.4f}}}$")
            else:
                cells.append(rf"${m:.4f} \pm {s:.4f}$")
        tex.append(f"{ds.capitalize()} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ablation_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    with open(os.path.join(RESULTS_DIR, "ablation_raw.csv"), "w") as f:
        f.write("dataset,strategy,seed,auc_pd_pf,auc_pf_tau,separability,"
                "final_lambda,train_seconds,git_commit\n")
        for r in rows:
            f.write(f"{r['dataset']},{r['strategy']},{r['seed']},"
                    f"{r['auc_pd_pf']:.6f},"
                    f"{r.get('auc_pf_tau', float('nan')):.6f},"
                    f"{r.get('separability', float('nan')):.6f},"
                    f"{r.get('final_lambda', float('nan')):.6f},"
                    f"{r.get('train_seconds', 0)},{r.get('git_commit', '')}\n")

    print(f"\nWrote {RESULTS_DIR}/ablation_table.tex and ablation_raw.csv")


def phase_grid(phase):
    if phase == "A":
        # xai_alignment must complete before cosine (calibration dependency)
        order = ["fixed", "residual", "xai_alignment", "cosine", "xai_shuffled"]
        return [(PRIMARY_DATASET, a, s) for a in order for s in SEEDS]
    return [(d, a, s) for d in DATASETS for a in MAIN_ARMS for s in SEEDS
            if d != PRIMARY_DATASET]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phase", choices=["A", "B"])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run Phase B even if the gate fails (not advised)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.aggregate_only:
        aggregate(load_all()); return
    if args.gate:
        gate(load_all()); return
    if args.dry_run:
        print("DRY RUN — one cell only.\n")
        ok = execute_cell(PRIMARY_DATASET, "fixed", 0) == "ok"
        print("\nWiring OK. Next: python run_ablation.py --phase A" if ok
              else "\nFix the adapter in run_single() first.")
        return
    if not args.phase:
        ap.error("specify --phase A or --phase B (or --gate / --aggregate-only)")

    if args.phase == "B" and not args.force:
        if not gate(load_all()):
            print("\nPhase B not started. Use --force to override.")
            return

    grid = phase_grid(args.phase)
    print(f"\nPhase {args.phase}: {len(grid)} cells, "
          f"{ITERATIONS * EPOCHS_PER_ITER} epochs each (fixed a priori)\n")

    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for i, (ds, st, sd) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}]")
        counts[execute_cell(ds, st, sd)] += 1

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — {counts['ok']} run, "
          f"{counts['skipped']} skipped, {counts['failed']} failed")

    if args.phase == "A":
        gate(load_all())
    aggregate(load_all())


if __name__ == "__main__":
    sys.exit(main())
