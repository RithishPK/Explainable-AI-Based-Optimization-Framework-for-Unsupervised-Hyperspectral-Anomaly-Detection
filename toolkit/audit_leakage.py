"""
audit_leakage.py — find ground-truth-based model selection in your repo.

I cannot see your code, so this does the looking for you. It statically scans
.py and .ipynb files for the patterns that produce model-selection leakage,
and reports file:line so you can go fix them.

Run from the repo root:

    python audit_leakage.py
    python audit_leakage.py --path C:\\Users\\rithi\\OneDrive\\Desktop\\HAD PROJECT

Then paste the output back and I can tell you which hits are real and which
are false positives.

Exit code 1 if any HIGH-severity hit is found, so you can wire it into CI.
"""

import argparse
import ast
import io
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Patterns. HIGH = almost certainly leakage. MEDIUM = inspect manually.
# ---------------------------------------------------------------------------

PATTERNS = [
    # --- selecting a checkpoint / epoch by AUC ------------------------------
    ("HIGH", r"\bbest[_\s]*auc\b",
     "A 'best AUC' variable implies scanning checkpoints against ground truth."),
    ("HIGH", r"\bbest[_\s]*(epoch|iter|ckpt|checkpoint|model|state)\b",
     "Selecting a best epoch/checkpoint — check what criterion picks it."),
    ("HIGH", r"(max|np\.max|np\.nanmax)\s*\(\s*[^)]*auc",
     "max() over AUC values is peak-selection, not a fixed budget."),
    ("HIGH", r"(argmax|np\.argmax)\s*\(\s*[^)]*auc",
     "argmax over AUC selects an epoch using labels."),
    ("HIGH", r"auc[_a-z0-9]*\s*>\s*best",
     "Classic 'if auc > best' checkpoint-tracking loop."),
    ("HIGH", r"\.max\s*\(\s*\)\s*",
     "Bare .max() — check whether the receiver is an AUC list or trace."),

    # --- ground truth reaching the training loop ---------------------------
    ("MEDIUM", r"\b(gt|ground_truth|label|labels|y_true|mask_gt)\b",
     "Ground truth referenced — confirm it is used for evaluation only."),
    ("MEDIUM", r"select_best_by_auc\s*=\s*True",
     "Explicit best-by-AUC selection is enabled."),

    # --- non-determinism that makes 'seeds' decorative ---------------------
    ("MEDIUM", r"cudnn\.benchmark\s*=\s*True",
     "cudnn.benchmark=True breaks determinism; seeds will not reproduce."),
    ("MEDIUM", r"shuffle\s*=\s*True",
     "Shuffling without a seeded generator makes runs non-reproducible."),

    # --- things that quietly change the reported number --------------------
    ("MEDIUM", r"\bearly[_\s]*stop",
     "Early stopping on a label-derived metric is also leakage."),
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints",
             "venv", ".venv", "env", "ablation_results"}


def iter_source_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".py", ".ipynb")):
                yield os.path.join(dirpath, fn)


def extract_lines(path):
    """Return [(lineno, text)] for .py, or [(cell:line, text)] for .ipynb."""
    try:
        if path.endswith(".ipynb"):
            with io.open(path, encoding="utf-8") as f:
                nb = json.load(f)
            out = []
            for ci, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                for li, line in enumerate(cell.get("source", []), start=1):
                    out.append((f"cell{ci}:{li}", line.rstrip("\n")))
            return out
        with io.open(path, encoding="utf-8", errors="replace") as f:
            return list(enumerate(f.read().splitlines(), start=1))
    except Exception as e:
        print(f"  ! could not read {path}: {e}", file=sys.stderr)
        return []


def scan(root):
    hits = {"HIGH": [], "MEDIUM": []}
    n_files = 0

    for path in iter_source_files(root):
        n_files += 1
        rel = os.path.relpath(path, root)
        for lineno, text in extract_lines(path):
            stripped = text.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for severity, pat, note in PATTERNS:
                if re.search(pat, text, flags=re.IGNORECASE):
                    hits[severity].append((rel, lineno, stripped[:110], note))
    return hits, n_files


def check_final_vs_peak(root):
    """
    Look for the specific safe pattern: reporting trace[-1] rather than
    max(trace). Presence is reassuring; absence is not proof of a bug.
    """
    good, bad = [], []
    for path in iter_source_files(root):
        rel = os.path.relpath(path, root)
        for lineno, text in extract_lines(path):
            if re.search(r"auc[_a-z0-9]*\s*\[\s*-\s*1\s*\]", text, re.I):
                good.append((rel, lineno, text.strip()[:90]))
            if re.search(r"(max|argmax)\s*\([^)]*auc[^)]*\)", text, re.I):
                bad.append((rel, lineno, text.strip()[:90]))
    return good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        return 2

    print("=" * 74)
    print(f"LEAKAGE AUDIT — {root}")
    print("=" * 74)

    hits, n_files = scan(root)
    print(f"\nScanned {n_files} source files.\n")

    for sev in ("HIGH", "MEDIUM"):
        rows = hits[sev]
        print("-" * 74)
        print(f"{sev}  ({len(rows)} hits)")
        print("-" * 74)
        if not rows:
            print("  none\n")
            continue
        seen_notes = set()
        for rel, lineno, text, note in rows:
            print(f"  {rel}:{lineno}")
            print(f"      {text}")
            if note not in seen_notes:
                print(f"      -> {note}")
                seen_notes.add(note)
        print()

    good, bad = check_final_vs_peak(root)
    print("-" * 74)
    print("FINAL-vs-PEAK reporting")
    print("-" * 74)
    print(f"  trace[-1] style (safe)   : {len(good)}")
    print(f"  max/argmax style (unsafe): {len(bad)}")
    for rel, lineno, text in bad:
        print(f"      {rel}:{lineno}  {text}")

    print("\n" + "=" * 74)
    if hits["HIGH"]:
        print("VERDICT: HIGH-severity hits present. Reported AUC is suspect")
        print("until each one is inspected. Paste this output back for triage.")
        print("=" * 74)
        return 1
    print("VERDICT: no HIGH-severity patterns found. Still confirm manually")
    print("that the reported AUC is the final-iteration value.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
