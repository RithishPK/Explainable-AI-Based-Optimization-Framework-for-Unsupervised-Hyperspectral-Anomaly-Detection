# Explainable AI-Based Optimization Framework for Unsupervised Hyperspectral Anomaly Detection

**Major Project — Visvesvaraya Technological University (VTU), Belagavi**
Nitte Meenakshi Institute of Technology (NMIT), Bengaluru — Dept. of Computer Science and Business Systems

A self-supervised hyperspectral anomaly detector built on **MSNet + Enhanced Separation Training (SeT+)**, extended with an **Integrated Gradients (IG) explainability layer** and two proposed optimization mechanisms — an XAI-guided adaptive loss weight (λ) and multiscale gating — together with a rigorous analysis of whether those mechanisms actually work.

> **Honest-result notice.** This project reports both design *and* evidence. The reproduced MSNet+SeT+ backbone and the explainability layer are validated. The two proposed optimization mechanisms (XAI-mask adaptive-λ and multiscale gating) are shown — analytically and by measurement — to be **circular / self-referential**, and are therefore documented as negative/future-work results rather than claimed improvements. See [Scope, Status & Limitations](#scope-status--limitations).

---

## Overview

Hyperspectral anomaly detection (HAD) flags pixels whose spectra depart from the surrounding background, without labelled training data. Reconstruction-based self-supervised detectors train a small network to reconstruct a single scene's background and score each pixel by its reconstruction error.

This framework:

1. **Reproduces** the MSNet multiscale convolutional encoder–decoder with SeT+ separation training, under a reporting rule fixed in advance.
2. **Adds** a gradient-based explainability layer (Integrated Gradients + Grad-CAM) producing spatial saliency maps and per-band importance rankings.
3. **Investigates** using the IG attribution signal to steer the SeT+ suppression weight λ during training (adaptive-λ) and to gate multiscale feature fusion — and establishes, with proof and measurement, why that signal cannot act as a genuine controller.
4. **Delivers** an interactive Streamlit application for the full workflow: dataset upload → band selection → training → detection → explainability → export.

---

## What's validated vs. proposed

| # | Item | Status |
|---|------|--------|
| C1 | XAI-mask alignment adaptive-λ signal is self-referential (mask and IG target are the same reconstruction-error field; corr ≈ 0.98 on Coast, alignment saturates > 0.97 from iteration 2, λ pins at its ceiling) | **Established (negative result)** |
| C2 | Residual-ratio λ signal is monotonic by construction and cannot decrease — not a controller | **Established (analytical)** |
| C3 | Faithful MSNet+SeT+ reproduction on difficulty-stratified scenes; correction of a previously quoted 0.9975 Coast figure (a ground-truth-selected peak) to 0.9972 at a fixed iteration | **Established** |
| C4 | IG implementation verified to accumulate *signed* gradients (completeness axiom holds); reflection padding added to bottleneck convolutions to reduce border artefacts | **Established (correctness)** |
| C5 | Background-targeted IG variant lowers mask correlation from ≈ 0.98 to 0.77 — a real change, but saturation remains | **Partial** |
| C6 | Multiscale gating (reuses the same IG saliency map to gate decoder-side fusion) — inherits the circularity of C1 | **Proposed / untested (future work)** |

---

## Key results

Single-run AUC(Pd, Pf), seed 0. Coast is evaluated at a fixed outer iteration (5); Aviris and ABU-Airport use the residual-λ schedule and therefore report **MSNet+SeT backbone** behaviour, not the XAI controller.

| Scene | MSNet + SeT | RX Detector (baseline) | Margin over RX |
|-------|:-----------:|:----------------------:|:--------------:|
| Coast (easy) | **0.9972** | 0.9907 | +0.007 |
| Aviris (harder) | **0.9837** | 0.8866 | +0.097 |
| ABU-Airport (hardest) | **0.9524** | 0.8221 | +0.130 |

The reproduction lands within 0.0008 of the original MSNet publication on Coast, and the detector's margin over RX **widens as scenes get harder** — i.e. the backbone's advantage is most visible on difficult scenes, where RX headroom is largest.

> All reported scores are single-seed on single scenes and are flagged as such. Multi-seed mean ± std evaluation is listed as required future work.

---

## How it works

```
Hyperspectral scene (.mat)
        │
        ▼
 OPBS band selection  ──►  reduce to N_bs = 64 informative bands  ──►  Z-score normalisation
        │
        ▼
 MSNet encoder–decoder (3 CED layers, multiscale summation fusion)
        │        trained under SeT+ loss:  L_SeT+ = L_BR + λ · L_AS
        │        (background reconstruction + mask-guided anomaly suppression)
        ▼
 Soft mask update per outer iteration  ◄── [Phase-2 controller, optional/opt-in]
        │                                    IG attribution → XAI-mask alignment → λ recalibration
        ▼
 Detection map = pixel-wise reconstruction error
        │
        ▼
 Explainability: Integrated Gradients saliency + Grad-CAM + per-band importance
        │
        ▼
 Visualisation & export (ROC, λ/gate trajectories, CSV, model checkpoint)
```

The core MSNet backbone is kept **unmodified** except for one justified change (reflection padding in the bottleneck convolutions). Every Phase-2 experimental feature is **opt-in and defaults to off**, so disabling it reduces the system exactly to the validated baseline.

---

## Repository structure

```
.
├── main.py                     # Demo entry point (runs a single-scene experiment)
├── model/MSNet/                # MSNet encoder–decoder architecture
├── SeT/                        # Enhanced Separation Training (train, loss, mask, detect)
├── adaptive_lambda.py          # Phase-2 adaptive-λ / multiscale-gating controller (experimental)
├── xai_core.py                 # Integrated Gradients + Grad-CAM explainability
├── select_bands.py             # OPBS band selection
├── unsupervised_selection.py   # Unsupervised band/feature selection utilities
├── metric.py, metrics_extended.py   # AUC / ROC evaluation metrics
├── run_multi_dataset.py        # Multi-scene evaluation runner
├── probe_v14.py, verify_v14.py # Diagnostic / verification scripts
├── streamlit_app/              # Interactive Streamlit application (app, utils, xai_utils)
├── toolkit/                    # Ablations, leakage audit, IG completeness verification
├── dataset/                    # Hyperspectral scenes (.mat) + loaders
│   ├── coast/  aviris/  abu_airport/  sandiego/  indian_pines/
├── requirements.txt
├── CHANGES.md, CHANGES_v14.md  # Version history
└── MSNet.png                   # Architecture diagram
```

---

## Getting started

### 1. Environment

Python 3.10–3.12. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

GPU execution (NVIDIA, CUDA) is supported and recommended for larger scenes; a CPU-only fallback works for smaller scenes such as Coast.

### 2. Run a quick experiment

```bash
python main.py
```

This runs a single-scene experiment (Coast by default). Add other scenes under `dataset/` following the existing loader format.

### 3. Launch the interactive app

```bash
streamlit run streamlit_app/app.py
```

The Streamlit interface provides dataset upload (up to 1 GB, with legacy MATLAB v7.3/HDF5 fallback), band-count and training controls, a GPU/device diagnostic, detection and RX baseline maps, ROC curves, λ/gate trajectories, saliency maps, and CSV/checkpoint export.

---

## Datasets

The repository ships several benchmark hyperspectral scenes as MATLAB `.mat` files, auto-detected by array shape rather than variable name:

| Scene | Ground truth | Notes |
|-------|:------------:|-------|
| Coast | ✓ | Primary scene (AVIRIS, 100×100×204) |
| Aviris | ✓ | Harder scene, evaluated |
| ABU-Airport | ✓ | Hardest scene, evaluated |
| Indian Pines | ✓ | Multi-class ground truth |
| San Diego | — | No ground truth in supplied file (no AUC) |

Preprocessing applies OPBS band selection (default `N_bs = 64`) followed by Z-score normalisation.

---

## Scope, Status & Limitations

This project deliberately separates **what the evidence supports** from **what is proposed but unproven**:

- **The adaptive-λ controller is circular.** The soft mask and the IG attribution target are both derived from the same reconstruction-error field, so the alignment signal is positive by construction, saturates above 0.97 from the second iteration, and never turns negative — reducing the method to fixed-weight training. This is proven analytically and confirmed numerically (contribution C1).
- **The residual-ratio λ signal cannot control λ** either: separation training's own objective makes the ratio increase monotonically, so it can only ramp toward a ceiling (contribution C2).
- **Multiscale gating is untested** and inherits the same circularity; it is not enabled for any reported result (C6).
- **All reported scores are single-seed on single scenes.** No mean ± standard deviation is available yet.
- The cross-scene results (Aviris, ABU-Airport) use the residual-λ schedule and therefore characterise the **backbone**, not the XAI controller.

**Future work** (each with multiple seeds before any wider sweep): a range-matched cosine λ schedule as a scheduling-only control; plain gradient magnitude in place of IG; a spatially shuffled saliency control; multi-seed three-scene evaluation with the XAI controller enabled; and — the central open problem — designing a **non-circular** attribution target genuinely distinct from the detection statistic.

---

## Team

**Dept. of Computer Science and Business Systems, NMIT Bengaluru**

| Name | USN |
|------|-----|
| Chinmay Deepak Chandavar | 1NT23CB013 |
| Rithish P K | 1NT23CB045 |
| Sanjay Antony | 1NT23CB047 |

**Project Guide:** Dr. Sudha Y, Assistant Professor

---

## Acknowledgement & Citation

This work builds directly on **MSNet** and uses the OPBS band selection algorithm. If you use this code, please cite the original MSNet paper:

```bibtex
@article{liu2024msnet,
  title   = {MSNet: Self-Supervised Multi-Scale Network with Enhanced Separation Training for Hyperspectral Anomaly Detection},
  author  = {Liu, Haijun and Su, Xi and Shen, Xiangfei and Zhou, Xichuan},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {62},
  year    = {2024},
  doi     = {10.1109/TGRS.2024.3411009}
}
```

The authors thank Dr. Sudha Y and the Department of Computer Science and Business Systems, NMIT Bengaluru.

---

## License

See [`LICENSE`](LICENSE). The MSNet backbone retains the terms of its original authors; please respect the upstream license when reusing that component.
