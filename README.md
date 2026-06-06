# Explainable AI-Based Optimization Framework for Unsupervised Hyperspectral Anomaly Detection

**Major Project — Phase 1 | Academic Year 2025–26**
**Visvesvaraya Technological University (VTU), Belagavi**
**Nitte Meenakshi Institute of Technology — Dept. of Computer Science and Business Systems**

---

## Team

| Name | USN |
|---|---|
| Chinmay Deepak Chandavar | 1NT23CB013 |
| Rithish P K | 1NT23CB045 |
| Sanjay Antony | 1NT23CB047 |

**Guide:** Dr. Sudha Y, Assistant Professor, Dept. of CSBS

---

## Project Overview

Hyperspectral Anomaly Detection (HAD) aims to identify pixels or regions in a hyperspectral image that differ significantly from the surrounding background — without any labeled training data. This is critical for applications in defense surveillance, environmental monitoring, mineral exploration, and disaster assessment.

Existing deep learning-based HAD methods suffer from three core problems:
- **Anomaly leakage** — anomaly pixels are incorrectly reconstructed as background during training
- **Limited interpretability** — models offer no explanation for *why* a pixel is flagged as anomalous
- **High computational cost** — transformer-based models are impractical for real-time deployment

This project addresses all three by building an **Explainable AI-enhanced MSNet framework** for unsupervised hyperspectral anomaly detection.

---

## Phase 1 — What's Implemented

Phase 1 delivers a fully validated prototype with 4 functional modules:

| FR ID | Requirement | Status |
|---|---|---|
| FR1 | Hyperspectral data loading and preprocessing (OPBS band selection, normalization) | ✅ PASS |
| FR2 | Anomaly detection using MSNet + SeT+ separation training | ✅ PASS |
| FR3 | Visualization — ROC curves, anomaly heatmaps, ground truth comparison | ✅ PASS |
| FR4 | Explainable AI — spatial saliency maps + spectral band importance rankings | ✅ PASS |

---

## Architecture

The system is a 7-module pipeline:

```
DataLoader (.mat file)
       ↓
Preprocessor (OPBS patch extraction)
       ↓
MSNet Encoder (multi-scale, 3 CED layers)
       ↓
MSNet Decoder (multi-scale background reconstruction)
       ↓
SeT+ Loss Module (L_BR + λ·L_AS, iterative soft mask)
       ↓
Anomaly Score Map (pixel-wise L2 reconstruction error)
       ↓
XAI Module (gradient-based saliency + spectral band importance)
       ↓
Visualization Output (PNG: heatmap, RGB composite, detection map)
```

**Key design choices:**
- **MSNet** uses summation-based multiscale feature fusion (not concatenation like U-Net) — avoids the Identical Mapping Problem (IMP)
- **SeT+ loss** = Background Reconstruction Loss (L_BR) + Anomaly Suppression Loss (L_AS), with iterative soft-mask updating to prevent anomaly leakage
- **XAI** uses gradient-based saliency over encoder feature maps to produce spatial anomaly saliency maps and per-band importance rankings

---

## Dataset

- **Dataset Used:** Coast Hyperspectral Dataset
- **Format:** `.mat` file
- **Shape:** 100 × 100 × 204 (H × W × Bands)
- **Bands Selected (OPBS):** 64
- **Anomaly clusters in ground truth:** 5

The dataset is included in `MSNet-main.zip` along with the original MSNet codebase.

---

## Results — Phase 1

### Anomaly Detection Performance (Coast Dataset)

| Method | AUC Score |
|---|---|
| RX Detector (baseline) | 0.9907 |
| **MSNet + SeT+ (ours)** | **0.9975** |
| Improvement | **+0.0069** |

### Training Configuration

| Parameter | Value |
|---|---|
| Device | CPU |
| Encoder–Decoder Layers | 3 |
| Total Parameters | 92,976 |
| Training Iterations | 10 |
| Epochs per Iteration | 150 |
| Total Epochs | 1500 |

### AUC Convergence

AUC rises from ~0.9675 at epoch 150 to above 0.9970 by epoch 750, then stabilizes — demonstrating the effectiveness of the iterative SeT+ mask updating.

### XAI Output

- **Top spectral band:** Band 3 (normalized gradient importance = 1.000)
- **Spatial saliency map:** Generated for all detected anomaly regions, correlates strongly with ground truth clusters
- Full XAI dashboard: spatial saliency + per-band importance bar chart across all 64 OPBS-selected bands

---

## Repository Structure

```
├── MSNet_Phase1_Complete (1).ipynb   # Main notebook: preprocessing, detection, XAI, visualization
├── MSNet-main.zip                    # Dataset + original MSNet codebase
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install torch numpy scipy matplotlib scikit-learn
```

### Steps

1. Unzip `MSNet-main.zip` — contains the Coast `.mat` dataset and base MSNet code
2. Open `MSNet_Phase1_Complete (1).ipynb` in Jupyter Notebook or Google Colab
3. Run all cells sequentially — the notebook is self-contained and handles:
   - Data loading and OPBS preprocessing
   - MSNet training with SeT+ loss
   - Anomaly score computation and ROC curve generation
   - XAI saliency map and spectral band importance output

**Recommended:** Google Colab with T4 GPU (CPU also works; ~500MB storage required)

---

## Phase 2 — Planned Work

- Evaluation on SanDiego, Abu Airport, and HYDICE Urban datasets
- Benchmarking against SSHAD, CWIMamba, and FS²CCTrans
- SHAP and Integrated Gradients for more robust XAI
- Precision, Recall, F1, Detection Rate (DR), and False Alarm Rate (FAR) reporting
- Streamlit web interface for interactive detection and XAI visualization
- Generic `.mat` dataset loader for any hyperspectral file

---

## References

1. H. Liu et al., "MSNet: Self-Supervised Multiscale Network With Enhanced Separation Training for Hyperspectral Anomaly Detection," IEEE TGRS, vol. 62, 2024.
2. S. Liu et al., "Adaptive Dual-Domain Learning for Hyperspectral Anomaly Detection With State-Space Models," IEEE TGRS, vol. 63, 2025.
3. X. He et al., "CWIMamba: Cross-Scale Windowed Integration State Space Model for Hyperspectral Anomaly Detection," IEEE TGRS, vol. 63, 2025.
4. G. Zhang et al., "Frequency–Spatial–Spectral Joint Analysis with Criss-Cross Transformer for Hyperspectral Anomaly Detection," IEEE TGRS, 2023.
5. S. F. Eshpulatovich et al., "Hyperspectral Anomaly Detection With Enhanced Spectral Graph Transformer Network," IEEE Access, 2023.
