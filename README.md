# Fast Learning — Analysis Code for Renard, Foustoukos et al. (2026)

Analysis pipeline for a two-photon calcium imaging study of rapid cortical plasticity during sensory learning in mice.

---

## Scientific Background

This repository contains the full analysis code for a study investigating how individual neurons in the somatosensory cortex reorganize their activity during fast, within-session learning of a whisker detection task.

Mice were imaged across five sessions relative to the key learning day (days −2, −1, 0, +1, +2). A central question is whether learning-related plasticity is gradual or abrupt — and whether spontaneous neural reactivations during inter-trial intervals predict or drive behavioral improvement.

Key constructs analyzed:

- **Learning Modulation Index (LMI)**: a per-cell metric quantifying how much a neuron's sensory response changes across the learning session
- **Single-cell sigmoid plasticity**: fitting sigmoid vs. linear vs. flat models to trial-by-trial responses to identify cells with abrupt, step-like changes in activity on the learning day
- **Neural reactivations**: spontaneous events during no-stimulus trials whose correlation to the learned stimulus template exceeds a surrogate-based threshold; examined for their relationship to behavioral performance
- **Population-level decoding**: SVM and logistic regression classifiers trained on single-trial population vectors to probe when and how quickly neural representations of the stimulus emerge
- **Cosine similarity matrices**: trial-by-trial representational geometry computed across days to track the evolution of the population code

---

## Repository Structure

```
src/
├── preprocessing/              # Data ingestion and signal extraction
│   ├── processing_tiff_files/  # TIFF deinterleaving and rewriting
│   ├── processing_calcium_imaging/  # Suite2p pipeline, dF/F computation
│   ├── processing_session_files/    # Session stitching and config validation
│   └── processing_tensor_data/     # Generation of trial-aligned xarray tensors
│
├── core_analysis/              # Modular analysis components
│   ├── behavior/               # Behavioral performance, learning curve fitting
│   ├── day0/                   # Within-session plasticity, sigmoid fitting, LMI
│   ├── across_days/            # Multi-day decoding, correlation matrices, LMI tracking
│   ├── reactivations/          # Spontaneous reactivation detection and statistics
│   ├── projectors_contributions/   # Projection-neuron-specific analyses
│   └── illustrations/          # Calcium trace panels, rasters, FOV overlays
│
├── manuscript/                 # Reproduction scripts for each figure panel
│   ├── figure_1/ – figure_4/   # Main figures (panels a–j)
│   └── supp_1/ – supp_4/       # Supplementary figures
│
├── utils/                      # Shared utility modules
│   ├── utils_imaging.py        # dF/F extraction, ROI handling
│   ├── utils_behavior.py       # Trial parsing, lick analysis, performance metrics
│   ├── utils_io.py             # NWB file I/O, path resolution, session database
│   └── utils_plot.py           # Shared plotting styles and helpers
│
└── projection_gui/             # GUI tool for FOV registration across days
```

Data are stored in [NWB format](https://www.nwb.org/) and loaded via a companion `NWB_analysis` library.

---

## Methods and Technical Stack

| Category | Tools / Approaches |
|---|---|
| **Calcium imaging** | Suite2p (ROI detection, neuropil correction), custom dF/F pipeline |
| **Data formats** | NWB (Neurodata Without Borders), xarray tensors, Pandas DataFrames |
| **Dimensionality reduction** | PCA, LDA |
| **Decoding** | SVM (`SVC`), Logistic Regression, Ridge Classifier, Random Forest — cross-validated with `StratifiedKFold` |
| **Representational geometry** | Cosine similarity matrices, Spearman/Pearson correlation |
| **Single-cell modeling** | Sigmoid curve fitting (`scipy.optimize`), model comparison (sigmoid vs. linear vs. flat) |
| **Reactivation detection** | Template correlation + surrogate-threshold approach (per-mouse 99th-percentile baseline) |
| **Statistics** | Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman, bootstrap CIs, FDR correction |
| **Bayesian modeling** | PyMC for behavioral learning curve fitting |
| **Parallel processing** | `joblib` (up to 35 cores) |
| **Visualization** | Matplotlib, Seaborn; publication-quality SVG/PDF output |
| **Language / environment** | Python 3.9, conda |

---

## Reproducing the Figures

Each figure panel has a dedicated script under `src/manuscript/`. Scripts are self-contained: they load preprocessed data, run the relevant analysis, and save output to a configured directory.

```bash
# Example: reproduce Figure 1b (behavioral performance across learning days)
python src/manuscript/figure_1/figure_1b.py
```

For analyses that are computationally expensive (decoding, reactivation surrogates), scripts accept a `mode` flag — `'compute'` to run and cache results, `'analyze'` to load cached results and generate plots.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fast-learning
pip install -e .
```


---

## Citation

> Renard A.\*, Foustoukos G.\*, et al. eLife, 2026. Rapid cortical reorganization tracks goal-directed sensorimotor learning in real time.
