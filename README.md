# Fast Learning — Analysis Code for Renard, Foustoukos et al. (2026)

Analysis pipeline for a two-photon calcium imaging study of rapid cortical plasticity during sensory learning in mice.  
Preprint: [bioRxiv 2026.05.11.724293](https://doi.org/10.64898/2026.05.11.724293) — *eLife*, 2026.

---

## Scientific Background

This repository contains the full analysis code for a study investigating how individual neurons in layer 2/3 of the primary somatosensory barrel cortex (wS1) reorganize their activity during fast, within-session learning of a whisker detection task.

Head-fixed, water-restricted mice were trained to lick in response to a brief deflection of the C2 whisker. Following 5–7 days of auditory pre-training (Days −2, −1), the novel whisker stimulus was introduced on Day 0 and continued for two additional days (Days +1, +2). Mice were randomly assigned to a rewarded (R+, n = 19) or unrewarded (R−, n = 16) group. Rewarded mice learned to detect the whisker stimulus within a single session, with behavioral divergence between groups emerging after just 22 whisker trials (~14 minutes). Pharmacological (muscimol) and optogenetic (VGAT-ChR2) inactivation of barrel cortex, but not forepaw somatosensory cortex, prevented this rapid learning, establishing a causal role for wS1.

The same layer 2/3 neurons were longitudinally imaged across all five days (3,210 neurons in 19 R+ mice; 2,846 neurons in 16 R− mice) using GCaMP6f calcium imaging to track how neural representations of the whisker stimulus evolved in parallel with behavior.

---

## Key Findings

The analysis establishes four main results:

1. **Rapid population reorganization.** Whisker-evoked responses progressively increased in R+ mice and decreased in R− mice across training days. A per-cell Learning Modulation Index (LMI) computed by ROC analysis shows the entire LMI distribution shifted positive in R+ mice (Kolmogorov–Smirnov test, p = 4×10⁻⁵⁴), with a significantly larger fraction of neurons acquiring positive LMI (Mann–Whitney U, p = 8×10⁻⁶). Pathway specificity: wS2-projecting neurons showed large learning-related changes while wM1-projecting neurons did not.

2. **Trial-by-trial decoding tracks learning.** A logistic regression classifier trained to discriminate pre-learning (Days −2, −1) from post-learning (Days +1, +2) passive whisker trial population vectors achieved significantly higher accuracy in R+ than R− mice (p = 1.6×10⁻³). Projecting Day 0 active trial activity onto the pre/post learning axis revealed a progressive, continuous shift in R+ mice whose slope was significantly positive (Wilcoxon, p = 0.03) and correlated with behavioral learning curves (p = 0.02).

3. **Representational geometry.** Trial-by-trial cosine similarity matrices (200 × 200, across 5 days × 40 passive trials) showed increasing within-day similarity in R+ mice and a significantly larger network reorganization index in R+ vs R− (p = 1.2×10⁻³).

4. **Online reactivations linked to plasticity.** Spontaneous reactivations of the whisker-evoked ensemble template detected during catch trials were more frequent in R+ mice (p = 0.014 on Day 0). Neurons that gained stimulus responsiveness through learning participated preferentially in reactivations (R+: r = 0.239, p = 1×10⁻⁴¹; R−: r = −0.067), suggesting a reward-gated, online selection mechanism for plasticity on the timescale of minutes.

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
│   ├── behavior/               # Behavioral performance, Bayesian learning curves
│   ├── day0/                   # Within-session sigmoid plasticity, LMI
│   ├── across_days/            # Multi-day decoding, cosine similarity matrices
│   ├── reactivations/          # Reactivation detection, surrogate thresholds, LMI–participation
│   ├── projectors_contributions/   # wS2- and wM1-projecting neuron analyses
│   └── illustrations/          # Calcium trace panels, rasters, FOV overlays
│
├── manuscript/                 # One script per figure panel
│   ├── figure_1/ – figure_4/   # Main figures
│   └── supp_1/ – supp_4/       # Supplementary figures
│
├── utils/                      # Shared utility modules
│   ├── utils_imaging.py        # dF/F extraction, ROI handling
│   ├── utils_behavior.py       # Trial parsing, lick analysis, performance metrics
│   ├── utils_io.py             # NWB file I/O, path resolution, session database
│   └── utils_plot.py           # Shared plotting styles and helpers
│
└── projection_gui/             # GUI tool for cross-day FOV registration (SimpleElastix)
```

Data are stored in [NWB format](https://www.nwb.org/) and loaded via a companion [NWB_analysis](https://github.com/LSENS-BMI-EPFL/NWB_converter) library. The full dataset will be deposited on Zenodo.

---

## Methods and Technical Stack

| Category | Tools / Approaches |
|---|---|
| **Calcium imaging** | Suite2p (ROI detection, neuropil correction: F = F_soma − 0.7 × F_neuropil), custom maximin-filter dF/F |
| **Data formats** | NWB (Neurodata Without Borders), xarray trial-aligned tensors, Pandas DataFrames |
| **Single-cell plasticity** | LMI via ROC/AUC (permutation test, 1,000 shuffles); sigmoid vs. linear vs. flat model comparison (`scipy.optimize`) |
| **Dimensionality reduction** | PCA, LDA |
| **Population decoding** | Logistic regression (L2, z-scored features, 10-fold stratified CV); SVM, Ridge Classifier, Random Forest |
| **Representational geometry** | Cosine similarity matrices (200 × 200); network reorganization index |
| **Reactivation detection** | Template correlation (Pearson) on catch trials; per-mouse surrogate thresholds (1,000 circular shifts, 99th percentile); `scipy.signal.find_peaks` (150 ms min distance, prominence 0.15) |
| **Behavioral modeling** | Bayesian state-space model (PyMC, Gaussian random walk in logit space, MCMC) for trial-by-trial lick probability |
| **Statistics** | Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Kolmogorov–Smirnov, Pearson/Spearman correlation, Benjamini–Hochberg FDR, bootstrap CIs |
| **Parallel processing** | `joblib` (up to 35 cores for per-mouse analyses) |
| **Visualization** | Matplotlib, Seaborn; SVG/PDF output for publication panels |
| **Language / environment** | Python 3.9, conda |

---

## Reproducing the Figures

Each panel has a dedicated script under `src/manuscript/`. Scripts load preprocessed data, run the analysis, and save output to a configured directory.

```bash
# Example: reproduce Figure 1b (behavioral performance across learning days)
python src/manuscript/figure_1/figure_1b.py
```

Computationally heavy analyses (decoding, reactivation surrogates) use a `mode` flag:

```python
mode = 'compute'   # run analysis and cache results
mode = 'analyze'   # load cached results and generate plots
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fast-learning
pip install -e .
```


---

## Citation

Renard A.\*, Foustoukos G.\*, Iuga M., Bech P., Bisi A., Dard R.F., Crochet S.\*, Petersen C.C.H.\*  
**Rapid cortical reorganization tracks goal-directed sensorimotor learning in real time.**  
*eLife*, 2026. DOI: [10.64898/2026.05.11.724293](https://doi.org/10.64898/2026.05.11.724293)

\* Equal contribution / co-corresponding authors.
