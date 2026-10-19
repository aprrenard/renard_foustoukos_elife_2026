# fast-learning

[![CI](https://github.com/LSENS-BMI-EPFL/renard_foustoukos/actions/workflows/ci.yml/badge.svg)](https://github.com/LSENS-BMI-EPFL/renard_foustoukos/actions/workflows/ci.yml)
[![eLife](https://img.shields.io/badge/eLife-10.7554%2FeLife.111818.1-087acc)](https://doi.org/10.7554/eLife.111818.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Analysis code for

> Renard A.\*, Foustoukos G.\*, Iuga M., Bech P., Bisi A., Dard R.F., Crochet S.\*, Petersen C.C.H.\*
> **Rapid cortical reorganization tracks goal-directed sensorimotor learning in real time.**
> *eLife* (2026), reviewed preprint. [doi:10.7554/eLife.111818.1](https://doi.org/10.7554/eLife.111818.1)
>
> \* equal contribution

It goes from the NWB files of the published dataset to every figure panel,
with the source data and statistics of each panel.

## Installation

Python ≥ 3.11. With conda:

```bash
git clone https://github.com/LSENS-BMI-EPFL/renard_foustoukos.git fast-learning
cd fast-learning
conda env create -f environment.yml      # creates the 'fast-learning' env and installs the package
conda activate fast-learning
```

or in any virtual environment: `pip install -e .` The exact package versions
used for the paper are in [`requirements-lock.txt`](requirements-lock.txt).
Optional extras: `.[preprocessing]` (suite2p, for the upstream steps) and
`.[gui]` (projection-neuron labelling GUI).

## Data and configuration

The dataset (NWB files, session metadata, processed tensors) will be
deposited on Zenodo. Copy [`config.example.yaml`](config.example.yaml) to
`config.yaml` and set `data_root` (the downloaded dataset) and `output_root`
(where processed data and figures are written). Input folders are only read.

## Reproducing the figures

```bash
python run_all.py --list                  # the steps, in order
python run_all.py --stage figures         # all figures from the processed data (~15 min)
python run_all.py --only figure_6 supp_4  # some figures
python run_all.py --stage all             # everything from the NWB files (several hours)
python figures/figure_5/figure_5b_d.py    # or any script on its own
```

Each figure script writes, in `<output_root>/figures/<figure>/` (e.g.
`Figure1/`, `Figure1_SupFig1/`), the panel (PDF by default; set
`figure_formats: [pdf, svg]` in `config.yaml` for SVG too) with its plotted
data (`*_data.csv`) and statistics (`*_stats.csv`).
Each step's log goes to `<output_root>/logs/`. Computations are seeded, so
reruns give identical results.

## How the code is organised

```
src/fast_learning/   the analysis library: paths and metadata, tensors, LMI,
                     decoding, reactivation detection, participation,
                     similarity, statistics, plotting
pipeline/            NWB files -> processed data, one numbered step per product
  upstream/          raw imaging -> NWB inputs (suite2p, dF/F), for reference
figures/             one script per figure panel (or group of panels)
run_all.py           runs the pipeline and the figures in order
tests/               unit tests of the core computations (pytest)
exploratory/         earlier and exploratory analyses, not needed for the paper
```

| Step | Produces |
|---|---|
| `01_session_flags` | trial ranges analysed per session (end of engagement, passive mapping block); shipped with the data, off by default |
| `02_tensors` | per-mouse dF/F tensors (cells × trials × time) with the trial table |
| `03_behavior_tables` | trial-level behaviour tables (imaging, particle test, muscimol, optogenetics) |
| `04_learning_curves` | Bayesian learning curves and learning trial per session |
| `05_lmi` | learning modulation index of each neuron, with shuffle significance |
| `06_decoder` | per-mouse pre/post-learning decoder (weights reused by Fig. 5f–i) |
| `07_reactivations` | reactivation events (surrogate thresholds, template correlation), mouse selection |
| `08_participation` | participation of each neuron in reactivations, measured in held-out cells (events detected with the other half of the cells); all-cell version for comparison |
| `09_pairwise_correlations` | pre-stimulus correlations between projection neurons |

### Figures

<!-- figure-table:start -->
| Panel(s) | Script | Content |
|---|---|---|
| **Figure 1** | | |
| 1b | [`figures/figure_1/figure_1b.py`](figures/figure_1/figure_1b.py) | Example behavioral performance across learning days |
| 1c | [`figures/figure_1/figure_1c.py`](figures/figure_1/figure_1c.py) | Behavioral performance of the imaging mice across days |
| 1d | [`figures/figure_1/figure_1d.py`](figures/figure_1/figure_1d.py) | Day 0 performance across whisker trials, R+ vs R- |
| **Figure 1 – figure supplement 1** | | |
| 1-S1a | [`figures/figure1_supfig1/figure1_supfig1_a.py`](figures/figure1_supfig1/figure1_supfig1_a.py) | First hit trial on Day 0 for R+ vs R- |
| 1-S1b | [`figures/figure1_supfig1/figure1_supfig1_b.py`](figures/figure1_supfig1/figure1_supfig1_b.py) | Whisker lick probability on Days 0, +1, +2 for R+ vs R- |
| 1-S1c | [`figures/figure1_supfig1/figure1_supfig1_c.py`](figures/figure1_supfig1/figure1_supfig1_c.py) | Particle test — whisker hit rate and false alarm rate across ON / OFF / ON periods (R+ mice only) |
| 1-S1d | [`figures/figure1_supfig1/figure1_supfig1_d.py`](figures/figure1_supfig1/figure1_supfig1_d.py) | Single-trial whisker hit rate across Day 0 trials, R+ vs R- (non-realigned) |
| 1-S1e | [`figures/figure1_supfig1/figure1_supfig1_e.py`](figures/figure1_supfig1/figure1_supfig1_e.py) | Fitted learning curve across Day 0 trials, R+ vs R-, realigned to each mouse's first hit trial |
| 1-S1f | [`figures/figure1_supfig1/figure1_supfig1_f.py`](figures/figure1_supfig1/figure1_supfig1_f.py) | Day 0 lick probability for whisker, auditory, and no-stim trial types on a common time axis (minutes from session start) |
| 1-S1g, h | [`figures/figure1_supfig1/figure1_supfig1_g_h.py`](figures/figure1_supfig1/figure1_supfig1_g_h.py) | Reaction times for auditory, whisker, and no-stim hit trials |
| **Figure 2** | | |
| 2b–d | [`figures/figure_2/figure_2b_d.py`](figures/figure_2/figure_2b_d.py) | Muscimol inactivation of wS1 or fpS1 during learning |
| 2e–g | [`figures/figure_2/figure_2e_g.py`](figures/figure_2/figure_2e_g.py) | Optogenetic inactivation of wS1 or fpS1 during learning |
| **Figure 2 – figure supplement 1** | | |
| 2-S1a–d | [`figures/figure2_supfig1/figure2_supfig1_a_d.py`](figures/figure2_supfig1/figure2_supfig1_a_d.py) | Muscimol inactivation of wS1 or fpS1 during task execution, in expert mice |
| **Figure 2 – figure supplement 2** | | |
| 2-S2a–l | [`figures/figure2_supfig2/figure2_supfig2_a_l.py`](figures/figure2_supfig2/figure2_supfig2_a_l.py) | Learning on the first day without inactivation, compared with naive mice learning on Day 0 |
| **Figure 3** | | |
| 3a | [`figures/figure_3/figure_3a.py`](figures/figure_3/figure_3a.py) | FOV with LMI overlay + calcium transient illustration for the same mouse |
| 3b | [`figures/figure_3/figure_3b.py`](figures/figure_3/figure_3b.py) | Raster plot of single-cell PSTH activity across learning days for GF314 |
| 3c | [`figures/figure_3/figure_3c.py`](figures/figure_3/figure_3c.py) | Grand-average mapping-trial PSTHs for all cells across learning days, R+ vs R- |
| 3d, e | [`figures/figure_3/figure_3d_e.py`](figures/figure_3/figure_3d_e.py) | Pre vs post learning response comparison (all cells) |
| 3f, g | [`figures/figure_3/figure_3f_g.py`](figures/figure_3/figure_3f_g.py) | Learning Modulation Index (LMI) for all cells |
| 3h–j | [`figures/figure_3/figure_3h_j.py`](figures/figure_3/figure_3h_j.py) | Trial-by-trial correlation matrices and network reorganization metrics |
| **Figure 5** | | |
| 5b–d | [`figures/figure_5/figure_5b_d.py`](figures/figure_5/figure_5b_d.py) | Decoding analyses (Figure 3l-n of the submitted version; 5a is a schematic) |
| 5f | [`figures/figure_5/figure_5f.py`](figures/figure_5/figure_5f.py) | Example mice — behaviour and decoder decision value during Day 0 |
| 5g | [`figures/figure_5/figure_5g.py`](figures/figure_5/figure_5g.py) | Progressive learning during Day 0 — population average behaviour and decoder value, R+ vs R- |
| 5h | [`figures/figure_5/figure_5h.py`](figures/figure_5/figure_5h.py) | Slope analysis of the progressive learning decoder — R+ vs R- |
| 5i | [`figures/figure_5/figure_5i.py`](figures/figure_5/figure_5i.py) | Correlation between decoder decision value and behavioural performance across Day-0 whisker trials — R+ vs R- |
| **Figure 6** | | |
| 6a | [`figures/figure_6/figure_6a.py`](figures/figure_6/figure_6a.py) | Reactivation heatmap illustration (mouse AR127, single day) |
| 6b | [`figures/figure_6/figure_6b.py`](figures/figure_6/figure_6b.py) | Example correlation traces across days (mouse AR127) |
| 6c | [`figures/figure_6/figure_6c.py`](figures/figure_6/figure_6c.py) | Reactivation rate across days, R+ vs R- |
| 6d, e | [`figures/figure_6/figure_6d_e.py`](figures/figure_6/figure_6d_e.py) | Reactivation participation rate vs LMI |
| **Figure 2 – supplement** | | |
| S2b | [`figures/supp_2/supp_2b.py`](figures/supp_2/supp_2b.py) | Example mapping-trial PSTHs for three illustrative cells across learning days |
| S2c | [`figures/supp_2/supp_2c.py`](figures/supp_2/supp_2c.py) | Relationship between classifier weights and the Learning Modulation Index (LMI) |
| S2d | [`figures/supp_2/supp_2d.py`](figures/supp_2/supp_2d.py) | Decoding accuracy vs percentage of most-modulated cells retained |
| **Figure 3 – supplement** | | |
| S3b | [`figures/supp_3/supp_3b.py`](figures/supp_3/supp_3b.py) | Grand-average mapping-trial PSTHs for projection neurons (wS2 and wM1) across learning days, R+ vs R- |
| S3c, d, g, h | [`figures/supp_3/supp_3c_d_g_h.py`](figures/supp_3/supp_3c_d_g_h.py) | Pre vs post learning responses for projection neurons |
| S3e, f, i, j | [`figures/supp_3/supp_3e_f_i_j.py`](figures/supp_3/supp_3e_f_i_j.py) | Proportions and distributions of LMI for projection neurons |
| S3k, l | [`figures/supp_3/supp_3k_l.py`](figures/supp_3/supp_3k_l.py) | CDF comparison of wS2 vs wM1 projector neurons |
| S3m | [`figures/supp_3/supp_3m.py`](figures/supp_3/supp_3m.py) | Pairwise correlations between projection neurons (wS2-wS2 and wM1-wM1 pairs) during a 2 s pre-stimulus quiet window, compared pre vs post learning. Mapping trials only |
| **Figure 4 – supplement** | | |
| S4a, b | [`figures/supp_4/supp_4a_b.py`](figures/supp_4/supp_4a_b.py) | Participation above chance, controlling for each cell's activity level |
| **Revision analyses (in progress)** | | |
| — | [`figures/revisions/behavior_day0_initial_licking.py`](figures/revisions/behavior_day0_initial_licking.py) | Licking to the whisker stimulus at the start of Day 0 (R2 public 1, R1 recommendation 1, R3 public 1) |
| — | [`figures/revisions/behavior_dprime.py`](figures/revisions/behavior_dprime.py) | Behavior quantified via d' (signal detection theory) instead of whisker hit rate |
| — | [`figures/revisions/behavior_state_summary.py`](figures/revisions/behavior_state_summary.py) | Session-level behavioral state summary (total water reward, session duration, total trial count) per mouse x session, across days and reward groups |
| — | [`figures/revisions/figure_3_passive_adaptation.py`](figures/revisions/figure_3_passive_adaptation.py) | Adaptation within the passive block, and the learning-related response changes in early vs late passive trials (R3 public 3) |
| — | [`figures/revisions/figure_3f_LMIshuffles.py`](figures/revisions/figure_3f_LMIshuffles.py) | LMI distribution vs. shuffled null |
| — | [`figures/revisions/figure_3g_i_signal_noise_drift.py`](figures/revisions/figure_3g_i_signal_noise_drift.py) | Response strength, trial-to-trial noise and day-to-day drift of whisker responses |
| — | [`figures/revisions/figure_6c_event_timing.py`](figures/revisions/figure_6c_event_timing.py) | Are reactivation events over-represented when mice can lick? |
| — | [`figures/revisions/figure_6c_previous_outcome.py`](figures/revisions/figure_6c_previous_outcome.py) | Does the reactivation rate depend on the outcome of the previous trial? |
| — | [`figures/revisions/figure_6c_shuffle_control.py`](figures/revisions/figure_6c_shuffle_control.py) | Shuffled-template reactivation detection |
| — | [`figures/revisions/movement_state_summary.py`](figures/revisions/movement_state_summary.py) | Whisker, tongue and jaw movement during the passive whisker trials, across days (R2 public 6) |
| — | [`figures/revisions/thresholds_robustness.py`](figures/revisions/thresholds_robustness.py) | Robustness of the reactivation results to the participation threshold and to the detection threshold |
<!-- figure-table:end -->

The table is generated from the scripts' docstrings: `python tools/update_figure_table.py`.

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests on synthetic data
ruff check .    # lint
ruff format .   # format
```

## Citation and license

Please cite the article above ([`CITATION.cff`](CITATION.cff)). The code is
released under the [MIT license](LICENSE).
