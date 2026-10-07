"""
Figure 4i-j revision: participation measured in cells that do not define the events.

Circularity concern: reactivation events are moments when population activity
matches the whisker template, and each cell contributes to that match in
proportion to its template weight. Cells with a strong whisker response
(LMI-positive cells after learning) therefore help create the events they are
then counted as participating in.

Cross-validated control: for each mouse, the cells are split at random into
two halves. Events are detected with half A only (half-A template, half-A
activity, detection threshold recomputed for half A from circular-shift
surrogates of the pre-learning days, as in pipeline step 07), and the
participation of the half-B cells, which played no part in the detection, is
measured in those events; then the halves are swapped. Over N_SPLITS random
splits, each cell is measured N_SPLITS times as a held-out cell; its held-out
participation rate per day is the mean over splits. The same events give the
in-sample participation of the detection-half cells, for comparison.

Everything else is as in Figure 4i-j: correct-rejection trials over the whole
trial, events detected within each trial, participation of a cell = fraction
of valid events in which its baseline-subtracted dF/F (+/- 150 ms) reaches
the participation threshold (10%), mouse selection of the main analysis,
mouse-days with >= 3 valid events.

  4i: day-0 held-out participation vs LMI, LMM (1 | mouse) per reward group.
  4j: held-out participation across days, LMI+ vs LMI- cells, per-mouse day
      slopes tested against zero (Wilcoxon, n = mice).

The split detection runs when its cached CSV is missing or with --recompute.

Outputs: <figures_dir>/revisions/figure_4i_j_heldout_cells/output/
    heldout_participation.csv (mouse x split x cell x day, held-out and in-sample),
    figure_4i_heldout(.pdf, _stats.csv), figure_4j_heldout(.pdf, _data.csv, _stats.csv),
    and the same with _insample for the detection-half cells.
"""

import argparse
import importlib.util
import os
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fast_learning import imaging, paths, participation as pt, reactivations as rx

N_SPLITS = 10
N_SURROGATES = 500  # per pre-learning day, for each half's detection threshold
PERCENTILE = 99
SEED = 0
N_JOBS = 35
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_4i_j_heldout_cells', 'output')
CACHE_CSV = os.path.join(OUTPUT_DIR, 'heldout_participation.csv')

# Panel functions of Figure 4i-j (figure scripts are not a package).
_spec = importlib.util.spec_from_file_location(
    'figure_4i_j', os.path.join(os.path.dirname(__file__), '..', 'figure_4', 'figure_4i_j.py')
)
fig4ij = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig4ij)


def _rng(mouse, *keys):
    return np.random.default_rng([SEED, zlib.crc32(mouse.encode()), *keys])


def half_threshold(mouse, raw_by_day, templates, cells, split):
    """Detection threshold for a subset of cells: median over surrogates of
    the PERCENTILE-th percentile of the template correlation of circularly
    shifted data, pooled over the pre-learning days (as step 07)."""
    pooled = []
    for day in rx.PRELEARNING_DAYS:
        if day not in raw_by_day:
            continue
        data = raw_by_day[day][cells]
        res = rx.compute_surrogate_thresholds(
            data,
            templates[day][cells],
            N_SURROGATES,
            0,
            (PERCENTILE,),
            rng=_rng(mouse, split, int(cells[0]), day + 10),
        )
        pooled.append(res[PERCENTILE]['surrogate_percentiles'])
    return float(np.median(np.concatenate(pooled))) if pooled else np.nan


def process_mouse(mouse, reward_group, threshold=pt.PARTICIPATION_THRESHOLD):
    raw_x = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=False
    )
    sub_x = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=True
    )
    rois = raw_x['roi'].values
    raw_by_day, sub_by_day, templates = {}, {}, {}
    for day in rx.DAYS:
        raw_tr, n = rx.select_trials_by_type(raw_x.sel(trial=raw_x['day'] == day))
        if n < pt.MIN_NOSTIM_TRIALS:
            continue
        sub_tr, _ = rx.select_trials_by_type(sub_x.sel(trial=sub_x['day'] == day))
        raw_by_day[day] = np.nan_to_num(raw_tr.values.reshape(len(rois), -1))
        sub_by_day[day] = np.nan_to_num(sub_tr.values)  # cells x trials x time
        templates[day], _ = rx.create_whisker_template(mouse, day, rx.THRESHOLD_DFF)

    rows = []
    n_cells = len(rois)
    for split in range(N_SPLITS):
        perm = _rng(mouse, split).permutation(n_cells)
        halves = [np.sort(perm[: n_cells // 2]), np.sort(perm[n_cells // 2 :])]
        for h in (0, 1):
            detect, heldout = halves[h], halves[1 - h]
            thr = half_threshold(mouse, raw_by_day, templates, detect, split)
            if np.isnan(thr):
                continue
            for day in raw_by_day:
                _, n_trials, n_t = sub_by_day[day].shape
                corr = rx.compute_template_correlation(raw_by_day[day][detect], templates[day][detect])
                events = rx.detect_reactivation_events(
                    corr, thr, rx.MIN_EVENT_DISTANCE_FRAMES, rx.PROMINENCE, n_timepoints=n_t
                )
                rates, n_valid = pt.participation_from_3d(sub_by_day[day], events, n_t, n_trials, threshold)
                if rates is None:
                    continue
                for role, idx in (('heldout', heldout), ('insample', detect)):
                    for i in idx:
                        rows.append(
                            dict(
                                mouse_id=mouse,
                                reward_group=reward_group,
                                split=split,
                                role=role,
                                roi=rois[i],
                                day=day,
                                participation_rate=rates[i],
                                n_events=n_valid,
                                threshold=thr,
                            )
                        )
    return rows


def per_day_table(cells_df, role):
    """Per cell x day: mean participation over splits (reliable splits only),
    in the format of the participation step (participation.rates_csv)."""
    d = cells_df[(cells_df['role'] == role) & (cells_df['n_events'] >= pt.MIN_EVENTS_FOR_RELIABILITY)]
    out = d.groupby(['mouse_id', 'day', 'roi']).agg(
        participation_rate=('participation_rate', 'mean'), n_events=('n_events', 'mean')
    )
    out = out.reset_index()
    out['reliable'] = True
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure 4i-j with participation of held-out cells.')
    parser.add_argument('--recompute', action='store_true', help='rerun the split detection even if cached')
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.recompute or not os.path.exists(CACHE_CSV):
        selected = rx.load_participation_mice()
        results = pd.read_pickle(os.path.join(rx.RESULTS_DIR, f'reactivation_results_p{PERCENTILE}.pkl'))
        groups = {m: 'R+' for m in results['r_plus_results']} | {m: 'R-' for m in results['r_minus_results']}
        mice = [m for m in groups if m in selected]
        print(f'{len(mice)} mice, {N_SPLITS} splits, {N_SURROGATES} surrogates per pre-learning day and half')
        rows = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(process_mouse)(m, groups[m]) for m in mice)
        cells_df = pd.DataFrame([r for mouse_rows in rows for r in mouse_rows])
        cells_df.to_csv(CACHE_CSV, index=False)
    cells_df = pd.read_csv(CACHE_CSV)

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    groups = cells_df.groupby('mouse_id')['reward_group'].first().to_dict()
    for role in ('heldout', 'insample'):
        per_day = per_day_table(cells_df, role)
        merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day), lmi_df, groups)
        fig4ij.panel_i_participation_vs_lmi(merged, output_dir=OUTPUT_DIR, filename=f'figure_4i_{role}')
        fig4ij.panel_j_participation_across_days(
            merged, per_day, output_dir=OUTPUT_DIR, filename=f'figure_4j_{role}'
        )
    print(f'Saved to {OUTPUT_DIR}')
