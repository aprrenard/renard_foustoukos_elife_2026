"""Pipeline step 08: participation of cells in reactivation events.

Participation is measured in held-out cells by default (--cells heldout): the
events are detected with one random half of the cells and participation is
measured in the other half, over N_SPLITS splits (see
fast_learning.participation). This removes the circularity of measuring a
cell's participation in events it helped detect. --cells all measures
participation of all cells in the events of step 07, for comparison.

Uses the participation mouse selection of step 07 (mice with >= 3 day-0
events; see fast_learning.reactivations).

Parts (all by default, or a subset with --only):
    events  held-out only: events detected by each half of the cells, per split
            and day (surrogate threshold per half; the slow part)
    rates   per-cell participation rates per day, at each participation
            threshold, and per-cell baseline / day-0 / post rates merged
            with LMI                                         (Fig. 4i-j)
            Held-out: also the in-sample rates (cells of the detecting half,
            same events), for comparison.
    day0    day-0 participation rate, spontaneous transient frequency and
            LMI per cell (needs 'rates' at the main threshold) (Supp. 4a-b)
    binary  binary participation per cell-day from the circular-shift
            test, merged with LMI                             (Supp. 4c)

Inputs:  <processed_dir>/reactivation/reactivation_results_p99.pkl,
         mouse_selection.csv, <processed_dir>/lmi_results.csv, tensors.
Outputs: <processed_dir>/reactivation/
             heldout_events.pkl                               (held-out)
             cell_participation_rates_per_day_thr<N><sfx>.csv
             participation_lmi_merged_thr<N><sfx>.csv
             supp4ab_lmi_data_day0<sfx>.csv
             binary_participation_with_lmi<sfx>.csv
         <sfx>: none for held-out cells, _insample (rates only), _allcells;
         with --nolick (no-lick control) followed by _nolick.

Usage:
    python pipeline/08_participation.py [--cells heldout|all] [--only events rates day0 binary] [--nolick]
"""

import argparse
import os
import pickle

import pandas as pd
from joblib import Parallel, delayed

from fast_learning import paths, participation as pt, reactivations as rx


# ============================================================================
# Parameters
# ============================================================================

N_JOBS = 35

RESULTS_DIR = pt.RESULTS_DIR
LMI_RESULTS_CSV = os.path.join(paths.processed_dir, 'lmi_results.csv')


# ============================================================================
# Inputs
# ============================================================================


def load_selected_results(results_file):
    """Reactivation results of the selected mice, and their reward groups."""
    print(f"\nLoading reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    included = rx.load_participation_mice()
    r_plus = {m: r for m, r in data['r_plus_results'].items() if m in included}
    r_minus = {m: r for m, r in data['r_minus_results'].items() if m in included}
    excluded = sorted((set(data['r_plus_results']) | set(data['r_minus_results'])) - included)
    print(f"Selected {len(r_plus)} R+ and {len(r_minus)} R- mice; excluded: {excluded or 'none'}")
    reward_groups = {m: 'R+' for m in data['r_plus_results']}
    reward_groups.update({m: 'R-' for m in data['r_minus_results']})
    return {**r_plus, **r_minus}, reward_groups


def load_heldout_events(nolick=False):
    path = pt.heldout_events_pkl(nolick)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run this step with --only events first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_rates(per_day_df, reward_groups, threshold, nolick, cells):
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day_df), lmi_df, reward_groups)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_day_df.to_csv(pt.rates_csv(threshold, nolick, cells), index=False)
    merged.to_csv(pt.merged_csv(threshold, nolick, cells), index=False)
    print(f"Saved: {pt.rates_csv(threshold, nolick, cells)} ({len(per_day_df)} cell-days)")
    print(f"Saved: {pt.merged_csv(threshold, nolick, cells)} ({len(merged)} cells)")


# ============================================================================
# Parts: held-out cells
# ============================================================================


def run_events(results, nolick=False):
    """Events detected by each half of the cells, per split and day."""
    print(
        f"\nHeld-out detection for {len(results)} mice: {pt.N_SPLITS} splits x 2 halves, "
        f"{pt.N_SURROGATES_HALF} surrogates per pre-learning day and half"
    )
    detections = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(pt.detect_heldout_events)(mouse, rx.trial_window(nolick)) for mouse in results
    )
    out = {
        'parameters': dict(
            n_splits=pt.N_SPLITS,
            n_surrogates=pt.N_SURROGATES_HALF,
            percentile=pt.DETECTION_PERCENTILE,
            seed=pt.HELDOUT_SEED,
            window=rx.trial_window(nolick),
        ),
        'mice': {d['mouse']: d for d in detections},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(pt.heldout_events_pkl(nolick), 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved: {pt.heldout_events_pkl(nolick)}")


def _heldout_mouse(detection, window, what):
    """Rates per split ('rates') or binary participation ('binary') for one mouse."""
    _, days = pt.load_heldout_data(detection['mouse'], window)
    if what == 'rates':
        return pt.heldout_split_rates(detection, days)
    return pt.heldout_circular_shift(detection, days)


def run_rates_heldout(reward_groups, nolick=False):
    """Held-out (and in-sample) participation rates, at each threshold."""
    events = load_heldout_events(nolick)
    split_dfs = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_heldout_mouse)(det, rx.trial_window(nolick), 'rates') for det in events['mice'].values()
    )
    split_df = pd.concat([d for d in split_dfs if d is not None], ignore_index=True)
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        for role in ('heldout', 'insample'):
            print(f"\n--- {role} participation rates, threshold {threshold} ---")
            sel = split_df[(split_df['role'] == role) & (split_df['threshold'] == threshold)]
            _save_rates(pt.average_over_splits(sel), reward_groups, threshold, nolick, role)


def run_binary_heldout(reward_groups, nolick=False):
    """Binary participation of held-out cells (circular-shift test)."""
    events = load_heldout_events(nolick)
    print(f"\nCircular-shift test on held-out events, {len(events['mice'])} mice ({pt.N_SHIFTS} shifts)")
    dfs = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_heldout_mouse)(det, rx.trial_window(nolick), 'binary') for det in events['mice'].values()
    )
    _save_binary(
        pd.concat([d for d in dfs if d is not None], ignore_index=True), reward_groups, nolick, 'heldout'
    )


# ============================================================================
# Parts: all cells (events of step 07)
# ============================================================================


def run_rates_all(results, reward_groups, nolick=False):
    """Participation rates of all cells per day and merged with LMI, at each threshold."""
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        print(f"\n--- participation rates (all cells), threshold {threshold} ---")
        results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(pt.process_mouse_participation)(
                mouse,
                results[mouse],
                participation_threshold=threshold,
                window=rx.trial_window(nolick),
            )
            for mouse in results
        )
        per_day_df = pd.concat([df for _, df in results_list if df is not None], ignore_index=True)
        _save_rates(per_day_df, reward_groups, threshold, nolick, 'all')


def run_binary_all(results, reward_groups, nolick=False):
    """Binary participation of all cells per cell-day (circular-shift test)."""
    print(
        f"\nRunning circular-shift test for {len(results)} mice "
        f"({pt.N_SHIFTS} shifts x {len(pt.DAYS)} days each) ..."
    )
    raw = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(pt.process_mouse_circular_shift)(mouse, results[mouse], pt.N_SHIFTS, rx.trial_window(nolick))
        for mouse in results
    )
    _save_binary(
        pd.concat([df for _, df in raw if df is not None], ignore_index=True), reward_groups, nolick, 'all'
    )


# ============================================================================
# Parts: both
# ============================================================================


def _save_binary(participation_df, reward_groups, nolick, cells):
    participation_df['reward_group'] = participation_df['mouse_id'].map(reward_groups)
    lmi_df = pt.add_lmi_category(pd.read_csv(LMI_RESULTS_CSV))
    merged = pd.merge(
        participation_df,
        lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category']],
        on=['mouse_id', 'roi'],
        how='inner',
    )
    merged.to_csv(pt.binary_csv(nolick, cells), index=False)
    print(f"Saved: {pt.binary_csv(nolick, cells)}")


def run_day0(reward_groups, nolick=False, cells='heldout'):
    """Day-0 participation rate, transient frequency and LMI per cell."""
    part_df = pd.read_csv(pt.rates_csv(pt.PARTICIPATION_THRESHOLD, nolick, cells))
    part_df = part_df[part_df['day'] == 0][['mouse_id', 'roi', 'participation_rate']].copy()
    if len(part_df) == 0:
        raise RuntimeError("No day-0 participation data found.")

    transient_parts = []
    for mouse_id in part_df['mouse_id'].unique():
        print(f"  Computing transient freq for {mouse_id}...")
        transient_parts.append(pt.transient_freq_per_cell(mouse_id, day=0, window=rx.trial_window(nolick)))
    transient_df = pd.concat([d for d in transient_parts if len(d) > 0], ignore_index=True)

    lmi_df = pd.read_csv(LMI_RESULTS_CSV)[['mouse_id', 'roi', 'lmi', 'lmi_p']]
    merged = part_df.merge(transient_df, on=['mouse_id', 'roi'], how='inner')
    merged = merged.merge(lmi_df, on=['mouse_id', 'roi'], how='inner')
    merged['reward_group'] = merged['mouse_id'].map(reward_groups)
    merged = merged.dropna(subset=['reward_group', 'transient_freq', 'participation_rate', 'lmi'])

    merged.to_csv(pt.day0_csv(nolick, cells), index=False)
    print(f"Saved: {pt.day0_csv(nolick, cells)}  ({len(merged)} cells, {merged['mouse_id'].nunique()} mice)")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parts = ['events', 'rates', 'day0', 'binary']
    parser = argparse.ArgumentParser(description='Pipeline step 08: participation in reactivations.')
    parser.add_argument('--only', nargs='+', choices=parts, default=parts)
    parser.add_argument(
        '--cells',
        choices=['heldout', 'all'],
        default='heldout',
        help='held-out cells (main analysis) or all cells in the events of step 07',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (events of step 07 --nolick, -1 to +1 s)'
    )
    args = parser.parse_args()

    results, reward_groups = load_selected_results(
        os.path.join(rx.results_dir(args.nolick), 'reactivation_results_p99.pkl')
    )
    if args.cells == 'heldout':
        if 'events' in args.only:
            run_events(results, args.nolick)
        if 'rates' in args.only:
            run_rates_heldout(reward_groups, args.nolick)
        if 'binary' in args.only:
            run_binary_heldout(reward_groups, args.nolick)
    else:
        if 'rates' in args.only:
            run_rates_all(results, reward_groups, args.nolick)
        if 'binary' in args.only:
            run_binary_all(results, reward_groups, args.nolick)
    if 'day0' in args.only:
        run_day0(reward_groups, args.nolick, args.cells)
