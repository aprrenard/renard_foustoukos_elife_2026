"""Pipeline step 08: participation of cells in reactivation events.

Uses the reactivation events of step 07 and its participation mouse
selection (mice with >= 3 day-0 events; see fast_learning.reactivations).

Parts (all by default, or a subset with --only):
    rates   per-cell participation rates per day, at each participation
            threshold, and per-cell baseline / day-0 / post rates merged
            with LMI                                         (Fig. 4i-j)
    day0    day-0 participation rate, spontaneous transient frequency and
            LMI per cell (needs 'rates' at the main threshold) (Supp. 4a-b)
    binary  binary participation per cell-day from the circular-shift
            test, merged with LMI                             (Supp. 4c)

Inputs:  <processed_dir>/reactivation/reactivation_results_p99.pkl,
         mouse_selection.csv, <processed_dir>/lmi_results.csv, tensors.
Outputs: <processed_dir>/reactivation/
             cell_participation_rates_per_day_thr<N>.csv
             participation_lmi_merged_thr<N>.csv
             supp4ab_lmi_data_day0.csv
             binary_participation_with_lmi.csv

Usage:
    python pipeline/08_participation.py [--only rates day0 binary]
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


# ============================================================================
# Parts
# ============================================================================


def run_rates(results, reward_groups):
    """Participation rates per day and merged with LMI, at each threshold."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        print(f"\n--- participation rates, threshold {threshold} ---")
        results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(pt.process_mouse_participation)(
                mouse,
                results[mouse],
                participation_threshold=threshold,
            )
            for mouse in results
        )
        per_day_df = pd.concat([df for _, df in results_list if df is not None], ignore_index=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        per_day_df.to_csv(pt.rates_csv(threshold), index=False)

        merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day_df), lmi_df, reward_groups)
        merged.to_csv(pt.merged_csv(threshold), index=False)
        print(f"Saved: {pt.rates_csv(threshold)} ({len(per_day_df)} cell-days)")
        print(f"Saved: {pt.merged_csv(threshold)} ({len(merged)} cells)")


def run_day0(reward_groups):
    """Day-0 participation rate, transient frequency and LMI per cell."""
    part_df = pd.read_csv(pt.rates_csv(pt.PARTICIPATION_THRESHOLD))
    part_df = part_df[part_df['day'] == 0][['mouse_id', 'roi', 'participation_rate']].copy()
    if len(part_df) == 0:
        raise RuntimeError("No day-0 participation data found.")

    transient_parts = []
    for mouse_id in part_df['mouse_id'].unique():
        print(f"  Computing transient freq for {mouse_id}...")
        transient_parts.append(pt.transient_freq_per_cell(mouse_id, day=0))
    transient_df = pd.concat([d for d in transient_parts if len(d) > 0], ignore_index=True)

    lmi_df = pd.read_csv(LMI_RESULTS_CSV)[['mouse_id', 'roi', 'lmi', 'lmi_p']]
    merged = part_df.merge(transient_df, on=['mouse_id', 'roi'], how='inner')
    merged = merged.merge(lmi_df, on=['mouse_id', 'roi'], how='inner')
    merged['reward_group'] = merged['mouse_id'].map(reward_groups)
    merged = merged.dropna(subset=['reward_group', 'transient_freq', 'participation_rate', 'lmi'])

    merged.to_csv(pt.DAY0_CSV, index=False)
    print(f"Saved: {pt.DAY0_CSV}  ({len(merged)} cells, {merged['mouse_id'].nunique()} mice)")


def run_binary(results, reward_groups):
    """Binary participation per cell-day (circular-shift test), merged with LMI."""
    print(
        f"\nRunning circular-shift test for {len(results)} mice "
        f"({pt.N_SHIFTS} shifts x {len(pt.DAYS)} days each) ..."
    )
    raw = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(pt.process_mouse_circular_shift)(mouse, results[mouse], pt.N_SHIFTS) for mouse in results
    )
    participation_df = pd.concat([df for _, df in raw if df is not None], ignore_index=True)
    participation_df['reward_group'] = participation_df['mouse_id'].map(reward_groups)

    lmi_df = pt.add_lmi_category(pd.read_csv(LMI_RESULTS_CSV))
    merged = pd.merge(
        participation_df,
        lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category']],
        on=['mouse_id', 'roi'],
        how='inner',
    )
    merged.to_csv(pt.BINARY_CSV, index=False)
    print(f"Saved: {pt.BINARY_CSV}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline step 08: participation in reactivations.')
    parser.add_argument(
        '--only', nargs='+', choices=['rates', 'day0', 'binary'], default=['rates', 'day0', 'binary']
    )
    args = parser.parse_args()

    results, reward_groups = load_selected_results(
        os.path.join(rx.RESULTS_DIR, 'reactivation_results_p99.pkl')
    )
    if 'rates' in args.only:
        run_rates(results, reward_groups)
    if 'day0' in args.only:
        run_day0(reward_groups)
    if 'binary' in args.only:
        run_binary(results, reward_groups)
