"""
Supplementary Figure 4a-b: Participation above chance, controlling for each
cell's activity level.

A cell that is often active crosses the participation threshold at many
moments, events or not, so its participation rate partly reflects its
activity. Its chance rate is the participation expected if events occurred
at random times: the fraction of time points (those where events are
counted) at which its +/- 150 ms mean dF/F reaches the threshold, on the
same day and trials. Participation above chance = participation rate -
chance rate, per cell-day (fast_learning.participation).

Panel a: day-0 participation above chance vs LMI, as Fig. 4i. Stats: linear
         mixed-effects model excess ~ LMI + (1 | mouse), per reward group.
Panel b: participation above chance across days for LMI+ vs LMI- cells, as
         Fig. 4j. Stats: per-mouse day slope, Wilcoxon signed-rank test
         against zero (n = mice).

Participation is measured in held-out cells (events detected with the other
half of the cells, 10 random splits; as Fig. 4i-j). --cells all: all cells in
the events of step 07, for comparison.

Inputs:  participation rates with chance rates (pipeline/08_participation.py).
Outputs: <figures_dir>/supp_4/output/supp_4a_<k>sd<sfx>.pdf, _stats.csv and
         supp_4b_<k>sd<sfx>.pdf, _data.csv, _stats.csv; <sfx> is empty for
         held-out cells, _allcells, then _nolick for the no-lick control.
"""

import argparse
import importlib.util
import os

import pandas as pd

from fast_learning import paths, participation

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_4', 'output')

# Panel functions of Figure 4i-j (figure scripts are not a package).
_spec = importlib.util.spec_from_file_location(
    'figure_4i_j', os.path.join(os.path.dirname(__file__), '..', 'figure_4', 'figure_4i_j.py')
)
fig4ij = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig4ij)


def excess_tables(merged_df, per_day_df):
    """(merged, per_day) with participation_rate replaced by excess_rate."""
    per_day = per_day_df.assign(participation_rate=per_day_df['excess_rate'])
    groups = merged_df.groupby('mouse_id')['reward_group'].first().to_dict()
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    merged = participation.merge_with_lmi(participation.aggregate_across_days(per_day), lmi_df, groups)
    return merged, per_day


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cells',
        choices=['heldout', 'all'],
        default='heldout',
        help='held-out cells (main) or all cells in the events of step 07',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (-1 to +1 s); output names end in _nolick'
    )
    args = parser.parse_args()
    sfx = participation.suffix(args.nolick, args.cells)
    print(f"Output directory: {OUTPUT_DIR}")

    for threshold in participation.PARTICIPATION_THRESHOLDS:
        tag = participation.thr_tag(threshold)
        print(f"\n--- participation threshold {threshold:g} x noise SD ({tag}{sfx}) ---")
        merged, per_day = excess_tables(*participation.load_participation(threshold, args.nolick, args.cells))
        fig4ij.panel_i_participation_vs_lmi(
            merged,
            output_dir=OUTPUT_DIR,
            filename=f'supp_4a_{tag}{sfx}',
            ylabel='Participation above chance (day 0)',
        )
        fig4ij.panel_j_participation_across_days(
            merged,
            per_day,
            output_dir=OUTPUT_DIR,
            filename=f'supp_4b_{tag}{sfx}',
            ylabel='Participation above chance',
            ylim=(None, 0.16),
        )
