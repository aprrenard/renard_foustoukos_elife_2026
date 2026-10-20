"""
Figure 3—figure supplement 3a-e: Stability of the passive whisker responses
within the passive block, and day-to-day drift between the baseline days
(-2/-1) and across learning (-1/0)

- a: within each day, correlation of the mean response vectors of the first
  and the last 20 of the 40 passive trials, and of the odd and the even
  trials (same number of trials, no separation in time); each mouse's median
  over the five days; Wilcoxon signed-rank per group.
- b: pattern correlation (noise-corrected; 1 = same pattern) within the
  passive block (first vs last 20 trials, median over days) and between
  consecutive days; within-block and -1/0 vs -2/-1 within group (Wilcoxon
  signed-rank, p-values above them) and R+ vs R- on -2/-1 (Mann-Whitney U).
- c-e: LMI computed between consecutive days (one day on each side, as
  pipeline step 05 otherwise): fraction of LMI+ cells (c), LMI- cells (d) and
  the net fraction (e, LMI+ minus LMI-), with the two-day LMI of Fig. 3
  (-2,-1 / +1,+2) for reference. Change from -2/-1 to -1/0 within group
  (Wilcoxon signed-rank, p-values above -1/0); net fraction against zero
  (Wilcoxon signed-rank, p-values above -2/-1 and the two-day LMI); R+ vs R-
  (Mann-Whitney U, stats file only).
Points: mean and bootstrapped 95% CI across mice. See fast_learning.drift.

Outputs: <figures_dir>/Figure3_SupFig3/figure3_supfig3_a_e.pdf,
figure3_supfig3_a_data.csv (mouse x day), figure3_supfig3_b_data.csv
(mouse x pair), figure3_supfig3_c_e_data.csv (mouse x pair),
figure3_supfig3_c_e_cells.csv (cell x pair), figure3_supfig3_a_stats.csv,
figure3_supfig3_b_stats.csv, figure3_supfig3_c_e_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fast_learning import database, paths
from fast_learning.drift import (
    DAY_PAIRS,
    FRACTIONS,
    GROUP_COLORS,
    LEARNING_PAIR,
    LMI_POSITIVE_THRESHOLD,
    REFERENCE,
    WITHIN,
    compute_metrics,
    drift_lmi,
    drift_lmi_stats,
    fractions,
    load_response_vectors,
    pair_label,
    signal_noise_stats,
    within_block,
    within_block_means,
    within_block_stats,
)
from fast_learning.plotting import panel_size, save_figure, set_style
from fast_learning.stats import format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure3_SupFig3')


def pair_points(ax, data, y, order, legend=False, x='pair'):
    """Mean and bootstrapped 95% CI across mice for each day pair."""
    sns.pointplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        dodge=0.3,
        markersize=3,
        linewidth=1,
        legend=legend,
        ax=ax,
    )
    ax.set_xlabel('Day pair')


def within_block_panel(ax, means, stats):
    """First/last vs odd/even correlation within each day."""
    splits = {'r_first_last': 'First vs\nlast 20', 'r_odd_even': 'Odd vs\neven'}
    long = means.reset_index().melt(
        id_vars=['reward_group', 'mouse_id'], value_vars=list(splits), var_name='split', value_name='r'
    )
    pair_points(ax, long, 'r', list(splits), legend=True, x='split')
    ax.set_xticks(range(len(splits)), list(splits.values()))
    ax.set_xlabel('Trials of the same day')
    pvalues(ax, 0.5, stats[stats['measure'] == 'r all days'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Pattern correlation')
    ax.legend(frameon=False, loc='lower left')


def pattern_drift_panel(ax, drift_df, stats):
    """Noise-corrected correlation within the passive block and between
    consecutive days."""
    order = [WITHIN] + [pair_label(*p) for p in DAY_PAIRS]
    pair_points(ax, drift_df, 'r_corrected', order)
    ax.set_xticks(range(len(order)), ['Within\nday', *order[1:]])
    ax.set_xlabel('')
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.6)
    # p-values above each pair: -1/0 vs -2/-1 within group (colored), R+ vs R- (black).
    for gi, rg in enumerate(['R+', 'R-', 'R+ vs R-']):
        for xi, pair in enumerate(order):
            row = stats[(stats['reward_group'] == rg) & (stats['measure'] == f'drift {pair}')]
            if len(row):
                ax.text(
                    xi,
                    1.04 + 0.06 * gi if rg in GROUP_COLORS else 1.04,
                    ('' if rg in GROUP_COLORS else 'R+ vs R−\n') + format_p(row['p_value'].iloc[0]),
                    ha='center',
                    va='bottom',
                    color=GROUP_COLORS.get(rg, 'black'),
                    fontsize=5,
                )
    ax.set_ylim(0.4, 1.25)
    ax.set_ylabel('Pattern correlation\n(noise-corrected)')


def pvalues(ax, x, rows):
    for gi, rg in enumerate(['R+', 'R-']):  # bottom to top
        r = rows[rows['reward_group'] == rg]
        if len(r):
            ax.text(
                x,
                1.0 + 0.07 * gi,
                format_p(r['p_value'].iloc[0]),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                color=GROUP_COLORS[rg],
                fontsize=5,
            )


def lmi_panel(ax, frac, stats, m):
    order = [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]
    pair_points(ax, frac.assign(pct=100 * frac[m]), 'pct', order)
    ax.axvline(len(DAY_PAIRS) - 0.5, color='grey', linewidth=0.4)
    ax.set_xticks(range(len(order)), [*order[:-1], 'LMI\n(2 days)'])
    rows = stats[stats['measure'] == m]
    pvalues(ax, order.index(pair_label(*LEARNING_PAIR)), rows[rows['pair'].str.contains('minus')])
    if m == 'net':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(-25, 25)
        for pair in [order[0], REFERENCE]:
            pvalues(ax, order.index(pair), rows[rows['pair'] == pair])
    else:
        ax.axhline(100 * (1 - LMI_POSITIVE_THRESHOLD), color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(0, 30)
    ax.set_ylabel(FRACTIONS[m])


if __name__ == '__main__':
    vectors = load_response_vectors()
    day_df, drift_df = compute_metrics(vectors)
    drift_stats = signal_noise_stats(day_df, drift_df)[0]
    drift_stats = drift_stats[drift_stats['measure'].str.startswith('drift')]
    wb = within_block(vectors)
    wb_means = within_block_means(wb)
    wb_stats = within_block_stats(wb, drift_df)
    within = wb_means['r_corrected'].reset_index().assign(pair=WITHIN)
    pattern_df = pd.concat([within, drift_df], ignore_index=True)
    pattern_stats = pd.concat([wb_stats[wb_stats['measure'] == f'drift {WITHIN}'], drift_stats])

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    _, _, _, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    groups = {
        m: database.get_mouse_reward_group_from_db(paths.db_path, m, db) for m in lmi_df['mouse_id'].unique()
    }
    cells = drift_lmi(groups, lmi_df)
    reference = lmi_df.assign(pair=REFERENCE, reward_group=lmi_df['mouse_id'].map(groups))
    frac = fractions(pd.concat([cells, reference[cells.columns.intersection(reference.columns)]]))
    lmi_stats = drift_lmi_stats(frac)

    set_style()
    fig, axes = plt.subplots(1, 5, figsize=panel_size(5))
    within_block_panel(axes[0], wb_means, wb_stats)
    pattern_drift_panel(axes[1], pattern_df, pattern_stats)
    for ax, m in zip(axes[2:], FRACTIONS):
        lmi_panel(ax, frac, lmi_stats, m)
    sns.despine()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_e.pdf'))
    plt.close(fig)

    wb.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_data.csv'), index=False)
    wb_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_stats.csv'), index=False)
    pattern_df.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_b_data.csv'), index=False)
    pattern_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_b_stats.csv'), index=False)
    frac.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_c_e_data.csv'), index=False)
    cells.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_c_e_cells.csv'), index=False)
    lmi_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_c_e_stats.csv'), index=False)
    print(wb_stats.drop(columns=['test']).round(4).to_string(index=False))
    print(
        drift_stats[['reward_group', 'measure', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
    print(
        lmi_stats[['pair', 'measure', 'reward_group', 'median_r_plus', 'median_r_minus', 'p_value']]
        .round(4)
        .to_string(index=False)
    )
