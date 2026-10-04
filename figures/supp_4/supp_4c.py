"""
Supplementary Figure 4c: Proportion of cells participating in reactivation
across days for LMI+ vs LMI- cells (binary participation).

Binary participation is determined by circular-shift control: a cell x day is
classified as 'participating' if the cell's real participation rate around
reactivation events exceeds the 95th percentile of a null distribution built
from N_SHIFTS circular shifts of the neural data (same shift applied to all
cells simultaneously, preserving inter-cell correlations).

Panel: Proportion of cells participating across days (-2 to +2) separately
       for LMI+ vs LMI- cells. Per-mouse averages with individual
       trajectories. Stats: linear trend across days with mice as the unit:
       one proportion-vs-day slope per mouse, then a Wilcoxon signed-rank
       test of the slopes against zero, for each of the four groups
       (R+ / R- x LMI+ / LMI-), as in Figure 4j.

Mice: those in the participation mouse selection (>= 3 reactivation events
on day 0; see fast_learning.reactivations). Within those, a mouse x day is
kept only if it has at least 3 valid events (fast_learning.participation).

Inputs:  binary participation per cell-day (pipeline/08_participation.py).
Outputs: <figures_dir>/supp_4/output/supp_4c.pdf, supp_4c_data.csv, supp_4c_stats.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fast_learning import paths, participation
from fast_learning.stats import format_p, per_mouse_slope_test, significance_stars
from fast_learning.plotting import save_figure, set_style, panel_size


DAYS = participation.DAYS
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_4', 'output')


# ============================================================================
# Panel: proportion of participating cells across days (LMI+ vs LMI-)
# ============================================================================


def panel_supp4c_proportion_across_days(
    df,
    output_dir=OUTPUT_DIR,
    filename='supp_4c',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4c: proportion of cells participating across days for
    LMI+ vs LMI- cells (binary participation).

    Per-mouse averages with individual trajectories. Stats: Kruskal-Wallis
    test (effect of day) run independently for each of the four groups
    (R+ positive LMI, R+ negative LMI, R- positive LMI, R- negative LMI).

    Saves:
        <filename>.svg         -- figure
        <filename>_data.csv    -- per-mouse x day x LMI-category proportions
        <filename>_stats.csv   -- Kruskal-Wallis results per group
    """
    set_style()

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    lmi_df = df[df['lmi_category'].isin(lmi_categories)].copy()

    mouse_day_prop = (
        lmi_df.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'], observed=True)['participating']
        .mean()
        .reset_index()
        .rename(columns={'participating': 'proportion'})
    )
    cell_counts = {
        (rg, cat): lmi_df[(lmi_df['reward_group'] == rg) & (lmi_df['lmi_category'] == cat)][
            ['mouse_id', 'roi']
        ]
        .drop_duplicates()
        .shape[0]
        for rg in reward_groups
        for cat in lmi_categories
    }

    # Linear trend across days, tested per mouse (n = mice)
    all_stats_rows = []
    tests = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_data = mouse_day_prop[
                (mouse_day_prop['reward_group'] == rg) & (mouse_day_prop['lmi_category'] == cat)
            ]
            test = per_mouse_slope_test(grp_data, y='proportion')
            if test is None:
                continue
            tests[(rg, cat)] = test
            all_stats_rows.append(
                {
                    'reward_group': rg,
                    'lmi_category': cat,
                    'test': 'Per-mouse day slope, Wilcoxon signed-rank (n = mice)',
                    'mean_day_slope': test['mean_slope'],
                    'median_day_slope': test['median_slope'],
                    'w_stat': test['w_stat'],
                    'p_value': test['p_value'],
                    'significance': significance_stars(test['p_value']),
                    'n_mice': test['n_mice'],
                }
            )
            print(
                f"  {rg} {cat} LMI: median slope={test['median_slope']:.4g}, p={test['p_value']:.4g}, n={test['n_mice']}"
            )

    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), sharey=True)
    plot_data_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_prop[mouse_day_prop['reward_group'] == rg]

        sns.barplot(
            data=grp,
            x='day',
            y='proportion',
            hue='lmi_category',
            hue_order=lmi_categories,
            palette=cat_colors,
            order=days_sorted,
            estimator=np.mean,
            errorbar=('ci', 95),
            capsize=0,
            err_kws={'linewidth': 1.5},
            alpha=0.7,
            ax=ax,
            seed=0,
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        # Individual mouse trajectories
        for j, cat in enumerate(lmi_categories):
            if j >= len(ax.containers):
                continue
            cat_grp = grp[grp['lmi_category'] == cat]
            x_centers = {
                days_sorted[k]: bar.get_x() + bar.get_width() / 2
                for k, bar in enumerate(ax.containers[j])
                if k < len(days_sorted)
            }
            for mouse_id in cat_grp['mouse_id'].unique():
                mdata = cat_grp[cat_grp['mouse_id'] == mouse_id].sort_values('day')
                mx = [x_centers[d] for d in mdata['day'] if d in x_centers]
                my = mdata['proportion'].values
                ax.plot(mx, my, '-', color=cat_colors[cat], linewidth=0.8, alpha=0.4, zorder=5)

        # Annotate Kruskal-Wallis results for each LMI group
        for j, cat in enumerate(lmi_categories):
            test = tests.get((rg, cat))
            p = test['p_value'] if test else float('nan')
            ax.text(
                0.02,
                0.97 - j * 0.12,
                f'{cat.capitalize()} LMI day slope: {format_p(p)}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                color=cat_colors[cat],
            )

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI-: {n_neg} cells)', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Proportion of cells participating' if i == 0 else '')
        ax.set_ylim(0, None)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            [f'{lab.capitalize()} LMI' for lab in labels],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=False,
        )
        sns.despine(ax=ax)

        plot_data_rows.append(grp)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.pdf'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(all_stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Output directory: {OUTPUT_DIR}")
    df = participation.load_binary_participation()

    print(
        f"\nDataset: {len(df)} cell-day records, "
        f"{df['mouse_id'].nunique()} mice, "
        f"{df[['mouse_id', 'roi']].drop_duplicates().shape[0]} unique cells"
    )

    panel_supp4c_proportion_across_days(df, filename='supp_4c')
