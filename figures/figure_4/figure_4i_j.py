"""
Figure 4i_j: Reactivation participation rate vs LMI.

Panel i: Scatter plot of day-0 participation rate vs LMI (one dot per cell),
         separately for R+ and R- mice. Stats: linear mixed-effects model
         participation_rate ~ LMI + (1 | mouse), so that cells recorded in
         the same mouse are not treated as independent.

Panel j: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells,
         showing per-mouse averages. Stats: linear trend across days with
         mice as the unit: one participation-vs-day slope per mouse, then a
         Wilcoxon signed-rank test of the slopes against zero, for each
         (reward group, LMI category).

Cells: reliable cells only (>= 3 reactivation events in the period, or on
the day for panel j). Mice: those in the participation mouse selection
(>= 3 reactivation events on day 0; see fast_learning.reactivations).

Inputs:  participation rates from pipeline/08_participation.py, at each
         participation threshold (10% main, 20% and 50% robustness checks).
Outputs: <figures_dir>/figure_4/output/figure_4i_<sel>_thr<N>.pdf, _stats.csv
         and figure_4j_<sel>_thr<N>.pdf, _data.csv, _stats.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fast_learning import paths, participation, reactivations as rx
from fast_learning.plotting import reward_palette, save_figure, set_style, panel_size
from fast_learning.stats import format_p, lmm_slope, per_mouse_slope_test, significance_stars


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]

# Trial selection of the reactivation events: --selection (see
# fast_learning.reactivations.SELECTIONS), default all no-stim trials.

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Panel i: scatter of day-0 participation rate vs LMI, mixed-model stats
# ============================================================================


def panel_i_participation_vs_lmi(merged_df, output_dir=OUTPUT_DIR, filename='figure_4i'):
    """Figure 4 Panel i: scatter of day-0 participation rate vs LMI.

    One dot per cell, separate subplots for R+ and R-. The line and the
    annotated statistic are the fixed-effect slope of
    learning_rate ~ lmi + (1 | mouse_id).

    Saves <filename>.pdf and <filename>_stats.csv (per reward group: slope,
    CI, p-value, intraclass correlation of the mouse term).
    """
    set_style()

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()
    if 'reliable_learning' in df.columns:
        df = df[df['reliable_learning']]

    reward_groups = ['R+', 'R-']
    # Dots are rasterized (cannot be recoloured in Illustrator) and opaque.
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), sharey=True)
    stats_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]
        x = grp['lmi'].values
        n_mice = grp['mouse_id'].nunique()

        ax.scatter(x, grp['learning_rate'].values, color=rg_colors[rg], s=2, linewidths=0, rasterized=True)

        if len(grp) >= 3 and n_mice >= 2:
            fit = lmm_slope(grp, 'learning_rate', 'lmi')
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, fit['slope'] * x_line + fit['intercept'], color='black', linewidth=1, zorder=5)
            ax.text(
                0.05,
                0.95,
                f"LMM slope = {fit['slope']:.3f}\n[{fit['ci_low']:.3f}, {fit['ci_high']:.3f}]\n{format_p(fit['p_value'])}",
                transform=ax.transAxes,
                va='top',
                ha='left',
            )
            stats_rows.append(
                {
                    'reward_group': rg,
                    'test': 'LMM learning_rate ~ lmi + (1 | mouse)',
                    'n_cells': len(grp),
                    'n_mice': n_mice,
                    'lmm_slope': fit['slope'],
                    'lmm_se': fit['se'],
                    'lmm_ci_low': fit['ci_low'],
                    'lmm_ci_high': fit['ci_high'],
                    'p_value': fit['p_value'],
                    'significance': significance_stars(fit['p_value']),
                    'icc_mouse': fit['icc_mouse'],
                    'method': fit['method'],
                }
            )
            print(f"  LMM {rg}: slope={fit['slope']:.4g}, p={fit['p_value']:.4g}, ICC={fit['icc_mouse']:.3f}")

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{rg}\n{len(grp)} cells, {n_mice} mice', fontweight='bold')
        ax.set_xlabel('LMI')
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '')
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel i saved: {os.path.join(output_dir, filename)}")


# ============================================================================
# Panel j: participation rate across days (LMI+ vs LMI-)
# ============================================================================


def panel_j_participation_across_days(merged_df, per_day_df, output_dir=OUTPUT_DIR, filename='figure_4j'):
    """Figure 4 Panel j: participation rate across days for LMI+ vs LMI- cells.

    Bars are means of per-mouse averages. Stats: per-mouse day slope,
    Wilcoxon signed-rank test against zero (n = mice), for each
    (reward_group, lmi_category).

    Saves <filename>.pdf, <filename>_data.csv (per mouse x day x LMI
    category averages) and <filename>_stats.csv.
    """
    set_style()

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    cat_labels = {'positive': 'Positive LMI', 'negative': 'Negative LMI'}
    reward_groups = ['R+', 'R-']

    lmi_cells = merged_df.loc[
        merged_df['lmi_category'].isin(lmi_categories),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    day_data = pd.merge(per_day_df, lmi_cells, on=['mouse_id', 'roi'], how='inner')
    if 'reliable' in day_data.columns:
        day_data = day_data[day_data['reliable']]

    mouse_day_avg = (
        day_data.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'], observed=True)[
            'participation_rate'
        ]
        .mean()
        .reset_index()
    )
    cell_counts = lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True).size().to_dict()

    stats_rows = []
    tests = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) & (mouse_day_avg['lmi_category'] == cat)
            ]
            test = per_mouse_slope_test(grp)
            if test is None:
                continue
            tests[(rg, cat)] = test
            stats_rows.append(
                {
                    'reward_group': rg,
                    'lmi_category': cat,
                    'test': 'Per-mouse day slope, Wilcoxon signed-rank (n = mice)',
                    'mean_day_slope': test['mean_slope'],
                    'median_day_slope': test['median_slope'],
                    'sd_day_slope': test['sd_slope'],
                    'w_stat': test['w_stat'],
                    'p_value': test['p_value'],
                    'significance': significance_stars(test['p_value']),
                    'n_mice': test['n_mice'],
                    'n_cells': cell_counts.get((rg, cat), 0),
                }
            )
            print(
                f"  {rg} {cat} LMI: median slope={test['median_slope']:.4g}, p={test['p_value']:.4g}, n={test['n_mice']}"
            )

    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), sharey=True)
    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_avg[mouse_day_avg['reward_group'] == rg]
        sns.barplot(
            data=grp,
            x='day',
            y='participation_rate',
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

        for j, cat in enumerate(lmi_categories):
            test = tests.get((rg, cat))
            text = (
                f'{cat_labels[cat]}: n.a.'
                if test is None
                else f"{cat_labels[cat]} day slope: {format_p(test['p_value'])}"
            )
            ax.text(
                0.02,
                0.97 - j * 0.09,
                text,
                transform=ax.transAxes,
                va='top',
                ha='left',
                color=cat_colors[cat],
            )

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}\nLMI+: {n_pos} cells, LMI−: {n_neg} cells', fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylabel('Participation rate' if i == 0 else '')
        ax.set_ylim(0, 0.4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            [cat_labels[lab] for lab in labels],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=False,
        )
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    mouse_day_avg.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel j saved: {os.path.join(output_dir, filename)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figure 4i-j: participation in reactivations vs LMI.')
    parser.add_argument('--selection', choices=list(rx.SELECTIONS), default='allnostim')
    SELECTION = parser.parse_args().selection
    print(f"Trial selection:  {SELECTION}")
    print(f"Output directory: {OUTPUT_DIR}")

    for threshold in participation.PARTICIPATION_THRESHOLDS:
        tag = participation.thr_tag(threshold)
        print(f"\n--- participation_threshold={threshold} ({tag}) ---")
        merged_df, per_day_df = participation.load_participation(threshold, SELECTION)
        print(
            f"Dataset: {len(merged_df)} cells, {len(per_day_df)} cell-day records, "
            f"{merged_df['mouse_id'].nunique()} mice"
        )
        panel_i_participation_vs_lmi(merged_df, filename=f'figure_4i_{SELECTION}_{tag}')
        panel_j_participation_across_days(merged_df, per_day_df, filename=f'figure_4j_{SELECTION}_{tag}')
