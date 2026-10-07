"""
Supplementary Figure 4a-b: Spontaneous activity controls for the
LMI-participation relationship.

Panel 4a: Scatter plot of cell participation rate vs spontaneous transient
frequency (Day 0). One dot per cell, color-coded by LMI (blue = negative,
red = positive, centered at 0). One panel per reward group. Stats: linear
mixed-effects model participation_rate ~ transient_freq + (1 | mouse).

Panel 4b: LMI vs participation rate, raw and controlling for spontaneous
transient frequency. Layout: 2 rows (R+, R-) x 2 columns (raw, partial).
Stats: mixed models with mouse as random intercept,
    raw:     participation_rate ~ lmi + (1 | mouse)
    partial: participation_rate ~ lmi + transient_freq + (1 | mouse)
The lmi slope of the partial model (transient frequency held fixed) is the
mixed-model analogue of the partial correlation. The right-hand column plots
residuals after regressing out transient frequency (added-variable plot) for
illustration; the reported statistics come from the mixed models.

Mice: those in the participation mouse selection (>= 3 reactivation events
on day 0; see fast_learning.reactivations).

Inputs:  day-0 participation, transient frequency and LMI per cell
         (pipeline/08_participation.py).
Outputs: <figures_dir>/supp_4/output/supp_4a.pdf, supp_4b.pdf and their
         _data.csv / _stats.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import linregress

from fast_learning import paths, participation
from fast_learning.plotting import reward_palette, save_figure, lmi_cmap, set_style, panel_size
from fast_learning.stats import format_p, lmm_slope


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_4', 'output')


# ============================================================================
# Panel 4a: participation rate vs transient frequency scatter (LMI color)
# ============================================================================


def panel_supp4a_scatter(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4a',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4a: scatter of participation rate vs transient frequency (Day 0).
    One dot per cell, colored by LMI (coolwarm colormap, centered at 0).
    One panel per reward group. Regression line and Pearson r annotated.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data used for the plot
        <filename>_stats.csv -- Pearson r and p per reward group
    """

    set_style()

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'transient_freq', 'participation_rate', 'reward_group'])

    lmi_abs_max = np.abs(merged['lmi']).max()
    norm = mcolors.TwoSlopeNorm(vmin=-lmi_abs_max, vcenter=0, vmax=lmi_abs_max)

    fig, axes = plt.subplots(1, 2, figsize=panel_size(2, w=1.15))
    stats_rows = []

    for i, reward_group in enumerate(['R+', 'R-']):
        ax = axes[i]
        gdata = merged[merged['reward_group'] == reward_group]

        if len(gdata) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_group, fontweight='bold')
            continue

        sc = ax.scatter(
            gdata['transient_freq'],
            gdata['participation_rate'],
            c=gdata['lmi'],
            cmap=lmi_cmap,
            norm=norm,
            alpha=0.6,
            s=2,
            linewidths=0,
        )

        fit = lmm_slope(gdata, 'participation_rate', 'transient_freq')
        x_range = np.linspace(gdata['transient_freq'].min(), gdata['transient_freq'].max(), 100)
        ax.plot(x_range, fit['slope'] * x_range + fit['intercept'], 'k-', linewidth=1)

        n_mice = gdata['mouse_id'].nunique()
        ax.text(
            0.05,
            0.95,
            f"LMM slope = {fit['slope']:.3f} [{fit['ci_low']:.3f}, {fit['ci_high']:.3f}]\n"
            f"{format_p(fit['p_value'])}\nn = {len(gdata)} cells, {n_mice} mice",
            transform=ax.transAxes,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

        plt.colorbar(sc, ax=ax, label='LMI')
        ax.set_xlabel('Transient frequency (events/min)', fontweight='bold')
        ax.set_ylabel('Participation rate' if i == 0 else '', fontweight='bold')
        ax.set_title(f'{reward_group}  (n={n_mice} mice, {len(gdata)} cells)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

        stats_rows.append(
            {
                'reward_group': reward_group,
                'test': 'LMM participation_rate ~ transient_freq + (1 | mouse)',
                'lmm_slope': fit['slope'],
                'lmm_ci_low': fit['ci_low'],
                'lmm_ci_high': fit['ci_high'],
                'p_value': fit['p_value'],
                'icc_mouse': fit['icc_mouse'],
                'method': fit['method'],
                'n_cells': len(gdata),
                'n_mice': n_mice,
            }
        )
        print(
            f"  {reward_group}: LMM slope={fit['slope']:.4g}, p={fit['p_value']:.4g}, n={len(gdata)} cells, {n_mice} mice"
        )

    fig.suptitle(
        'Participation Rate vs Transient Frequency (Day 0, colored by LMI)',
        fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.pdf'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    merged.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Panel 4b: partial correlation — LMI vs participation | transient freq
# ============================================================================


def panel_supp4b_partial_corr(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4b',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4b: added-variable scatter plots comparing raw and partial
    correlation between LMI and participation rate.

    Layout: 2 rows (R+, R-) x 2 columns (raw, partial).
      Left column : raw scatter of LMI vs participation_rate.
      Right column: residuals of LMI and participation_rate after regressing
                    each on transient_freq (partial regression / added-variable
                    plot). Slope and r equal the partial regression coefficient
                    and partial correlation.

    If the partial r (right) remains significant and close in magnitude to the
    raw r (left), spontaneous activity does not explain the LMI-participation
    relationship.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data with residuals per reward group
        <filename>_stats.csv -- raw and partial Pearson r / p per reward group
    """
    set_style()

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'participation_rate', 'transient_freq', 'reward_group'])

    group_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    def residuals(a, b):
        slope, intercept, _, _, _ = linregress(b, a)
        return a - (slope * b + intercept)

    def annotate(ax, label, fit):
        ax.text(
            0.05,
            0.95,
            f"{label}\nLMM slope = {fit['slope']:.3f} [{fit['ci_low']:.3f}, {fit['ci_high']:.3f}]\n{format_p(fit['p_value'])}",
            transform=ax.transAxes,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

    fig, axes = plt.subplots(2, 2, figsize=panel_size(2, 2), sharey=True)
    stats_rows = []
    data_rows = []

    for row, reward_group in enumerate(['R+', 'R-']):
        color = group_colors[reward_group]
        gdata = merged[merged['reward_group'] == reward_group].copy()

        if len(gdata) < 5:
            for col in range(2):
                axes[row, col].text(
                    0.5,
                    0.5,
                    'Insufficient data',
                    ha='center',
                    va='center',
                    transform=axes[row, col].transAxes,
                )
            continue

        lmi = gdata['lmi'].values
        part = gdata['participation_rate'].values
        freq = gdata['transient_freq'].values

        lmi_resid = residuals(lmi, freq)
        part_resid = residuals(part, freq)

        fit_raw = lmm_slope(gdata, 'participation_rate', 'lmi')
        fit_partial = lmm_slope(gdata, 'participation_rate', 'lmi', covariates=['transient_freq'])

        n_mice = gdata['mouse_id'].nunique()
        print(
            f"  {reward_group}  raw LMM slope={fit_raw['slope']:.3f} p={fit_raw['p_value']:.4g} | "
            f"partial LMM slope={fit_partial['slope']:.3f} p={fit_partial['p_value']:.4g}  "
            f"(n={len(lmi)} cells, {n_mice} mice)"
        )

        stats_rows.append(
            {
                'reward_group': reward_group,
                'test': 'LMM (1 | mouse): raw lmi; partial lmi + transient_freq',
                'raw_slope': fit_raw['slope'],
                'raw_ci_low': fit_raw['ci_low'],
                'raw_ci_high': fit_raw['ci_high'],
                'raw_p': fit_raw['p_value'],
                'partial_slope': fit_partial['slope'],
                'partial_ci_low': fit_partial['ci_low'],
                'partial_ci_high': fit_partial['ci_high'],
                'partial_p': fit_partial['p_value'],
                'raw_method': fit_raw['method'],
                'partial_method': fit_partial['method'],
                'n_cells': len(lmi),
                'n_mice': n_mice,
            }
        )

        gdata = gdata.copy()
        gdata['lmi_resid'] = lmi_resid
        gdata['part_resid'] = part_resid
        data_rows.append(gdata)

        for col, (x, y, label, fit, xlabel, ylabel) in enumerate(
            [
                (lmi, part, 'Raw', fit_raw, 'LMI', 'Participation rate'),
                (
                    lmi_resid,
                    part_resid,
                    'Partial (ctrl transient freq)',
                    fit_partial,
                    'LMI  (residual | transient freq)',
                    'Participation rate  (residual | transient freq)',
                ),
            ]
        ):
            ax = axes[row, col]
            ax.scatter(x, y, color=color, alpha=0.3, s=2, linewidths=0)

            # Line: least-squares fit of the plotted points (illustration).
            slope, intercept, _, _, _ = linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color='black', linewidth=1)

            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            annotate(ax, label, fit)

            ax.set_xlim(-1, 1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if col == 0 else '')
            title = 'Raw' if col == 0 else 'Partial  (ctrl transient freq)'
            ax.set_title(
                f'{reward_group} — {title}  (n={len(lmi)} cells, {n_mice} mice)',
                fontweight='bold',
            )
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

    fig.suptitle(
        'LMI vs Participation Rate: Raw and Partial (Day 0, LMM with mouse random intercept)',
        fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.pdf'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    data_csv = participation.DAY0_CSV
    print(f"Input:            {data_csv}")
    print(f"Output directory: {OUTPUT_DIR}")
    participation.load_day0()  # fails early if step 08 has not been run

    print("\nPlotting panel supp_4a...")
    panel_supp4a_scatter(data_csv, filename='supp_4a')

    print("\nPlotting panel supp_4b...")
    panel_supp4b_partial_corr(data_csv, filename='supp_4b')

    print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
