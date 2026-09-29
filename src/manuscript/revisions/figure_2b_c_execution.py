"""
Figure 2b-c revision: Muscimol inactivation during execution.

Addresses a reviewer comment asking whether the wS1/fpS1 muscimol
inactivation effect shown in Figure 2b-c (run during learning) is specific
to the learning phase, or also impairs performance in already-expert mice.
This script reruns the same analysis on a separate cohort inactivated
during execution instead of learning.

The execution paradigm has no recovery days. Instead, a muscimol day is
followed by an interleaved ringer (saline) control day and then a second
muscimol day: pre_-2, pre_-1, muscimol_1, ringer_1, muscimol_2. Only
muscimol (no optogenetic) inactivation exists for this cohort.

Unlike figure_2b_c.py, which reads a pre-built trial-level CSV
(behavior_muscimol.csv, built from learning sessions only), this script
first extracts its own trial-level table from NWB files for the execution
sessions via make_behavior_table (src/utils/utils_behavior.py) — the same
function originally used to build the learning CSV — and caches it to a
separate file so behavior_muscimol.csv is left untouched.

Execution modes:
    MODE = 'compute' : extract the trial table from NWB files, cache it,
                        then plot (slow, needs NWB/network-drive access)
    MODE = 'plot'     : load the previously cached CSV and plot only

Figures and CSVs are saved to
    io.manuscript_output_dir/revisions/figure_2b_c_execution/output/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_behavior import make_behavior_table
from src.utils.utils_plot import stim_palette, reward_palette


MODE = 'compute'

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'revisions', 'figure_2b_c_execution', 'output')
TABLE_PATH = os.path.join(io.processed_dir, 'behavior', 'behavior_muscimol_execution.csv')

INACTIVATION_LABELS = ['pre_-2', 'pre_-1', 'muscimol_1', 'ringer_1', 'muscimol_2']
DAYS_OF_INTEREST = ['muscimol_1', 'ringer_1', 'muscimol_2']
DAY_LABELS = ['M1', 'Ringer', 'M2']


# ============================================================================
# Data extraction
# ============================================================================

def _compute_behavior_table(table_path=TABLE_PATH):
    """Extract trial-level behavior data for execution-inactivation sessions
    from NWB files and cache it to `table_path`.
    """
    session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
        io.db_path, io.nwb_dir, experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['execution'],
        pharma_day=INACTIVATION_LABELS,
    )
    print(f"Building execution behavior table: {len(session_list)} sessions, "
          f"{len(mice_list)} mice")

    table = make_behavior_table(
        nwb_list, session_list, io.db_path, cut_session=True,
        stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml,
    )

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"Execution behavior table saved to: {table_path}")

    return table


def _load_behavior_table(table_path=TABLE_PATH):
    table_path = io.adjust_path_to_host(table_path)
    return pd.read_csv(table_path)


# ============================================================================
# Panel b: Muscimol inactivation across days (execution)
# ============================================================================

def panel_b_muscimol_timecourse_execution(
    table=None,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 2 Panel b (execution revision): Muscimol inactivation
    timecourse during execution.

    Shows performance across pre-injection, muscimol, and ringer-control
    days for wS1 and fpS1 inactivation mice.

    Args:
        table: Pre-loaded trial-level behavior table (optional, will load
            or compute if None)
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    if table is None:
        table = _compute_behavior_table() if MODE == 'compute' else _load_behavior_table()

    db_path = io.db_path
    nwb_dir = io.nwb_dir

    fpS1_mice = io.select_mice_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='execution',
        pharma_area='fpS1',
    )

    wS1_mice = io.select_mice_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='execution',
        pharma_area='wS1',
    )
    print(f"wS1 mice (execution): {wS1_mice}")
    print(f"fpS1 mice (execution): {fpS1_mice}")

    table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
    table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

    _, _, _, db = io.select_sessions_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['execution'],
        pharma_day=INACTIVATION_LABELS,
    )

    table = pd.merge(
        table,
        db[['mouse_id', 'session_id', 'pharma_day']],
        on=['mouse_id', 'session_id'],
        how='left'
    )

    data = table.groupby(
        ['mouse_id', 'session_id', 'pharma_day', 'area'],
        as_index=False
    )[['outcome_c', 'outcome_a', 'outcome_w']].agg('mean')

    data['pharma_day'] = pd.Categorical(
        data['pharma_day'],
        categories=INACTIVATION_LABELS,
        ordered=True
    )
    data = data.sort_values(by=['mouse_id', 'pharma_day'])

    data['outcome_c'] = data['outcome_c'] * 100
    data['outcome_a'] = data['outcome_a'] * 100
    data['outcome_w'] = data['outcome_w'] * 100

    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        rc={
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none'
        }
    )

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # ========================================================================
    # Left panel: wS1 inactivation
    # ========================================================================
    ax = axes[0]

    for imouse in wS1_mice:
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_c', estimator=np.mean,
            color=stim_palette[2], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_a', estimator=np.mean,
            color=stim_palette[0], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_w', estimator=np.mean,
            color=reward_palette[1], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )

    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='pharma_day', y='outcome_c', order=INACTIVATION_LABELS,
        color=stim_palette[2], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='pharma_day', y='outcome_a', order=INACTIVATION_LABELS,
        color=stim_palette[0], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='pharma_day', y='outcome_w', order=INACTIVATION_LABELS,
        color=reward_palette[1], ax=ax, linewidth=2
    )

    ax.set_title('wS1')

    # ========================================================================
    # Right panel: fpS1 inactivation
    # ========================================================================
    ax = axes[1]

    for imouse in fpS1_mice:
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_c', estimator=np.mean,
            color=stim_palette[2], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_a', estimator=np.mean,
            color=stim_palette[0], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='pharma_day', y='outcome_w', estimator=np.mean,
            color=reward_palette[1], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )

    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='pharma_day', y='outcome_c', order=INACTIVATION_LABELS,
        color=stim_palette[2], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='pharma_day', y='outcome_a', order=INACTIVATION_LABELS,
        color=stim_palette[0], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='pharma_day', y='outcome_w', order=INACTIVATION_LABELS,
        color=reward_palette[1], ax=ax, linewidth=2
    )

    ax.set_title('fpS1')

    for ax in axes:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['-2', '-1', 'M 1', 'Ringer', 'M 2'])
        ax.set_xlabel('Muscimol inactivation during execution')
        ax.set_ylabel('Lick probability (%)')

    sns.despine(trim=True)

    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_2b_execution.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    data_file = os.path.join(save_path, 'figure_2b_execution_data.csv')
    data.to_csv(data_file, index=False)

    print(f"Figure 2b (execution) saved to: {output_file}")
    print(f"Figure 2b (execution) data saved to: {data_file}")

    return data


# ============================================================================
# Panel c: Bar plot quantification for M1, Ringer, M2 (execution)
# ============================================================================

def panel_c_muscimol_barplot_execution(
    data=None,
    days_of_interest=DAYS_OF_INTEREST,
    day_labels=DAY_LABELS,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 2 Panel c (execution revision): Bar plot comparison of
    wS1 vs fpS1 during execution.

    Shows whisker trial performance for M1, Ringer, M2 with bar plots and
    statistical comparisons between wS1 and fpS1 inactivation.

    Args:
        data: Pre-processed data from panel_b (optional, will compute if None)
        days_of_interest: List of pharma_day labels to compare
        day_labels: Corresponding display labels for days
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    if data is None:
        data = panel_b_muscimol_timecourse_execution(save_path=save_path)

    day_data = data[data['pharma_day'].isin(days_of_interest)].copy()
    day_data['day_label'] = day_data['pharma_day'].map(
        dict(zip(days_of_interest, day_labels))
    )

    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=day_data,
        x='day_label',
        y='outcome_w',
        hue='area',
        palette=[reward_palette[1]],
        width=0.3,
        dodge=True
    )

    sns.swarmplot(
        data=day_data,
        x='day_label',
        y='outcome_w',
        hue='area',
        dodge=True,
        color=stim_palette[2],
        alpha=0.6
    )

    plt.xlabel('Day')
    plt.ylabel('Whisker Performance (%)')
    plt.ylim([0, 100])
    plt.legend(title='Area')
    sns.despine()

    stats = []
    for day, label in zip(days_of_interest, day_labels):
        df_day = day_data[day_data['pharma_day'] == day]
        group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
        group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']

        stat, p_value = mannwhitneyu(
            group_wS1, group_fpS1,
            alternative='two-sided'
        )
        stats.append({'day': label, 'statistic': stat, 'p_value': p_value})

        ax = plt.gca()
        xpos = day_labels.index(label)
        ypos = 95

        if p_value < 0.001:
            plt.text(xpos, ypos, '***', ha='center', va='bottom',
                    color='black', fontsize=14)
        elif p_value < 0.01:
            plt.text(xpos, ypos, '**', ha='center', va='bottom',
                    color='black', fontsize=14)
        elif p_value < 0.05:
            plt.text(xpos, ypos, '*', ha='center', va='bottom',
                    color='black', fontsize=14)

        plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom',
                color='black', fontsize=10)

    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_2c_execution.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    data_file = os.path.join(save_path, 'figure_2c_execution_data.csv')
    stats_file = os.path.join(save_path, 'figure_2c_execution_stats.csv')
    day_data.to_csv(data_file, index=False)
    pd.DataFrame(stats).to_csv(stats_file, index=False)

    print(f"Figure 2c (execution) saved to: {output_file}")
    print(f"Figure 2c (execution) data saved to: {data_file}")
    print(f"Figure 2c (execution) statistics saved to: {stats_file}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    table = _compute_behavior_table() if MODE == 'compute' else _load_behavior_table()

    data = panel_b_muscimol_timecourse_execution(table=table)

    panel_c_muscimol_barplot_execution(data=data)
