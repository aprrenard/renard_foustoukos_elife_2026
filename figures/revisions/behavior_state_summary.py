"""
Reviewer check: session-level behavioral state summary (total water reward,
session duration, total trial count) per mouse x session, across days and
reward groups.

Addresses the concern that behavioral engagement at the end of sessions --
when passive/mapping trials are presented -- might differ between R+/R- mice
and across training days.

Reward logic (5 uL per rewarded trial):
    R+ : auditory hit (auditory_stim==1 & lick_flag==1)
         + whisker hit (whisker_stim==1 & lick_flag==1)
    R- : auditory hit only (auditory_stim==1 & lick_flag==1)

Two comparisons:
    across groups: R+ vs R- on each day (Mann-Whitney U, one session per mouse);
    within groups: pre-learning (days -2, -1) vs post-learning (days +1, +2),
        each mouse's mean over the two days of each period, paired Wilcoxon
        signed-rank test within each reward group (n = mice).

Trial table is recomputed fresh via make_behavior_table() (not loaded from
a precomputed CSV) for the imaging cohort, same selection pattern as
exploratory/behavior/behavior.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from fast_learning import paths, database
from fast_learning.behavior import make_behavior_table
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import significance_stars as _significance_stars
from fast_learning.stats import format_p


# ============================================================================
# Parameters
# ============================================================================

REWARD_UL_PER_TRIAL = 5
DAYS = [-2, -1, 0, 1, 2]
PRE_DAYS = [-2, -1]
POST_DAYS = [1, 2]
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'behavior_state_summary', 'output')


# ============================================================================
# Data
# ============================================================================


def load_behavior_table():
    """Fresh trial table for the imaging cohort (same selection pattern as
    behavior.py's mice_imaging block)."""
    db_path = paths.db_path
    nwb_dir = paths.nwb_dir

    mice_imaging = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes',
    )
    session_list, nwb_list, mice_list, db = database.select_sessions_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        day=["-2", "-1", '0', '+1', '+2'],
        mouse_id=mice_imaging,
    )
    table = make_behavior_table(
        nwb_list,
        session_list,
        db_path,
        cut_session=True,
        stop_flag_yaml=paths.stop_flags_yaml,
        trial_indices_yaml=paths.trial_indices_yaml,
    )
    return table


def compute_session_summary(table):
    """One row per (mouse_id, session_id, day, reward_group):
    n_trials, session_duration_min, total_reward_uL."""
    rows = []
    for (mouse, session, day, rg), g in table.groupby(['mouse_id', 'session_id', 'day', 'reward_group']):
        n_trials = len(g)
        duration_min = (g['start_time'].max() - g['start_time'].min()) / 60

        aud_hits = int(((g['auditory_stim'] == 1) & (g['lick_flag'] == 1)).sum())
        wh_hits = 0
        if rg == 'R+':
            wh_hits = int(((g['whisker_stim'] == 1) & (g['lick_flag'] == 1)).sum())
        reward_uL = REWARD_UL_PER_TRIAL * (aud_hits + wh_hits)

        rows.append(
            {
                'mouse_id': mouse,
                'session_id': session,
                'day': day,
                'reward_group': rg,
                'n_trials': n_trials,
                'session_duration_min': duration_min,
                'total_reward_uL': reward_uL,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Plot
# ============================================================================

METRICS = [
    ('n_trials', 'Number of trials'),
    ('session_duration_min', 'Session duration (min)'),
    ('total_reward_uL', 'Total water reward (uL)'),
]


def plot_session_summary(
    df, days=DAYS, output_dir=OUTPUT_DIR, filename='session_state_summary', save_format='svg', dpi=300
):
    """Bar plot of each metric across days, R+ vs R-, with Mann-Whitney U
    per day (mirrors figure_4h.py's panel_h_reactivation_rate pattern).

    Saves:
        <filename>.svg        -- figure (3 panels)
        <filename>_data.csv   -- per-session summary table
        <filename>_stats.csv  -- Mann-Whitney U results per day per metric
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(days)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    all_stats_rows = []

    for ax, (metric, ylabel) in zip(axes, METRICS):
        stats_rows = []
        p_values = []
        for day in days_sorted:
            r_plus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R+')][metric].dropna().values
            r_minus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R-')][metric].dropna().values
            if len(r_plus_vals) > 0 and len(r_minus_vals) > 0:
                stat, p = mannwhitneyu(r_plus_vals, r_minus_vals, alternative='two-sided')
            else:
                stat, p = np.nan, 1.0
            p_values.append(p)
            stats_rows.append(
                {
                    'metric': metric,
                    'test': 'Mann-Whitney U',
                    'day': day,
                    'R+_n': len(r_plus_vals),
                    'R-_n': len(r_minus_vals),
                    'R+_mean': np.nanmean(r_plus_vals) if len(r_plus_vals) else np.nan,
                    'R-_mean': np.nanmean(r_minus_vals) if len(r_minus_vals) else np.nan,
                    'statistic': stat,
                    'p_value': p,
                    'significance': _significance_stars(p),
                }
            )
        all_stats_rows.extend(stats_rows)

        sns.barplot(
            data=df,
            x='day',
            y=metric,
            hue='reward_group',
            order=days_sorted,
            errorbar=('ci', 95),
            palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
            hue_order=['R+', 'R-'],
            alpha=0.7,
            edgecolor='black',
            ax=ax,
            seed=0,
        )

        y_max = df[metric].max()
        y_range = y_max * 0.05
        width = 0.35
        for day_idx, (day, p) in enumerate(zip(days_sorted, p_values)):
            if not np.isnan(p):
                r_plus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R+')][metric].dropna()
                r_minus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R-')][metric].dropna()
                ci_plus = (
                    r_plus_vals.mean() + 1.96 * r_plus_vals.std() / np.sqrt(len(r_plus_vals))
                    if len(r_plus_vals) > 0
                    else 0
                )
                ci_minus = (
                    r_minus_vals.mean() + 1.96 * r_minus_vals.std() / np.sqrt(len(r_minus_vals))
                    if len(r_minus_vals) > 0
                    else 0
                )
                y1 = max(ci_plus, ci_minus)
                y2 = y1 + y_range
                x1, x2 = day_idx - width / 2, day_idx + width / 2
                ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], 'k-', linewidth=1)
                ax.text((x1 + x2) / 2, y2, format_p(p), ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Day', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(title='', fontsize=9)

    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    df.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved to: {output_dir}")


def plot_pre_post_within_group(df, output_dir=OUTPUT_DIR, filename='session_state_pre_post'):
    """Pre (days -2, -1) vs post (days +1, +2) learning within each reward group.

    Each mouse contributes its mean over the two days of each period (only
    mice with both periods). Paired Wilcoxon signed-rank test per reward group
    and metric (n = mice). Bars: mean across mice with bootstrapped 95% CI;
    lines: individual mice.

    Saves:
        <filename>.pdf         -- figure (one panel per metric)
        <filename>_data.csv    -- per-mouse pre and post means
        <filename>_stats.csv   -- Wilcoxon results per reward group and metric
    """
    set_style()
    period = np.where(df['day'].isin(PRE_DAYS), 'pre', np.where(df['day'].isin(POST_DAYS), 'post', None))
    d = df.assign(period=period).dropna(subset=['period'])
    metrics = [m for m, _ in METRICS]
    mouse_means = (
        d.groupby(['mouse_id', 'reward_group', 'period'])[metrics].mean().unstack('period').dropna()
    )  # columns: (metric, period)

    stats_rows = []
    fig, axes = plt.subplots(1, len(METRICS), figsize=panel_size(len(METRICS)))
    groups = ['R+', 'R-']
    colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    for ax, (metric, ylabel) in zip(axes, METRICS):
        long_rows = []
        for gi, rg in enumerate(groups):
            mm = mouse_means.xs(rg, level='reward_group')[metric]
            pre, post = mm['pre'].values, mm['post'].values
            try:
                stat, p = wilcoxon(pre, post) if len(mm) >= 2 else (np.nan, np.nan)
            except ValueError:  # e.g. all differences zero
                stat, p = np.nan, np.nan
            stats_rows.append(
                {
                    'metric': metric,
                    'reward_group': rg,
                    'test': 'Wilcoxon signed-rank, pre (days -2, -1) vs post (days +1, +2), n = mice',
                    'n_mice': len(mm),
                    'pre_mean': pre.mean(),
                    'post_mean': post.mean(),
                    'statistic': stat,
                    'p_value': p,
                    'significance': _significance_stars(p) if not np.isnan(p) else 'n.a.',
                }
            )
            for mouse, row in mm.iterrows():
                long_rows += [(rg, 'pre', row['pre']), (rg, 'post', row['post'])]
                x = [gi * 3, gi * 3 + 1]
                ax.plot(x, [row['pre'], row['post']], '-', color='grey', linewidth=0.5, alpha=0.6, zorder=3)
            top = np.nanmax(mm.values)
            ax.plot(
                [gi * 3, gi * 3, gi * 3 + 1, gi * 3 + 1],
                [top * 1.04, top * 1.07, top * 1.07, top * 1.04],
                'k-',
                linewidth=0.8,
            )
            ax.text(gi * 3 + 0.5, top * 1.08, format_p(p), ha='center', va='bottom')

        long = pd.DataFrame(long_rows, columns=['reward_group', 'period', 'value'])
        for gi, rg in enumerate(groups):
            sub = long[long['reward_group'] == rg]
            sns.barplot(
                data=sub,
                x=np.where(sub['period'] == 'pre', gi * 3, gi * 3 + 1),
                y='value',
                order=list(range(6)),
                color=colors[rg],
                alpha=0.7,
                edgecolor='black',
                errorbar=('ci', 95),
                seed=0,
                native_scale=True,
                ax=ax,
            )
        ax.set_xticks([0, 1, 3, 4], ['Pre', 'Post', 'Pre', 'Post'])
        ax.set_xlim(-0.7, 4.7)
        for gi, rg in enumerate(groups):
            ax.text(gi * 3 + 0.5, -0.2, rg, transform=ax.get_xaxis_transform(), ha='center', va='top')
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    out = mouse_means.copy()
    out.columns = [f'{m}_{per}' for m, per in out.columns]
    out.reset_index().to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(
        pd.DataFrame(stats_rows)[
            ['metric', 'reward_group', 'n_mice', 'pre_mean', 'post_mean', 'p_value']
        ].to_string(index=False)
    )
    print(f"Pre vs post figure and stats saved to: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Loading behavior table for imaging cohort...")
    table = load_behavior_table()
    print(
        f"Loaded {len(table)} trials across "
        f"{table['session_id'].nunique()} sessions, "
        f"{table['mouse_id'].nunique()} mice"
    )

    summary_df = compute_session_summary(table)
    print(f"\nSummary: {len(summary_df)} sessions")

    plot_session_summary(summary_df)
    plot_pre_post_within_group(summary_df)

    print("\n=== Mean per day / reward group ===")
    print(
        summary_df.groupby(['day', 'reward_group'])[['n_trials', 'session_duration_min', 'total_reward_uL']]
        .mean()
        .to_string()
    )
