"""
Figure 2—figure supplement 1 (revision): inactivation of wS1 in d', learning
after optogenetic inactivation, and inactivation during execution.

d' = Z(whisker hit rate) - Z(catch false-alarm rate), from trial counts with
the log-linear correction (fast_learning.stats.dprime). n = mice throughout.

Row 1, inactivation during learning, in d' (Fig. 2c and 2f use lick
probability):
- a: muscimol on Days 0/+1/+2; b: optogenetics on Day 0 (light off on Day +1).
  wS1 vs fpS1 on each day (Mann-Whitney U).

Row 2, learning on Day +1 after optogenetic inactivation of wS1 on Day 0,
compared with naive mice learning on Day 0 (R+ imaging mice). Only 4 of the 6
wS1 mice have a Day +1 session (MI012 and MI014 have none).
- c: d' in blocks of 10 whisker trials, first 80 whisker trials; each catch
  trial goes to the block of the preceding whisker trial (catch trials before
  the first whisker trial to the first block).
- d: d' of the first block (initial performance); e: slope of d' across blocks
  (rate of learning). Mann-Whitney U.

Row 3, muscimol during execution, in expert mice (a Ringer control day between
two muscimol days; no optogenetics for this cohort):
- f, g: lick probability on whisker, auditory and catch trials, wS1 and fpS1
  mice; h: d', wS1 vs fpS1 on each day (Mann-Whitney U).

Inputs: behavior_muscimol.csv, behavior_opto_learning.csv,
behavior_muscimol_execution.csv, behavior_imagingmice_table_5days_cut.csv
(pipeline/03_behavior_tables.py).

Outputs: <figures_dir>/revisions/figure_2_supp1_inactivation/output/
    inactivation_supplement.pdf,
    learning_dprime_data.csv (session: muscimol and optogenetics during
    learning), execution_data.csv (session), day1_blocks_data.csv (mouse x
    block), day1_mice_data.csv (mouse: first-block d', slope), stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu

from fast_learning import database, paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style, stim_palette
from fast_learning.stats import dprime, format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_2_supp1_inactivation', 'output')
BEHAVIOR_DIR = os.path.join(paths.processed_dir, 'behavior')

AREAS = ['wS1', 'fpS1']
AREA_COLORS = {'wS1': reward_palette[1], 'fpS1': '#8c8c8c'}
OUTCOME_COLORS = {'p_whisker': reward_palette[1], 'p_auditory': stim_palette[0], 'p_catch': stim_palette[2]}

# (table, database prefix, day labels in the database, days shown, tick labels)
MUSCIMOL = (
    'behavior_muscimol.csv',
    'pharma',
    'learning',
    ['muscimol_1', 'muscimol_2', 'muscimol_3'],
    ['Day 0', 'Day +1', 'Day +2'],
)
OPTO = ('behavior_opto_learning.csv', 'opto', 'learning', ['opto', 'recovery_1'], ['Day 0', 'Day +1'])
EXECUTION_DAYS = ['pre_-2', 'pre_-1', 'muscimol_1', 'ringer_1', 'muscimol_2']
EXECUTION = (
    'behavior_muscimol_execution.csv',
    'pharma',
    'execution',
    EXECUTION_DAYS,
    ['-2', '-1', 'M 1', 'Ringer', 'M 2'],
)

BLOCK_SIZE = 10  # whisker trials per block
N_BLOCKS = 8
MIN_BLOCK_TRIALS = 5  # whisker trials, for the last block of short sessions
NAIVE, DAY1 = 'Naive (R+ Day 0)', 'wS1 light, Day +1'
GROUP_COLORS = {NAIVE: reward_palette[1], DAY1: 'black'}


# ============================================================================
# Data
# ============================================================================


def load_inactivation_table(table, prefix, inactivation_type, days):
    """Trial table of an inactivation cohort with the inactivated area and the
    day label of each session from the database."""
    trials = pd.read_csv(paths.adjust_path_to_host(os.path.join(BEHAVIOR_DIR, table)))
    exclude = ['exclude', 'opto_exclude'] if prefix == 'opto' else ['exclude']
    flag = {'opto': 'optogenetic', 'pharma': 'pharmacology'}[prefix]
    for area in AREAS:
        mice = database.select_mice_from_db(
            paths.db_path,
            paths.nwb_dir,
            experimenters=None,
            exclude_cols=exclude,
            **{flag: 'yes', f'{prefix}_inactivation_type': inactivation_type, f'{prefix}_area': area},
        )
        trials.loc[trials.mouse_id.isin(mice), 'area'] = area
    _, _, _, db = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=exclude,
        **{f'{prefix}_inactivation_type': [inactivation_type], f'{prefix}_day': days},
    )
    db = db[['mouse_id', 'session_id', f'{prefix}_day']].rename(columns={f'{prefix}_day': 'inactivation_day'})
    return trials.merge(db, on=['mouse_id', 'session_id'], how='inner').dropna(subset=['area'])


def counts(trials, by):
    """Trial counts, lick probabilities and d' per group of trials."""
    out = trials.groupby(by, observed=True).agg(
        n_whisker=('outcome_w', 'count'),
        hits=('outcome_w', 'sum'),
        n_catch=('outcome_c', 'count'),
        false_alarms=('outcome_c', 'sum'),
        p_whisker=('outcome_w', 'mean'),
        p_auditory=('outcome_a', 'mean'),
        p_catch=('outcome_c', 'mean'),
    )
    out['dprime'] = dprime(out['hits'], out['n_whisker'], out['false_alarms'], out['n_catch'])
    return out.reset_index()


def session_table(cohort):
    table, prefix, inactivation_type, days, _ = cohort
    trials = load_inactivation_table(table, prefix, inactivation_type, days)
    sessions = counts(trials, ['mouse_id', 'session_id', 'area', 'inactivation_day'])
    sessions['inactivation_day'] = pd.Categorical(sessions['inactivation_day'], categories=days, ordered=True)
    return sessions.sort_values(['mouse_id', 'inactivation_day']), trials


def whisker_blocks(trials):
    """d' in consecutive blocks of BLOCK_SIZE whisker trials of one session."""
    s = trials.sort_values('trial_id')
    s = s[(s.whisker_stim == 1) | (s.no_stim == 1)].copy()
    s['block'] = ((s['trial_w'] - 1) // BLOCK_SIZE).ffill().bfill()
    s = s[s['block'] < N_BLOCKS]
    b = counts(s, ['block'])
    return b[b['n_whisker'] >= MIN_BLOCK_TRIALS]


def day1_learning(opto_trials):
    """Block d' of the wS1 optogenetics mice on Day +1 and of the naive R+
    imaging mice on Day 0; per mouse, first-block d' and slope across blocks."""
    naive = pd.read_csv(
        paths.adjust_path_to_host(os.path.join(BEHAVIOR_DIR, 'behavior_imagingmice_table_5days_cut.csv'))
    )
    naive = naive[(naive.day == 0) & (naive.reward_group == 'R+')]
    day1 = opto_trials[(opto_trials.area == 'wS1') & (opto_trials.inactivation_day == 'recovery_1')]
    blocks = []
    for group, trials in [(NAIVE, naive), (DAY1, day1)]:
        for mouse, s in trials.groupby('mouse_id'):
            blocks.append(
                whisker_blocks(s).assign(mouse_id=mouse, group=group, session_id=s.session_id.iloc[0])
            )
    blocks = pd.concat(blocks, ignore_index=True)
    blocks['trial'] = (blocks['block'] + 1) * BLOCK_SIZE
    mice = (
        blocks.groupby(['group', 'mouse_id'])
        .apply(
            lambda b: pd.Series(
                {
                    'first_block_dprime': b.loc[b.block == 0, 'dprime'].iloc[0],
                    'slope': linregress(b['block'], b['dprime']).slope,
                    'n_blocks': len(b),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return blocks, mice


# ============================================================================
# Statistics
# ============================================================================


def mwu_row(panel, label, a, b, names, measure):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = mannwhitneyu(a, b, alternative='two-sided')
    return {
        'panel': panel,
        'measure': measure,
        'day': label,
        'comparison': f'{names[0]} vs {names[1]}',
        'n_1': len(a),
        'n_2': len(b),
        'median_1': np.median(a),
        'median_2': np.median(b),
        'test': 'Mann-Whitney U',
        'statistic': res.statistic,
        'p_value': res.pvalue,
    }


def area_stats(sessions, panel, days, labels):
    rows = []
    for day, label in zip(days, labels):
        d = sessions[sessions.inactivation_day == day]
        rows.append(mwu_row(panel, label, *(d.loc[d.area == a, 'dprime'] for a in AREAS), AREAS, 'dprime'))
    return rows


def day1_stats(mice):
    return [
        mwu_row(
            'c-e', 'Day +1 vs Day 0', *(mice.loc[mice.group == g, m] for g in (DAY1, NAIVE)), (DAY1, NAIVE), m
        )
        for m in ['first_block_dprime', 'slope']
    ]


# ============================================================================
# Plotting
# ============================================================================


def area_bars(ax, sessions, stats, days, labels, title):
    """d' per day, wS1 vs fpS1: mean and bootstrapped 95% CI, dots = mice."""
    d = sessions[sessions.inactivation_day.isin(days)].assign(
        x=lambda t: t.inactivation_day.map(dict(zip(days, labels)))
    )
    kw = dict(data=d, x='x', y='dprime', order=labels, hue='area', hue_order=AREAS, ax=ax)
    sns.barplot(**kw, palette=AREA_COLORS, errorbar=('ci', 95), seed=0, width=0.6, alpha=0.7, legend=True)
    sns.stripplot(**kw, palette=dict.fromkeys(AREAS, 'black'), dodge=True, size=2, jitter=0.1, legend=False)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    for xi, label in enumerate(labels):
        p = stats.loc[stats.day == label, 'p_value'].iloc[0]
        ax.text(
            xi, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5
        )
    ax.set_ylim(-1, 4)
    ax.set_xlabel('')
    ax.set_ylabel("d' (whisker vs catch)")
    ax.set_title(title, pad=8)
    ax.legend(frameon=False, title='', loc='upper right')


def timecourse(ax, sessions, area, labels):
    """Lick probability of each trial type across days: lines = mice, dots =
    mean and bootstrapped 95% CI."""
    d = sessions[sessions.area == area].assign(x=lambda t: t.inactivation_day.cat.codes)
    for col, color in OUTCOME_COLORS.items():
        for _, m in d.groupby('mouse_id'):
            ax.plot(m['x'], 100 * m[col], color=color, alpha=0.4, linewidth=0.5)
        sns.pointplot(
            data=d.assign(v=100 * d[col]),
            x='x',
            y='v',
            native_scale=True,
            color=color,
            errorbar=('ci', 95),
            seed=0,
            markersize=3,
            linewidth=1,
            ax=ax,
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel('')
    ax.set_ylabel('Lick probability (%)')
    ax.set_title(f'{area}, n = {d.mouse_id.nunique()}')


def block_curves(ax, blocks):
    naive = blocks[blocks.group == NAIVE]
    sns.lineplot(
        data=naive,
        x='trial',
        y='dprime',
        color=GROUP_COLORS[NAIVE],
        errorbar=('ci', 95),
        seed=0,
        ax=ax,
        label=f'{NAIVE}, n = {naive.mouse_id.nunique()}',
    )
    day1 = blocks[blocks.group == DAY1]
    for _, m in day1.groupby('mouse_id'):
        ax.plot(m['trial'], m['dprime'], color=GROUP_COLORS[DAY1], alpha=0.4, linewidth=0.5)
    sns.lineplot(
        data=day1,
        x='trial',
        y='dprime',
        color=GROUP_COLORS[DAY1],
        errorbar=None,
        ax=ax,
        label=f'{DAY1}, n = {day1.mouse_id.nunique()}',
    )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(BLOCK_SIZE, (N_BLOCKS + 1) * BLOCK_SIZE, 2 * BLOCK_SIZE))
    ax.set_xlabel('Whisker trial (blocks of 10)')
    ax.set_ylabel("d' (whisker vs catch)")
    ax.set_ylim(-1, 4)
    ax.legend(frameon=False, loc='upper left')


def mouse_points(ax, mice, stats, measure, ylabel, ylim):
    order = [NAIVE, DAY1]
    kw = dict(
        data=mice, x='group', y=measure, order=order, hue='group', hue_order=order, palette=GROUP_COLORS
    )
    sns.stripplot(**kw, size=2.5, jitter=0.15, alpha=0.7, legend=False, ax=ax)
    sns.pointplot(**kw, errorbar=('ci', 95), seed=0, markersize=4, linewidth=1, legend=False, ax=ax)
    p = stats.loc[stats.measure == measure, 'p_value'].iloc[0]
    ax.text(0.5, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks([0, 1], ['Naive\nDay 0', 'wS1 light\nDay +1'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(*ylim)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)


def plot(
    muscimol, opto, execution, blocks, mice, stats, output_dir=OUTPUT_DIR, filename='inactivation_supplement'
):
    set_style()
    fig = plt.figure(figsize=panel_size(3, 3))
    gs = fig.add_gridspec(3, 3)
    area_bars(
        fig.add_subplot(gs[0, :2]),
        muscimol,
        stats[stats.panel == 'a'],
        *MUSCIMOL[3:],
        'Muscimol during learning',
    )
    area_bars(fig.add_subplot(gs[0, 2]), opto, stats[stats.panel == 'b'], *OPTO[3:], 'Optogenetics on Day 0')
    block_curves(fig.add_subplot(gs[1, 0]), blocks)
    day1 = stats[stats.panel == 'c-e']
    mouse_points(
        fig.add_subplot(gs[1, 1]), mice, day1, 'first_block_dprime', "d', first 10 whisker trials", (-1, 4)
    )
    mouse_points(
        fig.add_subplot(gs[1, 2]), mice, day1, 'slope', "Slope of d'\n(per block of 10 trials)", (-0.4, 0.6)
    )
    for col, area in enumerate(AREAS):
        timecourse(fig.add_subplot(gs[2, col]), execution, area, EXECUTION[4])
    exec_days, exec_labels = EXECUTION_DAYS[2:], EXECUTION[4][2:]
    area_bars(
        fig.add_subplot(gs[2, 2]),
        execution,
        stats[stats.panel == 'h'],
        exec_days,
        exec_labels,
        'Muscimol during execution',
    )
    sns.despine()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    muscimol, _ = session_table(MUSCIMOL)
    opto, opto_trials = session_table(OPTO)
    execution, _ = session_table(EXECUTION)
    blocks, mice = day1_learning(opto_trials)

    stats = pd.DataFrame(
        area_stats(muscimol, 'a', *MUSCIMOL[3:])
        + area_stats(opto, 'b', *OPTO[3:])
        + day1_stats(mice)
        + area_stats(execution, 'h', EXECUTION_DAYS[2:], EXECUTION[4][2:])
    )
    print(
        stats[['panel', 'measure', 'day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']]
        .round(3)
        .to_string()
    )
    print(mice.round(2).to_string())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.concat([muscimol.assign(cohort='muscimol'), opto.assign(cohort='optogenetics')]).to_csv(
        os.path.join(OUTPUT_DIR, 'learning_dprime_data.csv'), index=False
    )
    execution.to_csv(os.path.join(OUTPUT_DIR, 'execution_data.csv'), index=False)
    blocks.to_csv(os.path.join(OUTPUT_DIR, 'day1_blocks_data.csv'), index=False)
    mice.to_csv(os.path.join(OUTPUT_DIR, 'day1_mice_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'stats.csv'), index=False)
    plot(muscimol, opto, execution, blocks, mice, stats)
