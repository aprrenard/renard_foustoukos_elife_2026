"""
Figure 2—figure supplements 1-3: inactivation of wS1 during learning in d',
learning on the day after optogenetic inactivation, and inactivation during
execution.

d' = Z(whisker hit rate) - Z(catch false-alarm rate), from trial counts with
the log-linear correction (fast_learning.stats.dprime). Lick probability is
the fraction of whisker (or auditory, catch) trials with a lick, as in
Fig. 2. Bars: mean and bootstrapped 95% CI across mice; dots (swarm): mice. n = mice
throughout; all tests are Mann-Whitney U.

Supplement 1, inactivation during learning in d' (Fig. 2c and 2f use lick
probability):
- a: muscimol on Days 0/+1/+2; b: optogenetics on Day 0, light off on Day +1.
  wS1 vs fpS1 on each day.

Supplement 2, learning on Day +1 after optogenetic inactivation of wS1 on
Day 0 (light off), compared with naive mice learning on Day 0 (R+ imaging
mice). Only 4 of the 6 wS1 mice were trained on Day +1 (MI012 and MI014 were
not). Top row whisker lick probability, bottom row d':
- a: whole session; b: blocks of 20 trials (all trial types, early-lick
  trials excluded; block_id of Fig. 1B), first 10 blocks, which all sessions
  complete; c: first block.

Supplement 3, muscimol during execution in expert mice (a Ringer control day
between two muscimol days; no optogenetics in this cohort):
- a, b: lick probability on whisker, auditory and catch trials across days,
  wS1 and fpS1 mice; c: whisker lick probability and d: d', wS1 vs fpS1 on
  each day.

Inputs: behavior_muscimol.csv, behavior_opto_learning.csv,
behavior_muscimol_execution.csv, behavior_imagingmice_table_5days_cut.csv
(pipeline/03_behavior_tables.py).

Outputs: <figures_dir>/revisions/figure_2_supp_inactivation/output/
    figure_2_supp1_dprime.pdf, figure_2_supp2_day1.pdf,
    figure_2_supp3_execution.pdf,
    supp1_data.csv (session), supp2_sessions_data.csv (session),
    supp2_blocks_data.csv (session x block), supp3_data.csv (session),
    supp1_stats.csv, supp2_stats.csv, supp3_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import database, paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style, stim_palette
from fast_learning.stats import dprime, format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_2_supp_inactivation', 'output')
BEHAVIOR_DIR = os.path.join(paths.processed_dir, 'behavior')

AREAS = ['wS1', 'fpS1']
AREA_COLORS = {'wS1': reward_palette[1], 'fpS1': '#8c8c8c'}
OUTCOME_COLORS = {'p_whisker': reward_palette[1], 'p_auditory': stim_palette[0], 'p_catch': stim_palette[2]}

# (table, database prefix, inactivation type, day labels in the database, tick labels)
MUSCIMOL = (
    'behavior_muscimol.csv',
    'pharma',
    'learning',
    ['muscimol_1', 'muscimol_2', 'muscimol_3'],
    ['Day 0', 'Day +1', 'Day +2'],
)
OPTO = ('behavior_opto_learning.csv', 'opto', 'learning', ['opto', 'recovery_1'], ['Day 0', 'Day +1'])
EXECUTION = (
    'behavior_muscimol_execution.csv',
    'pharma',
    'execution',
    ['pre_-2', 'pre_-1', 'muscimol_1', 'ringer_1', 'muscimol_2'],
    ['-2', '-1', 'M 1', 'Ringer', 'M 2'],
)
EXECUTION_TESTED = slice(2, None)  # the muscimol and Ringer days

N_BLOCKS = 10  # blocks of 20 trials completed by every session
NAIVE, DAY1 = 'Naive, Day 0', 'wS1 light, Day +1'
GROUP_COLORS = {NAIVE: reward_palette[1], DAY1: 'black'}
MEASURES = {'p_whisker': 'Whisker lick probability (%)', 'dprime': "d' (whisker vs catch)"}
YLIMS = {'p_whisker': (0, 100), 'dprime': (-1, 4)}


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
    """Trial counts, lick probabilities (%) and d' per group of trials."""
    out = trials.groupby(by, observed=True).agg(
        n_whisker=('outcome_w', 'count'),
        hits=('outcome_w', 'sum'),
        n_catch=('outcome_c', 'count'),
        false_alarms=('outcome_c', 'sum'),
        p_whisker=('outcome_w', 'mean'),
        p_auditory=('outcome_a', 'mean'),
        p_catch=('outcome_c', 'mean'),
    )
    out[['p_whisker', 'p_auditory', 'p_catch']] *= 100
    out['dprime'] = dprime(out['hits'], out['n_whisker'], out['false_alarms'], out['n_catch'])
    return out.reset_index()


def session_table(cohort):
    """One row per session, with the day labels of the cohort in order."""
    table, prefix, inactivation_type, days, labels = cohort
    trials = load_inactivation_table(table, prefix, inactivation_type, days)
    sessions = counts(trials, ['mouse_id', 'session_id', 'area', 'inactivation_day'])
    sessions['day_label'] = pd.Categorical(
        sessions['inactivation_day'].map(dict(zip(days, labels))), categories=labels, ordered=True
    )
    return sessions.sort_values(['mouse_id', 'day_label']), trials


def day1_tables():
    """Sessions and blocks of 20 trials of the wS1 optogenetics mice on Day +1
    and of the naive R+ imaging mice on Day 0."""
    opto = load_inactivation_table(*OPTO[:4])
    day1 = opto[(opto.area == 'wS1') & (opto.inactivation_day == 'recovery_1')]
    naive = pd.read_csv(
        paths.adjust_path_to_host(os.path.join(BEHAVIOR_DIR, 'behavior_imagingmice_table_5days_cut.csv'))
    )
    naive = naive[(naive.day == 0) & (naive.reward_group == 'R+')]
    trials = pd.concat([naive.assign(group=NAIVE), day1.assign(group=DAY1)], ignore_index=True)
    sessions = counts(trials, ['group', 'mouse_id', 'session_id'])
    blocks = counts(trials[trials.block_id < N_BLOCKS], ['group', 'mouse_id', 'session_id', 'block_id'])
    blocks['block'] = blocks['block_id'].astype(int) + 1
    return sessions, blocks


# ============================================================================
# Statistics
# ============================================================================


def mwu_row(label, measure, a, b, names):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = mannwhitneyu(a, b, alternative='two-sided')
    return {
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


def area_stats(sessions, measure, labels, cohort):
    rows = []
    for label in labels:
        d = sessions[sessions.day_label == label]
        rows.append(mwu_row(label, measure, *(d.loc[d.area == a, measure] for a in AREAS), AREAS))
    return [dict(r, cohort=cohort) for r in rows]


def day1_stats(sessions, blocks):
    rows = []
    for measure in MEASURES:
        for label, d in [('session', sessions), ('first block', blocks[blocks.block == 1])]:
            rows.append(
                mwu_row(label, measure, *(d.loc[d.group == g, measure] for g in (DAY1, NAIVE)), (DAY1, NAIVE))
            )
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================


def bars(ax, data, x, order, hue, hue_order, colors, y, pvalues, legend=False):
    """Bars (mean, bootstrapped 95% CI) and dots (mice), with the p-value of
    each x category above it. Bars are split by hue within each x category
    unless hue is x."""
    kw = dict(data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=ax)
    sns.barplot(**kw, palette=colors, errorbar=('ci', 95), seed=0, width=0.6, alpha=0.7, legend=legend)
    if hue == x:
        sns.swarmplot(data=data, x=x, y=y, order=order, color='black', size=2, ax=ax)
    else:
        sns.swarmplot(**kw, palette=dict.fromkeys(hue_order, 'black'), dodge=True, size=2, legend=False)
    for xi, p in enumerate(pvalues):
        ax.text(
            xi, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5
        )
    if y == 'dprime':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(*YLIMS[y])
    ax.set_xlabel('')
    ax.set_ylabel(MEASURES[y])
    if legend:
        ax.legend(frameon=False, title='', loc='upper left', bbox_to_anchor=(1.0, 1.0))


def area_bars(ax, sessions, stats, labels, y, title, legend=False):
    d = sessions[sessions.day_label.isin(labels)]
    pvalues = [stats.query('measure == @y and day == @label')['p_value'].iloc[0] for label in labels]
    bars(ax, d, 'day_label', labels, 'area', AREAS, AREA_COLORS, y, pvalues, legend)
    ax.set_title(title, pad=8)


def group_bars(ax, data, stats, y, label, title):
    """Naive Day 0 vs wS1 Day +1, one value per mouse."""
    order = [NAIVE, DAY1]
    p = stats.query('measure == @y and day == @label')['p_value'].iloc[0]
    bars(ax, data, 'group', order, 'group', order, GROUP_COLORS, y, [])
    ax.text(0.5, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5)
    ax.set_xticks([0, 1], ['Naive\nDay 0', 'wS1 light\nDay +1'])
    ax.set_title(title, pad=8)


def block_curves(ax, blocks, y):
    for group, color in GROUP_COLORS.items():
        g = blocks[blocks.group == group]
        if group == DAY1:
            for _, m in g.groupby('mouse_id'):
                ax.plot(m['block'], m[y], color=color, alpha=0.4, linewidth=0.5)
        sns.lineplot(
            data=g,
            x='block',
            y=y,
            color=color,
            errorbar=('ci', 95) if group == NAIVE else None,
            seed=0,
            ax=ax,
            label=f'{group}, n = {g.mouse_id.nunique()}',
        )
    if y == 'dprime':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(1, N_BLOCKS + 1))
    ax.set_xlabel('Block (20 trials)')
    ax.set_ylabel(MEASURES[y])
    ax.set_ylim(*YLIMS[y])
    ax.legend(frameon=False, loc='lower right')


def timecourse(ax, sessions, area, labels):
    """Lick probability of each trial type across days: lines = mice, dots =
    mean and bootstrapped 95% CI."""
    d = sessions[sessions.area == area].assign(x=lambda t: t.day_label.cat.codes)
    for col, color in OUTCOME_COLORS.items():
        for _, m in d.groupby('mouse_id'):
            ax.plot(m['x'], m[col], color=color, alpha=0.4, linewidth=0.5)
        sns.pointplot(
            data=d,
            x='x',
            y=col,
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


def save(fig, filename, output_dir=OUTPUT_DIR):
    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)


def plot_supp1(muscimol, opto, stats):
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), gridspec_kw={'width_ratios': [3, 2]})
    area_bars(axes[0], muscimol, stats[stats.cohort == 'muscimol'], MUSCIMOL[4], 'dprime', 'Muscimol')
    area_bars(
        axes[1],
        opto,
        stats[stats.cohort == 'optogenetics'],
        OPTO[4],
        'dprime',
        'Optogenetics on Day 0',
        legend=True,
    )
    save(fig, 'figure_2_supp1_dprime')


def plot_supp2(sessions, blocks, stats):
    fig, axes = plt.subplots(2, 3, figsize=panel_size(3, 2), gridspec_kw={'width_ratios': [2, 3, 2]})
    for row, y in zip(axes, MEASURES):
        group_bars(row[0], sessions, stats, y, 'session', 'Whole session')
        block_curves(row[1], blocks, y)
        group_bars(row[2], blocks[blocks.block == 1], stats, y, 'first block', 'First block')
    save(fig, 'figure_2_supp2_day1')


def plot_supp3(execution, stats):
    labels = EXECUTION[4]
    fig, axes = plt.subplots(1, 4, figsize=panel_size(4))
    for ax, area in zip(axes, AREAS):
        timecourse(ax, execution, area, labels)
    for ax, y in zip(axes[2:], MEASURES):
        area_bars(ax, execution, stats, labels[EXECUTION_TESTED], y, '', legend=ax is axes[-1])
    axes[-1].set_ylim(0, 4)
    axes[-1].set_yticks(range(5))
    save(fig, 'figure_2_supp3_execution')


# ============================================================================
# Main
# ============================================================================


def write(df, name):
    df.to_csv(os.path.join(OUTPUT_DIR, f'{name}.csv'), index=False)


if __name__ == '__main__':
    set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    columns = ['measure', 'day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']

    # Supplement 1: learning, d'.
    muscimol, _ = session_table(MUSCIMOL)
    opto, _ = session_table(OPTO)
    supp1_stats = pd.DataFrame(
        area_stats(muscimol, 'dprime', MUSCIMOL[4], 'muscimol')
        + area_stats(opto, 'dprime', OPTO[4], 'optogenetics')
    )
    print(supp1_stats[['cohort', *columns]].round(3).to_string())
    write(pd.concat([muscimol.assign(cohort='muscimol'), opto.assign(cohort='optogenetics')]), 'supp1_data')
    write(supp1_stats, 'supp1_stats')
    plot_supp1(muscimol, opto, supp1_stats)

    # Supplement 2: Day +1 after optogenetics vs naive Day 0.
    sessions, blocks = day1_tables()
    supp2_stats = day1_stats(sessions, blocks)
    print(supp2_stats[columns].round(3).to_string())
    write(sessions, 'supp2_sessions_data')
    write(blocks, 'supp2_blocks_data')
    write(supp2_stats, 'supp2_stats')
    plot_supp2(sessions, blocks, supp2_stats)

    # Supplement 3: execution.
    execution, _ = session_table(EXECUTION)
    tested = EXECUTION[4][EXECUTION_TESTED]
    supp3_stats = pd.DataFrame(sum((area_stats(execution, y, tested, 'execution') for y in MEASURES), []))
    print(supp3_stats[columns].round(3).to_string())
    write(execution, 'supp3_data')
    write(supp3_stats, 'supp3_stats')
    plot_supp3(execution, supp3_stats)
