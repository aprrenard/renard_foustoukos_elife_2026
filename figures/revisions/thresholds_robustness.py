"""
Revision: robustness of the reactivation results to the participation
threshold and to the detection threshold.

Participation threshold (2.5 x the cell's noise SD main; 5 and 10 x). A cell
participates in an event if its baseline-subtracted dF/F averaged over
+/- 150 ms reaches k times its noise SD, the robust SD (median absolute
deviation x 1.4826) of that 300-ms mean over the trials analysed
(fast_learning.participation). Noise panel: distribution of the per-cell
thresholds in dF/F0, with the fixed 10% threshold of the submitted version.
Results at each threshold: Fig. 4i (day-0 LMM slope vs LMI) and Fig. 4j
(per-mouse day slopes, LMI+ / LMI-), on the participation rate and the rate
above chance (Supp. 4), held-out cells (pipeline step 08).

Detection threshold (surrogate percentile 99 main; 99.5 and 99.9). The
detection threshold is the median over circular-shift surrogates of this
percentile of the template correlation (step 07). Results at each percentile:
Fig. 4h event rates (events of step 07 at that percentile; Mann-Whitney U per
day) and Fig. 4i-j on held-out cells, with each half's events re-detected at
that percentile (computed here when missing or with --recompute, cached in
the output folder; about 20 min per percentile on 35 cores). Participation
threshold 2.5 x noise SD. Mice: the participation mouse selection (at p99)
throughout.

Panels per condition:
    4i: LMM slope of day-0 participation vs LMI, 95% CI, per reward group.
    4j: per-mouse slopes of participation vs day, for R+ / R- x LMI+ / LMI-
        (dots: mice; bar: mean, bootstrapped 95% CI, seed 0; p: Wilcoxon
        signed-rank against zero, n = mice).

Outputs: <figures_dir>/revisions/thresholds_robustness/output/
    participation_threshold.pdf, participation_noise.pdf,
    detection_threshold.pdf, stats.csv (all conditions), mouse_slopes.csv,
    heldout_events_p995.pkl, heldout_events_p999.pkl (cache).
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

from fast_learning import paths, participation as pt, reactivations as rx
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p, lmm_slope, per_mouse_slope_test

THRESHOLDS = pt.PARTICIPATION_THRESHOLDS  # noise SD units: 2.5 main, 5, 10
MAIN = pt.PARTICIPATION_THRESHOLD
FIXED_SUBMITTED = 0.10  # fixed dF/F0 threshold of the submitted version (noise panel)
PERCENTILES = [99, 99.5, 99.9]  # 99 main
N_JOBS = 35
GROUPS = [(rg, c) for rg in ['R+', 'R-'] for c in ['positive', 'negative', 'neutral']]
GROUP_LABELS = [f'{rg}\n{c}' for rg in ['R+', 'R−'] for c in ['LMI+', 'LMI−', 'n.m.']]
LMI_COLORS = {'positive': '#d62728', 'negative': '#1f77b4', 'neutral': '#a0a0a0'}
RG_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
MEASURES = {'rate': 'Participation rate', 'excess': 'Participation above chance'}
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'thresholds_robustness', 'output')


# ============================================================================
# Data
# ============================================================================


def heldout_tables(percentile, recompute=False):
    """{threshold: (merged, per_day)} of held-out cells at a detection percentile.

    p99 is read from step 08; other percentiles are computed here (main threshold)."""
    if percentile == pt.DETECTION_PERCENTILE:
        return {thr: pt.load_participation(thr) for thr in THRESHOLDS}

    cache = os.path.join(OUTPUT_DIR, f'heldout_events_{rx.percentile_tag(percentile)}.pkl')
    mice = sorted(rx.load_participation_mice())
    if recompute or not os.path.exists(cache):
        print(f'Held-out detection at p{percentile:g} for {len(mice)} mice ...')
        detections = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(pt.detect_heldout_events)(m, None, percentile) for m in mice
        )
        with open(cache, 'wb') as f:
            pickle.dump({d['mouse']: d for d in detections}, f)
    with open(cache, 'rb') as f:
        detections = pickle.load(f)

    def one(det):
        rois, days = pt.load_heldout_data(det['mouse'])
        sub_by_day = {d: v['sub'] for d, v in days.items()}
        noise = pt.noise_sd(sub_by_day)
        thresholds = {MAIN: pt.cell_thresholds(noise, [MAIN])[MAIN]}
        cells = pd.DataFrame(dict(mouse_id=det['mouse'], roi=rois[noise > 0]))
        rates = pt.heldout_split_rates(det, days, thresholds).merge(cells)
        return rates, pt.chance_rates(det['mouse'], rois, sub_by_day, thresholds)

    out = Parallel(n_jobs=N_JOBS)(delayed(one)(d) for d in detections.values())
    split_df = pd.concat([s for s, _ in out if s is not None], ignore_index=True)
    chance_df = pd.concat([c for _, c in out if c is not None], ignore_index=True)
    sel = split_df[split_df['role'] == 'heldout']
    per_day = pt.add_excess(pt.average_over_splits(sel), chance_df, MAIN)
    groups = _reward_groups()
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day), lmi_df, groups)
    return {MAIN: (merged, per_day)}


def _reward_groups():
    with open(os.path.join(rx.RESULTS_DIR, 'reactivation_results_p99.pkl'), 'rb') as f:
        res = pickle.load(f)
    return {m: 'R+' for m in res['r_plus_results']} | {m: 'R-' for m in res['r_minus_results']}


def with_measure(merged, per_day, measure):
    """(merged, per_day) with participation_rate = the rate or the rate above chance."""
    if measure == 'rate':
        return merged, per_day
    per_day = per_day.assign(participation_rate=per_day['excess_rate'])
    groups = merged.groupby('mouse_id')['reward_group'].first().to_dict()
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    return pt.merge_with_lmi(pt.aggregate_across_days(per_day), lmi_df, groups), per_day


def event_rates(percentile):
    """Per mouse x day reactivation rate (events/min) of step 07 at a percentile (Fig. 4h)."""
    with open(
        os.path.join(rx.RESULTS_DIR, f'reactivation_results_{rx.percentile_tag(percentile)}.pkl'), 'rb'
    ) as f:
        res = pickle.load(f)
    rows = [
        dict(mouse_id=m, reward_group=rg, day=day, rate=d['event_frequency'])
        for rg, key in [('R+', 'r_plus_results'), ('R-', 'r_minus_results')]
        for m, r in res[key].items()
        for day, d in r['days'].items()
    ]
    return pd.DataFrame(rows)


# ============================================================================
# Statistics
# ============================================================================


def condition_stats(merged, per_day, labels):
    """4i LMM slopes and 4j per-mouse slope tests for one condition."""
    rows, slopes = [], []
    day0 = merged.dropna(subset=['lmi', 'learning_rate'])
    day0 = day0[day0['reliable_learning']]
    for rg in ['R+', 'R-']:
        fit = lmm_slope(day0[day0['reward_group'] == rg], 'learning_rate', 'lmi')
        rows.append(
            dict(
                **labels,
                panel='4i',
                reward_group=rg,
                lmi_category='',
                estimate=fit['slope'],
                ci_low=fit['ci_low'],
                ci_high=fit['ci_high'],
                p_value=fit['p_value'],
                n=f"{(day0['reward_group'] == rg).sum()} cells",
            )
        )
    cells = merged.loc[
        merged['lmi_category'].isin(['positive', 'negative', 'neutral']),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    d = per_day.merge(cells, on=['mouse_id', 'roi'])
    d = d[d['reliable']]
    mouse_day = (
        d.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'])['participation_rate']
        .mean()
        .reset_index()
    )
    for rg, cat in GROUPS:
        test = per_mouse_slope_test(
            mouse_day[(mouse_day['reward_group'] == rg) & (mouse_day['lmi_category'] == cat)]
        )
        rows.append(
            dict(
                **labels,
                panel='4j',
                reward_group=rg,
                lmi_category=cat,
                estimate=test['median_slope'],
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=test['p_value'],
                n=f"{test['n_mice']} mice",
            )
        )
        slopes += [
            dict(**labels, reward_group=rg, lmi_category=cat, mouse_id=m, slope=s)
            for m, s in test['slopes'].items()
        ]
    return rows, slopes


# ============================================================================
# Figures
# ============================================================================


def plot_lmm(ax, stats, conditions, cond_key, cond_labels):
    """4i: LMM slope with 95% CI per condition and reward group."""
    s = stats[stats['panel'] == '4i']
    for gi, rg in enumerate(['R+', 'R-']):
        for ci, c in enumerate(conditions):
            r = s[(s[cond_key] == c) & (s['reward_group'] == rg)].iloc[0]
            x = ci + (gi - 0.5) * 0.3
            ax.errorbar(
                x,
                r['estimate'],
                yerr=[[r['estimate'] - r['ci_low']], [r['ci_high'] - r['estimate']]],
                fmt='o',
                color=RG_COLORS[rg],
                markersize=3,
                capsize=0,
                linewidth=1,
            )
            ax.text(
                x, r['ci_high'], format_p(r['p_value']), ha='center', va='bottom', rotation=90, fontsize=5
            )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(len(conditions)), cond_labels)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylabel('4i: LMM slope vs LMI (day 0)')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.6 * (top - bottom))  # room for the p-values
    for gi, rg in enumerate(['R+', 'R-']):
        ax.text(0.02 + 0.2 * gi, 0.02, rg, color=RG_COLORS[rg], transform=ax.transAxes)


def plot_slopes(ax, slopes, stats, title):
    """4j: per-mouse day slopes, four groups."""
    slopes = slopes.assign(
        group=[GROUPS.index((rg, c)) for rg, c in zip(slopes['reward_group'], slopes['lmi_category'])]
    )
    palette = [LMI_COLORS[c] for _, c in GROUPS]
    sns.barplot(
        data=slopes,
        x='group',
        y='slope',
        order=range(len(GROUPS)),
        hue='group',
        palette=palette,
        legend=False,
        errorbar=('ci', 95),
        seed=0,
        alpha=0.5,
        edgecolor='black',
        linewidth=0.5,
        err_kws={'linewidth': 1},
        ax=ax,
    )
    sns.stripplot(
        data=slopes,
        x='group',
        y='slope',
        order=range(len(GROUPS)),
        color='black',
        size=1.5,
        jitter=0.2,
        ax=ax,
    )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    top = slopes['slope'].max()
    for i, (rg, c) in enumerate(GROUPS):
        p = stats[(stats['panel'] == '4j') & (stats['reward_group'] == rg) & (stats['lmi_category'] == c)][
            'p_value'
        ].iloc[0]
        ax.text(i, top * 1.05, format_p(p), ha='center', va='bottom', rotation=90, fontsize=5)
    ax.set_ylim(top=top * 1.6)
    ax.set_xticks(range(len(GROUPS)), GROUP_LABELS)
    ax.set_xlabel('')
    ax.set_ylabel('4j: day slope (per mouse)')
    ax.set_title(title)


def figure_participation_threshold(stats, slopes):
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=panel_size(4, 2, w=1.3))
    for row, measure in enumerate(MEASURES):
        s = stats[(stats['measure'] == measure) & (stats['percentile'] == 99)]
        plot_lmm(axes[row, 0], s, THRESHOLDS, 'participation_threshold', [f'{t:g} SD' for t in THRESHOLDS])
        axes[row, 0].set_title(MEASURES[measure])
        for col, thr in enumerate(THRESHOLDS, 1):
            sel = lambda df: df[
                (df['measure'] == measure) & (df['percentile'] == 99) & (df['participation_threshold'] == thr)
            ]  # noqa: E731
            plot_slopes(axes[row, col], sel(slopes), sel(stats), f'{MEASURES[measure]}\nthreshold {thr:g} SD')
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'participation_threshold.pdf'))
    plt.close(fig)


def figure_noise(noise):
    """Per-cell participation thresholds (k x noise SD, in dF/F0) and the fixed 10%."""
    set_style()
    fig, ax = plt.subplots(figsize=panel_size(1))
    bins = np.logspace(np.log10(0.02), np.log10(3), 60)
    for k, color in zip(THRESHOLDS, ['black', '0.45', '0.7']):
        ax.hist(k * noise['noise_sd'], bins=bins, histtype='step', color=color, label=f'{k:g} SD')
    ax.axvline(FIXED_SUBMITTED, color='#d62728', linestyle='--', linewidth=0.8)
    ax.text(
        FIXED_SUBMITTED,
        1.0,
        'fixed 10%',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='bottom',
        color='#d62728',
    )
    ax.set_xscale('log')
    ax.set_xlabel('Participation threshold (dF/F0)')
    ax.set_ylabel('Cells')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9))
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'participation_noise.pdf'))
    plt.close(fig)


def figure_detection_threshold(stats, slopes, rates):
    set_style()
    fig, axes = plt.subplots(3, 4, figsize=panel_size(4, 3, w=1.3))
    for col, perc in enumerate(PERCENTILES, 1):
        ax = axes[0, col]
        r = rates[rates['percentile'] == perc]
        sns.pointplot(
            data=r,
            x='day',
            y='rate',
            hue='reward_group',
            hue_order=['R+', 'R-'],
            palette=RG_COLORS,
            errorbar=('ci', 95),
            seed=0,
            markersize=2,
            linewidth=1,
            dodge=0.2,
            legend=False,
            ax=ax,
        )
        top = ax.get_ylim()[1]
        for i, day in enumerate(sorted(r['day'].unique())):
            a = r[(r['day'] == day) & (r['reward_group'] == 'R+')]['rate']
            b = r[(r['day'] == day) & (r['reward_group'] == 'R-')]['rate']
            ax.text(
                i, top, format_p(mannwhitneyu(a, b).pvalue), ha='center', va='bottom', rotation=90, fontsize=5
            )
        ax.set_ylim(0, top * 1.5)
        ax.set_ylabel('4h: reactivations (events/min)')
        ax.set_xlabel('Day')
        ax.set_title(f'Detection p{perc:g}')
    axes[0, 0].axis('off')
    for row, measure in enumerate(MEASURES, 1):
        s = stats[(stats['measure'] == measure) & (stats['participation_threshold'] == MAIN)]
        plot_lmm(axes[row, 0], s, PERCENTILES, 'percentile', [f'p{p:g}' for p in PERCENTILES])
        axes[row, 0].set_title(MEASURES[measure])
        for col, perc in enumerate(PERCENTILES, 1):
            sel = lambda df: df[
                (df['measure'] == measure)
                & (df['participation_threshold'] == MAIN)
                & (df['percentile'] == perc)
            ]  # noqa: E731
            plot_slopes(axes[row, col], sel(slopes), sel(stats), f'{MEASURES[measure]}\ndetection p{perc:g}')
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'detection_threshold.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--recompute', action='store_true', help='rerun the held-out detections at p99.5 / p99.9'
    )
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stat_rows, slope_rows = [], []
    for perc in PERCENTILES:
        for thr, (merged, per_day) in heldout_tables(perc, args.recompute).items():
            for measure in MEASURES:
                labels = dict(percentile=perc, participation_threshold=thr, measure=measure)
                r, s = condition_stats(*with_measure(merged, per_day, measure), labels)
                stat_rows += r
                slope_rows += s
    stats, slopes = pd.DataFrame(stat_rows), pd.DataFrame(slope_rows)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'stats.csv'), index=False)
    slopes.to_csv(os.path.join(OUTPUT_DIR, 'mouse_slopes.csv'), index=False)

    rates = pd.concat([event_rates(p).assign(percentile=p) for p in PERCENTILES], ignore_index=True)
    noise = pt.load_participation(MAIN)[1].drop_duplicates(['mouse_id', 'roi'])[
        ['mouse_id', 'roi', 'noise_sd']
    ]
    q = noise['noise_sd'].quantile([0.25, 0.5, 0.75])
    print(f'Noise SD of the 300-ms mean: median {q[0.5]:.3f}, IQR {q[0.25]:.3f}-{q[0.75]:.3f} dF/F0')
    ratio = FIXED_SUBMITTED / noise['noise_sd']
    print(
        f'Fixed 10% = {ratio.median():.1f} noise SD (median; IQR {ratio.quantile(0.25):.1f}-{ratio.quantile(0.75):.1f})'
    )

    figure_participation_threshold(stats, slopes)
    figure_noise(noise)
    figure_detection_threshold(stats, slopes, rates)
    pd.set_option('display.width', 200)
    print(stats.drop(columns=['ci_low', 'ci_high']).to_string(index=False))
    print(f'Saved to {OUTPUT_DIR}')
