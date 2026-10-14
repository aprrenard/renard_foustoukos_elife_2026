"""
Revision: adaptation within the passive block, and the learning-related
response changes in early vs late passive trials (R3 public 3).

The 50 passive whisker trials are delivered at the end of each session. If
responses adapted within the block, or if the mice's state changed along it,
the early and late trials could give different results. Response of a trial:
dF/F (baseline-subtracted tensors) averaged over 0-300 ms after the stimulus
and over all cells of the mouse. Only the last N_TRIALS passive trials of each
day are used (all of them if fewer), as in the population analyses (templates,
decoders, similarity), so first / last N_SPLIT are trials 11-30 / 31-50 of a
50-trial block. Mice: imaging mice, as Fig. 3c.

a: response across the block, in bins of BIN trials, per day; mean across
   mice with bootstrapped 95% CI.
b: first vs last N_SPLIT trials of the block, each mouse averaged over its
   five days (adaptation within the block); Wilcoxon signed-rank (n = mice).
c, d: pre (days -2, -1) vs post (days +1, +2) learning, using only the first
   (c) or only the last (d) N_SPLIT trials of each day; Wilcoxon signed-rank
   per group (n = mice).
e: per cell, the post - pre response change computed from the first and from
   the last N_SPLIT trials; Spearman correlation across cells in each mouse,
   Wilcoxon test of the per-mouse correlations against zero.

Outputs: <figures_dir>/revisions/figure_3_passive_adaptation/output/
    passive_adaptation.pdf, passive_adaptation_bins.csv (mouse x day x bin),
    passive_adaptation_data.csv (mouse x day x part),
    passive_adaptation_cell_change_rho.csv (mouse), passive_adaptation_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

from fast_learning import database, imaging, paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p

DAYS = [-2, -1, 0, 1, 2]
PRE, POST = [-2, -1], [1, 2]
WIN = (0, 0.300)
N_TRIALS = 40  # last trials of the block, as the population analyses
BIN = 10  # trials per bin along the block
N_SPLIT = 20  # first / last trials of the block
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
DAY_PALETTE = sns.color_palette('viridis', len(DAYS))
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_3_passive_adaptation', 'output')


# ============================================================================
# Data
# ============================================================================


def mouse_responses(mouse):
    """Per day: responses (cells x trials, block order) averaged over WIN,
    for the last N_TRIALS trials of the block."""
    x = imaging.load_mouse_xarray(mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc')
    x = imaging.select_time(x, *WIN).mean(dim='time')
    return {d: x.sel(trial=x['day'] == d).values[:, -N_TRIALS:] for d in DAYS if (x['day'] == d).sum() > 0}


def summarize(mouse, rg, resp):
    bins, parts, cells = [], [], {}
    for day, r in resp.items():
        trial_mean = np.nanmean(r, axis=0)  # population response per trial
        for b in range(int(np.ceil(len(trial_mean) / BIN))):
            bins.append(
                dict(
                    mouse_id=mouse,
                    reward_group=rg,
                    day=day,
                    bin=b + 1,
                    response=np.nanmean(trial_mean[b * BIN : (b + 1) * BIN]),
                )
            )
        for part, sl in [('first', slice(0, N_SPLIT)), ('last', slice(-N_SPLIT, None))]:
            parts.append(
                dict(mouse_id=mouse, reward_group=rg, day=day, part=part, response=np.nanmean(trial_mean[sl]))
            )
            cells[(day, part)] = np.nanmean(r[:, sl], axis=1)
    # Per-cell post - pre change from the first and from the last trials.
    rho = np.nan
    if all((d, p) in cells for d in PRE + POST for p in ('first', 'last')):
        change = {
            p: np.mean([cells[(d, p)] for d in POST], axis=0) - np.mean([cells[(d, p)] for d in PRE], axis=0)
            for p in ('first', 'last')
        }
        ok = np.isfinite(change['first']) & np.isfinite(change['last'])
        rho = spearmanr(change['first'][ok], change['last'][ok])[0]
    return bins, parts, dict(mouse_id=mouse, reward_group=rg, rho_change_first_last=rho)


def build():
    _, _, mice, _ = database.select_sessions_from_db(
        paths.db_path, paths.nwb_dir, two_p_imaging='yes', experimenters=['AR', 'GF', 'MI']
    )
    groups = {m: database.get_mouse_reward_group_from_db(paths.db_path, m) for m in mice}
    resp = Parallel(n_jobs=20)(delayed(mouse_responses)(m) for m in mice)
    bins, parts, rhos = [], [], []
    for m, r in zip(mice, resp):
        b, p, c = summarize(m, groups[m], r)
        bins += b
        parts += p
        rhos.append(c)
    return pd.DataFrame(bins), pd.DataFrame(parts), pd.DataFrame(rhos)


# ============================================================================
# Statistics
# ============================================================================


def paired_row(test_name, rg, a, b, labels):
    res = wilcoxon(a, b)
    return dict(
        comparison=test_name,
        reward_group=rg,
        test='Wilcoxon signed-rank (n = mice)',
        n_mice=len(a),
        **{f'mean_{labels[0]}': np.mean(a), f'mean_{labels[1]}': np.mean(b)},
        n_mice_higher_first=int((np.asarray(a) > np.asarray(b)).sum()),
        statistic=res.statistic,
        p_value=res.pvalue,
    )


def compute_stats(parts, rhos):
    rows = []
    for rg in GROUP_COLORS:
        g = parts[parts['reward_group'] == rg]
        block = g.groupby(['mouse_id', 'part'])['response'].mean().unstack().dropna()
        rows.append(
            paired_row(
                'adaptation: first vs last trials (all days)', rg, block['first'], block['last'], ('a', 'b')
            )
        )
        for part in ('first', 'last'):
            gp = g[g['part'] == part]
            period = np.where(gp['day'].isin(PRE), 'pre', np.where(gp['day'].isin(POST), 'post', None))
            pp = gp.assign(period=period).dropna(subset=['period'])
            pp = pp.groupby(['mouse_id', 'period'])['response'].mean().unstack().dropna()
            rows.append(
                paired_row(f'pre vs post, {part} {N_SPLIT} trials', rg, pp['pre'], pp['post'], ('a', 'b'))
            )
        r = rhos.loc[rhos['reward_group'] == rg, 'rho_change_first_last'].dropna()
        res = wilcoxon(r)
        rows.append(
            dict(
                comparison='per-cell post-pre change, first vs last trials: Spearman per mouse',
                reward_group=rg,
                test='Wilcoxon signed-rank of per-mouse rho against 0',
                n_mice=len(r),
                mean_a=np.median(r),
                n_mice_higher_first=int((r > 0).sum()),
                statistic=res.statistic,
                p_value=res.pvalue,
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================


def paired_bars(ax, d, cols, labels, stats, comparison):
    for gi, rg in enumerate(GROUP_COLORS):
        g = d.xs(rg, level='reward_group')
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in g.iterrows():
            ax.plot(x, 100 * row[list(cols)].astype(float), '-', color='grey', linewidth=0.5, alpha=0.6)
        long = g.reset_index().melt(id_vars='mouse_id', value_vars=list(cols), var_name='c', value_name='v')
        sns.barplot(
            data=long.assign(x=np.where(long['c'] == cols[0], x[0], x[1]), v=100 * long['v']),
            x='x',
            y='v',
            order=list(range(5)),
            native_scale=True,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            edgecolor='black',
            errorbar=('ci', 95),
            seed=0,
            width=0.7,
            ax=ax,
        )
        p = stats.query('reward_group == @rg and comparison == @comparison')['p_value'].iloc[0]
        ax.text(
            x.mean(),
            1.0,
            format_p(p),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=5,
        )
        ax.text(
            x.mean(),
            1.08,
            rg,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            color=GROUP_COLORS[rg],
        )
    ax.set_xticks([0, 1, 3, 4], list(labels) * 2)
    ax.set_xlim(-0.7, 4.7)
    ax.set_xlabel('')
    ax.set_ylabel('Response (% ΔF/F0)')


def rho_panel(ax, rhos, stats):
    """Per mouse: Spearman correlation across cells of the post - pre change
    computed from the first and from the last N_SPLIT trials."""
    kw = dict(data=rhos, x='reward_group', y='rho_change_first_last', order=list(GROUP_COLORS), ax=ax)
    sns.barplot(
        **kw,
        hue='reward_group',
        palette=GROUP_COLORS,
        alpha=0.7,
        edgecolor='black',
        errorbar=('ci', 95),
        seed=0,
        width=0.6,
        legend=False,
    )
    sns.swarmplot(**kw, color='black', size=2)
    for gi, rg in enumerate(GROUP_COLORS):
        p = stats.query('reward_group == @rg and comparison.str.startswith("per-cell")')['p_value'].iloc[0]
        ax.text(
            gi, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5
        )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(-0.2, 1)
    ax.set_xlabel('')
    ax.set_ylabel(f'Cell response change (post − pre):\nfirst vs last {N_SPLIT} trials (Spearman r)')


def plot(bins, parts, rhos, stats, filename='passive_adaptation'):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=panel_size(3, 2))
    for ax, rg in zip(axes[0, :2], GROUP_COLORS):
        d = bins[bins['reward_group'] == rg].assign(response=lambda t: 100 * t['response'])
        sns.lineplot(
            data=d,
            x='bin',
            y='response',
            hue='day',
            palette=DAY_PALETTE,
            hue_order=DAYS,
            errorbar=('ci', 95),
            seed=0,
            ax=ax,
            legend=rg == 'R-',
            err_kws=dict(alpha=0.15),
        )
        ax.set_title(rg, color=GROUP_COLORS[rg])
        n_bins = N_TRIALS // BIN
        ax.set_xticks(range(1, n_bins + 1), [f'{(b - 1) * BIN + 1}–{b * BIN}' for b in range(1, n_bins + 1)])
        ax.set_xlabel(f'Passive trial (last {N_TRIALS})')
        ax.set_ylabel('Response (% ΔF/F0)')
        ax.set_ylim(0, 8)
    axes[0, 1].legend(frameon=False, title='Day', loc='upper right', ncol=2)
    block = parts.groupby(['reward_group', 'mouse_id', 'part'])['response'].mean().unstack()
    paired_bars(
        axes[0, 2],
        block,
        ('first', 'last'),
        ('First', 'Last'),
        stats,
        'adaptation: first vs last trials (all days)',
    )
    axes[0, 2].set_title(f'First vs last {N_SPLIT} trials, all days', pad=16)
    for ax, part in zip(axes[1, :2], ('first', 'last')):
        gp = parts[parts['part'] == part]
        period = np.where(gp['day'].isin(PRE), 'pre', np.where(gp['day'].isin(POST), 'post', None))
        pp = gp.assign(period=period).dropna(subset=['period'])
        pp = pp.groupby(['reward_group', 'mouse_id', 'period'])['response'].mean().unstack()
        paired_bars(ax, pp, ('pre', 'post'), ('Pre', 'Post'), stats, f'pre vs post, {part} {N_SPLIT} trials')
        ax.set_title(f'{part.capitalize()} {N_SPLIT} trials', pad=16)
    for ax in [axes[0, 2], *axes[1, :2]]:
        ax.set_ylim(0, 12)
    rho_panel(axes[1, 2], rhos, stats)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bins, parts, rhos = build()
    stats = compute_stats(parts, rhos)
    bins.to_csv(os.path.join(OUTPUT_DIR, 'passive_adaptation_bins.csv'), index=False)
    parts.to_csv(os.path.join(OUTPUT_DIR, 'passive_adaptation_data.csv'), index=False)
    rhos.to_csv(os.path.join(OUTPUT_DIR, 'passive_adaptation_cell_change_rho.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'passive_adaptation_stats.csv'), index=False)
    plot(bins, parts, rhos, stats)
    print(stats.drop(columns=['test']).round(4).to_string(index=False))
