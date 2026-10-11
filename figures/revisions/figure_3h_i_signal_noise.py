"""
Figure 3h-i revision: what drives the change in within-day similarity?

Signal and noise power of each day's responses, pre vs post learning, and
day-to-day drift of the response pattern: noise-corrected correlation of the
mean response vectors of consecutive days. The correlation of the two days'
mean vectors is divided by the geometric mean of their split-half
reliabilities (Spearman-Brown corrected), so that 1 means the same pattern
on both days. Days -2 and -1 are the only consecutive days without
learning (mice are auditory experts; performance keeps improving after
day 0), so -2 -> -1 measures drift. Day 0 responses are recorded after the
learning session, so -1 -> 0 spans learning. Wilcoxon signed-rank test of
-1 -> 0 against -2 -> -1 within mice, and Mann-Whitney U test of drift
(-2 -> -1) between groups, n = mice.

Outputs: <figures_dir>/revisions/figure_3h_i_signal_noise/output/
    signal_noise.pdf, signal_noise_data.csv (mouse x day),
    signal_noise_stats.csv, drift_data.csv (mouse x day pair).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon

from fast_learning import database, imaging, paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.similarity import (
    DAYS,
    N_MAP_TRIALS,
    WIN,
    compute_similarity_matrix,
    compute_within_day_metrics,
)
from fast_learning.stats import format_p, significance_stars

PRE_DAYS = [-2, -1]
POST_DAYS = [1, 2]
BASELINE_PAIR = (-2, -1)  # the only consecutive days without learning
LEARNING_PAIR = (-1, 0)  # spans the first whisker learning session
DAY_PAIRS = [BASELINE_PAIR, LEARNING_PAIR]
N_SPLITS = 100
SEED = 0
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_3h_i_signal_noise', 'output')


# ============================================================================
# Data: the response vectors of Fig. 3h-i
# ============================================================================


def load_response_vectors():
    """{mouse: (reward_group, cells x trials array, day of each trial)},
    with the selection of figure_3h_j.load_and_process_data (all cells,
    last N_MAP_TRIALS mapping trials per day, mean over WIN)."""
    _, _, mice, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    out = {}
    for mouse in mice:
        xarr = imaging.load_mouse_xarray(mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc')
        xarr = xarr.sel(trial=xarr['day'].isin(DAYS))
        n_trials = xarr[0, :, 0].groupby('day').count(dim='trial').values
        if np.any(n_trials < N_MAP_TRIALS):
            print(f'Not enough mapping trials for {mouse}, skipped.')
            continue
        d = xarr.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))
        d = imaging.select_time(d, WIN[0], WIN[1]).mean(dim='time')
        out[mouse] = (database.get_mouse_reward_group_from_db(paths.db_path, mouse, db), d)
    return out


# ============================================================================
# Signal / noise decomposition
# ============================================================================


def signal_noise(trials):
    """(signal power, noise power) of a cells x trials array."""
    n = trials.shape[1]
    noise = trials.var(axis=1, ddof=1).sum()
    mu = trials.mean(axis=1)
    return mu @ mu - noise / n, noise


def split_half_reliability(trials, rng, n_splits=N_SPLITS):
    """Spearman-Brown corrected split-half reliability of the mean pattern."""
    n, h = trials.shape[1], trials.shape[1] // 2
    r = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        r.append(np.corrcoef(trials[:, idx[:h]].mean(1), trials[:, idx[h:]].mean(1))[0, 1])
    r = np.mean(r)
    return 2 * r / (1 + r)


def pair_label(a, b):
    return f'{a:+d} → {b:+d}'


def compute_metrics(vectors):
    day_rows, drift_rows = [], []
    rng = np.random.default_rng(SEED)
    for mouse, (rg, d) in vectors.items():
        X, day_of = d.values, d['day'].values
        cosine = compute_within_day_metrics([compute_similarity_matrix(d, 'cosine')], [mouse], rg).iloc[0]
        for day in DAYS:
            s, n = signal_noise(X[:, day_of == day])
            day_rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': rg,
                    'day': day,
                    'signal_power': s,
                    'noise_power': n,
                    'snr': s / n,
                    'cosine_observed': cosine[f'within_day{day:+d}'],
                    'cosine_predicted': s / (s + n),
                }
            )
        mean = {day: X[:, day_of == day].mean(1) for day in DAYS}
        rel = {day: split_half_reliability(X[:, day_of == day], rng) for day in DAYS}
        for a, b in DAY_PAIRS:
            r = pearsonr(mean[a], mean[b])[0]
            drift_rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': rg,
                    'pair': pair_label(a, b),
                    'r': r,
                    'reliability_a': rel[a],
                    'reliability_b': rel[b],
                    'r_corrected': r / np.sqrt(rel[a] * rel[b]),
                }
            )
    return pd.DataFrame(day_rows), pd.DataFrame(drift_rows)


# ============================================================================
# Statistics
# ============================================================================


def pre_post_means(day_df, metrics):
    period = day_df['day'].map({d: 'pre' for d in PRE_DAYS} | {d: 'post' for d in POST_DAYS})
    return (
        day_df.assign(period=period)
        .dropna(subset=['period'])
        .groupby(['reward_group', 'mouse_id', 'period'])[metrics]
        .mean()
    )


def compute_stats(day_df, drift_df):
    metrics = ['signal_power', 'noise_power', 'snr', 'cosine_observed', 'cosine_predicted']
    pp = pre_post_means(day_df, metrics).unstack('period')
    rows = []
    for rg in ['R+', 'R-']:
        g = pp.loc[rg]
        for m in metrics:
            stat, p = wilcoxon(g[(m, 'pre')], g[(m, 'post')])
            rows.append(
                {
                    'reward_group': rg,
                    'measure': m,
                    'test': 'Wilcoxon signed-rank, pre (days -2, -1) vs post (days +1, +2)',
                    'n_mice': len(g),
                    'median_pre': g[(m, 'pre')].median(),
                    'median_post': g[(m, 'post')].median(),
                    'statistic': stat,
                    'p_value': p,
                    'significance': significance_stars(p),
                }
            )
        g = drift_df[drift_df['reward_group'] == rg].pivot(
            index='mouse_id', columns='pair', values='r_corrected'
        )
        baseline, learning = g[pair_label(*BASELINE_PAIR)], g[pair_label(*LEARNING_PAIR)]
        stat, p = wilcoxon(learning - baseline)
        rows.append(
            {
                'reward_group': rg,
                'measure': f'drift {pair_label(*LEARNING_PAIR)}',
                'test': f'Wilcoxon signed-rank, noise-corrected correlation {pair_label(*LEARNING_PAIR)} vs {pair_label(*BASELINE_PAIR)}',
                'n_mice': len(g),
                'median_pre': baseline.median(),
                'median_post': learning.median(),
                'statistic': stat,
                'p_value': p,
                'significance': significance_stars(p),
            }
        )
    g = {
        rg: drift_df[(drift_df['reward_group'] == rg) & (drift_df['pair'] == pair_label(*BASELINE_PAIR))][
            'r_corrected'
        ]
        for rg in ['R+', 'R-']
    }
    stat, p = mannwhitneyu(g['R+'], g['R-'])
    rows.append(
        {
            'reward_group': 'R+ vs R-',
            'measure': f'drift {pair_label(*BASELINE_PAIR)}',
            'test': 'Mann-Whitney U, noise-corrected correlation (median_pre: R+, median_post: R-)',
            'n_mice': len(g['R+']) + len(g['R-']),
            'median_pre': g['R+'].median(),
            'median_post': g['R-'].median(),
            'statistic': stat,
            'p_value': p,
            'significance': significance_stars(p),
        }
    )
    r, _ = pearsonr(day_df['cosine_observed'], day_df['cosine_predicted'])
    rows.append(
        {
            'reward_group': 'all',
            'measure': 'observed vs predicted within-day cosine',
            'test': 'Pearson r across mouse-days (descriptive)',
            'n_mice': day_df['mouse_id'].nunique(),
            'median_pre': np.nan,
            'median_post': r,
            'statistic': r,
            'p_value': np.nan,
            'significance': '',
        }
    )
    return pd.DataFrame(rows), pp


# ============================================================================
# Figure
# ============================================================================


def paired_panel(ax, pp, metric, ylabel, stats, ylim):
    for gi, rg in enumerate(['R+', 'R-']):
        g = pp.loc[rg][metric]
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in g.iterrows():
            ax.plot(x, [row['pre'], row['post']], '-', color='grey', linewidth=0.5, alpha=0.6)
        # Bars: mean across mice with bootstrapped 95% CI (seeded); lines: mice.
        long = g.reset_index().melt(id_vars='mouse_id', value_vars=['pre', 'post'], value_name='v')
        sns.barplot(
            data=long.assign(x=np.where(long['period'] == 'pre', x[0], x[1])),
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
            zorder=1,
        )
        p = stats.query('reward_group == @rg and measure == @metric')['p_value'].iloc[0]
        top = np.nanmax(g.values)
        ax.plot(
            [x[0], x[0], x[1], x[1]], [top * 1.04, top * 1.08, top * 1.08, top * 1.04], 'k-', linewidth=0.8
        )
        ax.text(x.mean(), top * 1.09, format_p(p), ha='center', va='bottom')
        ax.text(x.mean(), -0.2, rg, transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.set_xticks([0, 1, 3, 4], ['Pre', 'Post', 'Pre', 'Post'])
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")


def drift_panel(ax, drift_df, stats):
    """Noise-corrected correlation of consecutive days, mean and 95% CI across mice."""
    order = [pair_label(*p) for p in DAY_PAIRS]
    sns.pointplot(
        data=drift_df,
        x='pair',
        y='r_corrected',
        order=order,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        dodge=0.3,
        markersize=3,
        linewidth=1,
        ax=ax,
    )
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.6)
    # p-values above each pair: -1/0 vs -2/-1 within group (colored), R+ vs R- (black).
    for gi, rg in enumerate(['R+', 'R-', 'R+ vs R-']):
        for xi, pair in enumerate(order):
            row = stats[(stats['reward_group'] == rg) & (stats['measure'] == f'drift {pair}')]
            if len(row):
                y = 1.04 + 0.06 * gi
                ax.text(
                    xi,
                    y,
                    ('' if rg in GROUP_COLORS else 'R+ vs R−: ') + format_p(row['p_value'].iloc[0]),
                    ha='center',
                    va='bottom',
                    color=GROUP_COLORS.get(rg, 'black'),
                    fontsize=5,
                )
    ax.set_ylim(0.4, 1.2)
    ax.set_xticks(range(len(order)), [f'{a:+d}/{b:+d}'.replace('+0', '0') for a, b in DAY_PAIRS])
    ax.set_xlabel('Day pair')
    ax.set_ylabel('Day-to-day pattern correlation\n(noise-corrected)')
    ax.legend(frameon=False, loc='lower left')


def plot(day_df, drift_df, pp, stats, output_dir=OUTPUT_DIR, filename='signal_noise'):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3))
    paired_panel(axes[0], pp, 'signal_power', 'Signal power', stats, ylim=(0, 15))
    paired_panel(axes[1], pp, 'noise_power', 'Noise power', stats, ylim=(0, 20))

    drift_panel(axes[2], drift_df, stats)
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    vectors = load_response_vectors()
    print(
        f"{sum(v[0] == 'R+' for v in vectors.values())} R+ and {sum(v[0] == 'R-' for v in vectors.values())} R- mice"
    )
    day_df, drift_df = compute_metrics(vectors)
    stats, pp = compute_stats(day_df, drift_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    day_df.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_data.csv'), index=False)
    drift_df.to_csv(os.path.join(OUTPUT_DIR, 'drift_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_stats.csv'), index=False)
    plot(day_df, drift_df, pp, stats)

    print(
        stats[['reward_group', 'measure', 'n_mice', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
    print(drift_df.groupby(['reward_group', 'pair'])[['r', 'r_corrected']].median().round(3).to_string())
    print(f'Saved to {OUTPUT_DIR}')
