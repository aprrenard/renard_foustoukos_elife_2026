"""
Figure 3h-i revision: what drives the change in within-day similarity?

Outputs: <figures_dir>/revisions/figure_3h_i_signal_noise/output/
    signal_noise.pdf, signal_noise_data.csv (mouse x day),
    signal_noise_stats.csv, pattern_preservation_data.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wilcoxon

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


def compute_metrics(vectors):
    day_rows, pattern_rows = [], []
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
        pre, post = X[:, np.isin(day_of, PRE_DAYS)], X[:, np.isin(day_of, POST_DAYS)]
        r_raw = pearsonr(pre.mean(1), post.mean(1))[0]
        rel_pre, rel_post = split_half_reliability(pre, rng), split_half_reliability(post, rng)
        (s_pre, _), (s_post, _) = signal_noise(pre), signal_noise(post)
        pattern_rows.append(
            {
                'mouse_id': mouse,
                'reward_group': rg,
                'n_cells': X.shape[0],
                'r_pre_post': r_raw,
                'reliability_pre': rel_pre,
                'reliability_post': rel_post,
                'r_corrected': r_raw / np.sqrt(rel_pre * rel_post),
                'signal_amplitude_ratio': np.sqrt(max(s_post, 0) / s_pre),
            }
        )
    return pd.DataFrame(day_rows), pd.DataFrame(pattern_rows)


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


def compute_stats(day_df, pattern_df):
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
        pg = pattern_df[pattern_df['reward_group'] == rg]
        stat, p = wilcoxon(pg['r_corrected'] - 1)
        rows.append(
            {
                'reward_group': rg,
                'measure': 'r_corrected (pre vs post pattern)',
                'test': 'Wilcoxon signed-rank against 1 (uniform scaling)',
                'n_mice': len(pg),
                'median_pre': np.nan,
                'median_post': pg['r_corrected'].median(),
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
        ax.bar(
            x,
            [g['pre'].median(), g['post'].median()],
            width=0.7,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            edgecolor='black',
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


def plot(day_df, pattern_df, pp, stats, output_dir=OUTPUT_DIR, filename='signal_noise'):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3))
    paired_panel(axes[0], pp, 'signal_power', 'Signal power', stats, ylim=(0, 15))
    paired_panel(axes[1], pp, 'noise_power', 'Noise power', stats, ylim=(0, 20))

    ax = axes[2]
    sns.swarmplot(
        data=pattern_df,
        x='reward_group',
        y='r_corrected',
        order=['R+', 'R-'],
        hue='reward_group',
        palette=GROUP_COLORS,
        size=3,
        legend=False,
        ax=ax,
    )
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(0, 1)
    for gi, rg in enumerate(['R+', 'R-']):
        p = stats.query("reward_group == @rg and measure.str.startswith('r_corrected')", engine='python')[
            'p_value'
        ].iloc[0]
        ax.text(gi, 1.02, format_p(p), ha='center', va='bottom')
    ax.set_xlabel('')
    ax.set_ylabel('Pre-post pattern correlation\n(noise-corrected)')
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
    day_df, pattern_df = compute_metrics(vectors)
    stats, pp = compute_stats(day_df, pattern_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    day_df.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_data.csv'), index=False)
    pattern_df.to_csv(os.path.join(OUTPUT_DIR, 'pattern_preservation_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_stats.csv'), index=False)
    plot(day_df, pattern_df, pp, stats)

    print(
        stats[['reward_group', 'measure', 'n_mice', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
    print(
        pattern_df.groupby('reward_group')[
            ['r_pre_post', 'reliability_pre', 'reliability_post', 'r_corrected', 'signal_amplitude_ratio']
        ]
        .median()
        .round(3)
        .to_string()
    )
    print(f'Saved to {OUTPUT_DIR}')
