"""
Figure 3g-i revision: response strength, trial-to-trial noise and day-to-day
drift of whisker responses.

Days -2 and -1 are the only consecutive days without learning (mice are
auditory experts; performance keeps improving after day 0), so -2/-1
measures drift. Day 0 responses are recorded after the learning session, so
-1/0 spans learning. n = mice throughout.

Top row, the response vectors of Fig. 3h-i (last 40 passive trials per day):
- Signal power (squared norm of the trial-averaged response, corrected for
  noise) and noise power (summed trial-to-trial variance), pre (days -2, -1)
  vs post (days +1, +2): Wilcoxon signed-rank test.
- Day-to-day pattern drift: correlation of the mean response vectors of two
  consecutive days, divided by the geometric mean of their split-half
  reliabilities (Spearman-Brown corrected), so that 1 means the same pattern
  on both days. -1/0 vs -2/-1 within mice (Wilcoxon signed-rank), and
  -2/-1 between groups (Mann-Whitney U).

Bottom row, the LMI of pipeline step 05 (ROC of each cell's response, trial-
label shuffles) computed between consecutive days, one day on each side.
Same response window (0-300 ms), baseline (-1-0 s), all passive whisker
trials of each day, 1000 shuffles, LMI+ at p >= 0.975 and LMI- at
p <= 0.025. Per mouse and pair: fraction of LMI+ and LMI- cells and the net
fraction (LMI+ minus LMI-); drift without a consistent direction gives as
many LMI+ as LMI- cells. The step-05 LMI (two days on each side, so twice as
many trials) is shown for reference; its fractions are not directly
comparable to the one-day pairs.
- Change from -2/-1 to -1/0 within mice (Wilcoxon signed-rank): p-values above
  -1/0 (R+ bottom, R- top).
- Net fraction against zero (Wilcoxon signed-rank; p-values above -2/-1 and the
  step-05 LMI) and between groups (Mann-Whitney U; stats file only).

Outputs: <figures_dir>/revisions/figure_3g_i_signal_noise_drift/output/
    signal_noise_drift.pdf,
    signal_noise_data.csv (mouse x day), drift_data.csv (mouse x day pair),
    signal_noise_stats.csv,
    drift_lmi_data.csv (mouse x day pair), drift_lmi_cells.csv (cell x day
    pair: LMI, p), drift_lmi_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, rankdata, wilcoxon

from fast_learning import database, imaging, paths
from fast_learning.participation import LMI_NEGATIVE_THRESHOLD, LMI_POSITIVE_THRESHOLD
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
RESPONSE_WIN = (0, 0.300)  # LMI, as pipeline step 05
BASELINE_WIN = (-1, 0)
N_SHUFFLES = 1000
SEED = 0
REFERENCE = 'LMI (-2,-1/+1,+2)'
FRACTIONS = {
    'positive': 'LMI+ cells (%)',
    'negative': 'LMI− cells (%)',
    'net': 'LMI+ minus LMI− cells (%)',
}
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}

OUTPUT_DIR = os.path.join(
    paths.manuscript_output_dir, 'revisions', 'figure_3g_i_signal_noise_drift', 'output'
)


def pair_label(a, b):
    return f'{a:+d}/{b:+d}'.replace('+0', '0')


# ============================================================================
# Signal, noise and pattern drift: the response vectors of Fig. 3h-i
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


def pre_post_means(day_df, metrics):
    period = day_df['day'].map({d: 'pre' for d in PRE_DAYS} | {d: 'post' for d in POST_DAYS})
    return (
        day_df.assign(period=period)
        .dropna(subset=['period'])
        .groupby(['reward_group', 'mouse_id', 'period'])[metrics]
        .mean()
    )


def signal_noise_row(rg, measure, test, n, median_pre, median_post, stat, p):
    return {
        'reward_group': rg,
        'measure': measure,
        'test': test,
        'n_mice': n,
        'median_pre': median_pre,
        'median_post': median_post,
        'statistic': stat,
        'p_value': p,
        'significance': significance_stars(p) if np.isfinite(p) else '',
    }


def signal_noise_stats(day_df, drift_df):
    metrics = ['signal_power', 'noise_power', 'snr', 'cosine_observed', 'cosine_predicted']
    pp = pre_post_means(day_df, metrics).unstack('period')
    base, learn = pair_label(*BASELINE_PAIR), pair_label(*LEARNING_PAIR)
    drift = drift_df.pivot(index=['reward_group', 'mouse_id'], columns='pair', values='r_corrected')
    rows = []
    for rg in ['R+', 'R-']:
        g = pp.loc[rg]
        for m in metrics:
            rows.append(
                signal_noise_row(
                    rg,
                    m,
                    'Wilcoxon signed-rank, pre (days -2, -1) vs post (days +1, +2)',
                    len(g),
                    g[(m, 'pre')].median(),
                    g[(m, 'post')].median(),
                    *wilcoxon(g[(m, 'pre')], g[(m, 'post')]),
                )
            )
        g = drift.loc[rg]
        rows.append(
            signal_noise_row(
                rg,
                f'drift {learn}',
                f'Wilcoxon signed-rank, noise-corrected correlation {learn} vs {base}',
                len(g),
                g[base].median(),
                g[learn].median(),
                *wilcoxon(g[learn] - g[base]),
            )
        )
    x, y = drift.loc['R+'][base], drift.loc['R-'][base]
    rows.append(
        signal_noise_row(
            'R+ vs R-',
            f'drift {base}',
            'Mann-Whitney U, noise-corrected correlation (median_pre: R+, median_post: R-)',
            len(x) + len(y),
            x.median(),
            y.median(),
            *mannwhitneyu(x, y),
        )
    )
    r, _ = pearsonr(day_df['cosine_observed'], day_df['cosine_predicted'])
    rows.append(
        signal_noise_row(
            'all',
            'observed vs predicted within-day cosine',
            'Pearson r across mouse-days (descriptive)',
            day_df['mouse_id'].nunique(),
            np.nan,
            r,
            r,
            np.nan,
        )
    )
    return pd.DataFrame(rows), pp


# ============================================================================
# LMI between two consecutive days
# ============================================================================


def auc_rows(ranks, post_idx, n_post):
    """AUC of each row for the trials in post_idx against the others, from ranks."""
    n_pre = ranks.shape[-1] - n_post
    return (ranks[..., post_idx].sum(-1) - n_post * (n_post + 1) / 2) / (n_pre * n_post)


def lmi_two_days(pre, post, rng, n_shuffles=N_SHUFFLES):
    """LMI = 2 (AUC - 0.5) of each cell (rows) and its shuffle percentile, as
    imaging.compute_roc (scipy percentileofscore, kind='rank'), vectorised
    over cells with the same label shuffles for all cells."""
    X = np.concatenate([pre, post], axis=1)
    n, n_post = X.shape[1], post.shape[1]
    ranks = rankdata(X, axis=1)
    auc = auc_rows(ranks, np.arange(n - n_post, n), n_post)
    null = np.stack([auc_rows(ranks, rng.permutation(n)[:n_post], n_post) for _ in range(n_shuffles)], 1)
    below = (null < auc[:, None]).sum(1)
    at_or_below = (null <= auc[:, None]).sum(1)
    p = (below + at_or_below + (at_or_below > below)) * 0.5 / n_shuffles
    return 2 * (auc - 0.5), p


def mouse_drift_lmi(mouse, rng):
    """Cell x day pair LMI and p of one mouse, preprocessed as pipeline step 05."""
    data = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc', subtracted=False
    )
    data = data - np.nanmean(imaging.select_time(data, *BASELINE_WIN), axis=2, keepdims=True)
    resp = imaging.select_time(data, *RESPONSE_WIN).mean(dim='time')
    days = resp['day'].values
    rows = []
    for a, b in DAY_PAIRS:
        if not (np.any(days == a) and np.any(days == b)):
            continue
        lmi, p = lmi_two_days(resp.values[:, days == a], resp.values[:, days == b], rng)
        rows.append(
            pd.DataFrame(
                dict(
                    mouse_id=mouse,
                    roi=resp['roi'].values,
                    pair=pair_label(a, b),
                    lmi=lmi,
                    lmi_p=p,
                    n_trials_a=int((days == a).sum()),
                    n_trials_b=int((days == b).sum()),
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


def drift_lmi(groups, lmi_df):
    """Cell x pair LMIs of all mice, seeded per mouse."""
    mice = sorted(lmi_df['mouse_id'].unique())
    seeds = np.random.SeedSequence(SEED).spawn(len(mice))
    cells = pd.concat(
        [mouse_drift_lmi(m, np.random.default_rng(s)) for m, s in zip(mice, seeds)], ignore_index=True
    )
    cells['reward_group'] = cells['mouse_id'].map(groups)
    return cells


def fractions(cells):
    """Per mouse and pair, from cell counts (so that equal counts give exactly equal fractions)."""
    pos = (cells['lmi_p'] >= LMI_POSITIVE_THRESHOLD).astype(int)
    neg = (cells['lmi_p'] <= LMI_NEGATIVE_THRESHOLD).astype(int)
    cells = cells.assign(positive=pos, negative=neg, net=pos - neg)
    return (
        cells.groupby(['mouse_id', 'reward_group', 'pair'])
        .agg(**{m: (m, 'mean') for m in FRACTIONS}, n_cells=('roi', 'size'))
        .reset_index()
    )


def lmi_row(pair, measure, rg, test, x_plus, x_minus, stat, p):
    return dict(
        pair=pair,
        measure=measure,
        reward_group=rg,
        test=test,
        median_r_plus=x_plus.median(),
        median_r_minus=x_minus.median(),
        n_r_plus=len(x_plus),
        n_r_minus=len(x_minus),
        statistic=stat,
        p_value=p,
        significance=significance_stars(p),
    )


def drift_lmi_stats(frac):
    """Change from -2/-1 to -1/0 of each fraction; net fraction against zero and R+ vs R-."""
    base, learn = pair_label(*BASELINE_PAIR), pair_label(*LEARNING_PAIR)
    rows = []
    for m in FRACTIONS:
        w = frac.pivot(index=['mouse_id', 'reward_group'], columns='pair', values=m).reset_index()
        d = {rg: (w[learn] - w[base])[w['reward_group'] == rg].dropna() for rg in ['R+', 'R-']}
        for rg, x in d.items():
            rows.append(
                lmi_row(
                    f'{learn} minus {base}', m, rg, 'Wilcoxon signed-rank', d['R+'], d['R-'], *wilcoxon(x)
                )
            )
    for pair in [base, learn, REFERENCE]:
        g = {rg: frac[(frac['pair'] == pair) & (frac['reward_group'] == rg)]['net'] for rg in ['R+', 'R-']}
        for rg, x in g.items():
            rows.append(lmi_row(pair, 'net', rg, 'Wilcoxon signed-rank vs 0', g['R+'], g['R-'], *wilcoxon(x)))
        rows.append(
            lmi_row(
                pair, 'net', 'R+ vs R-', 'Mann-Whitney U', g['R+'], g['R-'], *mannwhitneyu(g['R+'], g['R-'])
            )
        )
    return pd.DataFrame(rows)


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
    ax.set_xlabel('')


def pair_points(ax, data, y, order, legend=False):
    """Mean and bootstrapped 95% CI across mice for each day pair."""
    sns.pointplot(
        data=data,
        x='pair',
        y=y,
        order=order,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        dodge=0.3,
        markersize=3,
        linewidth=1,
        legend=legend,
        ax=ax,
    )
    ax.set_xlabel('Day pair')


def pattern_drift_panel(ax, drift_df, stats):
    """Noise-corrected correlation of consecutive days."""
    order = [pair_label(*p) for p in DAY_PAIRS]
    pair_points(ax, drift_df, 'r_corrected', order, legend=True)
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.6)
    # p-values above each pair: -1/0 vs -2/-1 within group (colored), R+ vs R- (black).
    for gi, rg in enumerate(['R+', 'R-', 'R+ vs R-']):
        for xi, pair in enumerate(order):
            row = stats[(stats['reward_group'] == rg) & (stats['measure'] == f'drift {pair}')]
            if len(row):
                ax.text(
                    xi,
                    1.04 + 0.06 * gi if rg in GROUP_COLORS else 1.04,
                    ('' if rg in GROUP_COLORS else 'R+ vs R−\n') + format_p(row['p_value'].iloc[0]),
                    ha='center',
                    va='bottom',
                    color=GROUP_COLORS.get(rg, 'black'),
                    fontsize=5,
                )
    ax.set_ylim(0.4, 1.2)
    ax.set_ylabel('Day-to-day pattern correlation\n(noise-corrected)')
    ax.legend(frameon=False, loc='lower left')


def pvalues(ax, x, rows):
    for gi, rg in enumerate(['R+', 'R-']):  # bottom to top
        r = rows[rows['reward_group'] == rg]
        if len(r):
            ax.text(
                x,
                1.0 + 0.07 * gi,
                format_p(r['p_value'].iloc[0]),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                color=GROUP_COLORS[rg],
                fontsize=5,
            )


def lmi_panel(ax, frac, stats, m):
    order = [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]
    pair_points(ax, frac.assign(pct=100 * frac[m]), 'pct', order)
    ax.axvline(len(DAY_PAIRS) - 0.5, color='grey', linewidth=0.4)
    ax.set_xticks(range(len(order)), [*order[:-1], 'LMI\n(2 days)'])
    rows = stats[stats['measure'] == m]
    pvalues(ax, order.index(pair_label(*LEARNING_PAIR)), rows[rows['pair'].str.contains('minus')])
    if m == 'net':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(-25, 25)
        for pair in [order[0], REFERENCE]:
            pvalues(ax, order.index(pair), rows[rows['pair'] == pair])
    else:
        ax.axhline(100 * (1 - LMI_POSITIVE_THRESHOLD), color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(0, 30)
    ax.set_ylabel(FRACTIONS[m])


def plot(pp, sn_stats, drift_df, frac, lmi_stats, output_dir=OUTPUT_DIR, filename='signal_noise_drift'):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=panel_size(3, 2))
    paired_panel(axes[0, 0], pp, 'signal_power', 'Signal power', sn_stats, ylim=(0, 15))
    paired_panel(axes[0, 1], pp, 'noise_power', 'Noise power', sn_stats, ylim=(0, 20))
    pattern_drift_panel(axes[0, 2], drift_df, sn_stats)
    for ax, m in zip(axes[1], FRACTIONS):
        lmi_panel(ax, frac, lmi_stats, m)
    sns.despine()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    vectors = load_response_vectors()
    print(
        f"{sum(v[0] == 'R+' for v in vectors.values())} R+ and {sum(v[0] == 'R-' for v in vectors.values())} R- mice"
    )
    day_df, drift_df = compute_metrics(vectors)
    sn_stats, pp = signal_noise_stats(day_df, drift_df)

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    _, _, _, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    groups = {
        m: database.get_mouse_reward_group_from_db(paths.db_path, m, db) for m in lmi_df['mouse_id'].unique()
    }
    cells = drift_lmi(groups, lmi_df)
    reference = lmi_df.assign(pair=REFERENCE, reward_group=lmi_df['mouse_id'].map(groups))
    frac = fractions(pd.concat([cells, reference[cells.columns.intersection(reference.columns)]]))
    lmi_stats = drift_lmi_stats(frac)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    day_df.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_data.csv'), index=False)
    drift_df.to_csv(os.path.join(OUTPUT_DIR, 'drift_data.csv'), index=False)
    sn_stats.to_csv(os.path.join(OUTPUT_DIR, 'signal_noise_stats.csv'), index=False)
    cells.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_cells.csv'), index=False)
    frac.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_data.csv'), index=False)
    lmi_stats.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_stats.csv'), index=False)
    plot(pp, sn_stats, drift_df, frac, lmi_stats)

    print(
        sn_stats[['reward_group', 'measure', 'n_mice', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
    print(frac.groupby(['reward_group', 'pair'])[list(FRACTIONS)].median().mul(100).round(1).to_string())
    print(
        lmi_stats[['pair', 'measure', 'reward_group', 'median_r_plus', 'median_r_minus', 'p_value']]
        .round(4)
        .to_string(index=False)
    )
    print(f'Saved to {OUTPUT_DIR}')
