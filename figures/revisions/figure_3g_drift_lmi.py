"""
Figure 3g revision: how many significant LMIs does day-to-day drift produce?

The LMI of pipeline step 05 (ROC of each cell's response, days -2/-1 vs
+1/+2, significance from trial-label shuffles) is computed here between
consecutive days, one day on each side. Days -2 and -1 are the only
consecutive days without learning (mice are auditory experts; performance
keeps improving after day 0), so -2/-1 measures drift. Day 0 responses are
recorded after the learning session, so -1/0 spans learning. Same response
window (0-300 ms), baseline (-1-0 s), all passive whisker trials of each
day, 1000 shuffles, LMI+ at p >= 0.975 and LMI- at p <= 0.025.

Per mouse and day pair: fraction of LMI+ and LMI- cells, of all significant
cells (LMI+ or LMI-), and the net fraction (LMI+ minus LMI-): drift without
a consistent direction gives as many LMI+ as LMI- cells. Stats, n = mice:
- change from -2/-1 to -1/0 within mice, each fraction (Wilcoxon signed-rank);
- net fraction against zero in each group (Wilcoxon signed-rank) and between
  groups (Mann-Whitney U), for each pair and for the step-05 LMI.
The step-05 LMI (two days on each side, so twice as many trials) is shown
for reference; its fractions are not directly comparable to the one-day
pairs. Stars above -1/0: change from -2/-1 (R+ bottom, R- top). In the net
panel, stars above -2/-1 and the step-05 LMI: net against zero.

Outputs: <figures_dir>/revisions/figure_3g_drift_lmi/output/
    drift_lmi.pdf, drift_lmi_data.csv (mouse x day pair), drift_lmi_stats.csv,
    drift_lmi_cells.csv (cell x day pair: LMI, p).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from fast_learning import database, imaging, paths
from fast_learning.participation import LMI_NEGATIVE_THRESHOLD, LMI_POSITIVE_THRESHOLD
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import significance_stars

RESPONSE_WIN = (0, 0.300)
BASELINE_WIN = (-1, 0)
N_SHUFFLES = 1000
SEED = 0
BASELINE_PAIR = (-2, -1)  # the only consecutive days without learning
LEARNING_PAIR = (-1, 0)  # spans the first whisker learning session
DAY_PAIRS = [BASELINE_PAIR, LEARNING_PAIR]
REFERENCE = 'LMI (-2,-1/+1,+2)'
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
FRACTIONS = {
    'positive': 'LMI+ cells (%)',
    'negative': 'LMI− cells (%)',
    'significant': 'LMI+ or LMI− cells (%)',
    'net': 'LMI+ minus LMI− cells (%)',
}

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_3g_drift_lmi', 'output')


def pair_label(a, b):
    return f'{a:+d}/{b:+d}'.replace('+0', '0')


# ============================================================================
# LMI between two days
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


# ============================================================================
# Fractions and statistics
# ============================================================================


def fractions(cells):
    """Per mouse and pair, from cell counts (so that equal counts give exactly equal fractions)."""
    pos = (cells['lmi_p'] >= LMI_POSITIVE_THRESHOLD).astype(int)
    neg = (cells['lmi_p'] <= LMI_NEGATIVE_THRESHOLD).astype(int)
    cells = cells.assign(positive=pos, negative=neg, significant=pos + neg, net=pos - neg)
    return (
        cells.groupby(['mouse_id', 'reward_group', 'pair'])
        .agg(**{m: (m, 'mean') for m in FRACTIONS}, n_cells=('roi', 'size'))
        .reset_index()
    )


def stat_row(pair, measure, rg, test, x_plus, x_minus, stat, p):
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


def compute_stats(frac):
    """Change from -2/-1 to -1/0 of each fraction; net fraction against zero and R+ vs R-."""
    base, learn = pair_label(*BASELINE_PAIR), pair_label(*LEARNING_PAIR)
    rows = []
    for m in FRACTIONS:
        w = frac.pivot(index=['mouse_id', 'reward_group'], columns='pair', values=m).reset_index()
        d = {rg: (w[learn] - w[base])[w['reward_group'] == rg].dropna() for rg in ['R+', 'R-']}
        for rg, x in d.items():
            rows.append(
                stat_row(
                    f'{learn} minus {base}', m, rg, 'Wilcoxon signed-rank', d['R+'], d['R-'], *wilcoxon(x)
                )
            )
    for pair in [base, learn, REFERENCE]:
        g = {rg: frac[(frac['pair'] == pair) & (frac['reward_group'] == rg)]['net'] for rg in ['R+', 'R-']}
        for rg, x in g.items():
            rows.append(
                stat_row(pair, 'net', rg, 'Wilcoxon signed-rank vs 0', g['R+'], g['R-'], *wilcoxon(x))
            )
        rows.append(
            stat_row(
                pair, 'net', 'R+ vs R-', 'Mann-Whitney U', g['R+'], g['R-'], *mannwhitneyu(g['R+'], g['R-'])
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Figure
# ============================================================================


def pair_points(ax, data, y, order):
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
        legend=False,
        ax=ax,
    )
    ax.axvline(len(DAY_PAIRS) - 0.5, color='grey', linewidth=0.4)
    ax.set_xticks(range(len(order)), [*order[:-1], 'LMI\n(2 days)'])
    ax.set_xlabel('Day pair')


def stars(ax, x, rows):
    for gi, rg in enumerate(['R+', 'R-']):  # bottom to top
        r = rows[rows['reward_group'] == rg]
        if len(r):
            ax.text(
                x,
                1.0 + 0.07 * gi,
                r['significance'].iloc[0],
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                color=GROUP_COLORS[rg],
                fontsize=5,
            )


def plot(frac, stats, output_dir=OUTPUT_DIR, filename='drift_lmi'):
    set_style()
    order = [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]
    learn = order.index(pair_label(*LEARNING_PAIR))
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=panel_size(len(FRACTIONS)))
    for ax, (m, label) in zip(axes, FRACTIONS.items()):
        pair_points(ax, frac.assign(pct=100 * frac[m]), 'pct', order)
        stars(ax, learn, stats[(stats['measure'] == m) & stats['pair'].str.contains('minus')])
        ax.set_ylabel(label)
        if m == 'net':
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
            ax.set_ylim(-25, 25)
            for pair in [order[0], REFERENCE]:
                stars(ax, order.index(pair), stats[(stats['measure'] == m) & (stats['pair'] == pair)])
        else:
            chance = 2 if m == 'significant' else 1
            ax.axhline(
                100 * chance * (1 - LMI_POSITIVE_THRESHOLD), color='grey', linestyle='--', linewidth=0.6
            )
            ax.set_ylim(0, 45 if m == 'significant' else 30)
    for rg in ['R+', 'R-']:
        axes[0].plot([], [], color=GROUP_COLORS[rg], label=rg)
    axes[0].legend(frameon=False, loc='upper left')
    sns.despine()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    mice = sorted(lmi_df['mouse_id'].unique())
    _, _, _, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    groups = {m: database.get_mouse_reward_group_from_db(paths.db_path, m, db) for m in mice}

    seeds = np.random.SeedSequence(SEED).spawn(len(mice))
    cells = pd.concat(
        [mouse_drift_lmi(m, np.random.default_rng(s)) for m, s in zip(mice, seeds)], ignore_index=True
    )
    cells['reward_group'] = cells['mouse_id'].map(groups)
    reference = lmi_df.assign(pair=REFERENCE, reward_group=lmi_df['mouse_id'].map(groups))
    frac = fractions(pd.concat([cells, reference[cells.columns.intersection(reference.columns)]]))
    stats = compute_stats(frac)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cells.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_cells.csv'), index=False)
    frac.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'drift_lmi_stats.csv'), index=False)
    plot(frac, stats)

    print(frac.groupby(['reward_group', 'pair'])[list(FRACTIONS)].median().mul(100).round(1).to_string())
    print(
        stats[['pair', 'measure', 'reward_group', 'median_r_plus', 'median_r_minus', 'p_value']]
        .round(4)
        .to_string(index=False)
    )
    print(f'Saved to {OUTPUT_DIR}')
