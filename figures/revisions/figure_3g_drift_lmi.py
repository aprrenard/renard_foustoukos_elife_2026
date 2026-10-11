"""
Figure 3g revision: how many significant LMIs does day-to-day drift produce?

The LMI of pipeline step 05 (ROC of each cell's response, days -2/-1 vs
+1/+2, significance from trial-label shuffles) is computed here between
consecutive days, one day on each side: -2/-1 and +1/+2 measure drift
without learning in between, -1/0 spans the learning session (day 0
responses are recorded after it), 0/+1 the day after. Same response window
(0-300 ms), baseline (-1-0 s), all passive whisker trials of each day,
1000 shuffles, LMI+ at p >= 0.975 and LMI- at p <= 0.025.

Per mouse and day pair: fraction of LMI+ and LMI- cells, and their
difference (net, LMI+ minus LMI-): drift without a consistent direction
gives as many LMI+ as LMI- cells. Stats on the net fraction, n = mice:
Wilcoxon signed-rank test against zero in each group, and Mann-Whitney U
test between groups. The step-05 LMI (two days on each side, so twice as
many trials) is shown and tested in the same way for reference; its
fractions are not directly comparable to the one-day pairs.
Net panel, stars from top to bottom: R+ vs R-, R- vs 0, R+ vs 0.

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
DAY_PAIRS = [(-2, -1), (-1, 0), (0, 1), (1, 2)]
BASELINE_PAIRS = [(-2, -1), (1, 2)]  # one day apart, no learning in between
REFERENCE = 'LMI (-2,-1/+1,+2)'
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
CATEGORIES = {'positive': 'LMI+', 'negative': 'LMI−'}

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
    cells = cells.assign(
        positive=cells['lmi_p'] >= LMI_POSITIVE_THRESHOLD, negative=cells['lmi_p'] <= LMI_NEGATIVE_THRESHOLD
    )
    return (
        cells.groupby(['mouse_id', 'reward_group', 'pair'])
        .agg(positive=('positive', 'mean'), negative=('negative', 'mean'), n_cells=('roi', 'size'))
        .reset_index()
    )


def compute_stats(frac):
    """Net fraction (LMI+ minus LMI-) per day pair: against zero per group, and R+ vs R-."""
    net = frac.assign(net=frac['positive'] - frac['negative'])
    rows = []
    for pair in [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]:
        g = {rg: net[(net['pair'] == pair) & (net['reward_group'] == rg)]['net'] for rg in ['R+', 'R-']}
        tests = [(rg, 'Wilcoxon signed-rank vs 0', *wilcoxon(x)) for rg, x in g.items()]
        tests.append(('R+ vs R-', 'Mann-Whitney U', *mannwhitneyu(g['R+'], g['R-'])))
        for rg, test, stat, p in tests:
            rows.append(
                dict(
                    pair=pair,
                    reward_group=rg,
                    measure='net fraction (LMI+ minus LMI-)',
                    test=test,
                    median_r_plus=g['R+'].median(),
                    median_r_minus=g['R-'].median(),
                    n_r_plus=len(g['R+']),
                    n_r_minus=len(g['R-']),
                    statistic=stat,
                    p_value=p,
                    significance=significance_stars(p),
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


def plot(frac, stats, output_dir=OUTPUT_DIR, filename='drift_lmi'):
    set_style()
    order = [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3))
    for ax, (cat, label) in zip(axes, CATEGORIES.items()):
        pair_points(ax, frac.assign(pct=100 * frac[cat]), 'pct', order)
        ax.axhline(100 * (1 - LMI_POSITIVE_THRESHOLD), color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(0, 30)
        ax.set_ylabel(f'{label} cells (%)')
    ax = axes[2]
    pair_points(ax, frac.assign(net=100 * (frac['positive'] - frac['negative'])), 'net', order)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(-25, 25)
    ax.set_ylabel('LMI+ minus LMI− cells (%)')
    for gi, rg in enumerate(['R+', 'R-', 'R+ vs R-']):  # bottom to top
        for _, r in stats[stats['reward_group'] == rg].iterrows():
            ax.text(
                order.index(r['pair']),
                1.0 + 0.07 * gi,
                significance_stars(r['p_value']),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                color=GROUP_COLORS.get(rg, 'black'),
                fontsize=5,
            )
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

    print(
        frac.groupby(['reward_group', 'pair'])[['positive', 'negative']]
        .median()
        .mul(100)
        .round(1)
        .to_string()
    )
    print(
        stats[['pair', 'reward_group', 'median_r_plus', 'median_r_minus', 'p_value']]
        .round(4)
        .to_string(index=False)
    )
    print(f'Saved to {OUTPUT_DIR}')
