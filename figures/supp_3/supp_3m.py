"""
Supplementary Figure 3m: Pairwise correlations between projection neurons
(wS2-wS2 and wM1-wM1 pairs) during a 2 s pre-stimulus quiet window,
compared pre vs post learning. Mapping trials only.

Stats: change of each pair's correlation (post - pre), tested with a linear
mixed-effects model change ~ 1 + (1 | mouse), so that pairs recorded in the
same mouse are not treated as independent.

Inputs:  pair-level correlations (pipeline/09_pairwise_correlations.py).
Outputs: <figures_dir>/supp_3/output/supp_3m_<group>.pdf, _data.csv (pairs),
         _mouse_means.csv (plotted, one value per mouse), _stats.csv.
Figure: mean over each mouse's pairs, pre and post (lines); bars = mean across
mice with bootstrapped 95% CI (seeded).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from fast_learning import paths, correlations
from fast_learning.plotting import save_figure, set_style, panel_size
from fast_learning.stats import format_p, lmm_mean


# ============================================================================
# Parameters
# ============================================================================

PAIR_TYPES = correlations.PAIR_TYPES

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Significance annotation helper
# ============================================================================


def add_p_value_bracket(ax, x1, x2, y, p_value):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c='black')
    ax.text((x1 + x2) / 2, y + h, format_p(p_value), ha='center', va='bottom')


# ============================================================================
# Main loop per reward group
# ============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_style()

    for reward_group in ['R+', 'R-']:
        corr_csv = correlations.correlations_csv(reward_group)
        if not os.path.exists(corr_csv):
            raise FileNotFoundError(f"{corr_csv} not found. Run pipeline/09_pairwise_correlations.py first.")
        print(f"\nLoading {corr_csv}")
        corr_df = pd.read_csv(corr_csv)

        print(f"Pairs: {corr_df['pair_type'].value_counts().to_dict()}")

        # ── Stats: per-pair change, mixed model with mouse as random intercept ─
        key = ['mouse_id', 'pair_type', 'roi_i', 'roi_j']
        paired = (
            corr_df.pivot_table(index=key, columns='period', values='correlation')
            .dropna(subset=['pre', 'post'])
            .reset_index()
        )
        paired['change'] = paired['post'] - paired['pre']
        stats_pair = []
        for pt in PAIR_TYPES:
            sub = paired[paired['pair_type'] == pt]
            if sub['mouse_id'].nunique() < 2:
                continue
            fit = lmm_mean(sub, 'change')
            stats_pair.append(
                {
                    'pair_type': pt,
                    'test': 'LMM change ~ 1 + (1 | mouse)',
                    'mean_pre': sub['pre'].mean(),
                    'sem_pre': sub['pre'].std() / np.sqrt(len(sub)),
                    'mean_post': sub['post'].mean(),
                    'sem_post': sub['post'].std() / np.sqrt(len(sub)),
                    'n_pairs': len(sub),
                    'n_mice': sub['mouse_id'].nunique(),
                    'lmm_change': fit['mean'],
                    'lmm_ci_low': fit['ci_low'],
                    'lmm_ci_high': fit['ci_high'],
                    'p_value': fit['p_value'],
                    'icc_mouse': fit['icc_mouse'],
                    'method': fit['method'],
                }
            )
            print(
                f"  {reward_group} {pt}: change={fit['mean']:.4g}, p={fit['p_value']:.3g}, n={len(sub)} pairs"
            )
        stats_pair_df = pd.DataFrame(stats_pair)

        stats_pair_df.to_csv(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_stats.csv'), index=False)
        corr_df.to_csv(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_data.csv'), index=False)

        # ── Figure: one value per mouse (mean over its pairs), the unit of the test ─
        mouse_means = paired.groupby(['mouse_id', 'pair_type'])[['pre', 'post']].mean().reset_index()
        mouse_means.to_csv(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_mouse_means.csv'), index=False)

        fig, axes = plt.subplots(1, 2, figsize=panel_size(2, w=0.7), sharey=True)
        for idx, pt in enumerate(PAIR_TYPES):
            ax = axes[idx]
            mm = mouse_means[mouse_means['pair_type'] == pt]
            # Bars: mean across mice, bootstrapped 95% CI (seeded) as in the other panels.
            long = mm.melt(
                id_vars='mouse_id', value_vars=['pre', 'post'], var_name='period', value_name='corr'
            )
            sns.barplot(
                data=long,
                x='period',
                y='corr',
                order=['pre', 'post'],
                color='lightgrey',
                edgecolor='black',
                width=0.6,
                errorbar=('ci', 95),
                seed=0,
                capsize=0.1,
                ax=ax,
            )
            for _, row in mm.iterrows():
                ax.plot(
                    [0, 1],
                    [row['pre'], row['post']],
                    '-o',
                    color='grey',
                    markersize=3,
                    linewidth=0.8,
                    alpha=0.7,
                )
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xticks([0, 1], ['Pre', 'Post'])
            ax.set_title(f'{pt}\n(n = {len(mm)} mice)', fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Pearson correlation\n(mean over pairs)' if idx == 0 else '')

            row = stats_pair_df[stats_pair_df['pair_type'] == pt]
            if not row.empty:
                top = mm[['pre', 'post']].max().max()
                add_p_value_bracket(
                    ax, 0, 1, top + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, row.iloc[0]['p_value']
                )

        plt.suptitle(f'Pre vs post learning ({reward_group})', y=1.02)
        plt.tight_layout()
        sns.despine()
        save_figure(fig, os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}.pdf'))
        print(f"Saved: supp_3m_{reward_group}.svg")
        plt.close()

    print("\nDone.")
