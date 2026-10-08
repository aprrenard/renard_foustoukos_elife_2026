"""
Supplementary figure (panel index TBD): LMI distribution vs. shuffled null.

Reviewer request: for R+ and R- independently, overlay the real LMI
distribution against a null LMI distribution built by chance (label
shuffling), to show the real distribution is more spread than chance alone
would produce.

LMI computation (data source, response/baseline windows, pre=[-2,-1] vs
post=[+1,+2] day pooling) exactly mirrors the "Compute LMI" section of
src/preprocessing/processing_tensor_data/stats_on_tensors.py. The null
distribution reuses the same shuffle procedure already used there for LMI
significance testing (imaging.compute_roc), via the new
return_shuffles=True option, which keeps every per-cell per-shuffle null LMI
value instead of collapsing them to a single percentile. All shuffle x cell
null values are pooled per reward group (not averaged per cell, which would
artificially shrink the null's spread, and not one shuffle per cell, which
would give too few null points for a stable histogram).

The null distribution (slow) is computed when its cache is missing or with
--recompute, and loaded from the cache otherwise:
    python figures/revisions/figure_3f_LMIshuffles.py [--recompute]

Real LMI values are loaded as-is from lmi_results.csv (already computed).

Statistics, with mice as the unit: in each mouse, the fraction of cells
whose LMI is significant by the shuffle test (lmi_p >= 0.975 or <= 0.025,
5% expected by chance), tested against 5% with a Wilcoxon signed-rank test
across mice; LMI+ and LMI- fractions are also tested against 2.5% each
(stats file only). The histograms pool cells for illustration.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from fast_learning import paths, database
from fast_learning import imaging
from fast_learning.participation import LMI_NEGATIVE_THRESHOLD as LMI_NEGATIVE_P
from fast_learning.participation import LMI_POSITIVE_THRESHOLD as LMI_POSITIVE_P
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p


# ============================================================================
# Parameters
# ============================================================================

RESPONSE_WIN = (0, 0.300)
BASELINE_WIN = (-1, 0)
N_SHUFFLES = 100
DAYS = ['-2', '-1', '0', '+1', '+2']

PROCESSED_DATA_DIR = paths.processed_dir
NULL_LMI_CSV = os.path.join(PROCESSED_DATA_DIR, 'lmi_null_shuffles.csv')
LMI_RESULTS_CSV = os.path.join(paths.processed_dir, 'lmi_results.csv')
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_3f_LMIshuffles')


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Null distribution computation
# ============================================================================


def compute_null_lmi_distribution(n_shuffles=N_SHUFFLES):
    """Recompute LMI shuffles for every mouse, pooling per-cell per-shuffle
    null LMI values into one long-format DataFrame.

    Mirrors stats_on_tensors.py's "Compute LMI" section exactly (same data
    source, windows, pre/post day pooling); the only difference is
    return_shuffles=True to keep the full per-shuffle null LMI values instead
    of only the significance percentile.

    Saves NULL_LMI_CSV.
    """
    db_path = paths.db_path
    nwb_path = paths.nwb_dir

    _, _, mice_list, _ = database.select_sessions_from_db(
        db_path,
        nwb_path,
        exclude_cols=['exclude', 'two_p_exclude'],
        experimenters=['AR', 'GF', 'MI'],
        day=DAYS,
        two_p_imaging='yes',
    )

    rows = []
    for mouse_id in mice_list:
        print(f'Processing {mouse_id}')
        reward_group = database.get_mouse_reward_group_from_db(db_path, mouse_id)

        # Same loader as 05_lmi.py, so artefact cells are excluded.
        data_mapping = imaging.load_mouse_xarray(
            mouse_id, paths.tensor_dir, 'tensor_xarray_mapping_data.nc', subtracted=False
        )
        data_mapping = data_mapping - np.nanmean(
            imaging.select_time(data_mapping, *BASELINE_WIN), axis=2, keepdims=True
        )

        data_pre = data_mapping.sel(trial=data_mapping.coords['day'].isin([-2, -1]))
        data_pre = imaging.select_time(data_pre, *RESPONSE_WIN).mean(dim='time')
        data_post = data_mapping.sel(trial=data_mapping.coords['day'].isin([1, 2]))
        data_post = imaging.select_time(data_post, *RESPONSE_WIN).mean(dim='time')

        _, _, lmi_shuffles = imaging.compute_roc(
            data_pre, data_post, nshuffles=n_shuffles, return_shuffles=True
        )

        # lmi_shuffles shape: (n_cells, n_shuffles) -- pool to long format.
        rows.append(
            pd.DataFrame(
                {
                    'mouse_id': mouse_id,
                    'reward_group': reward_group,
                    'null_lmi': lmi_shuffles.ravel(),
                }
            )
        )

    null_df = pd.concat(rows, ignore_index=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    null_df.to_csv(NULL_LMI_CSV, index=False)
    print(f"Saved: {NULL_LMI_CSV}  ({len(null_df)} rows)")
    return null_df


def load_null_lmi_distribution():
    if not os.path.exists(NULL_LMI_CSV):
        raise FileNotFoundError(
            f"Pre-computed null LMI data not found: {NULL_LMI_CSV}\nRun with --recompute first."
        )
    df = pd.read_csv(NULL_LMI_CSV)
    print(f"Loaded: {NULL_LMI_CSV}  ({len(df)} rows)")
    return df


def load_real_lmi():
    """Load real LMI values and assign reward groups (mirrors figure_3f_g.py)."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    lmi_mice = set(lmi_df['mouse_id'].unique())
    _, _, mice, _ = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    print(f"  lmi_results.csv mice ({len(lmi_mice)}): {sorted(lmi_mice)}")
    print(f"  DB two_p_imaging=='yes' mice ({len(mice)}): {sorted(mice)}")
    print(f"  Overlap: {len(lmi_mice & set(mice))} mice")
    for mouse in lmi_df['mouse_id'].unique():
        lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = database.get_mouse_reward_group_from_db(
            paths.db_path, mouse
        )
    lmi_df = lmi_df.loc[lmi_df['mouse_id'].isin(mice)]
    print(
        f"  lmi_df after DB filter: {len(lmi_df)} rows, "
        f"reward_group counts: {lmi_df['reward_group'].value_counts(dropna=False).to_dict()}"
    )
    return lmi_df


# ============================================================================
# Plot
# ============================================================================


def significant_fractions(lmi_df):
    """Per mouse: fraction of cells with a significant LMI (shuffle test,
    lmi_p >= 0.975 or <= 0.025, i.e. 5% expected by chance), and of LMI+ and
    LMI- cells (2.5% each expected)."""
    d = lmi_df.dropna(subset=['lmi_p'])
    return (
        d.assign(
            positive=d['lmi_p'] >= LMI_POSITIVE_P,
            negative=d['lmi_p'] <= LMI_NEGATIVE_P,
        )
        .assign(significant=lambda x: x['positive'] | x['negative'])
        .groupby(['reward_group', 'mouse_id'])[['significant', 'positive', 'negative']]
        .mean()
        .reset_index()
    )


def fraction_stats(fractions):
    """Wilcoxon signed-rank test (n = mice) of each fraction against chance."""
    rows = []
    for rg in ['R+', 'R-']:
        g = fractions[fractions['reward_group'] == rg]
        for col, chance in [('significant', 0.05), ('positive', 0.025), ('negative', 0.025)]:
            stat, p = wilcoxon(g[col] - chance)
            rows.append(
                {
                    'reward_group': rg,
                    'cells': {'significant': 'LMI+ or LMI-', 'positive': 'LMI+', 'negative': 'LMI-'}[col],
                    'test': f'Wilcoxon signed-rank of per-mouse fraction vs {chance:.1%} expected by chance',
                    'n_mice': len(g),
                    'chance': chance,
                    'mean_fraction': g[col].mean(),
                    'median_fraction': g[col].median(),
                    'n_mice_above_chance': int((g[col] > chance).sum()),
                    'statistic': stat,
                    'p_value': p,
                }
            )
    return pd.DataFrame(rows)


def plot_lmi_vs_null(lmi_df, null_df, output_dir=OUTPUT_DIR, filename='supp_3_LMI_shuffles'):
    """Real LMI distribution overlaid with the pooled shuffled-null LMI
    distribution, one panel per reward group, and the per-mouse fraction of
    LMI-significant cells against the 5% expected by chance.

    The histograms pool cells for illustration; the statistic is per mouse
    (fraction_stats).

    Saves:
        <filename>.pdf
        <filename>_stats.csv  -- per-group test of the fractions against chance
        <filename>_data.csv   -- per-mouse fractions
    """
    set_style()
    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    bin_edges = np.linspace(-1, 1, 31)

    fractions = significant_fractions(lmi_df)
    stats_df = fraction_stats(fractions)

    fig, axes = plt.subplots(1, 3, figsize=panel_size(3))
    for ax, rg in zip(axes[:2], reward_groups):
        real_vals = lmi_df.loc[lmi_df['reward_group'] == rg, 'lmi'].dropna().values
        null_vals = null_df.loc[null_df['reward_group'] == rg, 'null_lmi'].dropna().values
        sns.histplot(
            null_vals,
            bins=bin_edges,
            stat='probability',
            element='step',
            fill=False,
            color='dimgray',
            linewidth=0.8,
            label='Shuffled null',
            ax=ax,
        )
        sns.histplot(
            real_vals,
            bins=bin_edges,
            stat='probability',
            color=rg_colors[rg],
            alpha=0.5,
            linewidth=0,
            label='Real LMI',
            ax=ax,
        )
        ax.set_title(f'{rg}  ({len(real_vals)} cells)')
        ax.set_xlim(-1, 1)
        ax.set_xlabel('LMI')
        ax.set_ylabel('Probability' if rg == 'R+' else '')
        ax.legend(frameon=False)

    ax = axes[2]
    long = fractions.assign(percent=100 * fractions['significant'])
    sns.barplot(
        data=long,
        x='reward_group',
        y='percent',
        order=reward_groups,
        hue='reward_group',
        palette=rg_colors,
        legend=False,
        errorbar=('ci', 95),
        seed=0,
        alpha=0.7,
        edgecolor='black',
        ax=ax,
    )
    sns.stripplot(data=long, x='reward_group', y='percent', order=reward_groups, color='black', size=2, ax=ax)
    ax.axhline(5, color='grey', linestyle='--', linewidth=0.6)
    top = long['percent'].max()
    for i, rg in enumerate(reward_groups):
        p = stats_df.query('reward_group == @rg and cells == "LMI+ or LMI-"')['p_value'].iloc[0]
        ax.text(i, top * 1.05, format_p(p), ha='center', va='bottom')
    ax.set_ylim(0, top * 1.2)
    ax.set_xlabel('')
    ax.set_ylabel('LMI-significant cells (% per mouse)')
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    stats_df.to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    fractions.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, filename)}.pdf, _stats.csv, _data.csv")
    return stats_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real vs shuffled-null LMI distributions.')
    parser.add_argument(
        '--recompute', action='store_true', help='recompute the null LMI distribution even if it is cached'
    )
    args = parser.parse_args()
    print(f"Output directory: {OUTPUT_DIR}")

    if args.recompute or not os.path.exists(NULL_LMI_CSV):
        null_df = compute_null_lmi_distribution()
    else:
        null_df = load_null_lmi_distribution()

    lmi_df = load_real_lmi()
    stats_df = plot_lmi_vs_null(lmi_df, null_df)
    print("\n=== Real vs. shuffled-null LMI spread, per reward group ===")
    print(stats_df.to_string(index=False))
