"""
Revision: whisker, tongue and jaw movement during the passive whisker trials,
across days (R2 public 6).

Addresses the concern that the state of the mice during the passive epoch at
the end of each session might differ across days. Complements
behavior_state_summary.py (trials, session duration, water).

Data: DeepLabCut trial matrices exported by G. Foustoukos, one
'<mouse>_dlc.h5' file per mouse in DLC_DIR, for 12 mice (11 R+, one R-, so
only R+ mice are analysed). Each file holds, for every whisker trial after
disengagement with usable video (no lick), the traces of three movement
speeds, (n_trials, 3, 700) at 100 Hz with the stimulus at 0 s (-2 to 5 s):
    whisker_angular_speed  |d theta / dt| of the whisker (base to tip), deg/s
    tongue_speed           speed of the tongue, pixels/s
    jaw_speed              speed of the jaw, pixels/s

Per mouse and day, trials are averaged. Figures:
    movement_traces.pdf    mean speed across the trial (rows: speeds,
                           columns: days); mean and 95% CI across mice.
    movement_summary.pdf   mean speed in the post-stimulus window WINDOW, per
                           day; bars: mean across mice with bootstrapped 95%
                           CI, lines: mice. Friedman test across the five days
                           (repeated measures, mice with all five days).

Outputs: <figures_dir>/revisions/movement_state_summary/output/
    movement_traces.pdf, movement_summary.pdf,
    movement_data.csv (mouse x day x speed), movement_traces_data.csv
    (mouse x day x speed x time), movement_stats.csv.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
from scipy.stats import t as t_dist

from fast_learning import database, paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p

DLC_DIR = os.path.join(paths.dlc_dir, 'Speeds', 'DLCTrialMatrices')
DAYS = [-2, -1, 0, 1, 2]
WINDOW = (0.0, 1.0)  # s after stimulus onset
MIN_TRIALS = 5  # per mouse and day
SPEEDS = {
    'whisker_angular_speed': 'Whisker angular speed',
    'tongue_speed': 'Tongue speed',
    'jaw_speed': 'Jaw speed',
}
COLOR = reward_palette[1]
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'movement_state_summary', 'output')


# ============================================================================
# Data
# ============================================================================


def _text(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_mouse(path):
    """(traces by day {day: (n_trials, n_speeds, n_time)}, speed names, units,
    time) of one mouse."""
    with h5py.File(path, 'r') as h:
        traces = h['traces'][...]
        names = _text(h['feature_names'][...])
        units = _text(h['feature_units'][...])
        time = h['time'][...]
        days = h['trials/day_offset'][...]
    # Edge artefact of the speed traces: first and last samples take their
    # neighbour's value.
    traces[:, :, 0] = traces[:, :, 1]
    traces[:, :, -1] = traces[:, :, -2]
    return {d: traces[days == d] for d in DAYS if (days == d).sum() >= MIN_TRIALS}, names, units, time


def build_datasets():
    """Per mouse x day x speed: mean trace and mean in WINDOW (R+ mice)."""
    db = database.read_excel_db(paths.db_path)
    files = sorted(f for f in os.listdir(DLC_DIR) if f.endswith('_dlc.h5'))
    traces, means, skipped = [], [], []
    units = {}
    for f in files:
        mouse = f[: -len('_dlc.h5')]
        if database.get_mouse_reward_group_from_db(paths.db_path, mouse, db=db) != 'R+':
            skipped.append(mouse)
            continue
        by_day, names, unit_list, time = load_mouse(os.path.join(DLC_DIR, f))
        units.update(zip(names, unit_list))
        window = (time >= WINDOW[0]) & (time < WINDOW[1])
        for day, tr in by_day.items():
            mean_trace = tr.mean(axis=0)  # (n_speeds, n_time)
            for i, name in enumerate(names):
                traces.append(
                    pd.DataFrame(dict(mouse_id=mouse, day=day, speed=name, time=time, value=mean_trace[i]))
                )
                means.append(
                    dict(
                        mouse_id=mouse,
                        day=day,
                        speed=name,
                        n_trials=len(tr),
                        value=mean_trace[i, window].mean(),
                    )
                )
    print(f"R+ mice: {len({m['mouse_id'] for m in means})}; not R+: {skipped}")
    return pd.DataFrame(means), pd.concat(traces, ignore_index=True), units


# ============================================================================
# Statistics
# ============================================================================


def friedman_stats(means):
    rows = []
    for speed in SPEEDS:
        wide = means[means['speed'] == speed].pivot(index='mouse_id', columns='day', values='value')
        wide = wide.reindex(columns=DAYS).dropna()
        stat, p = friedmanchisquare(*(wide[d] for d in DAYS)) if len(wide) >= 3 else (np.nan, np.nan)
        rows.append(
            dict(
                speed=speed,
                window=f'{WINDOW[0]:g}-{WINDOW[1]:g} s',
                test='Friedman across days (n = mice with all days)',
                n_mice=len(wide),
                **{f'mean_day_{d:+d}': wide[d].mean() for d in DAYS},
                statistic=stat,
                p_value=p,
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================


def _mean_ci(values, ci=0.95):
    """Mean and t-based CI across mice at each time point (bootstrapping 700
    time points would be slow)."""
    g = values.groupby('time')['value']
    mean, sem, n = g.mean(), g.sem(), g.count()
    half = t_dist.ppf((1 + ci) / 2, np.maximum(n - 1, 1)) * sem
    return mean.index.values, mean.values, (mean - half).values, (mean + half).values


def plot_traces(traces, units, filename='movement_traces'):
    fig, axes = plt.subplots(
        len(SPEEDS), len(DAYS), figsize=panel_size(len(DAYS), len(SPEEDS), w=0.8), sharex=True
    )
    for i, (speed, label) in enumerate(SPEEDS.items()):
        for j, day in enumerate(DAYS):
            ax = axes[i, j]
            d = traces[(traces['speed'] == speed) & (traces['day'] == day)]
            t, m, lo, hi = _mean_ci(d)
            ax.plot(t, m, color=COLOR)
            ax.fill_between(t, lo, hi, color=COLOR, alpha=0.3, linewidth=0)
            ax.axvline(0, color='#FF9600', linewidth=0.8)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f'Day {day:+d}'.replace('+0', '0'))
            if j == 0:
                ax.set_ylabel(f'{label}\n({units[speed]})')
            if i == len(SPEEDS) - 1:
                ax.set_xlabel('Time (s)')
        ymax = max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_ylim(0, ymax)
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


def plot_summary(means, stats, units, filename='movement_summary'):
    fig, axes = plt.subplots(1, len(SPEEDS), figsize=panel_size(len(SPEEDS)))
    for ax, (speed, label) in zip(axes, SPEEDS.items()):
        d = means[means['speed'] == speed]
        for _, m in d.groupby('mouse_id'):
            m = m.set_index('day').reindex(DAYS)
            ax.plot(range(len(DAYS)), m['value'], color='grey', linewidth=0.5, alpha=0.6)
        sns.barplot(
            data=d, x='day', y='value', order=DAYS, color=COLOR, alpha=0.7, errorbar=('ci', 95), seed=0, ax=ax
        )
        p = stats.loc[stats['speed'] == speed, 'p_value'].iloc[0]
        ax.set_title(f'Friedman {format_p(p)}')
        ax.set_xticks(range(len(DAYS)), [f'{d:+d}'.replace('+0', '0') for d in DAYS])
        ax.set_xlabel('Day')
        ax.set_ylabel(f'{label}\n({units[speed]}, {WINDOW[0]:g}-{WINDOW[1]:g} s)')
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    means, traces, units = build_datasets()
    stats = friedman_stats(means)
    means.to_csv(os.path.join(OUTPUT_DIR, 'movement_data.csv'), index=False)
    traces.to_csv(os.path.join(OUTPUT_DIR, 'movement_traces_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'movement_stats.csv'), index=False)
    plot_traces(traces, units)
    plot_summary(means, stats, units)
    print(stats.round(3).to_string(index=False))
