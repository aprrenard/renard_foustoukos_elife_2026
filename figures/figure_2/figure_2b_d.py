"""
Figure 2b-d: Muscimol inactivation of wS1 or fpS1 during learning

One row of plots:
- b: lick probability on whisker, auditory and catch trials across days,
  wS1 and fpS1 mice (muscimol on Days 0, +1, +2: M 1-3; recovery days R 1-3);
- c: whisker lick probability on the muscimol days, wS1 vs fpS1;
- d: d' (whisker vs catch) on the same days, wS1 vs fpS1.
Bars: mean and bootstrapped 95% CI across mice; dots: mice. Mann-Whitney U
per day (n = mice). See fast_learning.inactivation.

Outputs: <figures_dir>/Figure2/figure_2b_d.pdf, figure_2b_data.csv (session),
figure_2c_stats.csv, figure_2d_stats.csv.
"""

import os

import matplotlib.pyplot as plt

from fast_learning import inactivation as inact
from fast_learning import paths
from fast_learning.plotting import panel_size, set_style

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure2')
COHORT = inact.MUSCIMOL_LEARNING
TESTED = ['M 1', 'M 2', 'M 3']  # Days 0, +1, +2
TESTED_LABELS = ['0', '+1', '+2']
XLABEL = 'Muscimol inactivation during learning'
NAME = ('figure_2b_d', 'figure_2b', 'figure_2c', 'figure_2d')


def generate_figure(cohort=COHORT, tested=TESTED, tested_labels=TESTED_LABELS, xlabel=XLABEL, names=NAME):
    sessions = inact.session_table(cohort)
    stats = {y: inact.area_stats(sessions, y, tested) for y in ('p_whisker', 'dprime')}
    shown = sessions.assign(
        day_label=sessions['day_label'].cat.rename_categories(
            {t: lab for t, lab in zip(tested, tested_labels)}
        )
    )

    set_style()
    widths = [len(cohort.days), len(cohort.days), 2 + len(tested), 2 + len(tested)]
    fig, axes = plt.subplots(1, 4, figsize=panel_size(4, w=1.2), gridspec_kw={'width_ratios': widths})
    for ax, area in zip(axes[:2], inact.AREAS):
        inact.timecourse(ax, sessions, area, list(cohort.labels), xlabel)
    for ax, y in zip(axes[2:], stats):
        s = stats[y].replace({'day': dict(zip(tested, tested_labels))})
        inact.area_bars(ax, shown, s, tested_labels, y, legend=ax is axes[-1])
        ax.set_xlabel('Day')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inact.save(fig, os.path.join(OUTPUT_DIR, f'{names[0]}.pdf'))

    sessions.to_csv(os.path.join(OUTPUT_DIR, f'{names[1]}_data.csv'), index=False)
    stats['p_whisker'].to_csv(os.path.join(OUTPUT_DIR, f'{names[2]}_stats.csv'), index=False)
    stats['dprime'].to_csv(os.path.join(OUTPUT_DIR, f'{names[3]}_stats.csv'), index=False)
    for y, s in stats.items():
        print(y, s[['day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']].round(4).to_string(index=False))


if __name__ == '__main__':
    generate_figure()
