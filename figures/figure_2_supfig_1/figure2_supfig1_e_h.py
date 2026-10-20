"""
Figure 2—figure supplement 1e-h: Muscimol inactivation of wS1 or fpS1 during
task execution, in expert mice

A separate cohort of mice trained to expert performance, with a muscimol day
(M 1), a Ringer control day and a second muscimol day (M 2) after two
baseline days. No optogenetic inactivation in this cohort.
- e, f: lick probability on whisker, auditory and catch trials across days,
  wS1 and fpS1 mice;
- g: whisker lick probability, wS1 vs fpS1 on each muscimol and Ringer day;
- h: d' (whisker vs catch) on the same days.
Bars: mean and bootstrapped 95% CI across mice; dots: mice. Mann-Whitney U
per day (n = mice). See fast_learning.inactivation.

Panels a-d (auditory performance during learning) are in
figure2_supfig1_a_d.py.

Outputs: <figures_dir>/Figure2_SupFig1/figure2_supfig1_e_h.pdf,
figure2_supfig1_e_h_data.csv (session), figure2_supfig1_g_stats.csv,
figure2_supfig1_h_stats.csv.
"""

import os

import matplotlib.pyplot as plt

from fast_learning import inactivation as inact
from fast_learning import paths
from fast_learning.plotting import panel_size, set_style

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure2_SupFig1')
COHORT = inact.MUSCIMOL_EXECUTION
TESTED = ['M 1', 'Ringer', 'M 2']


if __name__ == '__main__':
    sessions = inact.session_table(COHORT)
    stats = {y: inact.area_stats(sessions, y, TESTED) for y in ('p_whisker', 'dprime')}

    set_style()
    fig, axes = plt.subplots(1, 4, figsize=panel_size(4, w=1.1))
    for ax, area in zip(axes[:2], inact.AREAS):
        inact.timecourse(ax, sessions, area, list(COHORT.labels), 'Muscimol during execution')
    for ax, y in zip(axes[2:], stats):
        inact.area_bars(ax, sessions, stats[y], TESTED, y, legend=ax is axes[-1])
    axes[-1].set_ylim(-0.2, 4)  # d' never below 0 here (minimum -0.02)
    axes[-1].set_yticks(range(5))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inact.save(fig, os.path.join(OUTPUT_DIR, 'figure2_supfig1_e_h.pdf'))

    sessions.to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig1_e_h_data.csv'), index=False)
    stats['p_whisker'].to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig1_g_stats.csv'), index=False)
    stats['dprime'].to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig1_h_stats.csv'), index=False)
    for y, s in stats.items():
        print(y, s[['day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']].round(4).to_string(index=False))
