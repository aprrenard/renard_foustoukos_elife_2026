"""
Figure 2e-g: Optogenetic inactivation of wS1 or fpS1 during learning

One row of plots, as Figure 2b-d:
- e: lick probability on whisker, auditory and catch trials across days, wS1
  and fpS1 mice (light on Day 0 in all trials, from 0.1 s before the stimulus
  to the end of the 1-s response window; no light on Day +1);
- f: whisker lick probability on Days 0 and +1, wS1 vs fpS1;
- g: d' (whisker vs catch) on the same days, wS1 vs fpS1.
Bars: mean and bootstrapped 95% CI across mice; dots: mice. Mann-Whitney U
per day (n = mice). See fast_learning.inactivation.

Outputs: <figures_dir>/Figure2/figure_2e_g.pdf, figure_2e_data.csv (session),
figure_2f_stats.csv, figure_2g_stats.csv.
"""

import importlib.util
import os

from fast_learning import inactivation as inact

# Row builder of Figure 2b-d (figure scripts are not a package).
_spec = importlib.util.spec_from_file_location(
    'figure_2b_d', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_2b_d.py')
)
_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_row)

if __name__ == '__main__':
    _row.generate_figure(
        cohort=inact.OPTO_LEARNING,
        tested=['0', '+1'],
        tested_labels=['0', '+1'],
        xlabel='Optogenetic inactivation during learning',
        names=('figure_2e_g', 'figure_2e', 'figure_2f', 'figure_2g'),
    )
