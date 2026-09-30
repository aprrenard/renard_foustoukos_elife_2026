"""Reactivation analysis helpers.

Mouse selection for the participation analyses
----------------------------------------------
Participation rates are computed per reactivation event, so a mouse with
almost no events on day 0 has undefined or unreliable day-0 rates (with one
event, every cell's rate is exactly 0 or 1). Mice with fewer than
MIN_DAY0_EVENTS events on day 0, at the main detection threshold, are
therefore excluded from every participation analysis (Fig. 4i-j, Supp. 4a-c
and their revisions). Event-rate analyses (Fig. 4h and its controls) keep all
mice: there, zero events is a valid measurement, and excluding mice on it
would select on the outcome.

Within the selected mice, a mouse-day enters the per-day participation
statistics only if it has at least MIN_EVENTS_PER_DAY valid events (events
closer than 150 ms to a trial edge are not used). This is the same threshold,
so every day-0 panel uses exactly the selected mice.

reactivation_preprocessing.py writes the selection to
<processed_dir>/reactivation/mouse_selection.csv; the participation scripts
read it with load_participation_mice().
"""

import os

import pandas as pd

from fast_learning import paths

MIN_DAY0_EVENTS = 3
MIN_EVENTS_PER_DAY = MIN_DAY0_EVENTS
SELECTION_DAY = 0
MOUSE_SELECTION_CSV = os.path.join(paths.processed_dir, 'reactivation', 'mouse_selection.csv')


def compute_mouse_selection(results_data, min_events=MIN_DAY0_EVENTS, day=SELECTION_DAY):
    """One row per mouse: reward group, number of events on `day`, included.

    results_data is the content of a reactivation_results_p*.pkl file.
    A day missing from a mouse's results counts as zero events.
    """
    rows = []
    for key, group in [('r_plus_results', 'R+'), ('r_minus_results', 'R-')]:
        for mouse, res in results_data[key].items():
            n_events = int(res.get('days', {}).get(day, {}).get('total_events', 0))
            rows.append({'mouse_id': mouse, 'reward_group': group,
                         f'n_events_day{day}': n_events,
                         'included': n_events >= min_events})
    return pd.DataFrame(rows)


def load_participation_mice(path=MOUSE_SELECTION_CSV):
    """Set of mouse ids included in the participation analyses."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mouse selection not found: {path}\n"
            "Run reactivation_preprocessing.py (or its --selection-only option) first.")
    df = pd.read_csv(path)
    return set(df.loc[df['included'], 'mouse_id'])
