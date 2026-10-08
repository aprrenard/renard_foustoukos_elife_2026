import numpy as np
import pandas as pd

from fast_learning import participation as pt


def test_participation_from_3d():
    n_cells, n_trials, n_t = 3, 2, 60
    data = np.zeros((n_cells, n_trials, n_t))
    win = pt.EVENT_WINDOW_FRAMES
    events = [20, n_t + 30, 2]  # trial 0 t=20, trial 1 t=30, one too close to the edge
    data[0, 0, 20 - win : 20 + win + 1] = 0.5  # cell 0 active at event 1 only
    data[1, :, :] = 0.5  # cell 1 always active
    rates, n_valid = pt.participation_from_3d(data, events, n_t, n_trials)
    assert n_valid == 2
    np.testing.assert_allclose(rates, [0.5, 1.0, 0.0])


def test_participation_from_3d_no_valid_event():
    rates, n_valid = pt.participation_from_3d(np.zeros((2, 1, 30)), [0, 29], 30, 1)
    assert rates is None and n_valid == 0


def test_participation_rate_and_reliability():
    responses = pd.DataFrame(
        {'mouse_id': 'M', 'day': 0, 'roi': [1, 1, 1, 2, 2], 'participates': [True, False, True, True, True]}
    )
    rates = pt.compute_participation_rate(responses).set_index('roi')
    assert rates.loc[1, 'participation_rate'] == 2 / 3
    assert rates.loc[1, 'reliable'] == (3 >= pt.MIN_EVENTS_FOR_RELIABILITY)
    assert rates.loc[2, 'n_events'] == 2 and not rates.loc[2, 'reliable']


def test_aggregate_across_days():
    per_day = pd.DataFrame(
        {
            'mouse_id': 'M',
            'roi': 1,
            'day': [-2, -1, 0, 1, 2],
            'participation_rate': [0.1, 0.3, 0.5, 0.6, 0.8],
            'reliable': True,
        }
    )
    agg = pt.aggregate_across_days(per_day).iloc[0]
    assert agg['baseline_rate'] == 0.2
    assert agg['learning_rate'] == 0.5
    assert agg['post_rate'] == 0.7
    assert np.isclose(agg['delta_post'], 0.5)


def test_lmi_category():
    df = pt.add_lmi_category(pd.DataFrame({'lmi_p': [0.99, 0.5, 0.01, 0.975, 0.025]}))
    assert list(df['lmi_category']) == ['positive', 'neutral', 'negative', 'positive', 'negative']


def test_file_names():
    assert pt.thr_tag(0.1) == 'thr10' and pt.thr_tag(0.5) == 'thr50'
    assert pt.rates_csv(0.2).endswith('cell_participation_rates_per_day_thr20.csv')


def test_file_names_cells():
    assert pt.rates_csv(0.1, cells='all').endswith('thr10_allcells.csv')
    assert pt.merged_csv(0.1, nolick=True, cells='insample').endswith('thr10_insample_nolick.csv')
    assert pt.binary_csv(nolick=True).endswith('binary_participation_with_lmi_nolick.csv')


def test_split_halves():
    a, b = pt.split_halves('M', 11, 3)
    assert len(a) == 5 and len(b) == 6
    assert set(a).isdisjoint(b) and set(a) | set(b) == set(range(11))
    np.testing.assert_array_equal(a, pt.split_halves('M', 11, 3)[0])  # seeded
    assert not np.array_equal(a, pt.split_halves('M', 11, 4)[0])


def test_windowed_participation_matches_shifted_data():
    # Indexing the windowed trace at (event - shift) equals participation in
    # data rolled by shift, as in the all-cell circular-shift test.
    rng = np.random.default_rng(0)
    n_cells, n_trials, n_t = 4, 6, 40
    data = rng.normal(0.05, 0.1, (n_cells, n_trials, n_t))
    flat = data.reshape(n_cells, -1)
    n_frames = flat.shape[1]
    win = pt.EVENT_WINDOW_FRAMES
    events = np.array([tr * n_t + t for tr, t in [(0, 10), (2, 20), (3, win), (5, n_t - win - 1)]])
    part = pt.windowed_participation(flat)
    for shift in [0, 1, 7, 55, n_frames - 1]:
        rolled = np.roll(flat, shift, axis=1).reshape(n_cells, n_trials, n_t)
        expected, _ = pt.participation_from_3d(rolled, events, n_t, n_trials)
        np.testing.assert_allclose(part[:, (events - shift) % n_frames].mean(axis=1), expected)


def test_average_over_splits():
    m = pt.MIN_EVENTS_FOR_RELIABILITY
    split_df = pd.DataFrame(
        {
            'mouse_id': 'M',
            'day': 0,
            'roi': [1, 1, 1, 2],
            'split': [0, 1, 2, 0],
            'participation_rate': [0.2, 0.4, 0.9, 0.5],
            'n_events': [m, m + 2, m - 1, m - 1],  # split 2 of roi 1 and roi 2 unreliable
        }
    )
    out = pt.average_over_splits(split_df).set_index('roi')
    assert list(out.index) == [1]
    assert np.isclose(out.loc[1, 'participation_rate'], 0.3) and out.loc[1, 'n_splits'] == 2
