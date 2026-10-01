import numpy as np
import xarray as xr

from fast_learning import similarity
from fast_learning.stats import significance_stars


def test_significance_stars():
    assert [significance_stars(p) for p in (0.0001, 0.005, 0.03, 0.2)] == ['***', '**', '*', 'n.s.']
    assert significance_stars(0.05) == 'n.s.'
    assert significance_stars(0.2, ns='ns') == 'ns'
    assert significance_stars(float('nan'), na='n.a.') == 'n.a.'
    assert significance_stars(float('nan')) == 'n.s.'


def _vectors(n_cells=10):
    """Responses (cells x trials) for len(DAYS) days of N_MAP_TRIALS trials,
    with one fixed population pattern per day."""
    rng = np.random.default_rng(0)
    n = similarity.N_MAP_TRIALS
    patterns = rng.normal(size=(len(similarity.DAYS), n_cells))
    values = np.concatenate([np.repeat(p[:, None], n, axis=1) for p in patterns], axis=1)
    days = np.repeat(similarity.DAYS, n)
    return xr.DataArray(values, dims=['cell', 'trial'], coords={'day': ('trial', days)})


def test_similarity_matrix():
    v = _vectors()
    cm = similarity.compute_similarity_matrix(v, 'cosine')
    n = len(similarity.DAYS) * similarity.N_MAP_TRIALS
    assert cm.shape == (n, n)
    assert np.all(np.isnan(np.diag(cm)))
    # Identical trials within a day have cosine similarity 1.
    assert np.allclose(cm[0, 1 : similarity.N_MAP_TRIALS], 1)


def test_within_day_metrics():
    cm = similarity.compute_similarity_matrix(_vectors(), 'pearson')
    df = similarity.compute_within_day_metrics([cm], ['M'], 'R+')
    assert list(df['mouse_id']) == ['M']
    for day in similarity.DAYS:
        assert np.isclose(df[f'within_day{day:+d}'].iloc[0], 1)


def test_format_p():
    from fast_learning.stats import format_p

    assert format_p(0.0512) == 'p=0.051'
    assert format_p(0.00653) == 'p=0.007'
    assert format_p(0.0005) == 'p=5×10$^{-4}$'
    assert format_p(0.000564) == 'p=6×10$^{-4}$'
    assert format_p(3.2e-12) == 'p=3×10$^{-12}$'
    assert format_p(0.00096) == 'p=1×10$^{-3}$'
    assert format_p(float('nan')) == 'n.a.'
    assert format_p(0.2, prefix='KW p=') == 'KW p=0.200'


def test_lmm_accounts_for_mouse_clustering():
    import pandas as pd
    from fast_learning.stats import lmm_mean, lmm_slope

    rng = np.random.default_rng(0)
    # 8 mice whose offsets differ; no true effect within mice.
    rows = [
        (m, rng.normal(), off + rng.normal(0, 0.1))
        for m, off in enumerate(rng.normal(0, 1, 8))
        for _ in range(200)
    ]
    df = pd.DataFrame(rows, columns=['mouse_id', 'x', 'y'])
    fit = lmm_slope(df, 'y', 'x')
    assert fit['p_value'] > 0.01 and fit['icc_mouse'] > 0.9
    # Pooled pairs would make the mean change look highly significant; by mouse it is not.
    df['change'] = df['y']
    assert lmm_mean(df, 'change')['p_value'] > 0.01


def test_ks_permutation_respects_mice():
    import pandas as pd
    from scipy.stats import ks_2samp
    from fast_learning.stats import ks_permutation_test

    rng = np.random.default_rng(1)
    # 10 mice, 5 per group, strong mouse offsets but no group effect.
    rows = [
        (m, 'A' if m < 5 else 'B', off + rng.normal(0, 0.2))
        for m, off in enumerate(rng.normal(0, 1, 10))
        for _ in range(100)
    ]
    df = pd.DataFrame(rows, columns=['mouse_id', 'group', 'v'])
    pooled_p = ks_2samp(df[df.group == 'A'].v, df[df.group == 'B'].v).pvalue
    perm = ks_permutation_test(df, 'v', 'group', ('A', 'B'), 'between_mice', n_perm=500)
    assert pooled_p < 1e-6 and perm['p_value'] > 0.05
    # Within-mouse labels with a real shift: detected.
    df['ct'] = np.where(rng.random(len(df)) < 0.5, 'x', 'y')
    df.loc[df.ct == 'x', 'v'] += 0.3
    assert ks_permutation_test(df, 'v', 'ct', ('x', 'y'), 'within_mice', n_perm=200)['p_value'] < 0.01
