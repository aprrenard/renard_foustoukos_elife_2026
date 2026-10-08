"""Statistics helpers shared by the figure scripts."""

import math


def format_p(p, prefix='p='):
    """P-value label for figures: 'p=0.051' (three decimals) from 0.001 up,
    'p=5×10$^{-4}$' (one significant digit, rendered as a superscript) below,
    and 'p<10$^{-300}$' when the p-value underflows to 0.

    A NaN p-value gives 'n.a.'.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n.a.'
    if p >= 0.001:
        return f'{prefix}{p:.3f}'
    if p == 0:  # underflow: below the smallest representable double
        return prefix.replace('=', '<') + '10$^{-300}$'
    mantissa, exponent = f'{p:.0e}'.split('e')
    return f'{prefix}{mantissa}×10$^{{{int(exponent)}}}$'


def significance_stars(p, ns='n.s.', na=None):
    """'***' (p < 0.001), '**' (p < 0.01), '*' (p < 0.05), else `ns`.

    A NaN p-value gives `na` when it is set, and `ns` otherwise.
    """
    if na is not None and isinstance(p, float) and math.isnan(p):
        return na
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ns


def _mixed_fit(formula, df, group):
    """Fit a random-intercept mixed model. When the mouse variance is
    estimated at zero (boundary) its standard errors are undefined; the model
    then reduces to ordinary least squares, which is refitted with standard
    errors clustered by mouse. Returns (result, mouse variance, method)."""
    import warnings

    import numpy as np
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = MixedLM.from_formula(formula, groups=group, data=df).fit()
    re_var = float(result.cov_re.iloc[0, 0])
    if np.all(np.isfinite(result.bse)):
        return result, re_var, result.scale, 'LMM (1 | mouse)'
    groups = df[group].astype('category').cat.codes
    ols = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': groups})
    return ols, 0.0, ols.scale, 'OLS, mouse-clustered SE (mouse variance estimated at 0)'


def lmm_slope(df, y, x, group='mouse_id', covariates=()):
    """Mixed model y ~ x [+ covariates] + (1 | group): the fixed-effect slope
    of x with cells (or cell pairs) grouped by mouse, so that cells recorded in
    the same mouse are not treated as independent. With covariates, the slope
    of x is the one holding them fixed (the mixed-model analogue of a partial
    correlation).

    Returns a dict with slope, se, CI, p-value (Wald), intercept, the
    intraclass correlation of the mouse term and the method used.
    """
    formula = ' + '.join([f'{y} ~ {x}', *covariates])
    result, re_var, resid_var, method = _mixed_fit(formula, df, group)
    ci_low, ci_high = result.conf_int().loc[x]
    return {
        'slope': result.params[x],
        'se': result.bse[x],
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': result.pvalues[x],
        'intercept': result.params['Intercept'],
        'icc_mouse': re_var / (re_var + resid_var),
        'method': method,
    }


def lmm_mean(df, y, group='mouse_id'):
    """Mixed model y ~ 1 + (1 | group): tests whether the mean of y differs
    from zero with observations grouped by mouse (e.g. post - pre changes of
    cell pairs). Returns a dict like lmm_slope, for the intercept."""
    result, re_var, resid_var, method = _mixed_fit(f'{y} ~ 1', df, group)
    ci_low, ci_high = result.conf_int().loc['Intercept']
    return {
        'mean': result.params['Intercept'],
        'se': result.bse['Intercept'],
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': result.pvalues['Intercept'],
        'icc_mouse': re_var / (re_var + resid_var),
        'method': method,
    }


def per_mouse_slope_test(mouse_day_data, y='participation_rate', x='day'):
    """Linear trend across days tested with mice as the unit: one slope of y
    vs x per mouse (on that mouse's per-day means), then a Wilcoxon
    signed-rank test of the slopes against zero.

    Returns None with fewer than 2 mice having at least 2 days; otherwise
    the summary statistics and 'slopes' ({mouse: slope}).
    """
    import numpy as np
    from scipy.stats import linregress, wilcoxon

    slopes, mice = [], []
    for mouse, mdata in mouse_day_data.groupby('mouse_id'):
        mdata = mdata.dropna(subset=[y])
        if mdata[x].nunique() >= 2:
            slopes.append(linregress(mdata[x], mdata[y]).slope)
            mice.append(mouse)
    if len(slopes) < 2:
        return None
    slopes = np.array(slopes)
    try:
        w_stat, p_value = wilcoxon(slopes)
    except ValueError:  # e.g. all slopes zero
        w_stat, p_value = np.nan, np.nan
    return {
        'mean_slope': float(np.mean(slopes)),
        'median_slope': float(np.median(slopes)),
        'sd_slope': float(np.std(slopes, ddof=1)),
        'w_stat': float(w_stat),
        'p_value': float(p_value),
        'n_mice': len(slopes),
        'slopes': dict(zip(mice, slopes)),
    }


def ks_permutation_test(df, value, label, groups, shuffle, n_perm=10000, seed=0, mouse='mouse_id'):
    """Kolmogorov-Smirnov comparison of two groups of cells whose p-value
    respects that cells are recorded in mice.

    The statistic is the usual two-sample KS distance between the pooled cell
    values of groups[0] and groups[1]. Its null distribution is built by
    shuffling labels at the level where they are exchangeable under the null:
      shuffle='between_mice': the label is a property of the mouse (e.g. R+ vs
          R-); whole mice are reassigned to the groups, each keeping all its
          cells, with the number of mice per group fixed.
      shuffle='within_mice': the label is a property of the cell within mice
          (e.g. wS2 vs wM1); labels are shuffled among the cells of each
          mouse, so each mouse keeps its number of cells of each label.
    p = (1 + number of shuffles with a distance >= observed) / (1 + n_perm).
    """
    import numpy as np
    from scipy.stats import ks_2samp

    df = df[df[label].isin(groups)].dropna(subset=[value])
    values = df[value].to_numpy()
    labels = (df[label] == groups[0]).to_numpy()
    mice = df[mouse].to_numpy()

    def distance(is_first):
        return ks_2samp(values[is_first], values[~is_first]).statistic

    observed = distance(labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    if shuffle == 'between_mice':
        mouse_ids, inverse = np.unique(mice, return_inverse=True)
        mouse_label = np.zeros(len(mouse_ids), dtype=bool)
        mouse_label[inverse[labels]] = True
        if np.any(mouse_label[inverse] != labels):
            raise ValueError(f'{label} is not constant within mice.')
        for k in range(n_perm):
            null[k] = distance(rng.permutation(mouse_label)[inverse])
    elif shuffle == 'within_mice':
        blocks = [np.flatnonzero(mice == m) for m in np.unique(mice)]
        shuffled = labels.copy()
        for k in range(n_perm):
            for idx in blocks:
                shuffled[idx] = rng.permutation(labels[idx])
            null[k] = distance(shuffled)
    else:
        raise ValueError(shuffle)
    return {
        'ks_statistic': observed,
        'p_value': (1 + np.sum(null >= observed - 1e-12)) / (1 + n_perm),
        'n_perm': n_perm,
        'shuffle': shuffle,
        'n_cells': (int(labels.sum()), int((~labels).sum())),
        'n_mice': len(np.unique(mice)),
    }
