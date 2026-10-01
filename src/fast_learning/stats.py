"""Statistics helpers shared by the figure scripts."""

import math


def format_p(p, prefix='p='):
    """P-value label for figures: 'p=0.051' (three decimals) from 0.001 up,
    'p=5×10$^{-4}$' (one significant digit, rendered as a superscript) below.

    A NaN p-value gives 'n.a.'.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n.a.'
    if p >= 0.001:
        return f'{prefix}{p:.3f}'
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


def lmm_slope(df, y, x, group='mouse_id'):
    """Mixed model y ~ x + (1 | group): the fixed-effect slope of x with cells
    (or cell pairs) grouped by mouse, so that cells recorded in the same mouse
    are not treated as independent.

    Returns a dict with slope, se, CI, p-value (Wald), intercept, the
    intraclass correlation of the mouse term and the method used.
    """
    result, re_var, resid_var, method = _mixed_fit(f'{y} ~ {x}', df, group)
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

    Returns None with fewer than 2 mice having at least 2 days.
    """
    import numpy as np
    from scipy.stats import linregress, wilcoxon

    slopes = []
    for _, mdata in mouse_day_data.groupby('mouse_id'):
        mdata = mdata.dropna(subset=[y])
        if mdata[x].nunique() >= 2:
            slopes.append(linregress(mdata[x], mdata[y]).slope)
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
    }
