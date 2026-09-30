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
