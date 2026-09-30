"""Statistics helpers shared by the figure scripts."""


def significance_stars(p):
    """'***' (p < 0.001), '**' (p < 0.01), '*' (p < 0.05) or 'n.s.'.

    NaN gives 'n.s.'; callers that need 'n.a.' for missing tests check for
    NaN first.
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'
