import numpy as np
# import statsmodels.stats.weightstats as smws


def llincc(x, y):
    """
    Calculates Lin's concordance correlation coefficient.

    Usage:   llincc(x,y)    where x, y are equal-length arrays

    Args:
        x: A numpy array
        y: A second numpy array to compare against

    Returns:
        Lin's CC
    """
    covar = (np.cov(x, y)*(len(x)-1)/float(len(x)))[0, 1]  # correct denominator to n
    xvar = np.var(x)*(len(x)-1)/float(len(x))  # correct denominator to n
    yvar = np.var(y)*(len(y)-1)/float(len(y))  # correct denominator to n
    lincc = (2 * covar) / ((xvar+yvar) + ((np.mean(x)-np.mean(y))**2))
    return lincc


# def tost(a, b, dx=-0.5, dy=0.5):
#     """
#     Runs a Two One Sided T-test to test for similarity
#
#     Args:
#         a: Input array 1
#         b: Input array 2
#         dx: delta x
#         dy: delta y
#
#     Returns:
#         The resulting tost value
#     """
#     return smws.ttost_ind(a, b, dx, dy)[0]
