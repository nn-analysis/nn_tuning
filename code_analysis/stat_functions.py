import numpy as np
import statsmodels.stats.weightstats as smws


def llincc(x, y):
    """
    Calculates Lin's concordance correlation coefficient.

    Usage:   llincc(x,y)    where x, y are equal-length arrays
    Returns: Lin's CC
    """
    covar = (np.cov(x, y)*(len(x)-1)/float(len(x)))[0, 1]  # correct denom to n
    xvar = np.var(x)*(len(x)-1)/float(len(x))  # correct denom to n
    yvar = np.var(y)*(len(y)-1)/float(len(y))  # correct denom to n
    lincc = (2 * covar) / ((xvar+yvar) + ((np.mean(x)-np.mean(y))**2))
    return lincc


def tost(a, b, dx=-0.5, dy=0.5):
    return smws.ttost_ind(a, b, dx, dy)[0]
