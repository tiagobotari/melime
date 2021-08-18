import numpy as np


def mean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def covariance(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - mean(x, w)) * (y - mean(y, w))) / np.sum(w)


def correlation(x, y, w):
    """Weighted Correlation"""
    return covariance(x, y, w) / np.sqrt(covariance(x, x, w) * covariance(y, y, w))
