"""
Density class ..
"""

from abc import ABC, abstractmethod


class Density(ABC):

    def __init__(self):
        super().__init__()
        self.manifold = None

    @abstractmethod
    def fit(self, x):
        """
        Fit method, x should be the training data that the manifold/density will be estimated.
        :param x: data
        :return: self
        """
        pass

    @abstractmethod
    def sample_radius(self, x_exp, r=None, n_samples=1, random_state=None):
        """
        Sample data from the manifold/density at the locality of x_exp, a metric should be given.
        :param x_exp: array with point/instance where the neighborhood will be selected.
        :return: array with sampled data.
        """
        pass

    @abstractmethod
    def sample(self, n_samples=1, random_state=None):
        """
        Sample data from the manifold/density.
        :param n_samples:
        :param random_state:
        :return: array with sampled data.
        """
        pass
