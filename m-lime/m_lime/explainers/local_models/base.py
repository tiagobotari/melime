from abc import ABC, abstractmethod

import numpy as np


class LocalModelBase(ABC):
    """
    Base class to implement the local models.
    """
    def __init__(self, x_explain, y_p_explain, features_names, tol_convergence):
        self.x_explain = x_explain
        self.y_p_explain = y_p_explain
        self.features_names = features_names
        self.tol_convergence = tol_convergence
        self.previous_convergence = None
        self.n_previous_convergence = None
        self.convergence = False
        self.convergence_diffs = []

    @property
    @abstractmethod
    def measure_convergence(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def importance(self):
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, x_set, y_set):
        raise NotImplementedError

    def _measure_convergence(self, values):
        diff = None
        if self.previous_convergence is None:
            self.previous_convergence = values
            self.n_previous_convergence = len(values)
        else:
            diff = np.sum(np.abs(self.previous_convergence - values)) / self.n_previous_convergence
            self.convergence_diffs.append(diff)
            self.previous_convergence = values
            if diff < self.tol_convergence:
                self.convergence = True
            

        return diff

