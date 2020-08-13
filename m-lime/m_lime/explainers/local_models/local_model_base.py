from abc import ABC, abstractmethod

import copy
import numpy as np


class LocalModelBase(ABC):
    """
    Base class to implement the local models.
    """
    def __init__(self, x_explain, y_p_explain, feature_names, r, tol_convergence, save_samples=False):
        self.x_explain = x_explain
        self.y_p_explain = y_p_explain
        self.feature_names = feature_names
        self.tol_convergence = tol_convergence
        self.previous_convergence = None
        self.n_previous_convergence = None
        self.convergence_diffs = []
        self.r = r
        # Samples
        self.save_samples = save_samples
        self.x_samples = None
        self.y_samples = None
        self.weight_samples = None
        # Sample max/min
        self.y_p_local_model_max = None
        self.y_p_local_model_min = None
        self.y_p_max = None
        self.y_p_min = None
        # Convergence Variable
        self.convergence = False

    @property
    @abstractmethod
    def measure_convergence(self, chi_set=None, y_true=None):
        raise NotImplementedError

    @property
    @abstractmethod
    def importance(self):
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, x_set, y_set, weight_set=None):
        if self.save_samples:
            if self.x_samples is None:
                self.weight_samples = weight_set
                self.x_samples = x_set
                self.y_samples = y_set
            else:
                if weight_set is not None:
                    self.weight_samples =  np.append(self.weight_samples, weight_set, axis=0) 
                self.x_samples = np.append(self.x_samples, x_set, axis=0)
                self.y_samples = np.append(self.y_samples, y_set, axis=0)
        
    def predict(self, x):
        raise NotImplementedError

    def _measure_convergence_importance(self, values):
        diff = None
        if self.previous_convergence is None:
            self.previous_convergence = copy.deepcopy(values)
            self.n_previous_convergence = len(values)
        else:
            diff = np.sum(np.abs(self.previous_convergence - values)) / self.n_previous_convergence
            self.convergence_diffs.append(diff)
            self.previous_convergence = copy.deepcopy(values)
  
        return diff

    def explain(self):
        explanation = {}
        if self.feature_names is None:
            self.feature_names = [f'feature {e}' for e in range(len(self.importance))]
        explanation["feature_names"] = self.feature_names
        explanation["features"] = self.x_explain.reshape(-1)
        explanation["y_p"] = self.y_p_explain
        explanation["y_p_max"] = self.y_p_max
        explanation["y_p_min"] = self.y_p_min
        explanation["y_p_local_model"] = self.predict(self.x_explain)
        explanation["y_p_local_model_max"] = self.y_p_local_model_max
        explanation["y_p_local_model_min"] = self.y_p_local_model_min

        explanation["importances"] = self.importance
        explanation["ind_class_sorted"] = 0
        explanation["class_names"] = ["taget"]
        return explanation