from abc import ABC, abstractmethod

import copy
import numpy as np
from sklearn.preprocessing import StandardScaler

class LocalModelBase(ABC):
    """
    Base class to implement the local models.
    """

    def __init__(
        self,
        x_explain,
        chi_explain,
        y_p_explain,
        feature_names,
        target_names,
        class_index,
        r,
        tol_importance,
        tol_error,
        scale_data=False,
        save_samples=False,
    ):
        self.x_explain = x_explain
        self.chi_explain = chi_explain
        self.y_p_explain = y_p_explain
        self.feature_names = feature_names
        self.target_names = target_names
        self.class_index = class_index
        self.tol_importance = tol_importance
        self.tol_error = tol_error
        self.previous_convergence = None
        self.n_previous_convergence = None
        self.convergence_diffs = []
        self.erros_training = []
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
        if scale_data:
            self.scaler = StandardScaler()
        else:
            self.scaler = IdentityScaler()

    def measure_convergence(self, chi_set, y_true):
        y_p_local_model = self.model.predict(chi_set)
        # Difference of the importance.
        diff = self.measure_importances()
        # Error specific for the local model.
        error = self.measure_errors(y_true, y_p_local_model)
        self.erros_training.append(error)
        # Samples Min/Max
        self.min_max_predictions(y_p_local_model, y_p_black_box_model=y_true)
        # Test convergence.
        if diff is None:
            self.convergence = False
        elif error <= self.tol_error and diff < self.tol_importance:
            self.convergence = True
        else:
            self.convergence = False
        return diff, error, self.convergence

    @abstractmethod
    def measure_errors(self, y_true, y_p_local_model):
        raise NotImplementedError

    @abstractmethod
    def measure_importances(self):
        raise NotImplementedError

    def min_max_predictions(self, y_p_local_model=None, y_p_black_box_model=None):
        # Max and Min y_p_local_model_max value and y_p_max: y_true
        if y_p_black_box_model is not None:
            y_p_max = np.max(y_p_black_box_model)
            y_p_min = np.min(y_p_black_box_model)
            if self.y_p_max is None:
                self.y_p_max = y_p_max
                self.y_p_min = y_p_min
            else:
                self.y_p_max = np.max([self.y_p_max, y_p_max])
                self.y_p_min = np.min([self.y_p_min, y_p_min])

        if y_p_local_model is not None:
            y_p_local_model_max = np.max(y_p_local_model)
            y_p_local_model_min = np.min(y_p_local_model)
            if self.y_p_local_model_max is None:
                self.y_p_local_model_max = y_p_local_model_max
                self.y_p_local_model_min = y_p_local_model_min
            else:
                self.y_p_local_model_max = np.max([self.y_p_local_model_max, y_p_local_model_max])
                self.y_p_local_model_min = np.min([self.y_p_local_model_min, y_p_local_model_min])

    @property
    @abstractmethod
    def importance(self):
        raise NotImplementedError

    def partial_fit(self, x_set, y_set, weight_set=None):
        if self.save_samples:
            if self.x_samples is None:
                self.weight_samples = weight_set
                self.x_samples = x_set
                self.y_samples = y_set
            else:
                if weight_set is not None:
                    self.weight_samples = np.append(self.weight_samples, weight_set, axis=0)
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
            self.feature_names = [f"feature {e}" for e in range(len(self.importance))]

        x_explain = np.array(self.x_explain)
        chi_explain = np.array(self.chi_explain).reshape(1, -1)

        y_p = self.predict(chi_explain)
        if y_p is not None:
            y_p = y_p[0]
        explanation["chi_names"] = self.feature_names
        explanation["chi_values"] = chi_explain
        # explanation["feature_names"] = self.feature_names
        # explanation["features"] = chi_explain
        explanation["x_names"] = x_explain
        explanation["x_values"] = x_explain
        
        explanation["y_p"] = self.y_p_explain
        explanation["y_p_max"] = self.y_p_max
        explanation["y_p_min"] = self.y_p_min
        explanation["y_p_local_model"] = y_p
        explanation["y_p_local_model_max"] = self.y_p_local_model_max
        explanation["y_p_local_model_min"] = self.y_p_local_model_min
        explanation["error"] = self.erros_training[-1]

        explanation["importances"] = self.importance
        explanation["diff_convergence_importances"] = self.convergence_diffs[-1]
        explanation["index_class"] = self.class_index
        explanation["class_names"] = self.target_names
        return explanation


class IdentityScaler(StandardScaler):
    
    def transform(self, X, copy=None):
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        super().fit(X, y=y, **fit_params)
        return X