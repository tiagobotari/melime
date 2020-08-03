import numpy as np

from m_lime.explainers.local_models.base import LocalModelBase


class BasicStatistics(LocalModelBase):
    """
    Basic descriptive statistics for generating explanation.
    """
    def __init__(self, x_explain, y_p_explain, features_names, r=None, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.values = {}
        # self.values = {e: [] for e in self.features_names}

    def measure_convergence(self):
        return self._measure_convergence(self._coef_mean)

    @property
    def _coef_mean(self):
        results = self.calculate()
        return np.array([*results["mean"].values()])

    @property
    def importance(self):
        return self.calculate()

    def results(self):
        results = self.calculate()
        return results

    def partial_fit(self, x_set, y_set):
        # TODO: I need to improve this. Maybe put a matrix here with the positions.
        for x_i, y_i in zip(x_set, y_set):
            if x_i[0] in self.values:
                self.values[x_i[0]].append(y_i)
            else:
                self.values[x_i[0]] = [y_i]

    def calculate(self):
        mean = {}
        median = {}
        std = {}
        for key, values in self.values.items():
            if len(values) == 0:
                mean[key] = 0
                median[key] = 0
                std[key] = 0
                continue
            values_c = values - self.y_p_explain
            mean[key] = np.mean(values_c)
            median[key] = np.median(values_c)
            std[key] = np.std(values_c)
        return dict(mean=self.normalize_max(mean), median=self.normalize_max(median), std=self.normalize_max(std))

    @staticmethod
    def normalize_max(dict_in):
        max_values = np.max(np.abs([*dict_in.values()]))
        if max_values == 0.0:
            return dict_in
        dict_out = {key: value / max_values for key, value in dict_in.items()}
        return dict_out
