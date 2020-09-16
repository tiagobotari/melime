import numpy as np

from melime.explainers.local_models.local_model_base import LocalModelBase


class BasicStatistics(LocalModelBase):
    """
    Basic descriptive statistics for generating explanation.
    """

    # TODO: See paper “Algorithms for computing the sample variance: Analysis and recommendations.” for improments.
    def __init__(
        self,
        x_explain,
        chi_explain,
        y_p_explain,
        feature_names,
        target_names,
        class_index,
        r=None,
        tol_importance=0.001,
        tol_error=None,
        scale_data=False,
        save_samples=False,
    ):
        super().__init__(
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance,
            scale_data,
            save_samples,
        )
        self.values = {}
        # self.values = {e: [] for e in self.feature_names}

    def measure_importances(self):
        return self._measure_convergence_importance(self._coef_mean)

    def measure_errors(self):
        return 0.0

    def predict(self, x):
        return None

    def measure_convergence(self, chi_set, y_true):
        diff = self.measure_importances()
        error = self.measure_errors()
        self.erros_training.append(error)
        self.min_max_predictions(y_p_local_model=None, y_p_black_box_model=y_true)
        # Test convergence.
        if diff is None:
            self.convergence = False
        elif error <= self.tol_error and diff < self.tol_importance:
            self.convergence = True
        else:
            self.convergence = False
        return diff, error, self.convergence

    @property
    def _coef_mean(self):
        results = self.calculate()
        return np.array([*results["mean"].values()])

    @property
    def _coef_std(self):
        results = self.calculate()
        return np.array([*results["std"].values()])

    @property
    def importance(self):
        return self.calculate()

    def results(self):
        results = self.calculate()
        return results

    def partial_fit(self, x_set, y_set, weight_set=None):
        # TODO: Need to improve this. Maybe put a matrix here with the positions.
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
