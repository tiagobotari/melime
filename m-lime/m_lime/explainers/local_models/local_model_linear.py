import numpy as np

from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor
from sklearn import metrics

from m_lime.explainers.local_models.local_model_base import LocalModelBase


def transformer_identity(x):
    return x


class LocalModelLinear(LocalModelBase):
    def __init__(self, x_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=False):
        super().__init__(x_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
        self.model = None
        self.mse_error = 2.0 * self.tol_convergence
        self.mse_erros = []

    def measure_convergence(self, chi_set, y_true):
        y_p_local_model = self.model.predict(chi_set)
        # Max and Min y_p_local_model_max value and y_p_max: y_true
        y_p_local_model_max = np.max(y_p_local_model)
        y_p_local_model_min = np.min(y_p_local_model)
        y_p_max = np.max(y_true)
        y_p_min = np.min(y_true)
        if self.y_p_local_model_max is None:
            self.y_p_local_model_max = y_p_local_model_max
            self.y_p_local_model_min = y_p_local_model_min
            self.y_p_max = y_p_max
            self.y_p_min = y_p_min
        else:
            self.y_p_local_model_max = np.max([self.y_p_local_model_max, y_p_local_model_max])
            self.y_p_local_model_min = np.min([self.y_p_local_model_min, y_p_local_model_min])
            self.y_p_max = np.max([self.y_p_max, y_p_max])
            self.y_p_min = np.min([self.y_p_min, y_p_min])

        # Difference of the importance.
        diff = self._measure_convergence_importance(self.model.coef_)
        # Error specific for the local model.
        mse_error = metrics.mean_squared_error(y_true=y_true, y_pred=y_p_local_model)
        self.mse_erros.append(mse_error)
        # Test convergence.
        if diff is None:
            self.convergence = False
        elif mse_error <= self.tol_convergence and diff < self.tol_convergence:
            self.convergence = True
        else:
            self.convergence = False
        return diff, mse_error

    def predict(self, x):
        return self.model.predict(x)

    @property
    def importance(self):
        return self.model.coef_


class SGDRegressorMod(LocalModelLinear):
    def __init__(
        self,
        x_explain,
        y_p_explain,
        feature_names,
        r,
        tol_convergence=0.001,
        save_samples=False,
        l1_ratio=0.0,
        max_iter=100000,
        tol=0.001,
        learning_rate="adaptive",
        eta0=0.0005,
        n_iter_no_change=100,
        average=100,
        **kwargs
    ):
        super().__init__(x_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
        self.model = SGDRegressor(
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=1.0e-5,
            learning_rate=learning_rate,
            eta0=eta0,
            n_iter_no_change=n_iter_no_change,
            average=average,
            **kwargs
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.model.partial_fit(x_set, y_set, sample_weight=weight_set)


class RidgeMod(LocalModelLinear):
    def __init__(self, x_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=True):
        super().__init__(x_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
        self.model = Ridge(
            alpha=0.0001,
            fit_intercept=True,
            normalize=False,
            copy_X=False,
            max_iter=None,
            tol=1e-05,
            solver="auto",
            random_state=None,
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.model.fit(self.x_samples, self.y_samples, sample_weight=self.weight_samples)


class HuberRegressorMod(LocalModelLinear):
    def __init__(self, x_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=False):
        super().__init__(x_explain, y_p_explain, feature_names, r, tol_convergence)
        self.model = HuberRegressor(
            epsilon=1.35, max_iter=10000, alpha=0.001, warm_start=True, fit_intercept=True, tol=1e-05
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.model.fit(x_set, y_set, sample_weight=weight_set)
