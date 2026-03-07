import warnings

import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor, BayesianRidge, ARDRegression

import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS

from melime.explainers.local_models.local_model_base import LocalModelBase


def transformer_identity(x):
    return x


class LocalModelLinear(LocalModelBase):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=None,
            tol_error_instance=None,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = None
        self.mse_error = 2.0 * self.tol_importance

    def measure_errors(self, y_true, y_p_local_model):
        return metrics.mean_squared_error(y_true=y_true, y_pred=y_p_local_model)

    def measure_importances(self):
        return self._measure_convergence_importance(self.importance)

    def predict(self, x):
        x = self.scaler.transform(x)
        return self.model.predict(x)

    @property
    def importance(self):
        return self.model.coef_


class Stat(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
            scale_data=False,
            save_samples=True,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = None

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.fit(self.x_samples)
        x_set = self.scaler.transform(self.x_samples)
        x = sm.add_constant(x_set, prepend=False)
        if weight_set is None:
            self.model = sm.OLS(self.y_samples, x).fit()
        else:
            self.model = WLS(self.y_samples, x, weights=self.weight_samples).fit()

    def predict(self, x):
        x = self.scaler.transform(x)
        # x = sm.add_constant(x, prepend=False)
        x = np.c_[x, np.ones((x.shape[0], 1))]
        return self.model.predict(x)

    @property
    def importance(self):
        return self.model.params[1:]


class SGDRegressorMod(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
            scale_data=False,
            save_samples=False,
            grid_search=False,
            l1_ratio=0.15,
            max_iter=10000,
            tol=0.001,
            learning_rate="adaptive",
            eta0=0.001,
            early_stopping=False,
            n_iter_no_change=10000,
            average=False,
            **kwargs
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = SGDRegressor(
            l1_ratio=l1_ratio,
            alpha=0.001,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            eta0=eta0,
            n_iter_no_change=n_iter_no_change,
            early_stopping=early_stopping,
            average=average,
            warm_start=True,
            **kwargs
        )
        self.grid_search = grid_search

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.partial_fit(x_set)
        x_set = self.scaler.transform(x_set)

        if self.grid_search:
            warnings.warn("Grid Search Not implemented!!")
            self.grid_search = False
            parameters = {
                "alpha": 10.0 ** (-np.arange(2, 7)),
                "eta0": 1,
                "loss": ["squared_loss", "huber", "epsilon_insensitive"],
            }
            # grid_search = GridSearchCV(model, parameters, n_jobs=-1)
            # grid_search.fit(x_set, y_set)
        self.model.partial_fit(x_set, y_set, sample_weight=weight_set)
        # self.model.fit(x_set, y_set, sample_weight=weight_set)


class RidgeMod(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
            scale_data=False,
            save_samples=True,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = Ridge(
            alpha=0.001,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            max_iter=10000,
            tol=1e-05,
            solver="lsqr",
            random_state=None,
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.fit(self.x_samples)
        x_set = self.scaler.transform(self.x_samples)
        self.model.fit(x_set, self.y_samples, sample_weight=self.weight_samples)


class HuberRegressorMod(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = HuberRegressor(
            epsilon=1.35, max_iter=10000, alpha=0.001, warm_start=True, fit_intercept=True, tol=1e-05
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.partial_fit(x_set)
        x_set = self.scaler.transform(x_set)
        self.model.fit(x_set, y_set, sample_weight=weight_set)
        # self.model.fit(x_set, self.y_samples, sample_weight=self.weight_samples)


class ARDRegressionMod(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
            scale_data=False,
            save_samples=True,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = ARDRegression(
            n_iter=1000, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
            compute_score=False, fit_intercept=True,
            normalize=False, copy_X=False, verbose=False, threshold_lambda=10000.0)

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.fit(self.x_samples)
        x_set = self.scaler.transform(self.x_samples)
        self.model.fit(x_set, self.y_samples)  # , sample_weight=self.weight_samples)


class BayesianRidgeMod(LocalModelLinear):
    def __init__(
            self,
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance=0.001,
            tol_error=0.001,
            tol_error_instance=None,
            scale_data=False,
            save_samples=True,
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
            tol_error,
            tol_error_instance,
            scale_data,
            save_samples,
        )
        self.model = BayesianRidge(
            n_iter=1000, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
            alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True,
            normalize=False, copy_X=False, verbose=False)

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.fit(self.x_samples)
        x_set = self.scaler.transform(self.x_samples)
        self.model.fit(x_set, self.y_samples, sample_weight=self.weight_samples)
