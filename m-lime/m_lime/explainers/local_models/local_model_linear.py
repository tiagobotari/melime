import numpy as np

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor

from m_lime.explainers.local_models.local_model_base import LocalModelBase


def transformer_identity(x):
    return x


class LocalModelLinear(LocalModelBase):
    def __init__(
        self, x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=False
    ):
        super().__init__(x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
        self.model = None
        self.mse_error = 2.0 * self.tol_convergence
        self.mse_erros = []

    def measure_errors(self, y_true, y_p_local_model):
        return metrics.mean_squared_error(y_true=y_true, y_pred=y_p_local_model)

    def measure_importances(self):
        return self._measure_convergence_importance(self.model.coef_)

    def predict(self, x):
        return self.model.predict(x)

    @property
    def importance(self):
        return self.model.coef_


class SGDRegressorMod(LocalModelLinear):
    def __init__(
        self,
        x_explain,
        chi_explain,
        y_p_explain,
        feature_names,
        r,
        tol_convergence=0.001,
        save_samples=False,
        grid_search=False,
        l1_ratio=0.0,
        max_iter=100000,
        tol=0.001,
        learning_rate="adaptive",
        eta0=0.001,
        n_iter_no_change=100,
        average=False,
        **kwargs
    ):
        super().__init__(x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
        
        self.scaler = StandardScaler()
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
        self.grid_search = grid_search

    def predict(self, x):
        x = self.scaler.transform(x)
        return self.model.predict(x)
    
    def partial_fit(self, x_set, y_set, weight_set=None):
        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.partial_fit(x_set)
        
        if self.grid_search:
            self.grid_search = False
            parameters = {
                "alpha": 10.0 ** (-np.arange(2, 7)),
                "eta0": 1,
                "loss": ["squared_loss", "huber", "epsilon_insensitive"],
            }
            grid_search = GridSearchCV(model, parameters, n_jobs=-1)
            grid_search.fit(x_train, y_train)
        super().partial_fit(x_set, y_set, weight_set)
        x_set_t = self.scaler.transform(x_set)
        self.model.partial_fit(x_set_t, y_set, sample_weight=weight_set)


class RidgeMod(LocalModelLinear):
    def __init__(self, x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=True):
        super().__init__(x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence, save_samples)
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
    def __init__(
        self, x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence=0.001, save_samples=False
    ):
        super().__init__(x_explain, chi_explain, y_p_explain, feature_names, r, tol_convergence)
        self.model = HuberRegressor(
            epsilon=1.35, max_iter=10000, alpha=0.0001, warm_start=True, fit_intercept=True, tol=1e-05
        )

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.model.fit(x_set, y_set, sample_weight=weight_set)

