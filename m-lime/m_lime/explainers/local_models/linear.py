import numpy as np

from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor
from scipy.stats import multivariate_normal

from m_lime.explainers.local_models.base import LocalModelBase


class LocalModelLinear(LocalModelBase):
    def __init__(self, x_explain, y_p_explain, features_names, r, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.gaussian = multivariate_normal(
            mean=x_explain[0], cov=np.sqrt(x_explain.flatten().shape[0]) * .75)
        self.model = None

    def measure_convergence(self):
        return self._measure_convergence(self.model.coef_)

    @property
    def importance(self):
        return self.model.coef_


class SGDRegressorMod(LocalModelLinear):
    def __init__(
        self,
        x_explain,
        y_p_explain,
        features_names,
        r,
        tol_convergence=0.001, 
        l1_ratio=0.0,
        max_iter=100000,
        tol=0.001,   
        learning_rate="adaptive",
        eta0=0.0005,
        n_iter_no_change=100,
        average=100,
        **kwargs):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.model = SGDRegressor(
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            eta0=eta0,
            n_iter_no_change=n_iter_no_change,
            average=average,
            **kwargs
        )
        self.weight = None

    def partial_fit(self, x_set, y_set, sample_weight=None):
        self.model.partial_fit(x_set, y_set, sample_weight=sample_weight)


class RidgeMod(LocalModelLinear):
    def __init__(self, x_explain, y_p_explain, features_names, r, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.model = Ridge(
            alpha=0.0001,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            max_iter=None,
            tol=1e-05,
            solver="auto",
            random_state=None,
        )
        self.x_samples = None
        self.y_p = None
        self.sample_weight = None


    def partial_fit(self, x_set, y_set, sample_weight=None):
        if self.x_samples is None:
            self.sample_weight = self.gaussian.pdf(x_set)
            self.x_samples = x_set
            self.y_p = y_set
        else:
            self.sample_weight =  np.append(self.sample_weight, self.gaussian.pdf(x_set), axis=0)
            self.x_samples = np.append(self.x_samples, x_set, axis=0)
            self.y_p = np.append(self.y_p, y_set, axis=0)
        self.model.fit(self.x_samples, self.y_p, sample_weight=sample_weight)


class HuberRegressorMod(LocalModelLinear):
    def __init__(self, x_explain, y_p_explain, features_names, r, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.model = HuberRegressor(
            epsilon=1.35, max_iter=10000, alpha=0.0001, warm_start=True, fit_intercept=True, tol=1e-05
        )

    def partial_fit(self, x_set, y_set, sample_weight=None):
        sample_weight = self.gaussian.pdf(x_set)
        self.model.fit(x_set, y_set, sample_weight=sample_weight)
