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
        loss="squared_loss",
        penalty="l2",
        l1_ratio=0.0,
        fit_intercept=True,
        max_iter=100000,
        tol=1e-05,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="adaptive",
        eta0=0.0005,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=100,
        warm_start=False,
        average=100,
        alpha=0.003
        ):
        super().__init__(x_explain, y_p_explain, features_names, r, tol_convergence)
        self.model = SGDRegressor(
            loss=loss,
            penalty=penalty,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=False,
            average=average,
            alpha=alpha
        )
        self.weight = None

    def partial_fit(self, x_set, y_set, sample_weight=None):
        # sample_weight = self.gaussian.pdf(x_set)
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
