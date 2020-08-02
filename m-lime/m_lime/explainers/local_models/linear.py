import numpy as np

from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor

from m_lime.explainers.local_models.base import LocalModelBase


class SGDRegressorMod(LocalModelBase):
    def __init__(self, x_explain, y_p_explain, features_names, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, tol_convergence)
        self.model = SGDRegressor(
            loss="squared_loss",
            penalty="l2",
            alpha=0.001,
            l1_ratio=0.15,
            fit_intercept=True,
            max_iter=100000,
            tol=0.001,
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            random_state=None,
            learning_rate="adaptive",
            eta0=0.01,
            power_t=0.25,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=100,
            warm_start=False,
            average=False,
        )
        LocalModelBase.__init__(tol_convergence)

    @property
    def measure_convergence(self):
        return self._measure_convergence(self.model.coef_)

    def partial_fit(self, x_set, y_set):
        self.model.partial_fit(x_set, y_set)

    @property
    def importance():
        return self.coef_


class RidgeMod(LocalModelBase):
    def __init__(self, x_explain, y_p_explain, features_names, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, tol_convergence)
        self.model = Ridge(
            alpha=1.0,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            max_iter=None,
            tol=0.0001,
            solver="auto",
            random_state=None,
        )
        self.x_samples = None
        self.y_p = None

    def measure_convergence(self):
        return self._measure_convergence(self.model.coef_)

    @property
    def importance(self):
        return self.model.coef_

    def partial_fit(self, x_set, y_set):
        if self.x_samples is None:
            self.x_samples = x_set
            self.y_p = y_set
        else:
            self.x_samples = np.append(self.x_samples, x_set, axis=0)
            self.y_p = np.append(self.y_p, y_set, axis=0)
        self.model.fit(self.x_samples, self.y_p)


class HuberRegressorMod(LocalModelBase):
    def __init__(self, x_explain, y_p_explain, features_names, tol_convergence=0.001):
        super().__init__(x_explain, y_p_explain, features_names, tol_convergence)
        self.model = HuberRegressor(
            epsilon=1.35, max_iter=1000, alpha=0.0001, warm_start=True, fit_intercept=True, tol=1e-05
        )

    def measure_convergence(self):
        return self._measure_convergence(self.model.coef_)

    @property
    def importance(self):
        return self.model.coef_

    def partial_fit(self, x_samples, y_p):
        self.fit(x_samples, y_p)
