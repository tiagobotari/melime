import numpy as np

from sklearn.linear_model import SGDRegressor, Ridge, HuberRegressor


class SGDRegressorMod(SGDRegressor):
    def __init__(self
    , loss='squared_loss'
    , penalty='l2'
    , alpha=0.001
    , l1_ratio=0.15
    , fit_intercept=True
    , max_iter=100000
    , tol=0.001, shuffle=True, verbose=0
    , epsilon=0.1, random_state=None
    , learning_rate='adaptive', eta0=0.01, power_t=0.25
    , early_stopping=False, validation_fraction=0.1, n_iter_no_change=100
    , warm_start=False, average=False):
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio
            , fit_intercept=fit_intercept, max_iter=max_iter, tol=tol
            , shuffle=shuffle, verbose=verbose, epsilon=epsilon
            , random_state=random_state, learning_rate=learning_rate
            , eta0=eta0, power_t=power_t, early_stopping=early_stopping
            , validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change
            , warm_start=warm_start, average=average)


class RidgeMod(Ridge):
    def __init__(
        self, alpha=1.0, fit_intercept=True, normalize=False
        , copy_X=True, max_iter=None, tol=0.0001, solver='auto', random_state=None):
        super().__init__(
            alpha=alpha
            , fit_intercept=fit_intercept
            , normalize=normalize, copy_X=copy_X
            , max_iter=max_iter, tol=tol
            , solver=solver
            , random_state=random_state)
        self.x_samples = None
        self.y_p = None

    def partial_fit(self, x_samples, y_p):
        if self.x_samples is None:
            self.x_samples = x_samples
            self.y_p = y_p
        else:
            self.x_samples = np.append(self.x_samples, x_samples, axis=0)
            self.y_p = np.append(self.y_p, y_p, axis=0)
        self.fit(self.x_samples, self.y_p)


class HuberRegressorMod(HuberRegressor):
    def __init__(self, epsilon=1.35, max_iter=1000, alpha=0.0001, warm_start=True, fit_intercept=True, tol=1e-05):
        super().__init__(
            epsilon=epsilon
            , max_iter=max_iter
            , alpha=alpha, warm_start=warm_start
            , fit_intercept=fit_intercept
            , tol=tol)

    def partial_fit(self, x_samples, y_p):
        self.fit(x_samples, y_p)