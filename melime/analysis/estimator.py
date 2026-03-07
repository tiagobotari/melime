import numpy as np
from divergence import relative_entropy_from_kde, jensen_shannon_divergence_from_kde, \
    jensen_shannon_divergence_from_samples, relative_entropy_from_samples
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kde import KDEUnivariate

from experiments.utils.experimental_kl_divergence import InformationMeasurements as IM


def repeat_x(x_explain, n_samples):
    return np.repeat(
        x_explain, n_samples).reshape(x_explain.shape[1], -1).T


class GeneratorPerturbations:

    def __init__(self, x_train, seed=None):
        self.scaler = StandardScaler().fit(x_train)

        if seed is None:
            self.rnd = np.random.RandomState()
        else:
            self.rnd = np.random.RandomState(seed)

    def sample_r(self, x_explain, r, n_samples):
        x_explain_scaled = self.scaler.transform(x_explain)
        x_sampled = self.rnd.normal(
            x_explain_scaled[0], r, size=(n_samples, x_explain_scaled.shape[1]))
        return self.scaler.inverse_transform(x_sampled)

    def sample_r_col(self, x_explain, col, r, n_samples):
        n_features = x_explain.shape[1]
        x_explain_scaled = self.scaler.transform(x_explain)
        x_sampled = self.rnd.normal(
            x_explain_scaled[0, col], r, size=(n_samples, 1))
        x = repeat_x(x_explain, n_samples)
        x[:, col] = x_sampled[:, 0]
        return self.scaler.inverse_transform(x)

    def sample_radius(self, x_explain, r, n_samples, vec_r=None):
        n_features = x_explain.shape[1]
        if vec_r is None:
            vec = np.ones(n_features)
        else:
            vec = np.array(vec_r)
        x_explain_scaled = self.scaler.transform(x_explain)
        # TODO: build arbitrary covariance matrix with out diagonal terms
        cov = np.diag((vec * r) ** 2)
        x = self.rnd.multivariate_normal(x_explain_scaled[0], cov, size=n_samples)
        return self.scaler.inverse_transform(x)

    def plot_samples(self, x_explain, r, n_samples):
        x_sampled = self.sample(x_explain, r, n_samples)
        # plot_samples(x_sampled)
        raise NotImplementedError


class Estimator:

    def __init__(self, x_explain, x_sampled, y_sample, r, weights=None, method='histogram', parameters={}):
        self.r = r
        self.n_samples = x_sampled.shape[0]
        self.x_explain = x_explain
        # self.predict_proba = predict_proba
        self.method = method
        if self.n_samples == 0:
            self.y = []
            self.success = False
        else:
            # self.y = self.predict_proba(x_sampled).reshape(-1)
            self.y = y_sample
            self.x_sampled = x_sampled
            self.weights = weights
            self.success = True
            # Calculate mean
            self.mean = np.mean(self.y)
            self.std = np.std(self.y)
            self.median = np.median(self.y)
            self.ptp = np.ptp(self.y)
            self.min = np.min(self.y)
            self.max = np.max(self.y)
            # Probability
            self.p, self.bins = None, None
            self.estimate_probability(self.method, parameters=parameters)
            # Alternative
            # p0 = KernelDensity(bandwidth=0.01)
            # p0.fit(y0)
            # Other
            self.jensen_shannon_divergence = None
            self.kullback_leibler_divergence = None

    def as_dict(self):
        return dict(
            y=self.y,
            success=self.success,
            mean=self.mean,
            std=self.std,
            median=self.median,
            ptp=self.ptp,
            min=self.min,
            max=self.max,
            jensen_shannon_divergence=self.jensen_shannon_divergence,
            kullback_leibler_divergence=self.kullback_leibler_divergence,
            n_samples=self.n_samples
        )

    def estimate_probability(self, method, parameters=None):
        if parameters is None:
            parameters = {}
        if method == 'kde':
            self.p = KDEUnivariate(self.y)
            self.p.fit()
        elif method == 'histogram':
            bins = parameters.pop("bins", None)
            self.p, self.bins = self.estimate_probability_histogram(self.y, weights=self.weights, bins=bins)

    @staticmethod
    def estimate_probability_histogram(x, bins, weights):
        # Estimate the probability from y
        p, bins = np.histogram(x, bins=bins, density=True, weights=weights)
        return p, bins

    def entropy(self):
        if self.method == 'kde':
            return self.p.entropy
        else:
            return IM.entropy(self.p)
        # entropy_from_samples(y0, discrete=False)
        # entropy_from_kde(p0)

    def probability(self, x):
        return self.p.evaluate(x)

    @property
    def kde(self):
        return self.p

    def statistics(self):
        return dict(
            mean=self.mean,
            median=self.median,
            std=self.std,
            max=self.max,
            min=self.min,
            ptp=self.ptp,
        )

    def jensen_shannon_divergence_from_kde(self, kde_p):
        self.jensen_shannon_divergence = jensen_shannon_divergence_from_kde(kde_p, self.kde)
        return self.jensen_shannon_divergence

    def kullback_leibler_divergence_kde(self, kde_p):
        self.kullback_leibler_divergence = relative_entropy_from_kde(kde_p, self.kde)
        return self.kullback_leibler_divergence

    def calculate_relative_measurements(self, estimator):
        if self.method == 'kde':
            self.jensen_shannon_divergence_from_kde(estimator.kde)
            self.kullback_leibler_divergence_kde(estimator.kde)
        else:
            self.jensen_shannon_divergence = IM.jensen_shannon_divergence(estimator.p, self.p)
            self.kullback_leibler_divergence = IM.d_kl_discrete(estimator.p, self.p)
        return dict(
            jensen_shannon_divergence=self.jensen_shannon_divergence,
            kullback_leibler_divergence=self.kullback_leibler_divergence)

    def calculate_relative_measurements_sample(self, estimator):
        self.jensen_shannon_divergence = jensen_shannon_divergence_from_samples(estimator.y, self.y, discrete=False)
        self.kullback_leibler_divergence = relative_entropy_from_samples(estimator.y, self.y, discrete=False)
        return dict(
            jensen_shannon_divergence=self.jensen_shannon_divergence,
            kullback_leibler_divergence=self.kullback_leibler_divergence)


def create_estimator(x_explain, generator, predict_proba, r, r_vec_idx, vec_r,
                     n_samples=100000, bins=None, vec_r_fraction=False):
    if isinstance(bins, int):
        bins = np.linspace(0.0, 1.0, bins)
    vec_r1 = np.array(vec_r)

    if vec_r_fraction:
        vec_r1[:] += 0.1 * r
        vec_r1[r_vec_idx] = r
    else:
        vec_r1[r_vec_idx] = r

    x = generator.sample_radius(
        x_explain, r=vec_r1, n_samples=n_samples, include_explain=True)
    y = predict_proba(x).reshape(-1)
    weights = generator.weights(x_explain, x, vec_r1)

    return Estimator(
        x_explain, x, y_sample=y, r=vec_r1, weights=weights, method='histogram', parameters={'bins': bins})


# Test GeneratorPerturbations
def test_GeneratorPerturbations():
    n_samples = 100000
    rnd = np.random.RandomState(1234)
    x_train = rnd.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 2.0]], size=n_samples)
    std_train = np.std(x_train, axis=0)
    generator = GeneratorPerturbations(x_train, seed=1234)
    x_explain = np.array([[0.0, 0.0]])
    r = 0.05
    x = generator.sample(x_explain, r=r, n_samples=n_samples)
    std_x = np.std(x, axis=0)
    test = np.allclose(std_train * r, std_x, rtol=1e-03, atol=1e-04, equal_nan=False)
    return test


def test_estimator():
    import os
    import pickle

    import sklearn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Data
    data = datasets.load_iris()
    x_all = data.data
    y_all = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, train_size=0.80, random_state=123)
    # ML Model
    rf = RandomForestClassifier()
    filename = '../../experiments/rf_model_iris.bin'
    if os.path.exists(filename):
        rf = pickle.load(open(filename, 'rb'))
    else:
        rf.fit(x_train, y_train)
        pickle.dump(rf, open(filename, 'wb'))

    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    y_all_pred = rf.predict(data.data)

    print('Acurancy_score: ', sklearn.metrics.accuracy_score(y_test, y_test_pred))
    print('Random Forest Mean Square Error: ', np.mean((y_test_pred - y_test) ** 2))
    print('MSError when predicting the mean: ', np.mean((y_train.mean() - y_test) ** 2))
    # Instance to Explain
    i = 3
    # x_explain = x_test[i]
    x_explain = np.array([[6.0, 3., 5, 1.5]])
    y_explain = 1
    x_explain, y_explain, rf.predict(x_explain.reshape(1, -1))
    class_explain = rf.predict(x_explain.reshape(1, -1))[0]
    predict_proba = lambda x: rf.predict_proba(x)[:, class_explain].reshape(-1, 1)

    # Generator random samples
    generator = GeneratorPerturbations(x_train=x_train)
    n_samples = int(1e5)
    r = 0.5

    x_sampled = generator.sample(x_explain, r=r, n_samples=n_samples)
    estimator = Estimator(x_explain, x_sampled, predict_proba, r)
    print(estimator.statistics())


if __name__ == '__main__':
    test_estimator()
    # test_GeneratorPerturbations()
