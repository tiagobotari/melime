"""
Create a 2-d distributions and use two strategies of sampling, plot the final result.

"""
import sys
sys.path.append('../..')

import numpy as np
from density_lime.kernel_density_exp import KernelDensityExp
from matplotlib import pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
# import src.lime as lime
# import src.lime.lime_tabular


def data(n=50000):
    mu_l = [0, 0.5, 0.6, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    sigma_l = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    density = [0.25, 0.25, 0.25, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    X = np.empty((0, 2), dtype=np.float)

    for i in range(len(mu_l)):
        mu, sigma = mu_l[i], sigma_l[i]
        n_ = int(n*density[i])
        if n % 2 != 0: n_ += 1
        x1 = np.random.normal(mu, sigma, n_).reshape(-1, 2)
        X = np.concatenate((X, x1), axis=0)

    return X


class TestKDELocal2D(object):

    def __init__(self, n=50000, bandwidth=0.1):
        self.x_train = data(n)

        # KDE global!
        self.kde = KernelDensityExp(kernel='gaussian', bandwidth=bandwidth).fit(self.x_train)

    def plot_kde(self, ax):
        x = np.linspace(-0.3, 0.9, 100)
        y = np.linspace(-0.3, 0.9, 100)
        x, y = np.meshgrid(x, y)
        xy = np.vstack([x.ravel(), y.ravel()]).T
        z = self.kde.predict(xy)
        z = z.reshape(100, 100)
        levels = np.linspace(0, z.max(), 10)
        ax.scatter(self.x_train[:, 0], self.x_train[:, 1], c='gray', s=1)
        cp1 = ax.contour(x, y, z, levels=levels, colors='black', label='Global Kernel')

    def plot_kde_local(self, x_exp, ax, kernel=None, kernel_width=None):
        x_exp = np.array(x_exp)
        ax.scatter(x_exp[:, 0], x_exp[:, 1], c='red', s=100, label='x_exp')
        kde_local = self.kde.sub_kde(x_exp, r=0.1, kernel=kernel, kernel_width=kernel_width)
        x = np.linspace(x_exp[0, 0] - 0.5, x_exp[0, 0]+0.5, 100)
        y = np.linspace(x_exp[0, 1]-0.5, x_exp[0, 1]+0.5, 100)
        x, y = np.meshgrid(x, y)
        xy = np.vstack([x.ravel(), y.ravel()]).T
        z = kde_local.predict(xy)
        z = z.reshape(100, 100)
        levels = np.linspace(0, z.max(), 5)
        cp2 = ax.contour(x, y, z, levels=levels, colors='red', label='Final Local Kernel')


def test(x_exp=[[0.25, 0.25]]):
    """
    Some test of the implementation.
    :param x_exp:
    :param kde:
    :return:
    """


    #
    # KDE local, new estimation
    kde_local = kde_global.sub_kde(x_exp, r=0.1)
    x = np.linspace(-0.3, 0.9, 100)
    y = np.linspace(-0.3, 0.9, 100)
    x, y = np.meshgrid(x, y)
    xy = np.vstack([x.ravel(), y.ravel()]).T
    z = kde_local.score_samples(xy)
    z = z.reshape(100, 100)
    levels = np.linspace(0, z.max(), 5)
    cp2 = ax.contour(x, y, z, levels=levels, colors='red', label='Final Local Kernel')

    ax.legend()
    plt.show()


def plot_kde(X, bandwidth=0.02, kde=None, N=1000):
    """
    Routine to plot the KDE and data.
    :param X:
    :param bandwidth:
    :param kde:
    :param N:
    :return:
    """
    X = np.array(X)
    if kde is None:
        kde = KernelDensityExp(kernel='gaussian', bandwidth=bandwidth).fit(X)
    x1_max = np.max(X[:, 0])
    x1_min = np.min(X[:, 0])
    x2_max = np.max(X[:, 1])
    x2_min = np.min(X[:, 1])

    x = np.linspace(x1_min-0.4, x1_max+0.4, N)
    y = np.linspace(x2_min-0.4, x2_max+0.4, N)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = kde.score_samples(xy)
    Z = Z.reshape(N, N)
    levels = np.linspace(0, Z.max(), 10)
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=10)
    plt.contour(X, Y, Z, levels=levels, colors='pink')
    plt.show()


def lime():
    x_train = data(n=100000)

    plt.scatter(x_train[:, 0],x_train[:, 1])
    plt.show()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(x_train, y_train)

    sklearn.metrics.accuracy_score(y_test, rf.predict(x_test))



    explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

    i = np.random.randint(0, x_test.shape[0])
    exp = explainer.explain_instance(x_test[i], rf.predict_proba, num_features=2, top_labels=1)

    exp.show_in_notebook(show_table=True, show_all=False)



if __name__ == '__main__':
    lime()
    test_2()

