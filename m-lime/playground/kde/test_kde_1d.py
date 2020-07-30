"""
Create a 1-d distributions and use two strategies of sampling, plot the final result.

"""
import numpy as np
from src.density_lime.kernel_density_exp import KernelDensityExp
from matplotlib import pyplot as plt


def data_1d():
    mu_l = [-0.2, 0.2]
    sigma_l = [0.05, 0.05]
    X = np.empty((0, 1), dtype=np.float)
    for mu, sigma in zip(mu_l, sigma_l):
        x1 = np.random.normal(mu, sigma, 50000).reshape(-1, 1)
        X = np.concatenate((X, x1), axis=0)
    return X


def test_1d(x_exp=[[0.0]]):
    x_exp = np.array(x_exp)

    x = data_1d()

    # KDE global, black line!
    kde_global = KernelDensityExp(kernel='gaussian', bandwidth=0.05).fit(x)
    x_ = np.linspace(-0.7, 0.7, 1000).reshape(-1, 1)
    y_ = kde_global.predict(x_)
    y_normalizer = np.exp(kde_global.score_samples(x_exp))
    fig, ax = plt.subplots(1, 1)

    ax.plot([x_exp[0], x_exp[0]], [0, 1.], c='gray', linestyle='dashed')
    ax.plot(x_, y_, color='gray', lw=2)

    # KDE local_1, pink line!
    x_sample = kde_global.sample_radius(x_exp=x_exp, r=0.2, n_samples=20000)
    if x_sample.shape[0] != 0:
        kde_local_1 = KernelDensityExp(kernel='gaussian', bandwidth=0.01).fit(x_sample)

        x_ = np.linspace(x_exp[0, 0]-0.5, x_exp[0, 0]+0.5, 300).reshape(-1, 1)
        y_ = kde_local_1.predict(x_)
        y_ = y_normalizer*y_/kde_local_1.predict(x_exp)
        ax.plot(x_.reshape(-1), y_, color='green', lw=3, label='From Original KDE')

    # KDE local, new estimation
    kde_local = kde_global.sub_kde(x_exp, r=0.4)
    x_ = np.linspace(x_exp[0, 0]-0.5, x_exp[0, 0]+0.5, 300).reshape(-1, 1)
    y_ = kde_local.predict(x_)

    y_ = y_normalizer*y_/kde_local.predict(x_exp)
    #
    ax.plot(x_.reshape(-1), y_, color='red', lw=3, label='Final PDE')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    test_1d()


