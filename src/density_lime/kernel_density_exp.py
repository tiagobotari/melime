from functools import partial
import numpy as np

from scipy.special import gammainc
from scipy.stats import multivariate_normal

from sklearn.neighbors.ball_tree import BallTree, DTYPE
from  sklearn.neighbors.kd_tree import KDTree
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array, check_random_state, check_consistent_length

from matplotlib import pyplot as plt


def gaussian(x, kernel_width):
    """
    Returr a Normal density distribution centered in a mu.
    :param x: data to
    :param mu: center of the Gaussian.
    :param sigma: deviation of Gaussian function
    :return:
    """
    sigma = kernel_width
    normalize = (sigma * np.sqrt(2.0 * np.pi)) ** -1
    return normalize * np.exp(-0.5 * x ** 2 / sigma ** 2)


class KernelDensityExp(KernelDensity):
    """
    Modification of KenelDensity from sklearn do sample data around a specific point.
    # TODO: I do not remember what Exp mean in the name, we need to find a name
    """

    def sub_kde(self, x_exp, r=None, kernel=None, kernel_width=None):
        """
        Routine that return a KDE with selected r radius.
        :param x_exp:
        :param r:
        :param sample_weight:
        :return:
        """
        if kernel_width is None:
            kernel_width = 0.1
        if r is None:
            r = kernel_width*4.0
        if kernel is None:
            kernel = gaussian

        ind_ = self.tree_.query_radius(x_exp, r=r, return_distance=False)[0].astype(int)  # .tolist()
        # TODO: TB: Make the selection from the tree structure.
        # TODO: TB: I had implemented a version que transform the tree structure to numpy array
        # TODO: TB: and then do the selection. This is not a good strategy.
        # data = np.asarray(self.tree_.data[ind_])
        x = np.asarray(self.tree_.data)[ind_]
        if x.shape[0] == 0:
            return None

        kernel_ = partial(kernel, kernel_width=kernel_width)

        d = np.sqrt(np.sum((x - x_exp) ** 2, axis=1))
        sample_weight = kernel_(d)
        return KernelDensityExp(
            kernel='gaussian', bandwidth=self.bandwidth).fit(x, sample_weight=sample_weight)

    def predict(self, x, kernel_width=None):
        """
        Probability density function - PDF
        :param x:
        :param kernel_width:
        :return:
        """
        return np.exp(self.score_samples(x))

    def sample_radius(self, x_exp, n_min_kernels=10, r=None, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        x_exp : sample data around x_exp, a ball is define with radius r
        r: radius of the defined ball for sampling the data
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state : int, RandomState instance or None. default to None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """
        # TODO: TB: This TODO is from Sklearn
        # TODO: implement sampling for other valid kernel shapes
        if self.kernel not in ['gaussian', 'tophat']:
            raise NotImplementedError()

        # TODO: TB comment.
        #  Select the automatically the ball to find the kernel. Not sure if this is the best strategy.
        if r is None:
            len_n_kernel = 0
            r = 0.1
            while len_n_kernel < n_min_kernels:
                r += 0.1
                ind_ = self.tree_.query_radius(x_exp, r=r, return_distance=False)[0].astype(int)
                len_n_kernel = len(ind_)
        else:
            ind_ = self.tree_.query_radius(x_exp, r=r, return_distance=False)[0].astype(int)

        # TODO: TB: Make the selection from the tree structure.
        # TODO: TB: I had implemented a version que transform the tree structure to numpy array
        # TODO: TB: and then do the selection. This is not a good strategy.
        # data = np.asarray(self.tree_.data[ind_])
        data = np.asarray(self.tree_.data)
        data = data[ind_]
        if data.shape[0] == 0:
            return np.empty(data.shape)
        rng = check_random_state(random_state)
        u = rng.uniform(0, 1, size=n_samples)
        if self.tree_.sample_weight is None:
            i = (u * data.shape[0]).astype(np.int64)
        else:
            # TODO TB: Come back here!
            cumsum_weight = np.cumsum(np.asarray(self.tree_.sample_weight))
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)

        if self.kernel == 'gaussian':
            return np.atleast_2d(rng.normal(data[i], self.bandwidth))

        elif self.kernel == 'tophat':
            # we first draw points from a d-dimensional normal distribution,
            # then use an incomplete gamma function to map them to a uniform
            # d-dimensional tophat distribution.
            dim = data.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = row_norms(X, squared=True)
            correction = (gammainc(0.5 * dim, 0.5 * s_sq) ** (1. / dim)
                          * self.bandwidth / np.sqrt(s_sq))
            return data[i] + X * correction[:, np.newaxis]
