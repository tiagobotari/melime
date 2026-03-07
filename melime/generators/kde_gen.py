import numpy as np
from scipy.spatial import KDTree
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.stats import multivariate_normal

from melime.generators.gen_base import GenBase


class KDEGen(GenBase):

    def __init__(self, search_best=True, var_type=None, verbose=False, seed=None, **kwargs):
        super().__init__()
        self.transformer = False
        self.var_type = var_type
        self.rng = np.random.RandomState(seed)
        self.search_best = search_best
        self.verbose = verbose
        self.x_train = None
        self.tree_train = None
        self.sample_weight = None
        # self.bw = None
        self._cache_bw = {}

    def fit(self, x_train,  sample_weight=None):
        if self.var_type is None:
            self.var_type = 'c' * x_train.shape[1]
        self.x_train = x_train
        self.tree_train = KDTree(self.x_train, copy_data=False)
        self.sample_weight = sample_weight
        return self

    def select_data(self, x_explain, r, box=True):
        # 3 sigmas select above 99.7 of the data
        r_max = 3.5 * np.max(r)
        vec_r_max = 3.5 * r
        idx = self.tree_train.query_ball_point(x_explain, r=r_max)[0]
        data = self.x_train[idx]

        # Further selection
        if box:
            idx_s, data = select_from_box(x_explain, data, vec_r_max)
            idx = np.array(idx)[idx_s]
        return idx, data

    def calculate_kde_(self, x_explain, r, data, box=True,):
        # TODO: Limit the size that the cache_bw can grow.
        key = (x_explain, r, box)
        if key in self._cache_bw:
            return self._cache_bw[key]
        if self.verbose:
            print('Calculating Bandwidth from Multivariate Gaussian...')
        r = np.array(r)
        if self.search_best:
            # TODO: What is the best strategy with no neighbors instances are available?
            if data.shape[0] == 0:
                bw = r * 0.5
            elif data.shape[0] <= data.shape[1]:
                # bw = np.mean(np.abs(self.data_ - x_explain)*0.25, axis=0)
                bw = r * 0.5
            else:
                bw = KDEMultivariate(data=data, var_type=self.var_type, bw='cv_ls').bw
                # TODO: Improve algorithm to find the bw parameters, some values are very low.
                # TODO: This impose a lower boundary for bw.
                index_low_values = np.argwhere(bw < r[0]*0.1)
                bw[index_low_values] = r[index_low_values] * 0.1
        else:
            bw = KDEMultivariate(data=data, var_type=self.var_type, bw='normal_reference').bw
        if self.verbose:
            print('bw:', bw)
        self._cache_bw[key] = bw
        return bw

    def sample_radius(
            self,
            x_explain,
            r,
            n_samples=1,
            include_explain=True,
            kernel="gaussian",
            box=True,
            return_data=False,
            bw=None
    ):
        """
        Args:
           x_explain: instance where the local perturbation will be produced
           r: radius values used as the neighbourhood parameter
           n_samples: number of perturbed instances that will be generated
           include_explain: include the x_explain instance in the set of perturbations
           kernel: envelope function be applied on the estimation of p(x) and be used to generate the instances
           box: select instances that are in the range of 3.5*r inside of a "box"
           return_data: return the data around the instance that were used to estimate the local densidade
           bw: manually define the value of the bandwidth used in the local estimation
        Return:
            A numpy array with a set of perturbed instances, if return_data is True return also the used data.
        """
        x_explain = np.array(x_explain).reshape(1, -1)

        if len(np.shape(r)) > 0:
            r = np.array(tuple(r))

        if len(np.shape(r)) == 0:
            r = np.array([r] * x_explain.shape[1])
        idx, data = self.select_data(x_explain, r, box=box)

        if bw is None:
            bw = self.calculate_kde_(tuple(x_explain[0]), tuple(r), box=box, data=data)

        if include_explain:
            data = np.r_[data, x_explain]

        u = self.rng.uniform(0, 1, size=n_samples)
        if self.sample_weight is None:
            i = (u * data.shape[0]).astype(np.int64)
        else:
            # TODO: Fix that, it is not working!
            # Maybe it is a better strategy to sample data using the weights here.
            # self.sample_weight = self.weights(x_explain, data, r=0.01)
            if include_explain:
                self.sample_weight = np.r_[self.sample_weight, [1]]
            self.sample_weight = self.sample_weight.reshape(-1, 1)
            # Sample external to the Kernel, inputted during creation of the generator.
            sample_weight = self.sample_weight[idx]
            if include_explain:
                sample_weight = np._r[sample_weight, 1.0]
            cumsum_weight = np.cumsum(np.asarray(sample_weight))
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)

        if kernel == "gaussian":
            ii, counts = np.unique(i, return_counts=True)
            cov = np.diag(bw ** 2)
            x = np.empty(shape=(0, data.shape[1]))
            for j, count in zip(ii, counts):
                x = np.r_[x, self.rng.multivariate_normal(data[j], cov, size=count)]
        else:
            raise Exception()
        # if include_explain:
        #     x = np.r_[x, x_explain]

        if self.verbose > 1:
            print("Gaussian's centers:")
            print(data)

        if return_data:
            return np.atleast_2d(x), data
        return np.atleast_2d(x)

    @staticmethod
    def weights(x_explain, x, r):
        n_features = x.shape[1]
        if len(np.shape(r)) == 0:
            bandwidth = np.array([r] * n_features)
        else:
            bandwidth = np.array(r)
        # Calculate weights of the data
        cov = np.diag(bandwidth ** 2)
        f = multivariate_normal(x_explain[0], cov=cov)
        ff = lambda z: np.sqrt(f.pdf(z) / f.pdf(x_explain))
        return ff(x)

    def analysis_neighborhood(self, x_explain, r_max=0.5):
        y = []
        rr = np.arange(0.001, r_max, 0.00001)
        for r in rr:
            y += [len(self.tree_train.query_ball_point(x_explain, r=r)[0])]
        return rr, np.array(y)


def select_from_box(x_exp, data, vec_r):
    vec_r = np.array(vec_r)
    condition = np.ones(data.shape[0], dtype=bool)
    for i, r_i in enumerate(vec_r):
        if r_i is None:
            continue
        r_i_l = x_exp[0, i] - r_i
        r_i_h = x_exp[0, i] + r_i
        condition_i = ((data[:, i] > r_i_l) & (data[:, i] < r_i_h))
        condition = condition & condition_i
    idx = np.argwhere(condition)
    return idx, data[condition]


