"""
TODO: Probabilistic PCA? MiniBatchDictionaryLearning? MiniBatchSparsePCA? LinearDiscriminantAnalysis?
TODO: Independent Component Analysis
"""

from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.manifold import Isomap

from melime.generators.gen_base import GenBase
from melime.generators.kde_gen import KDEGen


class KDEPCAGen(GenBase):
    def __init__(self, kernel="gaussian", bandwidth=0.1, n_components=None, kde_params={}):
        super().__init__()
        self.pca = IncrementalPCA(n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_params = kde_params
        self.manifold = None
        self.len_data = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = KDEGen(kernel=self.kernel, bandwidth=self.bandwidth, **self.kde_params).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=20, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.pca.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state
        )
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample


class KDEKPCAGen(GenBase):
    def __init__(self, kernel="gaussian", bandwidth=0.1, n_components=None, kernel_pca="cosine"):
        super().__init__()
        # kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” |
        self.pca = KernelPCA(n_components=n_components, kernel=kernel_pca, fit_inverse_transform=True)  # , gamma=10)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = KDEGen(kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=20, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.pca.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state
        )
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample


class KDEIsomapGen(GenBase):
    # TODO: Isomap has no inverse transformation, maybe we can solve that in the future.
    # TODO: Look the the work: Inverse Methods for Manifold Learning from Kathleen Kay
    def __init__(self, kernel="gaussian", bandwidth=0.1, n_components=None, n_neighbors=20):
        super().__init__()
        self.transformer = Isomap(n_neighbors, n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None
        raise NotImplementedError

    def fit(self, x):
        x_pca = self.transformer.fit_transform(x)
        self.manifold = KDEGen(kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=20, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.transformer.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state
        )
        x_sample = self.transformer.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.transformer.inverse_transform(x_sample_pca)
        return x_sample
