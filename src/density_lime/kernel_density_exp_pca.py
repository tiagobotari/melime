
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.manifold import Isomap

from density_lime.kernel_density_exp import KernelDensityExp


# TODO: Implement Kernel PCA?
#  Probabilistic PCA? MiniBatchDictionaryLearning? MiniBatchSparsePCA? LinearDiscriminantAnalysis?
#  Independent Component Analysis
# TODO: Incorporate this as a children of KernelDensityExp or maybe create a abstract class. I am not sure.
class KernelDensityExpPCA(object):

    def __init__(self, kernel='gaussian', bandwidth=0.1, n_components=None):
        self.pca = IncrementalPCA(n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = KernelDensityExp(
            kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=50, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.pca.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample


class KernelDensityExpKernelPCA(object):

    def __init__(self, kernel='gaussian', bandwidth=0.1, n_components=None):
        # kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” |
        self.pca = KernelPCA(n_components=n_components, kernel='rbf', fit_inverse_transform=True)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = KernelDensityExp(
            kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=50, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.pca.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.pca.inverse_transform(x_sample_pca)
        return x_sample


# TODO: Isomap has no inverse transformation, maybe we can cameup with some solution for that.
#  I think this could be very useful in the future.
class KernelDensityExpIsomap(object):

    def __init__(self, kernel='gaussian', bandwidth=0.1, n_components=None, n_neighbors=20):
        # TODO: Optimization note: Isomap creates a tree structure, I do not need to recalculate that
        #  when creating the KernelDensityExp
        self.transformer = Isomap(n_neighbors, n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.transformer.fit_transform(x)
        self.manifold = KernelDensityExp(
            kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
        return self

    def sample_radius(self, x_exp, n_min_kernels=50, r=None, n_samples=1, random_state=None):
        x_exp_pca = self.transformer.transform(x_exp)
        x_sample_pca = self.manifold.sample_radius(
            x_exp_pca, n_min_kernels=n_min_kernels, r=r, n_samples=n_samples, random_state=random_state)
        x_sample = self.transformer.inverse_transform(x_sample_pca)
        return x_sample

    def sample(self, n_samples=1, random_state=None):
        x_sample_pca = self.manifold.sample(n_samples=n_samples, random_state=random_state)
        x_sample = self.transformer.inverse_transform(x_sample_pca)
        return x_sample
