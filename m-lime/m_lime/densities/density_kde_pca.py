from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.manifold import Isomap

from m_lime.densities.base import Density
from m_lime.densities.density_kde import DensityKDE


# TODO: Implement Kernel PCA?
#  Probabilistic PCA? MiniBatchDictionaryLearning? MiniBatchSparsePCA? LinearDiscriminantAnalysis?
#  Independent Component Analysis
# TODO: Incorporate this as a children of KernelDensityExp or maybe create a abstract class. I am not sure.
class DensityKDEPCA(Density):
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
        self.manifold = DensityKDE(kernel=self.kernel, bandwidth=self.bandwidth, **self.kde_params).fit(x_pca)
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


class DensityKDEKPCA(Density):
    def __init__(self, kernel="gaussian", bandwidth=0.1, n_components=None, kernel_pca="cosine"):
        super().__init__()
        # kernel: “linear” | “poly” | “rbf” | “sigmoid” | “cosine” |
        self.pca = KernelPCA(n_components=n_components, kernel=kernel_pca, fit_inverse_transform=True)  # , gamma=10)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.pca.fit_transform(x)
        self.manifold = DensityKDE(kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
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


# TODO: Isomap has no inverse transformation, maybe we can came up with some solution for that.
#  I think this could be very useful in the future.
#  Look the the work: Inverse Methods for Manifold Learning from Kathleen Kay
class DensityKDEIsomap(Density):
    def __init__(self, kernel="gaussian", bandwidth=0.1, n_components=None, n_neighbors=20):
        super().__init__()
        # TODO: Optimization note: Isomap creates a tree structure, I do not need to recalculate that
        #  when creating the KernelDensityExp
        self.transformer = Isomap(n_neighbors, n_components=n_components)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.manifold = None

    def fit(self, x):
        x_pca = self.transformer.fit_transform(x)
        self.manifold = DensityKDE(kernel=self.kernel, bandwidth=self.bandwidth).fit(x_pca)
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


if __name__ == "__main__":
    import torchvision.models.quantization as models

    model_fe = models.resnet18(pretrained=True, progress=True, quantize=True)
