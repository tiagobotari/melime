import numpy as np
from matplotlib import pyplot as plt


class Spiral(object):
    def __init__(self, r_a=1, r_b=0, error_x=None, error_y=None, n_samples=40000, theta_domain=None):
        self.r_a = r_a
        self.r_b = r_b
        self.feature_names = ["x1", "x2"]
        self.error_x = error_x
        self.error_y = error_y
        self.data, self.target = self.domain(theta_domain=theta_domain, n_samples=n_samples)

    def domain(self, n_samples=40000, seed=1654654, theta_domain=None):
        if theta_domain is None:
            theta_domain = [0, 8 * np.pi]
        random_state = np.random.RandomState(seed)
        distribution = random_state.rand(n_samples)
        theta = distribution * (theta_domain[1]) + theta_domain[0]
        return self.f(theta)

    def f(self, theta):

        if self.error_x is not None:
            epsilon_x1 = np.random.randn(len(theta)) * self.error_x
            epsilon_x2 = np.random.randn(len(theta)) * self.error_x
        else:
            epsilon_x1 = 0
            epsilon_x2 = 0

        if self.error_y is not None:
            epsilon_y = np.random.randn(len(theta)) * self.error_y
        else:
            epsilon_y = 0

        r = self.r_a * theta ** 1.0 + self.r_b
        r = r
        y = 0.5 * self.r_a * (theta * np.sqrt(1.0 + theta ** 2) + np.log(theta + np.sqrt(1.0 + theta ** 2)))
        y = y + epsilon_y

        x1 = r * np.cos(theta) + epsilon_x1
        x2 = r * np.sin(theta) + epsilon_x2
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        x = np.append(x1, x2, axis=-1)

        return x, y

    def sample_data_around(self, x, theta_0, data_size=10000):
        seed = 1478
        random_state = np.random.RandomState(seed)
        distribution = (random_state.rand(data_size) - 0.5) * 1.0
        theta = distribution + theta_0
        return self.f(theta)

    def plot(self):
        fig, ax = plt.subplots()
        cp = ax.scatter(self.data[:, 0], self.data[:, 1], c=self.target)
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        color = plt.colorbar(cp)
        color.set_label(label="Length Spiral", size=18)
        color.ax.tick_params(labelsize=14)
        return fig, ax


class SimpleSpiral(object):
    def __init__(self, r_a=1, r_b=0, error_x=0.2, error_y=None):
        self.r_a = r_a
        self.r_b = r_b
        self.error_x = error_x
        self.error_y = error_y

    def domain(self, data_size=40000, seed=1654654):
        random_state = np.random.RandomState(seed)
        distribution = random_state.rand(data_size)
        theta = distribution * (2 * np.pi) + 2 * np.pi
        return self.f(theta)

    def f(self, theta):

        if self.error_x is not None:
            epsilon_x1 = np.random.randn(len(theta)) * self.error_x
            epsilon_x2 = np.random.randn(len(theta)) * self.error_x
        else:
            epsilon_x1 = 0
            epsilon_x2 = 0

        if self.error_y is not None:
            epsilon_y = np.random.randn(len(theta)) * self.error_y
        else:
            epsilon_y = 0

        r = self.r_a * theta ** 1.0 + self.r_b
        r = r
        y = 0.5 * self.r_a * (theta * np.sqrt(1.0 + theta ** 2) + np.log(theta + np.sqrt(1.0 + theta ** 2)))
        y = y + epsilon_y

        x1 = r * np.cos(theta) + epsilon_x1
        x2 = r * np.sin(theta) + epsilon_x2
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        x = np.append(x1, x2, axis=-1)

        return x, y

    def sample_data_around(self, x, theta_0, data_size=10000):
        seed = 1478
        random_state = np.random.RandomState(seed)
        distribution = (random_state.rand(data_size) - 0.5) * 1.0
        theta = distribution + theta_0
        return self.f(theta)

