import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import metrics

from melime.explainers.local_models.local_model_statistics import BasicStatistics
from melime.explainers.local_models.local_model_linear import RidgeMod, HuberRegressorMod, SGDRegressorMod
from melime.explainers.local_models.local_model_tree import Tree

standard_local_models = {
    "BasicStatistics": BasicStatistics,
    "SGD": SGDRegressorMod,
    "Ridge": RidgeMod,
    "HuberRegressor": HuberRegressorMod,
    "Tree": Tree,
}

standard_weight_kernel = ["gaussian"]


def transformer_identity(x):
    return x


class Explainer:
    def __init__(
        self,
        model_predict,
        generator,
        local_model="BasicStatistics",
        feature_names=None,
        target_names=["target"],
        transformer=None,
        random_state=None,
        verbose=False,
    ):
        """
        Class to produce a local explanation for an instance from 
        a ML model
        :param model_predict: model that the explanation want to be generated.
        :param generator:  Generator class, manifold estimation object that will be used to sample data.
        :param local_model: linear model that will be used to generate the explanation.
        :param transformer: transformation to be applied to the features for generating the features used to explain
        :param random_state: seed for random condition.
        :param verbose: bool to control if information will be printed on screen.
        """
        self.feature_names = feature_names
        self.target_names = target_names
        self.model_predict = model_predict
        self.generator = generator
        self.random_state = random_state
        if transformer is None:
            self.transformer = transformer_identity
        self.verbose = verbose

        if isinstance(local_model, str):
            self.local_model_name = local_model
            if local_model in standard_local_models:
                self.local_algorithm = standard_local_models[self.local_model_name]
            else:
                raise Exception(
                    f"local_model should be in the list {[*standard_local_models]:}. "
                    + "You can also use our own linear model inheriting from LocalModelBase."
                )
        else:
            self.local_algorithm = local_model
            self.local_model_name = "custom"

        self.predictions_index = set()
        self.predictions_stat = {
            "count": defaultdict(int),
            "mean_probability": defaultdict(float),
            "std_probability": defaultdict(float),
        }

    def explain_instance(
        self,
        x_explain,
        r=None,
        class_index=0,
        n_samples=500,
        tol_importance=0.001,
        tol_error=0.001,
        local_mini_batch_max=100,
        weight_kernel=None,
        test_batch=False,
        scale_data=False,
        include_x_explain_train=True,
    ):
        """
        Generate an explanation for an instance from a ML model.
        :param x_explain: instance to be explained
        :param r: radius of the ball of the neighborhood
        :param class_index: class which an explanation will be created
        :param n_samples: number of samples for each epochs
        :param tol: tolerance of the change in the importance
        :param local_mini_batch_max: max number of local-mini-batch to generate the linear model
        :return: explanation in a dict with importance, see status
        """
        if self.generator.transformer:
            chi_explain = self.generator.transform(x_explain)
        else:
            chi_explain = self.transformer(x_explain)

        shape_input = list(x_explain.shape[1:])
        if weight_kernel is None:
            self.weight_kernel = None
        elif isinstance(weight_kernel, str):
            if weight_kernel == "gaussian":
                self.weight_kernel = multivariate_normal(mean=chi_explain[0], cov=0.5 * r ** 2.0).pdf
            else:
                raise Exception(
                    f"weight_kernel should be in the list {' '.join(standard_weight_kernel):}. "
                    + "You can also use our own kernel."
                )
        else:
            self.weight_kernel = weight_kernel

        diff_importance = None
        error_local_model = None

        y_p_explain = self.model_predict(x_explain)
        if len(y_p_explain.shape) == 2:
            y_p_explain = y_p_explain[0][class_index]
        else:
            y_p_explain = y_p_explain[0]

        self.local_model = self.local_algorithm(
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names=self.feature_names,
            target_names=self.target_names,
            class_index=class_index,
            r=r,
            tol_importance=tol_importance,
            tol_error=tol_error,
            scale_data=scale_data,
        )
        stats = {}
        con_fav_samples = ContrafactualExaples()
        self.generator.generated_data = None

        if test_batch:
            x_test_set = self.generator.sample_radius(x_explain, r=r, n_samples=n_samples)
            chi_test_set = self.transformer(x_test_set)
            y_test_set = self.model_predict(x_test_set)

        ## iteration count = `local_mini_batch_max`
        ## for iteration count times, generate data from generator function
        ## and test the convergence of surrogate model to see 
        ## whether it achieved the local model error and some other statistics, equation 2 in article
        for step in range(local_mini_batch_max):
            if self.generator.transformer:
                x_set, chi_set = self.generator.sample_radius(x_explain, r=r, n_samples=n_samples)
            else:
                x_set = self.generator.sample_radius(x_explain, r=r, n_samples=n_samples)
                chi_set = self.transformer(x_set)

            if x_set is None:
                warnings.warn("New sample set is None!")
                break
            elif x_set.shape[0] == 0:
                warnings.warn("New sample set is empty, try increase the r value!")
                break

            # Include the x_explain each local-mini-batch
            if include_x_explain_train:
                x_set = np.append(x_set, x_explain.reshape([1] + [*x_set[0].shape]), axis=0)
                chi_set = np.append(chi_set, chi_explain.reshape([1] + [*chi_set[0].shape]), axis=0)

            if self.weight_kernel is not None:
                weight_set = self.weight_kernel(chi_set)
            else:
                weight_set = None

            ## y_p is the prediction of the generated data (x_set)
            y_p = self.model_predict(x_set.reshape([-1] + shape_input))
            if len(y_p.shape) != 1:
                y_p = y_p[:, class_index]
            self.local_model.partial_fit(chi_set, y_p, weight_set)
            if test_batch:
                self.calc_error(chi_test_set, y_test_set, weight_set)
            diff_importance, error_local_model, converged_lc = self.local_model.measure_convergence(chi_set, y_p)
            con_fav_samples.insert_many(x_set, y_p)
            # self.plot_convergence(x_set, y_p, diff_importance, error_local_model)
            if self.verbose:
                print("########################")
                print(" Local-Mini-Batch", step)
                print("\tdiff_importance", "error_local_model")
                print("\t", diff_importance, error_local_model)
            if converged_lc:
                break
        ## if the convergence criteria wasn't achieved after a loop with iteration count of `local_mini_batch_max`
        ## then print that it wasn't converged
        if not self.local_model.convergence:
            warnings.warn(
                "Convergence tolerance (tol) was not achieved!\n"
                + f"Current difference in the importance {diff_importance}/{tol_importance}\n"
                + f"Current Error: {error_local_model}/{tol_importance}"
            )
        return self.local_model, con_fav_samples

    def calc_error(self, chi_set, y_set, weight_set):
        y_p_test_set = self.local_model.model.predict(chi_set)
        v1 = metrics.explained_variance_score(y_set, y_p_test_set, sample_weight=weight_set)
        v2 = metrics.mean_squared_error(y_set, y_p_test_set, sample_weight=weight_set)
        return v1, v2

    def plot_convergence(self, x_set, y_p, diff_importance, error_local_model):
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs[0, 0].scatter(x_set[:, 0], x_set[:, 1], c=y_p, s=10)
        axs[0, 0].scatter([x_set[0, 0]], [x_set[0, 1]], s=20, c="red")
        axs[1, 0].scatter(x_set[:, 0], x_set[:, 1], c=self.local_model.predict(x_set))
        axs[0, 1].scatter(x_set[:, 0], self.local_model.predict(x_set), c="green")
        axs[0, 1].scatter(x_set[:, 0], y_p, c="red", s=10)
        axs[1, 1].scatter(x_set[:, 1], self.local_model.predict(x_set), c="green")
        axs[1, 1].scatter(x_set[:, 1], y_p, c="red", s=10)
        print(self.local_model.importance)
        print("diff_importance", "Errors")
        print(diff_importance, error_local_model)
        plt.show()


class ContrafactualExaples(object):
    """
    Class to save the n_max top favarable and
    n_max top contrary samples found.
    """

    def __init__(self, n_max=5):
        self.n_max = n_max
        self.y_con = list()
        self.y_fav = list()
        self.samples_con = list()
        self.samples_fav = list()

    def insert_many(self, samples, ys):
        for sample, y in zip(samples, ys):
            self.insert(sample, y)

    def insert(self, sample, y):
        # Favorable Samples
        if len(self.y_fav) < self.n_max:
            self.y_fav.append(y)
            self.samples_fav.append(sample)
        else:
            if y > self.y_fav[-1]:
                self.y_fav[-1] = y
                self.samples_fav[-1] = sample
        indices_ = np.argsort(self.y_fav).reshape(-1)[::-1]
        self.y_fav = [self.y_fav[e] for e in indices_]
        self.samples_fav = [self.samples_fav[e] for e in indices_]

        # Contrary Samples
        if len(self.y_con) < self.n_max:
            self.y_con.append(y)
            self.samples_con.append(sample)
        else:
            if y < self.y_con[-1]:
                self.y_con[-1] = y
                self.samples_con[-1] = sample
        indices_ = np.argsort(self.y_con).reshape(-1)
        self.y_con = [self.y_con[e] for e in indices_]
        self.samples_con = [self.samples_con[e] for e in indices_]

    def print_results(self):
        print("Contrary:")
        for e, ee in zip(self.samples_con, self.y_con):
            print(e, ee)
        print("Favarable:")
        for e, ee in zip(self.samples_fav, self.y_fav):
            print(e, ee)
