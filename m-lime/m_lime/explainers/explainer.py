import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import metrics

from m_lime.explainers.local_models.local_model_statistics import BasicStatistics
from m_lime.explainers.local_models.local_model_linear import RidgeMod, HuberRegressorMod, SGDRegressorMod



standard_local_models = {
    "BasicStatistics": BasicStatistics,
    "SGD": SGDRegressorMod,
    "Ridge": RidgeMod,
    "HuberRegressor": HuberRegressorMod
}

standard_weight_kernel = ["gaussian"]


def transformer_identity(x):
    return x


class Explainer:
    
    def __init__(
        self,
        model_predict,
        density, 
        local_model="BasicStatistics", 
        feature_names=None,
        transformer=None, 
        random_state=None, 
        verbose=False
        ):
        """
        Class to produce a local explanation for an instance from 
        a ML model
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param local_model: linear model that will be used to generate the explanation.
        :param transformer: transformation to be applied to the features for generating the features used to explain
        :param random_state: seed for random condition.
        :param verbose: bool to control with information will be printed on screen.
        """
        self.feature_names=feature_names
        self.model_predict = model_predict
        self.density = density
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
                    +"You can also use our own linear model inheriting from LocalModelBase."
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
        tol=0.001,
        local_mini_batch_max=100,
        weight_kernel=None,
        test_batch=False 
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
        chi_explain = self.transformer(x_explain)
        
        if weight_kernel is None:
            self.weight_kernel=None
        elif isinstance(weight_kernel, str):
            if weight_kernel == "gaussian": 
                self.weight_kernel = multivariate_normal(mean=chi_explain[0], cov=0.5*r**2.).pdf
            else:
                raise Exception(
                    f"weight_kernel should be in the list {' '.join(standard_weight_kernel):}. "
                    +"You can also use our own kernel."
                )
        else:
            self.weight_kernel = weight_kernel

        diff = None
        y_p_explain = self.model_predict(x_explain)
        if len(y_p_explain.shape)==2:
            y_p_explain = y_p_explain[0][class_index]
        else: 
            y_p_explain = y_p_explain[0]

        self.local_model = self.local_algorithm(
            x_explain, y_p_explain, feature_names=self.feature_names, r=r, tol_convergence=tol)
        stats = {}
        con_fav_samples = ConFavExaples()
        self.density.generated_data = None
        
        if test_batch:
            x_test_set = self.density.sample_radius(x_explain, r, n_samples=n_samples)                
            chi_test_set = self.transformer(x_test_set)
            y_test_set = self.model_predict(x_test_set)

        for step in range(local_mini_batch_max):
            if self.density.transformer:
                x_set, chi_set = self.density.sample_radius(x_explain, r, n_samples=n_samples)
            else:
                x_set = self.density.sample_radius(x_explain, r, n_samples=n_samples)                
                chi_set = self.transformer(x_set)
            if x_set is None:
                break
            if self.weight_kernel is not None:
                weight_set = self.weight_kernel(chi_set)
            else:
                weight_set = None
            y_p = self.model_predict(x_set)
            # TODO: Look for the statistics.
            # self.stats_(y_p)
            if len(y_p.shape) != 1:
                y_p = y_p[:, class_index]
            con_fav_samples.insert_many(x_set, y_p)
            self.local_model.partial_fit(chi_set, y_p, weight_set)
            if test_batch:
                self.calc_error(chi_test_set, y_test_set)
            diff_importance, error_local_model = self.local_model.measure_convergence(chi_set, y_p)
            if self.verbose:
                print('########################')
                print(' Local-Mini-Batch', step)
                print('\tdiff_importance', 'error_local_model')
                print('\t', diff_importance, error_local_model)
            if self.local_model.convergence:
                break
        if not self.local_model.convergence:
            warnings.warn(
                "Convergence tolerance (tol) was not achieved!\n" 
                + f"Current difference in the importance {diff_importance}/{tol}\n"
                + f"Current Error: {error_local_model}/{tol}"
            )
        return self.results(), con_fav_samples, self.local_model

    def results(self):
        result = dict()
        result["stats"] = self.stats()
        result["importance"] = self.local_model.importance
        return result

    def stats_(self, y_p):
        class_index = np.argsort(y_p[:, :], axis=1)
        unique, counts = np.unique(class_index[:, -3:], return_counts=True)
        self.predictions_index.update(unique)
        for key, value in zip(unique, counts):
            self.predictions_stat["count"][key] += value
            self.predictions_stat["mean_probability"][key] += np.mean(y_p[:, key])
            self.predictions_stat["std_probability"][key] += np.std(y_p[:, key])

    def stats(self):
        results = dict()
        for key in self.predictions_index:
            results[key] = {
                "count": self.predictions_stat["count"][key],
                "mean_probability": self.predictions_stat["mean_probability"][key]
                / self.predictions_stat["count"][key],
                "std_probability": self.predictions_stat["std_probability"][key] / self.predictions_stat["count"][key],
            }
        return results

    def calc_error(self, chi_set, y_set):
        y_p_test_set = self.local_model.model.predict(chi_test_set)
        v1 = metrics.explained_variance_score(y_test_set, y_p_test_set, sample_weight=weight_set)
        v2 = metrics.mean_squared_error(y_test_set, y_p_test_set, sample_weight=weight_set)
        return v1, v2
                

class ConFavExaples(object):
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
