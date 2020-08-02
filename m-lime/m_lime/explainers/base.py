import warnings
from collections import defaultdict

import numpy as np

from m_lime.explainers.local_models.statistics import BasicStatistics


standard_local_models = {"BasicStatistics": BasicStatistics}

def transformer_identity(x):
    return x

class ExplainerBase:
    
    def __init__(
        self,
        model_predict,
        density, 
        local_model="BasicStatistics", 
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
        :self.transformer: transformation to be applied to the features to generate the features to explain
        :param random_state: seed for random condition.
        """
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
                    "local_model should be in the list {:}. "
                    "You can also use our own linear model inheriting from LocalModelBase."
                )
        else:
            self.local_model = local_model
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
        n_samples=2000,
        tol=0.0001,
        features_names=None,
        local_mini_batch_max=100
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
        diff = None
        y_p_explain = self.model_predict(x_explain)[0][class_index]
        self.local_model = self.local_algorithm(
            x_explain, y_p_explain, features_names=features_names, tol_convergence=tol)
        stats = {}
        con_fav_samples = ConFavExaples()
        self.density.generated_data = None
        for step in range(local_mini_batch_max):
            if self.density.transformer:
                x_set, chi_set = self.density.sample_radius(x_explain, r, n_samples=n_samples)
            else:
                x_set = self.density.sample_radius(x_explain, r, n_samples=n_samples)                
                chi_set = self.transformer(x_set)
            if x_set is None:
                break
            y_p = self.model_predict(x_set)[:, class_index]
            con_fav_samples.insert_many(x_set, y_p)
            self.local_model.partial_fit(chi_set, y_p)
            diff = self.local_model.measure_convergence()
            if self.local_model.convergence:
                break
        if not self.local_model.convergence:
            warnings.warn(
                "Convergence tolerance (tol) was not achieved!" + f"Current difference in the importance {diff}/{tol}"
            )
        return self.results(), con_fav_samples

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


class ConFavExaples(object):
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
        if len(self.y_con) < self.n_max:
            self.y_con.append(y)
            self.samples_con.append(sample)
        else:
            if y > self.y_con[-1]:
                self.y_con[-1] = y
                self.samples_con[-1] = sample
        indices_ = np.argsort(self.y_con).reshape(-1)[::-1]
        self.y_con = [self.y_con[e] for e in indices_]
        self.samples_con = [self.samples_con[e] for e in indices_]
        if len(self.y_fav) < self.n_max:
            self.y_fav.append(y)
            self.samples_fav.append(sample)
        else:
            if y < self.y_fav[-1]:
                self.y_fav[-1] = y
                self.samples_fav[-1] = sample
        indices_ = np.argsort(self.y_fav).reshape(-1)
        self.y_fav = [self.y_fav[e] for e in indices_]
        self.samples_fav = [self.samples_fav[e] for e in indices_]

    def print_results(self):
        print("Contrary:")
        for e, ee in zip(self.samples_con, self.y_con):
            print(e, ee)
        print("Favarable:")
        for e, ee in zip(self.samples_fav, self.y_fav):
            print(e, ee)
