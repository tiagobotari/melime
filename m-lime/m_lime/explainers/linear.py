import warnings
from collections import defaultdict
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from m_lime.explainers.local_models.linear import RidgeMod, HuberRegressorMod, SGDRegressorMod
from m_lime.explainers.base import ExplainerBase
from m_lime.explainers.base import ConFavExaples

standard_local_models = {
        "SGD": SGDRegressorMod,
         "Ridge": RidgeMod,
          "HuberRegressor": HuberRegressorMod
}
    

class ExplainLinear(object):
    # TODO: The explanation can be generated in a progressive way,
    #  we could generate more and more instances to minimize some criteria,
    #  one example could be the error of the linear model or a search to find
    #  a prediction from the sample that are desired.

    def __init__(
        self, model_predict, density, local_model="HuberRegressor", linear_param={}, random_state=None, verbose=False
    ):
        """
        Simple class to perform linear explanation from a model
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param random_state: seed for random condition.
        :param linear_model: linear model that will be used to generate the explanation.
        See standard_linear_models variable.
        """
        self.model_predict = model_predict
        self.density = density
        self.random_state = random_state
        self.verbose = verbose
        if isinstance(local_model, str):
            self.local_model_name = local_model
            if local_model in standard_local_models:
                self.local_model = standard_local_models[self.local_model_name]
            else:
                raise Exception(
                    "local_model should be in the list {:}. "
                    "You can also use our on linear model by instantiate in the linear_model."
                )
        else:
            self.local_model = local_model
            self.local_model_name = "custom"

    def explain_instance(self, x_explain, r=None, class_index=0, n_samples=2000, tol=0.0001, local_mini_batch_max=2000):
        """
        Generate a linear model to explain a instance locally
        :param x_explain: instance to be explained
        :param r: radius of the ball of the neighborhood
        :param class_index: class which an explanation will be created
        :param epochs: number of epochs to generate the linear model
        :param n_samples: number of samples for each epochs
        :param tol: tolerance of the change in the importance
        :param epochs_max: number max of epochs to achieve the tol
        :return: explanation in a dict with importance, see status
        """
        samples_generator, features = self.density.sample_radius(x_explain, r)
        
        importance_0 = None
        diff = tol * 10

        y_p_explain = self.model_predict(x_explain)[0][class_index]
        self.local_model = self.local_algorithm(x_explain, y_p_explain, features_names=features, tol_convergence=tol)
        stats = {}
        con_fav_samples = ConFavExaples()
        for x_set, chi_set in samples_generator:
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

        # for step in range(local_mini_batch_max):
        #     x_samples = self.density.sample_radius(
        #         x_exp=x_explain, r=r, random_state=self.random_state, n_samples=n_samples
        #     )
        #     y_p = self.model_predict(x_samples)
        #     # self.stats_(y_p)
        #     if len(y_p.shape) != 1:
        #         y_p = y_p[:, class_index]
        #     self.local_model.partial_fit(x_samples, y_p)

        #     if importance_0 is None:
        #         importance_0 = self.local_model.coef_.copy()
        #     else:
        #         diff = np.sum(np.abs(importance_0 - self.local_model.coef_))
        #         if diff < tol:
        #             return self.results()
        #         importance_0 = self.local_model.coef_.copy()

        #     if self.verbose:
        #         print(f"--- step: {step} ----")
        #         print("importance:", importance_0)
        #         print("importance diff:", diff)
        #         print("score:", self.local_model.score(x_samples, y_p))

        # if diff > tol:
        #     warnings.warn(
        #         "Convergence tolerance (tol) was not achieved!" + f"Current difference in the importance {diff}/{tol}"
        #     )

        return self.results()

    def fit_samples(self, x_samples, class_index):
        y_p = self.model_predict(x_samples)
        result = dict()
        result["most_predicted"] = self.most_predicted(y_p)
        # Linear Model
        self.local_model.fit(x_samples, y_p[:, class_index])
        result["cof"] = self.local_model.coef_
        return result

    def results(self):
        result = dict()
        # result['stats'] = self.stats()
        # Linear Model
        result["importance"] = self.local_model.coef_
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

    def find_contra_example(self, x_explain, r=0.1, n_samples=1000, epochs=100, target=None, class_target=None):
        """
        This routine should optimize a path from the source to a target instance
        defined by the prediction of the model.
        TODO: To implement this I will sample from instances from density  and find the
         best instance to the close target. With this new target, I will sample from it
         and optimize again. The optimization should be carried out up to convergence of the
         predicted target.
        :param x_explain:
        :param target:
        :return:
        """
        x_find = x_explain
        for epoch in range(epochs):
            x_samples = self.density.sample_radius(
                x_exp=x_find, r=r, random_state=self.random_state, n_samples=n_samples
            )
            y_p = self.model_predict(x_samples)
            class_index = np.argsort(y_p[:, :], axis=1)

        return self.results()
