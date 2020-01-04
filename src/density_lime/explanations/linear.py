from collections import defaultdict
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.metrics.pairwise import euclidean_distances

standard_linear_models = {
    'SGD': SGDRegressor
}


class ExplainLinear(object):
    # TODO: The explanation can be generated in a progressive way,
    #  we could generate more and more instances to minimize some criteria,
    #  one example could be the error of the linear model or a search to find
    #  a prediction from the sample that are desired.

    def __init__(self, model_predict, density, linear_model='SGD', random_state=None, verbose=False):
        """
        Simple class to perform linear explanation from a model
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param random_state: seed for random condition.
        :param linear_model: linear model that will be used to generate the explanation.
        See standard_linear_models variable.
        """
        self.density = density
        self.random_state = random_state
        if isinstance(linear_model, str):
            if linear_model in standard_linear_models:
                self.local_model = None
                self.local_model = standard_linear_models[linear_model]()
                self.model_linear = linear_model
            else:
                raise Exception(
                    'linear_model should be in the list {:}. '
                    'You can also use our on linear model by instantiate in the linear_model.')
        else:
            self.local_model = linear_model
            self.model_linear = 'custom'

        self.model_predict = model_predict

        self.predictions_index = set()
        self.predictions_stat = {
            'count': defaultdict(int)
            , 'mean_probability': defaultdict(float)
            , 'std_probability': defaultdict(float)
        }

    # def explain_instance(self, x_explain, r, class_index=0, n_samples=2000):
    #     x_samples = self.density.sample_radius(
    #         x_exp=x_explain
    #         , r=r
    #         , random_state=self.random_state
    #         , n_samples=n_samples
    #     )
    #     return self.fit_samples(x_samples, class_index)

    def explain_instance(self, x_explain, r, class_index=0, epochs=10, n_samples=2000, tol=0.1):
        importance_0 = None
        for epoch in range(epochs):
            x_samples = self.density.sample_radius(
                x_exp=x_explain
                , r=r
                , random_state=self.random_state
                , n_samples=n_samples
            )
            y_p = self.model_predict(x_samples)
            self.stats_(y_p)
            self.local_model.partial_fit(x_samples, y_p[:, class_index])

            if (epoch%5) == 0:
                if importance_0 is None:
                    importance_0 = self.local_model.coef_.copy()
                else:
                    diff = np.sum(np.abs(importance_0 - self.local_model.coef_))
                    print(diff)
                    importance_0 = self.local_model.coef_.copy()

        return self.results()

    # def fit_samples(self, x_samples, class_index):
    #     result = dict()
    #     result['most_predicted'] = self.most_predicted(y_p)
    #     # Linear Model
    #     self.local_model.fit(x_samples, y_p[:, class_index])
    #     result['cof'] = self.local_model.coef_
    #     return result

    def fit_samples(self, x_samples, class_index):
        y_p = self.model_predict(x_samples)
        result = dict()
        result['most_predicted'] = self.most_predicted(y_p)
        # Linear Model
        self.local_model.fit(x_samples, y_p[:, class_index])
        result['cof'] = self.local_model.coef_
        return result

    def results(self):
        result = dict()
        result['stats'] = self.stats()
        # Linear Model
        result['importance'] = self.local_model.coef_
        return result

    def stats_(self, y_p):
        class_index = np.argsort(y_p[:, :], axis=1)
        unique, counts = np.unique(class_index[:, -3:], return_counts=True)
        self.predictions_index.update(unique)
        for key, value in zip(unique, counts):
            self.predictions_stat['count'][key] += value
            self.predictions_stat['mean_probability'][key] += np.mean(y_p[:, key])
            self.predictions_stat['std_probability'][key] += np.std(y_p[:, key])

    def stats(self):
        results = dict()
        for key in self.predictions_index:
            results[key] = {
                'count': self.predictions_stat['count'][key]
                , 'mean_probability': self.predictions_stat['mean_probability'][key]/self.predictions_stat['count'][key]
                , 'std_probability': self.predictions_stat['std_probability'][key]/self.predictions_stat['count'][key]
            }

        return results

    def find_contra_example(self, x_explain, r=0.1, n_samples=1000, target=None, class_target=None):
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
                x_exp=x_find
                , r=r
                , random_state=self.random_state
                , n_samples=n_samples
            )
            y_p = self.model_predict(x_samples)
            class_index = np.argsort(y_p[:, :], axis=1)


        return self.results()
