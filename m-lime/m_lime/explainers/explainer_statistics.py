from collections import defaultdict
import numpy as np

np.seterr("raise")

from m_lime.explainers.explainer import Explainer
from m_lime.explainers.explainer import ConFavExaples


class ExplainStatistics(Explainer):
    def __init__(
        self,
        model_predict,
        density,
        local_model="BasicStatistics",
        feature_names=None,
        transformer=None,
        random_state=None,
        verbose=False,
    ):
        """
        Simple class to perform explanation using basic statistics of the predictions aroud an instance.
        This is useful when the space of the features is not well defined, for instance, texts.
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param random_state: seed for random condition.
        :param linear_model: linear model that will be used to generate the explanation.
        See standard_linear_models variable.
        """
        super().__init__(model_predict, density, local_model, random_state, verbose)

