import warnings
from collections import defaultdict
import numpy as np

from m_lime.explainers.explainer import Explainer
from m_lime.explainers.explainer import ConFavExaples
  

class ExplainerLinear(Explainer):

    def __init__(
        self,
        model_predict,
        density,
        local_model="HuberRegressor",
        feature_names=None,
        transformer=None,
        random_state=None,
        verbose=False
    ):
        """
        Simple class to perform linear explanation from a model
        :param model_predict: model that the explanation want to be generated.
        :param density:  Density class, manifold estimation object that will be used to sample data.
        :param random_state: seed for random condition.
        :param linear_model: linear model that will be used to generate the explanation.
        See standard_linear_models variable.
        """
        super().__init__(
            model_predict,
            density,
            local_model=local_model,
            feature_names=feature_names,
            transformer=transformer,
            random_state=random_state,
            verbose=verbose
        )

    def explain_instance(
        self,
        x_explain,
        r=None, class_index=0,
        n_samples=2000, tol=0.0001,
        local_mini_batch_max=100,
        weight_kernel='gaussian'
        ):
        return super().explain_instance(
            x_explain,
            r=r,
            class_index=class_index,
            n_samples=n_samples,
            tol=tol,
            local_mini_batch_max=local_mini_batch_max,
            weight_kernel=weight_kernel
        )

    def lime_explainer(self, local_model):
        from lime.lime_tabular import TableDomainMapper
        from lime import explanation
        mode='regression' 
        x_explain = local_model.x_explain
        predicted_value = local_model.y_p_explain
        min_y = 0.0
        max_y = 1.0
        model_regressor = self.model_predict

        domain_mapper = TableDomainMapper(
            local_model.feature_names,
            local_model.x_explain,
            local_model.importance,
            categorical_features=[],
            discretized_feature_names=None,
            feature_indexes=None
        )

        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=mode,
                                          class_names=['target'])
        
        ret_exp.scaled_data = local_model.importance
        if mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    local_model.importance,
                    [0],
                    [0],

                    # scaled_data,
                    # yss,
                    # distances,
                    label,
                    2,
                    model_regressor=model_regressor,
                    feature_selection='none')

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp