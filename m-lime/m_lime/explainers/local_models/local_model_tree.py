import numpy as np

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from m_lime.explainers.local_models.local_model_linear import LocalModelLinear


def transformer_identity(x):
    return x


class Tree(LocalModelLinear):
    def __init__(
        self,
        x_explain,
        chi_explain,
        y_p_explain,
        feature_names,
        target_names,
        class_index,
        r,
        tol_importance,
        tol_error,
        scale_data=False,
        save_samples=True,
        tree__max_depth=5,
    ):
        super().__init__(
            x_explain,
            chi_explain,
            y_p_explain,
            feature_names,
            target_names,
            class_index,
            r,
            tol_importance,
            tol_error,
            scale_data=scale_data,
            save_samples=save_samples,
        )
        self.model = tree.DecisionTreeRegressor(max_depth=tree__max_depth)

    def measure_importances(self):
        return self._measure_convergence_importance(self.importance)

    @property
    def importance(self):
        return self.model.feature_importances_

    def partial_fit(self, x_set, y_set, weight_set=None):
        super().partial_fit(x_set, y_set, weight_set)
        self.scaler.fit(self.x_samples)
        x_set = self.scaler.transform(self.x_samples)
        self.model.fit(x_set, self.y_samples, sample_weight=self.weight_samples)

    def plot_tree(self, max_depth=3, fontsize=14, proportion=True):
        from sklearn.tree import plot_tree
        return plot_tree(
            self.model,
            class_names="probability",
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=fontsize,
            proportion=proportion,
        )

