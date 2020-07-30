import numpy as np
from sklearn import datasets
import sklearn.ensemble
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

import lime.lime_tabular
from density_lime.lime_tabular_manifold import LimeTabularExplainerManifold

if __name__ == '__main__':
    data = datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data.data, data.target, train_size=0.80)
    df = pd.DataFrame(data.data, columns=data.feature_names)

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(x_train, y_train)

    categorical_features = []

    i = 5
    x_explain = x_test[i]
    df_1 = pd.DataFrame(x_explain.reshape(1, -1), columns=data.feature_names)
    print(x_explain, y_test[i], rf.predict(x_explain.reshape(1, -1)))

    explainer_mani = LimeTabularExplainerManifold(x_train, sample_around_instance=True, manifold='kernel-pca-kde',
                                                  feature_names=data.feature_names, class_names=['class'],
                                                  categorical_features=categorical_features, verbose=True,
                                                  mode='classification')

    exp_mani = explainer_mani.explain_instance_manifold(x_explain, rf.predict_proba, num_features=4)

    ax = explainer_mani.plot(x_train)
    explainer_mani.plot_samples(ax)
    explainer_mani.plot(x_explain.reshape(1, -1), ax)
    plt.show()


