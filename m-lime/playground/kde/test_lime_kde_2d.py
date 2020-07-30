sys.path.append('../..')

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import src.lime as lime
import src.lime.lime_tabular
from __future__ import print_function
np.random.seed(1)


iris = sklearn.datasets.load_iris()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)

sklearn.metrics.accuracy_score(y_test, rf.predict(x_test))



explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

i = np.random.randint(0, x_test.shape[0])
exp = explainer.explain_instance(x_test[i], rf.predict_proba, num_features=2, top_labels=1)

exp.show_in_notebook(show_table=True, show_all=False)

