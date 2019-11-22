# import sys

import numpy as np
import sklearn.ensemble
from matplotlib import pyplot as plt

from playground.aux.domain import SimpleSpiral

data = SimpleSpiral()
x, y = data.domain()

fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1], c=y)
plt.show()


rf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, train_size=0.80)

rf.fit(x_train, y_train)
print('Random Forest MSError', np.mean((rf.predict(x_test) - y_test) ** 2))
print('MSError when predicting the mean', np.mean((y_train.mean() - y_test) ** 2))


from density_lime.lime_tabular_manifold import LimeTabularExplainerManifold
#
categorical_features = []
explainer = LimeTabularExplainerManifold(
    x_train, feature_names=['x1', 'x2']
    , class_names=['target']
    , categorical_features=categorical_features
    , verbose=True
    , mode='regression'
    , manifold='kde')

i = 25
exp = explainer.explain_instance_manifold(x_test[i], rf.predict, num_features=5)
exp.as_pyplot_figure()
plt.show()