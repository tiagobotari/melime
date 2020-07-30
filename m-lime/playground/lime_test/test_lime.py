# import sys
# sys.path.append('../..')

from sklearn.datasets import load_boston
import sklearn.ensemble
import numpy as np

boston = load_boston()

rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(
    boston.data, boston.target, train_size=0.80)

rf.fit(train, labels_train)
print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))
print('MSError when predicting the mean', np.mean((labels_train.mean() - labels_test) ** 2))

categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()

from ... import lime
from ...lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    train, feature_names=boston.feature_names
    , class_names=['price'], categorical_features=categorical_features, verbose=True, mode='regression')
i = 25
exp = explainer.explain_instance(test[i], rf.predict, num_features=5)