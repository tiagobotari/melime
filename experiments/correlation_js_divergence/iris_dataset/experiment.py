import os
import pickle

import numpy as np
import shap
import sklearn
import lime.lime_tabular

from melime.analysis import test_interpretability
from melime.explainers.explainer import Explainer
from melime.generators.kde_gen import KDEGen

from util import save_pickle, load_data, load_json

# Load Data
x_train, y_train, x_test, y_test, feature_names, target_names = load_data()
# Load Model
rf = sklearn.ensemble.RandomForestClassifier()
filename = 'rf_model_iris.bin'
if os.path.exists(filename):
    rf = pickle.load(open(filename, 'rb'))


generator = KDEGen(verbose=0).fit(x_train)

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=feature_names, categorical_features=[],
    mode='classification', discretize_continuous=False
)

explainer_melime = Explainer(
    model_predict=rf.predict_proba,
    generator=generator,
    local_model='Ridge',
    feature_names=feature_names,
    target_names=target_names
)

explainer_shap = shap.Explainer

model_predict = rf.predict_proba

methods = test_interpretability.set_up_test_interpretability_methods(
    explainer_lime,
    explainer_melime,
    explainer_shap,
    model_predict,
    x_train,
    classification=True
)

get_estimators = methods['estimator']
get_importance_melime = methods['melime']
get_importance_lime = methods['lime']
get_importance_shap = methods['shap']

path_data = "./data"
tasks = load_json(f"{path_data}/tasks")

for i in range(len(tasks)):
    task_i = tasks[i]
    index = task_i["index"]
    x_explain = np.array(task_i["x_explain"])
    r_perturbation = task_i["r_perturbation"]
    vec_r_base = task_i["vec_r_base"]
    n_samples = task_i["n_samples"]
    file_name = task_i["file_name"]
    bins = np.array(task_i["bins"])
    predicted_class = rf.predict(x_explain)[0]
    estimators = get_estimators(
        x_explain, predicted_class, n_samples, vec_r=vec_r_base,
        generator=generator, bins=bins, r_perturbation=r_perturbation)
    task_i["estimators"] = estimators
    save_pickle(file_name=f"{path_data}/{file_name}", obj=task_i)
