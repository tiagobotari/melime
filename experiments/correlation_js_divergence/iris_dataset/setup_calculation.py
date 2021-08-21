import os.path

import numpy as np


# Load Data
from experiments.correlation_js_divergence.iris_dataset.util import load_data, save_json

x_train, y_train, x_test, y_test, feature_names, target_names = load_data()

bins = np.linspace(0, 1.0, 100)
vec_r_base = (0.2, 0.2, 0.2, 0.2)
r_perturbation = 0.5
n_samples = int(1e6)
path_data = "./data"
estimators_test = []
task_list = []
for i, x_i in enumerate(x_test[:10]):
    obj_estimators = dict(
        index=i,
        x_explain=x_i.reshape(1, -1).tolist(),
        r_perturbation=r_perturbation,
        vec_r_base=vec_r_base,
        n_samples=n_samples,
        bins=bins.tolist(),
        estimators=[],
        file_name=f"{i}.x_test_estimators"
    )
    task_list.append(obj_estimators)
save_json(f"{path_data}/tasks", task_list)
