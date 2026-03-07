import pickle
import json

from sklearn import datasets
from sklearn.model_selection import train_test_split


def save_json(file_name, data):
    with open(f"{file_name}.json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(file_name):
    with open(f"{file_name}.json", "r", encoding='utf-8') as f:
        return json.load(f)


def save_pickle(file_name, obj):
    with open(f"{file_name}.pickle", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(f"{file_name}.pickle", "rb") as handle:
        return pickle.load(handle)


def load_data():
    data = datasets.load_iris()
    x_all = data.data
    y_all = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, train_size=0.80, random_state=123)
    feature_names, target_names = data.feature_names, data.target_names
    return x_train, y_train, x_test, y_test, feature_names, target_names
