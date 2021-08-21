import numpy as np

from .estimator import Estimator, create_estimator
from .measurements import correlation


def set_up_test_interpretability_methods(
        explainer_lime, explainer_melime, explainer_shap, model_predict, x_train, classification=True):
    def get_importance_lime(
            x_explain, predicted_class):
        exp = explainer_lime.explain_instance(x_explain[0], model_predict)
        dict_imp = {e[0]: e[1] for e in exp.as_list()}
        importance = np.array([dict_imp[e] for e in exp.domain_mapper.feature_names])

        return importance / np.sum(np.abs(importance)), exp.local_pred[0]

    def get_importance_melime(
            x_explain, predicted_class,
            r=0.5,
            n_samples=1000,
            tol_importance=0.01,
            tol_error=0.05,
            local_mini_batch_max=40,
            weight_kernel='gaussian'
    ):
        explanation, counterfactual_examples = explainer_melime.explain_instance(
            x_explain=x_explain.reshape(1, -1),
            class_index=predicted_class,
            r=r,
            n_samples=n_samples,
            tol_importance=tol_importance,
            tol_error=tol_error,
            local_mini_batch_max=local_mini_batch_max,
            weight_kernel=weight_kernel
        )
        y_local_p = explanation.predict(x_explain)
        y_local_p = np.reshape(y_local_p, newshape=1)[0]
        importance = explanation.importance
        return importance / np.sum(np.abs(importance)), y_local_p

    def get_importance_shap(
            x_explain, predicted_class):
        if classification:
            f = lambda x: model_predict(x)[:, predicted_class]
        else:
            f = lambda x: model_predict(x)
        ex_shap = explainer_shap(f, x_train)
        shap_values = ex_shap(np.atleast_2d(x_explain))
        y_local_p = None
        return shap_values.values.reshape(-1), y_local_p

    def get_estimators(
            x_explain, predicted_class, n_samples, vec_r, generator, bins, classification=classification,
            r_perturbation=0.5):
        if classification:
            predict_proba = lambda x: model_predict(x)[:, predicted_class].reshape(-1, 1)
        else:
            predict_proba = lambda x: model_predict(x)
        estimators = []
        # x = generator.sample_radius(
        #     x_explain, r=vec_r, n_samples=n_samples, include_explain=True)
        # y = predict_proba(x).reshape(-1)
        # weights = generator.weights(x_explain, x, vec_r)
        # estimator0 = Estimator(
        #     x_explain, x, y, r=vec_r, weights=weights, method='histogram', parameters={'bins': bins})

        estimator0 = create_estimator(
            x_explain, generator, predict_proba, r=vec_r[0], r_vec_idx=0, vec_r=vec_r, n_samples=n_samples
            , bins=bins
        )

        estimators.append(estimator0)
        for col in range(x_explain.shape[1]):
            estimators += [create_estimator(
                x_explain, generator, predict_proba, r=r_perturbation, r_vec_idx=col, vec_r=vec_r, n_samples=n_samples
                , bins=bins
            )]
        return estimators

    return dict(
        lime=get_importance_lime,
        melime=get_importance_melime,
        shap=get_importance_shap,
        estimator=get_estimators
    )


def get_importance_fidelity_measurements(method, x_set, model, classification=True, **kwargs):
    importance_measurements = []
    fidelity_measurements = []
    for x_i in x_set:
        x_exp = x_i.reshape(1, -1)
        if classification:
            predicted_class = model.predict(x_exp)[0]
            y_p = model.predict_proba(x_exp)[0, predicted_class]
            importance, y_local_p = method(x_exp, predicted_class, **kwargs)
        else:
            y_p = model.predict(x_exp)[0]
            importance, y_local_p = method(x_exp, 0, **kwargs)
        if y_local_p is not None:
            fidelity_measurements.append(get_fidelity(y_p, y_local_p))
        importance_measurements.append(importance)
    fidelity_measurements = np.array(fidelity_measurements)
    return importance_measurements, fidelity_measurements


def get_fidelity(y_p, y_local_p):
    return y_local_p - y_p


def estimator_correlation(estimators):
    # Calculate Correlation
    c = []
    for col, estimator in enumerate(estimators):
        c.append(correlation(estimator.x_sampled[:, col], estimator.y, estimator.weights))
    return normalize(c)


def normalize(values):
    values = np.array(values)
    return values / np.sum(np.abs(values))


def distance_order(values1, values2, weighted=True):
    values_abs = np.abs(normalize(values1))
    order1 = np.argsort(values_abs)
    order2 = np.argsort(np.abs(values2)).tolist()
    diff = np.array([np.abs(i - order2.index(o)) for i, o in enumerate(order1)])
    if weighted:
        diff = diff * values_abs
    order_distance = np.mean(diff)
    return order_distance


def distance(values1, values2):
    values1 = normalize(values1)
    values2 = normalize(values2)
    return np.linalg.norm(values1 - values2)


def calculates_distances(corr, d_js, importance):
    # Correlation
    coor_distance = distance(corr, importance)
    coor_order = distance_order(corr, importance)
    # DJ divergence
    d_js_order = distance_order(d_js, np.abs(importance))
    d_js_distance = distance(d_js, np.abs(importance))
    return [coor_distance, coor_order, d_js_distance, d_js_order]


def calculate_distance_many(corr_l, d_js_l, importances):
    results = []
    for coor, d_js, importance in zip(corr_l, d_js_l, importances):
        results.append(calculates_distances(coor, d_js, importance))
    return np.array(results)


def calculate_js_divergence(estimator0, estimators):
    if estimator0.success is False:
        return dict(r=[estimator0.r], d_kl=None, d_js=None, mean=None, std=None)
    d_js = []
    r = []
    for estimator in estimators:
        if estimator.success is False:
            d_js += [None]
        _ = estimator.calculate_relative_measurements(estimator0)
        r += [estimator.r]
        d_js += [estimator.kullback_leibler_divergence]
    return d_js
