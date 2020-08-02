from sklearn.linear_model import HuberRegressor as Linear
# HuberRegressor, LinearRegression, SGDRegressor
def plot_local_predictions_(explainer_manifold, x_explain, rf):
    model_linear = Linear()
    print(x_explain)
    x_sample = explainer_manifold.data
    scaled_x_sample = (x_sample - explainer_manifold.scaler.mean_) / explainer_manifold.scaler.scale_
    y_p = rf.predict(x_sample)
    y_sample = np.empty(y_p.shape[0])
#     print('class to explain:', y_explain_index)
    y_sample = y_p  #[:, y_explain_index]

    scaled_var = False
    if scaled_var:
        model_linear.fit(scaled_x_sample, y_sample)
        y_linear = model_linear.predict(scaled_x_sample)
    else:
        model_linear.fit(x_sample, y_sample)
        y_linear = model_linear.predict(x_sample)

    importance = model_linear.coef_.reshape(-1)
    print('>>>>>>>>>>>>>>', model_linear.predict(x_explain.reshape(1, -1)))
    print(data.feature_names)
    print(importance)
    fig, axis = plt.subplots(1, 4, figsize=(15, 5))
    for j in range(x_sample.shape[1]):
        x_j = x_sample[:, j]
        axis[j].scatter(x_j, y_sample)
        axis[j].scatter(x_j, y_linear, s=1)
        axis[j].set_xlabel(data.feature_names[j])
    plot_prediction_sampling(rf, explainer_manifold)


def plot_density_target():
    classes_predicted_unique = np.unique(df_train['predicted'].values)
    columns = list(data.feature_names)
    fig, axis = plt.subplots(1, 4, figsize=(20, 5))
    axis = axis.reshape(-1)
    for i, (ax, col_i) in enumerate(zip(axis, columns)):
        for class_i in classes_predicted_unique:
            subset = df_train[df_train['predicted'] == class_i]
            sns.distplot(subset[col_i], hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3},
                         label=class_i, ax=ax)
            ax.axvline(x=x_explain[i], c='red')
        ax.set_xlabel(col_i, )
    plt.show()

    # from itertools import permutations
    # from sklearn.neighbors import KernelDensity

    # cols_per = permutations(columns, 2)
    # for i, (col_1, col_2) in enumerate(cols_per):
    #     if i == len(columns):
    #         break
    #     cs = plt.contourf(h, #levels=[10, 30, 50],
    #     extend='both', cmap='coolwarm')
    #     cs.cmap.set_over('red')
    #     cs.cmap.set_under('blue')
    #     cs.changed()
    #     plt.colorbar(cs)


def plot_prediction_sampling(rf, explainer_manifold):
    y_p = rf.predict_proba(explainer_manifold.data)
    axis = None
    for class_i in classes_predicted_unique:
        subset = df_train[df_train['predicted'] == class_i]
        if axis is None:
            axis = explainer_manifold.plot(subset[data.feature_names].values, figsize=(15, 15), alpha=0.6)
        else:
            explainer_manifold.plot(subset[data.feature_names].values, axis, alpha=0.6)

    # explainer_manifold.plot(df_train[data.feature_names].values, axis)
    for i in range(4):
        for j in range(4):
            if i == j:
                pass
            else:
                x = explainer_manifold.data[:, j]
                y = explainer_manifold.data[:, i]
                cp = axis[i, j].scatter(x, y, edgecolors='none', c=y_p[:, y_explain_index].reshape(-1), alpha=1.0,
                                        cmap='brg', s=1)

    axis = explainer_manifold.plot(x_explain.reshape(1, -1), axis)
    plt.colorbar(cp)