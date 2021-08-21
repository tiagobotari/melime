import numpy as np
from .estimator import create_estimator

from matplotlib import rcParams
from matplotlib import pyplot as plt

from melime.generators.kde_gen import KDEGen


# TODO: remover += for list and use append. profile python to find slow parts
def config_matplotlib():
    font_size = 10
    plt.rc('font', serif='Times')
    rcParams['axes.labelsize'] = font_size * 1.5
    rcParams['xtick.labelsize'] = font_size
    rcParams['ytick.labelsize'] = font_size
    rcParams['legend.fontsize'] = font_size
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.right'] = True
    rcParams['lines.linewidth'] = 2
    rcParams['font.family'] = 'times'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = False  # if latex is available change to True

    rcParams['axes.linewidth'] = 1.0

    rcParams['xtick.top'] = True  # draw ticks on the top side
    rcParams['xtick.major.size'] = 6.0
    rcParams['xtick.minor.size'] = 4.0
    rcParams['xtick.major.width'] = 1.0
    rcParams['xtick.minor.width'] = 0.8
    rcParams['xtick.direction'] = 'in'

    rcParams['ytick.left'] = True  # draw ticks on the left side
    rcParams['ytick.right'] = True  # draw ticks on the right side
    rcParams['ytick.labelleft'] = True  # draw tick labels on the left side
    rcParams['ytick.labelright'] = False  # draw tick labels on the right side
    rcParams['ytick.major.size'] = 6.0  # major tick size in points
    rcParams['ytick.minor.size'] = 4.0  # minor tick size in points
    rcParams['ytick.major.width'] = 1.0  # major tick width in points
    rcParams['ytick.minor.width'] = 0.8
    rcParams['ytick.direction'] = 'in'


def plot_information_metrics(result, axis=None, label=None):
    config_matplotlib()

    r = result['r_mean']
    if axis is None:
        fig, axis = plt.subplots(2, 2, figsize=(10, 8))
    else:
        fig = None
    axis = axis.flatten()
    axis[0].errorbar(r, result['d_kl_mean'], yerr=result['d_kl_std'], fmt='-o', capthick=2, capsize=2, label=label)
    axis[0].set_ylabel('KL Divergence')
    axis[1].errorbar(r, result['d_js_mean'], yerr=result['d_js_std'], fmt='-o', capthick=2, capsize=2)
    axis[1].set_ylabel('JS Divergence')
    axis[2].errorbar(r, result['mean_mean'], yerr=result['mean_std'], fmt='-o', capthick=2, capsize=2)
    axis[2].set_ylabel('Predicted Mean')
    axis[3].errorbar(r, result['std_mean'], yerr=result['std_std'], fmt='-o', capthick=2, capsize=2)
    axis[3].set_ylabel('Predicted Standard Desviation')
    axis[0].legend()
    for ax in axis:
        ax.set_xlabel('Radius - $r$')
    return fig, axis


def create_estimators(
        x_explain, generator, predict_proba, bins=1000,
        r_initial=0.01, r_final=0.5, r_delta=0.01,
        r_vec_idx=0, vec_r=(1.0, 0.0),
        n_samples=100000, verbose=0,
        # vec_bandwidth=None
        vec_r_fraction=False
):
    # TODO: Maybe I am not using this function.
    estimators = []
    for r in np.arange(r_initial, r_final + r_delta, r_delta):
        if verbose:
            print(f'{r:3.2f}', end='-')
        estimators += [create_estimator(
            x_explain, generator, predict_proba, r, r_vec_idx, vec_r, n_samples=n_samples
            # , vec_bandwidth=None
            , bins=bins
            , vec_r_fraction=vec_r_fraction
        )]

    return estimators


def experiment(
        x_explain,
        generator,
        predict_proba,
        bins=1000,
        r_initial=0.0002, r_final=0.1, r_delta=0.001,
        vec_r=(0.99, 0.01), r_vec_idx=0,
        n_samples=int(1e6),
        vec_r_fraction=False
):

    if isinstance(bins, int):
        bins_min, bins_max = estimation_bins(x_explain, generator, predict_proba, r_final, n_samples=int(1e6))
        bins = np.linspace(bins_min, bins_max, bins)

    estimators = create_estimators(
        x_explain, generator, predict_proba, r_initial=r_initial, r_final=r_final, r_delta=r_delta,
        vec_r=vec_r, r_vec_idx=r_vec_idx, n_samples=n_samples, bins=bins, vec_r_fraction=vec_r_fraction
    )
    return calculate(estimators[0], estimators[0:]), dict(estimators=estimators)


def average_experiments(
        x_explain,
        x_train,
        var_type,
        predict_proba,
        r_initial, r_final, r_delta,
        vec_r, r_vec_idx,
        bins=1000,
        n_samples=10000, n_repetition_experiment=1,
        vec_r_fraction=False
):
    generator = KDEGen(var_type=var_type).fit(x_train=x_train)
    results = []
    for i in range(n_repetition_experiment):
        print('-', end='')
        result, estimator = experiment(
            x_explain=x_explain,
            generator=generator,
            predict_proba=predict_proba,
            bins=bins,
            vec_r=vec_r,
            r_initial=r_initial, r_final=r_final, r_delta=r_delta,
            n_samples=n_samples,
            r_vec_idx=r_vec_idx,
            vec_r_fraction=vec_r_fraction
        )
        results += [result]

    result_stat = dict()
    for key in results[0].keys():
        if results[0][key] is None:
            result_stat[f"{key}_mean"] = None
            result_stat[f"{key}_std"] = None
            continue
        n_instances = len(results[0][key])
        values = np.empty(shape=(0, n_instances))
        for result_i in results[:]:
            values = np.r_[values, np.array(result_i[key]).reshape(-1, n_instances)]
        result_stat[f"{key}_mean"] = np.mean(values, axis=0)
        result_stat[f"{key}_std"] = np.std(values, axis=0)
    return result_stat


def calculate(estimator0, estimators):
    if estimator0.success is False:
        return dict(r=[estimator0.r], d_kl=None, d_js=None, mean=None, std=None)
    d_kl = []
    d_js = []
    mean = []
    std = []
    r = []
    for estimator in estimators:
        if estimator.success is False:
            d_kl += [None]
            d_js += [None]
        result = estimator.calculate_relative_measurements(estimator0)
        r += [estimator.r]
        d_kl += [estimator.jensen_shannon_divergence]
        d_js += [estimator.kullback_leibler_divergence]
        mean += [estimator.mean]
        std += [estimator.std]
    return dict(r=r, d_kl=d_kl, d_js=d_js, mean=mean, std=std)


def estimation_bins(x_explain, generator, predict_proba, r_max, n_samples=int(1e5)):
    vec_r1 = np.array([r_max]*x_explain.shape[1])

    x = generator.sample_radius(
        x_explain, r=vec_r1, n_samples=n_samples, include_explain=True)
    # weights = generator.weights(x_explain, x, vec_r1)
    # x = generator.sample_radius(x_explain, r=1.0, vec_r=vec_r1, n_samples=n_samples, include_explain=True)
    y = predict_proba(x)
    # TODO: Select the maximum from each dimension, not in the whole ball. Fix code below.
    # for i in range(x_explain.shape[1]):
    #     x, weights = generator.sample_radius(
    #         x_explain, bandwidth=vec_r1, n_samples=n_samples, include_explain=True)
    #     # x = generator.sample_radius(x_explain, r=1.0, vec_r=vec_r1, n_samples=n_samples, include_explain=True)
    #     y = predict_proba(x)

    return np.min(y), np.max(y)

# def figure_size():
#     WIDTH = 350.0  # the number latex spits out
#     FACTOR = 0.45  # the fraction of the width you'd like the figure to occupy
#     figwidthpt = WIDTH * FACTOR
#
#     inchesperpt = 1.0 / 72.27
#     golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
#
#     figwidthin = fig_width_pt * inches_per_pt  # figure width in inches
#     figheightin = fig_width_in * golden_ratio  # figure height in inches
#     fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
#     fig = plt.figure(figsize=fig_dims)
#     return fig


def calculate_plot(x_explain, y_explain, model):
    r_initial = 0.2
    r_final = 1.0
    r_delta = 0.1
    n_samples = int(1e5)
    vec_r_base = (r_initial, r_initial, r_initial, r_initial)
    predict_proba = lambda x: model.predict_proba(x)[:, y_explain].reshape(-1, 1)

    results = []
    for col in range(x_explain.shape[1]):
        results += [average_experiments(
            x_explain=x_explain,
            x_train=x_train, var_type='cccc',
            predict_proba=predict_proba,
            r_initial=r_initial, r_final=r_final, r_delta=r_delta,
            vec_r=vec_r_base, r_vec_idx=col, n_samples=n_samples, n_repetition_experiment=1, vec_r_fraction=False)]

    fig, axis = plot_information_metrics(results[0], label='x0')
    _ = plot_information_metrics(results[1], axis=axis, label='x1')
    _ = plot_information_metrics(results[2], axis=axis, label='x2')
    _ = plot_information_metrics(results[3], axis=axis, label='x3')
    # axis[0].set_ylim(0.5, 0.7)

    fig.tight_layout(pad=1.5)
    plt.show()
