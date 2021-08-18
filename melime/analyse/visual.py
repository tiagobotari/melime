import numpy as np
from matplotlib import pyplot as plt

from melime.explainers.visualizations.visualization import GridPlot


def plot_grid(x_explain, x, y, names=None, feature_names=None):
    names, c = get_names_color(names, y)
    axis, _ = GridPlot.plot(
        x=x, x_cols_name=feature_names, y=y, y_names=names, colors=c, alpha=0.2)
    # axis, _ = GridPlot.plot(
    #     x=x_all, x_cols_name=feature_names, y=y_all, y_names=names, colors=list(colors.values()), alpha=0.2)
    GridPlot.plot_instance(x_explain[0], axis)
    GridPlot.plot_instance(x_explain[0], axis)
    for ax in axis.ravel():
        start, end = ax.get_xlim()
        stepsize = 1
        ax.xaxis.set_ticks(np.arange(int(start), end, stepsize))
    return


def plot_samples(x_sampled, x_explain=None, normalize=True):
    if len(x_sampled.shape) == 1:
        x_sampled = x_sampled.reshape(-1, 1)
    n_instances = x_sampled.shape[0]
    n_features = x_sampled.shape[1]
    cols = int(np.sqrt(n_features))
    if cols ** 2 < n_features:
        rows = cols + 1
    else:
        rows = cols
    fig, axis = plt.subplots(rows, cols, figsize=(15, 5))
    if isinstance(axis, np.ndarray) is False:
        axis = np.array([axis])
    for i, ax in enumerate(axis.ravel()):
        plt.setp(ax.get_xticklabels(), fontsize=15)
        hist, bin_edges = np.histogram(x_sampled[:, i], bins=100)
        bins = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        width = bin_edges[0] - bin_edges[1]
        if normalize:
            hist = hist / np.max(hist)
        ax.bar(bins, hist, width=width)
        if x_explain is not None:
            ax.axvline(x_explain[0, i], ymin=0, ymax=1, c='black')
        mean = np.mean(x_sampled[:, i])
        ax.axvline(mean, ymin=0, ymax=1, c='gray')
        ax.set_title(f'{mean} +- {np.std(x_sampled[:, i])}')
    fig.tight_layout(pad=2.0)
    return fig, axis


def get_names_color(names=None, y=None, colors=None):
    if names is None:
        names = np.unique(y)
        names = {i: i for i in names}
    c = [colors[i] for i in names]
    return names, c


def plot_histogram_data(x_explain, x, y, names=None, feature_names=None):
    names, c = get_names_color(names, y)
    fig, axis = GridPlot.plot_histogram(
        x=x, x_cols_name=feature_names, y=y, y_names=names, colors=c, alpha=0.6)
    for i, ax in enumerate(axis):
        ax.axvline(x_explain[0][i], c="black", lw=3.0, linestyle=':', alpha=0.8)
        start, end = axis[i].get_xlim()
        axis[i].xaxis.set_ticks(np.arange(int(start), end, 1))
    #     axis[0].set_xlim([4 , 8])
    #     axis[1].set_xlim([1.8,4.5])
    #     axis[2].set_xlim([0.8,7.2])
    #     start, end = axis[3].get_xlim()
    #     axis[3].xaxis.set_ticks(np.arange(int(start), end, 0.5))
    axis[0].set_ylabel('Number of Instances', fontsize=18)
    plt.savefig('histogram_model_predictions.pdf')
