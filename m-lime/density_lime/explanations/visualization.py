import numpy as np
from matplotlib import pyplot as plt


def plot_importance(importance, shape=None, standardization=False, ):
    # TODO: Normalize the importance, give the option
    # TODO: add a variable shape
    if standardization:
        importance = importance/np.std(importance)

    if shape is None:
        n_importance = importance.shape
        n_size = int(np.sqrt(n_importance))
        shape = (n_size, n_size)

    max_importance = np.max(importance)
    min_importance = np.min(importance)

    importance = importance.reshape(28, 28)
    # importance = np.transpose(importance)
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)
    ax.reshape(-1)

    plot_importance_ax(
        fig, ax[0], importance, title='General Importance')

    positive = importance.copy()
    positive[positive < 0] = 0
    plot_importance_ax(
        fig, ax[1], positive, title='Positive Contribution')

    negative = importance.copy()
    negative[negative > 0] = 0
    plot_importance_ax(
        fig, ax[2], negative, title='Negative Contribution')

    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.95, 0.25, 0.04, 0.5])
    # fig.colorbar(cp, cax=cbar_ax)
    return fig, ax


def plot_importance_ax(fig, ax, importance, title, **kwarg):
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    cp = ax.imshow(
        importance, interpolation='none', cmap='jet', **kwarg)  #, origin='lower')
    fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.01)
    return cp
