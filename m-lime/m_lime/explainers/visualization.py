import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ImagePlot(object):
    @classmethod
    def plot_importance_negative_positive(cls, importance, shape=None, standardization=False):
        # TODO: Normalize the importance, give the option
        # TODO: add a variable shape
        if standardization:
            importance = importance / np.std(importance)
        if shape is None:
            n_importance = importance.shape
            n_size = int(np.sqrt(n_importance))
            shape = (n_size, n_size)

        max_importance = np.max(importance)
        min_importance = np.min(importance)
        max_scale = np.max(np.abs([max_importance, max_importance]))
        
        importance = importance.reshape(shape)

        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)
        ax.reshape(-1)

        cmap_ = cls.color_map()
        cp_importance = cls.plot_importance_(
            importance=importance, 
            title="General Importance", 
            fig=fig, ax=ax[0],
            cmap=cmap_,
            vmax=max_scale, 
            vmin=-max_scale
        )

        positive = importance.copy()
        positive[positive < 0] = 0
        cls.plot_importance_(
            importance=positive,
            title="Positive Contribution", 
            fig=fig, ax=ax[1], 
            cmap=cmap_,
            vmax=max_scale, 
            vmin=-max_scale
        )

        negative = importance.copy()
        negative[negative > 0] = 0
        cls.plot_importance_(
            importance=negative, 
            title="Negative Contribution", 
            fig=fig, ax=ax[2],
            cmap=cmap_, 
            vmax=max_scale, 
            vmin=-max_scale
        )
        # left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1
       
        return fig, ax
    
    @classmethod
    def plot_importance(cls, importance, shape=None, standardization=False):
        # TODO: Normalize the importance, give the option
        # TODO: add a variable shape
        if standardization:
            importance = importance / np.std(importance)
        if shape is None:
            n_importance = importance.shape
            n_size = int(np.sqrt(n_importance))
            shape = (n_size, n_size)

        max_importance = np.max(importance)
        min_importance = np.min(importance)
        max_scale = np.max(np.abs([max_importance, max_importance]))
        
        importance = importance.reshape(shape)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

        cmap_ = cls.color_map()
        cp_importance = cls.plot_importance_(
            importance=importance, 
            title="General Importance", 
            fig=fig, ax=ax,
            cmap=cmap_,
            vmax=max_scale, 
            vmin=-max_scale
        )

        return fig, ax

    @staticmethod
    def color_map():
        colors = cm.get_cmap('bwr', 200)
        scale_color = [*range(0, 50, 1)]+[*range(50, 80, 8)]
        scale_color1 = [*range(120, 120+30, 5)]+[*range(120+30, 200, 1)]
        newcolors = colors(scale_color+[*range(98,103)]+scale_color1)
        newcmp = matplotlib.colors.ListedColormap(newcolors)
        return newcmp

    @staticmethod
    def plot_importance_(importance, title, fig, ax, **kwarg):
        ax.set_title(title, fontsize=20)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        # plt.setp(ax.get_xticklabels(), visible=False)
        # plt.setp(ax.get_yticklabels(), visible=False)
        cp = ax.imshow(
                importance,
                interpolation="none",
                norm=matplotlib.colors.DivergingNorm(0),  
                **kwarg
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(cp, cax=cax)  #, fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=15)
        return cp

    @classmethod
    def plot_instances(cls, ax=None, x_=None, y_=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_title(y_, fontsize=18)
        colors = matplotlib.cm.get_cmap('Greys', 200)
        scale_color = [*range(0, 100, 8)]+[*range(100, 200, 1)]
        newcolors = colors(scale_color)
        newcmp = matplotlib.colors.ListedColormap(newcolors)
        return ax.imshow(x_, interpolation = 'none', cmap=newcmp)

class GridPlot(object):
    @classmethod
    def plot_samples(cls, ax, model=None, **kwargs):
        if cls.data is not None:
            if model is None:
                return cls.plot(cls.data, ax, s=1, **kwargs)
            else:
                return cls.plot(cls.data, ax, s=1, y=model.predict(cls.data), **kwargs)

    @classmethod
    def plot(
        cls,
        x,
        x_cols_name=None,
        plot_cols="all",
        y=None,
        y_names={},
        y_discrete=True,
        fig=None,
        ax=None,
        alpha=1.0,
        figsize=(10, 10),
        **kwargs
    ):
        """

        :param x:
        :param x_cols_name:
        :param plot_cols:
        :param y:
        :param y_names:
        :param y_discrete:
        :param fig:
        :param ax:
        :param alpha:
        :param figsize:
        :param kwargs:
        :return:
        """
        if x_cols_name is None:
            x_cols_name = np.arange(0, x.shape[1])
        if plot_cols == "all":
            plot_cols = np.arange(0, len(x_cols_name))

        if y_discrete:
            unique_y = np.unique(y)
            indices_plot = {y_i: np.argwhere(y == y_i) for y_i in unique_y}

        x_cols_name = np.array(x_cols_name)
        feature_names = x_cols_name[plot_cols]

        n_cols = len(feature_names)
        indices_cols = plot_cols

        selections = list(itertools.combinations_with_replacement(indices_cols, 2))

        if ax is None:
            fig, ax = plt.subplots(n_cols, n_cols, sharex="col", sharey="row", squeeze=False, figsize=figsize)
        for i, sel in enumerate(selections):
            col1 = sel[0]
            col2 = sel[1]
            axi = ax[col1, col2]
            axi_inv = ax[col2, col1]
            # configure
            axi_inv.grid(alpha=0.2)
            axi.grid(alpha=0.2)
            axi.tick_params(direction="in", which="both", top=True, right=True)
            axi_inv.tick_params(direction="in", which="both", top=True, right=True)
            if col1 == col2:
                # Plot Histogram
                x_ = x[:, col2]
                hist, bins, bin_edges, normalization = histogram(x_)
                if len(bins) > 1:
                    width = bins[0] - bins[1]
                else:
                    width = np.max(x) - np.min(x)
                if y_discrete:
                    color = "lightgray"
                else:
                    color = None
                main_his = axi.bar(bins, hist, align="center", edgecolor="lightgray", color=color, width=width)

                # Plot histogram per class of y
                if y_discrete:
                    width = width * 0.8
                    for y_i, indices_i in indices_plot.items():
                        x_ = x[indices_i, col2]
                        hist, bins, _, _ = histogram(
                            x_, bins=bin_edges, normalization=normalization, bin_edges=bin_edges
                        )
                        label_ = y_names.get(y_i, y_i)

                        axi.bar(bins, hist, align="center", label=label_, edgecolor=None, width=width, alpha=0.8)

            elif y is not None:
                if y_discrete:
                    for y_i, indices_i in indices_plot.items():
                        x_ = x[indices_i, col2]
                        y_ = x[indices_i, col1]

                        label_ = y_names.get(y_i, y_i)

                        cp = axi.scatter(x_, y_, label=label_, alpha=alpha, **kwargs)
                        cp = axi_inv.scatter(y_, x_, label=label_, alpha=alpha, **kwargs)
                else:
                    x_ = x[:, col2]
                    y_ = x[:, col1]
                    cp = axi.scatter(x_, y_, alpha=alpha, c=y, **kwargs)
                    cp = axi_inv.scatter(y_, x_, alpha=alpha, c=y, **kwargs)
            else:
                x_ = x[:, col2]
                y_ = x[:, col1]
                cp = None
                axi.scatter(x_, y_, alpha=alpha, c=y, **kwargs)
                axi_inv.scatter(y_, x_, alpha=alpha, c=y, **kwargs)

        for i, label in enumerate(feature_names):
            ax[-1, i].set_xlabel(label, fontdict={"fontsize": 12})
            ax[i, 0].set_ylabel(label, fontdict={"fontsize": 12})

        if y_discrete:
            handles, labels = ax[0, 1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="right")
            # ax[0, 1].legend(loc='upper center', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        if cp is not None:
            return ax, cp
        return ax

    @classmethod
    def plot_instance(cls, x_explain, axis):
        for i in range(axis.shape[0]):
            for j in range(axis.shape[1]):
                if i == j:
                    axis[i, j].axvline(x_explain[j], c="red", alpha=0.8)
                else:
                    axis[i, j].scatter([x_explain[j]], [x_explain[i]], s=150, c="red", marker="x")
                    axis[i, j].scatter([x_explain[j]], [x_explain[i]], s=40, c="red")


def histogram(x, bins=15, normalization=None, bin_edges=None):
    hist, bin_edges_ = np.histogram(x, bins=bins)
    if bin_edges is None:
        bin_edges = bin_edges_
    bins = []
    for i, edge in enumerate(bin_edges[:-1]):
        bins.append((edge + bin_edges[i + 1]) / 2)
    if normalization is None:
        normalization = np.max(hist)
    hist = (hist / normalization) * bin_edges[-1]
    return hist, bins, bin_edges, normalization


if __name__ == "__main__":
    # Simple test!
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize

    x = np.random.rand(1000) * 10
    norm1 = x / np.linalg.norm(x)
    norm2 = normalize(x[:, np.newaxis], axis=0).ravel()
    np.all(norm1 == norm2)
    # True

    df = pd.DataFrame(
        {
            "target": np.random.choice([0, 1, 2, 3], size=40),
            "x1": np.random.rand(40) * 5,
            "x2": np.random.rand(40) * 3,
            "x3": np.random.rand(40),
        }
    )
    x = df[["x1", "x2", "x3"]].values
    y = df["target"]

    y_names = {i: "label {:}".format(i) for i in range(4)}
    ax, cp = GridPlot.plot(x, x_cols_name=["x1", "x2", "x3"], plot_cols=[0, 1, 2], y=y, y_names=y_names)
    # plt.colorbar(cp)
    plt.legend()
    plt.show()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    origin: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap