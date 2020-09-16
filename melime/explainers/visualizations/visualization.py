import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.sparse import issparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from skimage import feature, transform


class ImagePlot(object):
    @classmethod
    def plot_importance_contrafactual(cls, explain_dict, contra, class_name, x_explain=None):
        y = "Prediction: f{np.argmax(y_explain):}"
        importances_ = explain_dict["importances"]

        fig = plt.figure(constrained_layout=False, figsize=(7.0, 3.5))
        offset_x = -0.14
        ax1 = fig.add_axes([0.1 + offset_x, 0.01, 0.5, 0.85])
        offset_x = -0.01
        x1 = 0.45 + offset_x
        w1 = 0.26
        x2 = x1 + w1 + 0.02
        ax2 = fig.add_axes([x1, 0.00, w1, 0.57])
        ax3 = fig.add_axes([x2, 0.00, w1, 0.57])
        ax1.set_title("Importance")
        a, cp_importance = cls.plot_importance(importances_, standardization=True, ax=ax1, x_explain=x_explain,)

        plot = cls.plot_instances(contra.samples_con[0].reshape(28, 28), ax=ax3)
        ax3.set_title(f"Contrary:", fontsize=14)
        plot = cls.plot_instances(contra.samples_fav[0].reshape(28, 28), ax=ax2)
        ax2.set_title(f"Favorable", fontsize=14)
        plt.annotate(
            f"Why is it classified as {class_name}?",
            xy=((x1 + x2 + w1) / 2.0, 0.88),
            xycoords="figure fraction",
            horizontalalignment="center",
            fontsize=22,
        )
        return fig, [ax1, ax2, ax3]

    @staticmethod
    def color_map():
        colors = cm.get_cmap("bwr", 201)
        scale_color = [*range(0, 50, 1)] + [*range(50, 80, 8)]
        scale_color1 = [*range(120, 150, 8)] + [*range(150, 200, 1)]
        newcolors = colors(scale_color + [*range(98, 104)] + scale_color1)
        newcmp = matplotlib.colors.ListedColormap(newcolors)
        # newcmp = colors
        return newcmp

    @classmethod
    def plot_importance(cls, importance, shape=None, standardization=False, ax=None, x_explain=None):
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

        cmap_ = cls.color_map()

        dx, dy = 0.05, 0.05
        xx = np.arange(0.0, importance.shape[1] + dx, dx)
        yy = np.arange(0.0, importance.shape[0] + dy, dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax

        cp_importance, cbar = cls.plot_importance_(
            importance=importance,
            title=" ",
            ax=ax,
            extent=extent,
            cmap=cmap_,
            vmax=max_scale,
            vmin=-max_scale
        )
        cbar.ax.set_xlabel("Importance", fontsize=18, labelpad=2)
        if x_explain is not None:
           cls.plot_mask(data=x_explain, ax=ax, extent=extent)

        return ax, cp_importance

    @staticmethod
    def plot_mask(data, ax, extent=None, dilation=3.0, alpha=0.5):
        data = data[0]
        mean = data if len(data.shape) == 2 else np.mean(data, axis=-1)
        in_image_upscaled = transform.rescale(mean, dilation, mode="constant")
        mask = feature.canny(in_image_upscaled).astype(float)
        mask[mask < 0.5] = np.nan
        # plot
        cmap = plt.get_cmap("Greys_r")
        ax.imshow(mask, extent=extent, interpolation="none", cmap=cmap, alpha=alpha)
        return ax

    @staticmethod
    def plot_importance_(importance, title, ax, extent=None, **kwarg):
        ax.set_title(title, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        cp = ax.imshow(importance, extent=extent, interpolation="none", **kwarg)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(cp, cax=cax, orientation="horizontal", fraction=0.07, anchor=(1.0, 0.0))
        cbar.ax.tick_params(labelsize=13, pad=-1, direction="in")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        return cp, cbar

    @classmethod
    def plot_instances(cls, x_=None, y_=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(y_, fontsize=18)
        colors = matplotlib.cm.get_cmap("Greys", 200)
        scale_color = [*range(0, 100, 8)] + [*range(100, 200, 1)]
        newcolors = colors(scale_color)
        newcmp = matplotlib.colors.ListedColormap(newcolors)
        return ax.imshow(x_, interpolation="none", cmap=newcmp)


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
        figsize=(8, 8),
        colors=None,
        **kwargs,
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
            fig, ax = plt.subplots(n_cols, n_cols, squeeze=False, figsize=figsize)  # sharex='col', sharey='row'
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
                cls._plot_histogram(axi, x, col2, y_names, indices_plot, y_discrete, colors_discrete=colors)
            elif y is not None:
                if y_discrete:
                    for i, (y_i, indices_i) in enumerate(indices_plot.items()):
                        color = colors[i]
                        x_ = x[indices_i, col2]
                        y_ = x[indices_i, col1]

                        label_ = y_names.get(y_i, y_i)

                        cp = axi.scatter(x_, y_, label=label_, alpha=alpha, c=color, **kwargs)
                        cp = axi_inv.scatter(y_, x_, label=label_, alpha=alpha, c=color, **kwargs)
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
            fig.legend(handles, labels, loc="upper center", borderaxespad=0.1, ncol=10)
        plt.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.97)
        if cp is not None:
            return ax, cp
        return ax

    @classmethod
    def plot_histogram(
        cls,
        x,
        x_cols_name=None,
        plot_cols="all",
        y=None,
        y_names={},
        y_discrete=True,
        fig=None,
        axis=None,
        alpha=0.8,
        figsize=(10, 3),
        colors=None,
        **kwargs,
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

        selections = indices_cols

        fig, axis = plt.subplots(1, n_cols, sharex="col", sharey="row", squeeze=False, figsize=figsize)
        axis = axis.flatten()

        for i, col1 in enumerate(selections):

            axi = axis[col1]
            # configure
            axi.grid(alpha=0.2)
            axi.tick_params(direction="in", which="both", top=True, right=True, labelsize=14)
            cls._plot_histogram(
                axi, x, col1, y_names, indices_plot, y_discrete, colors_discrete=colors, alpha=alpha, normalize=False
            )

        for i, label in enumerate(feature_names):
            axis[i].set_xlabel(label, fontdict={"fontsize": 15})

        if y_discrete:
            handles, labels = axis[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=10,
                fontsize=14,
                bbox_to_anchor=(0.0, 0.02, 1.0, 1.0),
                mode=True,
            )
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.96, top=0.85, wspace=0.1, hspace=0.0)
        return fig, axis

    @classmethod
    def _plot_histogram(
        cls, axi, x, col2, y_names, indices_plot, y_discrete, colors_discrete=None, alpha=0.8, normalize=True
    ):
        # Plot Histogram
        x_ = x[:, col2]
        hist, bins, bin_edges, normalization = histogram(x_, normalize=normalize)
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
            width = width * 0.9
            delta_width = 0.9  # 0.8**(1/len(indices_plot.items()))
            if colors_discrete is None:
                colors_discrete = list(mcolors.BASE_COLORS.keys())

            for i, (y_i, indices_i) in enumerate(indices_plot.items()):
                x_ = x[indices_i, col2]
                hist, bins, _, _ = histogram(
                    x_, bins=bin_edges, normalization=normalization, bin_edges=bin_edges, normalize=normalize
                )
                label_ = y_names.get(y_i, y_i)
                axi.bar(
                    bins,
                    hist,
                    align="center",
                    label=label_,
                    edgecolor=None,
                    width=width,
                    alpha=alpha,
                    color=colors_discrete[i],
                )
                width *= delta_width

    @classmethod
    def plot_instance(cls, x_explain, axis):
        for i in range(axis.shape[0]):
            for j in range(axis.shape[1]):
                if i == j:
                    axis[i, j].axvline(x_explain[j], c="red", alpha=0.8)
                else:
                    axis[i, j].scatter([x_explain[j]], [x_explain[i]], s=150, c="red", marker="x")
                    axis[i, j].scatter([x_explain[j]], [x_explain[i]], s=40, c="red")


def histogram(x, bins=15, normalization=None, bin_edges=None, normalize=True):
    hist, bin_edges_ = np.histogram(x, bins=bins)
    if bin_edges is None:
        bin_edges = bin_edges_
    bins = []
    for i, edge in enumerate(bin_edges[:-1]):
        bins.append((edge + bin_edges[i + 1]) / 2)
    if normalize:
        if normalization is None:
            normalization = np.max(hist)
        hist = (hist / normalization) * bin_edges[-1]
    return hist, bins, bin_edges, normalization


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
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
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def label_bar(rects, ax, labels=None):
    colors = ["blue", "orange"]
    for rect, color in zip(rects, colors):
        width = rect.get_width()
        rect.set_color("r")
        ax.annotate(
            "{:3.2f}".format(width),
            xy=(rect.get_width() / 2, rect.get_y() - 0.2 + rect.get_height() / 2),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=30,
        )


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


class ExplainText(object):
    @classmethod
    def plot(cls, importances, words):
        from IPython.core.display import HTML

        # Color scale.
        n = 100
        value = 150
        r_ = np.concatenate(
            (   
                [255] * (4 * n),
                np.linspace(255, value, n),
                np.linspace(value, 75, n),
                np.linspace(75, 50,  n),
                np.linspace(50, 0, n),
            )
        ).astype(np.int)
        g_ = np.concatenate(
            (
                np.linspace(0, 50, n),
                np.linspace(50, 75, n),
                np.linspace(75, value, n),
                np.linspace(value, 255, n),
                np.linspace(255, value, n),
                np.linspace(value, 75, n),
                np.linspace(75, 50, n),
                np.linspace(50, 0, n),
            )
        ).astype(np.int)
        b_ = np.concatenate(
            (
                np.linspace(0, 50, n),
                np.linspace(50, 75, n),
                np.linspace(75, value, n),
                np.linspace(value, 255, n),
                [255] * (4 * n)
            )
        ).astype(np.int)
        
        def f(x):
            result = np.array(4 * n * x + 4 * n)
            result = result.astype(np.int)
            return result

        indices = f(np.array(importances))

        def html_color(indices, words):
            """background-image: linear-gradient(90deg,
            rgba(130, 0, 0, 0.5), rgb(160, 0, 0, 0.5), rgba(255, 0, 0, 0.5),
            rgb(255, 255, 225, 0.5),
            rgba(0, 0, 255, 0.5), rgba(0, 0, 160, 0.5), rgba(0, 0, 130, 0.5))"""
            
            background = f"""
            background-image: linear-gradient(90deg,
            rgb(255, 0, 0),
            rgb(255, 50, 50),
            rgb(255, 75, 75),
            rgb(255, {value}, {value}),
            rgb(255, 255, 255),
            rgb({value}, {value}, 255),
            rgb(75, 75, 255),
            rgb(50, 50, 255),
            rgb(0, 0, 255)
            )"""

            string = ""
            for e, word in zip(indices, words):
                str_ = f"""
                <div style="background-color:rgba({r_[e]}, {g_[e]}, {b_[e]}, 1.0);
                padding:0px 2px 0px 2px;
                margin:2px 2px 0px 0px">
                <span style="color:black; font: 400 8px/0.85rem  serif;">{word} </span>
                </div>
                """
                string += str_
            string = f"""
            <div style="display:flex;  flex-direction:column; align-items:center">
            <span style="color:black">
            Importance
            </span>
            <div style="{background}; width:80%; height:10px">  </div>
            <div style="display: flex; width:82%; flex-direction: row; justify-content: space-between"> 
            <span style="color:black; font: 400 8px/0.85rem  serif;"> -1.0 </span>
            <span style="color:black; font: 400 8px/0.85rem  serif;"> 0.0 </span>
            <span style="color:black; font: 400 8px/0.85rem  serif;"> +1.0 </span>
            </div>

            <div style="display:flex; flex-wrap:wrap"> {string}   
            </div>
            </div>
            """
            return HTML(string)

        return html_color(indices, words)
