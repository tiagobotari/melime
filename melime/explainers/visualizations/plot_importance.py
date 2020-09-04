import numpy as np
from scipy.sparse import issparse

from matplotlib import pyplot as plt
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


def label_bar(rects, ax, labels=None, offset_y=0.4):
    colors = ["blue", "orange"]
    N = len(rects)
    if N > 28:
        font_size = 10
    else:
        font_size = 14
    for i in range(N):
        rect = rects[i]
        width = rect.get_width()
        if width != 0.0:
            text_width = "{:3.2f}".format(width)
        else:
            text_width = ""
        x = rect.get_width() / 2.0
        if abs(x) <= 0.06:
            x = x / abs(x) * 0.10

        y = (rect.get_y() + rect.get_height() / 2) - 0.225
        xy = (x, y)

        ax.annotate(
            text_width,
            xy=xy,
            xytext=(0, -1),  # 3 points vertical offset
            textcoords="offset points",
            # ha="center",
            va="bottom",
            size=font_size,
            color="black",
            horizontalalignment="center",
        )
        if rect.get_width() > 0:
            aling_text = "right"
            off_setx = -3
        else:
            aling_text = "left"
            off_setx = +3
        if labels is not None:
            text = labels[i]
            ax.annotate(
                text,
                xy=(rect.get_x(), y),
                xytext=(off_setx, -1),  # 3 points vertical offset
                textcoords="offset points",
                horizontalalignment=aling_text,
                verticalalignment="bottom",
                size=font_size,
            )


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


class SparseMatrix(object):
    """
    Transformation to be used in a sklearn pipeline
    check if a array is sparse.
    # TODO: The NLS, LLS, NNPredict should accept sparse array
    """

    def __init__(self):
        pass

    def fit(self):
        return self

    @staticmethod
    def transform(x):
        if issparse(x):
            return x.toarray()
        return x


def plot_betas(dict_exp, title="Atheism Class Feature Importance"):
    fig, ax = plt.subplots()
    width = 0.35
    ind = np.arange(len(dict_exp["words"][::-1]))
    bar_pos = plt.barh(ind - width / 2, dict_exp["betas_document"][::-1], width, label="Atheism")
    plt.title(title)
    bar_neg = plt.barh(ind + width / 2, dict_exp["betas_document_neg"][::-1], width, label="Christian")
    ax.set_yticks(ind)
    ax.set_yticklabels(dict_exp["words"][::-1])
    ax.grid(True, axis="y")
    leg = plt.legend()


class ExplainGraph(object):
    @classmethod
    def plot(cls, explanation):
        size_title = 18
        class_names = explanation["class_names"]
        feature_names = explanation["chi_names"]
        features = explanation["chi_values"]

        importances = explanation["importances"]
        diff_importances = explanation["diff_convergence_importances"]
        errors = explanation["error"]
        y_p = explanation["y_p"]
        y_local_model = explanation["y_p_local_model"]
        y_p_max = explanation["y_p_max"]
        y_p_min = explanation["y_p_min"]

        n_importance = len(importances)

        left = 0.07
        width = 1.0 - left * 2.0

        bottom, height = 0.1, 0.8
        bottom_h = left_h = left + width + 0.02
        h_space = 0.02
        v_space = 0.2

        font_size = 18
        top_margin = 0.95
        bottom_margin = 0.05

        fig = plt.figure(figsize=(12, 10))

        width_rect_left = width / 3.0

        # Prediction plot.
        b_target = 0.79
        h_target = 0.03
        center_target = left + width_rect_left / 2.0
        rect_target = [left, b_target, width_rect_left, h_target]
        ax_target = plt.axes(rect_target)
        #
        pading_title = 0.01

        plt.annotate(
            f"Predicted Target",
            xy=(center_target, top_margin - pading_title),
            xycoords="figure fraction",
            xytext=(0.0, 0.0),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=font_size,
            color="black",
            horizontalalignment="right",
        )
        # plt.annotate(
        #     f"Error: {errors:5.3e}",
        #     xy=(center_target, top_margin - pading_title - 0.06),
        #     xycoords="figure fraction",
        #     xytext=(0.0, 0.0),  # 3 points vertical offset
        #     textcoords="offset points",
        #     ha="center",
        #     va="bottom",
        #     size=int(font_size*0.7),
        #     color="black",
        #     horizontalalignment="right",
        # )
        ax_target = cls.plot_predictions(ax_target, y_p, y_p_min, y_p_max, y_local_model, y_name=class_names[0])

        # Features plot.
        b_features = 0.63
        h_features = 0.01
        rect_features = [left + 0.1, b_features, width_rect_left - 0.2, h_features]
        ax_features = plt.axes(rect_features)
        simpleaxis(ax_features)
        ax_features.set_xticks([])
        ax_features.set_yticks([])

        cell_text = features.reshape(-1, 1)  # [[e] for e in features]
        color = {0: "lightgray", 1: "white"}
        cell_colours = [[color[int(e % 2)]] for e in range(features.shape[1])]
        row_colours = [color[int(e % 2)] for e in range(features.shape[1])]
        columns = ["Value"]
        rows = feature_names

        font_size_cells = 20

        if n_importance > 15:
            y_table_size = 2.0 - 0.05 * (n_importance - 15)
            font_size_cells = 10
        else:
            y_table_size = 2.0
        the_table = plt.table(
            cellText=cell_text,
            cellColours=cell_colours,
            rowLabels=rows,
            rowColours=row_colours,
            colLabels=columns,
            cellLoc="center",
            fontsize=font_size_cells,
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(15)
        the_table.scale(1, y_table_size)
        ax_features.set_title("Features", fontsize=font_size)

        pading = 0.04
        width_rect_right = (width / 3.0) * 2.0 - pading

        delta = 0.07 * n_importance
        b_ = top_margin - delta
        if b_ < bottom_margin:
            b_ = bottom_margin

        h_ = top_margin - b_
        l_ = left + width_rect_left + h_space + pading
        rect_importance = [l_, b_, width_rect_right, h_ - 0.05]
        ax_importance = plt.axes(rect_importance)

        names = feature_names[::-1]
        vals = importances[::-1]
        center_feature_importance = l_ + width_rect_right / 2.0
        plt.annotate(
            "Feature Importance",
            xy=(center_feature_importance, top_margin - pading_title),
            xycoords="figure fraction",
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=size_title,
            color="black",
            horizontalalignment="right",
        )
        # plt.annotate(
        #     f"Diff. Importance.: {diff_importances:5.3e}",
        #     xy=(center_feature_importance, top_margin - pading_title - 0.06),
        #     xycoords="figure fraction",
        #     xytext=(0.0, 0.0),  # 3 points vertical offset
        #     textcoords="offset points",
        #     ha="center",
        #     va="bottom",
        #     size=int(font_size*0.7),
        #     color="black",
        #     horizontalalignment="right",
        # )
        # Importance
        ax_importance = cls.plot_feature_importance(ax=ax_importance, names=names, vals=vals, size_title=size_title)

        axs = [ax_target, ax_features, ax_importance]

        return fig, axs

    @staticmethod
    def plot_predictions(ax, y_p=None, y_p_min=None, y_p_max=None, y_local_model=None, y_name="", explanation=None):
        if explanation is not None:
            y_p = explanation["y_p"]
            y_local_model = explanation["y_p_local_model"]
            y_p_max = explanation["y_p_max"]
            y_p_min = explanation["y_p_min"]

        ax.axvline(x=y_p, ymin=0, ymax=1, color="tab:green", linewidth=4, label=f"Model Prediction: {y_p:5.4f}")
        ax.axvline(x=y_p_min, ymin=0, ymax=1, color="black", linewidth=3, linestyle=":")
        if y_local_model:
            ax.axvline(
                x=y_local_model,
                ymin=0,
                ymax=1,
                color="tab:orange",
                linewidth=3,
                linestyle="-",
                label=f"Local Model Prediction: {y_local_model:5.4f}",
            )
        ax.axvline(x=y_p_max, ymin=0, ymax=1, color="black", linewidth=3, linestyle=":", label="min/max values")
        ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.tick_params("x", labelsize=15)
        ax.set_xticks(np.linspace(y_p_min, y_p_max, 6))
        ax.set_xlabel(y_name, fontsize=15)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(45)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.15, 1), fontsize=15, frameon=False)
        delta = (y_p_max - y_p_min) * 0.015
        ax.set_xlim(xmin=y_p_min - delta, xmax=y_p_max + delta)
        return ax

    @staticmethod
    def plot_feature_importance(ax, names, vals, size_title=18):
        colors = ["tab:blue" if x > 0 else "tab:red" for x in vals]
        pos = np.arange(len(vals))
        rects2 = ax.barh(pos, vals, align="center", alpha=0.5, color=colors)

        label_bar(rects2, ax, labels=names)
        ax.axvline(0, color="black", lw=2)

        x_lim = np.max(np.abs(vals[:]))
        ax.set_xlim(-x_lim, x_lim)
        y_lim = np.array(ax.get_ylim())
        ax.set_ylim(y_lim + np.array([0, 0.8]))

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.annotate("  Positive ", xy=(0, y_lim[1] + 0.1), size=16, color="tab:blue", ha="left")
        ax.annotate("  Negative ", xy=(0, y_lim[1] + 0.1), size=16, color="tab:red", ha="right")
        simpleaxis(ax)
        return ax

    @staticmethod
    def plot_errors(explanation):
        fig, axis = plt.subplots(1, 2)
        x = range(2, len(explanation.convergence_diffs) + 2)
        y = explanation.convergence_diffs
        axis[0].scatter(x[:], y[:])
        axis[0].plot(x[:], y[:])
        axis[0].set_ylabel("Difference - Importance")
        axis[0].set_xlabel("Steps")
        x = range(1, len(explanation.erros_training) + 1)
        y = explanation.erros_training
        axis[1].scatter(x[:], y[:])
        axis[1].plot(x[:], y[:])
        axis[1].set_ylabel("Errors")
        axis[1].set_xlabel("Steps")
        fig.tight_layout(pad=3.0)
        return fig, axis


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)
    return ax


if __name__ == "__main__":
    n_features = 25
    explanation = {}
    explanation["feature_names"] = [f"$xsada_{{{i}}}$" for i in range(n_features)]
    explanation["features"] = [*range(n_features)]
    explanation["y_p"] = 0.58
    explanation["y_p_local_model"] = 0.55
    explanation["y_p_max"] = 0.20
    explanation["y_p_min"] = 0.88
    explanation["importances"] = np.random.uniform(size=n_features) - 0.5
    explanation["ind_class_sorted"] = 0
    explanation["class_names"] = ["taget"]

    exp = ExplainGraph()

    fig, ax = exp.plot(explanation)
    plt.savefig("figure.png")

