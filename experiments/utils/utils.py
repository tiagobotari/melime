from skimage import feature, transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib import cm


def color_map():
    colors = cm.get_cmap("bwr", 200)
    scale_color = [*range(0, 50, 1)] + [*range(50, 80, 2)]
    scale_color1 = [*range(120, 120 + 30, 2)] + [*range(120 + 30, 200, 1)]
    newcolors = colors(scale_color + [*range(98, 103)] + scale_color1)
    newcmp = matplotlib.colors.ListedColormap(newcolors)
    return newcmp


def plot(data, xi=None, cmap="RdBu_r", axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    cmap = color_map()

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap("Greys_r")
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode="constant")
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    cp = axis.imshow(data, extent=extent, interpolation="none", cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation="none", cmap=cmap_xi, alpha=alpha)
    # axis.axis('off')

    return axis, cp
