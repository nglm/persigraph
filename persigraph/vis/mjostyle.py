from matplotlib.collections import (
    EllipseCollection, LineCollection, PolyCollection
)
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap
from typing import List, Sequence, Union, Any, Dict, Tuple

def _make_segments(
    x: np.ndarray,   # shape (T)
    y: np.ndarray,   # shape (T)
) -> np.ndarray:
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def _make_polygons(
    mean: np.ndarray,     # shape (T, d)
    std_sup: np.ndarray,  # shape (T, d)
    std_inf: np.ndarray,  # shape (T, d)
) -> np.ndarray:
    """
    Create list of line segments from x and y coordinates, in the correct format
    for PolyCollection
    """
    polys = []
    N_points = len(mean)
    for i in range(N_points-1):
        polys.append([
            # std_inf at t
            [mean[i, 0]-std_inf[i, 0], mean[i, 1]-std_inf[i, 1]],
            # std_sup at t
            [mean[i, 0]+std_sup[i, 0], mean[i, 1]+std_sup[i, 1]],
            # std_sup at t+1
            [mean[i+1, 0]+std_sup[i+1, 0], mean[i+1, 1]+std_sup[i+1, 1]],
            # std_inf at t+1
            [mean[i+1, 0]-std_inf[i+1, 0], mean[i+1, 1]-std_inf[i+1, 1]],
        ])
    polys = np.asarray(polys)
    return polys


def add_mjo_mean(
    mean: np.ndarray,             # shape (T, d)
    std_inf: np.ndarray = None,   # shape (T, d)
    std_sup: np.ndarray = None,   # shape (T, d)
    cmap: ListedColormap = None,
    line_kw: dict = {'lw' : 10},
    fig_kw: dict = {},
) -> Tuple[PolyCollection, LineCollection]:
    """
    Return collections corresponding to the mean and std of the MJO

    :param mean: mean values
    :type mean: np.ndarray, shape (T, d)
    :param std_inf: std of the members below the mean
    :type std_inf: np.ndarray, shape (T, d), optional
    :param std_sup: std of the members above the mean
    :type std_sup: np.ndarray, shape (T, d), optional
    :param cmap: colormap (for timescale), defaults to None
    :type cmap: ListedColormap, optional
    :param line_kw: kw for the mean line, defaults to {'lw' : 10}
    :type line_kw: dict, optional
    :param fig_kw: Figure kw, defaults to {}
    :type fig_kw: dict, optional
    :return: 2 Collections corresponding to the std and the mean
    :rtype: Tuple[PolyCollection, LineCollection]
    """
    if cmap is None:
        cmap = plt.get_cmap('plasma').reversed()

    # Default colors equally spaced on [0,1]:
    z = np.linspace(0.0, 1.0, len(mean))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    if std_inf is not None and std_sup is not None:
        polys = _make_polygons(mean, std_inf, std_sup)
        polys = PolyCollection(polys, array=z, cmap=cmap, alpha=0.15)
    else:
        polys = None
    segments = _make_segments(mean[:, 0], mean[:, 1])
    segments = LineCollection(segments, array=z, cmap=cmap, **line_kw)
    return polys, segments

def add_mjo_member(
    rmm: np.ndarray,      # Shape (T, 2)
    cmap: ListedColormap = None,
    line_kw: dict = {'lw' : 0.8, "alpha":1},
) -> LineCollection:
    """
    Return a collection corresponding to the trajectory of the given member

    :param rmm1: RMM components (x and y axis)
    :type rmm1: np.ndarray
    :param cmap:  colormap (for timescale), defaults to None
    :type cmap: ListedColormap, optional
    :param line_kw: kw for the member line, defaults to {'lw' : 0.8, "alpha":1}
    :type line_kw: dict, optional
    :return: Collection member line
    :rtype: LineCollection
    """
    if cmap is None:
        cmap = plt.get_cmap('plasma').reversed()
    # Default colors equally spaced on [0,1]:
    z = np.linspace(0.0, 1.0, len(rmm))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    segments = _make_segments(x=rmm[:, 0], y=rmm[:, 1])
    lc = LineCollection(segments, array=z, cmap=cmap, **line_kw)
    return lc


def draw_mjo_classes(
    fig=None,
    ax=None,
    plot_kw : dict =  {'c' : 'black', 'lw':1, 'zorder':31},
):
    # Weak VS strong MJO
    circle = EllipseCollection(
        widths = 2,
        heights = 2,
        angles = 0,
        units = 'xy',
        edgecolors = plot_kw['c'],
        facecolors = "none",
        offsets = (0, 0),
        lw = plot_kw['lw'],
        zorder = plot_kw['zorder'],
        transOffset = ax.transData)
    ax.add_collection(circle)

    # Create the data for the 4 lines
    N_points = 1000
    x1 = np.linspace(-4,4,N_points)
    y1 = np.array([x for x in x1])
    y2 = np.array([0 for x in x1])
    y3 = np.array([-x for x in x1])
    y4 = np.array([x for x in x1])
    x4 = np.zeros(N_points)

    # Now let's remove the parts that are inside the circle
    cond1 = np.sqrt(x1**2+y1**2) >= 1
    cond2 = np.sqrt(x1**2+y2**2) >= 1
    cond3 = np.sqrt(x1**2+y3**2) >= 1
    cond4 = np.sqrt(x4**2+y4**2) >= 1

    x1_filt = x1[cond1]
    y1_filt = y1[cond1]

    x2_filt = x1[cond2]
    y2_filt = y2[cond2]

    x3_filt = x1[cond3]
    y3_filt = y3[cond3]

    x4_filt = x4[cond4]
    y4_filt = y4[cond4]

    # Plot lines
    ax.plot(x1_filt[x1_filt<0], y1_filt[x1_filt<0], **plot_kw)
    ax.plot(x1_filt[x1_filt>0], y1_filt[x1_filt>0], **plot_kw)
    ax.plot(x2_filt[x2_filt<0], y2_filt[x2_filt<0], **plot_kw)
    ax.plot(x2_filt[x2_filt>0], y2_filt[x2_filt>0], **plot_kw)
    ax.plot(x3_filt[x3_filt<0], y3_filt[x3_filt<0], **plot_kw)
    ax.plot(x3_filt[x3_filt>0], y3_filt[x3_filt>0], **plot_kw)
    ax.plot(x4_filt[y4_filt<0], y4_filt[y4_filt<0], **plot_kw)
    ax.plot(x4_filt[y4_filt>0], y4_filt[y4_filt>0], **plot_kw)
    return fig, ax
