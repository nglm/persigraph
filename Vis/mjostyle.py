from matplotlib.collections import (
    EllipseCollection, LineCollection, PolyCollection
)
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def _make_polygons(mean, std_sup, std_inf):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for PolyCollection
    """
    polys = []
    N_points = np.shape(mean)[-1]
    for i in range(N_points-1):
        polys.append([
            # std_inf at t
            [mean[0, i]-std_inf[0, i], mean[1, i]-std_inf[1, i]],
            # std_sup at t
            [mean[0, i]+std_sup[0, i], mean[1, i]+std_sup[1, i]],
            # std_sup at t+1
            [mean[0, i+1]+std_sup[0, i+1], mean[1, i+1]+std_sup[1, i+1]],
            # std_inf at t+1
            [mean[0, i+1]-std_inf[0, i+1], mean[1, i+1]-std_inf[1, i+1]],
        ])
    polys = np.asarray(polys)
    return polys


def add_mjo_mean(
    mean,
    std_inf,
    std_sup,
    z=None,
    cmap=None,
    line_kw = {'lw' : 10},
    fig_kw = {},
):
    """

    """
    if cmap is None:
        cmap = plt.get_cmap('plasma').reversed()
    # if ax is None:
    #     fig, ax = plt.subplots(**fig_kw)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, np.shape(mean)[-1])
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)

    polys = _make_polygons(mean, std_inf, std_sup)
    polys = PolyCollection(polys, array=z, cmap=cmap, alpha=0.15)
    segments = _make_segments(mean[0], mean[1])
    segments = LineCollection(segments, array=z, cmap=cmap, **line_kw)
    return polys, segments

def add_mjo_member(
    x,
    y,
    z=None,
    cmap=None,
    line_kw = {'lw' : 0.8, "alpha":1},
):
    """

    """
    if cmap is None:
        cmap = plt.get_cmap('plasma').reversed()

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)

    segments = _make_segments(x, y)
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
