import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, EllipseCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path
import time
import random
from math import exp, ceil, floor, sqrt
from typing import List
from PIL import ImageColor

from .analysis import sort_components_by, get_k_life_span, get_relevant_k, get_relevant_components
from . import Vertex
from . import Edge


# =========================================================================
# TODO!
# =========================================================================
#
# -------------------------------------------------------------------------
# TODO: Get a nice color map between 2 colors
# -------------------------------------------------------------------------
# colors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
# cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
# -------------------------------------------------------------------------
#
# -------------------------------------------------------------------------
# TODO:  Get a nice gradient between 2 nodes of different colors
# -------------------------------------------------------------------------
#
# def colorline(
#     x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
#         linewidth=3, alpha=1.0):
#     """
#     http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#     http://matplotlib.org/examples/pylab_examples/multicolored_line.html
#     Plot a colored line with coordinates x and y
#     Optionally specify colors in the array z
#     Optionally specify a colormap, a norm function and a line width
#     """

#     # Default colors equally spaced on [0,1]:
#     if z is None:
#         z = np.linspace(0.0, 1.0, len(x))

#     # Special case if a single number:
#     if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
#         z = np.array([z])

#     z = np.asarray(z)

#     segments = make_segments(x, y)
#     lc = Path.LineCollection(
#         segments, array=z, cmap=cmap, norm=norm,
#         linewidth=linewidth, alpha=alpha
#         )

#     ax = plt.gca()
#     ax.add_collection(lc)

#     return lc


# def make_segments(x, y):
#     """
#     Create list of line segments from x and y coordinates, in the correct format
#     for LineCollection: an array of the form numlines x (points per line) x 2 (x
#     and y) array
#     """

#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     return segments

# -------------------------------------------------------------------------
# TODO:  Get a nice gradient for the std too
# -------------------------------------------------------------------------
#


def __uniform_polygon(g, edges):
    '''
    Define a polygon representing a uniform distribution

    #FIXME: Outdated since multivariate

    Return a nested list (1, 1 polygon)
    '''

    [to_uniforms, from_to_uniforms, from_uniforms] = edges
    polys = []

    if to_uniforms:
        t_start = g.time_axis[to_uniforms[0].time_step]
        t_end = g.time_axis[to_uniforms[0].time_step + 1]
        polys += [[
            # std_inf at t
            (t_start, e.v_start.info["params"][0] - e.v_start.info["params"][2]),
            # std_sup at t
            (t_start, e.v_start.info["params"][0] + e.v_start.info["params"][3]),
            # sup at t+1
            (t_end,   e.v_end.info["params"][1]),
            # inf at t+1
            (t_end,   e.v_end.info["params"][0])]
            for e in to_uniforms
        ]


    if from_to_uniforms:
        t_start = g.time_axis[from_to_uniforms[0].time_step]
        t_end = g.time_axis[from_to_uniforms[0].time_step + 1]
        polys += [[
            # inf at t
            (t_start, e.v_start.info["params"][0]),
            # sup at t
            (t_start, e.v_start.info["params"][1]),
            # sup at t+1
            (t_end,   e.v_end.info["params"][1]),
            # inf at t+1
            (t_end,   e.v_end.info["params"][0])]
            for e in from_to_uniforms
        ]

    if from_uniforms:
        t_start = g.time_axis[from_uniforms[0].time_step]
        t_end = g.time_axis[from_uniforms[0].time_step + 1]
        polys += [[
            # inf at t
            (t_start, e.v_start.info["params"][0]),
            # sup at t
            (t_start, e.v_start.info["params"][1]),
            # std_sup at t+1
            (t_end,   e.v_end.info["params"][0] + e.v_end.info["params"][3]),
            # std_inf at t+1
            (t_end,   e.v_end.info["params"][0] - e.v_end.info["params"][2]),]
            for e in from_uniforms
        ]

    return polys




def sort_components(
    components,
    threshold_m: int = 1,
    threshold_l: float = 0.00,
):
    components = [
        c for c in components
        if (c.nb_members > threshold_m and c.life_span > threshold_l )
    ]
    if components:
        components = sort_components_by(
            components, criteron='life_span', descending=False
        )[0]       # [0] because sort_components_by returns a nested list
        if components:
            # VERTICES
            if isinstance(components[0], Vertex):
                gaussians = [
                    c for c in components
                    if c.info['type'] in ['gaussian','KMeans','Naive']
                ]
                uniforms = [ c for c in components if c.info['type'] == 'uniform' ]
            # EDGES
            elif isinstance(components[0],Edge):
                gaussians = [
                    c for c in components
                    if (
                        c.v_start.info['type'] in ['gaussian','KMeans','Naive']
                    and
                        c.v_end.info['type'] in ['gaussian','KMeans', 'Naive']
                    )]
                from_to_uniforms = [
                    c for c in components
                    if (c.v_start.info['type'] == 'uniform')
                    and (c.v_end.info['type'] == 'uniform')
                    ]
                to_uniforms = [
                    c for c in components
                    if (c.v_start.info['type'] != 'uniform')
                    and (c.v_end.info['type'] == 'uniform')
                    ]
                from_uniforms = [
                    c for c in components
                    if (c.v_start.info['type'] == 'uniform')
                    and (c.v_end.info['type'] != 'uniform')
                    ]
                uniforms = [to_uniforms, from_to_uniforms, from_uniforms]
    else:
        gaussians = []
        uniforms = []

    return gaussians, uniforms

def plot_gaussian_vertices(
    g,
    vertices,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
    lw_min=0.5,
    lw_max=8,
    f=linear,
    color_list = get_list_colors(51),
    max_opacity=False,
    ax=None,
):
    collect = vdraw(vertices)
    ax.add_collection(circles)
    return ax

def plot_uniform_edges(
    g,
    edges,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    lw_min=0.3,
    lw_max=8,
    f=linear,
    color_list = get_list_colors(51),
    show_std = False,
    max_opacity=False,
    ax=None,
):
    [to_uniforms, from_to_uniforms, from_uniforms] = edges

    # This must respect the order in polys (see __uniform_polygon)
    alphas = []
    alphas += [f(e.life_span) for e in to_uniforms]
    alphas += [f(e.life_span) for e in from_to_uniforms]
    alphas += [f(e.life_span) for e in from_uniforms]

    # The color of a uniform edge is grey
    colors = np.asarray(
        [color_list[0] for e in to_uniforms]
        + [color_list[0] for e in from_to_uniforms]
        + [color_list[0] for e in from_uniforms]
    ).reshape((-1, 4))

    if max_opacity:
        colors[:,3] = 1
    else:
        colors[:,3] = alphas

    polys = __uniform_polygon(g, edges)


    colors[:,3] /= 1
    polys = PolyCollection(polys, facecolors=colors)
    ax.add_collection(polys)

    return ax

def plot_gaussian_edges(
    g,
    edges,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    lw_min=0.3,
    lw_max=8,
    f=linear,
    color_list = get_list_colors(51),
    show_std = False,
    max_opacity=False,
    ax=None,
):
    if edges:

        # Colors
        # The color of an edge is the color of its start vertex
        colors = np.asarray(
            [color_list[e.v_start.info['brotherhood_size'][0]] for e in edges]
        ).reshape((-1, 4))

        # Alphas
        if max_opacity:
            colors[:,3] = 1
        else:
            colors[:,3] = [f(e.life_span) for e in edges]

        # Sizes
        lw = np.asarray([
            f(e.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
            for e in edges
        ])

        # matplotlib.collections.Collection
        lines = __edges_lines(g, edges)
        lines = LineCollection(
            lines,
            colors=colors,
            linewidths=lw,)
        ax.add_collection(lines)

        if show_std:
            colors[:,3] /= 6

            # matplotlib.collections.Collection
            polys = __std_polygon(g, edges)
            polys = PolyCollection(polys, facecolors=colors)
            ax.add_collection(polys)

    return ax

def plot_vertices(
    g,
    t,
    vertices,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    color_list = get_list_colors(51),
    lw_min=0.5,
    lw_max=20,
    f=sigmoid,
    max_opacity=False,
    ax=None,
):

    if ax is None:
        ax = plt.gca()
    if not isinstance(vertices, list):
        vertices = [vertices]

    # Keep only the vertices respecting the thresholds
    gaussians, uniforms = sort_components(
        vertices,
        threshold_m = threshold_m,
        threshold_l = threshold_l,
    )
    ax = plot_gaussian_vertices(
        g,
        gaussians,
        c1 = c1,
        c2 = c2,
        lw_min=lw_min,
        lw_max=lw_max,
        f=f,
        max_opacity=max_opacity,
        ax=ax,
    )
    # ax = plot_uniform_vertices(
    #     g,
    #     uniforms,
    #     f=f,
    #     ax=ax,
    # )
    return ax

def plot_edges(
    g,
    t,
    edges,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    lw_min=0.3,
    lw_max=20,
    color_list = get_list_colors(51),
    show_std = False,
    show_uniform = False,
    f=sigmoid,
    max_opacity=False,
    ax=None,
):

    if ax is None:
        ax = plt.gca()
    if not isinstance(edges, list):
        edges = [edges]

    # Keep only edges respecting the thresholds
    gaussians, uniforms = sort_components(
        edges,
        threshold_m = threshold_m,
        threshold_l = threshold_l,
    )
    ax = plot_gaussian_edges(
        g,
        gaussians,
        c1 = c1,
        c2 = c2,
        lw_min=lw_min,
        lw_max=lw_max,
        show_std = show_std,
        f=f,
        max_opacity=max_opacity,
        ax=ax,
    )
    if show_uniform:
        ax = plot_uniform_edges(
            g,
            uniforms,
            c1 = c1,
            c2 = c2,
            lw_min=lw_min,
            lw_max=lw_max,
            show_std = show_std,
            f=f,
            max_opacity=max_opacity,
            ax=ax,
        )

    return ax

def _nrows_ncols(n: int) -> Tuple[int, int]:
    """
    Get a number a rows and columns from a number of axes

    :param n: Number of axes (ex: dimension of physical variable)
    :type n: int
    :return: Number of rows and columns of the figure
    :rtype: Tuple[int, int]
    """
    if n == 1:
        nrows = 1
        ncols = 1
    else:
        nrows = floor(sqrt(n))
        ncols = ceil(n/nrows)
    return nrows, ncols

def plot_as_graph(
    g,
    s:int = None,
    i_var: int = None,
    show_vertices: bool = True,
    show_edges: bool = True,
    show_std: bool = True,
    show_uniform: bool = False,
    threshold_m:int = 0,
    threshold_l:float = 0.00,
    max_opacity: bool = False,
    fig = None,
    axs = None,
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
):
    if i_var is None:
        i_range = range(g.d)
    if axs is None:
        fig, axs = plt.subplots(**fig_kw)
        axs.autoscale()

    color_list = get_list_colors(g.k_max)
    if s is None:
        title = "All steps"
        for t in range(g.T):
            if show_vertices:
                ax = plot_vertices(
                    g, t, g._vertices[t],
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list, max_opacity=max_opacity,
                    ax=ax,
                )
            if show_edges and (t < g.T-1):
                ax = plot_edges(
                    g, t, g._edges[t],
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list, max_opacity=max_opacity,
                    show_std = show_std, show_uniform=show_uniform,
                    ax=ax
                )
    else:
        title = "step = " + str(s)
        for t in range(g.T):
            if show_vertices:
                vertices = g.get_alive_vertices(
                    steps = s,
                    t = t,
                    get_only_num = False,
                )

                ax = plot_vertices(
                    g, t, vertices,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list, max_opacity=max_opacity,
                    ax=ax
                )
            if (t < g.T-1) and show_edges:
                edges = g.get_alive_edges(
                    steps = s,
                    t = t,
                    get_only_num = False,
                )
                ax = plot_edges(
                    g, t, edges,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list, max_opacity=max_opacity,
                    show_std = show_std, show_uniform=show_uniform,
                    ax=ax,
                )
    ax.autoscale()
    ax.set_xlabel(ax_kw.pop('xlabel', "Time (h)"))
    ax.set_ylabel(ax_kw.pop('ylabel', ""))
    ax.set_xlim([g.time_axis[0],g.time_axis[-1]])
    ax.set_title(title)
    return fig, ax





def k_plot(
    g,
    k_max=8,
    life_span=None,
    fig = None,
    ax = None,
    show0 = False,
    show_legend = True,
    fig_kw: dict = {"figsize" : (5,3)},
    ax_kw: dict = {},
):
    """
    Spaghetti plots of ratio scores for each number of clusters
    """
    k_max = min(k_max, g.k_max)
    colors = get_list_colors(g.N)
    if life_span is None:
        life_span = get_k_life_span(g, k_max)

    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
    if show0:
        k_range = range(k_max)
    else:
        k_range = range(1, k_max)
    for k in k_range:
        ax.plot(
            g.time_axis, life_span[k],
            c=colors[k], label='k='+str(k)
            )
        if show_legend:
            ax.legend()
        ax.set_xlabel(ax_kw.pop('xlabel', 'Time (h)'))
        ax.set_ylabel(ax_kw.pop('ylabel', 'Life span'))
        ax.set_ylim([0,1])
    return fig, ax, life_span

def _draw_arrow(
    g,
    ax,
    k,
    t_start,
    t_end,
    offset_arrow = +25,
    offset_text = +5,
    width = 5,
    txt_kw = {'size':18}
):
    colors = get_list_colors(g.N)
    x_start = (
        (g.time_axis[t_start] - g.time_axis[0])
        / (g.time_axis[-1] - g.time_axis[0])
    )
    x_end = (
        (g.time_axis[t_end] - g.time_axis[0])
        / (g.time_axis[-1] - g.time_axis[0])
    )
    x_mid = (x_end + x_start) / 2

    if t_start == t_end:
        arrowstyle = '|-|, widthA=.5, widthB=.5'
    else:
        arrowstyle = '|-|'

    # Arrow
    ax.annotate(
        '',
        xy=(x_start, offset_arrow),
        xycoords=('axes fraction', 'axes points'),
        #xytext=(x_end+0.001, 0),
        xytext=(x_end, 0),
        textcoords=('axes fraction', 'offset points'),
        arrowprops=dict(arrowstyle=arrowstyle, color=colors[k]),
    )
    # Thicker Arrow
    ax.annotate(
        '',
        xy=(x_start+0.0025, offset_arrow),
        xycoords=('axes fraction', 'axes points'),
        #xytext=(x_end+0.001, 0),
        xytext=(x_end-0.0025, 0),
        textcoords=('axes fraction', 'offset points'),
        arrowprops=dict(
            width=width, headwidth=0, headlength=0.0001, color=colors[k]
        ),
    )

    # Text
    ax.annotate(
        k,
        xy=(x_mid, offset_text),
        xycoords=('axes fraction', 'axes points'),
        xytext=(-4, 0),
        textcoords=('offset points', 'offset points'),
        color=colors[k], **txt_kw
    )

def annot_ax(
    g,
    ax,
    relevant_k = None,
    k_max = 8,
    arrow_kw = {}
):
    k_max = min(k_max, g.k_max)
    # For each time step, get the most relevant number of clusters
    if relevant_k is None:
        relevant_k = get_relevant_k(g, k_max=k_max)

    # init
    t_start = 0
    k_curr, _ = relevant_k[0]
    for t, (k, _) in enumerate(relevant_k[1:]):
        if k != k_curr:
            t_end = t
            _draw_arrow(
                g, ax = ax, k=k_curr, t_start=t_start, t_end=t_end,
            )
            k_curr = k
            t_start = t
    # last arrow if not already done
    _draw_arrow(
        g, ax = ax, k = k_curr, t_start = t_start, t_end = -1
    )

    return ax


def plot_most_revelant_components(
    g,
    relevant_components = None,
    relevant_k = None,
    k_max = 8,
    show_vertices: bool = True,
    show_edges: bool = True,
    show_std: bool = True,
    threshold_m:int = 0,
    threshold_l:float = 0.00,
    max_opacity:bool = True,
    fig = None,
    ax = None,
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
):
    k_max = min(k_max, g.k_max)
    # For each time step, get the most relevant number of clusters
    if relevant_components is None:
        vertices, edges = get_relevant_components(
            g, relevant_k=relevant_k, k_max=k_max)
    else:
        vertices, edges = relevant_components

    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
        ax.autoscale()

    color_list = get_list_colors(k_max)
    for t in range(g.T):
        if show_vertices:
            ax = plot_vertices(
                g, t, vertices[t],
                threshold_m=threshold_m, threshold_l=threshold_l,
                color_list = color_list, max_opacity=max_opacity,
                ax=ax,
            )
        if show_edges and (t < g.T-1):
            ax = plot_edges(
                g, t, edges[t],
                threshold_m=threshold_m, threshold_l=threshold_l,
                color_list = color_list, max_opacity=max_opacity,
                show_std = show_std,
                ax=ax
            )
    ax.autoscale()
    ax.set_xlabel(ax_kw.pop('xlabel', "Time (h)"))
    ax.set_ylabel(ax_kw.pop('ylabel', ""))
    ax.set_xlim([g.time_axis[0], g.time_axis[-1]])
    ax.set_title('Only most relevant components')
    return fig, ax


def plot_overview(
    g,
    relevant_components = None,
    k_max = 8,
    show_vertices: bool = True,
    show_edges: bool = True,
    show_std: bool = True,
    threshold_m:int = 0,
    threshold_l:float = 0.00,
    max_opacity:bool = True,
    fig = None,
    ax = None,
    fig_kw: dict = {},
    ax_kw: dict = {},
):
    if fig is None:
        fig = plt.figure(
            figsize = fig_kw.pop('figsize', (40,12)), tight_layout=True
        )
    gs = fig.add_gridspec(nrows=2, ncols=5)

    # Plot entire graph
    ax0 = fig.add_subplot(gs[:, 0:2])
    _, ax0 = plot_as_graph(
        g, show_vertices=show_vertices, show_edges=show_edges,
        show_std=show_std, ax=ax0)
    ax0.set_title("Entire graph")
    ax0.set_xlabel("Time (h)")
    ax0.set_ylabel("Values")

    # Arrows on entire graph
    ax0 = annot_ax(g, ax=ax0)

    # k_plot
    ax1 = fig.add_subplot(gs[0, 2], sharex=ax0)
    _, ax1, _ = k_plot(g, k_max = 5, ax=ax1)
    #ax1.set_xlabel("Time")
    ax1.set_ylabel("Relevance")
    ax1.set_title('Number of clusters: relevance')

    # most relevant components
    ax2 = fig.add_subplot(gs[:, 3:], sharey=ax0, sharex=ax0)
    _, ax2 = plot_most_revelant_components(
        g, k_max=k_max, show_vertices=show_vertices,
        show_edges=show_edges, show_std=show_std, max_opacity=True, ax=ax2)
    ax2.set_title("Most relevant components")
    #ax2.set_xlabel("Time")
    ax2.set_ylabel("Values")

    # Arrows on most relevant components
    ax2 = annot_ax(g, ax=ax2)

    return fig, fig.axes


def __init_make_gif():

    return None

def __update_make_gif(
    s,
    g,
    show_vertices: bool = True,
    show_edges: bool = True,
    threshold_m:int = 1,
    threshold_l:float = 0.00,
    cumulative=True,
    ax = None,
    verbose = False,
):
    if not cumulative:
        ax.collections = []
        ax.artists = []
        # ax.set_xlim(g.min_time_step, g.max_time_step)
        # ax.set_ylim(g.min_value-1, g.max_value+1)
    if verbose:
        print(s)
    fig, ax = plot_as_graph(
        g,
        s = s,
        show_vertices = show_vertices,
        show_edges = show_edges,
        threshold_m = threshold_m,
        threshold_l = threshold_l,
        ax = ax,
    )


def make_gif(
    g,
    show_vertices: bool = True,
    show_edges: bool = True,
    threshold_m:int = 1,
    threshold_l:float = 0.01,
    cumulative=True,
    ax = None,
    fig = None,
    fig_kw: dict = {"figsize" : (5,5)},
    ax_kw: dict = {'xlabel' : "Time (h)",
                   'ylabel' : "Temperature (°C)"},
    verbose=False,
    max_iter=None,
):
    """
    FIXME: Outdated
    """

    fig, ax = plt.subplots(**fig_kw)
    ax.set_xlim(g.min_time_step, g.max_time_step)
    ax.set_ylim(g.min_value-1, g.max_value+1)

    if max_iter is None:
        max_iter = g.nb_steps

    fargs = (
        g,
        show_vertices,
        show_edges,
        threshold_m,
        threshold_l,
        cumulative,
        ax,
        verbose
    )

    ani = FuncAnimation(
        fig,
        func = __update_make_gif,
        fargs = fargs,
        frames = max_iter,
        init_func = None,
    )
    t_end = time.time()
    return ani



    # Update drawing the next step
    # If cumulative, simply 'add' the current step to the previous ones
    # If not, the current frame is composed of the current step only




def plot_barcodes(
    barcodes,
    c1 = np.array([254.,0.,0.,1.]),
    c2 = np.array([254.,254.,0.,1.]),
):
    """
    FIXME: Outdated
    """
    if not isinstance(barcodes[0], list):
        barcodes = [barcodes]
    for t in range(len(barcodes)):
        fig, ax = plt.subplots(figsize=(10,10))
        colors = []
        lines = []
        c = np.zeros_like(c1, dtype=float)
        for i, (start, end, r_members) in enumerate(barcodes[t]):
            c[:3] = (r_members*c1[:3] + (1-r_members)*c2[:3])/255.
            c[-1] = 1.
            lines.append(((i, start),(i, end)))
            colors.append(c)
        lines = LineCollection(lines,colors=np.asarray(colors))
        ax.add_collection(lines)
        ax.autoscale()
        ax.set_xlabel("Components")
        ax.set_ylabel("Life span")
        plt.show()

def plot_bottleneck_distances(
    bn_distances,
    c1 = np.array([254.,0.,0.,1.]),
    c2 = np.array([254.,254.,0.,1.]),
):
    """
    FIXME: Outdated
    """
    if isinstance(bn_distances[0], list):
        bn_distances = [bn_distances]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(bn_distances)
    ax.autoscale()
    ax.set_xlabel("Time")
    ax.set_ylabel("Bottleneck distance")
    plt.show()
    return fig, ax
