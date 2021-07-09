import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from typing import List

from .analysis import get_k_life_span, get_relevant_k, get_relevant_components
from ..Vis import PGraphStyle
from ..Vis.commonstyle import nrows_ncols, get_list_colors




def plot_as_graph(
    g,
    s:int = None,
    fig = None,
    axs = None,
    pgstyle = None,
    pgstyle_kw: dict = {},
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
):
    if axs is None:
        nrows, ncols = nrows_ncols(g.d)
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, **fig_kw)

    if pgstyle is None:
        pgstyle = PGraphStyle(**pgstyle_kw)

    #TODO: Check if we can put this in pgstyle instead
    color_list = get_list_colors(g.k_max)

    if s is None:
        axs_collections = pgstyle.gdraw(g)
        for ax, collections in zip(axs.flat, axs_collections):
            ax.add_collection(collections)
    else:
        raise NotImplementedError("Cannot display a specific step of the graph")
    # ax.autoscale()
    # ax.set_xlabel(ax_kw.pop('xlabel', "Time (h)"))
    # ax.set_ylabel(ax_kw.pop('ylabel', ""))
    # ax.set_xlim([g.time_axis[0],g.time_axis[-1]])
    # ax.set_title(title)
    return fig, axs





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
