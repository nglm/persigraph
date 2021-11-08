import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import time
from typing import List, Union


from .analysis import get_k_life_span, get_relevant_k, get_relevant_components
from ..DataAnalysis.preprocess import to_polar, to_cartesian
from ..Vis import PGraphStyle
from ..Vis.commonstyle import nrows_ncols, get_list_colors
from ..Vis.barstyle import draw_arrow
from ..Vis.mjostyle import draw_mjo_classes, add_mjo_member ,add_mjo_mean
from ..utils.lists import to_list

def plot_mean_std(
    g = None,
    members = None,
    time_axis = None,
    fig = None,
    axs = None,
    fig_kw: dict = {"figsize" : (20,8)},
    ax_kw: dict = {},
    plt_kw : dict = {'lw' : 3, 'ls' : '--', 'color' :'grey'}
):
    if members is None:
        if g is None:
            raise ValueError("No members were given")
        else:
            members = g.members
    if time_axis is None:
        if g is None:
            time_axis = np.arange(members.shape[-1])
        else:
            time_axis = g.time_axis
    if len(members.shape) > 2:
        d = members.shape[1]
    else:
        d = 1

    if axs is None:
        nrows, ncols = nrows_ncols(d)
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            squeeze = False,
            **fig_kw)
    mean = np.mean(members, axis=0)
    std_sup = np.std(members, axis=0)
    std_inf = np.std(members, axis=0)
    for i, ax in enumerate(axs.flat):
        lw = plt_kw.pop("lw", 3)
        ax.plot(time_axis, mean[i], lw = lw, **plt_kw)
        ax.plot(time_axis, mean[i] + std_sup[i], lw=lw/2, **plt_kw)
        ax.plot(time_axis, mean[i] - std_inf[i], lw=lw/2, **plt_kw)
    return fig, axs

def plot_members(
    g = None,
    members = None,
    time_axis = None,
    fig = None,
    axs = None,
    fig_kw: dict = {"figsize" : (20,8)},
    ax_kw: dict = {},
    plt_kw : dict = {'lw' : 0.6, 'color' :'lightgrey'}
):
    if members is None:
        if g is None:
            raise ValueError("No members were given")
        else:
            members = g.members
    if time_axis is None:
        if g is None:
            time_axis = np.arange(members.shape[-1])
        else:
            time_axis = g.time_axis
    if len(members.shape) > 2:
        d = members.shape[1]
    else:
        d = 1

    if axs is None:
        nrows, ncols = nrows_ncols(d)
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            squeeze = False,
            **fig_kw)
    for i, ax in enumerate(axs.flat):
        for m in members[:,i,:]:
            ax.plot(time_axis, m,  **plt_kw)
    return fig, axs

def plot_mjo_mean_std(
    g = None,
    members = None,
    fig = None,
    ax = None,
    show_classes = True,
    show_std = False,
    polar = True,  # True if members are in polar coordinate
    fig_kw: dict = {"figsize" : (15,15), 'tight_layout':True, },
):
    if members is None:
        if g is None:
            raise ValueError("No members were given")
        else:
            members = g.members
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
    if show_classes:
        fig, ax = draw_mjo_classes(fig, ax)
    mean = np.mean(members, axis=0)
    std_sup = np.std(members, axis=0)
    std_inf = np.std(members, axis=0)
    if polar:
        mean = to_cartesian(mean)
        std_inf = None
        std_sup = None
    polys, segments = add_mjo_mean(mean, std_inf, std_sup)
    if not polar and show_std:    # There is no poly if polar
        ax.add_collection(polys)
    ax.add_collection(segments)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    return fig, ax

def plot_mjo_members(
    g = None,
    members = None,
    fig = None,
    ax = None,
    show_classes = True,
    fig_kw: dict = {"figsize" : (15,15), 'tight_layout':True, },
    ax_kw: dict = {},
    plt_kw : dict = {'lw' : 0.8}
):
    if members is None:
        if g is None:
            raise ValueError("No members were given")
        else:
            members = g.members
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
    if show_classes:
        fig, ax = draw_mjo_classes(fig, ax)
    for rmm in members:
        ax.add_collection(add_mjo_member(rmm, line_kw=plt_kw))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    return fig, ax

def plot_as_graph(
    g,
    s:int = None,
    t = None,
    vertices = None,
    edges = None,
    relevant_k = None,
    fig = None,
    axs = None,
    pgstyle = None,
    pgstyle_kw: dict = {},
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
    # ax_title: str = 'Entire Graph',
    # xlabel: Union[str, List[str]] = 'Time (h)',
    # ylabel: Union[str, List[str]] = 'Values'
):
    """
    Plot the entire graph

    If ``vertices`` is not specified, plot all the vertices in the graph.
    Same holds for ``edges``. If ``relevant_k`` is not specified, take the automated solution
    """
    if axs is None:
        nrows, ncols = nrows_ncols(g.d)
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            squeeze = False,
            **fig_kw)

    if vertices is None:
        vertices = g._vertices
    elif not isinstance(vertices, list):
        vertices = to_list(vertices)

    if edges is None:
        edges = g._edges
    elif not isinstance(edges, list):
        edges = to_list(edges)

    #TODO: Check if we can put this in pgstyle instead
    color_list = get_list_colors(g.k_max)

    if pgstyle is None:
        pgstyle = PGraphStyle(color_list = color_list, **pgstyle_kw)

    if s is None:
        # Get all the collections associated with vertices, edges and std
        axs_collections = pgstyle.gdraw(
            axs, g, vertices=vertices, edges=edges,  t=t)
        for ax, collections in zip(axs.flat, axs_collections):

            # Add the collections to their corresponding ax
            for collect in collections:
                ax.add_collection(collect)

            # Add suggestion bar (not a collection)
            if pgstyle.show_bar:
                suggestion_bar(
                    g, ax, relevant_k=None, k_max=g.k_max,
                    color_list=color_list)

            ax.autoscale()
            ax.set_xlim([g.time_axis[0],g.time_axis[-1]])
            # ax.set_title(ax_title)
            # ax.set_xlabel(xlabel)
            # ax.set_ylabel(ylabel)
    else:
        raise NotImplementedError("Cannot display a specific step of the graph")

    # ax.set_xlabel(ax_kw.pop('xlabel', "Time (h)"))
    # ax.set_ylabel(ax_kw.pop('ylabel', ""))
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
    colors = get_list_colors(k_max)
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

def k_legend(color_list, k_max=8, lw=4):
    k_max = min(len(color_list), k_max)
    color_list = color_list[:k_max]
    handles = [Line2D([0], [0], color=c, lw=lw) for c in color_list[1:]]
    labels = ['k='+str(i) for i in range(1, len(color_list))]
    return (handles, labels)

def draw_legend(ax, color_list, k_max=None, size=12):
    ax.legend(
        *k_legend(color_list=color_list, k_max=k_max),
        prop={'size' : size})
    ax.axis('off')
    return ax

def suggestion_bar(
    g,
    ax,
    relevant_k = None,
    k_max = 8,
    color_list = None,
    arrow_kw = {}
):
    """
    Plot the suggestion bar

    If ``relevant_k`` is not specified, take the automated solution
    """
    k_max = min(k_max, g.k_max)
    if color_list is None:
        color_list = get_list_colors(k_max)
    # For each time step, get the most relevant number of clusters
    # If a custom value of relevant k is not given, take the automated
    # solution
    if relevant_k is None:
        relevant_k = get_relevant_k(g, k_max=k_max)

    # init
    t_start = 0
    k_curr, _ = relevant_k[0]
    for t, (k, _) in enumerate(relevant_k[1:]):
        if k != k_curr:
            t_end = t
            draw_arrow(
                g, ax = ax, k=k_curr, t_start=t_start, t_end=t_end,
                color_list = color_list,
            )
            k_curr = k
            t_start = t
    # last arrow if not already done
    draw_arrow(
        g, ax = ax, k = k_curr, t_start = t_start, t_end = -1,
        color_list = color_list,
    )

    return ax


def plot_most_revelant_components(
    g,
    t = None,
    relevant_components = None,
    relevant_k = None,
    k_max = 8,
    fig = None,
    axs = None,
    pgstyle = None,
    pgstyle_kw: dict = {'max_opacity' : True},
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
):
    """
    Plot only the most relevant components (automated or user-adjusted)

    If ``relevant_k`` nor ``relevant_components`` are specified, take
    the automated solution
    """
    k_max = min(k_max, g.k_max)
    # For each time step, get the most relevant number of clusters
    if relevant_components is None:
        vertices, edges = get_relevant_components(
            g, relevant_k=relevant_k, k_max=k_max)
    else:
        vertices, edges = relevant_components

    #TODO: Check if we can put this in pgstyle instead
    color_list = get_list_colors(k_max)

    if pgstyle is None:
        pgstyle = PGraphStyle(color_list = color_list, **pgstyle_kw)

    fig, axs =  plot_as_graph(
        g,
        t = t,
        vertices = vertices,
        edges = edges,
        fig = fig,
        axs = axs,
        pgstyle = pgstyle,
        fig_kw = fig_kw,
        ax_kw = ax_kw,
    )

    return fig, axs


def plot_overview(
    g,
    t = None,
    relevant_components = None,
    relevant_k = None,
    k_max = 8,
    fig = None,
    axs = None,
    pgstyle = None,
    pgstyle_kw: dict = {},
    fig_kw: dict = {"figsize" : (20,12)},
    ax_kw: dict = {},
):
    """
    Plot:

    - The entire graph with its suggestion bar
    - The most relevant components (automated or user-adjusted) with its
    corresponding bar
    - The k_plot
    - The legend
    """
    if fig is None:
        fig_kw["figsize"] = fig_kw.pop('figsize', (20*g.d+10,12))
        fig = plt.figure(**fig_kw, tight_layout=True)
    ncols = 12
    gs = fig.add_gridspec(nrows=2*g.d+1, ncols=ncols)

    # Create axs
    axs01 = []
    axs02 = []
    for i in range(g.d):
        # For entire graph view
        axs01.append(fig.add_subplot(gs[2*i:2*(i+1), 0:ncols//2]))
        # Most relevant component view
        axs02.append(fig.add_subplot(gs[2*i:2*(i+1), ncols//2:]))
        # We can not really share axis without that if they have been created
        # separately
        # Remove useless yaxis
        axs02[i].sharey(axs01[i])
        axs02[i].tick_params(labelleft=False)
        if i>0:
            # Remove useless xaxis
            axs01[i-1].sharex(axs01[i])
            axs01[i-1].tick_params(labelbottom=False)
            axs02[i-1].sharex(axs02[i])
            axs02[i-1].tick_params(labelbottom=False)

    # Add ax titles
    axs01[0].set_title("Entire Graph")
    axs02[0].set_title("Most relevant Components")

    axs01 = np.array(axs01)
    axs02 = np.array(axs02)
    # k plot
    axs03 = fig.add_subplot(gs[-1, 0:ncols//4])
    # legend
    axs04 = fig.add_subplot(gs[-1, 0:ncols//4+1])


    # Plot entire graph
    fix, axs01 = plot_as_graph(
        g,
        t = t,
        fig = fig,
        axs = axs01,
        pgstyle = pgstyle,
        fig_kw = fig_kw,
        ax_kw = ax_kw,
    )

    # Plot most relevant components
    fix, axs02 = plot_most_revelant_components(
        g,
        t = t,
        relevant_components = relevant_components,
        relevant_k = relevant_k,
        k_max = k_max,
        fig = fig,
        axs = axs02,
        pgstyle = pgstyle,
        fig_kw = fig_kw,
        ax_kw = ax_kw,
    )

    # k_plot
    _, axs03, _ = k_plot(g, k_max = k_max, ax=axs03, show_legend=False)
    # legend
    color_list = get_list_colors(g.k_max)

    axs04 = draw_legend(ax=axs04, color_list=color_list, k_max = k_max)


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




# def plot_barcodes(
#     barcodes,
#     c1 = np.array([254.,0.,0.,1.]),
#     c2 = np.array([254.,254.,0.,1.]),
# ):
#     """
#     FIXME: Outdated
#     """
#     if not isinstance(barcodes[0], list):
#         barcodes = [barcodes]
#     for t in range(len(barcodes)):
#         fig, ax = plt.subplots(figsize=(10,10))
#         colors = []
#         lines = []
#         c = np.zeros_like(c1, dtype=float)
#         for i, (start, end, r_members) in enumerate(barcodes[t]):
#             c[:3] = (r_members*c1[:3] + (1-r_members)*c2[:3])/255.
#             c[-1] = 1.
#             lines.append(((i, start),(i, end)))
#             colors.append(c)
#         lines = LineCollection(lines,colors=np.asarray(colors))
#         ax.add_collection(lines)
#         ax.autoscale()
#         ax.set_xlabel("Components")
#         ax.set_ylabel("Life span")
#         plt.show()

# def plot_bottleneck_distances(
#     bn_distances,
#     c1 = np.array([254.,0.,0.,1.]),
#     c2 = np.array([254.,254.,0.,1.]),
# ):
#     """
#     FIXME: Outdated
#     """
#     if isinstance(bn_distances[0], list):
#         bn_distances = [bn_distances]
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.plot(bn_distances)
#     ax.autoscale()
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Bottleneck distance")
#     plt.show()
#     return fig, ax
