import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from math import exp
from PersistentGraph.analysis import sort_components_by



def sigmoid(
    x,
    range0_1 = True,
    shift=3.,
    a=6.,
    f0=0.7,
    f1=6,
):
    # Here we will get f(0) = 0 and f(1) = 1
    res = 1/(1+exp(-(a*x-shift))) + (2*x-1)/(1+exp(shift))
    # Use min and max because of precision error
    res = min(max(0, res), 1)
    # Here we will get f(0) = f0 and f(1) = f1
    if not range0_1:
        res = f1*res + f0*(1-x)
    return res

def plot_vertices(
    g,
    vertices,
    t: int,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    lw_min=0.5,
    lw_max=20,
    f=sigmoid,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if not isinstance(vertices, list):
        vertices = [vertices]
    # sort by ratio life so that the older components are easier to see
    vertices = sort_components_by(
        vertices, criteron='ratio_life', descending=False
    )[0]
    # Keep only the vertices respecting the thresholds
    vertices = [
        v for v in vertices
        if v.nb_members >= threshold_m and v.ratio_life >= threshold_l
    ]
    values = [v.value for v in vertices]
    alphas = [f(v.ratio_life) for v in vertices]

    colors = np.asarray(
        [(f(v.ratio_life)*c1 + (1-f(v.ratio_life))*c2) for v in vertices]
    ).reshape((-1, 4)) / 255

    colors[:,3] = alphas

    lw = np.asarray([
        f(v.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
        for v in vertices
    ])
    n = len(values)
    if n == 1:
        col = ax.scatter(g.time_axis[t], values, c=colors, s=lw**2)
    else:
        col = ax.scatter([g.time_axis[t]]*n, values, c=colors, s=lw**2)

    #ax.add_artist(col)
    return ax



def plot_edges(
    g,
    edges,
    t,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    lw_min=0.3,
    lw_max=20,
    f=sigmoid,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if not isinstance(edges, list):
        edges = [edges]
    # sort by ratio life so that the older components are easier to see
    edges = sort_components_by(
        edges, criteron='ratio_life', descending=False
    )[0]
    # Keep only edges respecting the thresholds
    edges = [
        e for e in edges
        if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    ]
    alphas = [f(e.ratio_life) for e in edges]

    colors = np.asarray(
        [(f(e.ratio_life)*c1 + (1-f(e.ratio_life))*c2)
        for e in edges]
    ).reshape((-1, 4)) / 255
    colors[:,3] = alphas

    lw = np.asarray([
        f(e.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
        for e in edges
    ])
    lines = [
        (
        (g.time_axis[t],   g.vertices[t][e.v_start].value),
        (g.time_axis[t+1], g.vertices[t+1][e.v_end].value)
        ) for e in edges
    ]

    lines = LineCollection(lines,colors=colors, linewidths=lw)
    ax.add_artist(lines)
    return ax

def plot_as_graph(
    g,
    s:int = None,
    show_vertices: bool = True,
    show_edges: bool = True,
    threshold_m:int = 1,
    threshold_l:float = 0.00,
    ax = None,
    fig = None,
    fig_kw: dict = {"figsize" : (24,12)},
    ax_kw: dict = {'xlabel' : "Time (h)",
                   'ylabel' : "Temperature (°C)"}
):
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
        ax.autoscale()
        ax.set_xlabel(ax_kw['xlabel'])
        ax.set_ylabel(ax_kw['ylabel'])
        ax.set_facecolor("whitesmoke")
        #ax.set_title(title)
    ax.set_facecolor("whitesmoke")
    if s is None:
        title = "All steps"
        for t in range(g.T):
            if show_vertices:
                ax = plot_vertices(
                    g,g.vertices[t],t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax,
                )
            if show_edges and (t < g.T-1):
                ax = plot_edges(
                    g,g.edges[t],t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax
                )
    else:
        title = "step s = " + str(s)
        for t in range(g.T):
            if show_vertices:
                vertices_key = g.get_alive_vertices(s = s, t = t)
                vertices = [g.vertices[t][key] for key in vertices_key]
                ax = plot_vertices(
                    g, vertices, t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax,
                )
            if (t < g.T-1) and show_edges:
                edges_key = g.get_alive_edges(s = s, t = t)
                edges = [g.edges[t][key] for key in edges_key]
                ax = plot_edges(
                    g, edges, t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax,
                )
        ax.set_title(title)

    return fig, ax


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
    if isinstance(bn_distances[0], list):
        bn_distances = [bn_distances]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(bn_distances)
    ax.autoscale()
    ax.set_xlabel("Time")
    ax.set_ylabel("Bottleneck distance")
    plt.show()
    return fig, ax
