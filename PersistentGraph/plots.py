import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from math import exp
from analysis import sort_by_ratio_life

# ------------
# Source:
# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
# ------------

def colorFader(
    c1="#98FB98", # PaleGreen
    c2="#FF4500", # OrangeRed
    mix=0,
    ctype="hex",
):
    """
    Interpolates between 2 colors

    0 <= mix <= 1
    mix=0: Darkred
    mix=1: Yellow

    :param c1: [description], defaults to "#8b0000"
    :type c1: str, optional
    :return: [description]
    :rtype: [type]
    """
    if ctype == "hex":
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex(mix*c1 + (1-mix)*c2)

# def get_edge_lw(
#     g,
#     e,
#     lw_max = 5,
#     lw_min = 1
# ):
#     return (e.nb_members/g.N)*(lw_max-lw_min) + lw_min


# def get_alpha(g, obj):
#     s_born = obj.s_born
#     s_death = obj.s_death
#     ratio = max(
#         ((g.distances[s_born] - g.distances[s_death])
#         / g.distances[0]),
#         0.01
#     )

#     return ratio

def plot_vertices(
    g,
    vertices,
    t: int,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if not isinstance(vertices, list):
        vertices = [vertices]
    # sort by ratio life so that the older components are easier to see
    vertices = sort_by_ratio_life(vertices, descending=False)[0]
    values = [
        v.value
        for v in vertices
        if v.nb_members >= threshold_m and v.ratio_life >= threshold_l
    ]
    # Iterable alpha and colors. Source:
    # https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
    alphas = [
        v.ratio_life
        for v in vertices
        if v.nb_members >= threshold_m and v.ratio_life >= threshold_l
    ]
    colors = np.asarray([
        (v.ratio_members*c1 + (1-v.ratio_members)*c2)
        for v in vertices
        if v.nb_members >= threshold_m and v.ratio_life >= threshold_l
    ])
    # To understand the '/255' see source:
    # https://stackoverflow.com/questions/57113398/matplotlib-scatter-fails-with-error-c-argument-has-n-elements-which-is-not-a
    colors /=255.
    colors[:,3] = alphas
    lw = 1
    n = len(values)
    if n == 1:
        ax.scatter(g.time_axis[t], values, c=colors, lw=lw)
    else:
        ax.scatter([g.time_axis[t]]*n, values, c=colors, lw=[lw]*n)
    return ax

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
    if not range0_1:
        res = f1*res + f0*(1-x)
    return res

def plot_edges(
    g,
    edges,
    t,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    threshold_m: int = 1,
    threshold_l: float = 0.00,
    lw_min=1,
    lw_max=7,
    f=sigmoid,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if not isinstance(edges, list):
        edges = [edges]
    # sort by ratio life so that the older components are easier to see
    edges = sort_by_ratio_life(edges, descending=False)[0]
    f0 = sigmoid(0)
    alphas = [
        f(e.ratio_life)
        for e in edges
        if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    ]
    # colors = np.asarray([
    #     (e.ratio_members*c1 + (1-e.ratio_members)*c2)
    #     for e in edges
    #     if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    # ]).reshape((-1, 4))

    colors = np.asarray([
        (f(e.ratio_members)*c1 + (1-f(e.ratio_members))*c2)
        for e in edges
        if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    ]).reshape((-1, 4))
    colors = colors / 255
    colors[:,3] = alphas
    # lw = np.asarray([
    #     (e.ratio_members*(lw_max-lw_min) + lw_min)
    #     for e in edges
    #     if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    # ])
    lw = np.asarray([
        f(e.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
        for e in edges
        if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    ])
    lines = [
        (
        (g.time_axis[t],   g.vertices[t][e.v_start].value),
        (g.time_axis[t+1], g.vertices[t+1][e.v_end].value)
        ) for e in edges
        if e.nb_members >= threshold_m and e.ratio_life >= threshold_l
    ]

    lines = LineCollection(lines,colors=colors, linewidths=lw)
    ax.add_collection(lines)
    return ax

def plot_as_graph(
    g,
    s:int = None,
    show_vertices: bool = True,
    show_edges: bool = True,
    threshold_m:int = 1,
    threshold_l:float = 0.00,
    fig_kw: dict = {"figsize" : (24,12)}
):
    fig, ax = plt.subplots(**fig_kw)
    ax.set_facecolor("white")
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
                vertices_key = g.get_alive_vertices(s, t)
                vertices = [g.vertices[t][key] for key in vertices_key]
                ax = plot_vertices(
                    g, vertices, t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax,
                )
            if (t < g.T-1) and show_edges:
                edges_key = g.get_alive_edges(s, t)
                edges = [g.edges[t][key] for key in edges_key]
                ax = plot_edges(
                    g, edges,t,
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    ax=ax,
                )
    ax.autoscale()
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(title)
    return fig, ax

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