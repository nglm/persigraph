# from persisentgraph import PersistentGraph
# from vertex import Vertex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

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

def get_edge_lw(
    g,
    e,
    lw_max = 5,
    lw_min = 1
):
    return (e.nb_members/g.N)*(lw_max-lw_min) + lw_min


def get_alpha(g, obj):
    s_born = obj.s_born
    s_death = obj.s_death
    ratio = max(
        ((g.distances[s_born] - g.distances[s_death])
        / g.distances[0]),
        0.01
    )

    return ratio

def plot_vertices(
    ax,
    g,
    vertices,
    t,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
):
    if not isinstance(vertices, list):
        vertices = [vertices]
    #vertices = [g.vertices[t][v_num] for v_num in vertices]
    values = [v.value for v in vertices]
    # Iterable alpha and colors. Source:
    # https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
    alphas = [v.ratio_life for v in vertices]
    colors = np.asarray([
        (v.ratio_members*c1 + (1-v.ratio_members)*c2) for v in vertices
    ])
    # To understand the '/255' see source:
    # https://stackoverflow.com/questions/57113398/matplotlib-scatter-fails-with-error-c-argument-has-n-elements-which-is-not-a
    colors /=255.
    colors[:,3] = alphas
    lw = 1
    condition = (colors < 0) | (colors > 1)
    if np.any(condition):
        print(colors)
        print(np.ma.masked_where(condition, colors))
    # for v_num in vertices:
    #     v = g.vertices[t][v_num]
    #     color = colorFader(mix=(v.nb_members/g.N))
    #     alpha = get_alpha(g, v)
    #     lw = 1
    #     values.append[v.value]
    n = len(values)
    if n == 1:
        ax.scatter(t, values, c=colors, lw=lw)
    else:
        ax.scatter([t]*len(values), values, c=colors, lw=[lw]*len(values))
    #lines = LineCollection(lines, linewidths=lw)
    #ax.add_collection(lines)
    return ax

def plot_edges(
    ax,
    g,
    edges,
    t,
    c1 = np.array([254,0,0,0]),
    c2 = np.array([254,254,0,0]),
):
    if not isinstance(edges, list):
        edges = [edges]
    alphas = [e.ratio_life for e in edges]
    colors = np.asarray([
        (e.ratio_members*c1 + (1-e.ratio_members)*c2) for e in edges
    ])
    colors /= 255
    colors[:,3] = alphas
    lw = 1
    lines = [
        (
        (t,   g.vertices[t][e.v_start].value),
        (t+1, g.vertices[t+1][e.v_end].value)
        ) for e in edges
    ]

    lines = LineCollection(lines,colors=colors, linewidths=[lw]*len(lines))
    ax.add_collection(lines)
    return ax

def plot_as_graph(
    g,
    s:int = None,
    show_vertices: bool = True,
    show_edges: bool = True,
):
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    if s is None:
        title = "All steps"
        for t in range(g.T):
            if show_vertices:
                ax = plot_vertices(ax,g,g.vertices[t],t)
            if show_edges and (t < g.T-1):
                ax = plot_edges(ax,g,g.edges[t],t)
    else:
        title = "step s = " + str(s)
        for t in range(g.T):
            if show_vertices:
                vertices_key = g.get_alive_vertices(s, t)
                vertices = [g.vertices[t][key] for key in vertices_key]
                ax = plot_vertices(ax, g, vertices, t)
            if (t < g.T-1) and show_edges:
                edges_key = g.get_alive_edges(s, t)
                edges = [g.edges[t][key] for key in edges_key]
                ax = plot_edges(ax, g, edges,t)
    ax.autoscale()
    ax.set_title(title)
    return fig, ax

