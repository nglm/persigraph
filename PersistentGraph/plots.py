# from persisentgraph import PersistentGraph
# from vertex import Vertex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------
# Source:
# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
# ------------

def colorFader(
    c1="#98FB98", # PaleGreen
    c2="#FF4500", # OrangeRed
    mix=0,
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
        0.05
    )

    return ratio

def plot_vertices(
    ax,
    g,
    vertices,
    t,
):
    if not isinstance(vertices, list):
        vertices = [vertices]
    for v_num in vertices:
        v = g.vertices[t][v_num]
        color = colorFader(mix=(v.nb_members/g.N))
        alpha = get_alpha(g, v)
        lw = 5
        ax.scatter([t], [v.value], c=color, alpha=alpha, lw=lw)
    return ax

def plot_edges(
    ax,
    g,
    edges,
    t,
):
    if not isinstance(edges, list):
        edges = [edges]
    for e_num in edges:
        e = g.edges[t][e_num]
        color = colorFader(mix=(e.nb_members/g.N))
        alpha = get_alpha(g, e)
        lw = 5
        v_start = g.vertices[t][e.v_start]
        v_end = g.vertices[t+1][e.v_end]
        ax.plot(
            [t, t+1],
            [v_start.value, v_end.value],
            c=color,
            alpha=alpha,
            lw=lw)
    return ax


def plot_as_graph(
    g,
    s:int = None,
):
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    if s is None:
        steps = list(range(g.nb_steps))
        title = "All steps"
    else:
        steps = [s]
        title = "step s = " + str(s)
    for s in steps:
        for t in range(g.T):
            vertices = g.get_alive_vertices(s, t)
            ax = plot_vertices(ax, g, vertices, t)
            if (t < g.T-1):
                edges = g.get_alive_edges(s, t)
                ax = plot_edges(ax, g, edges,t)
    ax.set_title(title)
    return fig, ax

    # for each time steps plot vertices at this
    # for each time steps plots edges

