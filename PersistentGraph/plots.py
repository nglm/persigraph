import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, EllipseCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
import random
from math import exp
from PersistentGraph.analysis import sort_components_by
from PersistentGraph import Vertex
from PersistentGraph import Edge
from typing import List
from PIL import ImageColor

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

# See https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=8
COLOR_BREWER = [
    "#377eb8", # Blue
    "#ff7f00", # Orange
    "#4daf4a", # Green
    "#984ea3", # Purple
    "#a65628", # Brown
    "#f781bf", # Pink
    "#e41a1c", # Red
    "#ffff33", # Yellow
]

COLOR_BREWER_RGB = [np.array(ImageColor.getcolor(c, "RGB"))/255 for c in COLOR_BREWER]

COLOR_BREWER_RGBA = [np.r_[c, np.ones(1)] for c in COLOR_BREWER_RGB]

# We will try to use color brewer instead
# def get_list_colors(
#     N,
#     seed: int = 22
# ):
#     cm = plt.get_cmap('tab10', lut=N)
#     list_colors = [cm(i) for i in range(N)]
#     random.Random(seed).shuffle(list_colors)
#     return list_colors

def get_list_colors(
    N: int,
) -> List:
    """
    Repeat COLOR_BREWER list until we get exactly N colors

    :param N: Number of colors desired
    :type N: int
    :return: List of colors (taken from COLOR_BREWER list)
    :rtype: List
    """
    n_cb = len(COLOR_BREWER)
    list_colors = []
    for i in range(1 + N//n_cb) :
        list_colors += COLOR_BREWER_RGBA
    return list_colors[:N]



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
        res = f1*res + f0*(1-res)
    return res



def linear(
    x,
    range0_1 = True,
    shift=0.,
    a=1.,
    f0=0.7,
    f1=6,
):
    # Use min and max because of precision error
    res = min(max(0, x), 1)
    # Here we will get f(0) = f0 and f(1) = f1
    if not range0_1:
        res = f1*res + f0*(1-res)
    return res


def __std_polygon(g, edges):
    '''
    Define a polygon representing the uncertainty of each edge in edges

    Return a nested list (nb edges, 1 polygon)
    '''
    t_start = g.time_axis[edges[0].time_step]
    t_end = g.time_axis[edges[0].time_step + 1]
    polys = [[
        (t_start, e.v_start.info["params"][0] - e.v_start.info["params"][1]),
        (t_start, e.v_start.info["params"][0] + e.v_start.info["params"][1]),
        (t_end,   e.v_end.info["params"][0] + e.v_end.info["params"][1]),
        (t_end,   e.v_end.info["params"][0] - e.v_end.info["params"][1]),]
        for e in edges
    ]
    return polys



def __edges_lines(g, edges):
    '''
    Define a line representing the edge for each edge in edges

    Return a nested list (nb edges, 1 line)
    '''
    t_start = g.time_axis[edges[0].time_step]
    t_end = g.time_axis[edges[0].time_step + 1]
    lines = [
        (
        (t_start,   e.v_start.info['params'][0]),
        (t_end,     e.v_end.info['params'][0])
        ) for e in edges
    ]
    return lines

def __vertices_circles(
    g,
    vertices,
    lw_min=0.5,
    lw_max=15,
    f=sigmoid,
    ):
    """
    Define a circle representing each vertex in vertices
    """
    t = g.time_axis[vertices[0].time_step]
    offsets = [(t, v.info['params'][0]) for v in vertices ]
    return offsets




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
        )[0]       # [0] because nents_by returns a nested list
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
                uniforms = [
                    c for c in components
                    if (c.v_start.info['type'] == 'uniform')
                    and (c.v_end.info['type'] == 'uniform')
                    ]
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
    lw_max=15,
    f=linear,
    color_list = get_list_colors(51),
    ax=None,
):
    if vertices:
        values = [ v.info['params'][0] for v in vertices ]
        alphas = [ f(v.life_span) for v in vertices ]

        # colors = np.asarray(
        #     [(f(v.life_span)*c1 + (1-f(v.life_span))*c2) for v in vertices]
        # ).reshape((-1, 4)) / 255
        colors = np.asarray(
            [color_list[v.info['brotherhood_size']] for v in vertices]
        ).reshape((-1, 4))

        colors[:,3] = alphas

        lw = np.asarray([
            f(v.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
            for v in vertices
        ])

        offsets = __vertices_circles(g, vertices,)
        circles = EllipseCollection(
            widths=lw,
            heights=lw,
            angles=0,
            units='points',
            facecolors=colors,
            offsets=offsets,
            transOffset=ax.transData,)
        ax.add_collection(circles)
    return ax

# def plot_uniform_vertices(
#     g,
#     vertices,
#     c2 = np.array([0,0,0,0]),
#     c1 = np.array([254,254, 254, 0]),
#     f=sigmoid,
#     ax=None,
# ):
#     if vertices:
#         up_lines = [ v.info['params'][0] for v in vertices ]
#         down_lines = [ v.info['params'][1] for v in vertices ]
#         alphas = [ f(v.life_span) for v in vertices ]

#         colors = np.asarray(
#             [(f(v.life_span)*c1 + (1-f(v.life_span))*c2) for v in vertices]
#         ).reshape((-1, 4)) / 255

#         colors[:,3] = alphas

#         t = vertices[0].time_step
#         t_range =  [g.time_axis[t]]
#         if t > 0:
#             t_range = [g.time_axis[t-1]] + t_range
#         if t < g.T-1:
#             t_range.append(g.time_axis[t+1])
#         n = len(t_range)
#         ax.fill_between(t_range, up_lines[0], down_lines[0], facecolor=colors)
#     return ax

def plot_gaussian_edges(
    g,
    edges,
    c1 = np.array([254,0,0,1]),
    c2 = np.array([254,254,0,1]),
    lw_min=0.3,
    lw_max=15,
    f=linear,
    color_list = get_list_colors(51),
    show_std = False,
    ax=None,
):
    if edges:
        alphas = [f(e.life_span) for e in edges]
        # colors = np.asarray(
        #     [(f(e.life_span)*c1 + (1-f(e.life_span))*c2)
        #     for e in edges]
        # ).reshape((-1, 4)) / 255
        colors = np.asarray(
            [color_list[e.v_start.info['brotherhood_size']] for e in edges]
        ).reshape((-1, 4))
        colors[:,3] = alphas

        lw = np.asarray([
            f(e.ratio_members, range0_1=False, f0=lw_min, f1=lw_max)
            for e in edges
        ])

        colors[:,3] = alphas
        lines = __edges_lines(g, edges)
        lines = LineCollection(
            lines,
            colors=colors,
            linewidths=lw,)
        ax.add_collection(lines)

        if show_std:

            polys = __std_polygon(g, edges)
            colors[:,3] /= 6
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
    f=sigmoid,
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
        ax=ax,
    )
    #ax = plot_uniform_edges()

    return ax

def plot_as_graph(
    g,
    s:float = None,
    show_vertices: bool = True,
    show_edges: bool = True,
    show_std: bool = True,
    threshold_m:int = 0,
    threshold_l:float = 0.00,
    fig = None,
    ax = None,
    fig_kw: dict = {"figsize" : (24,12)},
    ax_kw: dict = {'xlabel' : "Time (h)",
                   'ylabel' : "Temperature (°C)"}
):
    if ax is None:
        fig, ax = plt.subplots(**fig_kw)
        ax.autoscale()
        ax.set_xlabel(ax_kw['xlabel'])
        ax.set_ylabel(ax_kw['ylabel'])
        #ax.set_facecolor("whitesmoke")
        #ax.set_title(title)
    #ax.set_facecolor("whitesmoke")
    color_list = get_list_colors(g.N)
    if s is None:
        title = "All steps"
        for t in range(g.T):
            if show_vertices:
                ax = plot_vertices(
                    g, t, g.vertices[t],
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list,
                    ax=ax,
                )
            if show_edges and (t < g.T-1):
                ax = plot_edges(
                    g, t, g.edges[t],
                    threshold_m=threshold_m, threshold_l=threshold_l,
                    color_list = color_list,
                    show_std = show_std,
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
                    color_list = color_list,
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
                    color_list = color_list,
                    show_std = show_std,
                    ax=ax,
                )
    ax.autoscale()
    ax.set_xlabel(ax_kw['xlabel'])
    ax.set_ylabel(ax_kw['ylabel'])
    ax.set_title(title)
    return fig, ax

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
