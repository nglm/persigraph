from persisentgraph import PersistentGraph
from vertex import Vertex
import numpy as np
import matplotlib.pyplot as plt
from colour import Color

red = Color("Darkredred")
colors = list(red.range_to(Color("green"),10))

def ratio(g, objects):
    ratio = []
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects
        s_born = obj.s_born
        s_death = obj.s_death
        ratio.append(
            (g.distances[s_born] - g.distances[s_death])
            / g.distances[0]
        )
    return ratio

def plot_vertices(
    axs,
    vertices,
    t,
):

    return None

def plot_edges(
    axs,
    edges,
    t,
):
    return None


def plot_as_graph(
    g,
    s,
):
    fig, axs = plt.subplots()
    for t in range(g.T):
        vertices = g.get_alive_vertices(s, t)
        edges = g.get_alive_edges(s, t)
        plot_vertices(axs, vertices, t)
        plot_edges(axs, edges,t)

    # for each time steps plot vertices at this
    # for each time steps plots edges

