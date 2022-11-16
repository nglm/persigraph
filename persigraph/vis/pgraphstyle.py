import numpy as np
from typing import List, Sequence, Union, Any, Dict

from . import VertexStyle, EdgeStyle, UncertaintyStyle

from .commonstyle import get_list_colors

from ..utils.lists import to_iterable
from ..persistentgraph import Vertex
from ..persistentgraph import Edge
from ..persistentgraph.analysis import sort_components_by

class PGraphStyle():


    def __init__(
        self,
        m_min: int = 0,
        l_min: float = 0.,
        show_uniform: bool = False,
        show_uncertainty: bool = True,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_bar: bool = True,
        color_list: List = None,
        lw_min: float = 0.5,
        lw_max: float = 8.,
        max_opacity: bool = False,
        vertices = None,
        edges = None,
        uncertainty = None,
    ):
        self.m_min = m_min
        self.l_min = l_min
        self.show_uniform = show_uniform
        self.show_uncertainty = show_uncertainty
        self.show_vertices = show_vertices
        self.show_edges = show_edges
        self.show_bar = show_bar
        if color_list is None:
            self.color_list = get_list_colors
        else:
            self.color_list = color_list

        self.lw_min = lw_min
        self.lw_max = lw_max
        self.max_opacity = max_opacity
        if vertices is None:
            self.vertices = VertexStyle(
                max_opacity = max_opacity,
                lw_min = lw_min,
                lw_max = lw_max,
                color_list = color_list,
            )
        else:
            self.vertices = vertices
        if edges is None:
            self.edges = EdgeStyle(
                max_opacity = max_opacity,
                lw_min = lw_min,
                lw_max = lw_max,
                color_list = color_list,
            )
        else:
            self.edges = edges
        if uncertainty is None:
            self.uncertainty = UncertaintyStyle(
                max_opacity = max_opacity,
                lw_min = lw_min,
                lw_max = lw_max,
                color_list = color_list,
            )
        else:
            self.uncertainty = uncertainty


    def sort_components(
        self,
        g,
        components,
    ):
        components = [
            c for c in components
            if (c.nb_members > self.m_min and c.life_span > self.l_min )
        ]
        types = ['gaussian','KMeans','Naive']
        u_types = ['uniform']
        # If there are still components matching the thresholds
        if components:

            # Sort by life span to emphasize components that live longer
            components = sort_components_by(
                components, criteron='life_span', descending=False
            )[0]     # [0] because sort_components_by returns a nested list

            # --------------------- VERTICES ---------------------------
            if isinstance(components[0], Vertex):
                gaussians = [
                    c for c in components if c.info['type'] not in u_types
                ]
                uniforms = [
                    c for c in components if c.info['type'] in u_types
                ]
            # ----------------------- EDGES ----------------------------
            elif isinstance(components[0], Edge):
                gaussians = [
                    c for c in components
                    if (
                        g._vertices[c.time_step][c.v_start].info['type']
                        not in u_types
                    and
                        g._vertices[c.time_step + 1][c.v_end].info['type']
                        not in u_types
                    )]
                from_to_uniforms = [
                    c for c in components
                    if (g._vertices[c.time_step][c.v_start].info['type'] in u_types)
                    and (g._vertices[c.time_step + 1][c.v_end].info['type'] in u_types)
                    ]
                to_uniforms = [
                    c for c in components
                    if (g._vertices[c.time_step][c.v_start].info['type'] not in u_types)
                    and (g._vertices[c.time_step + 1][c.v_end].info['type'] in u_types)
                    ]
                from_uniforms = [
                    c for c in components
                    if (g._vertices[c.time_step][c.v_start].info['type'] in u_types)
                    and (g._vertices[c.time_step + 1][c.v_end].info['type'] not in u_types)
                    ]
                uniforms = [to_uniforms, from_to_uniforms, from_uniforms]
        else:
            gaussians = []
            uniforms = []

        return gaussians, uniforms

    def gdraw(
        self,
        axs,
        g,
        vertices=None,
        edges=None,
        relevant_k=None,
        t=None):
        """
        Returns an aggregation of collections
        """
        if t is None:
            t_range = range(g.T)
        else:
            t_range = to_iterable(t)

        axs_collections = [[] for _ in range(g.d)]
        for t in t_range:

            # Collections for vertices if necessary
            if self.show_vertices:

                # Keep only vertices respecting the thresholds,
                # and distinguish between gaussian and uniform vertices
                gaussians, uniforms = self.sort_components(g, vertices[t])

                # Collections for gaussian vertices
                axs_collect = self.vertices.cdraw(
                    g,
                    gaussians,
                    f_collect_kw = {"axs": axs.flat}
                )
                for tot_col, part_col in zip(axs_collections, axs_collect):
                    tot_col.append(part_col)

                # Collections for uniform vertices if necessary
                if self.show_uniform:
                    raise NotImplementedError('Cannot display uniform vertices')

            # Collections for edges if necessary
            if (self.show_edges or self.show_uncertainty) and (t < g.T-1):

                # Keep only edges respecting the thresholds,
                # and distinguish between gaussian and uniform edges
                gaussians, uniforms = self.sort_components(g, edges[t])

                # Collections for gaussian edges
                if self.show_edges:
                    axs_collect = self.edges.cdraw(g, gaussians)
                    for tot_col, part_col in zip(axs_collections, axs_collect):
                        tot_col.append(part_col)

                # Collections for uniform edges if necessary
                if self.show_uniform:
                    raise NotImplementedError('Cannot display uniform edges')

                # Collections for uncertainty if necessary
                if self.show_uncertainty:

                    # Collections for uncertainty
                    axs_collect = self.uncertainty.cdraw(g, gaussians)
                    for tot_col, part_col in zip(axs_collections, axs_collect):
                        tot_col.append(part_col)

        return axs_collections


