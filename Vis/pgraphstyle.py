import numpy as np
from typing import List, Sequence, Union, Any, Dict



class PGraphStyle():


    def __init__(
        self,
        threshold_m: int,
        threshold_l: int,
        show_uniform: bool,
        show_std: bool,
        show_vertices: bool,
        show_edges: bool,
        color_list,
        lw_min: float,
        lw_max: float,
    ):
        self.threshold_m = threshold_m
        self.threshold_l = threshold_l



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

    def gdraw(self):
        """
        Returns an aggregation of collections
        """
        # t in range:

        # # sort components

        # # create collections for vertices if necessary
        # # # create collections for uniform vertices if necessary

        # # create collections for edges if necessary
        # # # create collections for uniform edges if necessary

        # # create collections for std if necessary

        # # return collections


