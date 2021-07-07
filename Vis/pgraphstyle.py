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
        color_vertices,
        color_edges,
        alpha_vertices,
        alpha_edges,
        f_lw,
        f_opacity,
        lw_min: float,
        lw_max: float,
        draw_vertices
        draw_edges,
        draw_uncertainty,


    ):
        self.threshold_m = threshold_m
        self.threshold_l = threshold_l



