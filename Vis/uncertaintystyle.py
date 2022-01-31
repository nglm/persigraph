from matplotlib.collections import PolyCollection
from typing import List, Sequence, Union, Any, Dict, Tuple

from . import ComponentStyle

from ..utils.functions import linear

class UncertaintyStyle(ComponentStyle):

    def __init__(
        self,
        max_opacity: bool = False,
        lw_min: float = 0.5,
        lw_max: float = 8,
        color_list: List = None,
    ):
        super().__init__(
            max_opacity = max_opacity,
            lw_min = lw_min,
            lw_max = lw_max,
            color_list = color_list,
        )

    def f_component(self, g, c, i, f_component_kw = {}):
        t_start = g.time_axis[c.time_step]
        t_end = g.time_axis[c.time_step + 1]
        v_start = g._vertices[c.time_step][c.v_start]
        v_end = g._vertices[c.time_step + 1][c.v_end]
        polys = (
            # std_inf at t
            (t_start, v_start.info["mean"][i] - v_start.info["std_inf"][i]),
            # std_sup at t
            (t_start, v_start.info["mean"][i] + v_start.info["std_sup"][i]),
            # std_sup at t+1
            (t_end,   v_end.info["mean"][i] + v_end.info["std_sup"][i]),
            # std_inf at t+1
            (t_end,   v_end.info["mean"][i] - v_end.info["std_inf"][i])
        )
        return polys

    def f_color(self, g, c, f_color_kw = {}):
        return g._vertices[c.time_step][c.v_start].info['brotherhood_size'][0]

    def f_alpha(self, g, c, f_alpha_kw = {}):
        return linear(c.life_span, range0_1 = True)/6

    def alpha_function(self, g, components, f_alpha_kw = {}):
        if self.max_opacity:
            alphas = 1/6
        else:
            alphas = [ self.f_alpha(g, c, f_alpha_kw) for c in components ]
        return alphas

    def f_collect(self, objects, colors, lw, f_collect_kw = {}):
        polys = PolyCollection(objects, facecolors=colors)
        return polys