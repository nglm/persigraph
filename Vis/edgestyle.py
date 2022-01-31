from matplotlib.collections import LineCollection
from typing import List, Sequence, Union, Any, Dict, Tuple

from . import ComponentStyle

class EdgeStyle(ComponentStyle):

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
        line = (
            (t_start, g._vertices[c.time_step][c.v_start].info['mean'][i]),
            (t_end,   g._vertices[c.time_step + 1][c.v_end].info['mean'][i])
        )
        return line

    def f_color(self, g,  c, f_color_kw = {}):
        return g._vertices[c.time_step][c.v_start].info['brotherhood_size'][0]

    def f_collect(self, objects, colors, lw, f_collect_kw = {}):
        lines = LineCollection(
            objects,
            colors = colors,
            linewidths = lw)
        return lines

