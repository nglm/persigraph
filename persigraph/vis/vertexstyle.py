from matplotlib.collections import EllipseCollection
from typing import List, Sequence, Union, Any, Dict, Tuple

from . import ComponentStyle

class VertexStyle(ComponentStyle):

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
        return (g.time_axis[c.time_step], c.info['mean'][i])

    def f_color(self, g, c, f_color_kw = {}):
        return c.info['k'][0]

    def f_collect(self, objects, colors, lw, f_collect_kw = {}):
        circles = EllipseCollection(
            widths = lw,
            heights = lw,
            angles = 0,
            units = 'points',
            facecolors = colors,
            offsets = objects,
            transOffset = f_collect_kw['axs'][f_collect_kw['i']].transData)
        return circles
