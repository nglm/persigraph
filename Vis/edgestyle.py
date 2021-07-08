from matplotlib.collections import LineCollection

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

    def f_component(self, g, c, i):
        t_start = g.time_axis[c.time_step]
        t_end = g.time_axis[c.time_step + 1]
        line = (
            (t_start, c.v_start.info['mean'][i]),
            (t_end,   c.v_end.info['mean'][i])
        )
        return line

    def f_color(self, c):
        return c.v_start.info['brotherhood_size'][0]

    def f_collect(objects, colors, lw):
        lines = LineCollection(
            objects,
            colors = colors,
            linewidths = lw)
        return lines

