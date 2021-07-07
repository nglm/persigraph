
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

    def f_component(self, t, c):
        return (g.time_axis[c.time_step], c.info['mean'])

    def f_color(self, c):
        return c.info['brotherhood_size'][0]

    def f_collect(objects, colors, lw):
        circles = EllipseCollection(
            widths = lw,
            heights = lw,
            angles = 0,
            units = 'points',
            facecolors = colors,
            offsets = objects,
            transOffset = ax.transData)
        return circles
