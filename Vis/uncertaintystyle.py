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

    def f_component(self, g, c):
        t_start = g.time_axis[c.time_step]
        t_end = g.time_axis[c.time_step + 1]
        polys = (
            # std_inf at t
            (t_start, c.v_start.info["mean"] - c.v_start.info["std_inf"]),
            # std_sup at t
            (t_start, c.v_start.info["mean"] + c.v_start.info["std_sup"]),
            # std_sup at t+1
            (t_end,   c.v_end.info["mean"] + c.v_end.info["std_sup"]),
            # std_inf at t+1
            (t_end,   c.v_end.info["mean"] - c.v_end.info["std_inf"])
        )
        return polys

    def f_color(self, c):
        return c.v_start.info['brotherhood_size'][0]

    def f_alpha(self, c):
        return linear(c.life_span, range0_1 = True)/6

    def f_collect(objects, colors, lw):
        polys = PolyCollection(objects, facecolors=colors)
        return polys