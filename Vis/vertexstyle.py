
class VertexStyle():

    def __init__(
        self,
        color_function,
        alpha_function,
        vertex_function,
        size_function,
        f_alpha,
        f_lw,
        max_opacity: bool = False,
        lw_min: float = 0.5,
        lw_max: float = 8,
        color_list: List = None,
    ):
        if color_function is None:
            self.color_function = self._color_function
        else:
            self.color_function = color_function

        if alpha_function is None:
            self.alpha_function = self._alpha_function
        else:
            self.alpha_function = alpha_function

        if vertex_function is None:
            self.vertex_function = self._vertex_function
        else:
            self.vertex_function = vertex_function

        if f_alpha is None:
            self.f_alpha = self._f_alpha
        else:
            self.f_alpha = f_alpha

        if f_lw is None:
            self.f_lw = self._f_lw
        else:
            self.f_lw = f_lw

        if color_list is None:
            

        self.max_opacity = max_opacity
        self.lw_min = lw_min
        self.lw_max = lw_max


    def _vertex_function(
        g,
        vertices,
    ):
        """
        Define a circle representing each vertex in vertices
        """
        t = g.time_axis[vertices[0].time_step]
        values = [(t, v.info['mean']) for v in vertices ]
        return values

    def _f_lw(self, x):
        return linear(x, range0_1 = False, f0=self.lw_min, f1=self.lw_max)

    def _f_alpha(self, x):
        return linear(x, range0_1 = True)

    def _color_function(self, vertices):
        # The color of a vertex is the color of its smallest brotherhood size
        colors = np.asarray(
                [
                    self.color_list[v.info['brotherhood_size'][0]]
                    for v in vertices]
            ).reshape((-1, 4))
        return colors

    def _alpha_function(self, vertices):
        if self.max_opacity:
            alphas = 1
        else:
            alphas = [ self.f_alpha(v.life_span) for v in vertices ]
        return alphas

    def _size_function(self, vertices):
        return np.asarray([ self.f_lw(v.ratio_members) for v in vertices ])

    def vdraw(self, g, vertices):
        if vertices:

            colors = self.color_function(vertices)
            colors[:,3] = self.alpha_function(vertices)
            lw = self.size_function(vertices)
            values = self._vertex_function(g, vertices)

            # matplotlib.collections.Collection
            circles = EllipseCollection(
                widths=lw,
                heights=lw,
                angles=0,
                units='points',
                facecolors=colors,
                offsets=values,
                transOffset=ax.transData,)
            return circles
