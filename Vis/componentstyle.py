
import numpy as np
from abc import ABC, abstractmethod

from ..utils.functions import linear

class ComponentStyle(ABC):

    def __init__(
        self,
        max_opacity: bool = False,
        lw_min: float = 0.5,
        lw_max: float = 8,
        color_list: List = None,
    ):
        self.max_opacity = max_opacity
        self.lw_min = lw_min
        self.lw_max = lw_max
        self.color_list = color_list

    @abstractmethod
    def f_component(self, g, c, i):
        pass

    def component_function(
        g,
        components,
    ):
        """
        Generates a nested list of objects to generate collections
        """
        # One list of objects for each variable dimension
        axs_objects = [
            [self.f_component(g, c, i) for c in components]
            for i in range(g.d)
        ]
        return axs_objects

    @abstractmethod
    def f_color(self, c):
        pass

    def color_function(self, components):
        colors = np.asarray(
            [self.color_list[self.f_color(c)] for c in components]
        ).reshape((-1, 4))
        return colors

    def f_alpha(self, c):
        return linear(c.life_span, range0_1 = True)

    def alpha_function(self, components):
        if self.max_opacity:
            alphas = 1
        else:
            alphas = [ self.f_alpha(c) for c in components ]
        return alphas

    def f_size(self, c):
        sizes = linear(
            c.ratio_members,
            range0_1 = False,
            f0=self.lw_min,
            f1=self.lw_max
        )
        return sizes

    def size_function(self, components):
        return np.asarray([ self.f_size(c) for c in components ])

    @abstractmethod
    def f_collect(object, colors, lw):
        pass

    def cdraw(self, g, components):
        if components:

            # Colors, alphas and lw are common between all subvertices
            colors = self.color_function(components)
            colors[:,3] = self.alpha_function(components)
            lw = self.size_function(components)

            # One vertex gives g.d objects
            axs_objects = self.component_function(g, components)

            # matplotlib.collections.Collection
            # One collection for each in [0..g.d-1]
            axs_collect = [
                self.f_collect(objects = objects, colors = colors, lw = lw)
                for objects in axs_objects
            ]
            return axs_collect
