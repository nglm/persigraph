
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Any, Dict, Tuple

from ..utils.functions import linear, sigmoid

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
    def f_component(self, g, c, i, f_component_kw = {}):
        pass

    def component_function(
        self,
        g,
        components,
        f_component_kw = {},
    ):
        """
        Generates a nested list of objects to generate collections
        """
        # One list of objects for each variable dimension
        axs_objects = [
            [self.f_component(g, c, i, f_component_kw) for c in components]
            for i in range(g.d)
        ]
        return axs_objects

    @abstractmethod
    def f_color(self, c, f_color_kw = {}):
        pass

    def color_function(self, components, f_color_kw = {}):
        colors = np.asarray(
            [self.color_list[self.f_color(c, f_color_kw)] for c in components]
        ).reshape((-1, 4))
        return colors

    def f_alpha(self, c, f_alpha_kw = {}):
        return sigmoid(c.life_span, range0_1=True)

    def alpha_function(self, components, f_alpha_kw = {}):
        if self.max_opacity:
            alphas = 1
        else:
            alphas = [ self.f_alpha(c, f_alpha_kw) for c in components ]
        return alphas

    def f_size(self, c, f_size_kw = {}):
        sizes = linear(
            c.ratio_members,
            range0_1 = False,
            f0=self.lw_min,
            f1=self.lw_max
        )
        return sizes

    def size_function(self, components, f_size_kw = {}):
        return np.asarray([ self.f_size(c, f_size_kw) for c in components ])

    @abstractmethod
    def f_collect(self, object, colors, lw, f_collect_kw = {}):
        pass

    def cdraw(
        self,
        g,
        components,
        f_color_kw: dict = {},
        f_alpha_kw: dict = {},
        f_size_kw: dict = {},
        f_component_kw: dict = {},
        f_collect_kw: dict = {},
        ):
        if components:

            # Colors, alphas and lw are common between all subvertices
            colors = self.color_function(components, f_color_kw)
            colors[:,3] = self.alpha_function(components, f_alpha_kw)
            lw = self.size_function(components, f_size_kw)

            # One vertex gives g.d objects
            axs_objects = self.component_function(g, components, f_component_kw)

            # matplotlib.collections.Collection
            # One collection for each in [0..g.d-1]
            axs_collect = [
                self.f_collect(
                    objects = objects,
                    colors = colors,
                    lw = lw,
                    f_collect_kw = {'i' : i, **f_collect_kw}
                ) for (i, objects) in enumerate(axs_objects)
            ]

            return axs_collect
