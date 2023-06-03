from typing import Sequence, List, Dict, Any, Tuple
from . import Component
from ..utils.check_variable import check_int_positive
from ..utils._clustering import compute_cluster_params

import numpy as np

class Edge(Component):
    """
    Graph Component that link 2 vertices of a PersistentGraph
    """





    @staticmethod
    def info(v, members: List[int]):
        """
        Compute edge params based on v.info["X"]

        v.info["X"] contains aligned X values
        """
        ind_X = v.index_members(members)
        X = v.info["X"][ind_X]
        info = compute_cluster_params(X)
        return info

    @staticmethod
    def ratio_scores(v_start, v_end) -> Tuple[List[float]]:
        """
        Compute scores and ratios based on vertices

        :param v_start: v_start
        :type v_start: Vertex
        :param v_end: v_end
        :type v_end: Vertex
        :return: [score_birth, score_death] and [ratio_birth, ratio_death]
        :rtype: Tuple[List[float]]
        """
        # ------------ policy 1: common life span --------------------
        argbirth = np.argmax([v_start.score_ratios[0], v_end.score_ratios[0]])
        argdeath = np.argmin([v_start.score_ratios[1], v_end.score_ratios[1]])

        # ------------ policy 2: min of life span --------------------
        # argbirth = np.argmin([v_start.life_span, v_end.life_span])
        # argdeath = argbirth

        # Note that score birth and death might not be consistent but
        # The most important thing is the ratios which must be consistent
        score_birth = [v_start.scores[0], v_end.scores[0]][argbirth]
        score_death = [v_start.scores[1], v_end.scores[1]][argdeath]
        scores = [score_birth, score_death]

        ratio_birth = [v_start.score_ratios[0], v_end.score_ratios[0]][argbirth]
        ratio_death = [v_start.score_ratios[1], v_end.score_ratios[1]][argdeath]
        ratios = [ratio_birth, ratio_death]

        return scores, ratios

    def __init__(
        self,
        info_start: Dict[str, Any],
        info_end: Dict[str, Any],
        v_start: int,
        v_end: int,
        t: int = None,
        num: int = None,
        members : List[int] = None,
        scores: Sequence[float] = None,
        score_ratios: Sequence[float] = None,
        score_bounds: Sequence[float] = None,
        total_nb_members: int = None,
    ):
        super().__init__(
            t=t,
            num=num,
            members = members,
            scores=scores,
            score_ratios = score_ratios,
            score_bounds = score_bounds,
            total_nb_members = total_nb_members,
        )

        self.v_start = v_start
        self.v_end = v_end
        self.info_start = info_start
        self.info_end = info_end

    def is_equal_to(
        self,
        e,
    ) -> bool:
        """
        Check if 2 edges are equal

        :param e: edge to compare to
        :type e: Edge
        :return: Boolean indicating whether e and self are equal
        :rtype: bool
        """
        res = False
        if (self.time_step == e.time_step
            and e.v_start == self.v_start
            and e.v_end == self.v_end
        ):
            res = True
        return res

    @property
    def v_start(self) -> int:
        """
        Vertex num from which Edge comes

        :rtype: int
        """
        return self.__v_start

    @v_start.setter
    def v_start(self, v_start: int):
        if v_start is not None:
            check_int_positive(v_start, 'Vertex start')
            self.__v_start = int(abs(v_start))

    @property
    def v_end(self) -> int:
        """
        Vertex num to which Edge goes

        :rtype: int
        """
        return self.__v_end


    @v_end.setter
    def v_end(self, v_end: int):
        if v_end is not None:
            check_int_positive(v_end, 'Vertex end')
            self.__v_end = int(abs(v_end))


    @property
    def info_start(self) ->  Dict[str, Any]:
        """
        Info related to the start of the edge

        :rtype: Dict[str, Any]
        """
        return self.__info_start

    @info_start.setter
    def info_start(self, info_start: Dict[str, Any]):
        self.__info_start = info_start


    @property
    def info_end(self) ->  Dict[str, Any]:
        """
        Info related to the end of the edge

        :rtype: Dict[str, Any]
        """
        return self.__info_end

    @info_end.setter
    def info_end(self, info_end: Dict[str, Any]):
        self.__info_end = info_end
