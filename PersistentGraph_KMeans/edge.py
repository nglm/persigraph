from PersistentGraph_KMeans.component import Component
from PersistentGraph_KMeans.vertex import Vertex
from typing import Sequence, List

class Edge(Component):

    def __init__(
        self,
        v_start: Vertex,
        v_end: Vertex,
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
        Vertex from which Edge comes

        :rtype: int
        """
        return self.__v_start

    @v_start.setter
    def v_start(self, v_start: int):
        if v_start is not None:
            # if (v_start < 0):
            #     raise ValueError("v should be > O")
            # self.__v_start = int(abs(v_start))
            self.__v_start = v_start

    @property
    def v_end(self) -> Vertex:
        """
        Vertex to which Edge goes

        :rtype: int
        """
        return self.__v_end


    @v_end.setter
    def v_end(self, v_end: Vertex):
        if v_end is not None:
            # if (v_end < 0):
            #     raise ValueError("v should be > O")
            # self.__v_end = int(abs(v_end))
            self.__v_end = v_end


