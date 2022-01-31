from TopEns.utils.check_variable import check_int_positive
from . import Component
from typing import Sequence, List, Dict, Any

class Edge(Component):
    """
    Graph Component that link 2 vertices of a PersistentGraph
    """

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
