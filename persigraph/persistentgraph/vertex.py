from . import Component
from ..utils.check_variable import check_int_positive
from ..utils.sorted_lists import are_equal
from typing import Sequence, List, Dict, Any

class Vertex(Component):

    def __init__(
        self,
        info: Dict[str, Any],
        t:int = None,
        num: int = None,
        members: List[int] = None,
        score_ratios: Sequence[float] = None,
        total_nb_members: int = None,
    ):
        super().__init__(
            t=t,
            num=num,
            members = members,
            score_ratios = score_ratios,
            total_nb_members = total_nb_members,
        )
        self.info = info
        self.__e_to = []
        self.__e_from = []

    def is_equal_to(
        self,
        v = None,
        members: List[int] = None,
        time_step: int = None,
        v_type: str = None
    ) -> bool:
        """
        Check if 2 vertices are equal.

        Assume that ``members`` is an ordered list.

        :param v: vertex to compare to
        :type v: Vertex
        :return: Boolean indicating whether v and self are equal
        :rtype: bool
        """
        if v is not None:
            members = v.members
            time_step = v.time_step
            v_type = v.info['type']
        if (self.time_step == time_step
            and v_type == self.info['type']
            ):
            return are_equal(self.members, members)
        else:
            return False

    def index_members(self, members: Sequence[int]) -> List[int]:
        """
        Find indices of `members` in `Vertex.members`

        Assume that `members` is a subset of `Vertex.members`. Useful
        when computing info of potential edges going to/coming from
        `Vertex`.

        :param members: _description_
        :type members: Sequence[int]
        :return: Indices of members represented by `Vertex`
        :rtype: List[int]
        """
        return [self.members.index(m) for m in members]

    def add_edge_to(
        self,
        e: int
    ) -> None:
        """
        Append `e` to `e_to`, representing an edge coming to `Vertex`

        :param e: num that represents an edge coming to `Vertex`
        :type e: int
        """
        check_int_positive(e, 'edge to')
        self.__e_to.append(int(e))

    def add_edge_from(
        self,
        e: int
    ) -> None:
        """
        Append `e` to `e_from`, representing an edge coming from `Vertex`

        :param e: num that represents an edge coming from `Vertex`
        :type e: int
        """
        check_int_positive(e, 'edge from')
        self.__e_from.append(int(e))

    @property
    def info(self) ->  Dict[str, Any]:
        """
        Info related to the cluster Vertex represents

        Must contain a 'type' key, representing the type of cluster
        (uniform, gaussian, etc).

        Available keys:

        - `X`: representing $X_{t, vert}[:,:,mid_w]$, that is to say,
        aligned member values at $t$ if DTW is used, and original
        member values otherwise.
        - `mean`: representing the cluster center. If a time window is
        used, the midpoint of the time window is used to compute the
        mean. If DTW is used, DBA is used instead of the usual
        definition of the mean.
        - `std`: representing the cluster uncertainty.
        - `std_inf`: representing the cluster uncertainty under `mean`
        - `std_sup`: representing the cluster uncertainty above `mean`

        :rtype: Dict[str, Any]
        """
        return self.__info

    @info.setter
    def info(self, info: Dict[str, Any]):
        if "type" not in info.keys():
            raise ValueError("Vertex.info must have a 'type' key")
        self.__info = info

    @property
    def e_from(self) -> List[int]:
        """
        List of edges num coming from `Vertex`.

        :return: List of edges num coming from `Vertex`.
        :rtype: List[int]
        """
        return self.__e_from

    @property
    def e_to(self) -> List[int]:
        """
        List of edges num coming to `Vertex`.

        :return: List of edges num coming to `Vertex`.
        :rtype: List[int]
        """
        return self.__e_to

