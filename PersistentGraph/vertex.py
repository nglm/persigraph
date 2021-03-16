from . import Component
from typing import Sequence, List, Dict, Any

class Vertex(Component):

    def __init__(
        self,
        info: Dict[str, Any],
        t:int = None,
        num: int = None,
        members: List[int] = None,
        scores: Sequence[float] = None,
        score_ratios: Sequence[float] = None,
        score_bounds: float = None,
        total_nb_members: int = None,
    ):
        super().__init__(
            t=t,
            num=num,
            members = members,
            scores = scores,
            score_ratios = score_ratios,
            score_bounds = score_bounds,
            total_nb_members = total_nb_members,
        )
        self.info = info


    def is_equal_to(
        self,
        v = None,
        members: List[int] = None,
        nb_members: int = None,
        time_step: int = None,
        v_type: str = None
    ) -> bool:
        """
        Check if 2 vertices are equal

        Assume that ``members`` is an ordered list

        :param v: vertex to compare to
        :type v: Vertex
        :return: Boolean indicating whether v and self are equal
        :rtype: bool
        """
        if v is not None:
            members = v.members
            time_step = v.time_step
            nb_members = v.nb_members
            v_type = v.info['type']
        else:
            if nb_members is None:
                nb_members = len(members)
        if (self.time_step == time_step
            and nb_members == self.nb_members
            and v_type == self.info['type']
            ):
            res = True
            for i in range(nb_members):
                if members[i] != self.members[i]:
                    res = False
                    break
        else:
            res = False
        return res

    @property
    def info(self) ->  Dict[str, Any]:
        """
        Info related to the cluster Vertex represents

        Must contain a 'type' key, representing the type of cluster
        (uniform, guaussian, etc)

        :rtype: Dict[str, Any]
        """
        return self.__info

    @info.setter
    def info(self, info: Dict[str, Any]):
        if "type" not in info.keys():
            raise ValueError("Vertex.info must have a 'type' key")
        self.__info = info

