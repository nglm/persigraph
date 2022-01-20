from scipy.spatial.distance import euclidean
import numpy as np
from typing import List, Sequence

from ..utils.check_variable import check_O1_range, check_positive, check_all_positive
from ..utils.sorted_lists import has_element, get_common_elements, bisect_search


class Component():
    """
    Base class for Vertex and Edge, components of a PersistentGraph
    """

    key_incr:int = 0

    def __init__(
        self,
        t: int = None,
        num: int = None,
        members : List[int] = [],
        scores: Sequence[float] = None,
        score_ratios: Sequence[float] = None,
        score_bounds: float = None,
        total_nb_members: int = None,
    ):
        self.__key: int = Component.key_incr
        self.num = num
        self.time_step = t
        self.members = members
        self.scores = scores
        self.compute_ratio_members(total_nb_members = total_nb_members)
        if score_ratios is None:
            self._compute_ratio_scores(score_bounds = score_bounds)
        else:
            self.score_ratios = score_ratios

        Component.key_incr += 1

    def reset_key_incr(self):
        Component.key_incr = 0

    def has_member(self, m: int) -> bool:
        """
        Check if a member belongs to the component

        Assume that Component.members are sorted

        :param m: index of the member to find
        :type m: int
        :return: True if self has this member, False otherwise
        :rtype: List[int]
        """
        return has_element(self.__members, m, len_l = self.nb_members)

    def get_common_members(
        self,
        cmpt,
    ) -> List[int]:
        """
        Return the common members between self and cmpt

        Assume that Component.members are sorted

        :param cmpt: Component to compare to
        :type cmpt: Component
        :return: Common members
        :rtype: List[int]
        """
        return get_common_elements(self.__members, cmpt.members)

    def compute_ratio_members(self, total_nb_members):
        if total_nb_members is None or self.nb_members is None:
            self.__ratio_members = None
        else:
            ratio_members = self.nb_members/total_nb_members
            check_O1_range(ratio_members, 'Ratio members')
            self.__ratio_members = ratio_members

    def _compute_ratio_scores(
        self,
        score_bounds = None,
        ):
        if score_bounds is None or self.scores is None:
            self.__ratio_scores = None
        else:
            # Normalizer so that ratios are within 0-1 range
            norm = np.abs(score_bounds[0] - score_bounds[1])

            # BIRTH
            # If score_birth is ``None`` or 0 it means that the component is
            # alive since the very beginning
            if self.scores[0] is None:
                ratio_birth = 0.
            else:
                ratio_birth = np.abs(self.scores[0] - score_bounds[0]) / norm


            # DEATH
            # If score_death is ``None`` it means that the component is not
            # dead at the end
            if self.scores[1] is None:
                ratio_death = 1.
            else:
                ratio_death = np.abs(self.scores[1] - score_bounds[0]) / norm

            self.score_ratios = [ratio_birth, ratio_death]




    @property
    def key(self) -> int:
        """
        Number of the component

        :rtype: int
        """
        return self.__key

    @property
    def num(self) -> int :
        """
        Number of the component (unique at that time step)

        :rtype: int
        """
        return self.__num

    @num.setter
    def num(self, num: int):
        """
        Number of the component (unique at that time step)

        :type num: int
        :raises ValueError: If ``num`` is not > 0
        """
        if num is not None:
            check_positive(num, 'num')
            self.__num = int(abs(num))
        else:
            num = None

    @property
    def time_step(self) -> int:
        """
        Time step at which Component exists

        :rtype: int
        """
        return self.__time_step

    @time_step.setter
    def time_step(self, t: int):
        if (t < 0):
            raise ValueError("t should be >= 0")
        self.__time_step = int(t)

    @property
    def members(self) -> List[int]:
        """
        Ordered list of members indices belonging to Component

        :rtype: List[int]
        """
        return self.__members

    @members.setter
    def members(self, members: List[int]):
        if members is not None:
            check_all_positive(members, 'members')
            self.__members = [int(m) for m in members]
            self.__nb_members = len(members)
        else:
            self.__members = []


    @property
    def ratio_members(self) -> float:
        """
        Ratio of members belonging to Component: ``nb_members`` / N

        :rtype: float
        """
        return self.__ratio_members

    @property
    def nb_members(self) -> int:
        """
        Number of members belonging to Component

        :rtype: int
        """
        return self.__nb_members

    @nb_members.setter
    def nb_members(self, nb_members):
        self.__nb_members = int(nb_members)


    @property
    def scores(self) -> Sequence[float]:
        """
        Sequence (score_birth, score_death) "scores" depends on the method used.

        Naive method: max distance between members
        GMM: average log-likelihood of the members

        :rtype: Sequence[float]
        """
        return self.__scores

    @scores.setter
    def scores(self, scores: Sequence[float]):
        if scores is None:
            self.__scores = None
        else:
            if len(scores) != 2:
                raise ValueError("scores should be a sequence of 2 elements")
            else:
                self.__scores = scores

    @property
    def score_ratios(self) -> Sequence[float]:
        """
        Sequence (ratio_score_birth, ratio_score_death) "scores" depends on
        the method used.

        Naive method: max distance between members
        GMM: average log-likelihood of the members

        :rtype: Sequence[float]
        """
        return self.__score_ratios

    @score_ratios.setter
    def score_ratios(self, score_ratios):
        if score_ratios is None:
            self.__score_ratios = None
            self.__life_span = None
        else:
            if len(score_ratios) != 2:
                raise ValueError("score_ratios should be a sequence of 2 elements")

            check_O1_range(score_ratios[0], 'Ratio birth')
            check_O1_range(score_ratios[1], 'Ratio death')

            self.__score_ratios = score_ratios

            # LIFE SPAN
            # Note: ratio death must always be >= ratio_birth
            # Note: life_span is with 0-1 range
            life_span = euclidean(score_ratios[0], score_ratios[1])
            self.__life_span = np.around(min(max(life_span,0), 1), 3)


    @property
    def life_span(self) -> float:
        """
        Life span of Component: ``ratio_death`` - ``ratio_birh``

        :rtype: float
        """
        return self.__life_span











