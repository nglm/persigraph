import numpy as np
from typing import List, Sequence

from ..utils.check_variable import (
    check_O1_range, check_int_positive, check_all_int_positive,
    check_is_leq,
)
from ..utils.sorted_lists import has_element, get_common_elements,
from ..utils.lists import union_intervals, intersection_intervals

class Component():
    """
    Base class for Vertex and Edge, components of a PersistentGraph
    """

    key_incr:int = 0

    @staticmethod
    def ratio_intersection(
        ratios1: List[Sequence[float]],
        ratios2: List[Sequence[float]]
    ) -> List[Sequence[float]]:
        """
        Compute intersection of ratios

        :param ratios1: First list of ratios
        :type ratios1: List[Sequence[float]]
        :param ratios2: Second list of ratios
        :type ratios2: List[Sequence[float]]
        :return: Intersection of list of ratios
        :rtype: List[Sequence[float]]
        """

        return intersection_intervals(ratios1, ratios2)

    @staticmethod
    def ratio_union(
        ratios1: List[Sequence[float]],
        ratios2: List[Sequence[float]]
    ) -> List[Sequence[float]]:
        """
        Compute union of ratios

        :param ratios1: First list of ratios
        :type ratios1: List[Sequence[float]]
        :param ratios2: Second list of ratios
        :type ratios2: List[Sequence[float]]
        :return: Intersection of list of ratios
        :rtype: List[Sequence[float]]
        """
        return union_intervals(ratios1, ratios2)

    @staticmethod
    def contemporaries(c1, c2, verbose=False) -> bool:
        """
        Check if two components are contemporaries

        Useful before defining an edge for example

        :param c1: Component 1
        :type c1: Component
        :param c2: Component 2
        :type c2: Component
        :param verbose: if warnings should be printed, defaults to False
        :type verbose: bool, optional
        :return: True if they are contemporaries, False otherwise
        :rtype: bool
        """
        # If c1 is dead before c2 is even born
        # Or if c2 is dead before c1 is even born
        ratios = Component.ratio_intersection(c1.score_ratios, c2.score_ratios)
        if ratios == []:
            if verbose:
                print("WARNING: Components are not contemporaries")
                print("c1 ratios: ", c1.score_ratios)
                print("c2 ratios: ", c2.score_ratios)
            return False
        else:
            return True

    @staticmethod
    def have_common_members(c1, c2) -> bool:
        """
        Efficiently checks if two components have common members
        """
        return bool(set(c1.members).intersection(set(c2.members)))

    @staticmethod
    def common_members(c1, c2, verbose=False) -> List[int]:
        """
        Return common members of 2 components

        Useful before defining an edge for example

        :param c1: Component 1
        :type c1: Component
        :param c2: Component 2
        :type c2: Component
        :param verbose: if warnings should be printed, defaults to False
        :type verbose: bool, optional
        :return: List of common members
        :rtype:  List[int]
        """
        members = c1.get_common_elements(c2)

        if verbose and not members:
            print("WARNING: No common members")
        return members

    def __init__(
        self,
        t: int = None,
        num: int = None,
        members : List[int] = [],
        score_ratios: Sequence[float] = None,
        total_nb_members: int = None,
    ):
        self.__key: int = Component.key_incr
        self.num = num
        self.time_step = t
        self.members = members
        self._compute_ratio_members(total_nb_members = total_nb_members)
        self.score_ratios = score_ratios

        Component.key_incr += 1

    def reset_key_incr(self):
        Component.key_incr = 0

    def is_alive(self, ratio: float) -> bool:
        """
        Check if the component is alive at that ratio

        :param ratio: _description_
        :type ratio: float
        :return: True if the component is alive at that ratio
        :rtype: bool
        """
        alive = False
        for (ratio_birth, ratio_death) in self.__score_ratios:
            alive |= ratio > ratio_birth and ratio <= ratio_death
        return alive

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

    def _compute_ratio_members(self, total_nb_members: int) -> None:
        """
        We use this extra auxiliary function on top of the setter
        to use `total_nb_members`.

        :param total_nb_members: Size of the ensemble
        :type total_nb_members: int
        """
        if total_nb_members is None or self.nb_members is None:
            self.__ratio_members = None
        else:
            ratio_members = self.nb_members/total_nb_members
            self.ratio_members = ratio_members

    @property
    def key(self) -> int:
        """
        Number of the component (unique in the entire graph).

        :rtype: int
        """
        return self.__key

    @property
    def num(self) -> int :
        """
        Number of the component (unique at that time step).

        :rtype: int
        """
        return self.__num

    @num.setter
    def num(self, num: int):
        """
        Number of the component (unique at that time step).

        :type num: int
        :raises ValueError: If ``num`` is not > 0
        """
        if num is not None:
            check_int_positive(num, 'num')
            self.__num = int(num)
        else:
            self.__num = None

    @property
    def time_step(self) -> int:
        """
        Time step at which Component exists.

        :rtype: int
        """
        return self.__time_step

    @time_step.setter
    def time_step(self, t: int):
        check_int_positive(t, 'time_step')
        self.__time_step = int(t)

    @property
    def members(self) -> List[int]:
        """
        Ordered list of members indices belonging to Component.

        :rtype: List[int]
        """
        return self.__members

    @members.setter
    def members(self, members: List[int]):
        if members is not None:
            check_all_int_positive(members, 'members')
            self.__members = [int(m) for m in members]
            self.__nb_members = len(members)
        else:
            self.__members = []

    @property
    def ratio_members(self) -> float:
        """
        Ratio of members belonging to Component: ``nb_members`` / N.
        By definition, `ratio_members` is within [0, 1] range.

        :rtype: float
        """
        return self.__ratio_members

    @ratio_members.setter
    def ratio_members(self, ratio_members):
        check_O1_range(ratio_members, 'Ratio members')
        self.__ratio_members = ratio_members

    @property
    def nb_members(self) -> int:
        """
        Number of members belonging to Component.

        :rtype: int
        """
        return self.__nb_members

    @nb_members.setter
    def nb_members(self, nb_members):
        self.__nb_members = int(nb_members)

    @property
    def score_ratios(self) -> Sequence[float]:
        """
        List of sequences (`ratio_birth`, `ratio_death`).

        By convention, `ratio_birth` is worse (in terms of corresponding
        scores) than `ratio_death`. By convention, `ratio_birth` is
        lower than `ratio_death` (in terms of ratio).

        This means that `ratio_birth` <= `ratio_death` even
        when `score_birth` > `score_death`.
        Both `ratio_birth` and`ratio_death` are within $[0,
        1]$ range.

        Score ratios for components are derived from score ratios of
        local steps in the graph $r_{t,s}$. See
        `PersistentGraph.local_steps` for more information on step score
        ratios and step life spans.

        Definitions of score ratios and life spans in graph
        local steps:

        - $r_{t,s}$ = (score_{t,s} - sco / score_bounds_{t]})
        - The "improvement" of assuming $k_t,s$ is defined as
        $r_{t,s} - r_{t,s-1}$
        - The "cost" of assuming $k_t,s$ is defined as
        $r_{t,s+1} - r_{t,s}$
        - By default, the "life span" of the assumption $k_t,s$ is
        defined as its improvement. Note that according to this
        definition of life span, `ratio_scores` refers to the death
        ratio of the step. See `PersistentGraph.local_steps` for more
        information on scores, ratios and life spans.

        For a component `c`, the life span is defined as the
        improvement of ratio between the last assumption before `c` was
        created and the ratio of the best assumption where `c` is still
        alive, that means ratio at which `c` dies.

        :rtype: List[Sequence[float]]
        """
        return self.__score_ratios

    @score_ratios.setter
    def score_ratios(self, score_ratios):
        if score_ratios is None:
            self.__score_ratios = None
            self.__life_span = None
        else:
            life_span = 0
            self.__score_ratios = []
            for ratios in score_ratios:
                if len(ratios) != 2:
                    raise ValueError("score_ratios should be a sequence of 2 elements")

                check_O1_range(ratios[0], 'Ratio birth')
                check_O1_range(ratios[1], 'Ratio death')
                check_is_leq(ratios, '[ratio_birth, ratio_death]')

                self.__score_ratios.append(ratios)

                # LIFE SPAN
                # Note: ratio death must always be >= ratio_birth
                # Note: life_span is with 0-1 range
                life_span += ratios[1] - ratios[0]
            self.__life_span = life_span

    @property
    def life_span(self) -> float:
        """
        Life span of Component: ``ratio_death`` - ``ratio_birth``. This
        holds regardless of the type of score, convention on the
        definition of life span and type of component (vertex or edge).

        See `PersistentGraph.local_steps` for more information on
        scores, ratios and life spans.

        :rtype: float
        """
        return self.__life_span











