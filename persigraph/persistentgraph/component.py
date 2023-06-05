import numpy as np
from typing import List, Sequence

from ..utils.check_variable import (
    check_O1_range, check_int_positive, check_all_int_positive,
    check_is_leq,
)
from ..utils.sorted_lists import has_element, get_common_elements, bisect_search


class Component():
    """
    Base class for Vertex and Edge, components of a PersistentGraph
    """

    key_incr:int = 0

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

        if (
            c1.score_ratios[1] <= c2.score_ratios[0]
            or c2.score_ratios[1] <= c1.score_ratios[0]
        ):
            if verbose:
                print("WARNING: Components are not contemporaries")
                print("c1 scores: ", c1.score_ratios)
                print("c2 scores:   ", c2.score_ratios)
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

    def is_alive(self, ratio: float) -> bool:
        """
        Check if the component is alive at that ratio

        :param ratio: _description_
        :type ratio: float
        :return: True if the component is alive at that ratio
        :rtype: bool
        """
        return (
            ratio > self.__score_ratios[0]
            and ratio <= self.__score_ratios[1]
        )

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
        """
        On scores:
        - By convention, `score_bound = (score_worst, score_best)`
        - By convention, `score_birth` is worse than `score_death`.
        - Note that if a score is better when maximized (respectively
        minimized), `score_birth` is lower (resp higher) than
        `score_death`.
        - Note that, depending on the type of score used, `score_birth`
        and / or `score_death` could be negative.

        On ratios:
        - By convention, `ratio_score_birth` is worse (in terms of
        corresponding scores) than `ratio_score_death`.
        - By convention, `ratio_score_birth` is lower than
        `ratio_score_death` (in terms of ratio).
        - Both `ratio_score_birth` and`ratio_score_death` are within
        $[0, 1]$ range.

        This means that `ratio_score_birth` <= `ratio_score_death` even
        when `score_birth` > `score_death`.

        """
        if score_bounds is None or self.scores is None:
            self.__score_ratios = None
        else:
            # SPECIAL CASE, if all score are equal, favor the case k=1
            if score_bounds[0] == score_bounds[1]:
                ratio_birth = 0
                if self.ratio_members == 1:
                    ratio_death = 1
                else:
                    ratio_death = 0
            else:
                # Normalizer so that ratios are within 0-1 range
                norm = np.abs(score_bounds[0] - score_bounds[1])

                # BIRTH
                # If score_birth is ``None`` or 0 it means that the component is
                # alive since the very beginning
                if self.scores[0] is None:
                    ratio_birth = 0.
                else:
                    ratio_birth = np.abs(self.scores[0]-score_bounds[0]) / norm

                # DEATH
                # If score_death is ``None`` it means that the component is not
                # dead at the end
                if self.scores[1] is None:
                    ratio_death = 1.
                else:
                    ratio_death = np.abs(self.scores[1]-score_bounds[0]) / norm

            self.score_ratios = [ratio_birth, ratio_death]

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
    def scores(self) -> Sequence[float]:
        """
        Sequence (`score_birth`, `score_death`).

        For vertices, `score_birth` is worse than `score_death` by
        convention.

        Note that if a score is better when maximized (respectively
        minimized), `score_birth` is lower (resp higher) than
        `score_death`.

        Note that, depending on the type of score used, `score_birth`
        and / or `score_death` could be negative.

        For edges, `scores` is rather irrelevant as different time steps
        could be mixed.

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
        Sequence (`ratio_score_birth`, `ratio_score_death`).

        By convention, `ratio_score_birth` is worse (in terms of
        corresponding scores) than `ratio_score_death`.
        By convention, `ratio_score_birth` is lower than
        `ratio_score_death` (in terms of ratio).

        This means that `ratio_score_birth` <= `ratio_score_death` even
        when `score_birth` > `score_death`.
        Both `ratio_score_birth` and`ratio_score_death` are within $[0,
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
            check_is_leq(score_ratios, '[ratio_birth, ratio_death]')

            self.__score_ratios = score_ratios

            # LIFE SPAN
            # Note: ratio death must always be >= ratio_birth
            # Note: life_span is with 0-1 range
            life_span = score_ratios[1] - score_ratios[0]
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











