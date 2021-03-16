import numpy as np
from typing import List, Sequence, Union, Any, Dict
from bisect import bisect, bisect_right, insort
import time
import pickle
from scipy.spatial.distance import euclidean

from PersistentGraph import Vertex
from PersistentGraph import Edge
from PersistentGraph import _pg_kmeans, _pg_naive
from utils.sorted_lists import (
    insert_no_duplicate, concat_no_duplicate, reverse_bisect_right
)

class PersistentGraph():
    _SCORES_TO_MINIMIZE = [
        'inertia',
        'max_inertia',
        'min_inertia',
        'variance',
        'min_variance',
        'max_variance',
        ]

    _SCORES_TO_MAXIMIZE = []


    def __init__(
        self,
        members: np.ndarray,
        time_axis: np.ndarray = None,
        weights: np.ndarray = None,
        score_is_improving: bool = False,
        precision: int = 13,
        score_type: str = 'inertia',
        zero_type: str = 'uniform',
        model_type: str = 'KMeans',
        k_max : int = None,
        name: str = None,
    ):
        """
        Initialize an empty graph

        :param members: (N, T)-array. Original data, ensemble of time series

        - N: number of members
        - T: length of the time series

        :type members: np.ndarray

        :param time_axis: (T)-array. Time axis, mostly used for plotting,
        defaults to None, in that case, ``time_axis = np.arange(T)``
        :type time_axis: np.ndarray, optional

        :param weights: Only here for parameter compatibility with the naive
        version, defaults to None
        :type weights: np.ndarray, optional

        :param score_is_improving: Is the score improving throughout the
        algorithm steps? (Is, ``score_birth`` 'worse' than ``score_death``),
        defaults to False
        :type score_is_improving: bool, optional

        :param precision: Score precision, defaults to 13
        :type precision: int, optional

        :param score_type: Define how the score is computed,
        defaults to 'inertia'. Scores available:

        - inertia
        - min / max_inertia
        - variance
        - min / max_variance

        :type score_type: str, optional
        :param zero_type: Define the "0" cluster, defaults to 'uniform'
        :type zero_type: str, optional

        :param pre_prune: Are new step close to their previous step
        pre-pruned? Pre-pruned means their corresponding vertices are
        never created and the step never stored, defaults to False.
        FIXME: OUTDATED IMPLEMENTATION
        :type pre_prune: bool, optional

        :param pre_prune_threshold: Threshold determining which steps
        are pre-pruned, defaults to 0.30
        FIXME: OUTDATED IMPLEMENTATION
        :type pre_prune_threshold: float, optional

        :param post_prune: FIXME: NOT IMPLEMENTED YET
        :type post_prune: bool, optional

        :param post_prune_threshold: Threshold determining which steps
        are post-pruned,, defaults to 0.05
        FIXME: NOT IMPLEMENTED YET
        :type post_prune_threshold: float, optional

        :param verbose: Level of verbosity (defaults to False):

        - 0 (False) Nothing
        - 1 (True) Steps overview
        - 2 Steps overview + more details about vertices and edges
        :type verbose: Union[bool,int], optional

        :param name: Name of the graph object, mostly used for saving,
        defaults to None
        :type name: str, optional
        :raises ValueError: [description]
        """


        # --------------------------------------------------------------
        # ---------------------- About the data ------------------------
        # --------------------------------------------------------------

        self._members = np.copy(members)  #Original Data

        # Variable dimension
        shape = members.shape
        if len(shape) < 3:
            self._d = int(1)
        else:
            self._d = shape[0]

        self._N = shape[0]  # Number of members (time series)
        self._T = shape[1]  # Length of the time series

        # Shared x-axis values among the members
        if time_axis is None:
            self._time_axis = np.arange(self.T)
        else:
            self._time_axis = time_axis

        if weights is None:
            self._weights = np.ones_like(time_axis, dtype = float)
        else:
            self._weights = np.array(weights)

        # --------------------------------------------------------------
        # --------- About the graph:   graph's attributes --------------
        # --------------------------------------------------------------

        # True if we should consider only relevant scores
        self._pre_prune = True
        self._pre_prune_threshold = 0
        # True if we should remove vertices with short life span
        self._post_prune = False
        self._post_prune_threshold = 0
        if k_max is None:
            self._k_max = self.N
        else:
            self._k_max = max(int(k_max), 1)
        # Determines how to cluster the members
        self._model_type = model_type
        # True if the score is improving with respect to the algo step
        if model_type == "Naive":
            self._score_is_improving = True
            # To know if we start with N clusters or 1
            self._n_clusters_range = range(1, self.k_max + 1)
        else:
            self._score_is_improving = score_is_improving
            self._n_clusters_range = range(self.k_max, 0,-1)
        # Score type, determines how to measure how good a model is
        self._set_score_type(score_type)
        # Determines how to measure the score of the 0th component
        self._zero_type = zero_type
        # Total number of iteration of the algorithm
        self._nb_steps = 0
        # Local number of iteration of the algorithm
        self._nb_local_steps = np.zeros(self.T, dtype = int)
        # Total number of vertices/edges created at each time step
        self._nb_vertices = np.zeros((self.T), dtype=int)
        self._nb_edges = np.zeros((self.T-1), dtype=int)
        # Nested list (time, nb_vertices/edges) of vertices/edges
        # Here are stored Vertices/Edges themselves, not only their num
        self._vertices = [[] for _ in range(self.T)]
        self._edges = [[] for _ in range(self.T-1)]
        # Nested list (time, nb_local_steps) of dict storing info about
        # The successive steps
        self._local_steps = [[] for _ in range(self.T)]
        # Dict of lists containing step info stored in increasing step order
        self._sorted_steps = {
            'time_steps' : [],
            'local_step_nums' : [],
            'ratio_scores' : [],
            'scores' : [],
            'params' : [],
        }

        # Score precision
        if precision <= 0:
            raise ValueError("precision must be a > 0 int")
        else:
            self._precision = int(precision)

        if name is None:
            name = score_type + zero_type

        # --------------------------------------------------------------
        # --------- About the graph:   algo's helpers ------------------
        # --------------------------------------------------------------

        # To find members' vertices and edges more efficiently
        #
        # Nested list (time, local steps) of arrays of size N
        # members_v_distrib[t][local_step][i] is the vertex num of the
        # ith member at t at the given local step
        self._members_v_distrib = [
            [] for _ in range(self.T)
        ]
        # To find local steps vertices and more efficiently
        #
        # v_at_step[t]['v'][local_step][i] is the vertex num of the
        # ith alive vertex at t at the given local step
        #
        # v_at_step[t]['global_step_nums'][local_step] is the global step
        # num associated with 'local_step' at 't'
        self._v_at_step = [
            {
                'v' : [],
                'global_step_nums' : []
             } for _ in range(self.T)
        ]

        # Same as above EXCEPT that here local steps don't really refer to
        # the algo's local steps since they will be new edges at t whenever
        # there is a new local step at t OR at t+1
        self._e_at_step = [
            {
                'e' : [],
                'global_step_nums' : []
             } for _ in range(self.T)
        ]

        if self._maximize:
            self._best_scores = -np.inf*np.ones(self.T)
            self._worst_scores = np.inf*np.ones(self.T)
            self._zero_scores = None
        else:
            self._best_scores = np.inf*np.ones(self.T)
            self._worst_scores = -np.inf*np.ones(self.T)
            self._zero_scores = None

        self._are_bounds_known = False
        self._norm_bounds = None
        self._verbose = False
        self._quiet = False

    def _set_score_type(self, score_type):
        if self._model_type == "Naive":
            self._maximize = False
            self._score_type = "max_diameter"
        else:
            if score_type in self._SCORES_TO_MAXIMIZE:
                self._maximize = True
            elif score_type in self._SCORES_TO_MINIMIZE:
                self._maximize = False
            else:
                raise ValueError(
                    "Choose an available score_type"
                    + str(self._SCORES_TO_MAXIMIZE + self._SCORES_TO_MINIMIZE)
                    )
            self._score_type = score_type


    def _get_model_parameters(
        self,
        X,
        t = None,
    ):
        if self._model_type == "KMeans":
            model_kw, fit_predict_kw = _pg_kmeans.get_model_parameters(
                self,
                X = X,
            )
        elif self._model_type == "Naive":
            model_kw, fit_predict_kw = _pg_naive.get_model_parameters(
                self,
                X = X,
                t = t,
            )

        return model_kw, fit_predict_kw



    def _clustering_model(
        self,
        X,
        model_kw : Dict = {},
        fit_predict_kw : Dict = {},
        ):
        if self._model_type == 'KMeans':
            (
                clusters,
                clusters_info,
                step_info,
                model_kw,
            ) = _pg_kmeans.clustering_model(
                self,
                X = X,
                model_kw = model_kw,
                fit_predict_kw = fit_predict_kw,
            )
        elif self._model_type == 'Naive':
            (
                clusters,
                clusters_info,
                step_info,
                model_kw,
            ) = _pg_naive.clustering_model(
                self,
                X = X,
                model_kw = model_kw,
                fit_predict_kw = fit_predict_kw,
            )

        return clusters, clusters_info, step_info, model_kw

    def _is_earlier_score(self, score1, score2, or_equal=True):
        return (
            self.better_score(score1, score2, or_equal)
            != self._score_is_improving
            or (score1 == score2 and or_equal)
        )

    def _is_relevant_score(
        self,
        previous_score,
        score,
        or_equal = True,
    ):
        # # Case if it is the first step
        # if previous_score is None:
        #     res = True
        # # General case
        # else:
        curr_is_better = self.better_score(
            score, previous_score, or_equal=or_equal
            )
        res = curr_is_better == self._score_is_improving
            # and self._pre_prune
            # and (
            #     abs(score-previous_score)
            #     / abs(self.worst_score(previous_score, score))
            #     > self._pre_prune_threshold
            #     )
            # or (not self._pre_prune and
            #     self.better_score(score, previous_score))

        return res

    def better_score(self, score1, score2, or_equal=False):
        # None means that the score has not been reached yet
        # So None is better if score is improving
        if score1 is None:
            return self._score_is_improving
        elif score2 is None:
            return not self._score_is_improving
        elif score1 == score2:
            return or_equal
        elif score1 > score2:
            return self._maximize
        elif score1 < score2:
            return not self._maximize
        else:
            print(score1, score2)
            raise ValueError("Better score not determined")


    def argbest(self, score1, score2):
        if self.better_score(score1, score2):
            return 0
        else:
            return 1

    def best_score(self, score1, score2):
        if self.argbest(score1, score2) == 0:
            return score1
        else:
            return score2

    def argworst(self, score1, score2):
        if self.argbest(score1, score2) == 0:
            return 1
        else:
            return 0

    def worst_score(self, score1, score2):
        if self.argworst(score1, score2) == 0:
            return score1
        else:
            return score2

    def _compute_ratio_score(
        self,
        score,
        score_bounds = None,
        ):
        """
        Inspired by the similar method in component
        """
        if score_bounds is None or score is None:
            ratio_score = None
        else:
            # Normalizer so that ratios are within 0-1 range
            norm = euclidean(score_bounds[0], score_bounds[1])

            if score is None:
                ratio_score = 1
            else:
                ratio_score = euclidean(score, score_bounds[0]) / norm

        return ratio_score

    def _add_vertex(
        self,
        info: Dict[str, Any],
        t: int,
        members: List[int],
        scores: Sequence[float],
        local_step: int,
    ):
        """
        Add a vertex to the current graph

        :param info: Info related to the cluster the vertex represents
        :type info: Dict[str, Any]
        :param t: time step at which the vertex should be added
        :type t: int
        :param members: Ordered list of members indices represented by the
        vertex
        :type members: List[int]
        :param scores: (score_birth, score_death)
        :type scores: Sequence[float]
        :param local_step: Local step of the algorithm
        :type local_step: int
        :return: The newly added vertex
        :rtype: Vertex
        """

        # Create the vertex
        num = self._nb_vertices[t]
        v = Vertex(
            info = info,
            t = t,
            num = num,
            members = members,
            scores = scores,
            score_bounds = None,  # Unknown at that point
            total_nb_members = self.N,
        )

        # Update the graph with the new vertex
        self._nb_vertices[t] += 1
        self._vertices[t].append(v)
        self._members_v_distrib[t][local_step][members] = num
        insort(self._v_at_step[t]['v'][local_step], v.num)

        return v

    def _add_edge(
        self,
        v_start: Vertex,
        v_end: Vertex,
        t: int,
        members: List[int],
    ):
        """
        Add an adge to the current graph

        :param v_start: Number (unique at ``t``) of the vertex from which the
        Edge comes
        :type v_start: Vertex
        :param v_end: Number (unique at ``t+1``) of the vertex to which Edge
        goes
        :type v_end: Vertex
        :param t: time step at which the edge should be added
        :type t: int
        :param members: Ordered list of members indices represented by the
        edge
        :type members: List[int]
        :return: The newly added edge
        :rtype: Edge
        """

        # If v_start is dead before v_end is even born
        # Or if v_end is dead before v_start is even born
        if (
            v_start.score_ratios[1] < v_end.score_ratios[0]
            or v_end.score_ratios[1] < v_start.score_ratios[0]
        ):
            if not self._quiet:
                print("v_start scores: ", v_start.score_ratios)
                print("v_end scores: ", v_end.score_ratios)
                print("WARNING: Vertices are not contemporaries")
        # Create the edge
        argbirth = np.argmax([v_start.score_ratios[0], v_end.score_ratios[0]])
        argdeath = np.argmin([v_start.score_ratios[1], v_end.score_ratios[1]])

        # Note that score birth and death might not be consistent but
        # The most important thing is the ratios which must be consistent
        score_birth = [v_start.scores[0], v_end.scores[0]][argbirth]
        score_death = [v_start.scores[1], v_end.scores[1]][argdeath]

        ratio_birth = [v_start.score_ratios[0], v_end.score_ratios[0]][argbirth]
        ratio_death = [v_start.score_ratios[1], v_end.score_ratios[1]][argdeath]
        if (ratio_death < ratio_birth):
            if not self._quiet:
                print(
                    "WARNING: ratio death smaller than ratio birth!",
                    ratio_death, ratio_birth
                )

        e = Edge(
            v_start = v_start,
            v_end = v_end,
            t = t,
            num = self._nb_edges[t],
            members = members,
            scores = [score_birth, score_death],
            score_ratios = [ratio_birth, ratio_death],
            total_nb_members = self.N,
        )

        # Update the graph with the new edge
        self._nb_edges[t] += 1
        self._edges[t].append(e)
        insort(self._e_at_step[t]['e'][-1], e.num)
        return e


    def _kill_vertices(
        self,
        t:int,
        vertices: List[int],
        score_death: float = None,
    ):
        """
        Kill vertices

        :param t: Time step at which the vertices should be killed
        :type t: int
        :param vertices: Vertex or list of vertices to be killed
        :type vertices: List[Vertex]
        :param score_death: (Open interval) best score at which a vertex
        is still alive
        :type score_death: float

        """
        if score_death is not None:
            if not isinstance(vertices, list):
                vertices = [vertices]
            for v in vertices:
                self._vertices[t][v].scores[1] = score_death


    def _keep_alive_edges(
        self,
        t:int,
        edges: List[int],
        ratio: float = None,
    ):
        """
        Keep edges that are not dead yet at that ratio

        Assume that edges are already born (This is the edges's counterpart
        of "kill_vertices" function)
        """
        if ratio is not None:
            if not isinstance(edges, list):
                edges = [edges]
            return [
                e for e in edges if ratio < self._edges[t][e].score_ratios[1]
                ]


    def get_local_step_from_global_step(
        self,
        step,
        t,
    ):
        """
        Find the local step corresponding to the given global step


        ``self.local_steps[t][s]['global_step_num']`` gives indeed the global
        steps corresponding exactly to a given local step. But one local step
        may live during more than 1 global step, if the global steps concern
        other time steps.

        :param step: [description]
        :type step: [type]
        :param t: [description]
        :type t: [type]
        :return: [description]
        :rtype: [type]
        """
        #TODO: Not all cases are implemented yet
        s = bisect_right(
            self._v_at_step[t]['global_step_nums'],
            step,
            hi = self._nb_local_steps[t],
        )
        # We want the local step's global step to be equal or inferior
        # to the step given
        s -= 1
        return s

    def get_e_local_step_from_global_step(
        self,
        step,
        t,
    ):
        """
        Find the local step corresponding to the given global step


        ``self.local_steps[t][s]['global_step_num']`` gives indeed the global
        steps corresponding exactly to a given local step. But one local step
        may live during more than 1 global step, if the global steps concern
        other time steps.

        :param step: [description]
        :type step: [type]
        :param t: [description]
        :type t: [type]
        :return: [description]
        :rtype: [type]
        """
        s = bisect_right(self._e_at_step[t]['global_step_nums'], step)
        # We want the local step's global step to be equal or inferior
        # to the step given
        s -= 1
        return s

    def get_alive_vertices(
        self,
        scores: Union[Sequence[float], float] = None,
        steps: Union[Sequence[int], int] = None,
        t: Union[Sequence[int],int] = None,
        get_only_num: bool = True
    ) -> Union[List[Vertex], List[int]]:
        """
        Extract alive vertices

        If ``t`` is not specified then returns a nested list of
        alive vertices for each time steps

        :param scores: Scores at which vertices should be alive
        :type scores: float, optional
        :param t: Time step from which the vertices should be extracted
        :type t: int, optional
        :param get_only_num: Return the compononent or its num only?
        :type get_only_num: bool, optional
        """
        # -------------------- Initialization --------------------------
        v_alive = []
        if t is None:
            return_nested_list = True
            t_range = range(self.T)
        else:
            if isinstance(t, int):
                return_nested_list = False
                t_range = [t]
            else:
                return_nested_list = True
                t_range = t

        # If Scores and steps are None,
        #  Then return the vertices that are still alive at the end
        if scores is None and steps is None:
            for t in t_range:
                v_alive_t = self._v_at_step[t][-1]
            if not get_only_num:
                v_alive_t = [
                    self._vertices[t][v_num] for v_num in v_alive_t
                ]
            v_alive.append(v_alive_t)
        # Else: scores or steps is specified
        else:
            if steps is None:
                #TODO: not implemented yet:
                #
                # Find the step with a bisect search on the local scores
                # And then proceed normaly as if steps was given
                print("not implemented yet")

            # If a single step was given
            if isinstance(steps, int):
                steps = [steps]

            # -------------------- Main part ---------------------------
            for t in t_range:
                v_alive_t = []
                for s in steps:
                    local_s = self.get_local_step_from_global_step(step=s, t=t)

                    # If the global step occured before the initialization step at
                    # t then return local_s = -1 and there is no alive vertices
                    # to add
                    if local_s != -1:
                        v_alive_t = concat_no_duplicate(
                            v_alive_t,
                            self._v_at_step[t]['v'][local_s],
                            copy = False,
                        )
                    #print("s, t, local_s", s, t, local_s)
                if not get_only_num:
                    v_alive_t = [
                        self._vertices[t][v_num] for v_num in v_alive_t
                    ]
                v_alive.append(v_alive_t)

        if not return_nested_list:
            v_alive = v_alive[0]
        return v_alive

    def get_alive_edges(
        self,
        scores: Union[Sequence[float], float] = None,
        steps: Union[Sequence[int], int] = None,
        t: Union[Sequence[int],int] = None,
        get_only_num: bool = True
    ) -> Union[List[Vertex], List[int]]:
        """
        Extract alive edges

        If ``t`` is not specified then returns a nested list of
        alive edges for each time steps

        If ``scores`` is not specified then it will return edges that
        are still alive at the end of the algorithm

        :param scores: Scores at which edges should be alive
        :type s: float
        :param t: Time step from which the edges should be extracted
        :type t: int
        :param get_only_num: Return the compononent or its num only?
        :type get_only_num: bool, optional
        """
        e_alive = []
        if t is None:
            return_nested_list = True
            t_range = range(self.T-1)
        else:
            if isinstance(t, int):
                return_nested_list = False
                t_range = [t]
            else:
                return_nested_list = True
                t_range = t

        if steps is None:
            #TODO: not implemented yet:
            #
            # Find the step with a bisect search on the local scores
            # And then proceed normaly as if steps was given
            print("not implemented yet")

        # If a single step was given
        if isinstance(steps, int):
            steps = [steps]

        # -------------------- Main part ---------------------------
        for t in t_range:
            e_alive_t = []
            for s in steps:
                local_s = self.get_e_local_step_from_global_step(step=s, t=t)

                # If the global step occured before the initialization step at
                # t then return local_s = -1 and there is no alive edges
                # to add
                if local_s != -1:
                    e_alive_t = concat_no_duplicate(
                        e_alive_t,
                        self._e_at_step[t]['e'][local_s],
                        copy = False,
                    )
            if not get_only_num:
                e_alive_t = [
                    self._edges[t][e_num] for e_num in e_alive_t
                ]
            e_alive.append(e_alive_t)

        if not return_nested_list:
            e_alive = e_alive[0]
        return e_alive

    def _graph_initialization(self):
        """
        Initialize the graph with N components at each time step
        """
        if self._model_type == "KMeans":
            _pg_kmeans.graph_initialization(self)
        elif self._model_type == "Naive":
            _pg_naive.graph_initialization(self)


    def _compute_extremum_scores(self):
        if self._model_type == "KMeans":
            _pg_kmeans.compute_extremum_scores(self)
        elif self._model_type == "Naive":
            _pg_naive.compute_extremum_scores(self)


    def _construct_vertices(self):


        for t in range(self.T):
            if self._verbose:
                print(" ========= ", t, " ========= ")

            X = self._members[:, t].reshape(-1,1)
            # Get clustering model parameters required by the
            # clustering model
            model_kw, fit_predict_kw = self._get_model_parameters(
                    X = X,
                    t = t,
                )

            local_step = 0

            # each 1st local step is already done in 'graph_initialization'
            for n_clusters in self._n_clusters_range[1:]:

                # Update model_kw
                model_kw['n_clusters'] = n_clusters

                try :
                    # Fit & predict using the clustering model
                    (
                        clusters,
                        clusters_info,
                        step_info,
                        model_kw,
                    ) = self._clustering_model(
                        X,
                        model_kw = model_kw,
                        fit_predict_kw = fit_predict_kw,
                    )
                except ValueError as ve:
                    if not self._quiet:
                        print(str(ve))
                    continue
                score = step_info['score']

                # If the score is worse than the 0th component, stop there
                if self.better_score(self._zero_scores[t], score):
                    if self._verbose:
                        print(
                            "Score worse than 0 component: ",
                            self._zero_scores[t]," VS ", score
                        )
                    break

                # Consider this step only if it improves the score
                previous_score = self._local_steps[t][local_step]['score']
                #if self._is_relevant_score(previous_score, score):
                if self._is_relevant_score(
                    previous_score, score, or_equal=False
                ):

                    # -------------- New step ---------------
                    local_step += 1
                    if self._verbose:
                        msg = "n_clusters: " + str(n_clusters)
                        for (key,item) in step_info.items():
                            msg += "  " + key + ":  " + str(item)
                        print(msg)

                    self._local_steps[t].append(
                        {**{'param' : {"n_clusters" : n_clusters}},
                         **step_info
                        })
                    self._members_v_distrib[t].append(
                        np.zeros(self.N, dtype = int)
                    )
                    self._v_at_step[t]['v'].append([])
                    self._v_at_step[t]['global_step_nums'].append(None)
                    self._nb_steps += 1
                    self._nb_local_steps[t] += 1

                    # ------- Update vertices: Kill and Create ---------
                    # For each new component, check if it already exists
                    # Then kill and create vertices accordingly

                    # Alive vertices
                    alive_vertices = self._v_at_step[t]['v'][local_step-1][:]
                    v_to_kill = []
                    nb_v_created = 0
                    for i_cluster in range(n_clusters):

                        to_create = True
                        members = clusters[i_cluster]

                        # IF i_cluster already exists in alive_vertices
                        # THEN 'to_create' is then 'False'
                        # And update 'v_at_step' and 'members_v_distrib'
                        # ELSE, kill the former vertex of each of its members
                        # And create a new vertex
                        for i, v_alive_key in enumerate(alive_vertices):
                            v_alive = self._vertices[t][v_alive_key]
                            if (v_alive.is_equal_to(
                                members = members,
                                time_step = t,
                                v_type = self._model_type
                            )):
                                to_create = False
                                insort(
                                    self._v_at_step[t]['v'][local_step],
                                    v_alive_key
                                )
                                self._members_v_distrib[t][local_step][members] = v_alive_key
                                # No need to check v_alive anymore for the
                                # next cmpt

                                # Update brotherhood size to the smallest one
                                insort(
                                    v_alive.info['brotherhood_size'],
                                    n_clusters
                                )
                                del alive_vertices[i]
                                break
                        if to_create:
                            # For each of its members, find their former vertex
                            # and kill them
                            nb_v_created += 1
                            for m in members:
                                insert_no_duplicate(
                                    v_to_kill,
                                    self._members_v_distrib[t][local_step-1][m]
                                )

                            # --- Creating a new vertex ----
                            # NOTE: score_death is not set yet
                            # it will be set at the step at which v dies
                            v = self._add_vertex(
                                info = clusters_info[i_cluster],
                                t = t,
                                members = members,
                                scores = [score, None],
                                local_step = local_step,
                            )

                    # -----------  Kill Vertices -------------
                    self._kill_vertices(
                        t = t,
                        vertices = v_to_kill,
                        score_death = score,
                    )
                    if self._verbose == 2:
                        print("  ", nb_v_created, ' vertices created\n  ',
                              v_to_kill, ' killed')

                elif self._verbose:
                    print("n_clusters: ", n_clusters,
                          "Score not good enough:", score,
                          "VS", previous_score)

        if self._verbose:
            print(
                "nb steps: ", self._nb_steps,
                "\nnb_local_steps: ", self._nb_local_steps
            )
    def _compute_ratios(self):
        for t, v_t in enumerate(self._vertices):
            # Bounds order depends on score_is_improving
            if self._score_is_improving:
                score_bounds = (self._worst_scores[t], self._best_scores[t])
            else:
                score_bounds = (self._best_scores[t], self._worst_scores[t])

            # Ratios for vertices
            for v in v_t:
                v._compute_ratio_scores(score_bounds = score_bounds)

            # Ratios for local step scores
            for l_step in range(self._nb_local_steps[t]):
                score = self._local_steps[t][l_step]['score']
                ratio_score = self._compute_ratio_score(score, score_bounds)
                self._local_steps[t][l_step]['ratio_score'] = ratio_score


    def _sort_steps(self):

        # ====================== Initialization ==============================
        # Current local step (i.e step_t[i] represents the ith step at t)
        step_t = -1 * np.ones(self.T, dtype=int)

        # Find the ratio of the first algorithm step at each time step
        candidate_ratios = np.array([
            self._local_steps[t][0]["ratio_score"]
            for t in range(self.T)
        ])
        # candidate_time_steps[i] is the time step of candidate_ratios[i]
        candidate_time_steps = list(np.argsort( candidate_ratios ))

        # Now candidate_ratios are sorted in increasing order
        candidate_ratios = list(candidate_ratios[candidate_time_steps])

        i = 0
        while candidate_ratios:

            # ==== Find the candidate score with its associated time step ====
            # if self._score_is_improving:
            #     idx_candidate = -1
            # else:
            #     idx_candidate = 0
            idx_candidate = 0
            t = candidate_time_steps[idx_candidate]

            # Only for the first local step
            if step_t[t] == -1:
                step_t[t] += 1

            if self._verbose:
                print(
                    "Step ", i, " = ",
                    ' t: ', t,
                    'local step_num: ', step_t[t],
                    ' ratio_score: %.4f ' %candidate_ratios[idx_candidate]
                )

            # ==================== Update sorted_steps =======================
            self._sorted_steps['time_steps'].append(t)
            self._sorted_steps['local_step_nums'].append(step_t[t])
            self._sorted_steps['ratio_scores'].append(
                candidate_ratios[idx_candidate]
            )
            self._sorted_steps['scores'].append(
                self._local_steps[t][step_t[t]]["score"]
            )
            self._sorted_steps['params'].append(
                self._local_steps[t][step_t[t]]["param"]
            )

            # ==================== Update local_steps =======================
            self._local_steps[t][step_t[t]]["global_step_num"] = i
            self._v_at_step[t]['global_step_nums'][step_t[t]] = i


            # ======= Update candidates: deletion and insertion ==============

            # 1. Deletion:
            del candidate_ratios[idx_candidate]
            del candidate_time_steps[idx_candidate]

            # 2. Insertion if there are more local steps available:
            step_t[t] += 1
            if step_t[t] < self._nb_local_steps[t]:
                next_ratio = self._local_steps[t][step_t[t]]["ratio_score"]
                idx_insert = bisect(candidate_ratios, next_ratio)
                candidate_ratios.insert(idx_insert, next_ratio)
                candidate_time_steps.insert(idx_insert, t)

            i += 1

        if i != self._nb_steps:
            if not self._quiet:
                print(
                    "WARNING: number of steps sorted: ", i,
                    " But number of steps done: ", self._nb_steps
                )



    def _construct_edges(self):

        last_v_at_t = -1 * np.ones(self.T, dtype = int)
        local_step_nums = -1 * np.ones(self.T, dtype = int)
        for s in range(self.nb_steps):
            # Find the next time step
            t = self._sorted_steps['time_steps'][s]
            ratio =  self._sorted_steps['ratio_scores'][s]
            # Find the next local step at this time step
            local_step_nums[t] = self._sorted_steps['local_step_nums'][s]
            local_s = local_step_nums[t]

            # Find the new vertices (so vertices created at this step)
            new_vertices = [
                self._vertices[t][v] for v in self._v_at_step[t]['v'][local_s]
                if v > last_v_at_t[t]
            ]
            if new_vertices:
                # New vertices are sorted
                last_v_at_t[t] =  new_vertices[-1].num
            else:
                if not self._quiet:
                    print("WARNING NO NEW VERTICES")
                continue

            if self._verbose:
                print(
                    "Step ", s, " = ",
                    ' t: ', t,
                    ' local step_num: ', local_s,
                    ' nb new vertices: ', len(new_vertices)
                )
            # Prepare next edges' creation
            nb_new_edges_from = 0

            if ( (t < self.T - 1) and (local_step_nums[t + 1] != -1)):
                self._e_at_step[t]['e'].append(
                    self._keep_alive_edges(
                        t,
                        self.get_alive_edges(steps=s-1,t=int(t)),
                        ratio,
                        )
                    )
                self._e_at_step[t]['global_step_nums'].append(s)
            nb_new_edges_to = 0

            if ( (t > 0) and (local_step_nums[t - 1] != -1) ):
                self._e_at_step[t - 1]['e'].append(
                    self._keep_alive_edges(
                        t-1,
                        self.get_alive_edges(steps=s-1,t=int(t-1)),
                        ratio,
                        )
                    )
                self._e_at_step[t - 1]['global_step_nums'].append(s)

            for v_new in new_vertices:

                # ======== Construct edges from t-1 to t ===============
                if ( (t > 0) and (local_step_nums[t - 1] != -1) ):
                    step_num_start = local_step_nums[t-1]
                    v_starts = set([
                        self._vertices[t-1][self._members_v_distrib[t-1][step_num_start][m]]
                        for m in v_new.members
                    ])
                    for v_start in v_starts:
                        members = v_new.get_common_members(v_start)
                        e = self._add_edge(
                            v_start = v_start,
                            v_end = v_new,
                            t = t-1,
                            members = members,
                        )
                        if e is not None:
                            nb_new_edges_to += 1

                # ======== Construct edges from t to t+1 ===============
                if ( (t < self.T - 1) and (local_step_nums[t + 1] != -1)):
                    step_num_end = local_step_nums[t+1]
                    v_ends = set([
                        self._vertices[t+1][self._members_v_distrib[t+1][step_num_end][m]]
                        for m in v_new.members
                    ])
                    for v_end in v_ends:
                        nb_new_edges_from += 1
                        members = v_new.get_common_members(v_end)
                        e = self._add_edge(
                            v_start = v_new,
                            v_end = v_end,
                            t = t,
                            members = members,
                        )
                        if e is not None:
                            nb_new_edges_from += 1
            if self._verbose == 2:
                print("nb new edges going FROM t: ", nb_new_edges_from)
                print("nb new edges going TO t: ", nb_new_edges_to)

    def prune(self):
        """
        FIXME: Outdated
        """
        tot_count = 0
        for t in range(self.T):
            count = 0
            for i, v in enumerate(self._vertices[t]):
                if v.life_span < threshold:
                    del self._vertices[t][i]
                    count += 1
            if self._verbose:
                print("t = ", t, " nb vertices pruned = ", count)
            tot_count += count
            self._nb_vertices[t] -= count


    def construct_graph(
        self,
        k_max : int = None,
        pre_prune: bool = False,
        pre_prune_threshold: float = 0.30,
        post_prune: bool = False,
        post_prune_threshold: float = 0.05,
        verbose: Union[bool,int] = False,
        quiet: bool = False,
    ):

        self._verbose = verbose
        self._quiet = quiet
        self._pre_prune = pre_prune
        self._pre_prune_threshold = pre_prune_threshold
        self._post_prune = post_prune
        self._post_prune_threshold = post_prune_threshold

        self._compute_extremum_scores()

        # if self._verbose :
        #     print(" ========= Initialization ========= ")

        self._graph_initialization()

        t_start = time.time()
        if self._verbose:
            print("Construct vertices...")
        self._construct_vertices()
        t_end = time.time()
        if self._verbose:
            print('Vertices constructed in %.2f s' %(t_end - t_start))

        self._compute_ratios()

        if post_prune:
            t_start = time.time()
            if self._verbose:
                print("Prune vertices...")
            self.prune()
            t_end = time.time()
            if self._verbose:
                print('Vertices pruned in %.2f s' %(t_end - t_start))

        t_start = time.time()
        if self._verbose:
            print("Sort steps...")
        self._sort_steps()
        t_end = time.time()
        if self._verbose:
            print('Steps sorted in %.2f s' %(t_end - t_start))

        if self._verbose:
            print("Construct edges...")
        t_start = time.time()
        self._construct_edges()
        t_end = time.time()
        if self._verbose:
            print('Edges constructed in %.2f s' %(t_end - t_start))

    def save(self, filename = None, path=''):
        if filename is None:
            filename = self.name
        with open(path + filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename, path=''):
        with open(path + filename, 'rb') as f:
            self = pickle.load(f)



    @property
    def N(self) -> int :
        """Number of members in the ensemble

        :rtype: int
        """
        return self._N

    @property
    def T(self) -> int :
        """Length of the time series

        :rtype: int
        """
        return self._T

    @property
    def d(self) -> int :
        """Number of variables studied
        :rtype: int
        """
        return self._d

    @property
    def k_max(self):
        return self._k_max


    @property
    def members(self) -> np.ndarray:
        """Original data, ensemble of time series

        :rtype: np.ndarray[float], shape: (N,T,d)
        """
        return np.copy(self._members)

    @property
    def time_axis(self) -> np.ndarray:
        """
        Time axis, mostly used for plotting

        :rtype: np.ndarray[float], shape: T
        """
        return self._time_axis

    @property
    def nb_steps(self) -> int :
        """Total number of iteration on the graph

        :rtype: int
        """
        return self._nb_steps

    @property
    def nb_local_steps(self) -> np.ndarray :
        """Total number of local iteration on the graph

        :rtype: np.ndarray
        """
        return self._nb_local_steps

    @property
    def v_at_step(self):
        return self._v_at_step

    @property
    def e_at_step(self):
        return self._e_at_step


    @property
    def nb_vertices(self) -> np.ndarray:
        """
        Total number of vertices created at each time step (T)

        :rtype: np.ndarray[int], shape: T
        """
        return np.copy(self._nb_vertices)

    @property
    def nb_edges(self) -> np.ndarray:
        """
        Total number of edges created at each time step (T-1)

        ..note::
          ``nb_edges[t]`` are the edges going from vertices at ``t`` to
          vertices at ``t+1``

        :rtype: np.ndarray[int], shape: T-1
        """
        return np.copy(self._nb_edges)


    @property
    def edges(self) -> List[List[Edge]]:
        """
        Nested list of edges of the graph (T-1, nb_edges[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List[List[Edge]]
        """
        return self._edges

    @property
    def vertices(self) -> List[List[Vertex]]:
        """
        Nested list of vertices of the graph (T, nb_vertices[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List[List[Vertex]]
        """
        return self._vertices


    @property
    def local_steps(self) -> List[List[dict]]:
        """
        Nested list (time and steps) of scores

        :rtype: List[List[dict]]
        """
        return self._local_steps

    @property
    def sorted_steps(self) -> Dict[str, List]:
        """
        Sorted scores as used for each step of the algorithm

        :rtype: dict[str, List]
        """
        return self._sorted_steps

    @property
    def n_clusters_range(self):
        """
        [summary]
        """
        return self._n_clusters_range


    @property
    def parameters(self) -> dict:
        """
        Parameters of the graph

        :rtype: dict
        """
        dic = {
            "model_type" : self._model_type,
            "zero_type" : self._zero_type,
            "score_type" : self._score_type,
            "score_is_improving": self._score_is_improving,
        }
        return dic
