import numpy as np
from bisect import bisect, bisect_right, insort
import time
import pickle
import json
from typing import List, Sequence, Tuple, Union, Any, Dict

from . import Vertex
from . import Edge
from ._clustering_model import generate_all_clusters
from ._scores import (
    _set_score_type, _compute_ratio_scores, _compute_score_bounds
)
from .analysis import get_k_life_span, get_relevant_k

from ..utils.sorted_lists import (
    insert_no_duplicate, concat_no_duplicate, has_element
)
from ..utils.d3 import jsonify
from ..utils._clustering import compute_cluster_params


class PersistentGraph():


    def __init__(
        self,
        members: np.ndarray = None,
        time_axis: np.ndarray = None,
        weights: np.ndarray = None,
        precision: int = 13,
        score_type: str = 'max_inertia',
        zero_type: str = 'bounds',
        model_type: str = 'KMeans',
        k_max : int = None,
        name: str = None,
        model_kw: dict = {},
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

        if members is not None:
            self._members = np.copy(members)  #Original Data

            # Variable dimension
            shape = self._members.shape
            if len(shape) < 3:
                self._d = int(1)
                self._members = np.expand_dims(self._members, axis=1)
            else:
                self._d = shape[1]

            self._N = shape[0]   # Number of members (time series)
            self._T = shape[-1]  # Length of the time series

            # Shared x-axis values among the members
            if time_axis is None:
                self._time_axis = np.arange(self.T)
            else:
                self._time_axis = np.copy(time_axis)

            if weights is None:
                self._weights = np.ones((self.d, self.T), dtype = float)
            else:
                self._weights = np.array(weights)
                if len(self._weights.shape) < 2:
                    self._weights = np.expand_dims(self._weights, axis=0)

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
                self._k_max = min(max(int(k_max), 1), self.N)
            # Determines how to cluster the members
            self._model_type = model_type
            # Key-words related to the clustering model
            self._model_kw = {'precompute_centroids' : True}
            self._model_kw.update(model_kw)
            # Ordered number of clusters that will be tried
            self._n_clusters_range = range(self.k_max + 1)
            # Score type, determines how to measure how good a model is
            _set_score_type(self, score_type)
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
            self._life_span = None
            self._life_span_max = None
            self._life_span_min = None
            self._relevant_k = None
            self._max = None
            self._min = None

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
            else:
                self._best_scores = np.inf*np.ones(self.T)
                self._worst_scores = -np.inf*np.ones(self.T)
            self._zero_scores = np.nan*np.ones(self.T)
            self._max_life_span = 0

            self._are_bounds_known = False
            self._norm_bounds = None
            self._verbose = False
            self._quiet = False



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

        # Info specific to this vertex
        # (brotherhood_size computed in construct_vertices)
        if info['brotherhood_size'][-1]:
            info["type"] = self._model_type
        else:
            info["type"] = "uniform"
        X = self._members[members, :, t]
        info_cluster = compute_cluster_params(X)
        info.update(info_cluster)

        # Create the vertex
        num = self._nb_vertices[t]
        v = Vertex(
            info = info,
            t = t,
            num = num,
            members = members,
            scores = scores,
            score_bounds = None,  # Ratio will be computed when v is killed
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
        members: List[int],
    ):
        """
        Add an adge to the current graph

        Return None if the edge is malformed

        :param v_start: Vertex from which the edge comes
        :type v_start: Vertex
        :param v_end: Vertex to which the edge goes
        :type v_end: Vertex
        :param members: Ordered list of members indices represented by the
        edge
        :type members: List[int]
        :return: The newly added edge
        :rtype: Edge
        """
        t = v_start.time_step
        # If v_start is dead before v_end is even born
        # Or if v_end is dead before v_start is even born
        if not self._quiet:
            if (
                v_start.score_ratios[1] < v_end.score_ratios[0]
                or v_end.score_ratios[1] < v_start.score_ratios[0]
            ):
                print("v_start scores: ", v_start.score_ratios)
                print("v_end scores: ", v_end.score_ratios)
                print("WARNING: Vertices are not contemporaries")
                return None
            if not members:
                print("WARNING: No members in edge")
                return None

        # Create the edge
        argbirth = np.argmax([v_start.score_ratios[0], v_end.score_ratios[0]])
        argdeath = np.argmin([v_start.score_ratios[1], v_end.score_ratios[1]])

        # Note that score birth and death might not be consistent but
        # The most important thing is the ratios which must be consistent
        score_birth = [v_start.scores[0], v_end.scores[0]][argbirth]
        score_death = [v_start.scores[1], v_end.scores[1]][argdeath]

        ratio_birth = [v_start.score_ratios[0], v_end.score_ratios[0]][argbirth]
        ratio_death = [v_start.score_ratios[1], v_end.score_ratios[1]][argdeath]
        if not self._quiet:
            if (ratio_death < ratio_birth):
                print(
                    "WARNING: ratio death smaller than ratio birth!",
                    ratio_death, ratio_birth
                )
                return None

        # Compute info (mean, std inf/sup at start and end)
        X_start = self._members[members, :, t]
        info_start = compute_cluster_params(X_start)
        X_end = self._members[members, :, t+1]
        info_end = compute_cluster_params(X_end)

        e = Edge(
            info_start = info_start,
            info_end = info_end,
            v_start = v_start.num,
            v_end = v_end.num,
            t = t,
            num = self._nb_edges[t],
            members = members,
            scores = [score_birth, score_death],
            score_ratios = [ratio_birth, ratio_death],
            total_nb_members = self.N,
        )

        # Add edge number to v_start and v_end
        v_start.add_edge_from(e.num)
        v_end.add_edge_to(e.num)

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

        Update the self._max_life_span if relevant

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
                self._vertices[t][v]._compute_ratio_scores(
                    (self._best_scores[t], self._worst_scores[t])
                )

            if vertices:
                # Get the longest life span
                self._max_life_span = max(self._max_life_span, max(
                    [self._vertices[t][v].life_span for v in vertices]
                ))


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
        s = bisect_right(
            self._v_at_step[t]['global_step_nums'],
            step,
            hi = self._nb_local_steps[t],
        )
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


    def _construct_vertices(self, cluster_data, local_scores):
        for t in range(self.T):
            for step, (clusters, clusters_info) in enumerate(cluster_data[t]):
                n_clusters = len(clusters)
                score = self._local_steps[t][step]['score']

                # ---------------- Preliminary ---------------------
                self._members_v_distrib[t].append(
                    np.zeros(self.N, dtype = int)
                )
                self._v_at_step[t]['v'].append([])
                self._v_at_step[t]['global_step_nums'].append(None)


                # ---------- Update vertices: Kill and Create ----------
                # For each new vertex, check if it already exists
                # Then kill and create vertices accordingly

                # Alive vertices
                if step == 0:
                    alive_vertices = []
                else:
                    alive_vertices = self._v_at_step[t]['v'][step-1][:]
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
                    for i, v_key in enumerate(alive_vertices):
                        v_alive = self._vertices[t][v_key]
                        if (v_alive.is_equal_to(
                            members = members,
                            time_step = t,
                            v_type = self._model_type
                        )):
                            to_create = False
                            insort(
                                self._v_at_step[t]['v'][step],
                                v_key
                            )
                            self._members_v_distrib[t][step][members] = v_key

                            # Update brotherhood size to the smallest one
                            insort(
                                v_alive.info['brotherhood_size'],
                                n_clusters
                            )

                            # No need to check v_alive anymore for the
                            # next cmpt
                            del alive_vertices[i]
                            break

                    if to_create:
                        # For each of its members, find their former vertex
                        # and kill them
                        nb_v_created += 1
                        for m in members:
                            if step > 0:
                                insert_no_duplicate(
                                    v_to_kill,
                                    self._members_v_distrib[t][step-1][m]
                                )

                        # -------------- Create new vertex -------------
                        # NOTE: score_death is not set yet
                        # it will be set at the step at which v dies
                        info = clusters_info[i_cluster]
                        info['brotherhood_size'] = [n_clusters]
                        v = self._add_vertex(
                            info = info,
                            t = t,
                            members = members,
                            scores = [score, None],
                            local_step = step,
                        )

                # --------------------  Kill Vertices ------------------
                self._kill_vertices(
                    t = t,
                    vertices = v_to_kill,
                    score_death = score,
                )
                if self._verbose == 2:
                    print("  ", nb_v_created, ' vertices created\n  ',
                            v_to_kill, ' killed')

        if self._verbose:
            print(
                "nb steps: ", self._nb_steps,
                "\nnb_local_steps: ", self._nb_local_steps
            )

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
            idx_candidate = 0
            t = candidate_time_steps[idx_candidate]

            # Only for the first local step
            if step_t[t] == -1:
                step_t[t] += 1

            if self._verbose:
                print(
                    "Step", i, '  ||  '
                    't: ', t, '  ||  ',
                    'n_clusters: ',
                    self._local_steps[t][step_t[t]]["param"]['n_clusters'],
                    '  ||  ',
                    ' ratio_score: %.4f ' %candidate_ratios[idx_candidate]
                )

            # ==================== Update sorted_steps =======================
            self._sorted_steps['time_steps'].append(int(t))
            self._sorted_steps['local_step_nums'].append(int(step_t[t]))
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
            t = int(self._sorted_steps['time_steps'][s])
            ratio =  self._sorted_steps['ratio_scores'][s]
            # Find the next local step at this time step
            local_step_nums[t] = int(self._sorted_steps['local_step_nums'][s])
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
                            members = members,
                        )
                        if e is not None:
                            nb_new_edges_from += 1

            if self._verbose == 2:
                print("nb new edges going FROM t: ", nb_new_edges_from)
                print("nb new edges going TO t: ", nb_new_edges_to)

    def _compute_statistics(self):
        # Max/min (N, d, t)
        self._max = np.amax(self._members, axis=(1,2))
        self._min = np.amin(self._members, axis=(1,2))
        # max/min life span
        life_spans = []
        for v_t in self._vertices:
            life_spans += [v.life_span for v in v_t]
        self._life_span_max = np.amax(life_spans)
        self._life_span_min = np.amin(life_spans)
        return None

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

        # ================== Cluster all the data ======================
        t_start = time.time()
        if self._verbose:
            print("Clustering data...")
        cluster_data, local_scores = generate_all_clusters(self)
        t_end = time.time()
        if self._verbose:
            print('Data clustered in %.2f s' %(t_end - t_start))

        # ================== Compute score bounds ======================
        _compute_score_bounds(self, local_scores)


        # ================== Construct vertices ========================
        t_start = time.time()
        if self._verbose:
            print("Construct vertices...")
        self._construct_vertices(cluster_data, local_scores)
        t_end = time.time()
        if self._verbose:
            print('Vertices constructed in %.2f s' %(t_end - t_start))

        # ================= Compute ratio scores =======================
        _compute_ratio_scores(self)

        # =================== Sort global steps ========================
        t_start = time.time()
        if self._verbose:
            print("Sort steps...")
        self._sort_steps()
        t_end = time.time()
        if self._verbose:
            print('Steps sorted in %.2f s' %(t_end - t_start))

        # =================== Construct edges ==========================
        if self._verbose:
            print("Construct edges...")
        t_start = time.time()
        self._construct_edges()
        t_end = time.time()
        if self._verbose:
            print('Edges constructed in %.2f s' %(t_end - t_start))

        self._compute_statistics()
        self._life_span = get_k_life_span(self)
        relevant_k = get_relevant_k(self, self._life_span, self.k_max)
        dict_relevant_k = {}
        dict_relevant_k["k"] = [x[0] for x in relevant_k]
        dict_relevant_k["life_span"] = [x[1] for x in relevant_k]
        self._relevant_k = dict_relevant_k


    def get_relevant_components(
        self,
        selected_k: List[int] = None,
        k_max: int = 8,
        fill_holes: bool = True,
    ) -> Tuple[List[Vertex], List[Edge]]:
        """
        Return the most relevant vertices and edges

        :param relevant_k: Nested list of [k_relevant, life_span]
        for each time step, defaults to None
        :type selected_k: List[int], optional
        :param k_max: Max value of k considered, defaults to 8
        :type k_max: int, optional
        :return: Relevant vertices and edges
        :rtype: Tuple[List[Vertex], List[Edge]]
        """
        k_max = min(k_max, self.k_max)
        # For each time step, get the most relevant number of clusters
        if selected_k is None:
            relevant_k = get_relevant_k(self, k_max=k_max)
            selected_k = [k for [k, _] in relevant_k]

        relevant_vertices = [
            [
                v for v in self._vertices[t]
                if has_element(v.info['brotherhood_size'], selected_k[t])
            ] for t in range(self.T)
        ]

        relevant_edges = [
            [
                e for e in self._edges[t]
                if (
                    has_element(
                        self._vertices[e.time_step][e.v_start].info['brotherhood_size'],
                        selected_k[t]
                    ) and has_element(
                        self._vertices[e.time_step + 1][e.v_end].info['brotherhood_size'],
                        selected_k[t+1]
                    ))
            ] for t in range(self.T-1)
        ]

        # Some edges might be non-existant (edge k1 -> k2 does not exist)
        # So we will create edges that won't be attached to the graph but with
        # the necessary information so that they can be visualized
        if fill_holes:
            for t, edges in enumerate(relevant_edges):
                if edges == []:

                    # keep track of edge num for each t
                    e_num = self._nb_edges[t]

                    # Get start relevant vertices
                    v_starts = relevant_vertices[t]
                    v_ends = relevant_vertices[t+1]
                    for v_start in v_starts:
                        v_end_members = [
                            (v, v.get_common_members(v_start)) for v in v_ends
                            if v.get_common_members(v_start) != []
                        ]
                        for (v_end, members) in v_end_members:

                            # Compute info (mean, std inf/sup at start and end)
                            X_start = self._members[members, :, t]
                            info_start = compute_cluster_params(X_start)
                            X_end = self._members[members, :, t+1]
                            info_end = compute_cluster_params(X_end)

                            edges.append(Edge(
                                info_start=info_start,
                                info_end=info_end,
                                v_start = v_start.num,
                                v_end = v_end.num,
                                t = t,
                                num = e_num,
                                members = members,
                                total_nb_members = self.N,
                                score_ratios = [0, 1],
                            ))
                            e_num += 1

        return relevant_vertices, relevant_edges

    def save(
        self,
        filename: str = None,
        path: str = '',
        type:str = 'pg'
    ) -> None:
        """
        Save the graph (either as a class or a JSON file)

        :param filename: filename of the saved graph, defaults to None
        :type filename: str, optional
        :param path: path to the saved graph, defaults to ''
        :type path: str, optional
        :param type: type of the graph, either as a python class (.pg)
        or as a JSON file, defaults to 'pg'
        :type type: str, optional
        """
        if filename is None:
            filename = self.name
        if type == 'json':
            with open(path + filename + '.json', 'w', encoding='utf-8') as f:
                class_dict = jsonify(self)
                json_str = json.dumps(class_dict, f, indent=4)
                f.write(json_str)
        else:
            with open(path + filename + '.pg', 'wb') as f:
                pickle.dump(self, f)

    def load(
        self,
        filename: str = None,
        path: str = '',
    ) -> None:
        """
        Load a graph from a PersistentGraph file (.pg)

        :param filename: filename of the saved graph, defaults to None
        :type filename: str, optional
        :param path: path to the saved graph, defaults to ''
        :type path: str, optional
        """
        raise NotImplementedError("pg.load not working, use pickle instead")
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
    def k_max(self) -> int:
        """
        Max value of k considered

        :rtype: int
        """
        return self._k_max


    @property
    def members(self) -> np.ndarray:
        """Original data, ensemble of time series

        :rtype: np.ndarray[float], shape: (N, d, T)
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
    def max(self) -> np.ndarray:
        """
        Max member values for each variable and each t

        :rtype: np.ndarray[float], shape: (d, T)
        """
        return self._max

    @property
    def min(self) -> np.ndarray:
        """
        Min member values for each variable and each t

        :rtype: np.ndarray[float], shape: (d, T)
        """
        return self._min

    @property
    def life_span(self) -> Dict[int, List[float]]:
        """
        life span for all k and each t

        :rtype: Dict[int, List[float]]
        """
        return self._life_span

    @property
    def life_span_max(self) -> float:
        """
        Max life span

        :rtype: float
        """
        return self._life_span_max

    @property
    def life_span_min(self) -> float:
        """
        Min life span

        :rtype: float
        """
        return self._life_span_min

    @property
    def relevant_k(self) -> Dict[str, List]:
        """
        Dict of lists with 2 keys, "k" and "life_span"

        :rtype:
        """
        return self._relevant_k

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
    def v_at_step(self) -> List[dict]:
        """
        List of vertex num and global step for each t and each local step

        v_at_step[t] is a dict such that:

        - v_at_step[t]['v'][local_step][i] is the vertex num of the
        ith alive vertex at t at the given local step

        - v_at_step[t]['global_step_nums'][local_step] is the global step
        num associated with 'local_step' at 't'

        :rtype: List[dict]
        """
        return self._v_at_step

    @property
    def e_at_step(self) -> List[dict]:
        """
        List of edge num and global step for each t and each pseudo local step

        Local steps don't really refer to the algo's local steps since they
        will be new edges at t whenever there is a new local step at t
        OR at t+1

        :rtype: List[dict]
        """
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
    def n_clusters_range(self) -> Sequence[int]:
        """
        Range of number of clusters studied

        :rtype: Sequence[int]
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
        }
        return dic
