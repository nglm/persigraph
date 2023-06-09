import numpy as np
from bisect import bisect, bisect_right, insort
import time
import pickle
import json
from copy import deepcopy
from bisect import insort

from typing import List, Sequence, Tuple, Union, Any, Dict

from . import Vertex
from . import Edge
from . import Component
from ._set_default_properties import (
    _set_members, _set_zero, _set_model_class, _set_score_type
)
from ._clustering_model import generate_all_clusters, merge_clusters
from ._scores import _compute_ratio_scores, _compute_score_bounds
from ._analysis import k_info, get_relevant_k

from ..utils.sorted_lists import (
    insert_no_duplicate, concat_no_duplicate, has_element, are_equal
)
from ..utils.d3 import jsonify
from ..utils._clustering import compute_cluster_params


class PersistentGraph():

    def __init__(
        self,
        members: np.ndarray = None,
        time_axis: np.ndarray = None,
        time_window: int = 1,
        weights: np.ndarray = None,
        precision: int = 13,
        score_type: str = None,
        squared_radius: bool = False,
        DTW: bool = False,
        zero_type: str = 'bounds',
        model_class = None,
        k_max : int = 5,
        name: str = None,
        model_kw: dict = {},
        fit_predict_kw: dict = {},
        model_class_kw: dict = {},
    ):
        """
        Initialize an empty graph

        :param members: (N, d, T)-array. Original data, ensemble of time series

        - N: number of members
        - d: number of variables (default: 1)
        - T: length of the time series (default: 1)

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
            # After that pg._members is of shape (N, d, T) even if
            # d and T were initially omitted
            _set_members(self, members)
            _set_zero(self, zero_type)

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
            # ---------------- About the clustering method  ----------------
            # --------------------------------------------------------------

            # Max number of cluster considered
            if k_max is None:
                self._k_max = self.N
            else:
                self._k_max = min(max(int(k_max), 1), self.N)

            # Length of the time window
            self._w = min( max(int(time_window), 0), self.T)
            # Determines how to cluster the members
            _set_model_class(
                self, model_class, DTW, model_kw, fit_predict_kw,
                model_class_kw)
            # Ordered number of clusters that will be tried
            self._n_clusters_range = range(self.k_max + 1)
            # Score type, determines how to measure how good a model is
            _set_score_type(self, score_type)
            # Should we use a squared radius when clustering data?
            self._squared_radius = squared_radius

            if name is None:
                self._name = self._model_type + "_" + self._score_type
            else:
                self._name = name

            # --------------------------------------------------------------
            # --------- About the graph:   graph's attributes --------------
            # --------------------------------------------------------------

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
            self._k_info = None
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

            # --------------------------------------------------------------
            # --------- About the graph:   algo's helpers ------------------
            # --------------------------------------------------------------

            # To find members' vertices and edges more efficiently
            #
            # List of dictionary of arrays of length N
            # members_v_distrib[t][k][i] is the vertex num of the
            # ith member at t for the assumption k.
            self._members_v_distrib = [
                {k: np.zeros(self.N) for k in range(1, self._k_max+1)}
                for _ in range(self.T)
            ]

            if self._maximize:
                self._best_scores = -np.inf*np.ones(self.T)
                self._worst_scores = np.inf*np.ones(self.T)
            else:
                self._best_scores = np.inf*np.ones(self.T)
                self._worst_scores = -np.inf*np.ones(self.T)
            self._zero_scores = np.nan*np.ones(self.T)
            self._worst_k = np.zeros(self.T)

            self._are_bounds_known = False
            self._norm_bounds = None
            self._verbose = False
            self._quiet = False

    def _add_vertex(
        self,
        info: Dict[str, Any],
        t: int,
        members: List[int],
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
        :return: The newly added vertex
        :rtype: Vertex
        """

        info["type"] = self._model_type
        score_ratios = []
        for k in info["k"]:
            score_ratios = Component.ratio_union(
                score_ratios, [self._k_info[k]["score_ratios"][t]]
            )

        # Create the vertex
        num = self._nb_vertices[t]
        v = Vertex(
            info = info,
            t = t,
            num = num,
            members = members,
            score_ratios = score_ratios,
            total_nb_members = self.N,
        )

        # Update the graph with the new vertex
        self._nb_vertices[t] += 1
        self._vertices[t].append(v)
        for k in info["k"]:
            self._members_v_distrib[t][k][members] = num

        return v

    def _add_edge(
        self,
        v_start: Vertex,
        v_end: Vertex,
    ):
        """
        Add an edge to the current graph

        Return None if the edge is malformed

        :param v_start: Vertex from which the edge comes
        :type v_start: Vertex
        :param v_end: Vertex to which the edge goes
        :type v_end: Vertex
        :return: The newly added edge
        :rtype: Edge
        """
        t = v_start.time_step

        members = v_start.get_common_members(v_end)
        if not members:
            if not self._quiet:
                print("WARNING: No members in edge")
            return None

        # --------- Compute scores and ratios info --------------
        ratios = Component.ratio_intersection(
            v_start.score_ratios, v_end.score_ratios
        )

        # -------------- Compute edge info --------------

        # Compute info (mean, std inf/sup at start and end)
        # Option 1: Re-compute cluster params naively from original members
        # Note that this is not very consistent with DTW.....
        # X_start = self._members[members, :, t]
        # info_start = compute_cluster_params(X_start)
        # X_end = self._members[members, :, t+1]
        # info_end = compute_cluster_params(X_end)

        # Option 2: Use v.info["X"] that we are now storing!
        info_start = Edge.info(v_start, members)
        info_end = Edge.info(v_end, members)

        e = Edge(
            info_start = info_start,
            info_end = info_end,
            v_start = v_start.num,
            v_end = v_end.num,
            t = t,
            num = self._nb_edges[t],
            members = members,
            score_ratios = ratios,
            total_nb_members = self.N,
        )

        # Add edge number to v_start and v_end
        v_start.add_edge_from(e.num)
        v_end.add_edge_to(e.num)

        # Update the graph with the new edge
        self._nb_edges[t] += 1
        self._edges[t].append(e)
        return e

    def get_local_step_from_global_step(
        self,
        step: int,
        t: int,
        v_step: bool = True,
    ) -> int:
        """
        Find the (pseudo-)local step corresponding to the given global step

        ``self.local_steps[t][s]['global_step_num']`` gives indeed the global
        steps corresponding exactly to a given local step. But one local step
        may live during more than 1 global step, if the global steps concern
        other time steps.

        Local steps for edges are pseudo local steps since there is new edge
        step, if there is a new local step at `t` or `t+1`.

        :param step: global step
        :type step: int
        :param t: time step of interest
        :type t: int
        :param v_step: use v_step (local steps) or e_step (pseudo-local step)
        :type v_step: bool
        :return: local step corresponding to the global step
        :rtype: int
        """
        if v_step:
            comp_at_step = self.v_at_step()
        else:
            comp_at_step = self.e_at_step()
        s = bisect_right(
            comp_at_step[t]['global_step_nums'],
            step,
            hi = self._nb_local_steps[t],
        )
        s -= 1
        return s

    def get_alive_vertices(
        self,
        ratio: float = None,
        t: Union[Sequence[int],int] = None,
        get_only_num: bool = True
    ) -> Union[List[Vertex], List[int]]:
        """
        Extract alive vertices

        If ``t`` is not specified then returns a nested list of
        alive vertices for each time steps

        :param ratio: Ratio at which vertices should be alive
        :type ratio: float, optional
        :param t: Time step from which the vertices should be extracted
        :type t: int, optional
        :param get_only_num: Return the component or its num only?
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

        # -------------------- Using ratio ---------------------------
        for t in t_range:

            v_alive_t = [
                v for v in self._vertices[t] if v.is_alive(ratio)
            ]

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
        ratio: float = None,
        t: Union[Sequence[int],int] = None,
        get_only_num: bool = True
    ) -> Union[List[Vertex], List[int]]:
        """
        Extract alive edges

        If ``t`` is not specified then returns a nested list of
        alive edges for each time steps

        :param ratio: Ratio at which edges should be alive
        :type ratio: float, optional
        :param t: Time step from which the edges should be extracted
        :type t: int, optional
        :param get_only_num: Return the component or its num only?
        :type get_only_num: bool, optional
        """
        # -------------------- Initialization --------------------------
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

        # -------------------- Using ratio ---------------------------
        for t in t_range:

            e_alive_t = [
                e for e in self._edges[t] if e.is_alive(ratio)
            ]

            if not get_only_num:
                e_alive_t = [
                    self._edges[t][e_num] for e_num in e_alive_t
                ]
            e_alive.append(e_alive_t)

        if not return_nested_list:
            e_alive = e_alive[0]
        return e_alive

    def _construct_vertices(self, cluster_data):
        for t in range(self.T):
            # Create all vertices at t
            for cluster_data_t_k in cluster_data[t]:
                    # create all v
                    new_vertices = [
                        self._add_vertex(
                                info=info,
                                t=t,
                                members=cluster,
                            ) for (cluster, info) in cluster_data_t_k
                        ]

    def _sort_steps(self):
        """
        Sort all local steps into global steps based on the ratio score,
        from the smallest to the greatest
        """

        # ====================== Initialization ==============================
        # Current local step (i.e step_t[i] represents the ith step at t)
        step_t = np.zeros(self.T, dtype=int)

        # Find the ratio of the first algorithm step at each time step
        candidate_ratios = np.array([
            self._local_steps[t][0]["ratio_score"]
            for t in range(self.T)
        ])
        # candidate_time_steps[i] is the time step of candidate_ratios[i]
        candidate_time_steps = list(np.argsort( candidate_ratios ))

        # Now candidate_ratios are sorted in increasing order
        candidate_ratios = list(candidate_ratios[candidate_time_steps])

        global_step = 0
        while candidate_ratios:

            # ==== Find the candidate score with its associated time step ====
            idx_candidate = 0
            # it's already an int, but it's somehow a shallow copy without using
            # int()
            t = int(candidate_time_steps[idx_candidate])

            if self._verbose:
                print(
                    "Step", global_step, '  ||  '
                    't: ', t, '  ||  ',
                    'n_clusters: ',
                    self._local_steps[t][step_t[t]]["param"]['k'],
                    '  ||  ',
                    ' ratio_score: %.4f ' %candidate_ratios[idx_candidate]
                )

            # ==================== Update sorted_steps =======================
            self._sorted_steps['time_steps'].append(t)
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
            self._local_steps[t][step_t[t]]["global_step_num"] = global_step
            # self._v_at_step[t]['global_step_nums'][step_t[t]] = global_step

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

            global_step += 1

        if global_step != self._nb_steps:
            if not self._quiet:
                print(
                    "WARNING: number of steps sorted: ", global_step,
                    " But number of steps done: ", self._nb_steps
                )

    def _construct_edges(self):
        for t in range(self.T-1):
            v_starts = self._vertices[t]
            v_ends = self._vertices[t+1]

            for v_start in v_starts:
                nb_new_edges = 0
                # Take all vertices at t and all at t+1 If they are
                # contemporaries and have common members then draw an
                # edge
                v_ends_contemp = [
                    v_end for v_end in v_ends
                    if (Component.contemporaries(v_start, v_end)
                    and Component.have_common_members(v_start, v_end))
                ]
                for v_end in v_ends_contemp:

                    e = self._add_edge(
                        v_start = v_start,
                        v_end = v_end,
                    )
                    if e is not None:
                        nb_new_edges += 1

                if self._verbose == 2 and nb_new_edges == 0:
                    print(
                        "WARNING! \nt: {}, no new edges going from v: {}"
                        %(t, v_start.num)
                    )

    def e_at_step(self) -> List[dict]:
        """
        List of edge num and global step for each t and each pseudo local step

        Pseudo-local steps don't really refer to the algo's local steps since they will be new edges at t whenever there is a new local step at t
        OR at t+1

        :rtype: List[dict]
        """

        # Same as above EXCEPT that here local steps don't really refer to
        # the algo's local steps since they will be new edges at t whenever
        # there is a new local step at t OR at t+1
        e_at_step = [
            {
                'e' : [],
                'global_step_nums' : []
            } for _ in range(self.T)
        ]

        for t in range(self.T-1):
            s = [0, 0]
            ts= [t, t+1]
            nb_steps = [self._nb_local_steps[t], self._nb_local_steps[t+1]]

            # -- Base case: while there are steps in both t and t+1 ----
            while s[0] < nb_steps[0] and s[1] < nb_steps[1]:

                # Take the step that is the next between t and t+1
                steps = [
                    self._local_steps[t][s[0]],
                    self._local_steps[t+1][s[1]]
                ]
                ratios = [steps[0]['ratio_score'], steps[0]['ratio_score']]
                argmin = np.argmin(ratios)

                e_at_step[t]['e'].append([
                    e.num for e in self._edges[ts[argmin]]
                    if e.is_alive(steps[argmin]["ratio_score"])
                ])
                e_at_step[t]['global_step_nums'].append([
                    steps[argmin]["global_step_num"]
                ])

                s[argmin] += 1

            # When at least one local step group has no remaining local steps
            argmin = np.argmin(nb_steps)
            s_start = s[argmin]
            for step in self._local_steps[ts[argmin]][s_start:]:
                e_at_step[t]['e'].append([
                    e.num for e in self._edges[ts[argmin]]
                    if e.is_alive(step["ratio_score"])
                ])
                e_at_step[t]['global_step_nums'].append([
                    step["global_step_num"]
                ])

        return e_at_step

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
        # To find local steps vertices and more efficiently
        #
        # v_at_step[t]['v'][local_step][i] is the vertex num of the
        # ith alive vertex at t at the given local step
        #
        # v_at_step[t]['global_step_nums'][local_step] is the global step
        # num associated with 'local_step' at 't'
        v_at_step = [
            {
                'v' : [],
                'global_step_nums' : []
            } for _ in range(self.T)
        ]

        for t in range(self.T-1):
            # Add v_at_step
            for step in self._local_steps[t]:
                v_at_step[t]['v'].append([
                    v.num for v in self._edges[t]
                    if v.is_alive(step["ratio_score"])
                ])
                v_at_step[t]['global_step_nums'].append([
                    step["global_step_num"]
                ])
        return v_at_step

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
        self._quiet = (quiet or not verbose)
        self._pre_prune = pre_prune
        self._pre_prune_threshold = pre_prune_threshold
        self._post_prune = post_prune
        self._post_prune_threshold = post_prune_threshold

        # ================== Cluster all the data ======================
        t_start = time.time()
        if self._verbose:
            print("Clustering data...")
        cluster_data = generate_all_clusters(self)
        merge_clusters(cluster_data)
        t_end = time.time()
        if self._verbose:
            print('Data clustered in %.2f s' %(t_end - t_start))

        # ================== Compute score bounds ======================
        _compute_score_bounds(self)

        # ================= Compute ratio scores =======================
        _compute_ratio_scores(self)

        # =================== default k values =========================
        self._k_info = k_info(self)
        relevant_k = get_relevant_k(self)
        dict_relevant_k = {}
        dict_relevant_k["k"] = [x[0] for x in relevant_k]
        dict_relevant_k["life_span"] = [x[1] for x in relevant_k]
        self._relevant_k = dict_relevant_k

        # ================== Construct vertices ========================
        t_start = time.time()
        if self._verbose:
            print("Construct vertices...")
        self._construct_vertices(cluster_data)
        t_end = time.time()
        if self._verbose:
            print('Vertices constructed in %.2f s' %(t_end - t_start))

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

        # =================== Compute statistics =======================
        self._compute_statistics()


    def get_relevant_components(
        self,
        selected_k: List[int] = None,
        k_max: int = 8,
        fill_holes: bool = True,
    ) -> Tuple[List[Vertex], List[Edge]]:
        """
        Return a deep copy of most relevant vertices and edges

        Potentially fills holes in edges

        :type selected_k: List[int], optional
        :param k_max: Max value of k considered, defaults to 8
        :type k_max: int, optional
        :return: Relevant vertices and edges
        :rtype: Tuple[List[Vertex], List[Edge]]
        """
        k_max = min(k_max, self.k_max)
        # For each time step, get the most relevant number of clusters
        if selected_k is None:
            relevant_k = get_relevant_k(self)
            selected_k = [k for [k, _] in relevant_k]

        # ------------ Find vertices that represent such a k -----------
        relevant_vertices = [
            [
                deepcopy(v) for v in self._vertices[t]
                if has_element(v.info['k'], selected_k[t])
            ] for t in range(self.T)
        ]

        # ------------- Find edges between 2 relevant k ----------------
        relevant_edges = [
            [
                deepcopy(e) for e in self._edges[t]
                if (
                    has_element(
                        self._vertices[e.time_step][e.v_start].info['k'],
                        selected_k[t]
                    ) and has_element(
                        self._vertices[e.time_step + 1][e.v_end].info['k'],
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

                        # Find common members between v_start and v_end
                        v_end_members = [
                            (v, v.get_common_members(v_start)) for v in v_ends
                            if v.get_common_members(v_start) != []
                        ]
                        for (v_end, members) in v_end_members:

                            # Compute info (mean, std inf/sup at start and end)
                            info_start = Edge.info(v_start, members)
                            info_end = Edge.info(v_end, members)

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

                            # Add edge number to v_start and v_end
                            # Potentially useful for plotting
                            v_start.add_edge_from(e_num)
                            v_end.add_edge_to(e_num)

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
            class_dict = jsonify(self)
            json_str = json.dumps(class_dict, indent=4)
            with open(path + filename + '.json', 'w', encoding='utf-8') as f:
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
    def name(self) -> str :
        """Name of the graph object, used as filename by default when saved

        :rtype: str
        """
        return self._name

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
    def w(self) -> int :
        """Size of the sliding time window
        :rtype: int
        """
        return self._w

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
    def members_zero(self) -> np.ndarray:
        """Data used for the "zero component", ensemble of time series

        :rtype: np.ndarray[float], shape: (N, d, T)
        """
        return np.copy(self._members_zero)

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
    def k_info(self) -> Dict[int, List[float]]:
        """
        Life span and ratios of each assumptions k for all k and t.

        Available keys:

        - "score_ratios"
        - "life_span"

        Example: k_info[3]["life_span"][10] is the life span of k=3 and
        t = 10

        :rtype: Dict[int, List[float]]
        """
        return self._k_info

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
        Dict of lists with 2 keys, "k" and "life_span", both lists
        are of length T and represent respectively the most relevant k
        and its life span at each time step.

        :rtype: Dict[str, List]
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
        Sorted nested list (time and steps) of information on local steps.

        Steps are sorted in increasing order of ratio_scores.

        Available keys of self._local_steps[t][s]:
        - `param`
        - `score`
        - `ratio_score`
        - `global_step_num`

        Let's denote $k_{t,s}$ the assumption on the number of clusters at time
        step `t` and local step `s`; and $r_{t,s}$ its corresponding score
        ratio.

        - The "improvement" of assuming $k_{t,s}$ is defined as
        $r_{t,s} - r_{t,s-1}$
        - The "cost" of assuming $k_{t,s}$ is defined as
        $r_{t,s+1} - r_{t,s}$
        - By default, the "life span" of the assumption $k_{t,s}$ is defined as
        its improvement. Note that according to this definition of life span,
        `ratio_scores` refers to the death ratio of the step. See
        `k_info` for more information on how `ratio_scores` is used
        to compute life spans of steps.

        :rtype: List[List[dict]]
        """
        return self._local_steps

    @property
    def sorted_steps(self) -> Dict[str, List]:
        """
        Sorted steps as used for each step of the algorithm
        available keys (with values being lists of length nb_steps)
        - `time_steps`
        - `local_step_nums`
        - `ratio_scores`
        - `scores`
        - `params`

        Steps are sorted in increasing order of ratio_scores.

        :rtype: dict[str, List]
        """
        return self._sorted_steps

    @property
    def n_clusters_range(self) -> Sequence[int]:
        """
        Ordered range of number of clusters studied: [0, .., k_max]

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
            "squared_radius" : self._squared_radius,
            "DTW" : self._DTW,
            "model_class_kw" : self._model_class_kw,
            "model_kw" : self._model_kw,
            "fit_predict_kw" : self._fit_predict_kw,
        }
        return dic
