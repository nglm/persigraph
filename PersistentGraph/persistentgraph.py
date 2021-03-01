import numpy as np
from utils.kmeans import kmeans_custom, row_norms
from PersistentGraph.vertex import Vertex
from PersistentGraph.edge import Edge
from typing import List, Sequence, Union, Any, Dict
from utils.sorted_lists import bisect_search, insert_no_duplicate, concat_no_duplicate, reverse_bisect_left
from bisect import bisect, bisect_left, bisect_right, insort
import time
from scipy.spatial.distance import sqeuclidean, cdist
import pickle



class PersistentGraph():
    __SCORES_TO_MINIMIZE = [
        'inertia',
        'max_inertia',
        'min_inertia',
        'variance',
        'min_variance',
        'max_variance',
        ]

    __SCORES_TO_MAXIMIZE = []


    def __init__(
        self,
        members: np.ndarray,
        time_axis: np.ndarray = None,
        weights: np.ndarray = None,
        score_is_improving: bool = False,
        precision: int = 13,
        score_type: str = 'inertia',
        zero_type: str = 'uniform',
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
        FIXME: OUTDATED IMPLEMENTATION
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

        self.__members = np.copy(members)  #Original Data

        # Variable dimension
        shape = members.shape
        if len(shape) < 3:
            self.__d = int(1)
        else:
            self.__d = shape[0]

        self.__N = shape[0]  # Number of members (time series)
        self.__T = shape[1]  # Length of the time series

        # Shared x-axis values among the members
        if time_axis is None:
            self.__time_axis = np.arange(self.T)
        else:
            self.__time_axis = time_axis

        if weights is None:
            self.__weights = np.ones_like(time_axis, dtype = float)
        else:
            self.__weights = np.array(weights)

        # --------------------------------------------------------------
        # --------- About the graph:   graph's attributes --------------
        # --------------------------------------------------------------

        # True if we should consider only relevant scores
        self.__pre_prune = True
        self.__pre_prune_threshold = 0
        # True if we should remove vertices with short life span
        self.__post_prune = False
        self.__post_prune_threshold = 0
        # True if the score is improving with respect to the algo step
        self.__score_is_improving = score_is_improving
        # Score type, determines how to measure how good a model is
        self.__set_score_type(score_type)
        # Determines how to measure the score of the 0th component
        self.__zero_type = zero_type
        # Total number of iteration of the algorithm
        self.__nb_steps = 0
        # Local number of iteration of the algorithm
        self.__nb_local_steps = np.zeros(self.T, dtype = int)
        # Total number of vertices/edges created at each time step
        self.__nb_vertices = np.zeros((self.T), dtype=int)
        self.__nb_edges = np.zeros((self.T-1), dtype=int)
        # Nested list (time, nb_vertices/edges) of vertices/edges
        self.__vertices = [[] for _ in range(self.T)]
        self.__edges = [[] for _ in range(self.T-1)]
        # Nested list (time, nb_local_steps) of dict storing info about
        # The successive steps
        self.__local_steps = [[] for _ in range(self.T)]
        # Dict of lists containing step info stored in increasing step order
        self.__sorted_steps = {
            'time_steps' : [],
            'local_step_nums' : [],
            'scores' : [],
            'params' : [],
        }

        # Score precision
        if precision <= 0:
            raise ValueError("precision must be a > 0 int")
        else:
            self.__precision = int(precision)

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
        self.__members_v_distrib = [
            [] for _ in range(self.T)
        ]
        # To find local steps vertices and more efficiently
        #
        # v_at_step[t]['v'][local_step][i] is the vertex num of the
        # ith alive vertex at t at the given local step
        #
        # v_at_step[t]['global_step_nums'][local_step] is the global step
        # num associated with 'local_step' at 't'
        self.__v_at_step = [
            {
                'v' : [],
                'global_step_nums' : []
             } for _ in range(self.T)
        ]

        # Same as above EXCEPT that here local steps don't really refer to
        # the algo's local steps since they will be new edges at t whenever
        # there is a new local step at t OR at t+1
        self.__e_at_step = [
            {
                'e' : [],
                'global_step_nums' : []
             } for _ in range(self.T)
        ]

        if self.__maximize:
            self.__best_scores = -np.inf*np.ones(self.T)
            self.__worst_scores = np.inf*np.ones(self.T)
            self.__zero_scores = None
        else:
            self.__best_scores = np.inf*np.ones(self.T)
            self.__worst_scores = -np.inf*np.ones(self.T)
            self.__zero_scores = None

        self.__are_bounds_known = False
        self.__norm_bounds = None
        self.__verbose = False

    def __set_score_type(self, score_type):
        if score_type in self.__SCORES_TO_MAXIMIZE:
            self.__maximize = True
        elif score_type in self.__SCORES_TO_MINIMIZE:
            self.__maximize = False
        else:
            raise ValueError(
                "Choose an available score_type"
                + str(self.__SCORES_TO_MAXIMIZE + self.__SCORES_TO_MINIMIZE)
                )
        self.__score_type = score_type



    def __clustering_model(
        self,
        X,
        copy_X,
        model_type = 'KMeans',
        model_kw : Dict = {},
        fit_predict_kw : Dict = {},
        ):
        if model_type == 'KMeans':
            # Default kw values
            max_iter = model_kw.pop('max_iter', 200)
            n_init = model_kw.pop('n_init', 10)
            tol = model_kw.pop('tol', 1e-3)
            n_clusters = model_kw.pop('n_clusters')
            model = kmeans_custom(
                n_clusters = n_clusters,
                max_iter = max_iter,
                tol = tol,
                n_init = n_init,
                copy_x = False,
                **model_kw,
            )
            labels = model.fit_predict(copy_X, **fit_predict_kw)
            if model.n_iter_ == max_iter:
                raise ValueError('Kmeans did not converge')
            clusters_info = []
            clusters = []
            for label_i in range(n_clusters):
                # Members belonging to that clusters
                members = [m for m in range(self.N) if labels[m] == label_i]
                clusters.append(members)
                if members == []:
                    print("No members in cluster")
                    raise ValueError('No members in cluster')
                # Info related to this specific vertex
                clusters_info.append({
                    'type' : 'KMeans',
                    'params' : [
                        float(model.cluster_centers_[label_i]),
                        float(np.std(X[members])),
                        ],
                    'brotherhood_size' : n_clusters
                })
            score = self.__compute_score(
                model = model,
                X = X,
                clusters = clusters
            )
        return score, clusters, clusters_info

    def __is_relevant_score(
        self,
        score,
        previous_score
    ):
        # Case if it is the first step
        if previous_score is None:
            res = True
        # General case
        else:
            res = (
                self.better_score(previous_score, score, or_equal=False)
                # and self.__pre_prune
                # and (
                #     abs(score-previous_score)
                #     / abs(self.worst_score(previous_score, score))
                #     > self.__pre_prune_threshold
                #     )
                # or (not self.__pre_prune and
                #     self.better_score(score, previous_score))
            )
        return res


    def __compute_score(self, model=None, X=None, clusters=None):
        if self.__score_type == 'inertia':
            return np.around(model.inertia_, self.__precision)
        elif self.__score_type == 'max_inertia':
            score = 0
            for i_cluster, members in enumerate(clusters):
                score = max(
                    score,
                    np.sum(cdist(
                        X[members],
                        np.mean(X[members]).reshape(-1, 1) ,
                        metric='sqeuclidean'
                        )
                    ))
                return np.around(score, self.__precision)
        elif self.__score_type == 'min_inertia':
            score = np.inf
            for i_cluster, members in enumerate(clusters):
                score = min(
                    score,
                    np.sum(cdist(
                        X[members],
                        np.mean(X[members]).reshape(-1, 1) ,
                        metric='sqeuclidean'
                        )
                    ))
                return np.around(score, self.__precision)
        elif self.__score_type == 'variance':
            score = 0
            for i_cluster, members in enumerate(clusters):
                score += len(members)/self.N * np.var(X[members])
            return np.around(score, self.__precision)
        elif self.__score_type == 'max_variance':
            score = 0
            for i_cluster, members in enumerate(clusters):
                score = max(np.var(X[members]), score)
            return np.around(score, self.__precision)
        elif self.__score_type == 'min_variance':
            score = np.inf
            for i_cluster, members in enumerate(clusters):
                score = min(np.var(X[members]), score)
            return np.around(score, self.__precision)


    def better_score(self, score1, score2, or_equal=False):
        # None means that the score has not been reached yet
        # So None is better if score is improving
        if score1 is None:
            return self.__score_is_improving
        elif score2 is None:
            return not self.__score_is_improving
        elif score1 == score2:
            return or_equal
        elif score1 > score2:
            return self.__maximize
        elif score1 < score2:
            return not self.__maximize
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



    def __add_vertex(
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
        num = self.__nb_vertices[t]
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
        self.__nb_vertices[t] += 1
        self.__vertices[t].append(v)
        self.__members_v_distrib[t][local_step][members] = num
        insort(self.__v_at_step[t]['v'][local_step], v.num)

        return v

    def __add_edge(
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

        if (
            self.better_score(v_start.scores[1], v_end.scores[0], or_equal=True)
            or self.better_score(v_end.scores[1], v_start.scores[0], or_equal=True)
        ):
            print("v_start scores: ", v_start.scores)
            print("v_end scores: ", v_end.scores)
            print("WANRING: Vertices are not comtemporaries")
        # Create the edge
        argbirth = self.argworst(v_start.scores[0], v_end.scores[0])
        argdeath = self.argbest(v_start.scores[1], v_end.scores[1])
        # This if condition is what solved the problem of ratio birth > 1
        # That happens
        if v_start.scores[1] is None and v_end.scores[1] is None:
            if self.__score_is_improving:
                argdeath = self.argbest(
                    self.__best_scores[t],
                    self.__best_scores[t+1]
                )
            else:
                argdeath = self.argworst(
                    self.__worst_scores[t],
                    self.__worst_scores[t+1]
                )

        score_birth = [v_start.scores[0], v_end.scores[0]][argbirth]
        score_death = [v_start.scores[1], v_end.scores[1]][argdeath]
        if self.better_score(score_death, score_birth):
            print(
                "WARNING: score death better than score birth!",
                score_death, score_birth
            )

        e = Edge(
            v_start = v_start,
            v_end = v_end,
            t = t,
            num = self.__nb_edges[t],
            members = members,
            scores = [score_birth, score_death],
            score_bounds = [
                self.__best_scores[t+argdeath],
                self.__worst_scores[t+argdeath]
                ],
            total_nb_members = self.N,
        )

        # Update the graph with the new edge
        self.__nb_edges[t] += 1
        self.__edges[t].append(e)
        insort(self.__e_at_step[t]['e'][-1], e.num)
        return e


    def __kill_vertices(
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
                self.__vertices[t][v].scores[1] = score_death


    def __keep_alive_edges(
        self,
        t:int,
        edges: List[int],
        score: float = None,
    ):
        """
        Keep edges that are not dead yet at that score

        Assume that edges are already born (This is the edges's counterpart
        of "kill_vertices" function)
        """
        if score is not None:
            if not isinstance(edges, list):
                edges = [edges]
            return [
                e for e in edges
                if self.better_score(score, self.__edges[t][e].scores[1])
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

        if self.__score_is_improving:
            if self.__maximize:
                s = bisect_right(
                    self.__v_at_step[t]['global_step_nums'],
                    step,
                    hi = self.__nb_local_steps[t],
                    )
                # We want the local step's global step to be equal or inferior
                # to the step given
                s -= 1
            else:
                print("Not implemented yet")
        else:
            if self.__maximize:
                print("Not implemented yet")
            else:
                s = bisect_right(
                    self.__v_at_step[t]['global_step_nums'],
                    step,
                    hi = self.__nb_local_steps[t],
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
        #TODO: Not all cases are implemented yet

        if self.__score_is_improving:
            if self.__maximize:
                s = bisect_right(self.__e_at_step[t]['global_step_nums'], step)
                # We want the local step's global step to be equal or inferior
                # to the step given
                s -= 1
            else:
                print("Not implemented yet")
        else:
            if self.__maximize:
                print("Not implemented yet")
            else:
                s = bisect_right(self.__e_at_step[t]['global_step_nums'], step)
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
                v_alive_t = self.__v_at_step[t][-1]
            if not get_only_num:
                v_alive_t = [
                    self.__vertices[t][v_num] for v_num in v_alive_t
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
                            self.__v_at_step[t]['v'][local_s],
                            copy = False,
                        )
                    #print("s, t, local_s", s, t, local_s)
                if not get_only_num:
                    v_alive_t = [
                        self.__vertices[t][v_num] for v_num in v_alive_t
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
                        self.__e_at_step[t]['e'][local_s],
                        copy = False,
                    )
            if not get_only_num:
                e_alive_t = [
                    self.__edges[t][e_num] for e_num in e_alive_t
                ]
            e_alive.append(e_alive_t)

        if not return_nested_list:
            e_alive = e_alive[0]
        return e_alive

    def __graph_initialization(self):
        """
        Initialize the graph with N components at each time step
        """

        if self.__verbose:
            print(" ========= Initialization ========= ")
        for t in range(self.T):

            # Initialization
            self.__members_v_distrib[t].append(
                np.zeros(self.N, dtype = int)
            )
            self.__v_at_step[t]['v'].append([])
            self.__v_at_step[t]['global_step_nums'].append(None)

            # ======= Create one vertex per member and time step =======
            for i in range(self.N):
                info = {
                    'type' : 'KMeans',
                    'params' : [self.__members[i,t], 0.],
                    'brotherhood_size' : self.N
                }
                v = self.__add_vertex(
                    info = info,
                    t = t,
                    members = [i],
                    scores = [0, None],
                    local_step = 0
                )

            # ========== Finalize initialization step ==================

            self.__local_steps[t].append({
                'param' : {"n_clusters" : self.N},
                'score' : 0,
            })

            self.__nb_local_steps[t] += 1
            self.__nb_steps += 1

            if self.__verbose:
                print(" ========= ", t, " ========= ")
                print(
                    "n_clusters: ", 0,
                    "   score: ", 0
                )

    def __compute_extremum_scores(self):
        inertia_scores = ['inertia', 'max_inertia', 'min_inertia']
        variance_scores = ['variance', 'max_variance', 'min_variance']
        if self.__zero_type == 'uniform':
            mins = np.amin(self.__members, axis = 0)
            maxs = np.amax(self.__members, axis = 0)

            if self.__score_type in inertia_scores:
                self.__zero_scores = np.around(
                    self.N / 12 * (mins-maxs)**2,
                    self.__precision
                )
            elif self.__score_type in variance_scores:
                self.__zero_scores = np.around(
                    1 / 12 * (mins-maxs)**2,
                    self.__precision
                )
        elif self.__zero_type == 'data':
            if self.__score_type in inertia_scores:
                self.__zero_scores = np.around(
                    self.N * np.var(self.__members, axis = 0),
                    self.__precision
                )
            elif self.__score_type in variance_scores:
                self.__zero_scores = np.around(
                    np.var(self.__members, axis = 0),
                    self.__precision
                )
        # Compute the score of one component and choose the worst score
        for t in range(self.T):
            model_kw = {'n_clusters' : 1}
            X = self.__members[:,t].reshape(-1,1)
            score, _, _ = self.__clustering_model(
                X,
                X,
                model_type = 'KMeans',
                model_kw = model_kw,
            )
            self.__worst_scores[t] = self.worst_score(
                score,
                self.__zero_scores[t]
            )

        self.__best_scores = np.zeros(self.T)
        self.__norm_bounds = np.abs(self.__best_scores - self.__worst_scores)
        self.__are_bounds_known = True



    def __construct_vertices(self):

        for t in range(self.T):
            if self.__verbose:
                print(" ========= ", t, " ========= ")
            # The same N datapoints X are use for all n_clusters values
            # Furthermore the clustering method might want to copy X
            # Each time it is called and compute pairwise distances
            # We avoid doing that more than once
            # using copy_X and row_norms_X
            X = self.__members[:, t].reshape(-1,1)
            copy_X = np.copy(X)
            row_norms_X = row_norms(copy_X, squared=True)

            local_step = 0
            for n_clusters in range(self.N-1, 0,-1):

                # Fit & predict using the clustering model
                model_kw = {'n_clusters' : n_clusters}
                fit_predict_kw = {"x_squared_norms" : row_norms_X}
                try :
                    score, clusters, clusters_info = self.__clustering_model(
                        X,
                        copy_X,
                        model_type = 'KMeans',
                        model_kw = model_kw,
                        fit_predict_kw = fit_predict_kw,
                    )
                except ValueError:
                    print('Step ignored: one cluster without member')
                    continue

                # If the score is worse than the 0th component, stop there
                if self.better_score(self.__zero_scores[t], score):
                    if self.__verbose:
                        print(
                            "Score worse than 0 component: ",
                            self.__zero_scores[t]," VS ", score
                        )
                    break

                # Consider this step only if it improves the score
                previous_score = self.__local_steps[t][local_step]['score']
                if self.__is_relevant_score(score, previous_score):

                    # -------------- New step ---------------
                    local_step += 1
                    if self.__verbose:
                        print(
                            "n_clusters: ", n_clusters,
                            "   score: ", score
                        )

                    self.__local_steps[t].append({
                        'param' : {"n_clusters" : n_clusters},
                        'score' : score,
                    })
                    self.__members_v_distrib[t].append(
                        np.zeros(self.N, dtype = int)
                    )
                    self.__v_at_step[t]['v'].append([])
                    self.__v_at_step[t]['global_step_nums'].append(None)
                    self.__nb_steps += 1
                    self.__nb_local_steps[t] += 1

                    # ------- Update vertices: Kill and Create ---------
                    # For each new component, check if it already exists
                    # Then kill and create vertices accordingly

                    # Alive vertices
                    alive_vertices = self.__v_at_step[t]['v'][local_step-1][:]
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
                            v_alive = self.__vertices[t][v_alive_key]
                            if (v_alive.is_equal_to(
                                members = members,
                                time_step = t,
                                v_type = 'KMeans'
                            )):
                                to_create = False
                                insort(
                                    self.__v_at_step[t]['v'][local_step],
                                    v_alive_key
                                )
                                self.__members_v_distrib[t][local_step][members] = v_alive_key
                                # No need to check v_alive anymore for the
                                # next cmpt
                                del alive_vertices[i]
                                break
                        if to_create:
                            # For each of its members, find their former vertex
                            # and kill them
                            nb_v_created += 1
                            for m in members:
                                insert_no_duplicate(
                                    v_to_kill,
                                    self.__members_v_distrib[t][local_step-1][m]
                                )

                            # --- Creating a new vertex ----
                            # NOTE: score_death is not set yet
                            # it will be set at the step at which v dies
                            v = self.__add_vertex(
                                info = clusters_info[i_cluster],
                                t = t,
                                members = members,
                                scores = [score, None],
                                local_step = local_step,
                            )

                    # -----------  Kill Vertices -------------
                    self.__kill_vertices(
                        t = t,
                        vertices = v_to_kill,
                        score_death = score,
                    )
                    if self.__verbose == 2:
                        print("  ", nb_v_created, ' vertices created\n  ',
                              v_to_kill, ' killed')

                elif self.__verbose:
                    print("n_clusters: ", n_clusters,
                          "Score not good enough:", score,
                          "VS", previous_score)

        if self.__verbose:
            print(
                "nb steps: ", self.__nb_steps,
                "\nnb_local_steps: ", self.__nb_local_steps
            )
        # -------------------- Compute ratios ----------------------
        for t, v_t in enumerate(self.__vertices):
            for v in v_t:
                v.compute_ratio_scores(
                    score_bounds = (
                        self.__best_scores[t], self.__worst_scores[t]
                    )
                )

    def __sort_steps(self):

        # ====================== Initialization ==============================
        # Current local step (i.e step_t[i] represents the ith step at t)
        step_t = -1 * np.ones(self.T, dtype=int)

        # Find the score of the first algorithm step at each time step
        candidate_scores = np.array([
            self.__local_steps[t][0]["score"]
            for t in range(self.T)
        ])
        # candidate_time_steps[i] is the time step of candidate_scores[i]
        candidate_time_steps = list(np.argsort( candidate_scores ))

        # Now candidate_scores are sorted in increasing order
        candidate_scores = list(candidate_scores[candidate_time_steps])

        i = 0
        while candidate_scores:

            # ==== Find the candidate score with its associated time step ====
            if self.__maximize:
                idx_candidate = -1
            else:
                idx_candidate = 0
            t = candidate_time_steps[idx_candidate]

            # Only for the first local step
            if step_t[t] == -1:
                step_t[t] += 1

            if self.__verbose:
                print(
                    "Step ", i, " = ",
                    ' t: ', t,
                    'local step_num: ', step_t[t],
                    ' score: ', candidate_scores[idx_candidate]
                )

            # ==================== Update sorted_steps =======================
            self.__sorted_steps['time_steps'].append(t)
            self.__sorted_steps['local_step_nums'].append(step_t[t])
            self.__sorted_steps['scores'].append(
                candidate_scores[idx_candidate]
            )
            self.__sorted_steps['params'].append(
                self.__local_steps[t][step_t[t]]["param"]
            )

            # ==================== Update local_steps =======================
            self.__local_steps[t][step_t[t]]["global_step_num"] = i
            self.__v_at_step[t]['global_step_nums'][step_t[t]] = i


            # ======= Update candidates: deletion and insertion ==============

            # 1. Deletion:
            del candidate_scores[idx_candidate]
            del candidate_time_steps[idx_candidate]

            # 2. Insertion if there are more local steps available:
            step_t[t] += 1
            if step_t[t] < self.__nb_local_steps[t]:
                next_score = self.__local_steps[t][step_t[t]]["score"]
                idx_insert = bisect(candidate_scores, next_score)
                candidate_scores.insert(idx_insert, next_score)
                candidate_time_steps.insert(idx_insert, t)

            i += 1

        if i != self.__nb_steps:
            print(
                "WARNING: number of steps sorted: ", i,
                " But number of steps done: ", self.__nb_steps
            )



    def __construct_edges(self):

        last_v_at_t = -1 * np.ones(self.T, dtype = int)
        local_step_nums = -1 * np.ones(self.T, dtype = int)
        for s in range(self.nb_steps):
            # Find the next time step
            t = self.__sorted_steps['time_steps'][s]
            score =  self.__sorted_steps['scores'][s]
            # Find the next local step at this time step
            local_step_nums[t] = self.__sorted_steps['local_step_nums'][s]
            local_s = local_step_nums[t]

            # Find the new vertices (so vertices created at this step)
            new_vertices = [
                self.__vertices[t][v] for v in self.__v_at_step[t]['v'][local_s]
                if v > last_v_at_t[t]
            ]
            if new_vertices:
                # New vertices are sorted
                last_v_at_t[t] =  new_vertices[-1].num
            else:
                continue
                print("WARNING NO NEW VERTICES")

            if self.__verbose:
                print(
                    "Step ", s, " = ",
                    ' t: ', t,
                    ' local step_num: ', local_s,
                    ' nb new vertices: ', len(new_vertices)
                )
            # Prepare next edges' creation
            nb_new_edges_from = 0

            if ( (t < self.T - 1) and (local_step_nums[t + 1] != -1)):
                #self.__e_at_step[t]['e'].append([])
                self.__e_at_step[t]['e'].append(
                    self.__keep_alive_edges(
                        t,
                        self.get_alive_edges(steps=s-1,t=int(t)),
                        score,
                        )
                    )
                self.__e_at_step[t]['global_step_nums'].append(s)
            nb_new_edges_to = 0

            if ( (t > 0) and (local_step_nums[t - 1] != -1) ):
                #self.__e_at_step[t - 1]['e'].append([])
                self.__e_at_step[t - 1]['e'].append(
                    self.__keep_alive_edges(
                        t-1,
                        self.get_alive_edges(steps=s-1,t=int(t-1)),
                        score,
                        )
                    )
                self.__e_at_step[t - 1]['global_step_nums'].append(s)

            for v_new in new_vertices:

                # ======== Construct edges from t-1 to t ===============
                if ( (t > 0) and (local_step_nums[t - 1] != -1) ):
                    #step_num_start = max(local_step_nums[t-1], 0)
                    step_num_start =local_step_nums[t-1]
                    v_starts = set([
                        self.__vertices[t-1][self.__members_v_distrib[t-1][step_num_start][m]]
                        for m in v_new.members
                    ])
                    for v_start in v_starts:
                        members = v_new.get_common_members(v_start)
                        nb_new_edges_to += 1
                        self.__add_edge(
                            v_start = v_start,
                            v_end = v_new,
                            t = t-1,
                            members = members,
                        )

                # ======== Construct edges from t to t+1 ===============
                if ( (t < self.T - 1) and (local_step_nums[t + 1] != -1)):

                    #step_num_end = max(local_step_nums[t + 1], 0)
                    step_num_end = local_step_nums[t+1]
                    v_ends = set([
                        self.__vertices[t+1][self.__members_v_distrib[t+1][step_num_end][m]]
                        for m in v_new.members
                    ])
                    for v_end in v_ends:
                        nb_new_edges_from += 1
                        members = v_new.get_common_members(v_end)
                        self.__add_edge(
                            v_start = v_new,
                            v_end = v_end,
                            t = t,
                            members = members,
                        )
            if self.__verbose == 2:
                print("nb new edges going FROM t: ", nb_new_edges_from)
                print("nb new edges going TO t: ", nb_new_edges_to)

    def prune(self):
        """
        FIXME: Outdated
        """
        tot_count = 0
        for t in range(self.T):
            count = 0
            for i, v in enumerate(self.__vertices[t]):
                if v.life_span < threshold:
                    del self.__vertices[t][i]
                    count += 1
            if self.__verbose:
                print("t = ", t, " nb vertices pruned = ", count)
            tot_count += count
            self.__nb_vertices[t] -= count


    def construct_graph(
        self,
        pre_prune: bool = False,
        pre_prune_threshold: float = 0.30,
        post_prune: bool = False,
        post_prune_threshold: float = 0.05,
        verbose: Union[bool,int] = False,
    ):

        self.__verbose = verbose
        self.__pre_prune = pre_prune
        self.__pre_prune_threshold = pre_prune_threshold
        self.__post_prune = post_prune
        self.__post_prune_threshold = post_prune_threshold

        self.__compute_extremum_scores()

        if self.__verbose :
            print("Graph initialization...")
        self.__graph_initialization()

        t_start = time.time()
        if self.__verbose:
            print("Construct vertices...")
        self.__construct_vertices()
        t_end = time.time()
        if self.__verbose:
            print('Vertices constructed in %.2f s' %(t_end - t_start))

        if post_prune:
            t_start = time.time()
            if self.__verbose:
                print("Prune vertices...")
            self.prune()
            t_end = time.time()
            if self.__verbose:
                print('Vertices pruned in %.2f s' %(t_end - t_start))

        t_start = time.time()
        if self.__verbose:
            print("Sort steps...")
        self.__sort_steps()
        t_end = time.time()
        if self.__verbose:
            print('Steps sorted in %.2f s' %(t_end - t_start))

        if self.__verbose:
            print("Construct edges...")
        t_start = time.time()
        self.__construct_edges()
        t_end = time.time()
        if self.__verbose:
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
        return self.__N

    @property
    def T(self) -> int :
        """Length of the time series

        :rtype: int
        """
        return self.__T

    @property
    def d(self) -> int :
        """Number of variables studied
        :rtype: int
        """
        return self.__d

    @property
    def members(self) -> np.ndarray:
        """Original data, ensemble of time series

        :rtype: np.ndarray[float], shape: (N,T,d)
        """
        return np.copy(self.__members)

    @property
    def time_axis(self) -> np.ndarray:
        """
        Time axis, mostly used for plotting

        :rtype: np.ndarray[float], shape: T
        """
        return self.__time_axis

    @property
    def nb_steps(self) -> int :
        """Total number of iteration on the graph

        :rtype: int
        """
        return self.__nb_steps

    @property
    def nb_local_steps(self) -> np.ndarray :
        """Total number of local iteration on the graph

        :rtype: np.ndarray
        """
        return self.__nb_local_steps

    @property
    def v_at_step(self):
        return self.__v_at_step

    @property
    def e_at_step(self):
        return self.__e_at_step


    @property
    def nb_vertices(self) -> np.ndarray:
        """
        Total number of vertices created at each time step (T)

        :rtype: np.ndarray[int], shape: T
        """
        return np.copy(self.__nb_vertices)

    @property
    def nb_vertices_max(self) -> int:
        """
        Max number of vertices created at each time step

        :rtype: int
        """
        return int(self.N*(self.N+1)/2)

    @property
    def nb_edges(self) -> np.ndarray:
        """
        Total number of edges created at each time step (T-1)

        ..note::
          ``nb_edges[t]`` are the edges going from vertices at ``t`` to
          vertices at ``t+1``

        :rtype: np.ndarray[int], shape: T-1
        """
        return np.copy(self.__nb_edges)


    @property
    def edges(self) -> List[List[Edge]]:
        """
        Nested list of edges of the graph (T-1, nb_edges[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List[List[Edge]]
        """
        return self.__edges

    @property
    def vertices(self) -> List[List[Vertex]]:
        """
        Nested list of vertices of the graph (T, nb_vertices[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List[List[Vertex]]
        """
        return self.__vertices


    @property
    def local_steps(self) -> List[List[dict]]:
        """
        Nested list (time and steps) of scores

        :rtype: List[List[dict]]
        """
        return self.__local_steps

    @property
    def sorted_steps(self) -> Dict[str, List]:
        """
        Sorted scores as used for each step of the algorithm

        :rtype: dict[str, List]
        """
        return self.__sorted_steps

    @property
    def parameters(self) -> dict:
        """
        Parameters of the graph

        :rtype: dict
        """
        dic = {
            "zero_type" : self.__zero_type,
            "score_type" : self.__score_type,
            "score_is_improving": self.__score_is_improving,
            "pre-prune" : self.__pre_prune,
            "pre_prune_threshold" : self.__pre_prune_threshold,
            "post-prune" : self.__post_prune,
            "post_prune_threshold" : self.__post_prune_threshold,
        }
        return dic
