import numpy as np
from sklearn.metrics import pairwise_distances
from vertex import Vertex
from edge import Edge

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '/home/natacha/Documents/Work/python'))

from galib.tools.lists import get_indices_element

class PersistentGraph():
    #TODO: function to plot the graph
    #TODO: function to plot the distribution


    def __init__(
        self,
        members,
        weights=None,
    ):
        """
        Initialize the graph with the mean

        :param members: [description]
        :type members: [type]
        """

        shape = members.shape
        if len(shape) < 3:
            self.__d = int(1)  # number of variable
        else:
            self.__d = shape[0]
        self.__N = shape[-2]           # Number of members (time series)
        self.__T = shape[-1]           # Length of the time series
        self.__members = np.copy(members)   # Initial physical values
        self.__dist_matrix = None           # Distance between each members
        if weights is None:
            weights = np.ones((self.T,1))
        self.__dist_weights = weights       # Dist weights at each time step
        self.__compute_dist_matrix()

        # Total number of iteration required by the algorithm
        self.__nb_steps = int((self.N - 1)*self.T + 1)
        self.__steps = []        # List of (i,j,t) involved in each step
        self.__distances = []   # List of distance between i-j at each step
        self.__vertices = []     # Nested list of vertices (see Vertex class)
        self.__edges = []        # Nested list of edges (see Edge class)

        # T*d arrays, members statistics
        mean = np.mean(self.__members, axis=0)
        std = np.std(self.__members, axis=0)

        # Create one vertex and one edge for each time step
        for t in range(self.T):
            # TODO: add a more relevant reprensentative here...
            v = Vertex(
                representative=0,
                s_born=0,
                num=0,
                value=mean[t],
                std=std[t],
                nb_members = self.N,
            )
            self.__vertices.append([v])
            if t>0:
                e = Edge(
                    v_start=0,
                    v_end=0,
                    nb_members = self.N,
                    s_born=0,
                    num=0
                )
                self.__edges.append([e])

        # Total number of vertices created for each time step
        self.__nb_vertices = np.ones((self.T), dtype=int)
        # Total number of edges created for each time step
        self.__nb_edges = np.ones((self.T-1), dtype=int)
        # Distribution of members among vertices
        # All members are associated with the only vertex created for each t
        self.__M_v = np.zeros((self.__nb_steps,self.T,self.N), dtype=int)
        self.__dist_min = 0
        self.__dist_max = None


    def __compute_dist_matrix(
        self,
    ):
        """
        Compute the pairwise distance matrix for each time step

        .. warning::
          Distance to self is set to 0
        """
        dist = []
        # append is more efficient for list than for np
        for t in range(self.T):
            # if your data has a single feature use array.reshape(-1, 1)
            if self.d == 1:
                dist_t = pairwise_distances(self.__members[:,t].reshape(-1, 1))
            else:
                dist_t = pairwise_distances(self.__members[:,t])
            dist.append(dist_t/self.__dist_weights[t])
        self.__dist_matrix = np.asarray(dist)

    def __sort_dist_matrix(
        self,
    ):
        """
        Gives a vector of indices to sort distance_matrix
        """
        # Add NaN to avoid redundancy
        dist_matrix = np.copy(self.__dist_matrix)
        nb_zeros = 0
        for t in range(self.T):
            for i in range(self.N):
                dist_matrix[t,i,i:] = np.nan
                for j in range(i):
                    # If the distance is equal
                    if dist_matrix[t,i,j] == 0:
                        dist_matrix[t,i,j] = np.nan
                        nb_zeros += 1
        # Sort the matrix (NaN should be at the end)
        #idx = np.argsort(dist_matrix, axis=-1)
        # Source https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
        idx = np.dstack(np.unravel_index(
            np.argsort(dist_matrix.ravel()), (self.T, self.N, self.N)
        )).squeeze()
        # Keep only the first non-NaN elements
        sort_idx = idx[:int((self.T*self.N*(self.N-1)/2) - nb_zeros)]

        (t_min, i_min, j_min) = sort_idx[0]
        self.__dist_min = self.__dist_matrix[t_min, i_min, j_min]

        (t_max, i_max, j_max) = sort_idx[-1]
        self.__dist_max = self.__dist_matrix[t_max, i_max, j_max]

        return(sort_idx)


    def __add_vertex(
        self,
        s: int,
        t:int,
        members=None,
        value:float = None,
        std: float = None,
        representative: int = None,
        nb_members:int = None
    ):
        """
        Add a vertex to the current graph

        If ``members`̀  is not None then ``value`` and ``std``
        will be ignored and computed according to ``members``
        In this case and if ``representative`` is not specified then the first
        element of ``member`` is considered as the representative

        :param s: Current algorithm step, used to set ``Vertex.s_born``
        :type s: int
        :param t: Time step at which the vertex should be added
        :type t: int
        :param members: [description], defaults to None
        :type members: [type], optional
        :param value: [description], defaults to None
        :type value: float, optional
        :param std: [description], defaults to None
        :type std: float, optional
        :param representative: [description], defaults to None
        :type representative: int, optional
        """
        # creating the vertex
        v = Vertex(s_born=s)
        v.num = self.__nb_vertices[t]
        # In this case 'value', 'std' and 'representative' have to be specified
        if members is None:
            v.value = value
            v.std = std
            v.representative = int(representative)
            v.nb_members = nb_members
        # Else: compute attributes according to 'members'
        else:
            if representative is None:
                v.representative = int(members[0])
            else:
                v.representative = int(representative)
            members_values = np.asarray(
                [self.__members[i,t] for i in members]
            )
            v.value = np.mean(members_values, axis=0)
            v.std = np.std(members_values, axis=0)
            v.nb_members = len(members)

        # update the graph with the new vertex
        self.__nb_vertices[t] += 1
        self.__vertices[t].append(v)

    def __add_edge(
        self,
        s:int,
        t:int,
        v_start:int,
        v_end:int,
        nb_members:int,

    ):
        """
        Add an adge to the current graph

        :param s: Current algorithm step, used to set ``Edge.s_born``
        :type s: int
        :param t: Time step at which the edge should be added
        :type t: int
        :param v_start: [description]
        :type v_start: int
        :param v_end: [description]
        :type v_end: int
        :param nb_members: [description]
        :type nb_members: int
        """
        # Initialize edge
        e = Edge(
            v_start=v_start,
            v_end=v_end,
            nb_members = nb_members,
            s_born = s,
            num = self.__nb_edges[t]
        )

        # update the graph with the new edge
        self.__nb_edges[t] += 1
        self.__edges[t].append(e)


    def __kill_vertices(
        self,
        s:int,
        t:int,
        vertices,
        verbose:bool = False,
    ):
        """
        Kill vertices and all their edges

        .. note::
          the date of death is defined as the step 's' at which the vertex is
          unused for the first time

        :param s: Current algorithm step, used to set ``s_death``
        :type s: int
        :param t: Time step at which the vertices should be killed
        :type t: int
        :param vertices: Vertex or list of vertices to be killed
        :type vertices: [type]
        """
        if not isinstance(vertices, list):
            vertices = [vertices]
        for v in vertices:
            self.__vertices[t][v].s_death = s
            #self.set_ratio(self.__vertices[t][v])
            edges_to_v, edges_from_v = self.extract_edges_of_vertex(s,t,v)
            if verbose:
                print("edges to kill: ", edges_to_v, edges_from_v)
            self.__kill_edges(s, t-1, edges_to_v)
            self.__kill_edges(s, t, edges_from_v)

    def __kill_edges(
        self,
        s:int,
        t:int,
        edges,
    ):
        """
        Kill the given edges

        .. note::
          the date of death is defined as the step 's' at which the edge is
          unused for the first time

        :param s: Current algorithm step, used to set ``s_death``
        :type s: int
        :param t: Time step at which the edges should be killed
        :type t: int
        :param edges: Edge or list of edges to be killed
        :type edges: [type]
        """
        if not isinstance(edges, list):
            edges = [edges]
        for e in edges:
            self.__edges[t][e].s_death = s
            #self.set_ratio(self.__edges[t][e])

    def get_alive_vertices(
        self,
        s:int = None,
        t:int = None,
    ):
        """
        Extract alive vertices (their number to be more specific)

        If ``t`` is not specified then returns a nested list of
        alive vertices for each time steps

        :param s: Algorithm step at which vertices should be alive
        :type s: int
        :param t: Time step from which the vertices should be extracted
        :type t: int
        """
        if t is None:
            v_alive = []
            for t in range (self.T):
                v_alive.append(list(set(self.__M_v[s,t])))
        else:
            v_alive = list(set(self.__M_v[s,t][:]))
        return v_alive

    def get_alive_edges(
        self,
        s:int = None,
        t:int = None,
    ):
        """
        Extract alive edges (their number to be more specific)

        If ``t`` is not specified then returns a nested list of
        alive edges for each time steps

        :param s: Algorithm step at which edges should be alive
        :type s: int
        :param t: Time step at which the edges start
        :type t: int
        """
        e_alive = []
        if t is None:
            for t in range (self.T-1):
                e_t = []
                for e in self.__edges[t]:
                    if (e.s_born <= s) and (e.s_death > s or e.s_death == -1):
                        e_t.append(e.num)
                e_alive.append(e_t)
        else:
            for e in self.__edges[t]:
                if (e.s_born <= s) and (e.s_death > s or e.s_death == -1):
                    e_alive.append(e.num)
        return e_alive

    def extract_edges_of_vertex(
        self,
        s:int,
        t:int,
        v:int,
    ):
        """
        Extract all edges to and from a vertex

        :param t: [description]
        :type t: int
        :param v: [description]
        :type v: int
        """
        edges_to_v = []
        edges_from_v = []
        if (t>0 and t<self.T):
            for e in self.__edges[t-1]:
                if (
                    (e.v_end == v)
                    and (e.s_born <= s)
                    and (e.s_death > s or e.s_death == -1)
                ):
                    edges_to_v.append(e.num)
        if t<(self.T-1):
            for e in self.__edges[t]:
                if (
                    (e.v_start == v)
                    and (e.s_born <= s)
                    and (e.s_death > s or e.s_death == -1)
                ):
                    edges_from_v.append(e.num)
        return(edges_to_v, edges_from_v)


    def extract_representatives(
        self,
        s:int = None,
        t:int = None,
    ):
        """
        Extract representatives alive at the time step t

        If ``t`` is not specified then returns a nested list of
        representatives for each time steps

        ``s`` must be > 0
        """
        v_alive = self.get_alive_vertices(s,t)
        if t is None:
            rep_alive = []
            for t in range(self.T):
                rep_alive.append(
                    [self.__vertices[t][v].representative for v in v_alive[t]]
                )
        else:
            rep_alive = [self.__vertices[t][v].representative for v in v_alive]
        return rep_alive


    def __update_representatives(
        self,
        s:int,
        t:int,
        i:int,
        j:int,
    ):
        # Break the vertex v into 2 vertices represented by i and j
        v_to_break = self.__M_v[s, t, i]

        representatives = self.extract_representatives(s, t)
        representatives.remove(self.__vertices[t][v_to_break].representative)
        representatives += [i,j]
        return representatives


    def __associate_with_representatives(
        self,
        t:int,
        representatives,
    ):
        """
        For each member, find the corresponding representative
        """
        # extract distance to representatives
        dist = []
        for rep in representatives:
            dist.append(self.__dist_matrix[t,rep])

        dist = np.asarray(dist)     # (nb_rep, N) array
        # for each member, find the representative that is the closest
        idx = np.nanargmin(dist, axis=0)
        return [representatives[i] for i in idx]

    def __update_vertices(
        self,
        s: int,
        t: int,
        representatives,
        verbose: bool = False,
    ):

        # Re-distribute members among updated representatives
        new_rep_distrib = self.__associate_with_representatives(
            t,
            representatives,
        )

        # Previous members' distrib  among vertices
        prev_v_distrib = self.__M_v[s-1, t]
        v_to_kill = []      # List of vertices that must be killed
        rep_visited = []    # List of visited representatives
        # v_to_create is a nested list.
        # each 1st sublists' elt is the representative of a vertex to create.
        # Other elts are the members associated with it
        v_to_create = []
        to_create = []    # boolean specifying potential vertex creation

        # Check which members have changed their representative
        for member in range(self.N):

            # Previous vertex to which 'member' was associated
            v_prev = prev_v_distrib[member]
            # Get representative of this vertex
            rep_prev = self.__vertices[t][v_prev].representative
            # Get the new representative of this member
            rep_new = new_rep_distrib[member]

            # Check if 'rep_new' has already been visited
            [idx] = get_indices_element(
                my_list=rep_visited,
                my_element=rep_new,
                if_none=[-1]
            )

            # If rep_new has never been visited and should be added
            if idx == -1:
                # Note: idx is then the 'right' index of appended elts
                rep_visited.append(rep_new)

                if member == rep_new:
                    v_to_create.append([rep_new])
                else:
                    v_to_create.append([rep_new, member])
                # So far there is no reason to recreate this vertex
                to_create.append(False)
            # Else idx is then the index of the corresponding vertex
            elif rep_new != member:
                v_to_create[idx].append(member)

            # If its representative has changed then its previous vertex
            # must be killed
            if rep_new != rep_prev:

                v_to_kill.append(v_prev)

                # v associated to rep_new must be (re-)created
                to_create[idx] = True

                # v associated to rep_prev must be (re-)created
                [idx_rep_prev] = get_indices_element(
                    my_list=rep_visited,
                    my_element=rep_prev,
                    if_none=[-1]
                )
                if idx_rep_prev == -1:
                    rep_visited.append(rep_prev)
                    v_to_create.append([rep_prev])
                    to_create.append(True)
                # Else idx is then the index of the corresponding vertex
                else:
                    to_create[idx_rep_prev] = True


        # Remove multiple occurences of the same vertex key
        v_to_kill = list(set(v_to_kill))
        if verbose:
            print("Vertices killed: ", v_to_kill)
        # Kill the vertices
        self.__kill_vertices(s, t, v_to_kill, verbose=verbose)

        # Process new vertices
        for i, members in enumerate(v_to_create):
            if to_create[i]:
                # Add vertex to the graph
                self.__add_vertex(s,t, members=members)
                # Update M_v with the new vertex
                for m in members:
                    self.__M_v[s,t,m] = self.__nb_vertices[t]-1

    def __update_edges(
        self,
        s: int,
        t: int,
        new_vertices,
        verbose: bool = False,
    ):
        if not isinstance(new_vertices, list):
            new_vertices = [new_vertices]
        for v in new_vertices:
            # Find all the members in v
            members_in_v = get_indices_element(
                self.__M_v[s,t],
                v,
                all_indices = True,
                if_none = [-1],
            )
            v_ante_visited = []  # list of vertices going to v
            nb_members_ante = [] # number of members in each v_ante visited
            v_succ_visited = []  # list of vertices coming from v
            nb_members_succ = [] # number of members in each v_succ visited

            for m in members_in_v:
                if (t>0):
                    v_ante = self.__M_v[s, t-1,m] # Vertex of 'm' at t-1

                    # Check if v_ante has already been visited
                    [v_ante_idx] = get_indices_element(
                        v_ante_visited,
                        v_ante,
                        all_indices = False,
                        if_none = [-1],
                    )
                    # If it hasn't been visited then append 'v_ante'
                    if (v_ante_idx == -1):
                        v_ante_visited.append(v_ante)
                        nb_members_ante.append(1)
                    # Else increment 'nb_members_ante'
                    else:
                        nb_members_ante[v_ante_idx] += 1
                else:
                    v_ante_visited = []
                # Same with v_succ
                if t<(self.T-1):
                    v_succ = self.__M_v[s, t+1,m] # Vertex of 'm' at t+1
                    [v_succ_idx] = get_indices_element(
                        v_succ_visited,
                        v_succ,
                        all_indices = False,
                        if_none = [-1],
                    )
                    if (v_succ_idx == -1):
                        v_succ_visited.append(v_succ)
                        nb_members_succ.append(1)
                    else:
                        nb_members_succ[v_succ_idx] += 1
                else:
                    v_succ_visited = []

            for i, v_start in enumerate(v_ante_visited):
                self.__add_edge(
                    s = s,
                    t = t-1,
                    v_start = v_start,
                    v_end = v,
                    nb_members = nb_members_ante[i]
                )
            for i, v_end in enumerate(v_succ_visited):
                self.__add_edge(
                    s = s,
                    t = t,
                    v_start = v,
                    v_end = v_end,
                    nb_members = nb_members_succ[i]
                )


    def __compute_ratios(self):
        for t in range(self.T):
            for v in self.__vertices[t]:
                s_born = v.s_born
                s_death = v.s_death
                v.ratio_life = (
                    (self.__distances[s_born] - self.__distances[s_death])
                    / self.__distances[0]
                )
                v.ratio_members = v.nb_members/self.N
            if t<(self.T - 1):
                for e in self.__edges[t]:
                    s_born = e.s_born
                    s_death = e.s_death
                    e.ratio_life = (
                        (self.__distances[s_born] - self.__distances[s_death])
                        / self.__distances[0]
                    )
                    e.ratio_members = e.nb_members/self.N


    def construct_graph(
        self,
        verbose=False,
    ):
        # At s=0 the graph is initialized with one vertex per time step
        s=0

        # reverse argsort of the pairwise distance matrix
        sort_idx = self.__sort_dist_matrix()[::-1]

        # Take the 2 farthest members and the corresponding time step
        for (t_s, i_s, j_s) in sort_idx:

            # Iterate algo only if i_s and j_s are in the same vertex
            if (self.__M_v[s, t_s, i_s] == self.__M_v[s, t_s, j_s]):
                self.__steps.append((t_s, i_s, j_s))
                if verbose:
                    print(
                        "==== Step ", str(s+1), "====",
                        "(t, i, j) = ", (t_s, j_s, i_s),
                        "distance i-j: ", self.__dist_matrix[t_s,i_s,j_s]
                    )
                self.__distances.append(self.__dist_matrix[t_s,i_s,j_s])
                s += 1
                # Current distribution (to be updated after each vertex added)
                self.__M_v[s] = np.copy(self.__M_v[s-1])

                representatives = self.__update_representatives(s,t_s,i_s,j_s)
                if verbose:
                    print('new representatives: ', representatives)

                prev_nb_vertices = self.__nb_vertices[t_s]

                self.__update_vertices(
                    s=s,
                    t=t_s,
                    representatives=representatives,
                    verbose=verbose,
                )


                new_vertices = list(
                    range(prev_nb_vertices, self.__nb_vertices[t_s])
                )
                if verbose:
                    print("New vertices created: ", new_vertices)

                if (t_s<self.T-1):
                    prev_nb_edges = self.__nb_edges[t_s]

                self.__update_edges(
                    s=s,
                    t=t_s,
                    new_vertices=new_vertices,
                    verbose=verbose,
                )

                if verbose and (t_s<self.T-1):
                    print("New edges created: ",
                          list(range(prev_nb_edges, self.__nb_edges[t_s])))

        # update the total number of steps
        self.__nb_steps = s+1
        self.__M_v = self.__M_v[:s+1]

        # Compute the ratios for each member and each vertex
        self.__distances.append(0.)
        self.__compute_ratios()

    @property
    def N(self):
        """Number of members in the ensemble

        :rtype: int
        """
        return self.__N

    @property
    def T(self):
        """Length of the time series

        :rtype: int
        """
        return self.__T

    @property
    def d(self):
        """Dimension of the variable studied
        :rtype: int
        """
        return self.__d

    @property
    def nb_points(self):
        """Total number of points (N*T)

        :rtype: int
        """
        return self.__N*self.__T

    @property
    def nb_steps(self):
        """Total number of iteration on the graph

        :rtype: int
        """
        return self.__nb_steps

    @property
    def steps(self):
        """ (t,i,j) of each iterations


        :rtype: Tuple(int,int,int)
        """
        return self.__steps

    @property
    def distances(self):
        """ Distance between i-j at t for each iterations

        .. note::
          distances[-1] is set to ``0.``
          (associated to the graph representing members)

        :rtype: float
        """
        return self.__distances


    @property
    def members(self):
        """Original data, ensemble of time series

        :rtype: np.ndarray((N,T,d))
        """
        return np.copy(self.__members)

    @property
    def edges(self):
        """
        Nested list of edges of the graph (time, nb_edges[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List
        """
        return self.__edges

    @property
    def vertices(self):
        """
        Nested list of vertices of the graph (t, nb_vertices[t])

        .. note::
          This includes dead and alive vertices

        :rtype: List
        """
        return self.__vertices

    @property
    def M_v(self):
        """
        Distribution of members among vertices (nb_steps, T, N)

        :rtype: np.ndarray((nb_steps, T, N))
        """
        return np.copy(self.__M_v)

    @property
    def nb_vertices(self):
        """
        Total number of vertices created at each time step (T)

        :rtype: np.ndarray((T))
        """
        return np.copy(self.__nb_vertices)

    @property
    def nb_vertices_max(self):
        """
        Max number of vertices created at each time step

        :rtype: int
        """
        return int(self.N*(self.N+1)/2)

    @property
    def nb_edges(self):
        """
        Total number of edges created at each time step (T-1)

        ..note::
          ``nb_edges[t]`` are the edges going from vertices at ``t-1`` to
          vertices at ``t``

        :rtype: np.ndarray((T-1))
        """
        return np.copy(self.__nb_edges)

    @property
    def distance_matrix(self):
        """
        Pairwise distance matrix for each time step (T,N,N)

        .. warning::
          Distance to self is set to NaN

        :rtype: np.ndarray((T,N,N))
        """
        return np.copy(self.__dist_matrix)