import numpy as np
from sklearn.metrics import pairwise_distances
from vertex import Vertex
from galib.tools.lists import get_indices_element

class PersistentGraph():


    def __init__(
        self,
        members,
    ):
        # Initialized attributes
        shape = members.shape
        if len(shape) < 3:
            self.__d = 1
        else:
            self.__d = shape(0)
        self.__members = np.copy(members)   # Initial physical values
        self.__dist_matrix = None           # Distance between each members
        self.__compute_dist_matrix()
        self.__N: int = shape(-2)           # Number of members (time series)
        self.__T: int = shape(-1)           # Length of the time series
        self.__s:int = 0                    # current graph iteration
        # Total number of iteration required by the algorithm
        self.__nb_steps = (self.__N - 1)*self.__T + 1

        # Declared attributes
        self.__vertices = None     # Nested list of vertices (see Vertex class)
        self.__edges = None        # Nested list of edges (see Edge class)
        #self.__v_alive = None      # Nested list of currently alive vertices
        #self.__e_alive = None      # Nested list of currently alive edges
        # Total number of vertices created for each time step
        self.__nb_vertices = np.zeros((self.__T))
        # Distribution of members among vertices
        self.__M_v = np.zeros((self.__nb_steps,self.__T,self.__N))


    def __compute_dist_matrix(
        self,
    ):
        """
        Compute the pairwise distance matrix for each time step

        .. warning::
          Distance to self is set to NaN
        """
        dist = []
        # append is more efficient for list than for np
        for t in range(self.__T):
            dist_t = pairwise_distances(self.__members[t])
            np.fill_diagonal(dist_t, np.nan)
            dist.append(dist_t)
        self.__dist_matrix = np.asarray(dist)

    def __sort_dist_matrix(
        self,
    ):
        """
        Gives a vector of indices to sort distance_matrix
        """
        # Add NaN to avoid redundancy
        dist_matrix = np.copy(self.__dist_matrix)
        for i in range(self.__N):
            for j in range(i, self.__N):
                dist_matrix[i,j] = np.nan
        # Sort the matrix (NaN should be at the end)
        idx = np.argsort(dist_matrix, axis=None)
        # Keep only the first non-NaN elements
        sort_idx = idx[:self.__T*self.__N*(self.__N-1)/2]
        return(sort_idx)

    # def __get_members_from_representative(
    #     self,
    #     rep,
    #     rep_distribution,
    # ):

    def __add_vertex(
        self,
        value:float = None,
        std: float = None,
        t: int = None,
        representative: int = None,
    ):
        """
        Add a vertex to the current graph

        :param value: [description], defaults to None
        :type value: float, optional
        :param std: [description], defaults to None
        :type std: float, optional
        :param t: [description], defaults to None
        :type t: int, optional
        :param representative: [description], defaults to None
        :type representative: int, optional
        """
        # creating the vertex
        v = Vertex()
        v._Vertex__value = value
        v._Vertex__std = std
        v._Vertex__num = nb_vertices[t]
        v._Vertex__representative = int(representative)
        v.s_born = self.__s    #current step in the graph

        # update the graph with the new vertex
        self.__nb_vertices[t] += 1
        self.__vertices[t].append(v)

    def __kill_vertices(
        self,
        vertices
    ):
        if isinstance(vertices, list)
        v.s_death = self.__s

    def extract_alive_vertices(
        self,
        t:int = None,
        s:int = None,
    ):
        """
        Extract alive vertices (their number to be more specific)

        If ``t`` is not specified then returns a nested list of
        alive vertices for each time steps

        ``s`` must be > 0
        """
        if s is None:
            s = self.__s
        if t is None:
            v_alive = []
            for t in range (self.__T):
                v_alive.append(list(set(self.__M_v[s,t])))
        else:
            v_alive = list(set(self.__M_v[s,t]))
        return v_alive

    def extract_representatives(
        self,
        t:int = None,
        s:int = None,
    ):
        """
        Extract representatives alive at the time step t

        If ``t`` is not specified then returns a nested list of
        representatives for each time steps

        ``s`` must be > 0
        """
        if s is None:
            s = self.__s
        v_alive = self.extract_alive_vertices(t,s)
        if t is None:
            rep_alive = []
            for t in range(self.__T):
                rep_alive.append([v.representative for v in v_alive[t]])
        else:
            rep_alive = [v.representative for v in v_alive]
        return rep_alive

    def __associate_to_representatives(
        self,
        representatives,
        t,
    ):
        """
        For each member, find the corresponding representative

        s must be > 0
        """
        # extract distance to representatives
        dist = []
        for rep in representatives:
            dist.append(self.__dist_matrix[t,rep])

        dist = np.asarray(dist)     # (nb_rep, N) array
        # for each member, find the representative that is the closest
        # TODO: this does't take into account the distance to self and
        idx = np.nanargmin(dist, axis=0)
        return [representatives[i] for i in idx]

    def initialization(
        self,
    ):
        """
        Initialize the graph with the mean
        """

        # T*d arrays, members statistics
        mean = np.mean(self.__members, axis=0)
        std = np.std(self.__members, axis=0)

        # Create nodes
        for t in range(self.__T):
            v = Vertex()
            v._Vertex__value = mean[t]
            v._Vertex__std = std[t]
            self.__vertices[t].append(v)
            # TODO: add a reprensentative here...

        # There is exactly one vertex for each time step
        self.__nb_vertices = np.ones((self.__T))
        # All members are associated with the only vertex created for each t
        self.__M_v[0] = np.ones((self.__T, self.__N))[:,:]
        # TODO: add the edges here...


    def construct_graph(
        self,
    ):
        # Initialize the graph with the mean
        self.initialization()

        # argsort of the pairwise distance matrix
        sort_idx = self.__sort_dist_matrix()
        for (t_s, i_s, j_s) in sort_idx:
            s = self.__s

            # Are i_s and j_s in the same vertex?
            if (self.__M_v[s, t_s, i_s] == self.__M_v[s, t_s, i_s]):

                # Break the vertex k into 2 vertices i_s and j_s
                k = self.__M_v[s, t_s, i_s]
                representatives = self.__extract_representatives(t_s)
                representatives.remove(k)
                representatives += [i_s,j_s]

                # Re-distribute members among updated representatives
                new_rep_distrib = self.__associate_to_representative(
                    representatives,
                    t_s
                )

                # Kill unused vertices, create necessary ones
                prev_v_distrib = self.__M_v[self.__s, t_s]
                alive_vertices = self.extract_alive_vertices(t=t_s)
                v_to_kill = []
                # v_to_create is a nested list. The first element of the
                # sublists is the representative, others are members associated
                # with it
                v_to_create = []
                rep_v_to_create = []

                for i in range(self.__N):

                    v_prev = prev_v_distrib[i]
                    rep_prev = self.__vertices[t_s][v_prev].representative
                    rep_new = new_rep_distrib[i]
                    # if the representative of this member has changed
                    # then the vertex of rep_prev must be killed
                    # and we can add this member to
                    if rep_prev != rep_new:
                        # add to the "to_kill" list
                        v_to_kill.append(v_prev)

                        list_members = get_indices_element(
                            my_list=rep_v_to_create,
                            my_element=rep_new,
                            all_indices=False,
                            if_none=-1
                        )
                        # If this new rep is already taken for a new vertex...
                        if idx == -1:
                            # ... then we have another vertex to create with
                            # rep_new as representant
                            rep_v_to_create.append(rep_new)
                            v_to_create.append([rep_new])

                        else:
                            # ...then idx is the index of the new
                            # vertex in the queue to their creation

                            # So we just associate the member with this rep
                            v_to_create[idx].append(i)


                    # Remove multiple occurences of the same vertex key
                    v_to_kill = list(set(v_to_kill))

                # Add vertices

                # update the graph with the new vertex
                #self.__nb_vertices[t] +=
                #self.__vertices[t].append(v)

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
    def members(self):
        """Original data, ensemble of time series

        :rtype: np.ndarray((N,T,d))
        """
        return np.copy(self.__members)

    @property
    def edges(self):
        """
        Nested list of edges of the graph (time, start_v, end_v)

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
    def distance_matrix(self):
        """
        Pairwise distance matrix for each time step (T,N,N)

        .. warning::
          Distance to self is set to NaN

        :rtype: np.ndarray((T,N,N))
        """
        return np.copy(self.__dist_matrix)