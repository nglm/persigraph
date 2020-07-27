import numpy as np
from sklearn.metrics import pairwise_distances
from vertex import Vertex

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
        self.__members = np.copy(members)
        self.__dist_matrix = None
        self.__compute_dist_matrix()
        self.__N: int = shape(-2)
        self.__T: int = shape(-1)
        self.__nb_steps = (self.__N - 1)*self.__T + 1
        self.__s:int = 0 # current graph iteration

        # Declared attributes
        self.__vertices = None
        self.__edges = None
        self.__M_v = np.zeros((self.__nb_steps,self.__T,self.__N))
        self.__nb_vertices = np.zeros((self.__T)) # number of vertices created

    def __compute_dist_matrix(
        self,
    ):
        """
        Compute the pairwise distance matrix for each time step
        """
        dist = []
        # append is more efficient for list than for np
        for t in range(self.__T):
            dist_t = pairwise_distances(self.__members[t])
            dist.append(dist_t)
        self.__dist_matrix = np.asarray(dist)

    def __sort_dist_matrix(
        self,
    ):
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

    def __get_members_from_representative(
        self,
        rep,
        rep_distribution,
    ):



    def __add_vertex(
        self,
        value:float = None,
        std: float = None,
        t: int = None,
        representative: int = None,
    ):
        # creating the vertex
        v = Vertex()
        v._Vertex__value = value
        v._Vertex__std = std
        v._Vertex__num = nb_vertices[t]
        v._Vertex__representative = int(representative)
        v.s_born = self.__s
        # update the graph
        self.__nb_vertices[t] += 1
        self.__vertices[t].append(v)

    def __find_representatives(
        self,
        t,
    ):
        """
        Find representatives

        s must be > 0
        """
        # extract alive vertices
        vertices_t = list(set(self.__vertices[t]))
        # extract representatives
        representatives = [v.representative for v in vertices_t]
        return representatives

    def __associate_to_representatives(
        self,
        representatives,
        t,
    ):
        """
        Find vertex corresponding to each member

        s must be > 0
        """
        # extract distance to representatives
        dist = []
        for rep in representatives:
            dist.append(self.__dist_matrix[t,rep])
        dist = np.asarray(dist)     # (nb_rep, N) array
        # for each member, find the representative that is the closest
        idx = np.argmin(dist, axis=0)
        distribution = np.zeros((self.__N))
        return [representatives[i] for i in idx]



    def __kill_vertex(
        self,
        v
    ):
        v.s_death = self.__s

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
                representatives = self.__find_representatives(t_s)
                representatives.remove(k)
                representatives += [i_s,j_s]

                # Re-distribute, kill unused vertices and create necessary ones
                new_distrib = self.__associate_to_representatives(representatives,t_s)
                prev_distrib = self.__M_v[self.__s, t_s]
                list_vertices = self.__vertices[t_s]
                v_to_kill = []
                for i in range(self.__N):
                    rep_prev = prev_distrib[i]
                    rep_new = new_distrib[i]
                    if rep_prev != rep_new





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

        :rtype: List
        """
        return self.__edges

    @property
    def vertices(self):
        """
        Nested list of vertices of the graph (t, nb_vertices[t])

        :rtype: List
        """
        return self.__vertices

    @property
    def M_v(self):
        """
        Distribution of the members among the vertices (nb_steps, T, N)

        :rtype: np.ndarray((nb_steps, T, N))
        """
        return np.copy(self.__M_v)

    @property
    def nb_vertices(self):
        """
        Number of vertices created at each time step (T)

        :rtype: np.ndarray((T))
        """
        return np.copy(self.__nb_vertices)

    @property
    def distance_matrix(self):
        """
        Pairwise distance matrix for each time step (T,N,N)

        :rtype: np.ndarray((T,N,N))
        """
        return np.copy(self.__dist_matrix)