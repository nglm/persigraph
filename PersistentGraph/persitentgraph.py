import numpy as np

class PersistentGraph():


    def __init__(
        self,
        members,
    ):
        shape = members.shape
        if len(shape) < 3:
            self.__d = 1
        else:
            self.__d = shape(0)

        self.__members = np.copy(members)
        self.__N: int = shape(-2)
        self.__T: int = shape(-1)
        self.__nb_steps = (self.__N - 1)*self.__T + 1
        self.M_v = np.zeros((self.__N,self.__N,self.__T))
        self.__vertices = None
        self.__edges = None
        self.__s:int = 0 # current graph iteration


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

        self.M_v

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

        :rtype: np.ndarray([N,T,d])
        """
        return np.copy(self.__members)

    @property
    def edges(self):
        """
        Nested list of edges of the graph (time, start_v, end_v)
        """
        return self.__edges

    @property
    def vertices(self):
        """
        Nested list of vertices of the graph (time, key_v)
        """
        return self.__vertices
