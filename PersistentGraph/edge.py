
class Edge():
    key_incr:int = 0


    def __init__(
        self,
        v_start,
        v_end,
        # s_born: int = None,
        # s_death: int = None,
    ):
        self.key: int = Edge.key_incr
        self.start = v_start
        self.end = v_end
        self.nb_members: int = None  #TODO: find how many members there are
        self.s_born = None
        self.s_death = None
        Edge.key_incr += 1

