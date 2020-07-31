
class Edge():
    key_incr:int = 0


    def __init__(
        self,
        v_start:int,
        v_end:int,
        nb_members:int,
        s_born: int = None
    ):
        self.key: int = Edge.key_incr
        self.__num: int = None
        self.start = v_start
        self.end = v_end
        self.nb_members = nb_members
        self.s_born = s_born
        self.s_death = None
        Edge.key_incr += 1

