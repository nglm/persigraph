class Vertex():
    key_incr:int = 0


    def __init__(
        self,
        representative: int = None,
        # s_born: int = None,
        # s_death: int = None,
    ):
        self.__key: int = Vertex.key_incr
        self.__num: int = None
        self.__value: float = 0.
        self.__std: float = 1.
        self.__representative = representative
        self.__s_born:int = None
        self.__s_death:int = None
        Vertex.key_incr += 1



    def reset_key_incr(self):
        Vertex.key_incr = 0

    @s_born.setter
    def s_born(self, s_born):
        if self.__s_death is not None:
            s_born = min(self.__s_death-1, s_born)
        self.__s_born = int(max(s_born), 0)

    @s_death.setter
    def s_death(self, s_death):
        if self.__s_born is not None:
            s_death = max(self.__s_born+1, s_death)
        self.__s_death = int(max(s_death), 1)

    @property
    def s_born(self):
        return(self.__s_born)

    @property
    def s_death(self):
        return(self.__s_death)

    @property
    def key(self):
        return self.__key

    @property
    def num(self):
        return self.__num

    @property
    def value(self):
        return self.__value

    @property
    def std(self):
        return self.__std

    @property
    def representative(self):
        return self.__representative