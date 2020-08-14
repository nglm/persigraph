class Vertex():
    key_incr:int = 0


    def __init__(
        self,
        representative: int = None,
        s_born: int = None,
        num: int = None,
        value: float = None,
        std: float = None,
        nb_members: int = None
    ):
        self.__key = int(Vertex.key_incr)
        self.__num = num
        self.__value = value
        self.__std = std
        self.__nb_members = nb_members
        self.representative = representative
        self.__s_death = -1
        self.s_born = s_born
        self.__ratio_life = None
        self.__ratio_members = None
        Vertex.key_incr += 1

    def reset_key_incr(self):
        Vertex.key_incr = 0


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

    @property
    def nb_members(self):
        return self.__nb_members

    @property
    def ratio_members(self):
        return self.__ratio_members

    @property
    def ratio_life(self):
        return self.__ratio_life

    @num.setter
    def num(self, num):
        if num is not None:
            if (num < 0):
                raise ValueError("num should be > O")
            self.__num = int(abs(num))

    @nb_members.setter
    def nb_members(self, nb_members):
        if nb_members is not None:
            if (nb_members < 0):
                raise ValueError("number should be > O")
            self.__nb_members = int(abs(nb_members))


    @ratio_members.setter
    def ratio_members(self, ratio_members):
        if (ratio_members > 1) or (ratio_members < 0):
            raise ValueError("ratio should be within 0-1 range")
        self.__ratio_members = ratio_members


    @ratio_life.setter
    def ratio_life(self, ratio_life):
        if (ratio_life > 1) or (ratio_life < 0):
            raise ValueError("ratio should be within 0-1 range")
        self.__ratio_life = ratio_life

    @s_born.setter
    def s_born(self, s_born):
        if s_born is not None:
            if (s_born < 0):
                raise ValueError("s should be > 0")
            self.__s_born = int(max(s_born, 0))

    @s_death.setter
    def s_death(self, s_death):
        if s_death is not None:
            if (s_death < 0):
                raise ValueError("s should be > 0")
            self.__s_death = int(max(s_death, 1))

    @representative.setter
    def representative(self, rep):
        if rep is not None:
            if (rep < 0):
                raise ValueError("representative should be > 0")
            self.__representative = int(abs(rep))

    @value.setter
    def value(self, value):
        self.__value = value

    @std.setter
    def std(self, std):
        self.__std = std