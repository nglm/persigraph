class Component():

    key_incr:int = 0

    def __init__(
        self,
        s_born:int = 0,
        t:int = None,
        num:int = None,
        nb_members: int = None,
    ):
        self.__key: int = Component.key_incr
        self.num = num
        self.__s_death = -1
        self.s_born = s_born
        self.time_step = t
        self.nb_members = nb_members
        self.__ratio_life = None
        self.__ratio_members = None
        self.__r_born = None,
        self.__r_death = None
        self.__t = None
        Component.key_incr += 1

    def reset_key_incr(self):
        Component.key_incr = 0


    def update_life_info(
        self,
        distances,
        N,
        nb_steps,
    ):
        """
        Update:
          - ratio_life
          - ratio_member
          - r_born
          - r_death
          - s_death if cmpt still alive at the end

        :param distances: Matrix used as normalizer
        :type distances: [type]
        :param N: Matrix used as normalizer
        :type distances: int
        """
        self.r_born = distances[self.s_born] / distances[0]
        self.r_death = distances[self.s_death] / distances[0]
        self.ratio_life = self.r_born - self.r_death
        self.ratio_members = self.nb_members/N
        if self.s_death == -1:
            self.s_death = nb_steps



    @property
    def s_born(self):
        return(self.__s_born)

    @property
    def s_death(self):
        return(self.__s_death)

    @property
    def num(self):
        return self.__num

    @property
    def key(self):
        return self.__key

    @property
    def nb_members(self):
        return self.__nb_members

    @property
    def ratio_life(self):
        """
        (distances[s_born] - distances[s_death])/distances[0]

        with distances[-1] = 0.
        (associated to the graph representing members)

        :return: [description]
        :rtype: [type]
        """
        return self.__ratio_life

    @property
    def ratio_members(self):
        return self.__ratio_members

    @property
    def r_born(self):
        return self.__r_born

    @property
    def r_death(self):
        return self.__r_death

    @property
    def time_step(self):
        return self.__time_step

    @time_step.setter
    def time_step(self, t):
        if (t < 0):
            raise ValueError("t should be >= 0")
        self.__time_step = t


    @r_born.setter
    def r_born(self, r_born):
        if (r_born > 1) or (r_born < 0):
            raise ValueError("ratio should be within 0-1 range")
        self.__r_born = r_born

    @r_death.setter
    def r_death(self, r_death):
        if (r_death > 1) or (r_death < 0):
            raise ValueError("ratio should be within 0-1 range")
        self.__r_death = r_death

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
            self.__s_death = int(s_death)

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

