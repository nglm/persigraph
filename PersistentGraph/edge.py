from component import Component
class Edge(Component):

    def __init__(
        self,
        v_start:int,
        v_end:int,
        s_born: int = 0,
        t:int = None,
        num:int = None,
        nb_members:int = None,
    ):
        super().__init__(
            s_born=s_born,
            t=t,
            num=num,
            nb_members=nb_members,
        )
        self.v_start = v_start
        self.v_end = v_end

    @property
    def v_start(self):
        return self.__v_start

    @property
    def v_end(self):
        return self.__v_end

    @v_start.setter
    def v_start(self, v_start):
        if v_start is not None:
            if (v_start < 0):
                raise ValueError("v should be > O")
            self.__v_start = int(abs(v_start))

    @v_end.setter
    def v_end(self, v_end):
        if v_end is not None:
            if (v_end < 0):
                raise ValueError("v should be > O")
            self.__v_end = int(abs(v_end))


