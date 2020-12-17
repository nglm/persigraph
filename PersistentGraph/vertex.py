from PersistentGraph.component import Component

class Vertex(Component):

    def __init__(
        self,
        representative: int = None,
        value: float = None,
        std: float = None,
        s_birth: int = None,
        t:int = None,
        num: int = None,
        nb_members: int = None
    ):
        super().__init__(
            s_birth=s_birth,
            t=t,
            num=num,
            nb_members=nb_members,
        )
        self.__value = value
        self.__std = std
        self.representative = representative

    @property
    def value(self):
        return self.__value

    @property
    def std(self):
        return self.__std

    @property
    def representative(self):
        return self.__representative

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
