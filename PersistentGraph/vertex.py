class Vertex():
    key_incr:int = 0


    def __init__(
        self,
        representative: int = None,
        # s_born: int = None,
        # s_death: int = None,
    ):
        self.key: int = Vertex.key_incr
        self.mean: float = 0.
        self.std: float = 1.
        self.representative = representative
        Vertex.key_incr += 1

    def reset_key_incr(self):
        Vertex.key_incr = 0
