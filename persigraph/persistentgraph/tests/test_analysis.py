import numpy as np
from numpy.testing import assert_array_equal

from ..persistentgraph import PersistentGraph
from ..analysis import get_k_life_span
from ..plots import graph
from ...datasets import mini

def test_get_k_life_span():
    """
    Test that the sum of life spans is 1(-worst_ratio) for each t
    """
    members, time = mini()
    g = PersistentGraph(members, time)
    g.construct_graph(verbose=True)
    graph(g)

    life_spans = get_k_life_span(g)
    # from list of life_span for each k
    # to list of life_spans for each t
    # lf_t = [0 for _ in range(g.T)]
    # for t in range(g.T):
    #     for k, lf_k in life_spans.items():
    #         lf_t[t] += lf_k[t]

    lf_t = np.array([
        sum([lf_k[t] for lf_k in life_spans.values()])
        for t in range(g.T)
    ])

    msg = "Sum of life_span != 1" + str(lf_t)
    eps = 0.05
    lf_min = 1-eps
    lf_max = 1+eps
    assert np.all((lf_t >= lf_min)), msg
    assert np.all((lf_t <= lf_max)), msg
