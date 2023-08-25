import numpy as np
from numpy.testing import assert_array_equal
from pycvi.datasets import mini

from ..persistentgraph import PersistentGraph
from ..plots import graph


def test_k_info():
    """
    Test that the sum of life spans is 1 for each t
    """
    members, time = mini()
    g = PersistentGraph(members, time)
    g.construct_graph(verbose=True)
    graph(g)

    lf_t = np.array([
        sum([
            g.k_info[k]["life_span"][t] for k in g.k_range if k>0
        ]) for t in range(g.T)
    ])

    msg = "Sum of life_span != 1" + str(lf_t)
    eps = 0.05
    lf_min = 1-eps
    lf_max = 1+eps
    assert np.all((lf_t >= lf_min)), msg
    assert np.all((lf_t <= lf_max)), msg
