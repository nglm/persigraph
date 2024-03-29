import numpy as np
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from .._scores import SCORES
from .._clustering_model import CLUSTERING_METHODS
from ..persistentgraph import PersistentGraph
from ..plots import graph, overview
from pycvi.datasets import mini

def assert_sorted(l, msg=""):
    assert np.all([l[i] <= l[i+1] for i in range(len(l) - 1)]), msg


def test_init():
    members, time = mini()
    members_exp, time_exp = mini()
    g = PersistentGraph(members, time)
    # ------------------------------------------------------------------
    # test members in the graph don't change by doing the following:
    members[0] = 0
    time[1] = 0
    # ------------------------------------------------------------------
    output_np = [
        g.members,
        g.time_axis,
        (g.N, g.T, g.d),
    ]
    output_np_exp = [
        members_exp,
        time_exp,
        members_exp.shape,

    ]
    output = []
    output_exp = []

    for out, out_exp in zip(output, output_exp):
        assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
    for out, out_exp in zip(output_np, output_np_exp):
        assert_array_equal(out, out_exp)

def test_construct_graph():
    members, time = mini()
    g = PersistentGraph(members, time, w=1)
    g.construct_graph()
    overview(g)

    # ------------------------------------------------------------------
    # Test numbers of vertices / edges are consistent
    # ------------------------------------------------------------------
    output_np = [
        [len(edges) for edges in g.edges],
        [len(vertices) for vertices in g.vertices],
    ]
    output_np_exp = [
        g.nb_edges,
        g.nb_vertices,
    ]
    for out, out_exp in zip(output_np, output_np_exp):
        assert_array_equal(out, out_exp)

    fname = "test_construct_graph"
    g.save('tmp/'+fname, type="json")

def test_time_window():
    members, time = mini(multivariate=True)
    list_w = [t for t in range(1, len(time))]
    for w in list_w:
        g = PersistentGraph(members, time, w=w)
        g.construct_graph()
        fig, ax = overview(g)
        fname = "test_time_window_" + "time_window_" + str(w)
        fig.savefig('tmp/'+fname)
        g.save('tmp/'+fname, type="json")

def _square_radius(X: np.ndarray) -> np.ndarray:
    """
    Returns r*X with r = sqrt(RMM1**2 + RMM2**2)
    """
    # r = sqrt(RMM1**2 + RMM2**2)
    # r of shape (N, 1, T)
    r = np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
    # r*X gives the same angle but a squared radius
    return r*X

def test_transformer():
    members, time = mini(multivariate=True)
    list_transformers = [None, _square_radius]
    list_w = [1,2,3, len(time)]
    for transformer in list_transformers:
        for w in list_w:
            g = PersistentGraph(
                members,
                time,
                transformer=transformer,
                w=w,
            )
            g.construct_graph()
            fig, ax = overview(g)
            fname = (
                "test_transformer" + "time_window_" + str(w)
                + "squared_radius_" + str(transformer is not None)
            )
            fig.savefig('tmp/'+fname)
            g.save('tmp/'+fname, type="json")

def test_DTW():
    members, time = mini(multivariate=True)
    methods = CLUSTERING_METHODS["names"]
    DTWs = [False, True]
    list_transformers = [_square_radius, None]
    list_w = [1, max(1, len(time)//2), max(1, len(time)-1)]
    for i, m in enumerate(methods):
        for is_dtw in DTWs:
            for transformer in list_transformers:
                for w in list_w:
                    if not (
                        is_dtw and CLUSTERING_METHODS["classes-dtw"][i] is None
                    ):
                        if not is_dtw:
                            continue
                        g = PersistentGraph(
                            members, time, model_class=m, DTW=is_dtw,
                            transformer=transformer, w=w
                            )
                        g.construct_graph()
                        fig, ax = overview(g)
                        fname = "test_DTW_" + str(m) + "_" + str(is_dtw) + "_squared_" + str(transformer is not None) + "_w_" + str(w)
                        fig.savefig('tmp/'+fname)
                        g.save('tmp/'+fname, type="json")

def test_clustering_methods():
    members, time = mini()
    methods = CLUSTERING_METHODS["names"]
    for m in methods:
        print(m)
        g = PersistentGraph(members, time, model_class=m)
        g.construct_graph()
        fig, ax = overview(g)
        fname = "test_clustering_methods_" + "method_" + str(m)
        fig.savefig('tmp/'+fname)
        g.save('tmp/'+fname, type="json")

def test_agglomerative():
    members, time = mini()
    model_class = AgglomerativeClustering
    linkages = ["ward", "simple", "complete", "average"]
    list_model_kw = [{"linkage" : l} for l in linkages]
    for model_kw in list_model_kw:
        g = PersistentGraph(
            members,
            time,
            model_class=model_class,
            model_kw=model_kw
        )
        g.construct_graph()
        overview(g)

def test_gmm():
    members, time = mini()
    model_class = GaussianMixture
    #model_class_kw = {"k_arg_name" : "n_components"}
    model_class_kw = {}
    list_model_kw = [{}]
    for model_kw in list_model_kw:
        g = PersistentGraph(
            members,
            time,
            model_class=model_class,
            model_kw=model_kw,
            model_class_kw = model_class_kw
        )
        g.construct_graph()
        overview(g)

def test_scores():

    scores = SCORES
    for s in scores:
        print(s)
        for DTW in [True, False]:
            for multivariate in [True, False]:
                for window in [None, 1, 3]:
                    members, time = mini(multivariate=multivariate)
                    for scaler in [None, StandardScaler()]:
                        g = PersistentGraph(
                            members, time,
                            DTW=DTW, scaler=scaler, w=window,
                            score=s(),
                            )
                        g.construct_graph()

