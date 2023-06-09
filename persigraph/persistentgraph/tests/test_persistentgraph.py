import numpy as np
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from .._scores import SCORES, SCORES_TO_MINIMIZE
from .._clustering_model import CLUSTERING_METHODS
from ..persistentgraph import PersistentGraph
from ..plots import graph, overview
from ...datasets import mini

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
        (g.N, g.d, g.T),
        g.members_zero.shape,
    ]
    output_np_exp = [
        members_exp,
        time_exp,
        members_exp.shape,
        (g.N, g.d, g.T),

    ]
    output = []
    output_exp = []

    for out, out_exp in zip(output, output_exp):
        assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
    for out, out_exp in zip(output_np, output_np_exp):
        assert_array_equal(out, out_exp)

def test_construct_graph():
    members, time = mini()
    g = PersistentGraph(members, time)
    g.construct_graph()
    overview(g)

    # ------------------------------------------------------------------
    # Test numbers of vertices / edges / steps are consistent
    # ------------------------------------------------------------------
    output_np = [
        [len(edges) for edges in g.edges],
        [len(vertices) for vertices in g.vertices],
    ]
    output_np_exp = [
        g.nb_edges,
        g.nb_vertices,
    ]
    output = [
        sum(g.nb_local_steps),
        len(g.sorted_steps["scores"])
    ]
    output_exp = [
        g.nb_steps,
        g.nb_steps
    ]
    for out, out_exp in zip(output, output_exp):
        assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
    for out, out_exp in zip(output_np, output_np_exp):
        assert_array_equal(out, out_exp)

    # ------------------------------------------------------------------
    # Test vertices and edges sorted
    # ------------------------------------------------------------------
    for t in range(g.T):
        v_t = g.v_at_step()[t]['v']
        msg = "t: {:d} | vertices not sorted {}".format(t, v_t)
        (assert_sorted(v_t_s, msg) for v_t_s in v_t)

        e_t = g.e_at_step()[t]['e']
        msg = "t: {:d} | edges not sorted {}".format(t, e_t)
        (assert_sorted(e_t_s, msg) for e_t_s in e_t)
    fig, ax = overview(g)
    fname = "test_construct_graph"
    fig.savefig('tmp/'+fname)
    g.save('tmp/'+fname, type="json")

def test_sorted_steps():
    """
    Test that sorted steps is sorted in increasing order of ratio scores.
    """
    members, time = mini(multivariate=True)
    w = 3
    list_squared_radius = [True, False]
    DTWs = [False, True]
    for squared_radius in list_squared_radius:
        for DTW in DTWs:
            g = PersistentGraph(
                members, time, time_window=w,
                DTW=DTW, squared_radius=squared_radius)
            g.construct_graph()
            fig, ax = overview(g)
            fname = (
                "test_sorted_steps_" + "DTWs_" + str(DTW)
                + "squared_radius_" + str(squared_radius)
            )
            fig.savefig('tmp/'+fname)
            g.save('tmp/'+fname, type="json")

            sorted_steps = g.sorted_steps
            scores = sorted_steps["scores"]
            r_scores = sorted_steps["ratio_scores"]
            msg = "r_scores: " + str(r_scores)
            assert_sorted(r_scores, msg)

def test_time_window():
    members, time = mini(multivariate=True)
    list_w = [t for t in range(1, len(time))]
    for w in list_w:
        g = PersistentGraph(members, time, time_window=w)
        g.construct_graph()
        fig, ax = overview(g)
        fname = "test_time_window_" + "time_window_" + str(w)
        fig.savefig('tmp/'+fname)
        g.save('tmp/'+fname, type="json")

def test_squared_radius():
    members, time = mini(multivariate=True)
    list_squared_radius = [True, False]
    list_w = [1,2,3, len(time)]
    for squared_radius in list_squared_radius:
        for w in list_w:
            g = PersistentGraph(
                members,
                time,
                squared_radius=squared_radius,
                time_window=w,
            )
            g.construct_graph()
            fig, ax = overview(g)
            fname = (
                "test_squared_radius_" + "time_window_" + str(w)
                + "squared_radius_" + str(squared_radius)
            )
            fig.savefig('tmp/'+fname)
            g.save('tmp/'+fname, type="json")

def test_DTW():
    members, time = mini(multivariate=True)
    methods = CLUSTERING_METHODS["names"]
    DTWs = [False, True]
    list_squared_radius = [True, False]
    list_w = [1, max(1, len(time)//2), max(1, len(time)-1)]
    for i, m in enumerate(methods):
        for is_dtw in DTWs:
            for squared_radius in list_squared_radius:
                for w in list_w:
                    if not (
                        is_dtw and CLUSTERING_METHODS["classes-dtw"][i] is None
                    ):
                        if not is_dtw:
                            continue
                        g = PersistentGraph(
                            members, time, model_class=m, DTW=is_dtw,
                            squared_radius=squared_radius, time_window=w
                            )
                        g.construct_graph()
                        fig, ax = overview(g)
                        fname = "test_DTW_" + str(m) + "_" + str(is_dtw) + "_squared_" + str(squared_radius) + "_w_" + str(w)
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
    members, time = mini(multivariate=True)
    scores = SCORES_TO_MINIMIZE
    for s in scores:
        print(s)
        g = PersistentGraph(members, time, score_type=s)
        g.construct_graph()
        overview(g)

# def test_decreasing_distance():
#     """
#     Test that we take the distances in decreasing order
#     """
#     g = PersistentGraph(members, k_max=5)
#     g.construct_graph()
#     steps = g.steps
#     dist_matrix = g.distance_matrix
#     nb_steps = len(steps)
#     for k in range(0,nb_steps):  #First step is (-1, -1, -1)
#         (t_sup,i_sup,j_sup) = steps[k]
#         for l in range(k+1,nb_steps):
#             (t,i,j) = steps[l]
#             assert dist_matrix[t_sup,i_sup,j_sup] >= dist_matrix[t,i,j]



# def test_consistent_values():
#     list_g = []
#     g = PersistentGraph(members)
#     g.construct_graph()
#     list_g.append(g)
#     g = PersistentGraph(members_nc)
#     g.construct_graph()
#     list_g.append(g)
#     for g in list_g:
#         for t in range(g.T):
#             values = np.array([v.value for v in g.vertices[t]])
#             assert np.amin(values) == np.amin(g.members[:,t])
#             assert np.amax(values) == np.amax(g.members[:,t])

# def test_increment_nb_vertex():
#     g = PersistentGraph(members, k_max=5)
#     g.construct_graph()
#     M_v = g.M_v
#     for s in range(1,g.nb_steps):
#         (t,i,j) = g.steps[s]
#         assert len(np.unique(M_v[s-1,t])) < len(np.unique(M_v[s,t]))

# def test_one_vertex_per_member_last_step():

# def test_one_edge_per_member_last_step():
# print(myGraph.M_v)
# for list_v in myGraph.vertices:
#     print([v.num for v in list_v])
# print("steps", myGraph.steps)