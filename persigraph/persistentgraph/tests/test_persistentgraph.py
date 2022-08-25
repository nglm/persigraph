import numpy as np
from numpy.testing import assert_array_equal

from ..persistentgraph import PersistentGraph
from ...datasets import mini


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
    g = PersistentGraph(members, time)
    g.construct_graph()

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
    ]
    output_exp = [
        g.nb_steps,
    ]
    for out, out_exp in zip(output, output_exp):
        assert out == out_exp , "out: " + str(out) + " expected " + str(out_exp)
    for out, out_exp in zip(output_np, output_np_exp):
        assert_array_equal(out, out_exp)


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