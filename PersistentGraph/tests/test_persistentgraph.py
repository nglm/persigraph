import pytest
import numpy as np
from numpy.testing import assert_array_equal
from netCDF4 import Dataset



import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../../'))

from persistentgraph import PersistentGraph
from DataAnalysis.statistics import extract_variables

members = np.array([
    (0. ,1., 2., 1.,0.),
    (0. ,0, 0, 0,0),
    (0. ,-1, -2, -1, 0),
    (0 ,-0.5, -0.5, -0.5, -1),
])

nc = Dataset("tests/ec.ens.2020012900.sfc.meteogram.nc","r")
(list_var, var_names) = extract_variables(nc, var_names=["t2m"], ind_lat=np.array([0]), ind_long=np.array([0]))
members_nc = np.transpose(list_var[0].squeeze())

N_exp = int(4)
T_exp = int(5)
d_exp = int(1)
shape_dist_matrix_exp = (T_exp, N_exp, N_exp)
shape_members_exp = members.shape
nb_steps_exp = int((N_exp - 1)*T_exp + 1)
nb_vertices_exp = np.ones((T_exp), dtype=int)
nb_edges_exp = np.ones((T_exp-1), dtype=int)
M_v_exp = np.zeros((nb_steps_exp, T_exp, N_exp), dtype=int)

print(members)


#print("Distance_matrix: ", myGraph.distance_matrix)

def test_init():
    myGraph = PersistentGraph(members)
    output_int = [
        myGraph.N,
        myGraph.T,
        myGraph.d,
    ]
    output_it = [
        myGraph.nb_vertices,
        myGraph.nb_edges,
    ]

    output_int_exp = [
        N_exp,
        T_exp,
        d_exp,
    ]
    output_it_exp = [
        nb_vertices_exp,
        nb_edges_exp,
    ]
    for i in range(len(output_int)):
        assert output_int[i] == output_int_exp[i] , "output: " + str(output_int[i]) + " VS " + str(output_int_exp[i])
    for i in range(len(output_it)):
        assert_array_equal(output_it[i], output_it_exp[i])

def test_decreasing_distance():
    """
    Test that we take the distances in decreasing order
    """
    g = PersistentGraph(members)
    g.construct_graph()
    steps = g.steps
    dist_matrix = g.distance_matrix
    nb_steps = len(steps)
    for k in range(0,nb_steps):  #First step is (-1, -1, -1)
        (t_sup,i_sup,j_sup) = steps[k]
        for l in range(k+1,nb_steps):
            (t,i,j) = steps[l]
            assert dist_matrix[t_sup,i_sup,j_sup] >= dist_matrix[t,i,j]


# def test_nb_vertices():
#     g = PersistentGraph(members)
#     g.construct_graph()
#     for t in range(g.T):
#         assert g.nb_vertices[t] <= g.nb_vertices_max

# def test_members():
#     g = PersistentGraph(members_nc)
#     g.construct_graph()
#     assert np.shape(g.members) == (50,51)


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

def test_increment_nb_vertex():
    g = PersistentGraph(members)
    g.construct_graph()
    M_v = g.M_v
    for s in range(1,g.nb_steps):
        (t,i,j) = g.steps[s]
        assert len(np.unique(M_v[s-1,t])) < len(np.unique(M_v[s,t]))

# def test_one_vertex_per_member_last_step():

# def test_one_edge_per_member_last_step():
# print(myGraph.M_v)
# for list_v in myGraph.vertices:
#     print([v.num for v in list_v])
# print("steps", myGraph.steps)