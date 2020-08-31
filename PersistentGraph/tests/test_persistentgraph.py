import pytest
import numpy as np
from numpy.testing import assert_array_equal


# ------
# Source
# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to access persistentgraph
# ------

from persistentgraph import PersistentGraph

members = np.array([
    (0. ,1., 2., 1.,0.),
    (0. ,0, 0, 0,0),
    (0. ,-1, -2, -1, 0),
    (0 ,-0.5, -0.5, -0.5, -1),
])

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
        myGraph.nb_steps,
    ]
    output_it = [
        myGraph.M_v,
        myGraph.nb_vertices,
        myGraph.nb_edges,
        myGraph.distance_matrix.shape
    ]

    output_int_exp = [
        N_exp,
        T_exp,
        d_exp,
        nb_steps_exp,
    ]
    output_it_exp = [
        M_v_exp,
        nb_vertices_exp,
        nb_edges_exp,
        shape_dist_matrix_exp,
    ]
    for i in range(len(output_int)):
        assert output_int[i] == output_int_exp[i] , "output: " + str(output_int[i]) + " VS " + str(output_int_exp[i])
    for i in range(len(output_it)):
        assert_array_equal(output_it[i], output_it_exp[i])

def test_decreasing_distance():
    """
    Test that we take the distances in decreasing order
    """
    myGraph = PersistentGraph(members)
    myGraph.construct_graph()
    steps = myGraph.steps
    dist_matrix = myGraph.distance_matrix
    nb_steps = len(steps)
    for k in range(nb_steps):
        (t_sup,i_sup,j_sup) = steps[k]
        for l in range(k+1,nb_steps):
            (t,i,j) = steps[l]
            assert dist_matrix[t_sup,i_sup,j_sup] >= dist_matrix[t,i,j]


# print(myGraph.M_v)
# for list_v in myGraph.vertices:
#     print([v.num for v in list_v])
# print("steps", myGraph.steps)