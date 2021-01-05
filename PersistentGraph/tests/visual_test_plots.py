import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from persistentgraph import PersistentGraph
from plots import *


members = np.array([
    (0. ,1., 2., 1.,0.),
    (0. ,0, 0, 0,0),
    (0. ,-1, -2, -1, 0),
    (0 ,-0.5, -0.5, -0.5, -1),
])

g = PersistentGraph(members)
g.construct_graph(verbose=True)
print(g.nb_zeros)


# print(g.M_v)
# for t in range(g.T-1):
#     print("===============", t)
#     for e in g.edges[t]:
#         print("birth: ", e.s_birth, "death: ", e.s_death)
#     plt.show()

# for s in range(g.nb_steps):
#     fig, ax = plot_as_graph(g,s, show_vertices=True)
# plt.show()

# # fig, ax = plot_as_graph(g, show_vertices=True)
# plt.show()
# print(ax.collections)

# ax.collections = []
# print(ax.collections)
# print(ax.artists)

ani = make_gif(g, show_vertices=True, cumulative=False)
writer = PillowWriter(fps=2)
ani.save('test_gif.gif', writer=writer)