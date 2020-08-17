import numpy as np


# ------
# Source
# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # to access persistentgraph
# ------

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

# print(g.M_v)
# for t in range(g.T-1):
#     print("===============", t)
#     for e in g.edges[t]:
#         print("born: ", e.s_born, "death: ", e.s_death)

for s in range(g.nb_steps):
    fig, ax = plot_as_graph(g,s, show_vertices=False)

fig, ax = plot_as_graph(g, show_vertices=False)
plt.show()

