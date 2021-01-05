import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))


from persistentgraph import PersistentGraph
from plots import *


members = np.array([
    (0.1 ,  1.,    2.05,   1.1,   0.1 ),
    (0.,    0,     0,    0,    0.005 ),
    (-0.1,  -1,    -2,   -1,   0.05 ),
    (0.05,  -0.4,  -0.6, -0.5, -1),
])

g = PersistentGraph(
    members,
    time_axis = np.arange(5),
    maximize=False,
    score_type = 'max_variance',
    zero_type = 'uniform',
    )
print(members.shape)
g.construct_graph(
    verbose=2,
    pre_prune = False,
    post_prune = False)
print("number of vertices: ", [len(vt) for vt in g.vertices])
print("number of edges: ", [len(et) for et in g.edges])
# #print([[ (e.nb_members, e.score) for e in et] for et in g.vertices])
# print([[ (e.nb_members, e.scores) for e in et] for et in [g.vertices[0]]])
# print([[ (e.nb_members, e.scores) for e in et] for et in [g.edges[0]]])
# #print([[ (e.nb_members, e.ratio_score) for e in et] for et in g.vertices])
# print([[ (e.nb_members, e.ratio_scores) for e in et] for et in [g.edges[0]]])
print(g.time_axis)

for t in range(g.T):
    print(' -------------- ', t, ' -------------- ')
    print("num, value:", [ (v.num, v.info['params'][0]) for v in g.vertices[t] ])
    print(g.v_at_step[t])


for t in range(g.T):
    print(' -------------- ', t, ' -------------- ')
    print(g.time_axis[g.vertices[t][0].time_step])
    #print([ (v.info['params'][0]) for v in g.vertices[t] ])
    if t < g.T-1:
        print("v_end/start:", [ (e.v_start.num, e.v_end.num) for e in g.edges[t] ])
        print("scores:", [ (e.scores) for e in g.edges[t] ])
        print("life span:",[ (e.life_span) for e in g.edges[t] ])
        print("members:",[ (e.members) for e in g.edges[t] ])

print([v.num for v in g.vertices[0]])
print("---- Alive Vertices -------")
for s in range(g.nb_steps):
    print("s=",s, g.get_alive_vertices(steps=s))
print("---- Alive edges -------")
for s in range(g.nb_steps):
    print("s=",s, g.get_alive_edges(steps=s))


# print(g.sorted_steps['time_steps'])
# print(g.sorted_steps['local_step_nums'])

for t in range(g.T):
    print('--- ', t, '-----')
    print(g.local_steps[t])

# for s in range(g.nb_steps):
#     fig, ax = plot_as_graph(g,s, show_vertices=True)

fig, ax = plot_as_graph(
    g, show_vertices=True,
    show_edges=True,
    show_std = True)
plt.figure()
for m in g.members:
    plt.plot(m)
plt.show()

