import numpy as np
import warnings
from netCDF4 import Dataset

from .. import PersistentGraph
from ..plots import *
from ...Preprocessing.extraction import preprocess_meteogram


def warn(*args, **kwargs):
    pass


def main():
    warnings.warn = warn

    members = np.array([
        (0.1 ,  1.,    2.05,   1.1,   0.1 ),
        (0.005,    0,     0,    0,    0.005 ),
        (-0.1,  -1,    -2,   -1,   0.05 ),
        (0.05,  -0.4,  -0.6, -0.5, -1),
    ])

    members_bis = np.array([
        (0.2 ,  0.3,    1,   1.1,   0.6 ),
        (0.1,    0.15,     0.5,    0.6,    0.4 ),
        (-0.1,  -0.2,    -1,   -1.2,   -1 ),
        (-0.3,  -0.4,  -0.6, -0.7, -1),
    ])

    members_biv = np.ones((4,2,5))
    members_biv[:,0,:] = members
    members_biv[:,1,:] = members_bis
    members = members_biv
    print(members.shape)

    PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"
    # To get the right variable names and units
    filename = 'ec.ens.2020020200.sfc.meteogram.nc'
    nc = Dataset(PATH_DATA + filename,'r')
    var_names = ['tcwv']

    data_dict = preprocess_meteogram(
        filename = filename,
        path_data = PATH_DATA,
        var_names= var_names,
        ind_time=None,
        ind_members=None,
        ind_long=[0],
        ind_lat=[0],
        to_standardize = False,
        )

    #members = data_dict['members'][0]

    model_type = "KMeans"
    #model_type = "Naive"



    g = PersistentGraph(
        members,
        time_axis = np.arange(members.shape[-1]),
        score_type = 'inertia',
        zero_type = 'bounds',
        model_type = model_type,
        k_max=8,
        )
    print(members.shape)
    g.construct_graph(
        verbose=2,
        pre_prune = False,
        post_prune = False,
    )
    print("number of vertices: ", [len(vt) for vt in g.vertices])
    print("number of edges: ", [len(et) for et in g.edges])
    # #print([[ (e.nb_members, e.score) for e in et] for et in g.vertices])
    # print([[ (e.nb_members, e.scores) for e in et] for et in [g.vertices[0]]])
    # print([[ (e.nb_members, e.scores) for e in et] for et in [g.edges[0]]])
    # #print([[ (e.nb_members, e.ratio_score) for e in et] for et in g.vertices])
    # print([[ (e.nb_members, e.ratio_scores) for e in et] for et in [g.edges[0]]])
    print(g.time_axis)

    for t in range(g.T):
        print(' ============== ', t, ' ============== ')
        print("num, value:", [ (v.num, v.info['mean']) for v in g.vertices[t] ])
        print(g.v_at_step[t])


    for t in range(g.T):
        print(' ============== ', t, ' ============== ')
        print(' ----- ratio_scores ----- ')
        print("local_s['ratio_score'], local_s['score']: ", [(local_s['ratio_score'], local_s['score']) for local_s in g.local_steps[t] ])
        print(' ----- vertices ----- ')
        print("v.scores: ",  [ (v.scores) for v in g.vertices[t] ])
        print("v.life_span: ",  [ (v.life_span) for v in g.vertices[t] ])
        if t < g.T-1:
            print(' ----- edges ----- ')
            print("v_end/start:", [ (e.v_start, e.v_end) for e in g.edges[t] ])
            print("scores:", [ (e.scores) for e in g.edges[t] ])
            print("life span:",[ (e.life_span) for e in g.edges[t] ])
            print("members:",[ (e.members) for e in g.edges[t] ])

    print([v.num for v in g.vertices[0]])
    print("---- Alive Vertices -------")
    for s in range(g.nb_steps):
        "ratio=",g.sorted_steps['ratio_scores'][s]
    for s in range(g.nb_steps):
        print("s=",s ,g.get_alive_vertices(steps=s))
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

    # fig, ax = plot_as_graph(g)
    # plt.show()

    # fig, axs = plot_most_revelant_components(g)
    # plt.show()

    fig, ax = plot_overview(g)
    fig.savefig('tmp')
    plt.show()


    plt.figure()
    for m in g.members:
        plt.plot(m[0])
    # plt.show()

    plt.figure()
    for m in g.members:
        plt.plot(m[1])
    plt.show()

    # k_plot(g)
    # plt.show()

if __name__ == '__main__':
    main()