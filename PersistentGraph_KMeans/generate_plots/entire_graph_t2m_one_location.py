#!/usr/bin/env python3
import sys
import os
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from DataAnalysis.statistics import preprocess_data

TYPE_PG = 'KMeans'

if TYPE_PG == 'KMeans':
    from PersistentGraph_KMeans.persistentgraph import *
    from PersistentGraph_KMeans.plots import *
else:
    from PersistentGraph.persistentgraph import *
    from PersistentGraph.plots import *


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

score_type = 'variance'
zero_type = 'uniform'

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = (
    "/home/natacha/Documents/tmp/figs/PG/"
    + TYPE_PG
    + "/t2m/entire_graph/" + score_type + "/"
)

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and  fname.endswith(".nc")
]




# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def main():
    for filename in LIST_FILENAMES:

        list_var, list_names, time = preprocess_data(
            filename = filename,
            path_data = PATH_DATA,
            var_names=['t2m'],
            ind_time=None,
            ind_members=None,
            ind_long=[0],
            ind_lat=[0],
            to_standardize = False,
            )

        t2m = list_var[0]

        g = PersistentGraph(
                time_axis = time,
                members = t2m,
                score_is_improving = False,
                score_type = score_type,
                zero_type = zero_type,
        )
        g.construct_graph(
            verbose=True,
        )

        # ---------------------------
        # Plot entire graph
        # ---------------------------
        fig, ax = plot_as_graph(
            g, show_vertices=True, show_edges=True, show_std = True
        )

        fig_suptitle = (
            filename
            + "\nEntire graph, variable t2m \n"
        )

        ax.set_title(fig_suptitle)
        # ---------------------------
        # Save plot and graph
        # ---------------------------
        path_fig = PATH_FIG_PARENT + "plots/"
        name_fig = path_fig + filename[:-3] + ".png"
        makedirs(path_fig, exist_ok = True)
        fig.savefig(name_fig)
        plt.close()

        path_graph = PATH_FIG_PARENT + "graphs/"
        makedirs(path_graph, exist_ok = True)
        name_graph = path_graph + filename[:-3]
        g.save(name_graph)

main()