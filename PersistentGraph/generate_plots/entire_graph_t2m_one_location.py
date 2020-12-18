#!/usr/bin/env python3


#FIXME: 2020/12 Make sure it is still working after the clean-up
import sys
import os
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))  #to use PG submodules
sys.path.insert(1, os.path.join(sys.path[0], '../..'))  #to use DA submodules

from DataAnalysis.statistics import preprocess_data
from utils.lists import get_indices_element
from persistentgraph import PersistentGraph
from plots import plot_as_graph, plot_edges
from analysis import sort_components_by, get_contemporaries



# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/PG/t2m/"
#path_fig = "/home/natacha/Documents/tmp/figs/PG/t2m/entire_graph/"

# Choose which variables should be ploted
# type: List(str)
# Available variables:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”)
# if None: var_names = ["t2m","d2m","msl","u10","v10","tcwv"]
var_names=["t2m"]
# Choose which instants should be ploted
# type: ndarray(int)
ind_time=None
# Choose which members should be ploted
# type: ndarray(int)
ind_members=None
# Choose which longitude should be ploted
# type: ndarray(int)
#ind_long=[15]
ind_long=[0]
# Choose which latitude should be ploted
# type: ndarray(int)
#ind_lat=[10]
ind_lat=[0]

# Choose nb members threshold
threshold_m = 0
# Choose nb members threshold
threshold_l = 0

to_standardize = False

# Choose which files should be used
list_filenames = listdir(PATH_DATA)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]


# ---------------------------------------------------------
# script:
# ---------------------------------------------------------

type_op = ["max_distance"]
for weights in [True, False]:
    for op in type_op:
        for filename in list_filenames:

            list_var, list_names, time = preprocess_data(
                filename = filename,
                path_data = PATH_DATA,
                var_names = var_names,
                ind_time = ind_time,
                ind_members = ind_members,
                ind_long = ind_long,
                ind_lat = ind_lat,
                to_standardize = to_standardize,
                )
            t2m = list_var[0]

            if weights:
                weights_file = (
                    "/home/natacha/Documents/tmp/figs/global_variation_t2m/all_forecasts_"
                    + op
                    + ".txt"
                )
                weights_values = np.loadtxt(weights_file)
            else:
                weights_values = None
            g = PersistentGraph(
                members=t2m, time_axis=time, weights=weights_values
            )
            g.construct_graph()

            # ---------------------------
            # Plot entire graph
            # ---------------------------
            fig, ax = plot_as_graph(
                g, show_vertices=True, show_edges=True,
                threshold_m=threshold_m, threshold_l=threshold_l,
            )

            if weights:
                fig_suptitle = (
                    filename
                    + "\nEntire graph for variable t2m \n"
                    + "Type of weights: " + op
                )
            else:
                fig_suptitle = (
                    filename
                    + "\nEntire graph for variable t2m \n"
                    + "distance not weighted"
                )

            ax.set_title(fig_suptitle)
            path_fig = PATH_FIG_PARENT + "entire_graph/"

            if weights:
                name_fig = (
                    path_fig + op +"_"+ filename[:-3] + ".png"
                )
            else:
                name_fig = path_fig + filename[:-3] + ".png"
            makedirs(path_fig, exist_ok = True)

            fig.savefig(name_fig)
            plt.close()

            # ---------------------------
            # Plot only older edges and contemporaries
            # ---------------------------

            # sorted_edges = sort_by_ratio_life(g.edges)
            # older_edges = [e_t[0] for e_t in sorted_edges]
            # contemporaries_older_edges = [
            #     get_contemporaries(g, e) for e in older_edges
            # ]
            # plt.figure(figsize=(10,10))
            # ax = plt.gca()
            # for t in range(len(contemporaries_older_edges)):
            #     ax = plot_edges(
            #         g, contemporaries_older_edges[t], t, ax=ax,
            #         threshold_m=threshold_m,
            #         threshold_l=threshold_l,
            #     )
            # if weights:
            #     fig_suptitle = (
            #         filename
            #         + "\nOlder edges and contemporaries. for variable t2m \n"
            #         + "Type of weights: " + op
            #     )
            # else:
            #     fig_suptitle = (
            #         filename
            #         + "\nOlder edges and contemporaries. for variable t2m \n"
            #         + "distance not weighted"
            #     )

            # ax.set_title(fig_suptitle)
            # ax.set_xlabel("Time (h)")
            # ax.set_ylabel("Temperature (°C)")
            # ax.autoscale()
            # path_fig = PATH_FIG_PARENT + "older_edges_and_comtemp/"

            # if weights:
            #     name_fig = (
            #         path_fig
            #         + op
            #         +"_"+ filename[:-3] + ".png"
            #     )
            # else:
            #     name_fig = (
            #         path_fig
            #         + filename[:-3] + ".png"
            #     )
            # makedirs(path_fig, exist_ok = True)

            # plt.savefig(name_fig)
            # plt.close()
