#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from ...DataAnalysis.statistics import preprocess_data
from ...PersistentGraph import PersistentGraph
from ...PersistentGraph.plots import *



# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

PG_TYPE = 'Naive'
#PG_TYPE = 'KMeans'

SCORE_TYPE = 'max_diameter'

ZERO_TYPE = 'bounds'

var_names = ['t2m']

# Use
# - 'overview' if you want the overview plot (entire graph + k_plot +
# most relevant components)
# - 'inside' if you want graph and k_plots on the same fig
# - 'outside' if you want 2 figs
# - anything else if you just want the entire graph
show_k_plot = 'overview'

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = (
    "/home/natacha/Documents/tmp/figs/PG/"
    + PG_TYPE + "/" + str(var_names[0])
    + "/entire_graph/" + SCORE_TYPE + "/"
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
    if PG_TYPE == 'Naive':
        weights_range = [True, False]
    else:
        weights_range = [False]
    for weights in weights_range:
        for filename in LIST_FILENAMES:

            # --------------------------------------------
            # ----- Prepare folders and paths ------------
            # --------------------------------------------

            path_fig = PATH_FIG_PARENT + "plots/"
            name_fig = path_fig + filename[:-3]
            makedirs(path_fig, exist_ok = True)

            path_graph = PATH_FIG_PARENT + "graphs/"
            makedirs(path_graph, exist_ok = True)
            name_graph = path_graph + filename[:-3]

            # ---------------------------
            # Load and preprocess data
            # ---------------------------


            # To get the right variable names and units
            nc = Dataset(PATH_DATA + filename,'r')

            list_var, list_names, time = preprocess_data(
                filename = filename,
                path_data = PATH_DATA,
                var_names=var_names,
                ind_time=None,
                ind_members=None,
                ind_long=[0],
                ind_lat=[0],
                to_standardize = False,
                )

            members = list_var[0]

            if weights:
                weights_file = (
                    "/home/natacha/Documents/tmp/figs/global_variation_"
                    + var_names[0] +'/'
                    + "all_forecasts_max_distance.txt"
                )
                weights_values = np.loadtxt(weights_file)
            else:
                weights_values = None

            # ---------------------------
            # Construct graph
            # ---------------------------

            g = PersistentGraph(
                    time_axis = time,
                    members = members,
                    weights = weights_values,
                    score_type = SCORE_TYPE,
                    zero_type = ZERO_TYPE,
                    model_type = PG_TYPE,
                    k_max = 8,
            )
            g.construct_graph(
                verbose=True,
            )

            # ---------------------------------
            # Plot entire graph (with k_plot)
            # ---------------------------------

            ax0 = None
            fig0 = None
            if show_k_plot == 'inside' or show_k_plot == 'outside':
                if show_k_plot == 'inside':
                    fig0 = plt.figure(figsize = (25,15), tight_layout=True)
                    gs = fig0.add_gridspec(nrows=2, ncols=3)
                    ax0 = fig0.add_subplot(gs[:, 0:2])
                    ax1 = fig0.add_subplot(gs[0, 2], sharex=ax0)
                else:
                    ax1 = None
                fig1, ax1, _ = k_plot(g, k_max = 5, ax=ax1)
                ax1_title = 'Number of clusters: relevance'
                ax1.set_title(ax1_title)
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Relevance")

                if show_k_plot == 'outside':
                    fig1.savefig(name_fig + "_k_plots")

            ax_kw = {
                'xlabel' : "Time (h)",
                'ylabel' :  (
                    nc.variables[var_names[0]].long_name
                    + ' (' + nc.variables[var_names[0]].units + ')'
                )
                }

            fig_suptitle = (filename + "\n" +str(var_names[0]))

            # If overview:
            if show_k_plot == 'overview':
                fig0, ax0 = plot_overview(
                    g, k_max=8, show_vertices=True, show_edges=True,
                    show_std = True, ax_kw=ax_kw, ax = ax0, fig=fig0,
                )
                name_fig += '_overview'

            else:
                fig0, ax0 = plot_as_graph(
                    g, show_vertices=True, show_edges=True, show_std = True,
                    ax_kw=ax_kw, ax = ax0, fig=fig0,
                )

                ax0_title = 'Entire graph'
                ax0.set_title(ax0_title)


            if weights:
                fig_suptitle += ", with weights"
                name_fig += '_weights'
            fig0.suptitle(fig_suptitle)


            # ---------------------------
            # Save plot and graph
            # ---------------------------.
            name_fig += '.png'
            fig0.savefig(name_fig)
            plt.close()
            g.save(name_graph)

if __name__ == "__main__":
    main()
