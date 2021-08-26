#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...DataAnalysis.preprocess import preprocessing_mjo
from ...PersistentGraph import PersistentGraph
from ...PersistentGraph.plots import *



# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

PG_TYPE = 'Naive'
PG_TYPE = 'KMeans'

SCORE_TYPES = [
    'inertia',
    'mean_inertia',
    'weighted_inertia',
    'max_inertia',
    #'min_inertia',       # Shouldn't be used: taking min makes no sense
    # ----------
    'variance',
    'mean_variance',
    #'weighted_variance', # Shouldn't be used: favors very high k values
    #'min_variance',      # Shouldn't be used: taking min makes no sense
    'max_variance',
    # ----------
    #'diameter',      # WARNING: diameter should be used with weights
    #'max_diameter',  # WARNING: Max diameter should be used with weights
    # ----------
    'MedDevMean',
    'mean_MedDevMean',
    'max_MedDevMean',
    # ----------
    #'max_MedDevMed', # Shouldn't be used: see details below
]
SCORE_TYPES = ['max_inertia']


ZERO_TYPE = 'bounds'

save_spaghetti = True


#FIXME: Outdated option
# Use
# - 'overview' if you want the overview plot (entire graph + k_plot +
# most relevant components)
# - 'inside' if you want graph and k_plots on the same fig
# - 'outside' if you want 2 figs
# - anything else if you just want the entire graph
show_k_plot = 'overview'

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/MJO/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_ROOT = (
    "/home/natacha/Documents/tmp/figs/PG/"
    + PG_TYPE + "/mjo/entire_graph/"
)
PATH_SPAGHETTI = (
    "/home/natacha/Documents/tmp/figs/spaghetti/mjo/"
)

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [f for f in LIST_FILENAMES if f.endswith(".txt")]


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def main():
    # FIXME: Outdated option
    # IGNORED: no weight used
    if PG_TYPE == 'Naive':
        weights_range = [True, False]
    else:
        weights_range = [False]

    for weights in [False]:
        for score in SCORE_TYPES:
            path_parent = PATH_FIG_ROOT + score + "/"
            for filename in LIST_FILENAMES:

                # --------------------------------------------
                # ----- Prepare folders and paths ------------
                # --------------------------------------------

                name_spag = PATH_SPAGHETTI + filename[:-3] + ".png"
                makedirs(PATH_SPAGHETTI, exist_ok = True)

                path_fig = path_parent + "plots/"
                name_fig = path_fig + filename[:-3]
                makedirs(path_fig, exist_ok = True)

                path_graph = path_parent + "graphs/"
                makedirs(path_graph, exist_ok = True)
                name_graph = path_graph + filename[:-3]

                # ---------------------------
                # Load and preprocess data
                # ---------------------------
                f = PATH_DATA + filename

                members, time = preprocessing_mjo(filename = f)

                if weights:
                    pass
                    # weights_file = (
                    #     "/home/natacha/Documents/tmp/figs/global_variation_"
                    #     + var_names[0] +'/'
                    #     + "all_forecasts_max_distance.txt"
                    # )
                    # weights_values = np.loadtxt(weights_file)
                else:
                    weights_values = None

                # ---------------------------
                # Construct graph
                # ---------------------------

                g = PersistentGraph(
                        time_axis = time,
                        members = members,
                        weights = weights_values,
                        score_type = score,
                        zero_type = ZERO_TYPE,
                        model_type = PG_TYPE,
                        k_max = 8,
                )
                g.construct_graph(verbose=True)

                ax_kw = {
                    'xlabel' : "Time (h)",
                    'ylabel' :  'Values'
                    }

                fig_suptitle = filename

                # If overview:
                if show_k_plot == 'overview':
                    fig0, ax0 = plot_overview(
                        g, ax_kw=ax_kw, axs = None, fig= None,
                    )
                    name_fig += '_overview'

                else:
                    pass


                if weights:
                    pass
                    # fig_suptitle += ", with weights"
                    # name_fig += '_weights'
                fig0.suptitle(fig_suptitle)

                # ---------------------------
                # Save plot and graph
                # ---------------------------.
                name_fig += '.png'
                fig0.savefig(name_fig)
                plt.close()
                g.save(name_graph)

                if save_spaghetti:
                    fig_m, ax_m = plot_members(g)
                    fig_m.savefig(name_spag)
                    plt.close()

if __name__ == "__main__":
    main()
