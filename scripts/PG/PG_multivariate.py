#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import json
import matplotlib.pyplot as plt


from ...Preprocessing.extraction import preprocess_meteogram, jsonify, numpify
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
var_names = ['u10', 'v10']
#var_names = ['tcwv']
is_multivariate = len(var_names) > 1

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
PATH_FIG_ROOT = (
    "/home/natacha/Documents/tmp/figs/PG/"
    + PG_TYPE + "/" + '-'.join(var_names) + '/'
    + "/entire_graph/"
)
PATH_SPAGHETTI = (
    "/home/natacha/Documents/tmp/figs/spaghetti/" + '-'.join(var_names) + '/'
)

PATH_SPAG_FIG = PATH_SPAGHETTI + 'plots/'
PATH_SPAG_DICT = PATH_SPAGHETTI + 'dict/'
makedirs(PATH_SPAG_FIG, exist_ok = True)
makedirs(PATH_SPAG_DICT, exist_ok = True)

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and fname.endswith(".nc")
]


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------



def main():

    for filename in LIST_FILENAMES:

        # ---------------------------------------
        # Load and preprocess data
        # ---------------------------------------

        data_dict = preprocess_meteogram(
            filename = filename,
            path_data = PATH_DATA,
            var_names=var_names,
            ind_time=None,
            ind_members=None,
            ind_long=[0],
            ind_lat=[0],
            multivariate = is_multivariate,
            to_standardize = False,
            )

        if not is_multivariate:
            data_dict['members'] = data_dict['members'][0]
        members = data_dict['members']


        # ---------------------------------------
        # Spaghetti
        # ---------------------------------------
        if save_spaghetti:

            fig_m, ax_m = plot_members(
                members = members,
                time_axis = data_dict['time'],
                )

            for i, ax in enumerate(ax_m.flat):
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(
                    data_dict['long_names'][i]
                    + ' (' + data_dict['units'][i] + ')'
                )
            fig_m.suptitle(data_dict["filename"])

            # fig_m, ax_m = plot_mean_std(
            #         members = members,
            #         time_axis = time,
            #         fig = fig_m,
            #         axs=ax_m
            #         )

            fig_m.savefig(PATH_SPAG_FIG + data_dict["filename"] + ".png")
            plt.close()

            json_file = PATH_SPAG_DICT + data_dict["filename"]+".json"
            with open(json_file, 'w', encoding='utf-8') as f:
                res = jsonify(data_dict)
                json.dump(res, f, ensure_ascii=False, indent=4)

        # ---------------------------------------
        # Graphs
        # ---------------------------------------
        else:

            # IGNORED: no weight used
            if PG_TYPE == 'Naive':
                weights_range = [True, False]
            else:
                weights_range = [False]

            for weights in weights_range:

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

                for score in SCORE_TYPES:

                    # --------------------------------------------
                    # ----- Prepare folders and paths ------------
                    # --------------------------------------------
                    path_parent = PATH_FIG_ROOT + score + "/"

                    path_fig = path_parent + "plots/"
                    name_fig = path_fig + filename[:-3]
                    makedirs(path_fig, exist_ok = True)

                    path_graph = path_parent + "graphs/"
                    makedirs(path_graph, exist_ok = True)
                    name_graph = path_graph + filename[:-3]

                    # ---------------------------
                    # Construct graph
                    # ---------------------------

                    g = PersistentGraph(
                            time_axis = data_dict['time'],
                            members = members,
                            weights = weights_values,
                            score_type = score,
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

                    # ax0 = None
                    # fig0 = None
                    # if show_k_plot == 'inside' or show_k_plot == 'outside':
                    #     if show_k_plot == 'inside':
                    #         fig0 = plt.figure(figsize = (25,15), tight_layout=True)
                    #         gs = fig0.add_gridspec(nrows=2, ncols=3)
                    #         ax0 = fig0.add_subplot(gs[:, 0:2])
                    #         ax1 = fig0.add_subplot(gs[0, 2], sharex=ax0)
                    #     else:
                    #         ax1 = None
                    #     fig1, ax1, _ = k_plot(g, k_max = 5, ax=ax1)
                    #     ax1_title = 'Number of clusters: relevance'
                    #     ax1.set_title(ax1_title)
                    #     ax1.set_xlabel("Time")
                    #     ax1.set_ylabel("Relevance")

                    #     if show_k_plot == 'outside':
                    #         fig1.savefig(name_fig + "_k_plots")

                    ax_kw = {
                        'xlabel' : "Time (h)",
                        'ylabel' :  [
                            data_dict['short_names'][i]
                            + ' (' + data_dict['units'][i] + ')'
                            for i in range(len(data_dict['units']))
                        ]
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
                        # fig0, ax0 = plot_as_graph(
                        #     g, show_vertices=True, show_edges=True, show_std = True,
                        #     ax_kw=ax_kw, ax = ax0, fig=fig0,
                        # )

                        # ax0_title = 'Entire graph'
                        # ax0.set_title(ax0_title)


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

if __name__ == "__main__":
    main()
