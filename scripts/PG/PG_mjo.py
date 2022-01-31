#!/usr/bin/env python3
from os import listdir, makedirs
import numpy as np
import json
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_mjo, jsonify, to_polar
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

save_spaghetti = False
save_individual = False
save_mean = False
k_max = 5


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
PATH_SPAG_FIG = PATH_SPAGHETTI + 'plots/'
PATH_SPAG_DICT = PATH_SPAGHETTI + 'dict/'
makedirs(PATH_SPAG_FIG, exist_ok = True)
makedirs(PATH_SPAG_DICT, exist_ok = True)

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [f for f in LIST_FILENAMES if f.endswith(".txt")]


# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def main():
    weights = False
    weights_values = None
    for filename in LIST_FILENAMES:
        for smooth in [True, False]:

            # ---------------------------
            # Load and preprocess data
            # ---------------------------
            data_dict = preprocess_mjo(
                filename = filename,
                path_data = PATH_DATA,
                smooth = smooth,
            )
            members = data_dict['members']
            time = data_dict['time']

            # ---------------------------
            # Spaghetti
            # ---------------------------
            if save_spaghetti:

                # ---- Typical MJO plot ----
                fig_m, ax_m = plot_mjo_members(
                    members = members,
                    show_classes = True,
                    )

                # ------ add mean and std------
                fig_m, ax_m = plot_mjo_mean_std(
                        members = members,
                        show_classes = True,
                        polar = False,
                        show_std = False,
                        fig = fig_m,
                        ax=ax_m
                        )

                # ------ filenames ------
                name_spag = PATH_SPAG_FIG + data_dict["filename"]
                if smooth:
                    name_spag += '_smooth'

                # ------ save fig------
                name_spag += '.png'
                fig_m.savefig(name_spag)
                plt.close()

                # ------ save dict------
                for polar in [True, False]:
                    json_file = PATH_SPAG_DICT + data_dict["filename"]
                    if polar:
                        json_file += '_polar'
                        # Re-load data with polar coordinates
                        data_dict_tmp = preprocess_mjo(
                            filename = filename,
                            path_data = PATH_DATA,
                            smooth = False,
                            polar = polar,
                        )
                    else:
                        data_dict_tmp = data_dict.copy()
                    if smooth and not polar:
                        json_file += '_smooth'
                    json_file += ".json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        res = jsonify(data_dict_tmp)
                        json.dump(res, f, ensure_ascii=False, indent=4)

                # ---- Plot mean (polar and not polar) ----
                if save_mean:
                    for polar_mean in [True, False]:

                        if polar_mean:
                            members_tmp = to_polar(members)
                        else:
                            members_tmp = np.copy(members)

                        fig_m, ax_m = plot_mjo_mean_std(
                            members = members_tmp,
                            show_classes = True,
                            polar = polar_mean,
                            )

                        # ------ filenames ------
                        name_spag = (
                            PATH_SPAG_FIG + data_dict["filename"]
                            + "_mean_std"
                        )
                        if smooth:
                            name_spag += '_smooth'
                        if polar_mean:
                            name_spag += '_polarmean'

                        # ------ save fig------
                        name_spag += '.png'
                        fig_m.savefig(name_spag)
                        plt.close()

                # ---- Plot members one by one ----
                if save_individual:
                    for i in range(len(members)-1):
                        m = members[i:i+1]

                        fig_m, ax_m = plot_mjo_members(
                            members = m,
                            show_classes = True,
                            )

                        # ------ filenames ------
                        name_spag = PATH_SPAG_FIG + data_dict["filename"]
                        if smooth:
                            name_spag += '_smooth'
                        name_spag += '_' + str(i) + '.png'

                        # ------ save fig------
                        fig_m.savefig(name_spag)
                        plt.close()

                # ---- RMM1 RRM2 ----
                fig, ax = plot_members(
                    members = members,
                    time_axis = time,
                )
                fig, ax = plot_mean_std(
                    members = members,
                    time_axis = time,
                    fig = fig,
                    axs = ax,
                )

                # ------ filenames ------
                name_spag = PATH_SPAG_FIG + data_dict["filename"] + "_rmm"
                if smooth:
                    name_spag += '_smooth'
                name_spag += '.png'

                # ------ save fig------
                fig.savefig(name_spag)
                plt.close()

                # ---- Polar coordinates ----
                members_polar = to_polar(members)
                fig, ax = plot_members(
                    members = members_polar,
                    time_axis = time,
                )
                fig, ax = plot_mean_std(
                    members = members_polar,
                    time_axis = time,
                    fig = fig,
                    axs = ax,
                )

                # ------ filenames ------
                name_spag = PATH_SPAG_FIG + data_dict["filename"] + "_polar"
                if smooth:
                    name_spag += '_smooth'
                name_spag += '.png'

                # ------ save fig------
                fig.savefig(name_spag)
                plt.close()

            else:
                for score in SCORE_TYPES:
                    path_parent = PATH_FIG_ROOT + score + "/"

                    # --------------------------------------------
                    # ----- Prepare folders and paths ------------
                    # --------------------------------------------

                    path_fig = path_parent + "plots/"
                    name_fig = path_fig + data_dict["filename"]
                    makedirs(path_fig, exist_ok = True)

                    path_graph = path_parent + "graphs/"
                    makedirs(path_graph, exist_ok = True)
                    name_graph = path_graph + data_dict["filename"]

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
                            k_max = k_max,
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
                    g.save(name_graph, type='json')



if __name__ == "__main__":
    main()
