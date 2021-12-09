#!/usr/bin/env python3

#FIXME: 2020/12 Make sure it is still working after the clean-up



from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_meteogram
from ...utils.plt import from_list_to_subplots

# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG = "/home/natacha/Documents/tmp/figs/distplots_all_locations/"

# Choose which variables should be ploted
# type: List(str)
# Available variables:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”)
# if None: var_names = ["t2m","d2m","msl","u10","v10","tcwv"]
var_names=None
# Choose which instants should be ploted
# type: ndarray(int)
ind_time=None
# Choose which members should be ploted
# type: ndarray(int)
ind_members=None
# Choose which longitude should be ploted
# type: ndarray(int)
ind_long=None
# Choose which latitude should be ploted
# type: ndarray(int)
ind_lat=None

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and  fname.endswith(".nc")
]
LIST_FILENAMES = [LIST_FILENAMES[0]]


for use_log_tcwv in [False, False]:
    for to_standardize in [True, False]:
        for filename in LIST_FILENAMES:

            data_dict = preprocess_meteogram(
                filename = filename,
                path_data = PATH_DATA,
                var_names = var_names,
                ind_time = ind_time,
                ind_members = ind_members,
                ind_long = ind_long,
                ind_lat = ind_lat,
                to_standardize = to_standardize,
                )

            if to_standardize:
                fig_suptitle = (
                    "Bergen Forecast: "
                    + filename[:-3]
                    + "\n All grid points, All members, standardized values"
                )
            else:
                fig_suptitle = (
                    "Bergen Forecast: "
                    + filename[:-3]
                    + "\n All grid points, All members"
                )


            list_ax_titles = ["Variable: " + name for name in data_dict['short_names']]

            if to_standardize:
                list_xlabels = [
                    "Standardized values (1)" for _ in data_dict['short_names']
                ]
            else:
                list_xlabels = [""]

            kwargs = {
                "sharex" : False,
                "sharey" : False,
                "list_xlabels" : list_xlabels,
                "fig_suptitle" : fig_suptitle,
                "list_ax_titles" : list_ax_titles,
            }

            if to_standardize:
                kwargs['fit_show_mean'] = False
                kwargs['fit_show_std'] = False

            fig, axs = from_list_to_subplots(
                list_yvalues = data_dict['members'],
                plt_type = "distplot",
                show=False,
                **kwargs,
                )
            suffix = ""
            if use_log_tcwv:
                suffix += "_with_log"
            if to_standardize:
                suffix += "_std"

            name_fig = PATH_FIG + filename[:-3] + suffix + ".png"
            plt.savefig(name_fig)
            plt.close()