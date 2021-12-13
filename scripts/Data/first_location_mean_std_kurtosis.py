#!/usr/bin/env python3

#FIXME: 2020/12 Make sure it is still working after the clean-up



from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_meteogram, get_list_stats
from ...utils.plt import from_list_to_subplots

# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/first_location_std_mean_kurtosis/"

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
ind_long=np.array([0])
# Choose which latitude should be ploted
# type: ndarray(int)
ind_lat=np.array([0])

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and  fname.endswith(".nc")
]

list_type_plots = ["mean", "std", "kurtosis"]

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------

for use_log_tcwv in [False]:
    for use_standardise in [True]:
        for filename in LIST_FILENAMES:

            data_dict = preprocess_meteogram(
                filename = filename,
                path_data = PATH_DATA,
                var_names = var_names,
                ind_time = ind_time,
                ind_members = ind_members,
                ind_long = ind_long,
                ind_lat = ind_lat,
                to_standardize = use_standardise,
                )

            list_stats = get_list_stats(
                list_values=data_dict['members'],
            )

            for i_plot, type_plot in enumerate(list_type_plots):

                path_fig = PATH_FIG_PARENT + type_plot + "/"
                makedirs(path_fig, exist_ok = True)
                fig_suptitle = (
                    "Bergen Forecast: "
                    + filename[:-3]
                    + "\n First grid point, "
                    + type_plot
                )
                list_ax_titles = None
                list_ylabels = type_plot

                list_xlabels = ["Time (h)"]
                list_list_legends = data_dict['short_name']

                dict_kwargs = {
                    "fig_suptitle" : fig_suptitle,
                    "list_ax_titles" : list_ax_titles,
                    "list_xlabels" : list_xlabels,
                    "list_ylabels" : list_ylabels,
                    "list_list_legends" : list_list_legends,
                }

                fig, axs = from_list_to_subplots(
                    list_yvalues=list_stats[i_plot],
                    list_xvalues=data_dict['time'],
                    plt_type = "plot",
                    dict_kwargs = dict_kwargs,
                    show=False,
                    )

                name_fig = path_fig + filename[:-3] + ".png"
                plt.savefig(name_fig)
                plt.close()