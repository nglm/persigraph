#!/usr/bin/env python3

#FIXME: 2020/12 Make sure it is still working after the clean-up

from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_meteogram, get_list_std, get_list_average_values
from ...utils.plt import from_list_to_subplots


# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG = "/home/natacha/Documents/tmp/figs/avg_std_between_members_all_locations/"

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

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
to_standardize = True

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and  fname.endswith(".nc")
]
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

    list_std = get_list_std(
        list_var=data_dict['members'],
    )

    list_std = get_list_average_values(
        list_values=list_std,
    )

    fig_suptitle = (
        "Bergen Forecast: "
        + filename[:-3]
        + "\n Standard deviation between all members - averaged over all locations"
    )
    list_ax_titles = ["Variable: " + name for name in data_dict['short_name']]

    xlabel = "Time (h)"
    ylabel = "Standard deviation (on standardized values (1))"
    list_list_legend = [data_dict['short_name']]

    dict_kwargs = {
            "fig_suptitle" : fig_suptitle,
            "list_ax_titles" : list_ax_titles,
            "list_xlabels" : xlabel,
            "list_ylabels" : ylabel,
        }

    fig, axs = from_list_to_subplots(
        list_yvalues = np.array(list_std),
        list_xvalues = data_dict["time"],
        plt_type = "plot",
        dict_kwargs = dict_kwargs,
        show=False,
        )

    name_fig = PATH_FIG + filename[:-3] + ".png"
    plt.savefig(name_fig)
    plt.close()