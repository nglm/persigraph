#!/usr/bin/env python3


from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_meteogram, moving_average
from ...utils.plt import from_list_to_subplots


# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Choose which variables should be ploted
# type: List(str)
# Available variables:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”)
# if None: var_names = ["t2m","d2m","msl","u10","v10","tcwv"]
var_names=['tcwv']

# Absolute path to the files
# type: str
PATH_DATA= "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG = "/home/natacha/Documents/tmp/figs/moving_avg_first_location/"

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

# Choose which time window should be applied
list_windows = [1, 2 , 4, 6, 8, 10]

to_standardize = False

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

    list_list_avg_var = moving_average(
        list_var = data_dict["members"],
        list_windows = list_windows,
    )

    # One variable at a time
    for i,list_avg_var in enumerate(list_list_avg_var):

        fig_suptitle = (
            "Bergen Forecast: " + filename[:-3]
            + "\n Variable: " + data_dict["short_names"][i]
            + " First grid point, All members"
            )
        list_ax_titles = [
            "Moving average \n time_window = "
            + str(t) +" times steps ie " + str(t*6) + "h" for t in list_windows]
        xlabel = "Time (h)"
        if to_standardize:
            ylabel = "Standardized values (1)"
        else:
            ylabel = (
            nc.variables[var_names[0]].long_name
            + ' (' + nc.variables[var_names[0]].units + ')'
        )


        dict_kwargs = {
            "fig_suptitle" : fig_suptitle,
            "list_ax_titles" : list_ax_titles,
            "list_xlabels" : xlabel,
            "list_ylabels" : ylabel,
        }

        fig, axs = from_list_to_subplots(
            list_yvalues=list_avg_var,  # List[ndarray([n_lines, ] n_values )]
            list_xvalues=time, #ndarray(n_values)
            plt_type = "plot",
            dict_kwargs = dict_kwargs,
            show=False,
        )

        path_fig_tmp = PATH_FIG+ list_names[i] +"/"
        makedirs(path_fig_tmp, exist_ok = True)
        name_fig = path_fig_tmp + filename[:-3] + ".png"
        plt.savefig(name_fig)
        plt.close()