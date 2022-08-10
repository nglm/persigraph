#!/usr/bin/env python3
#FIXME: 2020/12 Make sure it is still working after the clean-up



from os import listdir, makedirs
import numpy as np
import matplotlib.pyplot as plt

from ...Preprocessing.extraction import preprocess_meteogram
from ...utils.plt import from_list_to_pairplots

# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
path_data = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
path_fig = "/home/natacha/Documents/tmp/figs/pairplots_first_location/"

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
ind_long=[0,10,15,3,12]
#ind_long=[0]
# Choose which latitude should be ploted
# type: ndarray(int)
ind_lat=[3,8,17,4,10]
#ind_lat=[0]

# Choose which files should be used
list_filenames = listdir(path_data)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]
list_filenames = [list_filenames[0]]

# Use log for tcwv
use_log_tcwv = True

# Use standardise
use_standardise = True

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
makedirs(path_fig, exist_ok = True)
for filename in list_filenames:

    data_dict = preprocess_meteogram(
        filename = filename,
        path_data = path_data,
        var_names = var_names,
        ind_time = ind_time,
        ind_members = ind_members,
        ind_long = ind_long,
        ind_lat = ind_lat,
        to_standardize = use_standardise,
        )

    if use_standardise:
        fig_suptitle = (
            "Bergen Forecast: "
            + filename[:-3]
            + "\n 5 grid point, All members, standardized values"
        )
    else:
        fig_suptitle = (
            "Bergen Forecast: "
            + filename[:-3]
            + "\n 5 grid point, All members"
        )

    ax = from_list_to_pairplots(
        list_values = data_dict['members'],
        fig_suptitle = fig_suptitle,
        list_labels = data_dict['short_name'],
        show = False,
    )
    suffix = "_5_grid_points"
    #suffix = ""
    if use_log_tcwv:
        suffix += "_with_log"
    if use_standardise:
        suffix += "_std"

    name_fig = path_fig + filename[:-3] + suffix + ".png"
    plt.savefig(name_fig)
    plt.close()