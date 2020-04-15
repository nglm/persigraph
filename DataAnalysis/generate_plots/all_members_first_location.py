#!/usr/bin/env python3
import sys
import os
from os import listdir, makedirs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

sys.path.append("/home/natacha/Documents/Work/")  # to import galib
sys.path.insert(1, os.path.join(sys.path[0], '..'))  #to use DataAnalysis submodules

from statistics import extract_variables, standardize
from galib.tools.lists import get_indices_element
from galib.tools.plt import from_list_to_subplots



# =========================================================
# Plot members one location
# with log and standardisation)
# =========================================================

# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
path_data = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
path_fig = "/home/natacha/Documents/tmp/figs/all_members_first_location/"

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
list_filenames = listdir(path_data)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]

# Allow print
descr = False

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
makedirs(path_fig, exist_ok = True)
for filename in list_filenames:

    print(filename)
    f = path_data + filename
    nc = Dataset(f,'r')

    # Extract the data, by default:
    # - All variables
    # - Entire time series
    # - All members
    # - One location
    (list_var,list_names) = extract_variables(
        nc=nc,
        var_names=var_names,
        ind_time=ind_time,
        ind_members=ind_members,
        ind_long=ind_long,
        ind_lat=ind_lat,
        descr=descr
    )

    # Take the log for the tcwv variable
    idx = get_indices_element(
        my_list=list_names,
        my_element="tcwv",
    )
    if idx != -1:
        for i in idx:
            list_var[i] = np.log(list_var[i])


    (list_scalers, list_stand_var) = standardize(
        list_var = list_var,
        each_loc = False,
    )

    list_stand_var = [np.swapaxes(var, 0,1) for var in list_stand_var]
    list_stand_var = [np.squeeze(var) for var in list_stand_var]

    # Set the initial conditions at time +0h
    time = np.array(nc.variables["time"])
    time -= time[0]

    fig_suptitle = "Bergen Forecast: " + filename[:-3] + "\n First grid point, All members"
    list_ax_titles = ["Variable: " + name for name in list_names]
    xlabel = "Time (h)"
    ylabel = "Standardized values (1)"

    fig, axs = from_list_to_subplots(
        list_yvalues=list_stand_var,  # List[ndarray([n_lines, ] n_values )]
        list_xvalues=time, #ndarray(n_values)
        plt_type = "plot",
        fig_suptitle = fig_suptitle,
        list_ax_titles = list_ax_titles,
        list_xlabels = xlabel,
        list_ylabels = ylabel,
        show=False,
        )

    name_fig = path_fig + filename[:-3] + ".png"
    plt.savefig(name_fig)
    plt.close()