#!/usr/bin/env python3

import sys
import os
from os import listdir, makedirs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from statistics import extract_variables, standardize, get_list_std, get_list_average_values
from utils.lists import get_indices_element
from utils.plt import from_list_to_subplots



# ---------------------------------------------------------
# Parameters:
# ---------------------------------------------------------

# Absolute path to the files
# type: str
path_data = "/home/natacha/Documents/Work/Data/Bergen/"


# Choose which variables should be ploted
# type: List(str)
# Available variables:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”)
# if None: var_names = ["t2m","d2m","msl","u10","v10","tcwv"]
var_names=["tcwv"]

# Choose the path where the figs will be saved
# type: str
path_fig = (
    "/home/natacha/Documents/tmp/figs/global_variation_"
    + var_names[0] + "/")

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
list_filenames = listdir(path_data)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
makedirs(path_fig, exist_ok = True)
type_op = ["std", "max_distance"]
for op in type_op:
    global_op_val = []
    for filename in list_filenames:
        print(filename)
        file_std = []

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
        )

        list_var = [np.swapaxes(var, 0,1) for var in list_var]
        list_var = [np.squeeze(var) for var in list_var]
        members = list_var[0]


        if ind_long is None:
            ind_long= np.arange(nc.variables["longitude"].size)
        if ind_lat is None:
            ind_lat= np.arange(nc.variables["latitude"].size)

        for i_lon in range(len(ind_long)):
            for i_lat in range(len(ind_lat)):
                if op == "std":
                    global_op_val.append(
                        np.std(members[:,:, i_lon, i_lat], axis = 0)
                    )
                elif  op == "max_distance":
                    global_op_val.append(
                        np.amax(members[:,:, i_lon, i_lat], axis = 0)
                        - np.amin(members[:,:, i_lon, i_lat], axis = 0)
                    )

    global_op_val = np.array(global_op_val)
    global_op_val = np.mean(global_op_val, axis=0)
    global_op_val_norm = global_op_val/np.amax(global_op_val)


    # Set the initial conditions at time +0h
    time = np.array(nc.variables["time"])
    time -= time[0]

    fig_suptitle = (
        "Global variation of " + op +" for variable " + var_names[0]
    )

    xlabel = "Time (h)"
    list_list_legend = [list_names]

    fig, axs = from_list_to_subplots(
        list_yvalues=np.array(global_op_val),  # List[ndarray([n_lines, ] n_values )]
        list_xvalues=time, #ndarray(n_values)
        plt_type = "plot",
        fig_suptitle = fig_suptitle,
        list_xlabels = xlabel,
        list_list_legends=list_list_legend,
        show=False,
        )

    name_fig = path_fig + "all_forecasts_" + op + ".png"
    name_file_std = path_fig + "all_forecasts_" + op + ".txt"
    name_file_weights = path_fig + "weights_" + op + ".txt"
    np.savetxt(name_file_std, global_op_val)
    np.savetxt(name_file_weights, global_op_val_norm)

    plt.savefig(name_fig)
    plt.close()
