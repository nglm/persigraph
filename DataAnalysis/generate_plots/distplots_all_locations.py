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
path_fig = "/home/natacha/Documents/tmp/figs/distplots_all_locations/"

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
list_filenames = listdir(path_data)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]
list_filenames = [list_filenames[0]]

# Allow print
descr = False

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
makedirs(path_fig, exist_ok = True)
for use_log_tcwv in [True, False]:
    for use_standardise in [True, False]:
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

            if use_log_tcwv:
                # Take the log for the tcwv variable
                idx = get_indices_element(
                    my_list=list_names,
                    my_element="tcwv",
                )
                if idx != -1:
                    for i in idx:
                        list_var[i] = np.log(list_var[i])

            if use_standardise:
                (list_scalers, list_var) = standardize(
                    list_var = list_var,
                    each_loc = False,
                )

            list_var = [np.swapaxes(var, 0,1) for var in list_var]
            list_var = [var.flatten() for var in list_var]

            if use_standardise:
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


            list_ax_titles = ["Variable: " + name for name in list_names]

            if use_standardise:
                list_xlabels = [
                    "Standardized values (1)" for name in list_names
                ]
            else:
                list_xlabels = [
                    "Values ("
                    + nc.variables[name].__dict__["units"]
                    +")" for name in list_names
                ]

            # if we used log on tcwv:
            if use_log_tcwv:
                for i in idx:
                    list_ax_titles[i] = "Variable: log(tcwv)"
                    if not use_standardise:
                        list_xlabels[i] = "Values (log("+nc.variables["tcwv"].__dict__["units"] +"))"

            kwargs = {
                "sharex" : False,
                "sharey" : False,
                "list_xlabels" : list_xlabels,
            }

            if use_standardise:
                kwargs['fit_show_mean'] = False
                kwargs['fit_show_std'] = False

            fig, axs = from_list_to_subplots(
                list_yvalues=list_var,  # List[ndarray([n_lines, ] n_values )]
                plt_type = "distplot",
                fig_suptitle = fig_suptitle,
                list_ax_titles = list_ax_titles,
                show=False,
                **kwargs,
                )
            suffix = ""
            if use_log_tcwv:
                suffix += "_with_log"
            if use_standardise:
                suffix += "_std"

            name_fig = path_fig + filename[:-3] + suffix + ".png"
            plt.savefig(name_fig)
            plt.close()