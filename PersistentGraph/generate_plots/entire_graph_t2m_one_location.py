#!/usr/bin/env python3
import sys
import os
from os import listdir, makedirs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

sys.path.append("/home/natacha/Documents/Work/python/")  # to import galib
sys.path.insert(1, os.path.join(sys.path[0], '..'))  #to use PG submodules
sys.path.insert(1, os.path.join(sys.path[0], '../..'))  #to use DA submodules

from DataAnalysis.statistics import extract_variables, standardize, get_list_std, get_list_average_values
from galib.tools.lists import get_indices_element
from galib.tools.plt import from_list_to_subplots
from persistentgraph import PersistentGraph
from plots import plot_as_graph, plot_barcodes




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
path_fig = "/home/natacha/Documents/tmp/figs/PG/t2m/entire_graph/"

# Choose which variables should be ploted
# type: List(str)
# Available variables:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”)
# if None: var_names = ["t2m","d2m","msl","u10","v10","tcwv"]
var_names=["t2m"]
# Choose which instants should be ploted
# type: ndarray(int)
ind_time=None
# Choose which members should be ploted
# type: ndarray(int)
ind_members=None
# Choose which longitude should be ploted
# type: ndarray(int)
ind_long=[0]
# Choose which latitude should be ploted
# type: ndarray(int)
ind_lat=[0]

# Choose nb members threshold
threshold = 2

# Choose which files should be used
list_filenames = listdir(path_data)
list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]

# Allow print
descr = False

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------
makedirs(path_fig, exist_ok = True)

type_op = ["std", "max_distance"]
for weights in [True, False]:
    for op in type_op:
        for filename in list_filenames:
            print(filename)
            f = path_data + filename
            nc = Dataset(f,'r')

            (list_var,list_names) = extract_variables(
                nc=nc,
                var_names=var_names,
                ind_time=ind_time,
                ind_members=ind_members,
                ind_long=ind_long,
                ind_lat=ind_lat,
                descr=descr
            )

            t2m = np.transpose(list_var[0]).squeeze()
            if weights:
                weights_file = (
                    "/home/natacha/Documents/tmp/figs/global_variation_t2m/all_forecasts_"
                    + op
                    + ".txt"
                )
                weights_values = np.loadtxt(weights_file)
            else:
                weights_values = None
            g = PersistentGraph(members=t2m, weights=weights_values)
            g.construct_graph()
            fig, ax = plot_as_graph(g, threshold=threshold)


            if weights:
                fig_suptitle = (
                    "Entire graph for variable t2m \n"
                    + "Type of weights: " + op
                )
            else:
                fig_suptitle = (
                    "Entire graph for variable t2m \n"
                    + "distance not weighted"
                )

            ax.set_title(fig_suptitle)

            if weights:
                name_fig = path_fig + op +"_"+ filename[:-3] + ".png"
            else:
                name_fig = path_fig + filename[:-3] + ".png"

            plt.savefig(name_fig)
            plt.close()
