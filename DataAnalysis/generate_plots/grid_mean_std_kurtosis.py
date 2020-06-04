import sys
import os
from os import listdir, makedirs
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

sys.path.append("/home/natacha/Documents/Work/")  # to import galib
sys.path.insert(1, os.path.join(sys.path[0], '..'))  #to use DataAnalysis submodules

from statistics import extract_variables, standardize
from galib.tools.lists import get_indices_element
from galib.tools.plt import pretty_subplots

import cartopy.crs as ccrs
import cartopy
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator
# from matplotlib.widgets import Slider



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
path_fig_parent = "/home/natacha/Documents/tmp/figs/grid_mean_std_kurtosis/"

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
ind_long=None
# Choose which latitude should be ploted
# type: ndarray(int)
ind_lat=None


# Choose the center of the map to plot
central_lon, central_lat = 0, 60
# Choose the extent of the map
# Choose the projection
nsp_trans = ccrs.NearsidePerspective()
# Used to transform coordinates
geodetic = ccrs.Geodetic()


# Choose which files should be used
#list_filenames = listdir(path_data)
# list_filenames = [fname for fname in list_filenames if fname.startswith("ec.ens.") and  fname.endswith(".nc")]
list_filenames = [
    "ec.ens.2020012900.sfc.meteogram.nc",
    "ec.ens.2020021012.sfc.meteogram.nc",
]

# Allow print
descr = False

# ---------------------------------------------------------
# script:
# ---------------------------------------------------------

for use_log_tcwv in [False]:
    for use_standardise in [True, False]:
        for filename in list_filenames:
            for type_plot in ["mean", "std", "kurtosis"]:

                path_fig = path_fig_parent + type_plot + "/"
                makedirs(path_fig, exist_ok = True)

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

                lon = np.flip(np.array(nc.variables["longitude"]))
                lat = np.flip(np.array(nc.variables["latitude"]))
                time = np.array(nc.variables["time"])
                time -= time[0]

                lon_0 = np.amin(lon)
                lon_1 = np.amax(lon)
                lat_0 = np.amin(lat)
                lat_1 = np.amax(lat)
                extent = [lon_0-5, lon_1+5, lat_0-2, lat_1]

                # Transform our coordinates
                lon2d,lat2d = np.meshgrid(lon, lat)

                trans_lon, trans_lat = np.zeros_like(lon2d), np.zeros_like(lat2d)
                for i, lon_i in enumerate(lon):
                #for i, lon_i in enumerate(np.flip(lon)):
                    for j, lat_j in enumerate(lat):
                    #for j, lat_j in enumerate(lat):
                        trans_lon[i,j], trans_lat[i,j] = nsp_trans.transform_point(lon_i, lat_j, geodetic)

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

                list_ax_titles = [type_plot + ", variable: " + name for name in list_names]

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

                if type_plot == "mean":
                    var = np.mean(list_var[0], axis = 1)
                elif type_plot == "std":
                    var = np.std(list_var[0], axis = 1)
                else:
                    var = kurtosis(list_var[0], axis = 1)
                vmin = np.amin(var)
                vmax = np.amax(var)

                # Draw the maps
                for t in range(len(time)):
                    if use_standardise:
                        fig_suptitle = (
                            "Bergen Forecast: "
                            + filename[:-3]
                            + "\n hours +"
                            + str(t*6)
                            + "\n standardized values"
                        )
                    else:
                        fig_suptitle = (
                            "Bergen Forecast: "
                            + filename[:-3]
                            + "\n hours +"
                            + str(t*6)
                        )
                    kwargs = {
                        "list_xlabels" : list_xlabels,
                        "fig_suptitle" : fig_suptitle,
                        "list_ax_titles" : list_ax_titles,
                    }
                    fig, axs = pretty_subplots(
                        **kwargs,
                    )
                    ax = plt.axes(
                        projection=ccrs.NearsidePerspective(
                            central_longitude=central_lon,
                            central_latitude=central_lat,
                        )
                    )
                    ax.set_extent(extent)

                    #ax.add_feature(cartopy.feature.OCEAN) # Add oceans
                    ax.gridlines()                        # Add gridlines
                    ax.coastlines(resolution='50m')       # Add coastlines
                    cmap = plt.get_cmap('RdBu_r')


                    cf = ax.pcolormesh(
                        trans_lon, trans_lat, var[t],
                        transform=nsp_trans, cmap=cmap,
                        vmin=vmin, vmax=vmax)

                    fig.colorbar(cf, ax=ax)

                    suffix = ""
                    if use_log_tcwv:
                        suffix += "_with_log"
                    if use_standardise:
                        suffix += "_std"
                    suffix += "_" + str(t)

                    name_fig = path_fig + filename[:-3] + suffix + ".png"
                    plt.savefig(name_fig)
                    plt.close()
