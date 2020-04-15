#!/usr/bin/env python3

# 20x20° area around and upstream of Bergen
# (50-70°N, 10°W-10°E) every 12 hours
# i.e 12 hours between each file
# The data has 6-hourly temporal resolution out to 15 days ahead.
# I am downloading the following parameters:
# --- 2 metre temperature (“t2m”),
# --- 2m-Dew point temperature (“d2m”),
# --- Mean sea-level pressure (“msl”),
# --- 10m-winds in East and North direction (“u10”, “v10”)
# --- total water vapour in the entire column above the grid point (“tcwv”).
# There are number = 50 members
# Length of the time series: 51
#
# So d=6, p=21, q=21, T=51, N=51
# So each file contains an array of dim = 6*21*21*51*50 = 6 747 300

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from galib.tools.plt import get_nrows_ncols_from_nplots, get_subplot_indices
from galib.tools.npy import running_mean


def moving_average(
    list_var,   # List(ndarray(n_time, n_members [, n_long, n_lat])
    list_windows,
):
# Source:
# https://www.quora.com/How-do-I-perform-moving-average-in-Python
    # Create a figure canvas and plot the original, noisy data
    # Compute moving averages using different window sizes
    n_windows = len(list_windows)
    n_dim = len(list_var[0].shape)
    if n_dim == 2:
        (n_time, n_members) = list_var[0].shape
    elif n_dim == 4:
        (n_time, n_members, n_long, n_lat) = list_var[0].shape
    list_list_var_mov_avg = []
    for var in list_var:
        list_var_avg = []
        for i, window in enumerate(list_windows):
            list_var_avg.append(np.copy(var))
            for i_members in range(n_members):
                if n_dim > 2:
                    for i_long in range(n_long):
                        for i_lat in range(n_lat):
                            vect = var[:,
                                       i_members,
                                       i_long,
                                       i_lat
                                       ]
                            list_var_avg[i][window-1:,
                                            i_members,
                                            i_long,
                                            i_lat
                                            ] = running_mean(vect, window)
                else:
                    vect = var[:,i_members]
                    list_var_avg[i][window-1:,i_members] = running_mean(vect, window)
        list_list_var_mov_avg.append(list_var_avg)
        # return  # List(List(ndarray(n_time, n_members [, n_long, n_lat])))
    return(list_list_var_mov_avg)

# =================================================
# Raw statistics on the data:
# - dimension
# - distribution
# =================================================



def extract_variables(
    nc,
    var_names=None,
    ind_time=None,
    ind_members=None,
    ind_long=None,
    ind_lat=None,
    descr: bool = False,
):
    """
    Extract given variables and corresponding columns

    The objective is to extract quickly usefull variables from
    all our nc files

    :param nc: Dataset (nc file) from which values are extracted
    :type nc: Dataset
    :param var_names:

        Names of the variables to extract,
        defaults to None, in this case
        ``` var_names =["t2m","d2m","msl","u10","v10","tcwv"]```

    :type var_names: List[string], optional
    :param ind_members:

        Index of the members to extract,
        defaults to None, in this case all the members are extracted

    :type ind_members: ndarray[int], optional
    :param descr:

        If True var dimensions are printed, defaults to False

    :type descr: bool, optional
    """
    if var_names is None:
        var_names =["t2m","d2m","msl","u10","v10","tcwv"]
    if ind_time is None:
        ind_time = np.arange(nc.variables["time"].size)
    if ind_members is None:
        ind_members = np.arange(nc.variables["number"].size)
    if ind_long is None:
        ind_long = np.arange(nc.variables["longitude"].size)
    if ind_lat is None:
        ind_lat = np.arange(nc.variables["latitude"].size)
    list_var = [np.array(nc.variables[name]) for name in var_names]
    list_var = [var[ind_time,:,:,:] for var in list_var]
    list_var = [var[:,ind_members,:,:] for var in list_var]
    list_var = [var[:,:,ind_long,:] for var in list_var]
    list_var = [var[:,:,:,ind_lat] for var in list_var]

    if descr:
        d = len(list_var)
        print("Total number of variables: ", d)
        for i in range(d):
            print("Variable: ", var_names[i], "Dimensions:", list_var[i].shape)
    return(list_var, var_names)
# (list_var, var_names) = extract_variables(nc)

# Extract each variable to study its dimension, distribution; etc
def extract_var_distrib(
    list_var,
    descr: bool = False,
    list_names = None
):
    """Flatten all columns for each variable

    The objective is to facilitate the study of the distribution of
    each variable.

    :param list_var:

        List of variables whose columns are to be flattened

    :type list_var: [type]
    :param list_names:

        List of names of the variables. Used only if descr=True,
        defaults to None, In that case each variable is referred as
        "var_i"

    :type list_names: [type], optional
    :param descr: [description], defaults to False
    :type descr: bool, optional
    """
    d = len(list_var)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(d)])
    var_distrib = np.array([var.flatten() for var in list_var])

    if descr:
        print("Total number of variables: ", d)
        for i in range(d):
            print("Variable: ", list_names[i], "Dimensions:", var_distrib[i].shape)
    return(var_distrib)
# var_distrib = extract_var_distrib(list_var, list_names)





# Plot the violin plot of the data
# Create one subplot for each row of data
def plot_violinplots(
    var_distrib,
    nrows:int = None,
    ncols:int = None,
    list_names=None,
    show=True,
):
    """Plot the violinplots of the data

    The objective is to get a better idea of the distribution of
    the different without doing any assumptions (not assuming that
    the values are normally distributed for instance)

    :param data: [description]
    :type data: [type]
    :param list_names: [description], defaults to None
    :type list_names: [type], optional
    :param nrows: [description], defaults to 1
    :type nrows: int, optional
    """
    if len(var_distrib.shape) == 1:
        # if we study only one variable
        d = 1
    else:
        d = len(var_distrib)
    (nrows, ncols) = get_nrows_ncols_from_nplots(d, nrows=nrows, ncols=ncols)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(d)])

    # Ploting violin plots, Source: Matplotlib examples
    # https://matplotlib.org/gallery/statistics/boxplot_vs_violin.html#sphx-glr-gallery-statistics-boxplot-vs-violin-py
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20))
    for i in range(d):
        idx = get_subplot_indices(i, ncols)
        axs[idx].violinplot(var_distrib[i])
        axs[idx].set_title(list_names[i])
    if show:
        plt.show()
    return fig, axs
# plot_violinplots(var_distrib, list_names=var_names, nrows=2, ncols=3)


# Sur ot


# For the entire grid, plot the value of one variable at a given instant

def pair_plots(
    var_distrib,
    same_fig:bool = True,
    nrows:int = None,
    ncols:int = None,
    list_names=None,
    show=True,
):
    # Plot the pairplots
    d = len(var_distrib)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(d)])
    (nrows, ncols) = get_nrows_ncols_from_nplots(d, nrows=nrows, ncols=ncols)

    # For each figure i, one subplot for the pair (i,j)
    for i in range(d-1):
        if same_fig:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20))
        for j in range(i+1,d):
            if same_fig:
                # One figure i for all the pair plots (i,_)
                k=j-i-1
                if ncols == 1:
                    # len(axs.shape)=1 if ncols=1
                    idx = k
                else:
                    idx = (int(k/ncols), k % ncols)
                axs[idx].scatter(var_distrib[i], var_distrib[j])
                axs[idx].set_title(list_names[i]+ " - " +list_names[j])
            else:
                # One pair plot for each var_i, var_j with j>i
                fig, axs = plt.subplots(figsize=(36, 20))
                axs.scatter(var_distrib[i], var_distrib[j])
                axs.set_title(list_names[i]+ " - " +list_names[j])
    if show:
        plt.show()
    return fig, axs

# extract only one member for each variable
# (list_var, var_names) = extract_variables(nc,ind_members=np.arange(1))
# var_distrib = extract_var_distrib(list_var, var_names)
# d = len(var_distrib)

# nrows=2
# ncols=3


# pair_plots()


# # Did it to show if the relation was more of an exponential
# fig, axs = plt.subplots(figsize=(36, 20))
# axs.scatter(var_distrib[0], np.log(var_distrib[5]))
# axs.set_title(var_names[0]+ " - log(" +var_names[5] +")")

# # Same but with a sqrt.
# fig, axs = plt.subplots(figsize=(36, 20))
# axs.scatter(var_distrib[0], np.sqrt(var_distrib[5]))
# axs.set_title(var_names[0]+ " - sqrt(" +var_names[5] +")")



def plot_members_one_location(
    list_var,   # list_var: list of ndarray(n_members, n_time)
    same_fig:bool = False,
    nrows:int = None,
    ncols:int = None,
    list_names=None,
    time=None,
    show=True,
):
    nvar = len(list_var)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(nvar)])
    if same_fig:
        (nrows, ncols) = get_nrows_ncols_from_nplots(nvar, nrows=nrows, ncols=ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20))
    # i goes through variables
    for i in range(nvar):
        (n_members, n_time) = list_var[i].shape
        if time is None:
            time = np.arange(n_time)
        if same_fig:
            if ncols == 1 or nrows == 1 :
                # len(axs.shape)=1 if ncols=1
                idx = i
            else:
                idx = (int(i/ncols), i % ncols)
            for j in range(n_members):
                axs[idx].plot(time, list_var[i][j])
            axs[idx].set_title(list_names[i])
        else:
            fig, axs = plt.subplots(figsize=(36, 20))
            for j in range(n_members):
                axs.plot(time, list_var[i][j])
            axs.set_title(list_names[i])
    if show:
        plt.show()
    return(fig, axs)

def get_list_spread(
    list_var,  # list_var: list of ndarray(n_members, n_time,  [n_long, n_lat])
    arg_spread: bool = False  # Should return spread or arg of spread?
):
    list_spread=[]
    # find min/max (or argmin/max) values for each variable
    for var in list_var:
        if arg_spread:
            arg_amin = np.argmin(var,axis=0)
            arg_amax = np.argmax(var,axis=0)
            list_spread.append((arg_amin,arg_amax))
        else:
            amin = np.squeeze(np.amin(var,axis=0, keepdims=True))
            amax = np.squeeze(np.amax(var,axis=0, keepdims=True))
            list_spread.append((amin,amax))
    # return List[(
        # ndarray(n_time, [n_long, n_lat]),
        # ndarray(n_time, [n_long, n_lat])
        # )]
    return(list_spread)

def get_list_mean_spread(
    list_spread,
):
    #list_spread: List[(
    # ndarray(n_time, n_long, n_lat),
    # ndarray(n_time, n_long, n_lat)
    # )]
    list_mean_spread = []
    for (amin, amax) in list_spread:
        spread = np.array(amax - amin)
        mean_spread = np.mean(spread, axis=(1,2))
        list_mean_spread.append(mean_spread)
    #list_mean_spread: List[ndarray(n_time)]
    return list_mean_spread


def get_list_std(
    list_var,  # list_var: list of ndarray(n_members, n_time,  [n_long, n_lat])
):
    list_std=[]
    # find std values for each variable at each time step
    for var in list_var:
        std = np.squeeze(np.std(var,axis=0, keepdims=True))
        list_std.append(std)
    # return List[ndarray(n_time, [n_long, n_lat])]
    return(list_std)

def get_list_average_values(
    list_values,  # List[ndarray(n_time, n_long, n_lat)]
):
    list_average_values = []
    for values in list_values:
        average_values = np.mean(values, axis=(1,2))
        list_average_values.append(average_values)
    #list_average_values: List[ndarray(n_time)]
    return list_average_values


def plot_list_time_series(
    list_time_series,  # List[ndarray(n_time)]
    list_names=None,
    time=None,
    show=True,
):
    nvar = len(list_time_series)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(nvar)])
    fig, axs = plt.subplots(figsize=(36, 20))
    for mean_spread in list_time_series:
        n_time = mean_spread.shape[0]
        if time is None:
            time = np.arange(n_time)
        axs.plot(time, mean_spread)
    if show:
        plt.show()
    return fig, axs

def plot_spread_one_location(
    list_spread,
    list_names=None,
    time=None,
    show=True,
):
    #list_spread: List[(
    # ndarray(n_time),
    # ndarray(n_time)
    # )]
    nvar = len(list_spread)
    if list_names is None:
        list_names = np.array(["var_"+str(i) for i in range(nvar)])
    fig, axs = plt.subplots(figsize=(36, 20))
    for (amin,amax) in list_spread:
        n_time = amin.shape[0]
        if time is None:
            time = np.arange(n_time)
        axs.plot(time, amax - amin)
    if show:
        plt.show()
    return fig, axs


# =========================================================
# Standardize the data (after using log)
# =========================================================


def standardize(
    list_var,  # List[ndarray(n_time, n_members, n_long, n_lat)]
    each_loc: bool = False, # if true return List[ndarray(n_long, n_lat)] else return List[Scaler]
):
    list_scalers = []
    list_stand_var = []
    for var in list_var:
        if len(var.shape) > 2:
            (n_time, n_members, n_long, n_lat) = var.shape
        else:
            (n_time, n_members) = var.shape
            each_loc = False
        if each_loc:
            scalers = np.empty((n_long, n_lat))
            stand_var = np.zeros_like(var)
            for i in range(n_long):
                for j in range(n_lat):
                    values_i_j = var[:,:,i,j].flatten()
                    scaler_i_j = StandardScaler()
                    scaler_i_j.fit(values_i_j)
                    scalers[i,j] = scaler_i_j
                    stand_var[:,:,i,j] = scaler_i_j.transform(var[:,:,i,j])
            list_scalers.append(scalers)
            list_stand_var.append(stand_var)
        else:
            values = np.transpose(var.reshape((1,-1)))
            scaler =  StandardScaler()
            scaler.fit(values)
            list_scalers.append(scaler)
            stand_var = np.zeros_like(var)
            if len(var.shape) > 2:
                for i in range(n_long):
                    for j in range(n_lat):
                        stand_var[:,:,i,j] = scaler.transform(var[:,:,i,j])
            else:
                stand_var = scaler.transform(var)
            list_stand_var.append(stand_var)
    return (list_scalers, list_stand_var)

# =========================================================
# Plot spread one location
# (again but with log and standardisation)
# =========================================================

# (list_var,list_names) = extract_variables(
#     nc,
#     var_names=None,
#     ind_time=None,
#     ind_members=None,
#     ind_long=np.array([0]),
#     ind_lat=np.array([0]),
#     descr=True
# )

# list_var[5] = np.log(list_var[5])
# list_var

# for var in list_var:
#     print(var.shape)
#     print(np.squeeze(var).shape)

# (list_scalers, list_stand_var) = standardize(
#     list_var = list_var,
#     each_loc = False,
# )

# list_stand_var = [np.transpose(var) for var in list_stand_var]
# time = np.array(nc.variables["time"])
# time -= time[0]

# for var in list_stand_var:
#     print(var.shape)

# plot_members_one_location(
#     list_var=np.squeeze(list_stand_var),
#     list_names=list_names,
#     time=time,
#     same_fig=True
# )

# list_spread = get_list_spread(
#     list_var=np.squeeze(list_stand_var),
# )

# plot_spread_one_location(
#     list_spread=list_spread,
#     list_names=list_names,
#     time=time,
# )

# =========================================================
# Plot the average spread across location
# (with log and standardisation)
# =========================================================

# (list_var,list_names) = extract_variables(
#     nc,
#     var_names=None,
#     ind_time=None,
#     ind_members=None,
#     ind_long=None,
#     ind_lat=None,
#     descr=True
# )

# list_var[5] = np.log(list_var[5])


# (list_scalers, list_stand_var) = standardize(
#     list_var = list_var,
#     each_loc = False,
# )
# list_stand_var = [np.swapaxes(var, 0,1) for var in list_stand_var]
# time = np.array(nc.variables["time"])
# time -= time[0]

# for var in list_stand_var:
#     print(var.shape)

# list_spread = get_list_spread(
#     list_var=list_stand_var,
#     arg_spread=False,
# )

# list_mean_spread = get_list_mean_spread(
#     list_spread=list_spread,
# )

# plot_list_time_series(
#     list_time_series=list_mean_spread,
#     list_names=list_names,
#     time=time,
# )

# list_std = get_list_std(
#     list_var=list_stand_var,
# )

# list_mean_std = get_list_average_values(
#     list_values=list_std,
# )

# plot_list_time_series(
#     list_time_series=list_mean_std,
#     list_names=list_names,
#     time=time,
# )
