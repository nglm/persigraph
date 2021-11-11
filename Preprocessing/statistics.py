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

from scipy.stats import norm, kurtosis
from typing import List, Sequence, Union, Any, Dict

from ..utils.plt import get_nrows_ncols_from_nplots, get_subplot_indices
from ..utils.npy import running_mean



def moving_average(
    list_var,   # List(ndarray(n_time, n_members [, n_long, n_lat])
    list_windows,
):
    """
    #FIXME: Outdated since multivariate

    :param list_var: [description]
    :type list_var: [type]
    :return: [description]
    :rtype: [type]
    """
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


def standardize(
    list_var,  # List[ndarray(n_members, n_time, n_long, n_lat)]
    each_loc: bool = False, # if true return List[ndarray(n_long, n_lat)] else return List[Scaler]
):

    #FIXME: Outdated since multivariate

    list_scalers = []
    list_stand_var = []
    for var in list_var:
        if len(var.shape) > 2:
            (n_members, n_time, n_long, n_lat) = var.shape
        else:
            (n_members, n_time) = var.shape
            each_loc = False
        if each_loc:
            print("Not implemented yet")
        else:
            if len(var.shape) > 2:
                print("Not implemented yet")
            else:
                mean = np.mean(var)
                std = np.std(var)
                list_stand_var.append((var - mean) / std)
                list_scalers.append([mean, std])
    return (list_scalers, list_stand_var)



# Extract each variable to study its dimension, distribution; etc
def extract_var_distrib(
    list_var,
    descr: bool = False,
    list_names = None
):
    """Flatten all columns for each variable

    #FIXME: Outdated since multivariate

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

    #FIXME: Outdated since multivariate

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

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36, 20))
    for i in range(d):
        idx = get_subplot_indices(i, ncols)
        axs[idx].violinplot(var_distrib[i])
        axs[idx].set_title(list_names[i])
    if show:
        plt.show()
    return fig, axs
# plot_violinplots(var_distrib, list_names=var_names, nrows=2, ncols=3)


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
        #std = np.squeeze(np.std(var,axis=0, keepdims=True))
        std = np.squeeze(np.std(var,axis=0))
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

def get_list_stats(
    list_values # List[ndarray(n_time, n_members, [n_long, n_lat])]
):
    list_std = []
    list_mean = []
    list_kurtosis = []
    for values in list_values:
        list_std.append(np.squeeze(np.std(values, axis=1)))
        list_mean.append(np.squeeze(np.mean(values, axis=1)))
        list_kurtosis.append(np.squeeze(kurtosis(values, axis=1)))
    # List[ndarray(n_values, n_time, [n_long, n_lat])]
    return ([np.array(list_mean), np.array(list_std), np.array(list_kurtosis)])

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
