#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from os import makedirs
import numpy as np
import matplotlib.pyplot as plt


from .generate_data import (
    generate_members,generate_params, get_param_values, generate_centers,
    linear, triangle, cosine
)



# In[ ]:

FIG_SIZE = (5,5)
FIG_SIZE2 = (10,5)
PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/toyexamples/"
SUB_DIRS = ["spaghettis/", "true_distrib/", "spaghettis_true_distrib/", "data/"]

# Available:
# - "2_gaussian"
# - "3_gaussian"
# - "N_gaussian"
# - "2_uniform"
to_generate = ["2_gaussian", "3_gaussian"]
generate_all = True

def get_paths(sub_dir):
    path_fig_sub_dir = PATH_FIG_PARENT + sub_dir
    path_fig = []
    # List of all the path subdirs needed
    for sub_dir in SUB_DIRS:
        path_fig.append(path_fig_sub_dir+sub_dir)
    # Create a directory for each path if necessary
    for path in path_fig:
        makedirs(path, exist_ok = True)
    return path_fig



def plot_spaghettis(
    xvalues,
    members,
    ax=None,
    fig=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    for m in members:
        ax.plot(xvalues, m, lw=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title("Spaghettis plot")
    return fig, ax

def plot_true_distribution(
    xvalues,
    centers,      # nb_clusters-List[np.ndarray(T)]
    param_values, # nb_clusters-List[np.ndarray(T)]
    ax=None,
    fig=None,
    distrib_type=np.random.normal,
    i_color=0,    # Starting color
):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    nb_clusters = len(centers)
    # If there is only one set of param values given that means that all
    # clusters share the same params
    if not isinstance(param_values, list):
        param_values = [param_values]*nb_clusters
    # If there is only one distrib type given that means that all
    # clusters share the distrib type
    if not isinstance(distrib_type, list):
        distrib_type = [distrib_type]*nb_clusters
    # Default matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i_center, center in enumerate(centers):
        # Assign a color to each cluster (max 10 colors in the cycle!)
        color = colors[i_color+i_center]
        if distrib_type[i_center] == np.random.normal:
            ax.plot(xvalues, center, c=color, lw=1, ls='-', label='center')
            scale_values = param_values[i_center]['scale']
            ax.plot(xvalues, center+scale_values, c=color, lw=0.5, ls='--', label='std')
            ax.plot(xvalues, center-scale_values, c=color, lw=0.5, ls='--')
        elif distrib_type[i_center] == np.random.uniform:
            radius_values = param_values[i_center]['high']
            ax.plot(xvalues, center+radius_values, c=color, lw=0.5, ls='--', label='bounds')
            ax.plot(xvalues, center-radius_values, c=color, lw=0.5, ls='--')

    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title("True distribution")
    ax.legend()
    return fig, ax

def plot_spaghettis_true_distribution(
    xvalues,
    members,
    centers,  # nb_clusters-List[np.ndarray(T)]
    param_values,
    distrib_type=np.random.normal,
):

    fig, axs = plt.subplots(ncols=2, figsize = FIG_SIZE2, sharey=True, tight_layout=True)
    _, axs[0] = plot_spaghettis(xvalues=xvalues, members=members, ax = axs[0])
    _, axs[1] = plot_true_distribution(
        xvalues=xvalues,
        centers=centers,
        ax = axs[1],
        distrib_type=distrib_type,
        param_values=param_values)
    return fig, axs

def compare(
    xvalues,
    members,
    param_values,
    centers,
    distrib_type=np.random.normal,
    name_fig="",
    path_fig=PATH_FIG_PARENT,
):

    print(" ----------- MEMBERS ----------- ")

    fig, ax = plot_spaghettis(xvalues=xvalues, members=members)
    fig.savefig(str(path_fig[0]+name_fig+".png"))
    plt.close()
    print(" --------- TRUE DISTRIB --------- ")

    fig, ax = plot_true_distribution(
        xvalues=xvalues,
        centers=centers,
        param_values=param_values,
        distrib_type = distrib_type,
    )
    fig.savefig(str(path_fig[1]+name_fig+".png"))
    plt.close()
    print(" -------- MEMBERS & DISTRIB -------- ")

    fig, ax = plot_spaghettis_true_distribution(
        xvalues = xvalues,
        members = members,
        centers= centers,
        param_values = param_values,
        distrib_type=  distrib_type
    )
    fig.savefig(str(path_fig[2]+name_fig+".png"))
    plt.close()

    # Save data
    np.save(str(path_fig[3]+name_fig+"_xvalues"), xvalues)
    np.save(str(path_fig[3]+name_fig+"_members"), members)


# In[ ]:

if "2_gaussian" in to_generate or generate_all:
    path_fig = get_paths("2/gaussian/")

    T = 31
    N = 50

    l_const = np.array(range(0,7,2))
    l_peak_value = np.array(range(0,7,2))
    l_std_min = np.array(range(1,7,2))
    l_std_max= np.array(range(1,10,2))


    nb_clusters = 2
    xvalues = np.arange(T)
    for std_min in l_std_min:
        for std_max in l_std_max:
            for const in l_const:
                for peak_value in l_peak_value:
                    name_fig = "std_"+str(std_min)+'-'+str(std_max)+"_peak_"+str(peak_value)+'_const_'+str(const)
                    weights = linear(xvalues, ymin=std_min, ymax=std_max)
                    if std_max != std_min:
                        np.save(path_fig[-1] + name_fig +"_weights", weights)
                        np.save(path_fig[-1] + name_fig +"bis_weights", weights)

                    params_kw = {'ymax': std_max, 'ymin':std_min}
                    params = generate_params(xvalues=xvalues, names = 'scale', func_kw=params_kw)
                    params_kw = [{'const' : -const, 'peak_value' : -peak_value}, {'const' : const, 'peak_value' : peak_value}]
                    centers = generate_centers(xvalues=xvalues, nb_clusters=nb_clusters, func=triangle, func_kw=params_kw)
                    members = generate_members(xvalues=xvalues, N=N, nb_clusters=nb_clusters, distrib_params=params, cluster_centers = centers)
                    param_values = get_param_values(params)
                    compare(
                        xvalues=xvalues,
                        members=members,
                        param_values=param_values,
                        centers=centers,
                        name_fig=name_fig,
                        path_fig=path_fig,
                    )

                    if std_max != std_min:
                        weights = linear(xvalues, ymin=std_max, ymax=std_min)
                        for i in range(len(centers)):
                            centers[i] = centers[i]/weights
                        members = generate_members(xvalues=xvalues, N=N, nb_clusters=nb_clusters, distrib_params=params, cluster_centers = centers)
                        name_fig = "std_"+str(std_min)+'-'+str(std_max)+"_peak_"+str(peak_value)+'_const_'+str(const)+"bis"
                        compare(
                            xvalues=xvalues,
                            members=members,
                            param_values=param_values,
                            centers=centers,
                            name_fig=name_fig,
                            path_fig=path_fig,
                        )


# In[ ]

if "3_gaussian" in to_generate or generate_all:
    path_fig = get_paths("3/gaussian/")
    T = 31
    N = 50

    l_const = np.array(range(0,7,2))
    l_peak_value = np.array(range(0,7,2))
    l_std_min = np.array(range(1,7,2))
    l_std_max= np.array(range(1,10,2))


    nb_clusters = 3
    xvalues = np.arange(T)
    for std_min in l_std_min:
        for std_max in l_std_max:
            for const in l_const:
                for peak_value in l_peak_value:
                    name_fig = "std_"+str(std_min)+'-'+str(std_max)+"_peak_"+str(peak_value)+'_const_'+str(const)
                    weights = linear(xvalues, ymin=std_min, ymax=std_max)
                    if std_max != std_min:
                        np.save(path_fig[-1] + name_fig +"_weights", weights)
                        np.save(path_fig[-1] + name_fig +"bis_weights", weights)

                    cluster_ratios = [0.40,0.40,0.20]
                    nb_gaussians = nb_clusters
                    distrib_type = [np.random.normal]*nb_gaussians

                    # ------- Generate gaussian params ------------ #
                    gaussian_kw = {'ymax':std_max,'ymin':std_min}

                    gaussian_params = generate_params(xvalues=xvalues, names = 'scale', func_kw=gaussian_kw)
                    gaussian_param_values = [get_param_values(gaussian_params)]*nb_gaussians
                    # ------- Generate triangles centers ------------ #
                    triangle_kw = [{'const' : -const, 'peak_value' : -peak_value}, {'const' : const, 'peak_value' : peak_value}]
                    centers_triangle = generate_centers(xvalues=xvalues, nb_clusters=2, func=triangle, func_kw=triangle_kw)
                    # ------- Generate linear centers ------------ #
                    line_kw = [{'slope' : 0, 'const' : 0}]
                    centers_line = generate_centers(xvalues=xvalues, nb_clusters=1, func=linear, func_kw=line_kw)
                    centers = np.concatenate([centers_triangle, centers_line], axis=0)

                    distrib_type = [np.random.normal, np.random.normal, np.random.normal]
                    members = generate_members(
                        xvalues = xvalues,
                        N = N,
                        nb_clusters = nb_clusters,
                        distrib_params = gaussian_params,
                        cluster_centers = centers,
                        cluster_ratios = cluster_ratios,
                    )

                    compare(
                        xvalues = xvalues,
                        members = members,
                        param_values = gaussian_param_values,
                        centers = centers,
                        name_fig = name_fig,
                        path_fig=path_fig,
                    )

                    if std_max != std_min:
                        weights = linear(xvalues, ymin=std_max, ymax=std_min)
                        for i in range(len(centers)):
                            centers[i] = centers[i]/weights
                        members = generate_members(
                            xvalues=xvalues,
                            N=N,
                            nb_clusters = nb_clusters,
                            distrib_params = gaussian_params,
                            cluster_centers = centers,
                            cluster_ratios = cluster_ratios,
                        )
                        name_fig = "std_"+str(std_min)+'-'+str(std_max)+"_peak_"+str(peak_value)+'_const_'+str(const)+"bis"
                        compare(
                            xvalues = xvalues,
                            members = members,
                            param_values = gaussian_param_values,
                            centers = centers,
                            name_fig = name_fig,
                            path_fig=path_fig,
                        )

# In[ ]


if "2_uniform" in to_generate or generate_all:
    path_fig = get_paths("2/uniform/")
    T = 31
    N = 50

    l_const = np.array(range(0,5,2))
    l_slope = np.around(np.array(range(0,5,2))/6,2)
    l_high_min = np.array(range(1,5,2))
    l_high_max = np.array(range(1,10,3))
    l_cluster_ratios = [[0.5, 0.5], [0.65, 0.35], [0.80, 0.20]]


    nb_clusters = 2
    xvalues = np.arange(T)
    for high_min in l_high_min:
        for high_max in l_high_max:
            for slope in l_slope:
                for const in l_const:
                    for cluster_ratios in l_cluster_ratios:
                        name_fig = "high_"+str(high_min)+"-"+str(high_max)+'_const_'+str(const)+'_slope_'+str(slope)+'_ratios_'+str(cluster_ratios)


                        nb_uniform = nb_clusters
                        distrib_type = [np.random.uniform]*nb_uniform

                        # ------- Generate uniform params ------------ #
                        uniform_kw = [{'ymin':high_min,'ymax':high_max}, {'ymin':-high_min,'ymax':-high_max}]
                        uniform_params = generate_params(xvalues=xvalues, names = ['high', 'low'], func_kw=uniform_kw)
                        uniform_param_values = [get_param_values(uniform_params)]*nb_uniform
                        # ------- Generate linear centers ------------ #
                        line_kw = [{'slope' : slope, 'const' : const}, {'slope' : -slope, 'const' : -const}]
                        centers_line = generate_centers(xvalues=xvalues, nb_clusters=2, func=linear, func_kw=line_kw)
                        centers = centers_line

                        distrib_type = [np.random.uniform, np.random.uniform]
                        members = generate_members(
                            xvalues = xvalues,
                            N = N,
                            nb_clusters = nb_clusters,
                            distrib_type = distrib_type,
                            distrib_params = uniform_params,
                            cluster_centers = centers,
                            cluster_ratios = cluster_ratios,
                        )

                        compare(
                            xvalues = xvalues,
                            members = members,
                            param_values = uniform_param_values,
                            centers = centers,
                            distrib_type = distrib_type,
                            name_fig = name_fig,
                            path_fig=path_fig,
                        )

# In[ ]


if "N_gaussian" in to_generate or generate_all:
    path_fig = get_paths("N/gaussian/")
    N = 10

    #l_const = np.array(range(0,5,2))
    l_slope = np.around(np.array(range(1,5,2))/3,2)
    l_std_min = np.array(range(1,7,2))
    l_std_max= np.array(range(1,10,2))
    cluster_ratios = [0.02]*N


    nb_clusters = N
    xvalues = np.arange(T)
    for std_min in l_std_min:
        for std_max in l_std_max:
            for slope in l_slope:
                name_fig = "std_"+str(std_min)+"-"+str(std_max)+'_slope_'+str(slope)


                nb_gaussian = nb_clusters
                distrib_type = [np.random.normal]*nb_gaussian

                # ------- Generate uniform params ------------ #
                gaussian_kw = [{'ymin':std_min,'ymax':std_max}]
                gaussian_params = generate_params(xvalues=xvalues, names = ['scale'], func_kw=gaussian_kw)
                gaussian_param_values = [get_param_values(gaussian_params)]*nb_gaussian
                # ------- Generate linear centers ------------ #
                center_slopes = np.linspace(-slope, slope, N)
                line_kw = [{'slope' : center_slopes[i], 'const' : 0} for i in range(N)]
                centers_line = generate_centers(xvalues=xvalues, nb_clusters=nb_clusters, func=linear, func_kw=line_kw)
                centers = centers_line

                members = generate_members(
                    xvalues = xvalues,
                    N = N,
                    nb_clusters = nb_clusters,
                    distrib_type = distrib_type,
                    distrib_params = gaussian_params,
                    cluster_centers = centers,
                    cluster_ratios = cluster_ratios,
                )

                compare(
                    xvalues = xvalues,
                    members = members,
                    param_values = gaussian_param_values,
                    centers = centers,
                    distrib_type = distrib_type,
                    name_fig = name_fig,
                    path_fig=path_fig,
                )
