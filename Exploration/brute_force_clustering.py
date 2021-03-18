import numpy as np
from sklearn.decomposition import PCA
import sys, os
from os import listdir, makedirs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from ..DataAnalysis.statistics import preprocess_data
from ..utils.plt import plot_mean_and_std



# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

# Absolute path to the files
# type: str
PATH_DATA = "/home/natacha/Documents/Work/Data/Bergen/"

# Choose the path where the figs will be saved
# type: str
PATH_FIG_PARENT = "/home/natacha/Documents/tmp/figs/Exploration/Brute_Force/"

# Choose which files should be used
LIST_FILENAMES = listdir(PATH_DATA)
LIST_FILENAMES = [
    fname for fname in LIST_FILENAMES
    if fname.startswith("ec.ens.") and  fname.endswith(".nc")
]

# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------

def find_pca_n_components(
    X,
    explained_var_threshold: float
    ) -> int:
    """

    :param X: data
    :type X: np.ndarray
    :param explained_var_threshold:
    ratio of explained variance we want to keep
    :type explained_var_threshold: float
    :return: Number of components to keep
    :rtype: int
    """


    model = PCA().fit(X)
    n_components = 0
    acc_variance = 0
    while acc_variance < explained_var_threshold:
        acc_variance += model.explained_variance_ratio_[n_components]
        n_components += 1
    print(n_components)
    return n_components

def PCA_pipeline(members):
    explained_var_threshold = 0.9
    n_components = find_pca_n_components(members, explained_var_threshold)
    model = PCA(n_components)
    X_red = model.fit_transform(members)

def get_list_members_from_dendogram(Z, color_list):
    # There are N-1 linkage for a dataset of N points
    N = len(Z) + 1
    distinct_colors = list(set(color_list))
    # C0 doesn't count as a cluster
    n_colors = len(distinct_colors) - 1
    clusters = { color : [] for color in distinct_colors}
    for i in range(N-1):
        # Take into account leaves only
        if Z[i,0] < N:
            clusters[color_list[i]].append(int(Z[i,0]))
        if Z[i,1] < N:
            clusters[color_list[i]].append(int(Z[i,1]))
    list_members = []
    # From a dict to a nested list
    for color, members in clusters.items():
        if color != "C0":
            list_members.append(members)
    return list_members




def hierarchical_clustering(X, filename):
    path_fig = PATH_FIG_PARENT + "Hierchichal/dendogram/"

    fig, ax = plt.subplots()

    linked = linkage(X, 'ward')
    model = dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    labels = list(set(model['ivl']))

    print("n clusters:", len(labels))

    list_members = get_list_members_from_dendogram(linked, model['color_list'])

    name_fig = path_fig + filename[:-3] + ".png"
    makedirs(path_fig, exist_ok = True)
    fig.savefig(name_fig)
    plt.close()
    return list_members

def hierarchical_pipeline(X, filename, x_axis=None):
    path_fig = PATH_FIG_PARENT + "Hierchichal/spaghetti/"
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    list_members = hierarchical_clustering(X, filename)
    if x_axis is None:
        x_axis = np.arange(X.shape[-1])

    fig, ax = plt.subplots(ncols=2, figsize=(20,10), sharey=True)
    for i_cluster, members in enumerate(list_members):
        color = colors[i_cluster]
        for m in members:
            ax[0].plot(X[m], color=color, lw=0.5)
        dict_kwargs = {
            'plot_mean_color' : color,
            'plot_mean_lw' : len(members)/5}
        _, ax[1] = plot_mean_and_std(
            yvalues = X[members],
            xvalues = x_axis,
            ax = ax[1],
            dict_kwargs = dict_kwargs,
        )

    name_fig = path_fig + filename[:-3] + ".png"
    makedirs(path_fig, exist_ok = True)
    fig.savefig(name_fig)
    plt.close()


def PCA_2_components_pipeline(
    X,
    filename,
    ):
    path_fig = PATH_FIG_PARENT + "PCA_2_cmpts/"
    model = PCA(2)
    X_red = model.fit_transform(X)

    fig, ax = plt.subplots()
    plt.scatter(X_red[:,0], X_red[:,1])

    name_fig = path_fig + filename[:-3] + ".png"
    makedirs(path_fig, exist_ok = True)

    fig.savefig(name_fig)
    plt.close()


def main():
    for filename in LIST_FILENAMES:

        list_var, list_names, time = preprocess_data(
            filename = filename,
            path_data = PATH_DATA,
            var_names=['t2m'],
            ind_time=None,
            ind_members=None,
            ind_long=[0],
            ind_lat=[0],
            to_standardize = False,
            )

        t2m = list_var[0]

        #PCA_2_components_pipeline(X = t2m, filename = filename)
        #find_pca_n_components(X = t2m, explained_var_threshold = 0.95)
        hierarchical_pipeline(X = t2m, x_axis=time, filename = filename)



main()


