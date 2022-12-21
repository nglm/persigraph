"""
This module deals with clustering-related matters but is totally
independant of the PersistentGraph as well as the score used and the
clustering model used
"""
from bisect import insort
from math import isnan
import numpy as np
import dtaidistance
from dtaidistance.clustering import medoids
from dtaidistance.clustering.hierarchical import Hierarchical
from dtaidistance.clustering.kmeans import KMeans
from typing import List, Sequence, Union, Any, Dict

from .sorted_lists import insert_no_duplicate



def sort_dist_matrix(
    pg,
    distance_matrix
):
    """
    Return a vector of indices to sort distance_matrix
    """
    # Add NaN to avoid redundancy
    dist_matrix = np.copy(distance_matrix)
    for i in range(pg.N):
        dist_matrix[i,i:] = np.nan
        for j in range(i):
            #If the distance is null
            if dist_matrix[i,j] == 0:
                dist_matrix[i,j] = np.nan
    # Sort the matrix (NaN should be at the end)
    # Source:
    # https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
    idx = np.dstack(np.unravel_index(
        np.argsort(dist_matrix.ravel()), (pg.N, pg.N)
    )).squeeze()

    # Keep only the first non-NaN elements
    for k, (i,j) in enumerate(idx):
        if isnan(dist_matrix[i,j]):
            idx_first_nan = k
            break
    idx = idx[:idx_first_nan]

    return idx[::-1]


def get_centroids(
    distance_matrix,
    sorted_idx,
    idx,
    members_r,
):
    # First step: only one cluster
    k = 0
    if idx == 0:
        rep_new = [0]

    # General case
    else:
        rep_new = []
        # Take the 2 farthest members and the corresponding time step
        for k, (i, j) in enumerate(sorted_idx[idx:]):

            # Iterate algo only if i_s and j_s are in the same vertex
            if members_r[i] == members_r[j]:

                # End algo if the 2 farthest apart members are equal
                if distance_matrix[i, j] == 0:
                    raise ValueError('Remaining members are now equal')


                # We'll break this vertex into 2 vertices represented by i and j
                rep_to_break = members_r[i]

                # Extract representatives of alive vertices
                for r in members_r:
                    # We want to remove rep_to_break from rep
                    if r != rep_to_break:
                        insert_no_duplicate(rep_new, r)
                # Now we want to add the new reps i,j replacing rep_to_break
                insort(rep_new, i)
                insort(rep_new, j)

                # stop for loop
                break

    idx += k + 1
    return rep_new, idx


def compute_cluster_params(
    cluster: np.ndarray,
) -> Dict:
    """
    Compute the mean, std, std_sup, inf, etc of a given cluster

    :param cluster: Values of the members belonging to that cluster
    :type cluster: np.ndarray, shape (N_members, d)
    :return: Dict of summary statistics
    :rtype: Dict
    """
    d = cluster.shape[1]
    mean = np.mean(cluster, axis=0).reshape(-1)  # shape: (d)
    std = np.std(cluster, axis=0).reshape(-1)    # shape: (d)
    std_inf = np.zeros(d)                        # shape: (d)
    std_sup = np.zeros(d)                        # shape: (d)

    # Get the members below/above the average
    for i in range(d):
        X_inf = [m for m in cluster[:, i] if m < mean[i]]
        X_sup = [m for m in cluster[:, i] if m > mean[i]]
        # Because otherwise the std goes up to the member if there is just
        # one member
        X_inf.append(mean[i])
        X_sup.append(mean[i])
        # How many members above/below the average
        n_inf = len(X_inf)
        n_sup = len(X_sup)
        X_inf = np.array(X_inf)
        X_sup = np.array(X_sup)
        std_inf[i] = np.sqrt( np.sum((X_inf - mean[i])**2) / n_inf )
        std_sup[i] = np.sqrt( np.sum((X_sup - mean[i])**2) / n_sup )


    cluster_params = {}
    cluster_params['mean'] = mean
    cluster_params['std'] = std
    cluster_params['std_inf'] = std_inf
    cluster_params['std_sup'] = std_sup
    return cluster_params

class KMeansDTW(KMeans):

    # def __init__(
    #     self, k, max_it=10, max_dba_it=10, thr=0.0001, drop_stddev=None,
    #     nb_prob_samples=None, dists_options=None, show_progress=True,
    #     initialize_with_kmedoids=False, initialize_with_kmeanspp=True,
    #     initialize_sample_size=None
    # ):
    #     super().__init__(
    #         k, max_it, max_dba_it, thr, drop_stddev, nb_prob_samples,
    #         dists_options, show_progress, initialize_with_kmedoids,
    #         initialize_with_kmeanspp, initialize_sample_size
    #     )
    def __init__( self, n_clusters=None):
        super().__init__(k=n_clusters)

    def fit_predict(self, X):
        return self.fit(X)

class AgglomerativeClusteringDTW(Hierarchical):

    # def __init__(
    #     self, dists_fun, dists_options, max_dist=..., merge_hook=None,
    #     order_hook=None, show_progress=True
    # ):
    #     super().__init__(
    #         dists_fun, dists_options, max_dist, merge_hook, order_hook,
    #         show_progress
    #     )

    def __init__(self, n_clusters=1):
        super().__init__(
            dtaidistance.dtw.distance_matrix_fast, {}, k=n_clusters
        )

    def fit_predict(self, X):
        return self.fit(X)

# class KMedoidsDTW(medoids.KMedoids):

#     # def __init__(
#     #     self, dists_fun, dists_options, k=None, initial_medoids=None,
#     #     show_progress=True
#     # ):
#     #     super().__init__(
#     #         dists_fun, dists_options, k, initial_medoids, show_progress
#     #     )

#     def __init__( self, n_clusters=1, ):
#         print("n_clusters", n_clusters)
#         super().__init__(dtaidistance.dtw.distance_matrix_fast, {}, k=n_clusters)

#     def fit_predict(self, X):
#         return self.fit(X)

# class KMedoids(medoids.KMedoids):

#     # def __init__(
#     #     self, dists_fun, dists_options, k=None, initial_medoids=None,
#     #     show_progress=True
#     # ):
#     #     super().__init__(
#     #         dists_fun, dists_options, k, initial_medoids, show_progress
#     #     )

#     def __init__( self, n_clusters=1, ):
#         print("n_clusters", n_clusters)
#         super().__init__(dtaidistance.ed.distance, {}, k=n_clusters)
#         print(self.dists_fun)
#         self.dists_options = {}
#         print(self.k)

#     def fit_predict(self, X):
#         return self.fit(X)
