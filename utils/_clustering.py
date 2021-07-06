"""
This module deals with clustering-related matters but is totally
independant of the PersistentGraph as well as the score used and the
clustering model used
"""
from bisect import insort
from math import isnan
import numpy as np


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
    #HERE_done
    d = cluster.shape[1]
    mean = np.mean(cluster, axis=0).reshape(-1)  # shape: (d)
    std = np.std(cluster, axis=0).reshape(-1)    # shape: (d)
    # Get the members below/above the average
    X_inf = np.array(
        [[m for i in range(d) if m < mean[i]] for m in cluster]
    )
    X_sup = np.array(
        [[m for i in range(d) if m >= mean[i]] for m in cluster]
    )
    # How many members above/below the average
    n_inf = len(X_inf)
    n_sup = len(X_sup)
    # if statement because of '<': No member below the average => std=0
    if n_inf == 0:
        std_inf = 0
    else:
        std_inf = np.sqrt(np.sum((X_inf - mean)**2, axis=0) / n_inf).reshape(-1)
    std_sup = np.sqrt(np.sum((X_sup - mean)**2) / n_sup).reshape(-1)

    cluster_params = {}
    cluster_params['mean'] = mean
    cluster_params['std'] = std
    cluster_params['std_inf'] = std_inf
    cluster_params['std_sup'] = std_sup
    return cluster_params