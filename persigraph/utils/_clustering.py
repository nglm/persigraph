"""
This module deals with clustering-related matters but is totally
independent of the PersistentGraph as well as the score used and the
clustering model used
"""
from bisect import insort
from math import isnan
import numpy as np
from sklearn.metrics import pairwise_distances
from tslearn.metrics import dtw_path
from tslearn.barycenters import softdtw_barycenter
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
    midpoint_w: int = None
) -> Dict:
    """
    Compute the mean, std, std_sup, inf, etc of a given cluster

    If `cluster` has shape `(N_clus, w_t, d)`, uses DTW. Otherwise, it
    must have a shape `(N_clus, w_t*d)` and the regular mean is used.

    If DTW is used, first compute the barycentric average of the
    cluster. Then consider only the midpoint of that barycenter to
    compute the mean and uses it as a reference to compute the standard
    deviation. Otherwise, uses the regular mean.

    :param cluster: Values of the members belonging to that cluster
    :type cluster: np.ndarray, shape (N_clus, w_t*d)`
    or `(N_clus, w_t, d)`
    :param t: center point of the time window w_t, used as reference in
    case of DTW, defaults to None
    Note that for the first time steps and the last time steps, the
    'center' of the time window is not necessarily in the middle on the
    window. E.g. for t=0 and w = 50, we have w_t = 26 and midpoint_w = 0
    :type t: int, optional
    :return: Dict of summary statistics
    :rtype: Dict
    """
    dims = cluster.shape
    cluster_params = {}

    # Regular case
    if len(dims) == 2:
        (N_clus, d) = dims
        mean = np.mean(cluster, axis=0).reshape(-1)  # shape: (d)
        X = np.copy(cluster)

    # DTW case
    elif len(dims) == 3:
        (N_clus, w_t, d) = dims
        # Take the barycenter as reference
        barycenter = softdtw_barycenter(cluster)
        cluster_params['barycenter'] = barycenter
        X = []
        # For each time series in the dataset, compare to the barycenter
        for ts in cluster:
            path, _ = dtw_path(
                barycenter, ts,
                global_constraint="sakoe_chiba", sakoe_chiba_radius=5
            )
            # Find time steps that match with the midpoint of the
            # barycenter
            ind = [
                path[i][1] for i in range(len(path))
                if path[i][0] == midpoint_w
            ]
            # Option 1: take all time steps that match the midpoint of
            # the barycenter
            # Here we can have len(X) > N_clus! because for a given time
            # series, multiple time step can correspond to the midpoint
            # of the barycenter find finding the dtw_path between that
            # time series and this cluster. Each element of X is of
            # shape (d)
            # X += [ts[i] for i in ind]

            # Option 2: take all the mean value for all time steps that
            # match the midpoint of the barycenter
            X.append(np.mean([ts[i] for i in ind], axis=0))
        X = np.array(X)
        mean = barycenter[midpoint_w]
    else:
        msg = (
            "Clusters should be of dimension (N, d*w_t) or (N, w_t, d)."
            + "and not " + str(dims)
        )
        raise ValueError(msg)

    std_inf = np.zeros(d)                        # shape: (d)
    std_sup = np.zeros(d)                        # shape: (d)
    for i in range(d):
        # Get the members below/above the average
        X_inf = [m for m in X[:, i] if m < mean[i]]
        X_sup = [m for m in X[:, i] if m > mean[i]]
        # Because otherwise the std goes up to the member if there is just
        # one member
        X_inf.append(mean[i])
        X_sup.append(mean[i])
        # How many members above/below the average
        n_inf = len(X_inf)
        n_sup = len(X_sup)
        X_inf = np.array(X_inf)
        X_sup = np.array(X_sup)
        std = np.std(X, axis=0).reshape(-1)    # shape: (d)
        std_inf[i] = np.sqrt( np.sum((X_inf - mean[i])**2) / n_inf )
        std_sup[i] = np.sqrt( np.sum((X_sup - mean[i])**2) / n_sup )

    # Member values (aligned with the barycenter if DTW was used)
    cluster_params['X'] = X
    cluster_params['center'] = mean
    cluster_params['disp'] = std
    cluster_params['disp_inf'] = std_inf
    cluster_params['disp_sup'] = std_sup
    return cluster_params

class Naive:

    def __init__(self, n_clusters, distance=pairwise_distances) -> None:
        self.k = n_clusters
        self.labels_ = None
        self.distance = distance

    def predict(self, X):
        # Find the diameters (and the points that define it) for each clusters
        # Associate all datapoints to its closest representative
        # Repeat
        clusters = [[i for i in range(len(X))]]
        rep = []
        for i in range(1, self.k +1):

            pdist_in_clusters = [self.distance(X[c]) for c in clusters]
            arg_diam_in_clusters = [
                np.unravel_index(d.argmax(), d.shape)
                for d in pdist_in_clusters
            ]
