import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Sequence, Union, Any, Dict
from math import isnan
from bisect import insort
from scipy.spatial.distance import sqeuclidean, cdist

from ._scores import _compute_cluster_params
from ..utils.sorted_lists import insert_no_duplicate

def _sort_dist_matrix(
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


def get_model_parameters(
    pg,
    X = None,
    t = None,
):
    # Compute pairwise distances
    distance_matrix = pairwise_distances(X) / pg._weights[t]
    # Argsort of pairwise distances
    sorted_idx = _sort_dist_matrix(pg, distance_matrix)
    # t is needed to access members_v_distrib[t][-1]
    fit_predict_kw = {
        "distance_matrix" : distance_matrix,
        "sorted_idx" : sorted_idx,
        't' : t,
        }
    # idx is needed to know which i, j are the next candidates
    # rep is needed to know whether i and j are in the same vertex
    model_kw = {
        'idx' : 0,
        'rep' : []
    }
    return model_kw, fit_predict_kw


def _fit(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
):
    t = fit_predict_kw['t']
    idx = model_kw['idx']
    members_r = model_kw['rep']

    # =============== Fit & predict part =======================
    # First step: only one cluster
    k = 0
    if idx == 0:
        members_r = [0 for _ in range(pg.N)]
        rep = [0]

    # General case
    else:
        rep = []
        # Take the 2 farthest members and the corresponding time step
        for k, (i, j) in enumerate(fit_predict_kw['sorted_idx'][idx:]):

            # Iterate algo only if i_s and j_s are in the same vertex
            if members_r[i] == members_r[j]:

                # End algo if the 2 farthest apart members are equal
                if fit_predict_kw['distance_matrix'][i, j] == 0:
                    raise ValueError('Remaining members are now equal')


                # We'll break this vertex into 2 vertices represented by i and j
                rep_to_break = members_r[i]

                # Extract representatives of alive vertices
                for r in members_r:
                    # We want to remove rep_to_break from rep
                    if r != rep_to_break:
                        insert_no_duplicate(rep, r)
                # Now we want to add the new reps i,j replacing rep_to_break
                insort(rep, i)
                insort(rep, j)

                # extract distance to representatives
                dist = []
                for r in rep:
                    dist.append(fit_predict_kw['distance_matrix'][r])

                dist = np.asarray(dist)     # (nb_rep, N) array
                # for each member, find the representative that is the closest
                members_r = [rep[r] for r in np.nanargmin(dist, axis=0)]

                # Stop for loop
                break
        # End algo if the 2 farthest apart members are equal
        if rep == []:
            raise ValueError('No new clusters')


    model_kw['idx'] = k + idx + 1
    model_kw['rep'] = members_r

    return rep, members_r, model_kw

def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
):

    # ====================== Fit & predict part =======================
    rep, members_r, model_kw = _fit(
        pg,
        X = X,
        model_kw = model_kw,
        fit_predict_kw = fit_predict_kw
    )
    idx = model_kw['idx']
    members_r = model_kw['rep']
    n_clusters = model_kw.pop('n_clusters')
    if len(rep) < n_clusters:
        raise ValueError('No members in cluster')

    # ================= clusters, cluster_info =================
    clusters_info = []
    clusters = []
    for i_cluster in range(n_clusters):

        # Members belonging to that cluster
        members = [
            m for m in range(pg.N) if members_r[m] == rep[i_cluster]
        ]
        clusters.append(members)

        # Info related to this specific vertex
        cluster_params =  _compute_cluster_params(X[members])
        cluster_params.append(rep[i_cluster])
        clusters_info.append({
            'type' : 'Naive',
            'params' : cluster_params,
            'brotherhood_size' : [n_clusters]
            })

    # ====================== step_info =========================
    # Add method specific info here if necessary
    # (Score is computed in persistentgraph)
    step_info = {}

    return clusters, clusters_info, step_info, model_kw
