import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Sequence, Union, Any, Dict
from math import isnan
from bisect import insort
from scipy.spatial.distance import sqeuclidean, cdist

from ..utils.sorted_lists import insert_no_duplicate


# TODO: Separate score computations


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
    model_kw = {'idx' : 0}
    return model_kw, fit_predict_kw


def graph_initialization(pg):
    """
    Initialize the graph with 1 components at each time step (mean)

    """
    cluster_data = [[] for _ in range(pg.T)]

    # Start inialization
    mean = np.mean(pg._members, axis=0)
    std = np.std(pg._members, axis=0)
    maxs = np.amax(pg._members, axis=0)
    mins = np.amin(pg._members, axis=0)
    scores = np.around(
                np.abs(maxs-mins) / pg._weights, pg._precision
            )

    clusters = [i for i in range(pg.N)]

    for t in range(pg.T):


        # ======= Create one vertex per time step =======

        cluster_info = {
            'type' : 'Naive',
            'params' : [mean[t], std[t], 0], #mean, std, rep
            'brotherhood_size' : [1]
        }
        cluster_data[t] = [clusters, clusters_info]

        # ========== Finalize initialization step ==================

        pg._local_steps[t].append({
            'param' : {"n_clusters" : 1},
            'score' : pg._worst_scores[t],
        })

        pg._nb_local_steps[t] += 1
        pg._nb_steps += 1

        if pg._verbose:
            print(" ========= ", t, " ========= ")
            print(
                "n_clusters: ", 1,
                "   score: ", pg._worst_scores[t]
            )

    return cluster_data

def compute_extremum_scores(pg):
    """
    Here all time steps share the same bounds
    """
    # zero score is not really defined for the naive method
    if pg._maximize:
        pg._zero_scores = -np.inf*np.ones(pg.T)
    else:
        pg._zero_scores = np.inf*np.ones(pg.T)
    maxs = np.amax(pg._members, axis=0)
    mins = np.amin(pg._members, axis=0)
    worst_score = np.around(
                np.max(np.abs(maxs-mins) / pg._weights), pg._precision
            )
    pg._worst_scores = np.ones(pg.T) * worst_score
    pg._best_scores = np.zeros(pg.T)
    pg._norm_bounds = np.abs(pg._best_scores - pg._worst_scores)
    pg._are_bounds_known = True


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    ):
    t = fit_predict_kw['t']
    idx = model_kw['idx']
    n_clusters = model_kw.pop('n_clusters')
    clusters = None

    # Take the 2 farthest members and the corresponding time step

    for k, (i, j) in enumerate(fit_predict_kw['sorted_idx'][idx:]):

        # Iterate algo only if i_s and j_s are in the same vertex
        if pg._members_v_distrib[t][-1][i] == pg._members_v_distrib[t][-1][j]:

            # End algo if the 2 farthest apart members are equal
            if fit_predict_kw['distance_matrix'][i, j] == 0:
                raise ValueError('Remaining members are now equal')

            # =============== Fit & predict part =======================

            # We'll break this vertex into 2 vertices represented by i and j
            v_to_break = pg._vertices[t][pg._members_v_distrib[t][-1][i]]
            rep_to_break = v_to_break.info['params'][2]
            v_to_break_j = pg._vertices[t][pg._members_v_distrib[t][-1][j]]
            rep_to_break_j = v_to_break_j.info['params'][2]

            # Extract representatives of alive vertices
            rep = []
            v_alive = [pg._vertices[t][v] for v in pg._v_at_step[t]['v'][-1]]
            for v in v_alive:
                # We want to remove rep_to_break from rep
                if v.info['params'][2] != rep_to_break:
                    insert_no_duplicate(rep, v.info['params'][2])
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

            # ========== clusters, cluster_info, step_info =============

            clusters_info = []
            clusters = []
            if len(set(rep)) < n_clusters:
                raise ValueError('No members in cluster')
            for i_cluster in range(n_clusters):
                # Members belonging to that clusters
                members = [
                    m for m in range(pg.N) if members_r[m] == rep[i_cluster]
                ]
                clusters.append(members)
                # Info related to this specific vertex
                clusters_info.append({
                    'type' : 'Naive',
                    'params' : [
                        np.mean(X[members]),
                        np.std(X[members]),
                        rep[i_cluster]
                        ],
                    'brotherhood_size' : [n_clusters]
                    })
            score = np.around(
                fit_predict_kw['distance_matrix'][i, j], pg._precision
            )
            step_info = {'score' : score, '(i,j)' : (i,j)}

            model_kw['idx'] = k + idx + 1
            # Stop for loop
            break
    if clusters is None:
        raise ValueError('No new clusters')

    return clusters, clusters_info, step_info, model_kw
