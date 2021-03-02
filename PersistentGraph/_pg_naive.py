import numpy as np
from typing import List, Sequence, Union, Any, Dict
from utils.kmeans import kmeans_custom
from utils.sorted_lists import insert_no_duplicate
from scipy.spatial.distance import sqeuclidean, cdist


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

    return idx[:idx_first_nan]


def get_model_parameters(
    pg,
    X = None,
    t = t,
):
    # Compute pairwise distances
    distance_matrix = pairwise_distances(X) / g._weights[t]
    #np.fill_diagonal(distance_matrix, np.nan)
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
    # Start inialization
    mean = np.mean(pg._members, axis=0)
    std = np.std(pg._members, axis=0)
    scores = np.linalg.norm(pg._members, ord=2, axis=0) / pg._weights
    members = [i for i in range(pg.N)]
    for t in range(pg.T):

        # Initialization
        pg._members_v_distrib[t].append(
            np.zeros(pg.N, dtype = int)
        )
        pg._v_at_step[t]['v'].append([])
        pg._v_at_step[t]['global_step_nums'].append(None)

        # ======= Create one vertex per time step =======

        info = {
            'type' : 'Naive',
            'params' : [pg._members[i,t], 0., 0], #mean, std, rep
            'brotherhood_size' : 1
        }
        v = pg._add_vertex(
            info = info,
            t = t,
            members = members,
            scores = [scores[t], None],
            local_step = 0
        )

        # ========== Finalize initialization step ==================

        pg._local_steps[t].append({
            'param' : {"n_clusters" : 1},
            'score' : scores[t],
        })

        pg._nb_local_steps[t] += 1
        pg._nb_steps += 1

        if pg._verbose:
            print(" ========= ", t, " ========= ")
            print(
                "n_clusters: ", 1,
                "   score: ", 0
            )

def compute_extremum_scores(pg):
    """
    Here all time steps share the same bounds
    """
    one_scores = np.linalg.norm(pg._members, ord=2, axis=0) / pg._weights
    self._worst_scores = np.ones(pg.T) * np.amax(one_scores)
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

    # Take the 2 farthest members and the corresponding time step
    for k, (i, j) in enumerate(fit_predict_kw['sorted_idx'][idx:]):

        # Iterate algo only if i_s and j_s are in the same vertex
        if pg._members_v_distrib[t][-1][i] == pg._members_v_distrib[t][-1][j]:

            # End algo if the 2 farthest apart members are equal
            if fit_predict_kw['distance_matrix'][i, j] == 0:
                raise ValueError('Remaining members are now equal')

            # =============== Fit & predict part =======================

            # We'll break this vertex into 2 vertices represented by i and j
            v_to_break = pg._members_v_distrib[t][-1][i]
            rep_to_break = v_to_break.info['params'][2]

            # Extract representatives of alive vertices
            rep = []
            for v_alive in pg._vertices[t][pg._v_at_step[t][-1]]:
                # We want to remove rep_to_break from rep
                if v_alive.info['params'][2] != rep_to_break
                    insert_no_duplicate(rep, v_alive.info['params'][2])
            # Now we want to add the new reps i,j replacing rep_to_break
            insert_no_duplicate(rep, i)
            insert_no_duplicate(rep, j)

            # extract distance to representatives
            dist = []
            for r in representatives:
                dist.append(fit_predict_kw['distance_matrix'][r])

            dist = np.asarray(dist)     # (nb_rep, N) array
            # for each member, find the representative that is the closest
            idx = np.nanargmin(dist, axis=0)

            # ========== clusters, cluster_info, step_info =============

            score = np.linalg.norm(pg._members[i,t]-pg._members[j,t] , ord=2)
            step_info = {'score' : score}
            clusters_info = []
            clusters = []
            for i_cluster in range(n_cluster):
                # Members belonging to that clusters
                members = [m for m in range(pg.N) if rep[m] == rep[i_cluster]]
                clusters.append(members)
                # Info related to this specific vertex
                clusters_info.append({
                    'type' : 'Naive',
                    'params' : [
                        np.mean(members),
                        np.std(members),
                        rep[i_cluster]
                        ],
                    })

            model_kw['idx'] = k + idx
            # Stop for loop
            break

    return clusters, clusters_info, step_info, model_kw
