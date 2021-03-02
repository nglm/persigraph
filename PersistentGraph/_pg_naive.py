import numpy as np
from typing import List, Sequence, Union, Any, Dict
from utils.kmeans import kmeans_custom
from scipy.spatial.distance import sqeuclidean, cdist

def compute_dist_matrix(pg, X):
    """
    Compute the pairwise distance matrix for each time step

    .. warning::

        Distance to self is set to 0
    """
    dist = []
    # append is more efficient for list than for np
    for t in range(self.T):
        # if your data has a single feature use array.reshape(-1, 1)
        if self.d == 1:
            dist_t = pairwise_distances(self.__members[:,t].reshape(-1, 1))
        else:
            dist_t = pairwise_distances(self.__members[:,t])
        dist.append(dist_t/self.__dist_weights[t])
    self.__dist_matrix = np.asarray(dist)

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
    # Argsort of pairwise distances
    sorted_idx = _sort_dist_matrix(pg, distance_matrix)
    fit_predict_kw = {
        "distance_matrix" : distance_matrix,
        "sorted_idx" : sorted_idx,
        't' : t,
        }

    model_kw = {}
    return model_kw, fit_predict_kw

def compute_score(pg, model=None, X=None, clusters=None):
    if pg._score_type == 'inertia':
        return np.around(model.inertia_, pg._precision)
    elif pg._score_type == 'max_inertia':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                score,
                np.sum(cdist(
                    X[members],
                    np.mean(X[members]).reshape(-1, 1) ,
                    metric='sqeuclidean'
                    )
                ))
            return np.around(score, pg._precision)
    elif pg._score_type == 'min_inertia':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(
                score,
                np.sum(cdist(
                    X[members],
                    np.mean(X[members]).reshape(-1, 1) ,
                    metric='sqeuclidean'
                    )
                ))
            return np.around(score, pg._precision)
    elif pg._score_type == 'variance':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += len(members)/pg.N * np.var(X[members])
        return np.around(score, pg._precision)
    elif pg._score_type == 'max_variance':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(np.var(X[members]), score)
        return np.around(score, pg._precision)
    elif pg._score_type == 'min_variance':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(np.var(X[members]), score)
        return np.around(score, pg._precision)

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
    copy_X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    ):


    # Take the 2 farthest members and the corresponding time step
    for (i_s, j_s) in sort_idx:

        # Iterate algo only if i_s and j_s are in the same vertex
        if (self.__M_v[s][t_s, i_s] == self.__M_v[s][t_s, j_s]):

            # End algo if the 2 farthest apart members are equal
            if self.__dist_matrix[t_s, i_s, j_s] == 0:
                break

            if verbose:
                print(
                    "==== Step ", str(s), "====",
                    "(i, j) = ", (j_s, i_s),
                    "distance i-j: ", self.__dist_matrix[t_s, i_s, j_s]
                )
            self.__steps.append((t_s, i_s, j_s))
            self.__distances.append(self.__dist_matrix[t_s, i_s, j_s])

            # List of new representatives
            representatives = self.__update_representatives(
                s=s,
                t=t_s,
                i=i_s,
                j=j_s,
                verbose=verbose,
            )




    # Default kw values
    n_clusters = model_kw.pop('n_clusters')
    model = kmeans_custom(
        n_clusters = n_clusters,
        max_iter = max_iter,
        tol = tol,
        n_init = n_init,
        copy_x = False,
        **model_kw,
    )
    labels = model.fit_predict(copy_X, **fit_predict_kw)
    if model.n_iter_ == max_iter:
        raise ValueError('Kmeans did not converge')
    clusters_info = []
    clusters = []
    for label_i in range(n_clusters):
        # Members belonging to that clusters
        members = [m for m in range(pg.N) if labels[m] == label_i]
        clusters.append(members)
        if members == []:
            if pg._quiet:
                print("No members in cluster")
            raise ValueError('No members in cluster')
        # Info related to this specific vertex
        clusters_info.append({
            'type' : 'KMeans',
            'params' : [
                float(model.cluster_centers_[label_i]),
                float(np.std(X[members])),
                ],
            'brotherhood_size' : n_clusters
        })

    score = compute_score(
            pg,
            model = model,
            X = X,
            clusters = clusters,
        )

    return score, clusters, clusters_info



