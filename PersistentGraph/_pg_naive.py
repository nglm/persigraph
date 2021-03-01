import numpy as np
from typing import List, Sequence, Union, Any, Dict
from utils.kmeans import kmeans_custom
from scipy.spatial.distance import sqeuclidean, cdist

def get_model_parameters(
    pg,
    X = None,
    ):
    

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
    mean = np.mean(pg_members, axis=0)
    std = np.std(pg_members, axis=0)
    scores = np.linalg.norm(pg_members, ord=2, axis=0)
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
    inertia_scores = ['inertia', 'max_inertia', 'min_inertia']
    variance_scores = ['variance', 'max_variance', 'min_variance']
    if pg._zero_type == 'uniform':
        mins = np.amin(pg._members, axis = 0)
        maxs = np.amax(pg._members, axis = 0)

        if pg._score_type in inertia_scores:
            pg._zero_scores = np.around(
                pg.N / 12 * (mins-maxs)**2,
                pg._precision
            )
        elif pg._score_type in variance_scores:
            pg._zero_scores = np.around(
                1 / 12 * (mins-maxs)**2,
                pg._precision
            )
    elif pg._zero_type == 'data':
        if pg._score_type in inertia_scores:
            pg._zero_scores = np.around(
                pg.N * np.var(pg._members, axis = 0),
                pg._precision
            )
        elif pg._score_type in variance_scores:
            pg._zero_scores = np.around(
                np.var(pg._members, axis = 0),
                pg._precision
            )
    # Compute the score of one component and choose the worst score
    for t in range(pg.T):
        model_kw = {'n_clusters' : 1}
        X = pg._members[:,t].reshape(-1,1)
        score, _, _ = pg._clustering_model(
            X,
            X,
            model_kw = model_kw,
        )
        pg._worst_scores[t] = pg.worst_score(
            score,
            pg._zero_scores[t]
        )

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

def __sort_dist_matrix(
    self,
):
    """
    Return a vector of indices to sort distance_matrix
    """
    # Add NaN to avoid redundancy
    dist_matrix = np.copy(self.__dist_matrix)
    for t in range(self.T):
        for i in range(self.N):
            dist_matrix[t,i,i:] = np.nan
            for j in range(i):
                #If the distance is null
                if dist_matrix[t,i,j] == 0:
                    dist_matrix[t,i,j] = np.nan
                    self.__nb_zeros += 1
    # Sort the matrix (NaN should be at the end)
    # Source:
    # https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
    idx = np.dstack(np.unravel_index(
        np.argsort(dist_matrix.ravel()), (self.T, self.N, self.N)
    )).squeeze()

    # Keep only the first non-NaN elements
    for k, (t,i,j) in enumerate(idx):
        if isnan(dist_matrix[t,i,j]):
            idx_first_nan = k
            break

    sort_idx = idx[:idx_first_nan]

    # Store min and max distances
    (t_min, i_min, j_min) = sort_idx[0]
    self.__dist_min = self.__dist_matrix[t_min, i_min, j_min]
    (t_max, i_max, j_max) = sort_idx[-1]
    self.__dist_max = self.__dist_matrix[t_max, i_max, j_max]

    return(sort_idx)