import numpy as np
from typing import List, Sequence, Union, Any, Dict
from utils.kmeans import kmeans_custom, row_norms

def get_model_parameters(
        pg,
        X,
):
    # The same N datapoints X are use for all n_clusters values
    # Furthermore the clustering method might want to copy X
    # Each time it is called and compute pairwise distances
    # We avoid doing that more than once
    # using copy_X and row_norms_X (in fit_predict_kw)
    copy_X = np.copy(X)
    row_norms_X = row_norms(copy_X, squared=True)
    fit_predict_kw = {
        "x_squared_norms" : row_norms_X,
        'X' : copy_X,
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
    Initialize the graph with N components at each time step
    """
    for t in range(pg.T):

        # Initialization
        pg._members_v_distrib[t].append(
            np.zeros(pg.N, dtype = int)
        )
        pg._v_at_step[t]['v'].append([])
        pg._v_at_step[t]['global_step_nums'].append(None)

        # ======= Create one vertex per member and time step =======
        for i in range(pg.N):
            info = {
                'type' : 'KMeans',
                'params' : [pg._members[i,t], 0.], #mean, std
                'brotherhood_size' : pg.N
            }
            v = pg._add_vertex(
                info = info,
                t = t,
                members = [i],
                scores = [0, None],
                local_step = 0
            )

        # ========== Finalize initialization step ==================

        pg._local_steps[t].append({
            'param' : {"n_clusters" : pg.N},
            'score' : 0,
        })

        pg._nb_local_steps[t] += 1
        pg._nb_steps += 1

        if pg._verbose:
            print(" ========= ", t, " ========= ")
            print(
                "n_clusters: ", pg.N,
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

        X = pg._members[:,t].reshape(-1,1)
        model_kw, fit_predict_kw = get_model_parameters(pg, X)
        model_kw['n_clusters'] = 1

        _, _, step_info = clustering_model(
            pg,
            X,
            model_kw = model_kw,
            fit_predict_kw = fit_predict_kw,
        )
        pg._worst_scores[t] = pg.worst_score(
            step_info['score'],
            pg._zero_scores[t]
        )

    pg._best_scores = np.zeros(pg.T)
    pg._norm_bounds = np.abs(pg._best_scores - pg._worst_scores)
    pg._are_bounds_known = True


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    ):

    # Default kw values
    max_iter = model_kw.pop('max_iter', 200)
    n_init = model_kw.pop('n_init', 10)
    tol = model_kw.pop('tol', 1e-3)
    n_clusters = model_kw.pop('n_clusters')
    model = kmeans_custom(
        n_clusters = n_clusters,
        max_iter = max_iter,
        tol = tol,
        n_init = n_init,
        copy_x = False,
        **model_kw,
    )
    labels = model.fit_predict(**fit_predict_kw)
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
    step_info = {'score' : score}

    return clusters, clusters_info, step_info