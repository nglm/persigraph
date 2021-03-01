import numpy as np
from typing import List, Sequence, Union, Any, Dict
from utils.kmeans import kmeans_custom

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
            score += len(members)/self.N * np.var(X[members])
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

    if pg._verbose:
        print(" ========= Initialization ========= ")
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
                'params' : [pg._members[i,t], 0.],
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
                "n_clusters: ", 0,
                "   score: ", 0
            )


def clustering_model(
    pg,
    X,
    copy_X,
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