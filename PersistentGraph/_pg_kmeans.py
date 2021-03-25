import numpy as np
from typing import List, Sequence, Union, Any, Dict

from ..utils.kmeans import kmeans_custom, row_norms
from ..utils._clustering import get_centroids
from ._scores import _compute_cluster_params



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
    # Default kw values
    model_kw = {
        'max_iter' : pg._model_kw.pop('max_iter', 100),
        'n_init' : pg._model_kw.pop('n_init', 10),
        'tol' : pg._model_kw.pop('tol', 1e-3),
    }
    if pg._model_kw['precompute_centroids']:
        # If we ive explicit centroids we don't need n_init parameter
        model_kw.pop('n_init')

    model_kw.update(pg._model_kw)
    return model_kw, fit_predict_kw


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
):

    # ====================== Fit & predict part =======================
    n_clusters = model_kw.pop('n_clusters')

    # If centroids are to be precomputed
    if pg._model_kw['precompute_centroids']:
        # Get new reprensatives and the index in the distance matrix
        rep_new, idx = get_centroids(
            distance_matrix = pg._model_kw['distance_matrix'],
            sorted_idx = pg._model_kw['sorted_idx'],
            idx = pg._model_kw['idx'],
            members_r = pg._model_kw['rep'],
        )
        if rep_new == []:
            raise ValueError('No new centroid')
        # Get the inital centroids
        model_kw['init'] = np.array([X[r] for r in rep_new]).reshape(-1, 1)
        pg._model_kw['idx'] = idx
        members_r = np.zeros(pg.N, dtype=int)

    model = kmeans_custom(
        n_clusters = n_clusters,
        copy_x = False,
        **model_kw,
    )
    labels = model.fit_predict(**fit_predict_kw)
    if model.n_iter_ == model_kw['max_iter']:
        raise ValueError('Kmeans did not converge')


    # ==================== clusters, cluster_info ======================
    clusters_info = []
    clusters = []
    for label_i in range(n_clusters):
        # Members belonging to that clusters
        members = [m for m in range(pg.N) if labels[m] == label_i]
        clusters.append(members)
        if members == []:
            raise ValueError('No members in cluster')

        # Info related to this specific vertex
        clusters_info.append({
            'type' : 'KMeans',
            'params' : _compute_cluster_params(X[members]),
            'brotherhood_size' : [n_clusters]
        })

        if pg._model_kw['precompute_centroids']:
            # Associate members with a representative according
            # to the clustering made by the model
            members_r[members] = rep_new[label_i]
            # Force representatives to be representated by themselves
            # (the clustering model might decide against)
            members_r[rep_new] = rep_new


    # ========================== step_info =============================
    # Add method specific info here if necessary
    # (Score is computed in persistentgraph)
    step_info = {}
    pg._model_kw['rep'] = members_r


    #TODO: add cluster center to model_kw for future clustering

    return clusters, clusters_info, step_info, model_kw