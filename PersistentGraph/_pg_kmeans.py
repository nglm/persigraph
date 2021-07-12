"""
This module depends on the clustering model and is meant to be called by
the _clustering module.
"""

import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple

from ..utils.kmeans import kmeans_custom, row_norms
from ..utils._clustering import get_centroids, compute_cluster_params



def get_model_parameters(
        pg,
        X: np.ndarray,
) -> Tuple[Dict, Dict]:
    """
    Initialize clustering model parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d) optional
    :return: 2 dict, for the model initialization and its fit method
    :rtype: Tuple[Dict, Dict]
    """
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
        'n_init' : pg._model_kw.pop('n_init', 20),
        'tol' : pg._model_kw.pop('tol', 1e-3),
    }

    model_kw.update(pg._model_kw)

    if pg._model_kw['precompute_centroids']:
        # If we ive explicit centroids we don't need n_init parameter
        model_kw['n_init'] = 1
    return model_kw, fit_predict_kw


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
) -> Tuple[List[List[int]], List[Dict], Dict, Dict]:
    """
    Generate a clustering instance with the given model/fit parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d) optional
    :param model_kw: Dict of kw for the model initalization, defaults to {}
    :type model_kw: dict, optional
    :param fit_predict_kw: Dict of kw for the fit_predict method, defaults to {}
    :type fit_predict_kw: dict, optional
    :return: All data corresponding to the generated clustering
    :rtype: Tuple[List[List[int]], List[Dict], Dict, Dict]
    """
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
        model_kw['init'] = np.array([X[r] for r in rep_new])
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
        info = {
            'type' : 'KMeans',
            'brotherhood_size' : [n_clusters]
        }
        info.update(compute_cluster_params(X[members]))
        clusters_info.append(info)

        if pg._model_kw['precompute_centroids']:
            # Associate members with a representative according
            # to the clustering made by the model
            members_r[members] = rep_new[label_i]

    if pg._model_kw['precompute_centroids']:
        # Force representatives to be representated by themselves
        # (the clustering model might decide against)
        #members_r[rep_new] != rep_new
        if np.any(members_r[rep_new] != rep_new):
            members_r[rep_new] = rep_new
            pg._model_kw['rep'] = members_r
            raise ValueError('Centroid was ignored')


    # ========================== step_info =============================
    # Add method specific info here if necessary
    # (Score is computed in persistentgraph)
    step_info = {}
    pg._model_kw['rep'] = members_r


    #TODO: add cluster center to model_kw for future clustering

    return clusters, clusters_info, step_info, model_kw