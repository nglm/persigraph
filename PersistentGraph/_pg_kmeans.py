import numpy as np
from typing import List, Sequence, Union, Any, Dict

from ..utils.kmeans import kmeans_custom, row_norms



def get_model_parameters(
        pg,
        X,
        t = None,
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
        't' : t,
        }
    # Default kw values
    model_kw = {
        'max_iter' : 100,
        'n_init' : 10,
        'tol' : 1e-3,
    }
    return model_kw, fit_predict_kw


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
):

    # ====================== Fit & predict part =======================
    t = fit_predict_kw.pop('t', None)
    n_clusters = model_kw.pop('n_clusters')
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
            'params' : [
                float(model.cluster_centers_[label_i]),
                float(np.std(X[members])),
                ],
            'brotherhood_size' : [n_clusters]
        })

    # ========================== step_info =============================
    # Add method specific info here if necessary
    # (Score is computed in persistentgraph)
    step_info = {}

    #TODO: add cluster center to model_kw for future clustering

    return clusters, clusters_info, step_info, model_kw