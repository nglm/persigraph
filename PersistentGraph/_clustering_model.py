
from sklearn.metrics import pairwise_distances

from . import _pg_kmeans, _pg_naive
from ..utils._clustering import sort_dist_matrix


def get_model_parameters(
    pg,
    X,
    t = None,
):
    # ----------- method specific key-words ------------------------
    if pg._model_type == "KMeans":
        model_kw, fit_predict_kw = _pg_kmeans.get_model_parameters(
            pg,
            X = X,
        )
    elif pg._model_type == "Naive":
        model_kw, fit_predict_kw = _pg_naive.get_model_parameters(
            pg,
            X = X,
            t = t,
        )
    # ----------- Common keywords ------------------------
    precompute_centroids = pg._model_kw['precompute_centroids']
    # 'pop' because we don't want it when we call clustering_model
    model_kw.pop("precompute_centroids", None)
    if precompute_centroids:
        # Compute pairwise distances
        #HERE!
        distance_matrix = pairwise_distances(X) / pg._weights[t]
        # Argsort of pairwise distances
        sorted_idx = sort_dist_matrix(pg, distance_matrix)
        pg._model_kw["distance_matrix"] = distance_matrix
        model_kw.pop("distance_matrix", None)
        pg._model_kw["sorted_idx"] = sorted_idx
        model_kw.pop("sorted_idx", None)
        pg._model_kw['idx'] = 0
        model_kw.pop("idx", None)
        pg._model_kw['rep'] = []
        model_kw.pop("rep", None)


    return model_kw, fit_predict_kw

def clustering_model(
    pg,
    X,
    model_kw : dict = {},
    fit_predict_kw : dict = {},
    ):
    if model_kw['n_clusters'] == 0:
        (
            clusters,
            clusters_info,
            step_info,
            model_kw,
        ) = pg._generate_zero_component(
            X = X,
            model_kw = model_kw,
            fit_predict_kw = fit_predict_kw,
        )
    else:
        if pg._model_type == 'KMeans':
            (
                clusters,
                clusters_info,
                step_info,
                model_kw,
            ) = _pg_kmeans.clustering_model(
                pg,
                X = X,
                model_kw = model_kw,
                fit_predict_kw = fit_predict_kw,
            )
        elif pg._model_type == 'Naive':
            (
                clusters,
                clusters_info,
                step_info,
                model_kw,
            ) = _pg_naive.clustering_model(
                pg,
                X = X,
                model_kw = model_kw,
                fit_predict_kw = fit_predict_kw,
            )

    return clusters, clusters_info, step_info, model_kw