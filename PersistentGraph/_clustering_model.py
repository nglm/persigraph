"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
from sklearn.metrics import pairwise_distances
import numpy as np
from bisect import bisect_left
from typing import List, Sequence, Union, Any, Dict, Tuple

from . import _pg_kmeans, _pg_naive
from ._scores import compute_score

from ..utils._clustering import sort_dist_matrix
from ..utils.sorted_lists import reverse_bisect_left


def get_model_parameters(
    pg,
    X: np.ndarray,
    t: int = None,
) -> Tuple[Dict, Dict]:
    """
    Initialize clustering model parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d) optional
    :param t: current time step (for weights), defaults to None
    :type t: int, optional
    :return: 2 dict, for the model initialization and its fit method
    :rtype: Tuple[Dict, Dict]
    """
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
        #HERE_done
        distance_matrix = pairwise_distances(X) * pg._weights[:, t]
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



def generate_zero_component(
    pg,
    X: np.ndarray,
    model_kw: dict = {},
    fit_predict_kw: dict = {},
) -> Tuple[List[List[int]], List[Dict], Dict, Dict]:
    """
    Create the 0 component of the graph for all time steps

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
    # ====================== Fit & predict part ========================
    if pg._zero_type == 'bounds':

        # Get the parameters of the uniform distrib using min and max
        #HERE_done axis
        mins = np.amin(X, axis=0)
        #HERE_done axis
        maxs = np.amax(X, axis=0)

    else:
        # I'm not sure if this type should be used at all actually.....
        # Get the parameters of the uniform distrib using mean and variance
        #HERE_done axis
        var = np.var(X, axis=0)
        #HERE_done axis
        mean = np.mean(X, axis=0)

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2

    # Generate a perfect uniform distribution
    steps = (maxs-mins) / (pg.N-1)
    #HERE_done (d, T)
    values = np.array([mins + i*steps for i in range(pg.N)])

    # ==================== clusters, cluster_info ======================
    # Compute the score of that distribution
    clusters = [[i for i in range(pg.N)]]
    clusters_info = [{
        'type' : 'uniform',
        'params' : [values[0], values[-1]], # lower/upper bounds
        'brotherhood_size' : [0]
    }]

    # ========================== step_info =============================
    step_info = {'values': values}

    return clusters, clusters_info, step_info, model_kw


def clustering_model(
    pg,
    X,
    model_kw : dict = {},
    fit_predict_kw : dict = {},
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
    if model_kw['n_clusters'] == 0:
        (
            clusters,
            clusters_info,
            step_info,
            model_kw,
        ) = generate_zero_component(
            pg = pg,
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



def generate_all_clusters(
    pg
):
    # --------------------- Preliminary ----------------------------
    # Use bisect left in order to favor the lowest number of clusters
    if pg._maximize:
        sort_fc = reverse_bisect_left
    else:
        sort_fc = bisect_left

    # temporary variable to help sort the local steps
    cluster_data = [[] for _ in range(pg.T)]
    local_scores = [[] for _ in range(pg.T)]

    for t in range(pg.T):
        if pg._verbose:
            print(" ========= ", t, " ========= ")

        # ------ clustering method specific parameters -------------
        #HERE_done (N, d, T)
        X = pg._members[:, :, t]
        # Get clustering model parameters required by the
        # clustering model
        model_kw, fit_predict_kw = get_model_parameters(
            pg,
            X = X,
            t = t,
        )


        for n_clusters in pg._n_clusters_range:

            # Update model_kw
            model_kw['n_clusters'] = n_clusters

            # ---------- Fit & predict using clustering model-------
            try :
                (
                    clusters,
                    clusters_info,
                    step_info,
                    model_kw,
                ) = clustering_model(
                    pg,
                    X,
                    model_kw = model_kw,
                    fit_predict_kw = fit_predict_kw,
                )
            except ValueError as ve:
                if not pg._quiet:
                    print(str(ve))
                continue

            # -------- Score corresponding to 'n_clusters' ---------

            if n_clusters == 0:
                score = compute_score(
                    pg,
                    X = step_info.pop('values'),
                    clusters = clusters,
                    t = t,
                )
                pg._zero_scores[t] = score
            else:
                score = compute_score(
                    pg,
                    X = X,
                    clusters = clusters,
                    t = t,
                )
            step_info['score'] = score

            # ---------------- Finalize local step -----------------
            # Find where should we insert this future local step
            idx = sort_fc(local_scores[t], score)
            local_scores[t].insert(idx, score)
            cluster_data[t].insert(idx, [clusters, clusters_info])
            pg._local_steps[t].insert(
                idx,
                {**{'param' : {"n_clusters" : n_clusters}},
                    **step_info
                }
            )
            pg._nb_steps += 1
            pg._nb_local_steps[t] += 1


            if pg._verbose:
                msg = "n_clusters: " + str(n_clusters)
                for (key,item) in step_info.items():
                    msg += '  ||  ' + key + ":  " + str(item)
                print(msg)



    return cluster_data, local_scores