"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
import numpy as np
from bisect import bisect_left
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from tslearn.clustering import TimeSeriesKMeans
from typing import List, Sequence, Union, Any, Dict, Tuple

from ._scores import compute_score
from ..utils._clustering import (
    compute_cluster_params,
)
from ..utils.sorted_lists import reverse_bisect_left

CLUSTERING_METHODS = {
    "names": [
        "KMeans", "TimeSeriesKMeans", "SpectralClustering", "GaussianMixture",
        "AgglomerativeClustering",
    ],
    "classes-standards": [
        KMeans, TimeSeriesKMeans, SpectralClustering, GaussianMixture,
        AgglomerativeClustering,
    ],
    "classes-dtw": [
        None, TimeSeriesKMeans, None, None,
        None,
    ],
}

def get_model_parameters(
    model_class,
    model_kw = {},
    fit_predict_kw = {},
    model_class_kw = {},
) -> Tuple[Dict, Dict]:
    """
    Initialize clustering model parameters

    :return: 2 dict, for the model initialization and its fit method
    :rtype: Tuple[Dict, Dict]
    """
    m_kw = {}
    ft_kw = {}
    mc_kw = {
        "k_arg_name" : "n_clusters",
        "X_arg_name" : "X"
    }
    # ----------- method specific key-words ------------------------
    if model_class == KMeans:
        m_kw = {
            'max_iter' : model_kw.pop('max_iter', 100),
            'n_init' : model_kw.pop('n_init', 20),
            'tol' : model_kw.pop('tol', 1e-3),
        }
        m_kw.update(model_kw)
        ft_kw.update(fit_predict_kw)
    elif model_class == GaussianMixture:
        mc_kw['k_arg_name'] = "n_components"
        mc_kw.update(model_class_kw)
    elif model_class == TimeSeriesKMeans:
        m_kw = {
            'metric' : 'dtw',
        }
        m_kw.update(model_kw)
    elif model_class == "Naive":
        raise NotImplementedError

    return m_kw, ft_kw, mc_kw


def generate_zero_component(
    pg,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate values to emulate the case k=0

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d) optional
    :return: Data emulating the k=0 based on X
    :rtype: np.ndarray
    """
    # ====================== Fit & predict part ========================
    if pg._zero_type == 'bounds':

        # Get the parameters of the uniform distrib using min and max
        mins = np.amin(X, axis=0)
        maxs = np.amax(X, axis=0)

    else:
        # I'm not sure if this type should be used at all actually.....
        # Get the parameters of the uniform distrib using mean and variance
        var = np.var(X, axis=0)
        mean = np.mean(X, axis=0)

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2

    # Generate a perfect uniform distribution
    steps = (maxs-mins) / (pg.N-1)
    X_zero = np.array([mins + i*steps for i in range(pg.N)])

    return X_zero

def clustering_model(
    pg,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    model_class_kw : Dict = {},
) -> Tuple[List[List[int]], List[Dict], Dict, Dict]:
    """
    Generate a clustering instance with the given model/fit parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param model_kw: Dict of kw for the model initialization, defaults to {}
    :type model_kw: dict, optional
    :param fit_predict_kw: Dict of kw for the fit_predict method, defaults to {}
    :type fit_predict_kw: dict, optional
    :return: All data corresponding to the generated clustering
    :rtype: Tuple[List[List[int]], List[Dict], Dict, Dict]
    """
    n_clusters = model_kw[model_class_kw["k_arg_name"]]
    X = fit_predict_kw[model_class_kw["X_arg_name"]]

    # ====================== Fit & predict part =======================
    model = pg._model_class(**model_kw)
    labels = model.fit_predict(**fit_predict_kw)

    # ==================== clusters ======================
    clusters = [ [] for _ in range(n_clusters)]
    for label_i in range(n_clusters):
        # Members belonging to that clusters
        members = [m for m in range(pg.N) if labels[m] == label_i]
        clusters[label_i] = members
        if members == []:
            raise ValueError('No members in cluster')

    return clusters

def _data_to_cluster(pg) -> np.ndarray:
    """
    Data to be clustered using sliding window.

    Note that if `pg.w` is even, it takes `t` and `t+1` time steps to cluster
    data at `t`. It is recommended to use odd numbers for `pg.w`. There is
    no padding for the sliding window and `stride=1`.
    So `T_clus = T - w + 1`, and `T_origin_to_clust[0..t//2] = T_clus[0]`

    :param pg: Persistent Graph
    :type pg: _type_
    :return: The data that will be clustered as a (N, d*w, T_clus) array
    :rtype: np.ndarray
    """
    # Correspondence of indices between T and T_clus
    T_clus = pg.T - pg.w + 1

    X = np.copy(pg._members)

    if pg._squared_radius:
        # r = sqrt(RMM1**2 + RMM2**2)
        # r of shape (N, 1, T)
        r = np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
        # r*X gives the same angle but a squared radius
        X = r*X

    X_clus = np.zeros((pg.N, pg.d*pg.w, T_clus))
    for t in range(T_clus):
        X_clus[:, :, t] = X[:,:,t:t+pg.w].reshape(pg.N, pg.d*pg.w)

    return X_clus

def _time_indices(pg):
    T_clus = pg.T - pg.w + 1
    T_ind = {'to_clus' : {}, 'to_origin' : {}}
    # From origin to cluster, length T
    T_ind["to_clus"].update({t: 0 for t in range(pg.w)})
    T_ind["to_clus"].update({int(t+pg.w): t+1 for t in range(T_clus-1)})
    T_ind["to_clus"].update({t: T_clus-1 for t in range(T_clus-1, pg.T)})

    # From cluster to origin, length T_clus but the first and last elements
    # could have multiple indices
    T_ind["to_origin"].update({0: [t for t in range(pg.w)]})
    T_ind["to_origin"].update({t+1: int(t+pg.w) for t in range(T_clus-1)})
    T_ind["to_origin"].update({T_clus-1 :t for t in range(T_clus-1, pg.T)})

    return T_ind


def generate_all_clusters(
    pg
):
    # --------------------- Preliminary ----------------------------
    # Use bisect left in order to favor the lowest number of clusters
    if pg._maximize:
        sort_fc = reverse_bisect_left
    else:
        sort_fc = bisect_left

    members_clus = _data_to_cluster(pg)
    T_clus = members_clus.shape[-1]

    # temporary variable to help remember clusters before merging
    # clusters_t_n[t][k][i] is a list of members indices contained in cluster i
    # for the clustering assuming k cluster at time step t.
    clusters_t_n = [{} for _ in range(T_clus)]

    for t in range(T_clus):

        # ------ clustering method specific parameters -------------
        X = members_clus[:, :, t]
        # Get clustering model parameters required by the
        # clustering model
        model_kw, fit_predict_kw, model_class_kw = get_model_parameters(
            pg._model_class,
            model_kw = pg._model_kw,
            fit_predict_kw = pg._fit_predict_kw,
            model_class_kw = pg._model_class_kw,
        )
        fit_predict_kw[model_class_kw["X_arg_name"]] = X

        for n_clusters in pg._n_clusters_range:

            if n_clusters == 0:
                clusters_t_n[t][n_clusters] = [[i for i in range(pg.N)]]
                continue

            # Update model_kw
            model_kw[model_class_kw["k_arg_name"]] = n_clusters

            # ---------- Fit & predict using clustering model-------
            try :
                clusters = clustering_model(
                    pg,
                    model_kw = model_kw,
                    fit_predict_kw = fit_predict_kw,
                    model_class_kw = model_class_kw,
                )
            except ValueError as ve:
                if not pg._quiet:
                    print(str(ve))
                continue

            clusters_t_n[t][n_clusters] = clusters

    # temporary variable to help sort the local steps
    # cluster_data[t] contains (clusters, clusters_info)
    cluster_data = [[] for _ in range(pg.T)]
    local_scores = [[] for _ in range(pg.T)]
    T_ind = _time_indices(pg)

    for t in range(pg.T):

        X = pg._members[:, :, t]
        for n_clusters in pg._n_clusters_range:

            # Difference between time step indices with/without sliding window
            clusters = clusters_t_n[T_ind["to_clus"][t]][n_clusters]

            # -------- Cluster infos for each cluster ---------
            clusters_info = [compute_cluster_params(X[c]) for c in clusters]

            # -------- Score corresponding to 'n_clusters' ---------

            if n_clusters == 0:
                score = compute_score(
                    pg,
                    X = generate_zero_component(pg, X),
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
            step_info = {"score" : score}

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
                print(" ========= ", t, " ========= ")
                msg = "n_clusters: " + str(n_clusters)
                for (key,item) in step_info.items():
                    msg += '  ||  ' + key + ":  " + str(item)
                print(msg)

    return cluster_data