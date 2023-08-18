"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
import numpy as np
from bisect import bisect_left, insort
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from tslearn.clustering import TimeSeriesKMeans
from typing import List, Sequence, Union, Any, Dict, Tuple

from pycvi.compute_scores import compute_score
from pycvi.cluster import (
    compute_cluster_params, prepare_data, sliding_window, get_clusters
)
from ..utils.sorted_lists import reverse_bisect_left, are_equal

CLUSTERING_METHODS = {
    "names": [
        "KMeans", "SpectralClustering", "GaussianMixture",
        "AgglomerativeClustering",
    ],
    "classes-standard": [
        KMeans, SpectralClustering, GaussianMixture,
        AgglomerativeClustering,
    ],
    "classes-dtw": [
        TimeSeriesKMeans, None, None,
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
        # m_kw = {
        #     'metric' : 'softdtw',
        # }
        m_kw = {
            'metric' : 'dtw',
            'metric_params' : {
                'global_constraint': "sakoe_chiba",
                'sakoe_chiba_radius' : 5,
                }
        }
        m_kw.update(model_kw)
    elif model_class == "Naive":
        raise NotImplementedError

    return m_kw, ft_kw, mc_kw

def _merge_clusters_aux(cluster_data1, cluster_data2):

    for i1, (c1, info1) in enumerate(cluster_data1):
        for i2, (c2, info2) in enumerate(cluster_data2):
            if are_equal(c1, c2):
                # Merge info
                [insort(info1['k'], k) for k in info2['k']]
                # Delete redundant cluster
                del cluster_data2[i2]

def merge_clusters(cluster_data):
    for cluster_data_t in cluster_data:
        # cluster_data_t = []
        n_clusterings = len(cluster_data_t)
        for i in range(n_clusterings-1):
            for j in range(i+1, n_clusterings):
                # Merge clusters that are the same between 2 clusterings
                _merge_clusters_aux(
                    cluster_data_t[i], cluster_data_t[j]
                )

def generate_all_clusters(
    pg
) -> List[Tuple[List[int], dict]]:
    """
    Generate all clustering data

    :param pg: Persistent Graph
    :type pg: _type_
    :return: All data corresponding to the generated clustering in a nested list
    clusters_t_n. each element of the nested list contain [clusters, clusters_info]
    :rtype: List[Tuple[List[int], dict]]
    """
    # --------------------------------------------------------------
    # --------------------- Preliminary ----------------------------
    # --------------------------------------------------------------
    # Use bisect left in order to favor the lowest number of clusters
    if pg._score_maximize:
        sort_fc = bisect_left
    else:
        sort_fc = reverse_bisect_left

    wind = sliding_window(pg.T, pg.w)
    # Use default transformer and scaler
    members_clus = prepare_data(pg.members, pg._DTW, wind,)
    members_clus0 = prepare_data(pg.members_zero, pg._DTW, wind)
    # No scaler/transformer
    members_params = prepare_data(
        pg.members, wind, transformer=None, scaler=None
    )
    members_params0 = prepare_data(
        pg.members_zero, wind, transformer=None, scaler=None
    )

    # --------------------------------------------------------------
    # --------------------- Find clusters --------------------------
    # --------------------------------------------------------------

    # temporary variable to help remember clusters before merging
    # clusters_t_n[t][k][i] is a list of members indices contained in cluster i
    # for the clustering assuming k cluster at time step t.
    clusters_t_n = [{} for _ in range(pg.T)]

    for t in range(pg.T):

        # ---- clustering method specific parameters --------
        # members_clus: list of length T of arrays of shape (N, w_t, d)
        # X: (N, w_t, d)
        X = members_clus[t]
        (N, w_t, d) = X.shape
        if not pg._DTW:
            # X: (N, w_t*d)
            X = X.reshape(N, w_t*d)

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

            # All members in the same cluster. Go to next iteration
            if n_clusters <= 1:
                clusters_t_n[t][n_clusters] = [[i for i in range(pg.N)]]
            # General case
            else:

                # Update model_kw
                model_kw[model_class_kw["k_arg_name"]] = n_clusters

                # ---------- Fit & predict using clustering model-------
                try :
                    clusters = get_clusters(
                        pg._model_class,
                        model_kw = model_kw,
                        fit_predict_kw = fit_predict_kw,
                        model_class_kw = model_class_kw,
                    )
                except ValueError as ve:
                    if not pg._quiet:
                        print(str(ve))
                    clusters_t_n[t][n_clusters] = None
                    continue

                clusters_t_n[t][n_clusters] = clusters

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    # temporary variable to help sort the local steps
    # cluster_data[t][s] contains (clusters, clusters_info)
    cluster_data = [[] for _ in range(pg.T)]
    local_scores = [[] for _ in range(pg.T)]

    for t in range(pg.T):

        for n_clusters in pg._n_clusters_range:
            # Take the data used for clustering while taking into account the
            # difference between time step indices with/without sliding window
            if n_clusters == 0:
                # members_clus: list of length T of arrays of shape (N, w_t, d)
                # X: (N, w_t, d)
                X = np.copy(members_clus0[t])
                X_params = np.copy(members_params0[t])
            else:
                X = np.copy(members_clus[t])
                X_params = np.copy(members_params[t])

            (N, w_t, d) = X.shape
            midpoint_w = wind["midpoint_w"][t]
            if not pg._DTW:
                # We take the entire time window into consideration for the
                # scores of the clusters
                # X: (N, d*w_t)
                X = X.reshape(N, w_t*d)
                # We take only the midpoint into consideration for the
                # parameters of the clusters
                # X_params: (N, d)
                X_params = X_params[:, midpoint_w, :]

            # Find cluster membership of each member
            clusters = clusters_t_n[t][n_clusters]

            # Go to next iteration if the model didn't converge
            if clusters is None:
                continue

            # -------- Cluster infos for each cluster ---------

            clusters_data = [
                (
                    c,
                    {
                        **{"k": [n_clusters]},
                        **compute_cluster_params(X_params[c], midpoint_w)
                    }
                ) for c in clusters
            ]
            clusters_data_score = [
                (c, compute_cluster_params(X[c], midpoint_w))
                for c in clusters
            ]

            # ------------ Score corresponding to 'n_clusters' ---------
            score = compute_score(
                pg._score,
                X = X,
                clusters = clusters_data_score,
            )
            if n_clusters == 0:
                pg._zero_scores[t] = score
                # don't insert the case k=0 in cluster_data
                # nor in local_steps, go straight to the next iteration
            elif pg._score_type == "monotonous" and score :
                pass
            else:
                step_info = {"score" : score}

                # ---------------- Finalize local step -----------------
                # Find where should we insert this future local step
                idx = sort_fc(local_scores[t], score)
                local_scores[t].insert(idx, score)
                cluster_data[t].insert(idx, clusters_data)
                pg._local_steps[t].insert(
                    idx,
                    {**{'param' : {"k" : n_clusters}},
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