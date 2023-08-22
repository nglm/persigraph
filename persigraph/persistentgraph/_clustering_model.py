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

from pycvi.compute_scores import compute_score, compute_all_scores
from pycvi.cluster import (
    compute_cluster_params, prepare_data, sliding_window, get_clusters,
    generate_all_clusterings,
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

def _merge_clusters_aux(
    clustering1: List[List[int]],
    clustering2: List[List[int]],
    l_k1: List[List[int]],
    l_k2: List[List[int]],
) -> None:
    """
    In place merging of 2 clusterings and their respective list of k.

    :param clustering1: clustering1
    :type clustering1: List[List[int]]
    :param clustering2: clustering2
    :type clustering2: List[List[int]]
    :param l_k1: list of k for each cluster in clustering1
    :type l_k1: List[List[int]]
    :param l_k2: list of k for each cluster in clustering2
    :type l_k2: List[List[int]]
    """

    for i1, c1 in enumerate(clustering1):
        for i2, c2 in enumerate(clustering2):
            if are_equal(c1, c2):
                # Merge info
                [insort(l_k1[i1], k) for k in l_k2[i2]]
                # Delete redundant cluster and l_k1
                del l_k2[i2]
                del clustering2[i2]

def merge_clusters(
    clusterings_t_k: List[Dict[List[List[int]]]],
) -> Tuple[List[Dict[List[List[int]]]], List[Dict[List[List[int]]]]]:
    """
    Returns the clusterings without duplicate clusters and "k" for each cluster

    `listk_t_k: List[Dict[List[List[int]]]]`
    listk_t_k[t_w][k][i] is the list of k for which the cluster i exists

    :param clusterings_t_k: _description_
    :type clusterings_t_k: List[Dict[List[List[int]]]]
    """
    # `clusterings_t_k: List[Dict[List[List[int]]]]`
    # `clusterings_t_k[t_w][k][i]` is a list of members indices contained
    # in cluster i for the clustering assuming k clusters for the
    # extracted time window t_w.

    # initialise a list of k for each cluster of each k of each t_w
    T_w = len(clusterings_t_k)
    listk_t_k = [
        {k: [[k] for _ in range(k)] for k in clusterings_t_k[t_w].key()}
        for t_w in range(T_w)]

    # Delete redundant clusters and merge their k
    for t_w in range(T_w):

        list_k = list(clusterings_t_k[t_w].key())
        # Iterate over the values of k in two nested loops
        # while always starting one element ahead in the second loop
        for i, k1 in enumerate(list_k):
            for j, k2 in enumerate(list_k[i+1:]):
                # Merge clusters that are the same between 2 clusterings
                _merge_clusters_aux(
                    clusterings_t_k[t_w][i], clusterings_t_k[t_w][j],
                    listk_t_k[t_w][i], listk_t_k[t_w][j],
                )
    return clusterings_t_k, listk_t_k

def generate_all_clusters(
    pg
) -> List[Tuple[List[int], dict]]:
    """
    Generate all clustering data

    :param pg: Persistent Graph
    :type pg: _type_
    :return: All data corresponding to the generated clustering in a nested list
    clusterings_t_k. each element of the nested list contain [clusters, clusters_info]
    :rtype: List[Tuple[List[int], dict]]
    """

    # --------------------------------------------------------------
    # --------------------- Find clusters --------------------------
    # --------------------------------------------------------------

    # Get clustering model parameters required by the
    # clustering model
    model_kw, fit_predict_kw, model_class_kw = get_model_parameters(
        pg._model_class,
        model_kw = pg._model_kw,
        fit_predict_kw = pg._fit_predict_kw,
        model_class_kw = pg._model_class_kw,
    )

    # `clusterings_t_k: List[Dict[List[List[int]]]]`
    # `clusterings_t_k[t_w][k][i]` is a list of members indices contained
    # in cluster i for the clustering assuming k clusters for the
    # extracted time window t_w.
    clusterings_t_k = generate_all_clusterings(
        pg.members, pg._model_class,
        n_clusters_range=range(pg.k_max),
        DTW=pg._DTW,
        time_window=pg.w,
        transformer=pg._transformer,
        scaler=pg._scaler,
        model_kw=model_kw,
        fit_predict_kw=fit_predict_kw,
        model_class_kw=model_class_kw
    )

    # --------------------------------------------------------------
    # -------- Compute score, cluster params, etc. -----------------
    # --------------------------------------------------------------

    # `scores_t_: List[Dict[float]]`
    # `scores_t_n[t_w][k]` is the score for the clustering assuming k
    # clusters for the extracted time window t_w
    scores_t_n = compute_all_scores(
        pg._score_type,
        pg._members,
        clusterings_t_k,
        transformer = pg._transformer,
        scaler = pg._scaler,
        DTW = pg._DTW,
        time_window = pg._w,
        dist_kwargs = {},
        score_kwargs = {},
    )

    T_w = len(scores_t_n)
    info = [
        [{
            **{"k": [n_clusters]},
            **compute_cluster_params(
                X_params[c],
                pg._sliding_window["midpoint_w"][t_w])
        }]
        for t_w in range(T_w)
    ]

    # # temporary variable to help sort the local steps
    # # cluster_data[t][s] contains (clusters, clusters_info)
    # cluster_data = [[] for _ in range(pg.T)]
    # local_scores = [[] for _ in range(pg.T)]

    # for t in range(pg.T):

    #     for n_clusters in pg._n_clusters_range:
    #         # Take the data used for clustering while taking into account the
    #         # difference between time step indices with/without sliding window
    #         if n_clusters == 0:
    #             # members_clus: list of length T of arrays of shape (N, w_t, d)
    #             # X: (N, w_t, d)
    #             X = np.copy(members_clus0[t])
    #             X_params = np.copy(members_params0[t])
    #         else:
    #             X = np.copy(members_clus[t])
    #             X_params = np.copy(members_params[t])

    #         (N, w_t, d) = X.shape
    #         midpoint_w = wind["midpoint_w"][t]
    #         if not pg._DTW:
    #             # We take the entire time window into consideration for the
    #             # scores of the clusters
    #             # X: (N, d*w_t)
    #             X = X.reshape(N, w_t*d)
    #             # We take only the midpoint into consideration for the
    #             # parameters of the clusters
    #             # X_params: (N, d)
    #             X_params = X_params[:, midpoint_w, :]

    #         # Find cluster membership of each member
    #         clusters = clusterings_t_k[t][n_clusters]

    #         # Go to next iteration if the model didn't converge
    #         if clusters is None:
    #             continue

    #         # -------- Cluster infos for each cluster ---------

    #         clusters_data = [
    #             (
    #                 c,
    #                 {
    #                     **{"k": [n_clusters]},
    #                     **compute_cluster_params(X_params[c], midpoint_w)
    #                 }
    #             ) for c in clusters
    #         ]
    #         clusters_data_score = [
    #             (c, compute_cluster_params(X[c], midpoint_w))
    #             for c in clusters
    #         ]

    #         # ------------ Score corresponding to 'n_clusters' ---------
    #         score = compute_score(
    #             pg._score,
    #             X = X,
    #             clusters = clusters_data_score,
    #         )
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