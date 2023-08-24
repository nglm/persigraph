"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
from bisect import insort
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from tslearn.clustering import TimeSeriesKMeans
from typing import List, Sequence, Union, Any, Dict, Tuple

from pycvi.compute_scores import compute_all_scores
from pycvi.cluster import (
    compute_cluster_params, generate_all_clusterings,
)
from ..utils.sorted_lists import are_equal

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
    clusterings_t_k: List[Dict[int, List[List[int]]]],
) -> Tuple[List[Dict[int, List[List[int]]]], List[Dict[int, List[List[int]]]]]:
    """
    Returns the clusterings without duplicate clusters and "k" for each cluster

    `listk_t_k: List[Dict[int, List[List[int]]]]`
    listk_t_k[t_w][k][i] is the list of k for which the cluster i exists

    :param clusterings_t_k: _description_
    :type clusterings_t_k: List[Dict[int, List[List[int]]]]
    """
    # `clusterings_t_k: List[Dict[int, List[List[int]]]]`
    # `clusterings_t_k[t_w][k][i]` is a list of members indices contained
    # in cluster i for the clustering assuming k clusters for the
    # extracted time window t_w.

    # initialise a list of k for each cluster of each k of each t_w
    # listk_t_k[t_w][k][i] contains a list of k values for the cluster i
    # in the the clustering k at t_w
    T_w = len(clusterings_t_k)
    listk_t_k = [
        {
            k: [[k] for _ in range(k)] if k>0 else [[0]]
            for k in clusterings_t_k[t_w].keys()
        } for t_w in range(T_w)]

    # Delete redundant clusters and merge their k
    for t_w in range(T_w):

        # k values of the clusterings at t_w
        list_k = list(clusterings_t_k[t_w].keys())
        # Iterate over the values of k in two nested loops
        # while always starting one element ahead in the second loop
        for i, k1 in enumerate(list_k):
            for j, k2 in enumerate(list_k[i+1:]):
                # Merge clusters that are the same between 2 clusterings
                _merge_clusters_aux(
                    clusterings_t_k[t_w][k1], clusterings_t_k[t_w][k2],
                    listk_t_k[t_w][k1], listk_t_k[t_w][k2],
                )
    return clusterings_t_k, listk_t_k

def generate_all_clusters(
    pg
) -> List[Dict[int, List[List[int]]]]:
    """
    Generate all clustering data

    :param pg: Persistent Graph
    :type pg: _type_
    :return: All data corresponding to the generated clustering in a nested list
    clusterings_t_k. each element of the nested list contain [clusters, clusters_info]
    :rtype: List[Dict[int, List[List[int]]]]
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

    # `clusterings_t_k: List[Dict[int, List[List[int]]]]`
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

    # `scores_t_: List[Dict[int, float]]`
    # `scores_t_n[t_w][k]` is the score for the clustering assuming k
    # clusters for the extracted time window t_w
    scores_t_n = compute_all_scores(
        pg._score,
        pg._members,
        clusterings_t_k,
        transformer = pg._transformer,
        scaler = pg._scaler,
        DTW = pg._DTW,
        time_window = pg._w,
        score_kwargs = {},
    )

    # Compute zero score if relevant
    if 0 in scores_t_n[0]:
        pg._zero_scores = [scores_t_n[t][0] for t in range(pg._T_w)]

    pg._local_steps = [
        [
            {
                "k": n_clusters,
                "score": score,
            }
            for (n_clusters, score) in scores_t_n[t_w].items()
        ] for t_w in range(pg._T_w)
    ]

    return clusterings_t_k