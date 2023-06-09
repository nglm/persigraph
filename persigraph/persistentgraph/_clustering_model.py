"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
import numpy as np
from bisect import bisect_left, insort
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from tslearn.clustering import TimeSeriesKMeans
from typing import List, Sequence, Union, Any, Dict, Tuple

from ._scores import compute_score
from ..utils._clustering import (
    compute_cluster_params,
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

def clustering_model(
    pg,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
    model_class_kw : Dict = {},
) -> List[List[int]]:
    """
    Generate a clustering instance with the given model/fit parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param model_kw: Dict of kw for the model initialization, defaults
    to {}
    :type model_kw: dict, optional
    :param fit_predict_kw: Dict of kw for the fit_predict method,
    defaults to {}
    :type fit_predict_kw: dict, optional
    :param model_class_kw: To know how X and n_clusters args are called in this
    model class, defaults to {}
    :type model_class_kw: dict, optional
    :return: Members affiliation to the generated clustering
    :rtype: List[List[int]]
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

def _data_to_cluster(
    pg,
    X: np.ndarray,
    window: dict,
    transform_data: bool = True
) -> List[np.ndarray]:
    """

    Data to be used for cluster scores and params, using sliding window.

    The index that should be used to compute cluster params
    of clustering that were computed using `X_clus` is called
    "midpoint_w" in the window dictionary

    :param pg: Persistent Graph
    :type pg: _type_
    :param X: _description_
    :type X: np.ndarray
    :param window: _description_
    :type window: dict
    :param transform_data: _description_, defaults to True
    :type transform_data: bool, optional
    :return: The data that will be clustered as a list of (N, w_t, d)
    arrays
    :rtype: List[np.ndarray]
    """
    # We keep the time dimension if we use DTW
    X_clus = []

    if pg._squared_radius and transform_data:
        # X of shape (N, d, T)
        # r = sqrt(RMM1**2 + RMM2**2)
        # r of shape (N, 1, T)
        r = np.sqrt(np.sum(np.square(X), axis=1, keepdims=True))
        # r*X gives the same angle but a squared radius
        X = r*X

    for t in range(pg.T):
        # X_clus: List of (N, w_t, d) arrays
        ind = window["origin"][t]
        X_clus.append(np.swapaxes(X[:,:,ind], 1, 2))
    return X_clus

def _sliding_window(T: int, w: int) -> dict:
    """
    Assuming that we consider an array of length `T`, and with indices
    `[0, ..., T-1]`.

    Windows extracted are shorter when considering the beginning and the
    end of the array. Which means that a padding is implicitly included.

    When the time window `w` is an *even* number, favor future time steps, i.e.,
    when extracting a time window around the datapoint t, the time window
    indices are [t - (w-1)//2, ... t, ..., t + w/2].
    When `w` is odd then the window is [t - (w-1)/2, ... t, ..., t + (w-1)/2]

    Which means that the padding is as follows:

    - beginning: (w-1)//2
    - end: w//2

    And that the original indices are as follows:

    - [0, ..., t + w//2], until t = (w-1)//2
    - [t - (w-1)//2, ..., t, ..., t + w//2] for datapoints in
    [(w-1)//2, ..., T-1 - w//2]
    - [t - (w-1)//2, ..., T-1] from t = T-1 - w//2
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply.
    - Note also that the left side of the end case is the same as the left
    side of the base case

    Window sizes:

    - (1 + w//2) at t=0, then t + (1 + w//2) until t = (w-1)//2
    - All datapoints from [(w-1)//2, ..., T-1 - w//2] have a normal
    window size.
    - (w+1)//2 at t=T-1, then T-1-t + (w+1)//2 from t = T-1 - w//2
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply

    Consider an extracted time window of length w_real (with w_real <= w,
    if the window was extracted at the beginning or the end of the array).
    The midpoint of the extracted window (i.e. the index in that window
    that corresponds to the datapoint around which the time window was
    extracted in the original array is:

    - 0 at t=0, then t, until t = pad_left, i.e. t = (w-1)//2
    - For all datapoints between, [(w-1)//2, ..., (T-1 - w//2)], the
    midpoint is (w-1)//2 (so it is the same as the base case)
    - w_real-1 at t=T-1, then w_real - (T-t), from t=T-1-pad_right, i.e.
    from t = (T-1 - w//2)
    - Note that for t = (w-1)//2 or t = (T-1 - w//2), both formulas apply.

    The midpoint in the original array is actually simply t
    """
    # Boundaries between regular cases and extreme ones
    ind_start = (w-1)//2
    ind_end = T-1 - w//2
    pad_l = (w-1)//2
    pad_r = (w//2)
    window = {}
    window["padding_left"] = pad_l
    window["padding_right"] = pad_r


    # For each t in [0, ..., T-1]...
    # ------- Real length of the time window -------
    window["length"] = [w for t in range(T)]
    window["length"][:ind_start] = [
        t + (1 + w//2)
        for t in range(0, ind_start)
    ]
    window["length"][ind_end:] = [
        T-1-t + (w+1)//2
        for t in range(ind_end, T)
    ]
    # ------- Midpoint in the window reference -------
    # Note that the end case is the same as the base case
    window["midpoint_w"] = [(w-1)//2 for t in range(T)]
    window["midpoint_w"][:ind_start]  = [ t for t in range(0, ind_start) ]
    # ------- Midpoint in the origin reference -------
    window["midpoint_o"] = [t for t in range(T)]
    # ------- Original indices -------
    window["origin"] = [
        # add a +1 to the range to include last original index
        list(range( (t - (w-1)//2),  (t + w//2) + 1))
        for t in range(T)
    ]
    window["origin"][:ind_start]  = [
        # add a +1 to the range to include last original index
        list(range(0, (t + w//2) + 1))
        for t in range(0, ind_start)
    ]
    window["origin"][ind_end:] = [
        # Note that the left side is the same as the base case
        list(range( (t - (w-1)//2),  (T-1) + 1 ))
        for t in range(ind_end, T)
    ]
    return window

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
) -> List:
    """
    Generate all clustering data

    :param pg: Persistent Graph
    :type pg: _type_
    :return: All data corresponding to the generated clustering in a nested list
    clusters_t_n. each element of the nested list contain [clusters, clusters_info]
    :rtype: List[Dict[int, Tuple[]]]
    """
    # --------------------------------------------------------------
    # --------------------- Preliminary ----------------------------
    # --------------------------------------------------------------
    # Use bisect left in order to favor the lowest number of clusters
    if pg._maximize:
        sort_fc = bisect_left
    else:
        sort_fc = reverse_bisect_left

    wind = _sliding_window(pg.T, pg.w)
    # all members_clus/params are lists of length T of arrays of
    # shape (N, w_t, d)
    members_clus = _data_to_cluster(pg, pg.members, wind)
    members_clus0 = _data_to_cluster(pg, pg.members_zero, wind)
    members_params = _data_to_cluster(pg, pg.members, wind, transform_data=False)
    members_params0 = _data_to_cluster(pg, pg.members_zero, wind, False)

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
                    clusters = clustering_model(
                        pg,
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
                pg,
                X = X,
                clusters_data = clusters_data_score,
                t = t,
            )
            if n_clusters == 0:
                pg._zero_scores[t] = score
                # don't insert the case k=0 in cluster_data
                # nor in local_steps, go straight to the next iteration
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