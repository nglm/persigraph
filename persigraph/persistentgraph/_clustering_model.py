"""
This module is supposed to manage any type of clustering model and to
call _pg_* for more model-dependant code.
"""
import numpy as np
from bisect import bisect_left
from sklearn.cluster import KMeans
from typing import List, Sequence, Union, Any, Dict, Tuple

from ._scores import compute_score
from ..utils.sorted_lists import reverse_bisect_left



def get_model_parameters(
    model_class,
    model_kw = {},
    fit_predict_kw = {},
) -> Tuple[Dict, Dict]:
    """
    Initialize clustering model parameters

    :return: 2 dict, for the model initialization and its fit method
    :rtype: Tuple[Dict, Dict]
    """
    m_kw = {}
    ft_kw = {}
    # ----------- method specific key-words ------------------------
    if model_class == KMeans:
        m_kw = {
            'max_iter' : model_kw.pop('max_iter', 100),
            'n_init' : model_kw.pop('n_init', 20),
            'tol' : model_kw.pop('tol', 1e-3),
        }
        m_kw.update(model_kw)
        ft_kw.update(fit_predict_kw)

    elif model_class == "Naive":
        raise NotImplementedError

    return m_kw, ft_kw


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
    values = np.array([mins + i*steps for i in range(pg.N)])

    # ==================== clusters, cluster_info ======================
    # Compute the score of that distribution
    clusters = [[i for i in range(pg.N)]]
    clusters_info = [{
        'type' : 'uniform',
        'params' : [values[0], values[-1]], # lower/upper bounds
        'brotherhood_size' : [0],
        'mean' : (values[0] + values[-1])/2
    }]

    # ========================== step_info =============================
    step_info = {'values': values}

    return clusters, clusters_info, step_info

def clustering_model(
    pg,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
) -> Tuple[List[List[int]], List[Dict], Dict, Dict]:
    """
    Generate a clustering instance with the given model/fit parameters

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param model_kw: Dict of kw for the model initalization, defaults to {}
    :type model_kw: dict, optional
    :param fit_predict_kw: Dict of kw for the fit_predict method, defaults to {}
    :type fit_predict_kw: dict, optional
    :return: All data corresponding to the generated clustering
    :rtype: Tuple[List[List[int]], List[Dict], Dict, Dict]
    """
    n_clusters = model_kw[pg._model_class_kw["k_arg_name"]]
    X = fit_predict_kw[pg._model_class_kw["X_arg_name"]]
    if n_clusters == 0:
        return generate_zero_component(pg, X, model_kw, fit_predict_kw)
    else:
        # ====================== Fit & predict part =======================
        model = pg._model_class(**model_kw)
        labels = model.fit_predict(**fit_predict_kw)

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
            info = {}
            clusters_info.append(info)

        # ========================== step_info =============================
        # Add method specific info here if necessary
        # (Score is computed in persistentgraph)
        step_info = {}

    return clusters, clusters_info, step_info


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
        X = pg._members[:, :, t]
        # Get clustering model parameters required by the
        # clustering model
        model_kw, fit_predict_kw = get_model_parameters(
            pg._model_class,
            model_kw = pg._model_kw,
            fit_predict_kw = pg._fit_predict_kw,
        )
        fit_predict_kw[pg._model_class_kw["X_arg_name"]] = X

        for n_clusters in pg._n_clusters_range:

            # Update model_kw
            model_kw[pg._model_class_kw["k_arg_name"]] = n_clusters

            # ---------- Fit & predict using clustering model-------
            try :
                (
                    clusters,
                    clusters_info,
                    step_info,
                ) = clustering_model(
                    pg,
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