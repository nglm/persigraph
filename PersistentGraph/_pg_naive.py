import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Sequence, Union, Any, Dict
from bisect import insort
from scipy.spatial.distance import sqeuclidean, cdist
from typing import List, Sequence, Union, Any, Dict

from ._scores import _compute_cluster_params
from ..utils.sorted_lists import insert_no_duplicate
from ..utils._clustering import get_centroids



def get_model_parameters(
    pg,
    X: np.ndarray = None,
    t: int = None,
):
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
    model_kw = {}
    fit_predict_kw = {}
    return model_kw, fit_predict_kw


def _fit(
    pg,
    X,
    model_kw : dict = {},
    fit_predict_kw : dict = {},
):

    # First step: only one cluster
    if pg._model_kw['idx'] == 0:
        members_r = [0 for _ in range(pg.N)]
        rep_new = [0]
        idx = 1

    # General case
    else:
        # Get new reprensatives and the index in the distance matrix
        rep_new, idx = get_centroids(
            distance_matrix = pg._model_kw['distance_matrix'],
            sorted_idx = pg._model_kw['sorted_idx'],
            idx = pg._model_kw['idx'],
            members_r = pg._model_kw['rep'],
        )

        # extract distance to representatives
        dist = []
        for r in rep_new:
            dist.append(pg._model_kw['distance_matrix'][r])

        dist = np.asarray(dist)     # (nb_rep, N) array
        # for each member, find the representative that is the closest
        members_r = [rep_new[r] for r in np.nanargmin(dist, axis=0)]

    if rep_new == []:
        raise ValueError('No new clusters')

    pg._model_kw['rep'] = members_r
    pg._model_kw['idx'] = idx

    return rep_new, members_r, model_kw

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
    rep, members_r, model_kw = _fit(
        pg,
        X = X,
        model_kw = model_kw,
        fit_predict_kw = fit_predict_kw
    )

    members_r = pg._model_kw['rep']
    n_clusters = model_kw.pop('n_clusters')
    if len(rep) < n_clusters:
        raise ValueError('No members in cluster')

    # ================= clusters, cluster_info =================
    clusters_info = []
    clusters = []
    for i_cluster in range(n_clusters):

        # Members belonging to that cluster
        members = [
            m for m in range(pg.N) if members_r[m] == rep[i_cluster]
        ]
        clusters.append(members)

        # Info related to this specific vertex
        cluster_params =  _compute_cluster_params(X[members])
        cluster_params.append(rep[i_cluster])
        clusters_info.append({
            'type' : 'Naive',
            'params' : cluster_params,
            'brotherhood_size' : [n_clusters]
            })

    # ====================== step_info =========================
    # Add method specific info here if necessary
    # (Score is computed in persistentgraph)
    step_info = {}

    return clusters, clusters_info, step_info, model_kw
