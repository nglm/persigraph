"""
This module deals with the score function and score ratio
It is totally clustering model independent
"""

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import pairwise_distances
from tslearn.metrics import cdist_soft_dtw
from typing import List, Sequence, Union, Any, Dict

SCORES = [
        'inertia',
        'mean_inertia',  # mean inertia and distortion are the same thing
        'weighted_inertia',
        'max_inertia',
        # ----------
        #'variance',     #TODO: Outdated since DTW
        # ----------
        #"diameter",     #TODO: Outdated since DTW
        #"max_diameter", #TODO: Outdated since DTW
        # ----------
        'MedDevMean',
        'mean_MedDevMean',
        'max_MedDevMean',
]

SUBSCORES = ["", "mean_", "median_", "weighted_", "min_", "max_"]

MAIN_SCORES_TO_MINIMIZE = [
    "inertia",
    #"variance",            #TODO: Outdated since DTW
    "MedDevMean",
    #"MeanDevMed",          #TODO: Outdated since DTW
    #"MedDevMed",           #TODO: Outdated since DTW
    #'diameter',            #TODO: Outdated since DTW
]

MAIN_SCORES_TO_MAXIMIZE = []

SCORES_TO_MINIMIZE = [p+s for s in MAIN_SCORES_TO_MINIMIZE for p in SUBSCORES]
SCORES_TO_MAXIMIZE = [p+s for s in MAIN_SCORES_TO_MAXIMIZE for p in SUBSCORES]


def f_inertia(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the inertia of ONE cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c, or (N_c, w, d) if DTW is used
    :type cluster: np.ndarray
    :return: inertia of that cluster
    :rtype: float
    """
    dims = cluster.shape
    if len(dims) == 2:
        score = cdist(
            cluster,
            cluster_info["mean"].reshape(1, -1),
            metric='sqeuclidean'
        )
    if len(dims) == 3:
        barycenter = np.expand_dims(cluster_info["barycenter"], 0)
        score = cdist_soft_dtw(
            cluster,
            barycenter,
            gamma=0.1
        )
    # Note cdist_soft_dtw_normalized should return positive values but
    # somehow doesn't!
    return np.sum(score)

def f_generalized_var(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the sample generalized variance of ONE cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: generalized variance of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        # a dxd array, representing the sample covariance
        sample_cov = np.cov(cluster, rowvar=False)
        if len(sample_cov.shape) > 1:
            return np.linalg.det(sample_cov)
        else:
            return sample_cov

def f_med_dev_mean(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the median deviation around the mean

    The variance can be seen as the mean deviation around the mean

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the median deviation around the mean of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        dims = cluster.shape
        if len(dims) == 2:
            return np.median(cdist(
                cluster,
                cluster_info["mean"].reshape(1, -1),
                metric='sqeuclidean'
            ))
        if len(dims) == 3:
            return np.median(cdist_soft_dtw(
                cluster,
                cluster_info["barycenter"],
                gamma=0.1
            ))

def f_mean_dev_med(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the mean deviation around the median

    The variance can be seen as the mean deviation around the mean

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the mean deviation around the median of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        dims = cluster.shape
        if len(dims) == 2:
            return np.mean(cdist(
                cluster,
                cluster_info["mean"].reshape(1, -1),
                metric='sqeuclidean'
            ))
        if len(dims) == 3:
            return np.mean(cdist_soft_dtw(
                cluster,
                cluster_info["barycenter"],
                gamma=0.1
            ))



def f_med_dev_med(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the median deviation around the median

    The variance can be seen as the mean deviation around the mean.
    Note that this would not penalize a vertex containing 2 clusters
    with slightly different size

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the mean deviation around the median of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        return np.median(cdist(
            cluster,
            np.median(cluster, axis=0, keepdims=True),
            metric='sqeuclidean'
        ))

def f_diameter(cluster: np.ndarray, cluster_info: dict = None) -> float:
    """
    Compute the diameter of the given cluster

    :param cluster: (N_c, d) array, representing a cluster of size N_c
    :type cluster: np.ndarray
    :return: the diameter of that cluster
    :rtype: float
    """
    if len(cluster) == 1:
        return 0
    else:
        return np.amax(pairwise_distances(cluster))

def compute_subscores(
    pg,
    X : np.ndarray,
    clusters: List[List[int]],
    main_score: str,
    f_score,
    clusters_info: Dict = None,
) -> float:
    """
    Compute the main score of a clustering and its associated subscores

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param X: Values of all members
    :type X: np.ndarray, shape: (N, d)
    :param clusters: Members ids of each cluster, defaults to None
    :type clusters: List[List[int]]
    :return: Score of the given clustering
    :rtype: float
    """
    prefixes = ["", "mean_", "weighted_"]
    score_tmp = [
            f_score(X[members], info)
            for members, info in zip(clusters, clusters_info)
        ]
    if (pg._score_type in [p + main_score for p in prefixes] ):
        score = sum(score_tmp)
        # Take the mean score by cluster
        if pg._score_type  == "mean_" + main_score:
            score /= len(clusters)
        # Take a weighted mean score by cluster
        elif pg._score_type == "weighted_" + main_score:
            score /= (len(clusters) / pg.N)
    # ------------------------------------------------------------------
    # Take the median score among all clusters
    elif pg._score_type == 'median_' + main_score:
        score = np.median(score_tmp)
    # ------------------------------------------------------------------
    # Take the max score among all clusters
    elif pg._score_type == 'max_' + main_score:
        score = max(score_tmp)
    # ------------------------------------------------------------------
    # Shouldn't be used: taking min makes no sense
    # Take the min score among all clusters
    elif pg._score_type == 'min_' + main_score:
        score = min(score_tmp)
    else:
        raise ValueError(
                pg._score_type + " has an invalid prefix."
                + "Please choose a valid score_type: "
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )
    return score


def compute_score(
    pg,
    model=None,
    X: np.ndarray = None,
    clusters: List[List[int]] = None,
    t: int = None,
    clusters_info: Dict = None,
) -> float :
    """
    Compute the score of a given clustering

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    :param model: sklearn model, defaults to None
    :type model: sklearn model, optional
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d*w) optional
    :param clusters: Members ids of each cluster, defaults to None
    :type clusters: List[List[int]], optional
    :param t: current time step (for weights), defaults to None
    :type t: int, optional
    :raises ValueError: [description]
    :return: Score of the given clustering
    :rtype: float
    """

    # TODO: add weights for scores that requires global bounds

    # ------------------------------------------------------------------
    # Inertia-based scores
    if (pg._score_type.endswith("inertia")):
        # sklearn gives the inertia
        if (
            pg._score_type == "inertia" and model is not None
            and hasattr(model, 'inertia_')
        ):
            score = model.inertia_
        else:
            score = compute_subscores(
                pg, X, clusters, "inertia", f_inertia,
                clusters_info=clusters_info)
    # ------------------------------------------------------------------
    # Variance based scores
    # Shouldn't be used: use inertia or distortion instead
    # Don't confuse generalized variance and total variation
    # Here it's generalized variance
    elif pg._score_type.endswith("variance"):
        score = compute_subscores(
            pg, X, clusters, "variance", f_generalized_var,
            clusters_info=clusters_info
        )
    # ------------------------------------------------------------------
    # Median around mean based scores
    elif pg._score_type.endswith("MedDevMean"):
        score = compute_subscores(
            pg, X, clusters, "MedDevMean", f_med_dev_mean,
            clusters_info=clusters_info)
    # Median around mean based scores
    elif pg._score_type.endswith("MeanDevMed"):
        score = compute_subscores(
            pg, X, clusters, "MeanDevMed", f_mean_dev_med,
            clusters_info=clusters_info)
    # Median around median based scores
    # Shouldn't be used, see f_med_dev_med
    elif pg._score_type.endswith("MedDevMed"):
        score = compute_subscores(
            pg, X, clusters, "MedDevMed", f_med_dev_med,
            clusters_info=clusters_info)
    # ------------------------------------------------------------------
    elif pg._score_type.endswith("diameter"):
        score = compute_subscores(
            pg, X, clusters, "diameter", f_diameter,
            clusters_info=clusters_info)
    # ------------------------------------------------------------------
    else:
        raise ValueError(
                pg._score_type
                + " is invalid. Please choose a valid score_type: "
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )
    if score < 0:
        raise ValueError(
            "Score can't be negative: " + str(score)
        )
    return np.around(score, pg._precision)


def _compute_score_bounds(
    pg,
) -> None:
    """
    Compare local_scores and zero_scores at t to find score bounds at t.
    The case k=0 is used only to be used as a potential score bound.
    It is never used to create a vertex in the graph and it doesn't have
    a life span.

    The score bounds are used to compute the ratio scores.
    By convention:
    - k_worst has life_span=0, r_birth=0 and r_death=0
    - k_best has r_death=1

    If all scores are equal, life_span=0, r_birth=0 and r_death=0 for all k
    except for k=1, where life_span=1, r_birth=0 and r_death=1

    :param pg: [description]
    :type pg: [type]
    """
    for t in range(pg.T):
        pg._worst_scores[t] = worst_score(
            pg, pg._zero_scores[t], pg._local_steps[t][-1]['score']
        )
        if pg._worst_scores[t] != pg._zero_scores[t]:
            # k that will automatically get a life span of 0
            pg._worst_k[t] = pg._local_steps[t][-1]['param']["n_clusters"]
        pg._best_scores[t] = best_score(
            pg, pg._zero_scores[t], pg._local_steps[t][0]['score']
        )
    if pg._global_bounds:
        worst_score_global = pg._worst_scores[0]
        best_score_global = pg._best_scores[0]
        for worst_t, best_t in zip(
            pg._worst_scores[1:],
            pg._best_scores[1:]
        ):
            worst_score_global = worst_score(
                pg,
                worst_score_global,
                worst_t
            )
            worst_score_global = worst_score(
                pg,
                worst_score_global,
                worst_t
            )
        pg._worst_scores[:] = worst_score_global
        pg._best_scores[:] = best_score_global

    pg._norm_bounds = np.abs(pg._worst_scores - pg._best_scores)
    pg._are_bounds_known = True


def _compute_ratio(
    score,
    score_bounds=None,
):
    """
    Inspired by the similar method in component
    """
    if score_bounds is None:
        ratio_score = None
    else:
        if score is None:
            ratio_score = 1
        else:
            # SPECIAL CASE
            if score_bounds[0] == score_bounds[1]:
                return 0
            else:
                # Normalizer so that ratios are within 0-1 range
                norm = np.abs(score_bounds[0] - score_bounds[1])
                return np.abs(score-score_bounds[0]) / norm

def _compute_ratio_scores(
    pg,
):
    """
    Compute the ratio scores and life span of local scores and vertices.

    Note that 'ratio_score' of steps refer to the birth ratio score.
    For more information on how life spans of steps are computed based on
    the ratio_score of steps, see `get_k_life_span`.

    Update the pg._max_life_span if relevant

    :param pg: [description]
    :type pg: [type]
    """
    for t in range(pg.T):
        score_bounds = (pg._best_scores[t], pg._worst_scores[t])

        # Ratios for local step scores
        for step in range(pg._nb_local_steps[t]):
            score = pg._local_steps[t][step]['score']
            ratio = _compute_ratio(score, score_bounds)
            pg._local_steps[t][step]['ratio_score'] = ratio

        # Compute score ratio of vertices that are still alive at the end
        for v in pg._v_at_step[t]['v'][-1]:
            pg._vertices[t][v]._compute_ratio_scores(
                score_bounds = score_bounds
            )
        if pg._v_at_step[t]['v'][-1]:
                # Get the longest life span
                pg._max_life_span = max(pg._max_life_span, max(
                    [
                        pg._vertices[t][v].life_span
                        for v in pg._v_at_step[t]['v'][-1]
                    ]
                ))




def better_score(pg, score1, score2, or_equal=False):
    # None means that the score has not been reached yet
    # So None is better if score is improving
    if score1 is None:
        return score1
    elif score2 is None:
        return score2
    elif score1 == score2:
        return or_equal
    elif score1 > score2:
        return pg._maximize
    elif score1 < score2:
        return not pg._maximize
    else:
        print(score1, score2)
        raise ValueError("Better score not determined")


def argbest(pg, score1, score2):
    if better_score(pg, score1, score2):
        return 0
    else:
        return 1

def best_score(pg, score1, score2):
    if argbest(pg, score1, score2) == 0:
        return score1
    else:
        return score2

def argworst(pg, score1, score2):
    if argbest(pg, score1, score2) == 0:
        return 1
    else:
        return 0

def worst_score(pg, score1, score2):
    if argworst(pg, score1, score2) == 0:
        return score1
    else:
        return score2

