import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import sqeuclidean, cdist, euclidean
from sklearn.metrics import pairwise_distances
from typing import List, Sequence, Union, Any, Dict

SCORES_TO_MINIMIZE = [
        'inertia',
        'mean_inertia',
        'weighted_inertia',
        'max_inertia',
        'min_inertia',       # Shouldn't be used: taking min makes no sense
        # ----------
        'distortion'
        # ----------         # Shouldn't be used: use inertia instead
        'variance',          # Shouldn't be used: use inertia instead
        'mean_variance',     # Shouldn't be used: use inertia instead
        'weighted_variance', # Shouldn't be used: favors very high k values
        'min_variance',      # Shouldn't be used: taking min makes no sense
        'max_variance',
        # ----------
        'diameter',      # WARNING: diameter should be used with weights
        'max_diameter',  # WARNING: Max diameter should be used with weights
        # ----------
        'MedDevMean',
        'mean_MedDevMean',
        'max_MedDevMean',
        # ----------
        'max_MedDevMed', # Shouldn't be used: see details below
]

SCORES_TO_MAXIMIZE = []


def _set_score_type(pg, score_type):
    if score_type in SCORES_TO_MAXIMIZE:
        pg._maximize = True
    elif score_type in SCORES_TO_MINIMIZE:
        pg._maximize = False
    else:
        raise ValueError(
            "Choose an available score_type"
            + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
        )
    if score_type in ['max_diameter']:
        pg._global_bounds = True
    else:
        pg._global_bounds = False
    pg._score_type = score_type

def compute_score(
    pg,
    model=None,
    X: np.ndarray = None,
    clusters: List[List[int]] = None,
    t: int = None
) -> float :
    """
    Compute the score of a given clustering

    :param pg:
    :type pg: PersistentGraph
    :param model: sklearn model, defaults to None
    :type model: sklearn model, optional
    :param X: Values of all members, defaults to None
    :type X: np.ndarray, shape: (N, d) optional
    :param clusters: Members ids of each cluster, defaults to None
    :type clusters: List[List[int]], optional
    :param t: current time step (for weights), defaults to None
    :type t: int, optional
    :raises ValueError: [description]
    :return: Score of the given clustering
    :rtype: float
    """
    #HERE_done

    # TODO: add weights for scores that requires global bounds

    # ------------------------------------------------------------------
    if pg._score_type == 'inertia':
        # sklearn gives the inertia
        if (model is not None) and hasattr(model, 'inertia_'):
            score = model.inertia_
        else:
            score = 0
            for i_cluster, members in enumerate(clusters):
                score += np.sum(cdist(
                        X[members],
                        np.mean(X[members], axis=0).reshape(-1),
                        metric='sqeuclidean'
                        )
                    )
    # ------------------------------------------------------------------
    elif pg._score_type in ['mean_inertia', 'distortion']:
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.sum(cdist(
                    X[members],
                    np.mean(X[members], axis=0).reshape(-1),
                    metric='sqeuclidean'
                    )
                )
        score /= len(clusters)
    # ------------------------------------------------------------------
    elif pg._score_type == 'weighted_inertia':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += (len(clusters) / pg.N)* np.sum(cdist(
                    X[members],
                    np.mean(X[members], axis=0).reshape(-1),
                    metric='sqeuclidean'
                    )
                )

    # ------------------------------------------------------------------
    elif pg._score_type == 'max_inertia':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                score,
                np.sum(cdist(
                    X[members],
                    np.mean(X[members], axis=0).reshape(-1),
                    metric='sqeuclidean'
                    )
                ))

    # ------------------------------------------------------------------
    # Shouldn't be used: taking min makes no sense
    elif pg._score_type == 'min_inertia':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(
                score,
                np.sum(cdist(
                    X[members],
                    np.mean(X[members], axis=0).reshape(-1),
                    metric='sqeuclidean'
                    )
                ))

    # ------------------------------------------------------------------
    elif pg._score_type in ['variance']:
        # Shouldn't be used: use inertia or distortion instead
        # Don't confuse generalized variance and total variation
        # Here it's generalized variance
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.linalg.det(np.cov(X[members], rowvar=True))
            #score += np.var(X[members], ddof=1)
    # ------------------------------------------------------------------
    elif pg._score_type in ['mean_variance']:
        # Shouldn't be used: use inertia instead
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.linalg.det(np.cov(X[members], rowvar=True))
        score /= len(clusters)
    # ------------------------------------------------------------------
    elif pg._score_type in ['weighted_variance']:
        # Shouldn't be used: use inertia instead
        # This should not be used, it favours very high values of k
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += (
                (len(clusters)/pg.N)
                * np.linalg.det(np.cov(X[members], rowvar=True))
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_variance':
        # Shouldn't be used: use inertia instead
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(np.linalg.det(np.cov(X[members], rowvar=True)), score)

    # ------------------------------------------------------------------
    # Shouldn't be used: taking min makes no sense
    # Shouldn't be used: use inertia instead
    elif pg._score_type == 'min_variance':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(np.linalg.det(np.cov(X[members], rowvar=True)), score)
    # ------------------------------------------------------------------
    elif pg._score_type == 'diameter':
        # WARNING: diameter should be used with weights
        # WARNING: Should not be used..... Does not penalize enough high
        # values of n_clusters
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.amax(pairwise_distances(X[members])) * pg._weights[:, t]
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_diameter':
        # WARNING: Max diameter should be used with weights
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.amax(pairwise_distances(X[members])) * pg._weights[:, t],
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.median(norm(X[members] - np.mean(X[members], axis=0)))
    # ------------------------------------------------------------------
    elif pg._score_type == 'mean_MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.median(norm(X[members] - np.mean(X[members], axis=0)))
        score /= len(clusters)
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.mean(X[members], axis=0))),
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'MeanDevMed':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.mean(norm(X[members] - np.median(X[members], axis=0)))
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MeanDevMed':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.mean(norm(X[members] - np.median(X[members], axis=0))),
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MedDevMed':
        # This would not penalize a vertex containing 2 clusters with
        # slightly different size
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.median(X[members], axis=0))),
                score
            )
    else:
        raise ValueError(
                "Choose an available score_type"
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )

    return np.around(score, pg._precision)


def _compute_score_bounds(
    pg,
    local_scores: List[float],
) -> None:
    """
    Compare local_scores and zero_scores at t to find score bounds at t

    :param pg: [description]
    :type pg: [type]
    :param local_scores: scores computed at a given time step
    :type local_scores: List[float]
    """
    for t in range(pg.T):
        pg._worst_scores[t] = worst_score(
            pg, pg._zero_scores[t], local_scores[t][-1]
        )
        pg._best_scores[t] = best_score(
            pg, pg._zero_scores[t], local_scores[t][0]
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
            ratio_score = (
                np.abs(score - score_bounds[0])
                / np.abs(score_bounds[0] - score_bounds[1])
            )
    return ratio_score

def _compute_ratio_scores(
    pg,
):
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

def _compute_cluster_params(
    cluster: np.ndarray,
) -> Dict:
    """
    Compute the mean, std, std_sup, inf, etc of a given cluster

    :param cluster: Values of the members belonging to that cluster
    :type cluster: np.ndarray, shape (N_members, d)
    :return: Dict of summary statistics
    :rtype: Dict
    """
    #HERE_done
    d = cluster.shape[1]
    mean = np.mean(cluster, axis=0).reshape(-1)  # shape: (d)
    std = np.std(cluster, axis=0).reshape(-1)    # shape: (d)
    # Get the members below/above the average
    X_inf = np.array(
        [[m for i in range(d) if m < mean[i]] for m in cluster]
    )
    X_sup = np.array(
        [[m for i in range(d) if m >= mean[i]] for m in cluster]
    )
    # How many members above/below the average
    n_inf = len(X_inf)
    n_sup = len(X_sup)
    # if statement because of '<': No member below the average => std=0
    if n_inf == 0:
        std_inf = 0
    else:
        std_inf = np.sqrt(np.sum((X_inf - mean)**2, axis=0) / n_inf).reshape(-1)
    std_sup = np.sqrt(np.sum((X_sup - mean)**2) / n_sup).reshape(-1)

    cluster_params = {}
    cluster_params['mean'] = mean
    cluster_params['std'] = std
    cluster_params['std_inf'] = std_inf
    cluster_params['std_sup'] = std_sup
    return cluster_params


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

