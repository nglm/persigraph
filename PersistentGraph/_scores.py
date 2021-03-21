import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import sqeuclidean, cdist, euclidean
from sklearn.metrics import pairwise_distances
from typing import List, Sequence, Union, Any, Dict

SCORES_TO_MINIMIZE = [
        'inertia',
        'max_inertia',
        'min_inertia',
        'variance',
        'min_variance',
        'max_variance',
        'max_diameter',
        'max_MedDevMean',
        'max_MedDevMed',
        'MedDevMean',
        'weighted_inertia',
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

def compute_score(pg, model=None, X=None, clusters=None, t=None):
    # TODO: add weights for scores that requires global bounds

    # ------------------------------------------------------------------
    if pg._score_type == 'inertia':
        if (model is not None) and hasattr(model, 'inertia_'):
            score = model.inertia_
        else:
            score = 0
            for i_cluster, members in enumerate(clusters):
                score += np.sum(cdist(
                        X[members],
                        np.mean(X[members], keepdims=True) ,
                        metric='sqeuclidean'
                        )
                    )
    # ------------------------------------------------------------------
    elif pg._score_type == 'weighted_inertia':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += pg.N/len(members)*np.sum(cdist(
                    X[members],
                    np.mean(X[members], keepdims=True) ,
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
                    np.mean(X[members], keepdims=True) ,
                    metric='sqeuclidean'
                    )
                ))

    # ------------------------------------------------------------------
    elif pg._score_type == 'min_inertia':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(
                score,
                np.sum(cdist(
                    X[members],
                    np.mean(X[members], keepdims=True) ,
                    metric='sqeuclidean'
                    )
                ))

    # ------------------------------------------------------------------
    elif pg._score_type in ['variance', 'distortion']:
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += len(members-1)/pg.N * np.var(X[members])

    # ------------------------------------------------------------------
    elif pg._score_type == 'max_variance':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(np.var(X[members]), score)

    # ------------------------------------------------------------------
    elif pg._score_type == 'min_variance':
        score = np.inf
        for i_cluster, members in enumerate(clusters):
            score = min(np.var(X[members]), score)
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_diameter':
        # WARNING: Max diameter should be used with weights
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.amax(pairwise_distances(X[members])) / pg._weights[t],
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.median(norm(X[members] - np.mean(X[members])))
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.mean(X[members]))),
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'MeanDevMed':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += np.mean(norm(X[members] - np.median(X[members])))
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MeanDevMed':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.mean(norm(X[members] - np.median(X[members]))),
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MedDevMed':
        # This would not penalize a vertex containing 2 clusters with
        # slightly different size
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.median(X[members]))),
                score
            )
    else:
        raise ValueError(
                "Choose an available score_type"
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )

    return np.around(score, pg._precision)


def _compute_zero_scores(pg):
    pg._zero_scores = np.zeros(pg.T)
    if pg._zero_type == 'bounds':

        # Get the parameters of the uniform distrib using min and max
        mins = np.amin(pg._members, axis = 0)
        maxs = np.amax(pg._members, axis = 0)

    else:
        # I'm not sure if this type should be used at all actually.....
        # Get the parameters of the uniform distrib using mean and variance
        var = np.var(pg._members, axis = 0)
        mean = np.mean(pg._members, axis = 0)

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2

    # Generate a perfectly uniform distribution
    steps = (maxs-mins) / (pg.N-1)
    values = np.array(
        [[mins[t] + i*steps[t] for i in range(pg.N)] for t in range(pg.T)]
    )

    # Compute the score of that distribution
    members = [[i for i in range(pg.N)]]
    for t in range(pg.T):
        pg._zero_scores[t] = compute_score(
            pg=pg,
            X = values[t].reshape(-1,1),
            clusters= members,
            t = t,
        )


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

def _compute_cluster_params(cluster):

    mean = np.mean(cluster)
    std = np.std(cluster)
    X_inf = np.array([m for m in cluster if m < mean])
    X_sup = np.array([m for m in cluster if m >= mean])
    # std_inf
    n_inf = len(X_inf)
    n_sup = len(X_sup)
    if n_inf == 0:
        std_inf = 0
    else:
        std_inf = np.sqrt( 1/n_inf * np.sum((mean-X_inf)**2) )
    if n_sup == 0:
        std_sup = 0
    else:
        std_sup = np.sqrt( 1/n_sup * np.sum((mean-X_sup)**2) )
    cluster_params = [mean, std, std_inf, std_sup]
    return cluster_params


# def _is_earlier_score(pg, score1, score2, or_equal=True):
#     return (
#         (better_score(pg, score1, score2, or_equal) != pg._score_is_improving)
#         or (score1 == score2 and or_equal)
#     )

# def _is_relevant_score(
#     pg,
#     previous_score,
#     score,
#     or_equal = True,
# ):
#     curr_is_better = better_score(pg, score, previous_score, or_equal=False)
#     res = (
#         (curr_is_better == pg._score_is_improving)
#         or (or_equal and previous_score == score)
#     )
#     return res

def better_score(pg, score1, score2, or_equal=False):
    # None means that the score has not been reached yet
    # So None is better if score is improving
    if score1 is None:
        #return pg._score_is_improving
        return score1
    elif score2 is None:
        #return not pg._score_is_improving
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

