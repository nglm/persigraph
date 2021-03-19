import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import sqeuclidean, cdist, euclidean
from sklearn.metrics import pairwise_distances

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


def set_score_type(pg, score_type):
    if pg._model_type == "Naive":
        pg._maximize = False
        pg._score_type = "max_diameter"
    else:
        if score_type in SCORES_TO_MAXIMIZE:
            pg._maximize = True
        elif score_type in SCORES_TO_MINIMIZE:
            pg._maximize = False
        else:
            raise ValueError(
                "Choose an available score_type"
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )
        pg._score_type = score_type

def compute_score(pg, model=None, X=None, clusters=None):

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
    if pg._score_type == 'weighted_inertia':
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
    elif pg._score_type == 'max_MedDevMean':
        # Max median deviation
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.mean(X[members]))),
                score
            )
    # ------------------------------------------------------------------
    elif pg._score_type == 'max_MedDevMed':
        # Max median deviation
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.median(norm(X[members] - np.median(X[members]))),
                score
            )

    # ------------------------------------------------------------------
    elif pg._score_type == 'max_diameter':
        # WARNING: Max diameter should be used with weights + shared
        # score bounds!
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.amax(pairwise_distances(X[members])),
                score
            )
    elif pg._score_type == 'MedDevMean':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += (pg.N/len(members) *
                np.median(norm(X[members] - np.mean(X[members])))
            )

    else:
        raise ValueError(
                "Choose an available score_type"
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )

    return np.around(score, pg._precision)


def compute_zero_scores(pg):
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
            clusters= members
        )




def _is_earlier_score(pg, score1, score2, or_equal=True):
    return (
        (better_score(pg, score1, score2, or_equal) != pg._score_is_improving)
        or (score1 == score2 and or_equal)
    )

def _is_relevant_score(
    pg,
    previous_score,
    score,
    or_equal = True,
):
    curr_is_better = better_score(pg, score, previous_score, or_equal=False)
    res = (
        (curr_is_better == pg._score_is_improving)
        or (or_equal and previous_score == score)
    )
    return res

def better_score(pg, score1, score2, or_equal=False):
    # None means that the score has not been reached yet
    # So None is better if score is improving
    if score1 is None:
        return pg._score_is_improving
    elif score2 is None:
        return not pg._score_is_improving
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

def _compute_ratio_score(
    score,
    score_bounds = None,
):
    """
    Inspired by the similar method in component
    """
    if score_bounds is None or score is None:
        ratio_score = None
    else:
        # Normalizer so that ratios are within 0-1 range
        norm = euclidean(score_bounds[0], score_bounds[1])

        if score is None:
            ratio_score = 1
        else:
            ratio_score = euclidean(score, score_bounds[0]) / norm

    return ratio_score