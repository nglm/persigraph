import numpy as np
from scipy.spatial.distance import sqeuclidean, cdist
from sklearn.metrics import pairwise_distances

SCORES_TO_MINIMIZE = [
        'inertia',
        'max_inertia',
        'min_inertia',
        'variance',
        'min_variance',
        'max_variance',
        'max_diameter',
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
                        np.mean(X[members]).reshape(-1, 1) ,
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
                    np.mean(X[members]).reshape(-1, 1) ,
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
                    np.mean(X[members]).reshape(-1, 1) ,
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
        score = 0
        for i_cluster, members in enumerate(clusters):
            score = max(
                np.amax(pairwise_distances(X[members])),
                score
            )
    else:
        #TODO: not implemented yet:
        raise ValueError(
                "Choose an available score_type"
                + str(SCORES_TO_MAXIMIZE + SCORES_TO_MINIMIZE)
            )

    return np.around(score, pg._precision)


def compute_zero_scores(pg):
    # inertia_scores = ['inertia', 'max_inertia', 'min_inertia']
    # variance_scores = ['variance', 'max_variance', 'min_variance']
    # diameter_scores = ['max_diameter']
    pg._zero_scores = np.zeros(pg.T)
    if pg._zero_type == 'bounds':

        # Get the parameters of the uniform distrib using min and max
        mins = np.amin(pg._members, axis = 0)
        maxs = np.amax(pg._members, axis = 0)

    elif pg._zero_type == 'var':
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
    for t in range(pg.T):
        pg._zero_scores[t] = compute_score(
            pg=pg,
            X= values[t].reshape(-1,1),
            clusters= [[i for i in range(pg.N)]]
        )

        # if pg._score_type in inertia_scores:
        #     pg._zero_scores = np.around(
        #         pg.N * np.var(pg._members, axis = 0),
        #         pg._precision
        #     )
        # elif pg._score_type in variance_scores:
        #     pg._zero_scores = np.around(
        #         np.var(pg._members, axis = 0),
        #         pg._precision
        #     )