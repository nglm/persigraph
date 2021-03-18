import numpy as np
from scipy.spatial.distance import sqeuclidean, cdist
from sklearn.metrics import pairwise_distances

def compute_score(pg, model=None, X=None, clusters=None):

    # ------------------------------------------------------------------
    if pg._score_type == 'inertia':
        if hasattr(model, 'inertia_'):
            score = model.inertia_
        else:
            #TODO: not implemented yet:
            raise NotImplementedError('inertia not implemented')

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
    elif pg._score_type == 'variance':
        score = 0
        for i_cluster, members in enumerate(clusters):
            score += len(members)/pg.N * np.var(X[members])

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
            score = max(pairwise_distances(X[members]), score)
    else:
        #TODO: not implemented yet:
        raise NotImplementedError(pg._score_type + ': not implemented score')

    return np.around(score, pg._precision)