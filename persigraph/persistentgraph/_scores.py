"""
This module deals with the score function and score ratio
It is totally clustering model independent
"""

import pycvi
from pycvi.compute_scores import best_score, worst_score
import numpy as np
from typing import List, Sequence, Union, Any, Dict, Tuple

SCORES = pycvi.SCORES
SUBSCORES = pycvi.compute_scores.SUBSCORES
MAIN_SCORES_TO_MINIMIZE = pycvi.compute_scores.MAIN_SCORES_TO_MINIMIZE
MAIN_SCORES_TO_MAXIMIZE = pycvi.compute_scores.MAIN_SCORES_TO_MAXIMIZE
SCORES_TO_MINIMIZE = [p+s for s in MAIN_SCORES_TO_MINIMIZE for p in SUBSCORES]
SCORES_TO_MAXIMIZE = [p+s for s in MAIN_SCORES_TO_MAXIMIZE for p in SUBSCORES]

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
        # _local_step n'est pas trié
        pg._worst_scores[t] = pg._score.worst_score([
            pg._local_steps[t][s]['score'] for s in len(pg._local_steps[t])
        ])
        pg._best_scores[t] = pg._score._best_score([
            pg._local_steps[t][s]['score'] for s in len(pg._local_steps[t])
        ])
    if pg._global_bounds:
        worst_score_global = pg._worst_scores[0]
        best_score_global = pg._best_scores[0]
        for worst_t, best_t in zip(
            pg._worst_scores[1:],
            pg._best_scores[1:]
        ):
            best_score_global = best_score(
                best_score_global,
                best_t,
                pg._score_maximize
            )
            worst_score_global = worst_score(
                worst_score_global,
                worst_t,
                pg._score_maximize
            )
        pg._worst_scores[:] = worst_score_global
        pg._best_scores[:] = best_score_global

    pg._norm_bounds = np.abs(pg._worst_scores - pg._best_scores)
    pg._are_bounds_known = True

def _compute_ratio_scores(
    pg,
):
    """
    Compute the ratio scores of local scores

    Note that 'ratio_score' of steps refer to the death ratio score.
    For more information on how life spans of steps are computed based on
    the ratio_score of steps, see `k_info`.

    :param pg: [description]
    :type pg: [type]
    """
    for t in range(pg.T):
        score_bounds = (pg._worst_scores[t], pg._best_scores[t])
        norm_bounds = np.abs(score_bounds[0] - score_bounds[1])

        # ------------------ ratio scores of local steps ---------------
        # Special case, all ratio score and life spans of that step will be
        # 0 expect for the case k=1, where
        # ratio_score=0 and life_span=1
        # See `k_info` for more info.
        if pg._worst_scores[t] == pg._best_scores[t]:
            for step in range(pg._nb_local_steps[t]):
                pg._local_steps[t][step]['ratio_score'] = 0
        else:

            for step in range(pg._nb_local_steps[t]):
                score = pg._local_steps[t][step]['score']
                ratio = np.abs(score - score_bounds[0]) / norm_bounds
                # If the score is absolute, we need a second round of
                # normalisation
                if True:
                    pass
                pg._local_steps[t][step]['ratio_score'] = ratio

