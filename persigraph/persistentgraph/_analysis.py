import numpy as np

from typing import List, Dict, Tuple, Any

from ..utils.lists import flatten
from .component import Component


def stats(components: List[List[Component]]) -> Dict[str, float]:
    """
    Compute basic statitistics on ``components``

    FIXME: Outdated
    Statistics available:

      - 'mean_ratio_life'
      - 'std_ratio_life'
      - 'min_ratio_life'
      - 'max_ratio_life'
      - 'mean_ratio_members'
      - 'std_ratio_members'
      - 'min_ratio_members'
      - 'max_ratio_members'

    :param components: List of graph components (vertices or edges)
    :type components: List[List[Component]]
    :return: A dictionary containing basic statitistics
    :rtype: Dict[str, float]
    """
    # Flatten the list, the time step information is not necessary here
    flat_cmpts = flatten(components)
    ratio_life = np.array([c.ratio_life for c in flat_cmpts])
    ratio_members = np.array([c.ratio_members for c in flat_cmpts])
    stats = {}
    stats['mean_ratio_life'] = np.mean(ratio_life)
    stats['std_ratio_life'] = np.std(ratio_life)
    stats['min_ratio_life'] = np.amin(ratio_life)
    stats['max_ratio_life'] = np.amax(ratio_life)
    stats['mean_ratio_members'] = np.mean(ratio_members)
    stats['std_ratio_members'] = np.std(ratio_members)
    stats['min_ratio_members'] = np.amin(ratio_members)
    stats['max_ratio_members'] = np.amax(ratio_members)
    return stats

def sort_components_by(components, criteron="life_span", descending=True):
    # components must be a nested list
    if not isinstance(components[0], list):
        components = [components]
    sorted_components = []
    def get_life_span(component):
        return component.life_span
    def get_ratio_members(component):
        return component.ratio_members
    if criteron=="ratio_members":
        key_func = get_ratio_members
    else:
        key_func = get_life_span
    for cmpts_t in components:
        sort_t = cmpts_t.copy()
        sort_t.sort(reverse=descending, key=key_func)
        sorted_components.append(sort_t)
    return sorted_components

def k_info(g) -> Dict[int, Dict[str, List[float]]]:
    """
    Get the life span and ratios for all k and t.

    To summarize:
    `k_info[k_curr]['life_span'][t] = r_curr - r_prev`

    Note:
    - g._local_steps[t][s]['ratio_score'] refers to the ratio of death of
    the local step.
    - there might be some 'holes' when steps are ignored
    their life span will then all be 0.
    - In case of equal r_scores, the smallest k value
    will be favored, the other ones keep their life span of value 0.
    - The case k=0 is used only to be used as a potential score bound.
    It is never used to create a vertex in the graph and it doesn't
    really have a life span. However if k=0 is indeed the worst ratio
    (as it should in a unimodal case) then, it helps define the second to
    the worst ratio.
    - If all ratios are equal, life_span=0, r_birth=0 and r_death=0 for all k
    except for k=1, where life_span=1, r_birth=0 and r_death=1

    e.g. if k=2,3 have the same r_score, then
    - life span[3] = 0
    - life span[2] = r_curr - r_prev where r_prev is the last r_score found
    without being equal to r_curr

    :param g: Graph
    :type g: PersistentGraph
    :return: life span and ratios of each assumption k for all k and each t
    :rtype: Dict[int, Dict[str, List[float]]]
    """

    # By default all life span are 0 (even if their corresponding step was
    # ignored). They might remain 0 in case of equal ratios
    k_infos = { k :
        {
            "life_span" : [0. for _ in range(g._T_w)],
            "score_ratios" : [[0., 0.] for _ in range(g._T_w)],
        } for k in range(1, g._k_max+1)
    }

    # ------------------------------------------------------------------
    # ------------------ Monotonous case -------------------------------
    # ------------------------------------------------------------------
    if g._score.score_type in ["monotonous" or "pseudo-monotonous"]:
        # Extract ratio for each k and each t
        for t in range(g._T_w):

            # to keep track of the last step and make sure smaller k are favored
            k_prev = 0
            k_curr = 0
            r_prev = 0
            r_curr = 0

            for step in g._local_steps[t]:
                if step['k'] != 0:
                    # ----- Initialisation -----------------
                    # By default all k have r_birth = r_death and life_span = 0
                    k_infos[step['k']]['score_ratios'][t] = [step['ratio_score'], step['ratio_score']]

                    # If ratios are equal, don't update, keep everything as
                    # initialized, but choose the "good" k_curr for the
                    # next iteration of k_prev
                    if step['ratio_score'] == r_curr:
                        if k_prev != 0:
                            k_curr = np.amin([step['k'], k_prev])
                    # Else if the ratio is not improving while it should
                    # or is improving when it shouldn't
                    elif not (
                        (step['ratio_score'] > r_curr) == (g._score.improve)
                    ):
                        # Ignore the step and go directly to the next
                        # iteration
                        pass
                    # Otherwise normal case:
                    # Update info of k_curr, with r_prev != r_curr
                    # and r_curr relevant
                    else:
                        # Previous step is the former current step
                        r_prev = r_curr
                        k_prev = k_curr

                        # current step is the new step
                        k_curr = step['k']
                        r_curr = step['ratio_score']

                        # Update
                        k_infos[k_curr]['life_span'][t] = r_curr - r_prev
                        k_infos[k_curr]['score_ratios'][t] = [r_prev, r_curr]

            # ------- Last step ---------
            # If we were in a series of equal ratios, find the "good" k_curr
            if r_curr == r_prev:
                k_curr = max(min(k_curr, k_prev), 1)
                k_infos[k_curr]['life_span'][t] = 1 - r_prev
                k_infos[k_curr]['score_ratios'][t] = [r_prev, 1]

    # ------------------------------------------------------------------
    # -------------------- Absolute case -------------------------------
    # ------------------------------------------------------------------
    else:
        for t in range(g._T_w):
            for step in g._local_steps[t]:
                if step['k'] != 0:
                    # Update all cases in the same way
                    k_curr = step['k']
                    r_curr = step['ratio_score']

                    k_infos[k_curr]['life_span'][t] = r_curr
                    k_infos[k_curr]['score_ratios'][t] = [0, r_curr]
    return k_infos

def get_relevant_k(
    g,
) -> List[List]:
    """
    For each time step, get the most relevant number of clusters

    :param g: Graph
    :type g: PersistentGraph

    :return: Nested list of [k_relevant, life_span_k] for each time step
    :rtype: List[List]
    """

    # list of t (k, k_info)
    relevant_k = [[1, 0.] for _ in range(g._T_w)]
    for t in range(g._T_w):
        for k, k_info_k in g.k_info.items():
            # Strict comparison to prioritize smaller k values
            if k_info_k["life_span"][t] > relevant_k[t][1]:
                relevant_k[t][0] = k
                relevant_k[t][1] = k_info_k["life_span"][t]
    return relevant_k