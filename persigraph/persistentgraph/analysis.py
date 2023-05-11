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

# #def# compute_barcodes(
#     components: List[List[Component]],
# ) -> List[List[Tuple[float]]]:
#     """
#     Compute a list of (sort of) barcodes for each time step

#     FIXME: Outdated
#     What we call a barcode here is simply a list of tuple
#     ``(r_birth, r_born, ratio_members)``.

#     - ``r_birth`` defines where the bar starts
#     - ``r_death`` defines where the bar dies
#     - ``ratio_members`` is additional information

#     They are not really barcodes as defined in the persistent homology
#     method because we do not build simplices.

#     :param components: List of graph components (vertices or edges)
#     :type components: List[List[Component]]
#     :return: A list of (sort of) barcodes for each time step
#     :rtype: List[List[Tuple[float]]]
#     """
#     if not isinstance(components[0], list):
#         components = [components]
#     barcodes = []
#     for t in range(len(components)):
#         bc_t = []
#         for c in components[t]:
#             bc_t.append((c.r_birth, c.r_death, c.ratio_members))
#         barcodes.append(bc_t)
#     return barcodes

# def# compute_bottleneck_distances(barcodes):
#     #FIXME: Outdated
#     diags = [np.array([bc_3[:-1] for bc_3 in bc_i]) for bc_i in barcodes]
#     bn_dist = []
#     for i in range(len(diags)-1):
#         bn_dist.append(bottleneck_distance(diags[i], diags[i+1], e=0))
#     # If only 2 barcodes were compared return a float
#     # Otherwise return a list of bn distances
#     if len(bn_dist)==1:
#         bn_dist = bn_dist[0]
#     return bn_dist

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

def get_k_life_span(
    g,
    k_max: int = 5,
) -> Dict[int, List[float]]:
    """
    Get the life span for all k [1, ..., k_max] and each t.
    To summarize:
    life_span[k_prev][t] = r_curr - r_prev

    Note:
    - there might be some 'holes' when steps are ignored
    their life span will then all be 0.
    - In case of equal r_scores, the smallest k value
    will be favored, the other ones keep their life span of value 0.
    - The case k=0 is used only to be used as a potential score bound.
    It is never used to create a vertex in the graph and it doesn't have
    a life span.
    - If all scores are equal, life_span=0, r_birth=0 and r_death=0 for all k
    except for k=1, where life_span=1, r_birth=0 and r_death=1

    e.g. if k=2,3 have the same r_score, thend
    - life span[3] = 0
    - life span[2] = r_curr - r_prev where r_prev is the last r_score found
    without being equal to r_curr

    :param g: [description]
    :type g: [type]
    :param k_max: Max value of k considered, defaults to 5
    :type k_max: int, optional
    :return: life span of k clusters for each k and each t
    :rtype: Dict[int, List[float]]
    """
    k_max = min(k_max, g.k_max)

    # By default all life span are 0 (even if their corresponding step was
    # ignored). They might remain 0 in case of equal r_scores
    life_span = { k : [0. for _ in range(g.T)] for k in range(1, k_max+1) }

    # Extract ratio scores for each k and each t
    for t in range(g.T):

        # to keep track of the last step and make sure smaller k are favored
        k_prev = None
        r_prev = 0
        k_eq = []

        for i, step in enumerate(g._local_steps[t]):
            # k_curr are not necessarily in increasing order
            k_curr = step['param']['n_clusters']
            r_curr = step['ratio_score']

            # Compute life span of the previous k visited
            if i>0:
                # If r_scores are equal, don't update life span,
                # just stack k values that share the same r_score
                if r_curr == r_prev:
                    if k_eq == []:
                        k_eq = [k_prev, k_curr]
                    else:
                        k_eq.append(k_curr)
                # Else compute life span of k_prev, with r_prev != r_curr
                else:
                    life_span[k_prev][t] = r_curr - r_prev
                    # If same ks were sharing the same r_score, keep their
                    # life span to 0 as initialized...
                    if k_eq:
                        sorted_k_eq = sorted(k_eq)
                        # ...except the smallest k, that will be updated
                        # at the next iteration (as being next k_prev)
                        k_curr = sorted_k_eq[0]
                    k_eq = []

            # Prepare next iteration
            r_prev = r_curr
            k_prev = k_curr

        # ------- Last step ---------
        # If we were in a series of equal scores, find the "good" k_prev
        if k_eq != []:
            k_prev = sorted(k_eq)[0]
        life_span[k_prev][t] = 1 - r_prev

    return life_span

def get_relevant_k(
    g,
    life_span: Dict[int, List[float]] = None,
    k_max: int = 8,
) -> List[List]:
    """
    For each time step, get the most relevant number of clusters

    :param g: Graph
    :type g: [type]
    :param life_span: life span of all k for all t, defaults to None
    :type life_span: Dict[int, List[float]], optional
    :param k_max: Max value of k considered, defaults to 8
    :type k_max: int, optional
    :return: Nested list of [k_relevant, life_span] for each time step
    :rtype: List[List]
    """
    k_max = min(k_max, g.k_max)
    if life_span is None:
        life_span = get_k_life_span(g, k_max)

    # list of t (k, life_span_k)
    relevant_k = [[0, 0.] for _ in range(g.T)]
    for t in range(g.T):
        for k, life_span_k in life_span.items():
            # Strict comparison to prioritize smaller k values
            if life_span_k[t] > relevant_k[t][1]:
                relevant_k[t][0] = k
                relevant_k[t][1] = life_span_k[t]
    return relevant_k



# #def# get_contemporaries(g, cmpt):
#     # FIXME: Outdated
#     t = cmpt.time_step
#     contemporaries = []
#     for s in range(cmpt.s_birth, cmpt.s_death):
#         c_alive_s = []
#         if isinstance(cmpt, Vertex):
#             c_alive_s = g.get_alive_vertices(s=s, t=t)
#         if isinstance(cmpt, Edge):
#             c_alive_s = g.get_alive_edges(s=s, t=t)
#         contemporaries += c_alive_s
#     # get edges from e_num
#     contemporaries = [g.edges[t][e_num] for e_num in contemporaries]
#     # remove duplicate
#     contemporaries = list(set(contemporaries))
#     return contemporaries
