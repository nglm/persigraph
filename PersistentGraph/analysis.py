import numpy as np
import matplotlib.pyplot as plt
from utils.lists import flatten
from gudhi import bottleneck_distance
from PersistentGraph.vertex import Vertex
from PersistentGraph.edge import Edge
from PersistentGraph.component import Component
from typing import List, Dict, Tuple

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

def compute_barcodes(
    components: List[List[Component]],
) -> List[List[Tuple[float]]]:
    """
    Compute a list of (sort of) barcodes for each time step

    FIXME: Outdated
    What we call a barcode here is simply a list of tuple
    ``(r_birth, r_born, ratio_members)``.

    - ``r_birth`` defines where the bar starts
    - ``r_death`` defines where the bar dies
    - ``ratio_members`` is additional information

    They are not really barcodes as defined in the persistent homology
    method because we do not build simplices.

    :param components: List of graph components (vertices or edges)
    :type components: List[List[Component]]
    :return: A list of (sort of) barcodes for each time step
    :rtype: List[List[Tuple[float]]]
    """
    if not isinstance(components[0], list):
        components = [components]
    barcodes = []
    for t in range(len(components)):
        bc_t = []
        for c in components[t]:
            bc_t.append((c.r_birth, c.r_death, c.ratio_members))
        barcodes.append(bc_t)
    return barcodes

def compute_bottleneck_distances(barcodes):
    #FIXME: Outdated
    diags = [np.array([bc_3[:-1] for bc_3 in bc_i]) for bc_i in barcodes]
    bn_dist = []
    for i in range(len(diags)-1):
        bn_dist.append(bottleneck_distance(diags[i], diags[i+1], e=0))
    # If only 2 barcodes were compared return a float
    # Otherwise return a list of bn distances
    if len(bn_dist)==1:
        bn_dist = bn_dist[0]
    return bn_dist

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
    k_max=8,
):
    k_max = min(k_max, g.k_max)
    r_scores = []
    life_span = {k : [] for k in range(1,g.k_max+1)}

    # Extract ratio scores for each k and each t
    for t in range(g.T):

        # init
        k_prev = g._n_clusters_range[0]
        r_scores.append([g._local_steps[t][0]['ratio_score']])

        for step in g._local_steps[t]:
            k_curr = step['param']['n_clusters']
            # Note: there might be some 'holes' when steps are ignored
            # their r_score will then all be 'step['ratio_score']'
            r_scores[-1] += abs(k_curr - k_prev)*[step['ratio_score']]
            k_prev = k_curr

        # Its length should therefore be N+1
        r_scores[-1] += (abs(g._n_clusters_range[-1] - k_prev)+1)*[1]

        # Compute life span for each k and each t
        for i in range(g.k_max):
            life_span[g._n_clusters_range[i]].append(
                r_scores[-1][i+1] - r_scores[-1][i]
                )
    return life_span

def get_relevant_k(
    g,
    life_span = None,
    k_max = 8,
):
    k_max = min(k_max, g.k_max)
    if life_span is None:
        life_span = get_k_life_span(g, k_max)

    # Not at all an efficient way to do it...
    #relevant_k = np.zeros((g.T, 2), dtype=float)
    relevant_k = [[0 for _ in range(2)] for _ in range(g.T)]
    for t in range(g.T):
        print('--------', t)
        for k, life_span_k in life_span.items():
            # Strict comparison to prioritize smaller k values
            if life_span_k[t] > relevant_k[t][1]:
                print(k, life_span_k[t], relevant_k[t][1])
                relevant_k[t][0] = k
                relevant_k[t][1] = life_span_k[t]
    print(relevant_k)
    return relevant_k


def get_contemporaries(g, cmpt):
    # FIXME: Outdated
    t = cmpt.time_step
    contemporaries = []
    for s in range(cmpt.s_birth, cmpt.s_death):
        c_alive_s = []
        if isinstance(cmpt, Vertex):
            c_alive_s = g.get_alive_vertices(s=s, t=t)
        if isinstance(cmpt, Edge):
            c_alive_s = g.get_alive_edges(s=s, t=t)
        contemporaries += c_alive_s
    # get edges from e_num
    contemporaries = [g.edges[t][e_num] for e_num in contemporaries]
    # remove duplicate
    contemporaries = list(set(contemporaries))
    return contemporaries
