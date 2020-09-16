import numpy as np
import matplotlib.pyplot as plt
from galib.tools.lists import flatten
from gudhi import bottleneck_distance

def basic_stats(components):
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
    components,
    distances,
    as_matrix=False,
    normalized=True
):
    if not isinstance(components[0], list):
        components = [components]
    barcodes = []
    if normalized:
        norm = distances[0]
    else:
        norm = 1.
    for t in range(len(components)):
        bc_t = []
        for c in components[t]:
            bc_t.append((
                distances[c.s_death]/norm,
                distances[c.s_born]/norm,
                c.ratio_members
            ))
        barcodes.append(bc_t)
    return barcodes

def compute_bottleneck_distances(barcodes):
    diags = [np.array([bc_3[:-1] for bc_3 in bc_i]) for bc_i in barcodes]
    bn_dist = []
    for i in range(len(diags)-1):
        bn_dist.append(bottleneck_distance(diags[i], diags[i+1], e=0))
    # If only 2 barcodes were compared return a float
    # Otherwise return a list of bn distances
    if len(bn_dist)==1:
        bn_dist = bn_dist[0]
    return bn_dist