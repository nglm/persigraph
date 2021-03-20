import numpy as np
from typing import List, Sequence, Union, Any, Dict

from ._scores import compute_score, worst_score
from ..utils.kmeans import kmeans_custom, row_norms



def get_model_parameters(
        pg,
        X,
):
    # The same N datapoints X are use for all n_clusters values
    # Furthermore the clustering method might want to copy X
    # Each time it is called and compute pairwise distances
    # We avoid doing that more than once
    # using copy_X and row_norms_X (in fit_predict_kw)
    copy_X = np.copy(X)
    row_norms_X = row_norms(copy_X, squared=True)
    fit_predict_kw = {
        "x_squared_norms" : row_norms_X,
        'X' : copy_X,
        }
    # Default kw values
    model_kw = {
        'max_iter' : 100,
        'n_init' : 10,
        'tol' : 1e-3,
    }
    return model_kw, fit_predict_kw



# def graph_initialization(pg):
#     """
#     Initialize the graph with k_max components at each time step
#     """
#     compute_zero_scores(pg)
#     cluster_data = [[] for _ in range(pg.T)]
#     for t in range(pg.T):

#         # No need to compute kmeans if we just want as many clusters as members
#         if pg.k_max == pg.N:
#             clusters_info = []
#             clusters = []
#             step_info = {'score' : 0}
#             # ======= Create one vertex per member and time step =======
#             for i in range(pg.N):
#                 clusters_info.append({
#                     'type' : 'KMeans',
#                     'params' : [pg._members[i,t], 0.], #mean, std
#                     'brotherhood_size' : [pg.k_max]
#                 })
#                 clusters.append([i])

#         # Otherwise we really need to call kmeans
#         else:
#             X = pg._members[:,t].reshape(-1,1)
#             model_kw, fit_predict_kw = get_model_parameters(pg, X)
#             model_kw['n_clusters'] = pg.k_max
#             clusters, clusters_info, step_info, model_kw = clustering_model(
#                 pg, X, model_kw, fit_predict_kw,
#             )

#         cluster_data[t] = [clusters, clusters_info]

#         # ========== Finalize initialization step ==================

#         pg._local_steps[t].append(
#             {**{'param' : {"n_clusters" : pg.k_max}},
#              **step_info
#             })

#         pg._nb_local_steps[t] += 1
#         pg._nb_steps += 1

#         if pg._verbose:
#             print(" ========= ", t, " ========= ")
#             print(
#                 "n_clusters: ", pg.k_max,
#                 "   score: ", step_info['score'],
#             )

#     return cluster_data

# def compute_extremum_scores(pg):

#     # Will compute the score got assuming 0 cluster
#     compute_zero_scores(pg)
#     # Compute the score of one component and choose the worst score

#     for t in range(pg.T):

#         X = pg._members[:,t].reshape(-1,1)
#         model_kw, fit_predict_kw = get_model_parameters(pg, X)
#         model_kw['n_clusters'] = 1

#         _, _, step_info, _ = clustering_model(
#             pg,
#             X,
#             model_kw = model_kw,
#             fit_predict_kw = fit_predict_kw,
#         )
#         pg._worst_scores[t] = worst_score(pg,
#             step_info['score'],
#             pg._zero_scores[t]
#         )

#     pg._best_scores = np.zeros(pg.T)
#     pg._norm_bounds = np.abs(pg._best_scores - pg._worst_scores)
#     pg._are_bounds_known = True


def clustering_model(
    pg,
    X,
    model_kw : Dict = {},
    fit_predict_kw : Dict = {},
):

    # ====================== Fit & predict part =======================
    n_clusters = model_kw.pop('n_clusters')
    model = kmeans_custom(
        n_clusters = n_clusters,
        copy_x = False,
        **model_kw,
    )
    labels = model.fit_predict(**fit_predict_kw)
    if model.n_iter_ == model_kw['max_iter']:
        raise ValueError('Kmeans did not converge')


    # ==================== clusters, cluster_info ======================
    clusters_info = []
    clusters = []
    for label_i in range(n_clusters):
        # Members belonging to that clusters
        members = [m for m in range(pg.N) if labels[m] == label_i]
        clusters.append(members)
        if members == []:
            raise ValueError('No members in cluster')
        
        # Info related to this specific vertex
        clusters_info.append({
            'type' : 'KMeans',
            'params' : [
                float(model.cluster_centers_[label_i]),
                float(np.std(X[members])),
                ],
            'brotherhood_size' : [n_clusters]
        })

    # ========================== step_info =============================
    score = compute_score(
            pg,
            model = model,
            X = X,
            clusters = clusters,
        )
    step_info = {'score' : score}

    #TODO: add cluster center to model_kw for future clustering

    return clusters, clusters_info, step_info, model_kw