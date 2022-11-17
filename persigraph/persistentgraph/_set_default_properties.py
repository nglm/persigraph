
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from ._scores import SCORES_TO_MAXIMIZE, SCORES_TO_MINIMIZE, SCORES
from ._clustering_model import CLUSTERING_METHODS

def _set_model_class(pg, model_class):
    """
    Set mode_class, allowing strings instead of sklearn class

    Note that custom classes are still possible
    """
    names = CLUSTERING_METHODS
    algos = [
        KMeans, SpectralClustering, GaussianMixture,
        AgglomerativeClustering
    ]
    default_names = [None, ""]
    default_algo = KMeans

    d = {n : a for n,a in zip(names,algos)}

    if model_class in default_names:
        pg._model_class = default_algo
    elif model_class in names:
        pg._model_class = d[model_class]
    elif type(model_class) == str:
        msg = (
            "Please select a valid clustering method name or give a "
            + "valid clustering method class"
        )
        raise ValueError(msg)
    else:
        pg._model_class = model_class

def _set_score_type(pg, score_type):

    default_names = [None, ""]
    default_score = "inertia"

    if score_type in SCORES_TO_MAXIMIZE:
        pg._maximize = True
    elif score_type in SCORES_TO_MINIMIZE:
        pg._maximize = False
    elif score_type in default_names:
        pg._maximize = True
        score_type = default_score
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