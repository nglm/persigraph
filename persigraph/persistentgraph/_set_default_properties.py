
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np

from ._scores import SCORES_TO_MAXIMIZE, SCORES_TO_MINIMIZE, SCORES
from ._clustering_model import CLUSTERING_METHODS

def _set_members(pg, members):
    """
    Set members, N, d, T
    """
    pg._members = np.copy(members)  #Original Data

    # Variable dimension
    shape = pg._members.shape

    # Assume that both d and T are "missing"
    if len(shape) == 1:
        pg._d = int(1)
        pg._T = int(1)
        pg._members = np.expand_dims(pg._members, axis=(1,2))
    # Assume that only d is missing
    elif len(shape) == 2:
        pg._d = int(1)
        pg._members = np.expand_dims(pg._members, axis=1)
    elif len(shape) == 3:
        pg._N = shape[0]  # Number of members (time series)
        pg._d = shape[1]  # Number of variables
        pg._T = shape[2]  # Length of the time series
    else:
        raise ValueError(
            "Invalid shape of members provided:" + str(shape)
            + ". Please provide a valid shape: (N,) or (N, T) or (N, d, T)"
        )

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