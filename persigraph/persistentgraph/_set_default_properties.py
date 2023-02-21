
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

def _set_model_class(pg, model_class, DTW: bool = False):
    """
    Set _model_class, and `_DTW` allowing strings instead of sklearn class

    Note that custom classes are still possible
    """
    names = CLUSTERING_METHODS["names"]
    algos_ed = {
        n : a for (n,a) in zip(names, CLUSTERING_METHODS["classes-standard"])
    }
    algos_dtw = {
        n : a for (n,a) in zip(names, CLUSTERING_METHODS["classes-dtw"])
    }

    default_names = [None, ""]
    default_algo = KMeans
    default_DTW = False

    # Base case
    if model_class in default_names:
        pg._model_class = default_algo
        pg._DTW = default_DTW
    # Usual case
    elif model_class in names:
        if DTW:
            if algos_dtw[model_class] is None:
                msg = (
                    "DTW is not available with " + model_class
                    + ". Please select a valid clustering method or "
                    + "use euclidean distance"
                )
                raise ValueError(msg)
            else:
                pg._model_class = algos_dtw[model_class]
                pg._DTW = True
        else:
            if algos_ed[model_class] is None:
                msg = (
                    "Euclidean distance is not available with " + model_class
                    + ". Please select a valid clustering method or "
                    + "use DTW"
                )
                raise ValueError(msg)
            else:
                pg._model_class = algos_ed[model_class]
                pg._DTW = False
    # Invalid option
    elif type(model_class) == str:
        msg = (
            "Please select a valid clustering method name or give a "
            + "valid clustering method class"
        )
        raise ValueError(msg)
    # Assume that a valid (python class, DTW) tuple was given
    else:
        pg._model_class = model_class
        pg._DTW = DTW

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