
from sklearn.cluster import KMeans
import numpy as np
from pycvi.cluster import generate_uniform, sliding_window

from ._scores import SCORES_TO_MAXIMIZE, SCORES_TO_MINIMIZE
from ._clustering_model import CLUSTERING_METHODS

def _set_members(pg, members):
    """
    Set members, N, d, T
    """
    pg._members = np.copy(members)  #Original Data

    # Variable dimension
    shape = pg._members.shape


    pg._N = shape[0]  # Number of members (time series)
    if pg._N < 2:
        raise ValueError(
            "At least members should be given (N>=2)" + str(shape[0])
        )
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

        pg._d = shape[1]  # Number of variables
        pg._T = shape[2]  # Length of the time series
    else:
        raise ValueError(
            "Invalid shape of members provided:" + str(shape)
            + ". Please provide a valid shape: (N,) or (N, T) or (N, d, T)"
        )

def _set_sliding_window(pg, w:int):
    """
    Set pg._sliding_window and pg._w
    """
    if w is None:
        pg._sliding_window = None
        pg._w = None
    else:
        pg._w = min( max(int(w), 1), pg.T)
        pg._sliding_window = sliding_window(pg.T, pg._w)

def _set_zero(pg, zero_type: str = "bounds"):
    """
    Set members_zero, zero_type

    Generate member values to emulate the case k=0

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    """
    pg._members_zero = generate_uniform(pg._members, zero_type)
    pg._zero_type = zero_type

def _set_model_class(
    pg,
    model_class,
    DTW: bool = False,
    model_kw: dict = {},
    fit_predict_kw: dict = {},
    model_class_kw: dict = {},
):
    """
    Set all properties related to the clustering model

    `_model_class`, `_DTW`, `_model_kw`, `_fit_predict_kw`,
    `_model_class_kw` and `_model_type`

    Allows for strings and classes.

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

    # User_friendly name
    pg._model_type = str(pg._model_class())[:-2]
    # To know how X and n_clusters args are called in this model class
    pg._model_class_kw = model_class_kw
    # Key-words related to the clustering model instantiation
    pg._model_kw = model_kw
    # Key-words related to the clustering model fit_predict method
    pg._fit_predict_kw = fit_predict_kw

def _set_score_type(pg, score_type):

    default_names = [None, ""]
    default_score = "inertia"
    default_maximize = False

    if score_type in SCORES_TO_MAXIMIZE:
        pg._score_maximize = True
    elif score_type in SCORES_TO_MINIMIZE:
        pg._score_maximize = False
    elif score_type in default_names:
        pg._score_maximize = default_maximize
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
    pg._score = score_type