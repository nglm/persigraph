
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

def _set_zero(pg, zero_type: str = "bounds"):
    """
    Set members_zero, zero_type

    Generate member values to emulate the case k=0

    :param pg: PersistentGraph
    :type pg: PersistentGraph
    """
    # Determines how to measure the score of the 0th component
    if zero_type == 'bounds':
        pg._zero_type = zero_type
    else:
        raise NotImplementedError("Only 'bounds' is implemented as `zero_type`")
        # I'm not sure if this type should be used at all actually.....
        # Get the parameters of the uniform distrib using mean and variance
        var = np.var(X, axis=0)
        mean = np.mean(X, axis=0)

        mins = (2*mean - np.sqrt(12*var)) / 2
        maxs = (2*mean + np.sqrt(12*var)) / 2

    # Get the parameters of the uniform distrib using min and max
    # We keep all the dims except the first one (Hence the 0) because
    # The number of members dimension will be added in members_0 in the
    # List comprehension
    mins = np.amin(pg._members, axis=0, keepdims=True)[0]
    maxs = np.amax(pg._members, axis=0, keepdims=True)[0]

    # Generate a perfect uniform distribution
    steps = (maxs-mins) / (pg.N-1)
    members_0 = np.array([mins + i*steps for i in range(pg.N)])
    pg._members_zero = members_0

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