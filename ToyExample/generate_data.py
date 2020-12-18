import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from utils.lists import flatten
from math import cos, sin, pi
from typing import List, Dict


# TODO: allow for multivariate gaussian in gauss_mixture

def gauss_mixture(
    list_params = [(2,1) , (-2,1)],  #List[(mu, sigma)]
    list_prop = [0.5, 0.5], # Proportion (weight) of each gaussian
    nsamples: int = 10000,
    shuffled_sample:bool = False,  # are sample ordered by sub-gaussian or shuffled?
):
    sample = np.zeros(nsamples)
    start = 0
    for i in range(len(list_params)-1):
        end = int(nsamples*list_prop[i])
        (mu, std) = list_params[i]
        weight = list_prop[i]
        sample[start:end] = np.random.normal(mu, std, int(nsamples*weight))
        start = end
    (mu, std) = list_params[-1]
    weight = list_prop[-1]
    sample[start:] = np.random.normal(mu, std, int(nsamples*weight))
    if shuffled_sample:
        sample = np.random.shuffle(sample)
    return sample


def identity(
    xvalues,
):
    return xvalues

def linear(
    xvalues,
    slope:float = 1.,    # Slope
    const:float = 0.,    # Constant term
    ymax: float = None,  # If defined by y max/min values
    ymin: float = None,
):
    # If slope and const defined by y max/min values:
    if ymax is not None:
        if ymin is None:
            ymin = 0
        slope = (ymax-ymin)/(xvalues.max() - xvalues.min())
        const = ymin - slope*xvalues.min()
    # Classic a*x + b
    yvalues = np.array([slope*x+const for x in xvalues])
    return yvalues

def cosine(
    xvalues,
    amp:float = 1.,    # Amplitude
    freq:float = 1.,   # Frequency
    const:float = 0.,  # Constant term
):
    yvalues = np.array([amp*cos(2*pi*freq*x)+const for x in xvalues])
    return yvalues

def triangle(
    xvalues,
    peak_value:float = 3.,         # Peak value (could be negative)
    nsteps_to_peak:int = 3,     # steps to reach a peak
    nsteps_break_up:int = 4,      # nsteps as a constant after peak
    nsteps_break_down:int = 3,      # nsteps  as a constant between triangles
    const:float = 0.,              # constant term
):
    # Value to add/subtract for each step to peak
    val_incr = peak_value/nsteps_to_peak
    T = xvalues.shape[0]

    # Initialization: start by going to peak
    to_peak = True  # Are we going up to peak or down?
    in_break = False # Is it a break?
    yvalues = [0]
    steps_since_new_direction = 1
    i=1
    while i < T:
        # ------------------------
        # Update direction
        # ------------------------
        # 1. BASE REACHED:
        # - to_peak=True
        # - in_break = True (if nsteps_break_down >0)
        if (
            not in_break and not to_peak
            and steps_since_new_direction == nsteps_to_peak
        ):
            to_peak = True
            in_break = nsteps_break_down > 0
            steps_since_new_direction = 0

        # 2. break_down FINISHED:
        # - in_break = False
        if (
            in_break and to_peak
            and steps_since_new_direction == nsteps_break_down
        ):
            in_break = False
            steps_since_new_direction = 0

        # 3. PEAK REACHED:
        # - to_peak = False
        # - in_break = True (if nsteps_break_up >0)
        if (
            not in_break and to_peak
            and steps_since_new_direction == nsteps_to_peak
        ):
            to_peak = False
            in_break = nsteps_break_up > 0
            steps_since_new_direction = 0

        # 4. break_up FINISHED:
        # - in_break = False
        if (
            in_break and not to_peak
            and steps_since_new_direction == nsteps_break_up
        ):
            in_break = False
            steps_since_new_direction = 0

        # ------------------------
        # Compute current value
        # ------------------------
        # If to_peak and in_break: const
        # If to_peak and not in_break: += val_incr
        # If not to_peak and in_break: peak_value
        # If not to_peak and not in_break: -= val_incr
        if to_peak:
            if in_break:
                val = 0
            else:
                val = yvalues[i-1] + val_incr
        else:
            if in_break:
                val = peak_value
            else:
                val = yvalues[i-1] - val_incr
        yvalues.append(val)


        steps_since_new_direction += 1
        i += 1
    yvalues = np.array(yvalues)+const
    return yvalues

def generate_centers(
    xvalues=None,
    T:int = 51,
    nb_clusters: int =2,
    func:list = None,          # callable or list
    func_kw:list = None,  # list(nb_clusters) of dict of func kwargs
):
    """
    Generate 'nb_clusters' time series representing cluster patterns

    :param xvalues: array of x values, defaults to None
    :type xvalues: np.ndarray(T), optional
    :param T: length of the time series, defaults to 51
    :type T: int, optional
    :param nb_clusters: number of cluster to generate, defaults to 2
    :type nb_clusters: int, optional
    :param func: functions s.t y=func(x), defaults to None
    :type func: list(nb_clusters) or Callable, optional
    :param func_kw: kwargs associated with func, defaults to None
    :type func_kw: list, optional
    :return: 'nb_clusters' time series representing cluster patterns
    :rtype: np.ndarray((nb_clusters, T))
    """
    # Either xvalues is given or a number of time steps T
    if xvalues is None:
        xvalues = np.arange(T)
    else:
        T = len(xvalues)
    # Get as many functions as centers to generate
    if func is None:
        func_kw = []
        func = []
        for cluster in range(nb_clusters):
            amplitude = 3.*(2+cluster)//2
            sign = (-1)**cluster
            func_kw.append({'peak_value': sign * amplitude})
            func.append(triangle)
    if not isinstance(func,list):
        func = [func]*nb_clusters
    # NOTE: this makes no sense to give the same param for all...
    cluster_centers = np.zeros((nb_clusters, T))
    for cluster in range(nb_clusters):
        cluster_centers[cluster] = func[cluster](
            xvalues,
            **func_kw[cluster],
        )
    return cluster_centers

def generate_params(
    names:list,
    xvalues=None,
    T:int = 51,
    func = None,  # Callable or list
    func_kw:list = None,  # Dict or list
):
    """
    Generate a list(T) of dictionaries corresponding to one cluster distrib
    """
    n_params = len(names)
    # Either xvalues is given or a number of time steps T
    if xvalues is None:
        xvalues = np.arange(T)
    else:
        T = len(xvalues)
    # Get as many functions as parameters to generate
    if func is None:
        func = linear
    if not isinstance(names,list):
        names = [names]
    if not isinstance(func,list):
        func = [func]*n_params
    if not isinstance(func_kw, list):
        func_kw = [func_kw]*n_params
    params_values = []
    for i, name in enumerate(names):
        # list(n_params) whose elements are array(T)
        params_values.append(func[i](xvalues, **func_kw[i]))

    params = []
    for t in range(T):
        dict_t = {}
        for i, name in enumerate(names):
            dict_t[name] = params_values[i][t]
        params.append(dict_t)
    return params  # list(T) of dict with n_params keys

def get_param_values(params: List[dict]):
    """
    Convert the List[dict] output of ``generate_params`` to Dict[str, List]

    This function is mostly used for ploting the true distribution.
    ``generate_members`` does not use this function

    :param params: [description]
    :type params: List[dict]
    """
    dict_of_values = {}
    for key in params[0].keys():
        # Initialize a list a value for each key
        dict_of_values[key] = [params[0][key]]
    for t in range(1,len(params)):
        for key in params[t].keys():
            dict_of_values[key].append(params[t][key])
    for key in params[0].keys():
        # Initialize a list a value for each key
        dict_of_values[key] = np.array(dict_of_values[key])
    return dict_of_values



def generate_members(
    N:int = 50,
    T:int = 51,
    xvalues = None,
    nb_clusters:int = 2,        # number of clusters
    cluster_centers = None,     # array(nb_clusters, T) center of each cluster
    cluster_ratios:list = None, #list(nb_clusters)
    distrib_type: list = None,  #list(nb_clusters) (callable)
    distrib_params:list = None, #list(nb_clusters,T) of dict
    shuffled_members:bool = True,
    chaotic:bool = False,
):
    # ---------------------
    # Initialization
    # ---------------------
    # Either xvalues is given or a number of time steps T
    if xvalues is None:
        xvalues = np.arange(T)
    else:
        T = len(xvalues)

    # If None: Pair of opposite triangles
    if cluster_centers is None:
        cluster_centers = generate_centers(
            xvalues=xvalues,
            nb_clusters=nb_clusters
        )

    # If None: Uniform distribution of member among clusters
    if cluster_ratios is None:
        cluster_ratios = [1/nb_clusters for _ in range(nb_clusters)]

    # If None: gaussian with increasing sigma
    if distrib_type is None:
        if distrib_params is None:
            distrib_params = generate_params(
                names='scale',
                xvalues=xvalues,
                func=linear,
                func_kw={'ymin':0.5,'ymax':2},
            )
        distrib_type = np.random.normal
    # If distrib_type is not a nested list then cluster centers share the
    # same distrib type
    if not isinstance(distrib_type, list):
        distrib_type = [distrib_type]*nb_clusters
    # If distrib_params is not a nested list then cluster centers share the
    # same set of parameters
    if not isinstance(distrib_params[0], list):
        distrib_params = [distrib_params]*nb_clusters

    # ---------------------
    # Data generation
    # ---------------------
    members = []
    cluster = 0  # Current cluster
    cum_ratio = cluster_ratios[0] # Cumulative ratio of members in clusters
    for m in range(N):
        # Update current cluster
        if m/N > cum_ratio and cluster<nb_clusters-1:
            cluster += 1
            cum_ratio += cluster_ratios[cluster]

        # Initialization of a member
        member = [] # current member
        distrib = distrib_type[cluster] # current distrib (callable)
        center = cluster_centers[cluster] # current center
        # Generate one member (complete time serie)
        for t in range(T):
            if chaotic and t>0:
                ref = member[t-1]
            else:
                ref = center[t]
            val = ref + distrib(**distrib_params[cluster][t])
            member.append(val)
        members.append(member)
    members = np.array(members)
    if shuffled_members:
        np.random.shuffle(members)
    return members










