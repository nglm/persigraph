import numpy as np
import pandas as pd
import re
from math import atan2
from netCDF4 import Dataset
from typing import List, Sequence, Union, Any, Dict, Tuple

from .statistics import standardize
from ..utils.lists import get_indices_element



def extract_from_meteogram(
    nc,
    var_names : Union[List[str], str] = None,
    ind_time: Union[np.ndarray, int] = None,
    ind_members: Union[np.ndarray, int] = None,
    ind_long: Union[np.ndarray, int] = None,
    ind_lat: Union[np.ndarray, int] = None,
    multivariate: bool = False,
) -> Dict:
    """
    Extract given variables and corresponding columns

    Warning: if multivariate, the time axis dimension comes before the
    variable dimension

    The objective is to extract quickly useful variables from
    all our nc files

    :param nc: Dataset (nc file) from which values are extracted
    :type nc: Dataset
    :param var_names:

        Names of the variables to extract,
        defaults to None, in this case
        ``` var_names =["t2m","d2m","msl","u10","v10","tcwv"]```

    :type var_names: Union[List[str], str], optional
    :param ind_time:

        Indices of time steps to extract,
        defaults to None, in this case all time steps are extracted

    :type ind_time: Union[np.ndarray, int], optional
    :param ind_members:

        Indices of members to extract,
        defaults to None, in this case all members are extracted

    :type ind_members: Union[np.ndarray, int], optional
    :param ind_long:

        Indices of longitudes to extract,
        defaults to None, in this case all elements are extracted

    :type ind_long: Union[np.ndarray, int], optional
    :param ind_lat:

        Indices of latitudes to extract,
        defaults to None, in this case all elements are extracted

    :type ind_lat: Union[np.ndarray, int], optional
    :param multivariate: Consider one multivariate variable, defaults to False
    :type multivariate: bool, optional
    :return: A dict containing variables, their corresponding names, time, etc.
    :rtype: Dict
    """
    if var_names is None:
        var_names =["t2m","d2m","msl","u10","v10","tcwv"]
    if isinstance(var_names, str):
        var_names = [var_names]
    if ind_time is None:
        ind_time = np.arange(nc.variables["time"].size)
    if ind_members is None:
        ind_members = np.arange(nc.variables["number"].size)
    if ind_long is None:
        ind_long = np.arange(nc.variables["longitude"].size)
    if ind_lat is None:
        ind_lat = np.arange(nc.variables["latitude"].size)
    list_var = [np.array(nc.variables[name]) for name in var_names]
    list_var = [var[ind_time,:,:,:] for var in list_var]
    list_var = [var[:,ind_members,:,:] for var in list_var]
    list_var = [var[:,:,ind_long,:] for var in list_var]
    list_var = [var[:,:,:,ind_lat] for var in list_var]

    # Now List of d (N, t, p, q) arrays
    list_var = [np.swapaxes(var, 0, 1).squeeze() for var in list_var]

    if multivariate:
        # Now (N, d, t, p, q) arrays
        list_var = np.swapaxes(list_var, 0, 1)
    else:
        # Now List of d (N, 1, t, p, q) arrays
        list_var = [np.expand_dims(var, axis=1) for var in list_var]

    d = {}
    d['members'] = list_var
    d['short_names'] = var_names
    d['time'] = nc.variables["time"][ind_time].data
    d['control'] = None
    d['long_names'] = [nc.variables[name].long_name for name in var_names]
    d['units'] = [nc.variables[name].units for name in var_names]

    return d


def preprocess_meteogram(
    filename: str,
    path_data: str = '',
    var_names: Union[List[str], str] = ['t2m'],
    ind_time: Union[np.ndarray, int] = None,
    ind_members: Union[np.ndarray, int] = None,
    ind_long: Union[np.ndarray, int] = 0,
    ind_lat: Union[np.ndarray, int] = 0,
    multivariate: bool = False,
    to_standardize: bool = False,
) -> Dict:
    """
    Extract and preprocess data

    :param filename: nc filename
    :type filename: str
    :param path_data: path to nc file, defaults to ''
    :type path_data: str, optional
    :param var_names:

        Names of the variables to extract,
        defaults to None, in this case
        ``` var_names =["t2m","d2m","msl","u10","v10","tcwv"]```

    :type var_names: Union[List[str], str], optional
    :param ind_time:

        Indices of time steps to extract,
        defaults to None, in this case all time steps are extracted

    :type ind_time: Union[np.ndarray, int], optional
    :param ind_members:

        Indices of members to extract,
        defaults to None, in this case all members are extracted

    :type ind_members: Union[np.ndarray, int], optional
    :param ind_long:

        Indices of longitudes to extract,
        defaults to None, in this case all elements are extracted

    :type ind_long: Union[np.ndarray, int], optional
    :param ind_lat:

        Indices of latitudes to extract,
        defaults to None, in this case all elements are extracted

    :type ind_lat: Union[np.ndarray, int], optional
    :param multivariate: Consider one multivariate variable, defaults to False
    :type multivariate: bool, optional
    :param to_standardize: Should variables be standardized, defaults to False
    :type to_standardize: bool, optional
    :return: Variables, their corresponding names, time axis and scalers
    :rtype: Tuple[Union[List[np.ndarray], np.ndarray], Union[List[str], str]]
    """

    print(filename)
    f = path_data + filename
    nc = Dataset(f,'r')

    data_dict = extract_from_meteogram(
        nc=nc,
        var_names=var_names,
        ind_time=ind_time,
        ind_members=ind_members,
        ind_long=ind_long,
        ind_lat=ind_lat,
        multivariate=multivariate,
    )

    # Extract date from filename
    i = re.search(r"\d\d\d\d", filename).start()
    data_dict['date'] = (
        filename[i:i+4] + '-'       # Year
        + filename[i+4:i+6] + '-'   # Month
        + filename[i+6:i+8] + '-'   # Day
        + filename[i+8:i+10]        # Hour
    )
    data_dict['filename'] = filename[:-3]

    # Take the log for the tcwv variable
    idx = get_indices_element(
        my_list=data_dict['short_names'],
        my_element="tcwv"
    )
    if idx != -1:
        if multivariate:
            raise NotImplementedError("Multivariate with log tcwv")
        for i in idx:
            data_dict['members'][i] = np.log(data_dict['members'][i])

    # Take Celsius instead of Kelvin
    if not to_standardize:
        idx = get_indices_element(
            my_list=data_dict['short_names'],
            my_element="t2m"
        )
        if idx != -1:
            if multivariate:
                raise NotImplementedError("Multivariate with t2m in °C")
            for i in idx:
                data_dict['members'][i] = data_dict['members'][i] - 273.15

    # If variables are to be standardized
    if to_standardize:
        if multivariate:
            raise NotImplementedError("Multivariate with standardization")
        (data_dict['scalers'], data_dict['members']) = standardize(
            list_var = data_dict['members'],
            each_loc = False,
        )

    # Set the initial conditions at time +0h
    data_dict['time'] -= data_dict['time'][0]
    return data_dict

def clear_double_space(filename):
    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

def extract_from_mjo(
    filename: str,
    path_data: str = '',
    smooth: bool = True,
) -> Dict:
    """
    Extract "members, time" from MJO datafiles

    The files are organized as follows:

    1. hours
    2. type of member (mean, control or perturbed)
    3. member's id
    4. RMM1
    5. RMM2
    6. radius
    7. MJO phase class

    :param filename: filename (including path)
    :type filename: str,
    :return: (members, time) of shape (N, 2, T) and (T) respectively
    :rtype: Dict
    """
    # Find files
    names = [x for x in 'abcdefghi'[:7]]
    df = pd.read_csv(path_data + filename, sep=' ', names=names, header=1)
    # Remove ensemble mean rows
    df = df[df['b'] != 'em']
    # Remove useless columns
    df = df[['a', 'c', 'd', 'e']]
    # Time steps
    time = list(set([h for h in df['a']]))
    time.sort()
    # Re-index time step in the df
    df[['a']] = df[['a']]//24-1
    # Reshape to (N, 2, T) array
    N = int(df[['c']].max())+1
    T = int(df[['a']].max())+1
    members = np.zeros((N, 2, T))
    # Find the right values
    for i in range(N):
        for t in range(T):
            values = df[(df.a == t) & (df.c == i)]
            members[i, 0, t] = float(values.d)
            members[i, 1, t] = float(values.e)
    if smooth:
        members = smoothing_mjo(members)

    d = {}
    d['time'] = np.array(time)
    d['members'] = members
    d['control'] = None
    d['units'] = "RMM"
    d["short_names"] = "rmm"
    d['long_names'] = 'Real-Time Multivariate Index'

    # Extract date from filename
    i = re.search(r"\d\d\d\d", filename).start()
    d['date'] = (
        filename[i:i+4] + '-'       # Year
        + filename[i+4:i+6] + '-'   # Month
        + filename[i+6:i+8] + '-'   # Day
        + filename[i+8:i+10]        # Hour
    )
    d['filename'] = filename[:-4]

    return d

def preprocess_mjo(
    filename: str,
    path_data: str = '',
    smooth: bool = False,
    ):
    clear_double_space(path_data + filename)
    return extract_from_mjo(
        filename=filename,
        path_data=path_data,
        smooth=smooth
    )

def jsonify(data_dict):
    res = dict(data_dict)
    for key, item in res.items():
        if isinstance(item, np.ndarray):
            res[key] = item.tolist()
    return res

def numpify(data_dict):
    res = dict(data_dict)
    for key, item in res.items():
        if isinstance(item, list):
            res[key] = np.array(item)
    return res

def _draw_mjo_line(values, arc=True):
    # values[0] is the first element inside the very weak circle
    # values[-1] is the last element inside the very weak circle
    # If there is just one step, return values[0]
    steps = len(values)
    if steps == 1:
        return np.array([values[0]]).T
    else:
        # Initialize the angle (note that if arc is false then phi is actually)
        # RMM2
        phi_start = values[0][1]
        phi_end = values[-1][1]
        if arc:
            # To polar
            polar = np.zeros_like(values)
            for p, v in zip(polar, values):
                # radius
                p[0] = np.sqrt(v[0]**2 + v[1]**2)
                # angle
                p[1] = (atan2(v[1], v[0])) % (2*np.pi)
            # If    (end -> 2pi -> start)

            phi_start = polar[0][1]
            phi_end = polar[-1][1]
            if abs(2*np.pi+phi_start -phi_end) < abs(phi_start -phi_end):
                # start is ahead of end yes, but if it is just a little bit
                # then go backward (4pi/3, 2pi/3 rule)
                if abs(2*np.pi+phi_start - phi_end) < 2*np.pi/3:
                    phi_start += 2*np.pi
            # Elif  (start -> 2pi -> end)
            elif abs(phi_start - (2*np.pi + phi_end)) < abs(phi_start -phi_end):
                # end if ahead of start, it's all good, just add 2pi to end
                phi_end += 2*np.pi
            # Else: the shortest path does not cross 2pi
            else:
                # if start is too ahead of end, do a loop (2/3, 1/3 rule)
                if (
                    (phi_start > phi_end)
                    and (abs(phi_start -phi_end) > 2*np.pi/3)
                ):
                    phi_end += 2*np.pi
        path = np.concatenate([
            [[p[0] for p in polar]],
            [np.linspace(phi_start, phi_end, num=steps, endpoint=True)]
        ], axis=0)
        if arc:
            # return to cartesian coordinates
            return to_cartesian(path)
        else:
            return path

def smoothing_mjo(members, cartesian=True, r=0.70):
    def cond(m, t):
        if cartesian:
            return (np.sqrt(m[0, t]**2 + m[1, t]**2) <= r)
        else:
            return (m[0, t] <= r)
    # If radius < r, find the next time step with
    # a radius > r and draw an arc between them
    # To know the direction of the arc use a 1/3 - 2-3 rule since it is
    # supposed to go anti-clockwise
    T = members.shape[-1]
    for m in members:
        t = 0
        while (t < T):
            values = []
            # If we enter the very weak circle
            if cond(m, t):
                t_start = t
                t_end = t + 1
                # The first element inside the very weak circle:
                values.append(np.copy(m[:, t]))
                while (t_end < T and cond(m, t_end)):
                    # The last element inside the very weak circle:
                    values.append(m[:, t_end])
                    t_end += 1
                m[:, t_start:t_end] =_draw_mjo_line(
                    values, arc=True
                )
                # update t for the next loop
                t = t_end
            else:
                t += 1
    return members


def _get_2pi_k(angles):
    #sign = np.sign(angles)
    T = len(angles)
    #next_neg = sign[0] > 0
    k = [0]*T
    for t in range(1, T):
        k[t] = k[t-1]
        # was positive and gets negative 'by the left' (+1: 4pi/3 rule)
        if (
            (angles[t-1] >= 0 and angles[t] < 0)
            and (abs(angles[t]+2*np.pi - angles[t-1]) <= 4*np.pi/3)
        ):
            k[t] += 1
        # was negative and gets positive 'by the left' (-1: 2pi/3 rule)
        elif (
            (angles[t] > 0 and angles[t-1] < 0)
            and (abs(angles[t-1]+2*np.pi - angles[t]) < 2*np.pi/3)
        ):
            k[t] -= 1
    return np.array(k)

def to_polar(members):
    members_conv = np.zeros_like(members)

    # if members is actually the mean for example
    if len(members.shape) == 2:
        # Radii
        members_conv[0, :] = np.sqrt(members[0, :]**2 + members[1, :]**2)
        # Phase (arctan2(x1, x2) = arctan(x1/x2)) (range: [-pi, pi])
        members_conv[1, :] = np.arctan2(members[1, :], members[0, :])
        # transform to [0, 2pi] range
        k = _get_2pi_k(members_conv[1, :])
        members_conv[1, :] += 2*np.pi*k
        assert np.all(members_conv[0, :]>=0)
    else:
        # Radii
        members_conv[:, 0, :] = np.sqrt(
            members[:, 0, :]**2 + members[:, 1, :]**2
        )
        # Phase (arctan2(x1, x2) = arctan(x1/x2)) (range: [-pi, pi])
        members_conv[:, 1, :] = np.arctan2(members[:, 1, :], members[:, 0, :])
        for m in members_conv[:, 1, :]:
            k = _get_2pi_k(m)
            m += 2*np.pi*k
        assert np.all(members_conv[:, 0, :]>=0)
    return members_conv

def to_cartesian(members):
    members_conv = np.zeros_like(members)
    # if members is actually the mean for example
    if len(members.shape) == 2:
        # RMM1 = r * cos(phi)
        members_conv[0, :] = members[0, :] * np.cos(members[1, :])
        # RMM2 = r * sin(phi)
        members_conv[1, :] = members[0, :] * np.sin(members[1, :])
    else:
        # RMM1 = r * cos(phi)
        members_conv[:, 0, :] = members[:, 0, :] * np.cos(members[:, 1, :])
        # RMM2 = r * sin(phi)
        members_conv[:, 1, :] = members[:, 0, :] * np.sin(members[:, 1, :])
    return members_conv
