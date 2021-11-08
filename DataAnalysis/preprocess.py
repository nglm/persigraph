import numpy as np
import pandas as pd
import warnings
from math import atan2
from typing import List, Sequence, Union, Any, Dict, Tuple

def clear_double_space(filename):
    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

def extract_from_files(
    filename: str,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
    :return: (members, time) of shape (N, 2, T) and () respectively
    :rtype: Tuple(ndarray, ndarray)
    """
    # Find files
    names = [x for x in 'abcdefghi'[:7]]
    df = pd.read_csv(filename, sep=' ', names=names, header=1)
    # Remove ensemble mean rows
    df = df[df['b'] != 'em']
    # Remove useless columns
    df = df[['a', 'c', 'd', 'e']]
    # Time steps
    time = list(set([h for h in df['a']]))
    time.sort()
    time = np.array(time)
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
    return members, time

def preprocessing_mjo(filename, smooth=False):
    clear_double_space(filename)
    return extract_from_files(filename, smooth=smooth)

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
