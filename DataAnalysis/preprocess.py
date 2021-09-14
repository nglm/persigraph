import numpy as np
import pandas as pd
import warnings
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
    members = to_polar(members)
    members = smoothing_mjo(members)
    members = to_cartesian(members)

    return members, time

def preprocessing_mjo(filename):
    clear_double_space(filename)
    return extract_from_files(filename)

def _draw_mjo_line(start, end, steps):
    # start is the first element inside the very weak circle
    # end is the last element inside the very weak circle
    # steps is the number of time steps spent inside this cercle
    # If there is just one step, return start

    if steps == 1:
        return np.array([start]).T
    else:
        return np.concatenate([
            [np.linspace(start[0], end[0], num=steps, endpoint=True)],
            [np.linspace(start[-1], end[-1], num=steps, endpoint=True)]
        ], axis=0)

# def _draw_mjo_arc(start, end, steps):
#     # If there is just one step, return start
#     # If 2 steps:  line from start to end
#     # If 'steps' >= 3:
#     # - 1 point from start to go to r=0.5 (constant phase)
#     # - 'steps'-2 points on the arc (until we reach phase_end)
#     # - 1 point to go to end ()
#     pass

def smoothing_mjo(members):
    # If radius < 0.5, find the next time step with
    # a radius > 0.5 and draw an arc between them
    # To know the direction of the arc use a 1/3 - 2-3 rule since it is
    # supposed to go anti-clockwise
    T = members.shape[-1]
    for m in members:
        t = 0
        while (t < T):
            # If we enter the very weak circle
            if m[0, t] <= 0.5:
                t_start = t
                t_end = t + 1
                # The first element inside the very weak circle:
                start = np.copy(m[:, t])
                while (t_end < T and m[0, t_end] <= 0.5):
                    t_end += 1
                # The last element inside the very weak circle:
                end = np.copy(m[:, t_end-1])
                m[:, t_start:t_end] =_draw_mjo_line(
                    start, end, t_end-t_start
                )
                # update t for the next loop
                t = t_end
            else:
                t += 1
        print(m)
    return members


def _get_2pi_k(angles):
    sign = np.sign(angles)
    T = len(sign)
    next_neg = sign[0] > 0
    k = [0]*T
    for i in range(1, T):
        k[i] = k[i-1]
        # If we wait for the next neg to add 2pi...
        if next_neg:
            # And this is indeed neg
            if sign[i] == 0:
                # And coming from upper left to lower left corner
                if (
                    (angles[i-1] % (2*np.pi)) > np.pi/2     # upper left
                    and (angles[i] % (2*np.pi)) < 3*np.pi/2 # lower left
                ):
                    k[i] += 1
                next_neg = False
        # We were in a neg phase
        else:
            # And we found a pos phase
            if sign[i] > 0:
                # And coming from lower right to upper right corner
                if (
                    (angles[i-1] % (2*np.pi)) > 3*np.pi/2   # lower right
                    and (angles[i] % (2*np.pi)) < np.pi/2 # upper right
                ):
                    next_neg = True
    return np.array(k)

def to_polar(members):
    members_conv = np.zeros_like(members)

    # if members is actually the mean for example
    if len(members.shape) == 2:
        # Radii
        members_conv[0, :] = np.sqrt(members[0, :]**2 + members[1, :]**2)
        # Phase (arctan2(x1, x2) = arctan(x1/x2)) (range: [-pi, pi])
        members_conv[1, :] = np.arctan(members[1, :], members[0, :])
        # transform to [0, 2pi] range
        k = _get_2pi_k(members_conv[1, :])
        members_conv[1, :] += 2*np.pi*k
    else:
        # Radii
        members_conv[:, 0, :] = np.sqrt(
            members[:, 0, :]**2 + members[:, 1, :]**2
        )
        # Phase (arctan2(x1, x2) = arctan(x1/x2)) (range: [-pi, pi])
        members_conv[:, 1, :] = np.arctan(members[:, 1, :], members[:, 0, :])
        for m in members_conv[:, 1, :]:
            k = _get_2pi_k(m)
            m += 2*np.pi*k
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
