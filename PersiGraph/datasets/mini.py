import numpy as np

members_with_dup_equal_dist = np.array([
    (0.1 ,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.05,  -0.4,  -0.6, -0.5,   -1),
])

members_with_equal_dist = np.array([
    (0.11,   1.,    2.,   1.,     0.1 ),
    (0.05,   0,     0,    0,      0.005 ),
    (0.1,   -1,    -2,   -1,      0.05 ),
    (0.005, -0.4,  -0.6, -0.5,   -1),
])

members = np.array([
    (0.11,   1.1,   2.1,  1.1,    0.1),
    (0.05,   0,     0,    0,      0.005),
    (0.1,   -1,    -2,   -1,      0.05),
    (0.005, -0.4,  -0.6, -0.55,  -1),
])

members_bis = np.array([
    ( 0.2 ,  0.3,    1,      1.1,    0.6),
    ( 0.1,   0.15,   0.5,    0.6,    0.4),
    (-0.1,  -0.2,   -1,     -1.2,   -1 ),
    (-0.3,  -0.4,   -0.6,   -0.7,   -1),
])

members_biv = np.ones((4,2,5))
members_biv[:,0,:] = members
members_biv[:,1,:] = members_bis

def mini(
    vtype = "univariate",
    time_scale=True,
    duplicates=False,
    equal_dist=False,
):
    (N, T) = members.shape
    time = np.arange(T)

    # To get a time axis different from the indices
    if time_scale:
        time *= 6
    if vtype == "multivariate":
        data = np.ones((N, 2, T))
        data[:,0,:] = members
        data[:,1,:] = members_bis
        data = members_biv
    else:
        if duplicates:
            data = members_with_dup_equal_dist
        else:
            if equal_dist:
                data = members_with_equal_dist
            else:
                data = members
        data = np.expand_dims(data, 1)
    return np.copy(data), np.copy(time)
