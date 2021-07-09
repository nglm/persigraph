import numpy as np
from PIL import ImageColor
from math import ceil, floor, sqrt
from typing import List, Sequence, Union, Any, Dict, Tuple


# See https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=8
COLOR_BREWER = [
    "#636363", # Grey
    "#377eb8", # Blue
    "#a65628", # Brown
    "#984ea3", # Purple
    "#e41a1c", # Red
    "#4daf4a", # Green
    "#ff7f00", # Orange
    "#f781bf", # Pink
    "#ffff33", # Yellow
]

COLOR_BREWER_RGB = [
    np.array(ImageColor.getcolor(c, "RGB"))/255 for c in COLOR_BREWER
    ]
COLOR_BREWER_RGBA = [np.r_[c, np.ones(1)] for c in COLOR_BREWER_RGB]


def get_list_colors(
    N: int = None,
) -> List:
    """
    Repeat COLOR_BREWER list until we get exactly N colors

    :param N: Number of colors desired
    :type N: int
    :return: List of colors (taken from COLOR_BREWER list)
    :rtype: List
    """
    if N is None:
        N = 100
    n_cb = len(COLOR_BREWER)
    list_colors = []
    for i in range(1 + N//n_cb) :
        list_colors += COLOR_BREWER_RGBA
    return list_colors[:(N+1)]


def nrows_ncols(n: int) -> Tuple[int, int]:
    """
    Get a number a rows and columns from a number of axes

    :param n: Number of axes (ex: dimension of physical variable)
    :type n: int
    :return: Number of rows and columns of the figure
    :rtype: Tuple[int, int]
    """
    if n == 1:
        nrows = 1
        ncols = 1
    else:
        nrows = floor(sqrt(n))
        ncols = ceil(n/nrows)
    return nrows, ncols