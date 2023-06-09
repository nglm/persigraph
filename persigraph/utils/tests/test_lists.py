from ..lists import union_intervals, intersection_intervals
import numpy as np
from numpy.testing import assert_array_equal

def test_union_intervals():
    l1 = [[0.1, 0.3], [0.6, 0.8]]
    l2 = []
    output_exp = l1
    output = union_intervals(l1, l2)
    assert output == output_exp
    output = union_intervals(l2, l1)
    assert output == output_exp
    l2 = [[0.2, 0.7]]
    output_exp = [[0.1, 0.8]]
    output = union_intervals(l1, l2)
    assert output == output_exp
    l2 = [[0.2, 0.4]]
    output_exp = [[0.1, 0.4], [0.6, 0.8]]
    output = union_intervals(l1, l2)
    assert output == output_exp

def test_intersection_intervals():
    l1 = [[0.1, 0.3], [0.6, 0.8]]
    l2 = []
    output_exp = []
    output = intersection_intervals(l1, l2)
    assert output == output_exp
    l2 = [[0.2, 0.7]]
    output_exp = [[0.2, 0.3], [0.6, 0.7]]
    output = intersection_intervals(l1, l2)
    assert output == output_exp
    l2 = [[0.2, 0.4]]
    output_exp = [[0.2, 0.3]]
    output = intersection_intervals(l1, l2)
    assert output == output_exp