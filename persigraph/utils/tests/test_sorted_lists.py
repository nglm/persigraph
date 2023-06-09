from ..sorted_lists import *

def test_has_element():

    l1 = [-1,2,3,6]

    elts = [
        0,
        6,
        4,
        -1,
        -2,
        8,
        2,
    ]

    res_has_expect = [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
    ]


    for i in range(len(elts)):
        print(i)
        args = {
            "l" : l1,
            'x' : elts[i]
            }
        output = has_element(**args)
        output_exp = res_has_expect[i]
        assert output == output_exp

def test_get_common_elements():

    output = get_common_elements([], [])
    output_exp = []
    assert output == output_exp

    l1 = [-1,2,3,6]
    l1_bis = [-1,0,3,6] # Because 0 seems to be a tricky case
    l2s = [
        [],
        [0],
        [0, 6],
        [0, 4],
        [-1],
        [6],
        [2, 3],
        [2, 3, 4],
        [-3],
        [8],
        l1,
    ]


    res_common_expect = [
        [],
        [],
        [6],
        [],
        [-1],
        [6],
        [2,3],
        [2,3],
        [],
        [],
        l1,
    ]

    res_common_expect_bis = [
        [],
        [0],
        [0,6],
        [0],
        [-1],
        [6],
        [3],
        [3],
        [],
        [],
        [-1, 3, 6],
    ]

    for i in range(len(l2s)):
        args = {
            "l1" : l1,
            'l2' : l2s[i]
            }
        output = get_common_elements(**args)
        output_exp = res_common_expect[i]
        assert output == output_exp
        args = {
            "l1" : l1_bis,
            'l2' : l2s[i]
            }
        output = get_common_elements(**args)
        output_exp = res_common_expect_bis[i]
        assert output == output_exp

        # Symmetry
        args = {
            "l2" : l1,
            'l1' : l2s[i]
            }
        output = get_common_elements(**args)
        output_exp = res_common_expect[i]
        assert output == output_exp
        args = {
            "l2" : l1_bis,
            'l1' : l2s[i]
            }
        output = get_common_elements(**args)
        output_exp = res_common_expect_bis[i]
        assert output == output_exp

def test_insert_no_duplicate():

    output = []
    insert_no_duplicate(output, -1)
    output_exp = [-1]
    assert output == output_exp

    l1 = [-1 ,0, 3, 8]
    l2s = [
        0,
        2,
        -2,
        9,
        8,
        -1,
    ]

    res_insert_no_dup = [
        l1,
        [-1 ,0, 2, 3, 8],
        [-2, -1 ,0, 3, 8],
        [-1 ,0, 3, 8, 9],
        l1,
        l1,
    ]

    for i in range(len(l2s)):
        args = {
            "l1" : l1,
            'l2' : l2s[i]
            }
        output = l1.copy()
        insert_no_duplicate(output, l2s[i])
        output_exp = res_insert_no_dup[i]
        assert output == output_exp

def test_concat_no_duplicate():
    l1 = [-1 ,0, 3, 8]
    l2s = [
        [],
        [0],
        [2],
        [-2, 2, 9],
        [8],
        [-1],
    ]

    res_concat_no_dup = [
        l1,
        l1,
        [-1 ,0, 2, 3, 8],
        [-2, -1 ,0, 2, 3, 8, 9],
        l1,
        l1,
    ]

    for i in range(len(l2s)):
        args = {
            "l1" : l1,
            'l2' : l2s[i]
            }
        output = concat_no_duplicate(**args)
        output_exp = res_concat_no_dup[i]
        assert output == output_exp

        # Symmetry
        args = {
            "l2" : l1,
            'l1' : l2s[i]
            }
        output = concat_no_duplicate(**args)
        output_exp = res_concat_no_dup[i]
        assert output == output_exp

def test_are_equal():
    l1s = [
        [], [0], [0], [0], [1,2,3], [1,2,3]
    ]
    l2s = [
        [], [1], [0], [0, 1], [1,2,3], [0,1,2]
    ]
    output_exp = [
        True, False, True, False, True, False
    ]
    for i in range(len(l1s)):
        output = are_equal(l1s[i], l2s[i])

        assert output == output_exp[i]