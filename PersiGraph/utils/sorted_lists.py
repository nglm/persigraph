
from typing import List, Tuple, Any, Sequence
from bisect import *

def bisect_search(
    a: Sequence[Any],
    x: Any,
    lo: int = 0,
    hi: int = None,
) -> int:
    """
    Binary search of ``x`` in ``a`` (``a`` sorted in increasing order)

    :param a: Sequence in which ``x`` has to be found
    :type a: Sequence[Any]
    :param x: Element to be found in ``a``
    :type x: Any
    :return: Index of ``x`` in ``a`` if found, otherwise ``-1``
    :rtype: int
    """
    if hi is None:
        hi = len(a)
    i = bisect_left(a, x, lo=lo, hi=hi)
    # If found we have ofc a[i] == x
    # If not found and x > to elt in a then i = len(a)
    # and therefore a[i] would raise an index error
    if i != hi and a[i] == x:
        return i
    else:
        return -1



def reverse_bisect_left(a, x, lo=0, hi=None):
    """
    Index where to insert x in a, assuming a is sorted in decreasing order

    :param a: Sequence
    :type a: Sequence[Any]
    :param x: Element
    :type x: Any
    :return: The index where x would inserted to respect the order while being
    the leftmost of its kind
    :rtype: int
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x >= a[mid]: hi = mid
        else: lo = mid+1
    return lo

def reverse_bisect_right(a, x, lo=0, hi=None):
    """
    Index where to insert x in a, assuming a is sorted in decreasing order

    :param a: Sequence
    :type a: Sequence[Any]
    :param x: Element
    :type x: Any
    :return: The index where x would inserted to respect the order while being
    the rightmost of its kind
    :rtype: int
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x > a[mid]: hi = mid
        else: lo = mid+1
    return lo


def insert_no_duplicate(a, x, lo=0, hi=None):
    """Insert item x in list a if x is not already in a and keep it
    sorted assuming a is sorted.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if hi is None:
        hi = len(a)
    if a:
        idx = bisect_right(a, x, lo=lo, hi=hi)
        if not (idx > 0  and a[idx-1] == x):
            a.insert(idx, x)
    else:
        a.insert(0,x)

def has_element(
    l: Sequence[Any],
    x: Any,
    len_l: int = None) -> bool:
    """
    Find out whether x is in l.

    Assume that l is sorted in increasing order

    :param l: Ordered Sequence
    :type l: Sequence[Any]
    :param x: Element searched for
    :type x: Any
    :param len_l: length of l, if already computed, defaults to None
    :type len_l: int, optional
    :return: True if the element is found in l
    :rtype: bool
    """
    # Bisect_search returns -1 if the element is not found
    return (bisect_search(l, x, lo=0, hi=len_l) >= 0)

def get_common_elements(
    l1: Sequence[Any],
    l2: Sequence[Any],
) -> List[Any]:
    """
    Find all elements belonging to both l1 and l2

    Assume that both l1 and l2 are sorted in increasing order

    :param l1: [description]
    :type l1: Sequence[Any]
    :param l2: [description]
    :type l2: Sequence[Any]
    :return: A sorted list containing the common elements
    :rtype: List[Any]
    """
    return [ e for e in l1 if has_element(l2, e) ]

def concat_no_duplicate(
    l1: Sequence[Any],
    l2: Sequence[Any],
    len_l1: int = None,
    len_l2: int = None,
    copy: bool = True,
) -> List[Any]:
    """
    Find all elements belonging either to l1 or l2

    Assume that both l1 and l2 are sorted in increasing order

    :param l1: [description]
    :type l1: Sequence[Any]
    :param l2: [description]
    :type l2: Sequence[Any]
    :return: A sorted list containing the common elements
    :rtype: List[Any]
    """
    if len_l1 is None:
            len_l1 = len(l1)
    if copy:
        if len_l2 is None:
            len_l2 = len(l2)
        if len_l1 <= len_l2:
            l_new = l1.copy()
            for x in l2:
                # hi != len_l1 because len_l1 increases at each step
                insert_no_duplicate(l_new, x, lo=0, hi=None)
            return l_new
        else:
            return concat_no_duplicate(l2, l1, len_l2, len_l1)
    else:
        for x in l2:
            insert_no_duplicate(l1, x, lo=0, hi=None)
        return l1

def concat_with_duplicate(
    l1: Sequence[Any],
    l2: Sequence[Any],
    len_l1: int = None,
    len_l2: int = None,
    copy: bool = True,
) -> List[Any]:
    """
    Find all elements belonging either to l1 or l2

    Assume that both l1 and l2 are sorted in increasing order

    :param l1: [description]
    :type l1: Sequence[Any]
    :param l2: [description]
    :type l2: Sequence[Any]
    :return: A sorted list containing the common elements
    :rtype: List[Any]
    """
    if len_l1 is None:
            len_l1 = len(l1)
    if copy:
        if len_l2 is None:
            len_l2 = len(l2)
        if len_l1 <= len_l2:
            l_new = l1.copy()
            for x in l2:
                insort(l_new, x)
            return l_new
    else:
        for x in l2:
            insort(l1, x)
        return l1

def remove_duplicate(
    l: Sequence
):
    """
    Assume that l is sorted (increasing OR decreasing order)

    :param l: [description]
    :type l: Sequence
    :return: [description]
    :rtype: [type]
    """
    return [x for i,x in enumerate(l) if i>0 and x != l[i-1]]
