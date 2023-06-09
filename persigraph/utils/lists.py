from typing import List, Tuple, Any, Sequence, Iterable, Union

def get_indices_element(
    my_list: List[Any],
    my_element: Any,
    all_indices: bool = True,
    if_none: Any = -1,
):
    """
    Return the first or all indices of an element

    :param my_list: List in which 'my_element' is searched
    :type my_list: List[Any]
    :param my_element: Element that is to be found
    :type my_element: Any
    :param all_indices: Return all indices?, defaults to True
    :type all_indices: bool, optional
    :param if_none: Return value if element not found, defaults to -1
    :type if_none: int, optional
    :return: Index of the element in the list
    :rtype: int if element found otherwise type(if_none)
    """
    if my_list is None:
        res = if_none
    indices = [i for i,x in enumerate(my_list) if x == my_element]
    res = indices
    # if my_element is not in the list
    if not indices:
        res = if_none
    # if we only want one index
    elif not all_indices:
        res = [indices[0]]
    return res  # List[int] or Type(if_none)

def flatten(
    list_of_list: List[Any],
) -> List[Any]:
    """
    Recursively flatten nested list

    :param list_of_list: (Potentially) nested list
    :type list_of_list: List[Any]
    """
    flat_list = []
    # Base case: not even a list
    if not isinstance(list_of_list, list):
        flat_list = [list_of_list]
    # Base case: empty list
    elif list_of_list == []:
        flat_list = []
    # Recursive call
    else:
        for alist in list_of_list:
            flat_alist = flatten(alist)
            flat_list += flat_alist
    return(flat_list)

def to_iterable(x: Any) -> List:
    """
    Return ``x`` if ``x`` is iterable, otherwise return ``[x]``

    :param x: Element to convert to iterable
    :type x: Any
    :return: ``x`` if ``x`` is iterable, otherwise ``[x]``
    :rtype: List
    """
    if isinstance(x, Iterable):
        return x
    else:
        return [x]

def to_list(x: Any) -> List:
    """
    Return ``x`` if ``x`` is a list, otherwise return ``[x]``

    :param x: Element to convert to a list
    :type x: Any
    :return: ``x`` if ``x`` is a list, otherwise ``[x]``
    :rtype: List
    """
    if isinstance(x, List):
        return x
    else:
        return [x]

def is_sorted(l: List) -> bool:
    """
    Return True if ``l`` is sorted

    :param l: List to check
    :type l: List
    :return: True if ``l`` is sorted, otherwise ``False``
    :rtype: bool
    """
    return (all(l[i] <= l[i + 1] for i in range(len(l)-1)))

def intersection_intervals(
    intervals1: List[Sequence[float]],
    intervals2: List[Sequence[float]]
) -> List[Sequence[float]]:
    """
    Compute intersection of intervals

    By "interval" we mean a sequence (x1, x2) such that x1 <= x2.

    :param intervals1: First list of intervals
    :type intervals1: List[Sequence[float]]
    :param intervals2: Second list of intervals
    :type intervals2: List[Sequence[float]]
    :return: Intersection of list of intervals
    :rtype: List[Sequence[float]]
    """
    intervals = []
    for inter1 in intervals1:
        for inter2 in intervals2:

            start_inter = max(inter1[0], inter2[0])
            end_inter = min(inter1[1], inter2[1])

            if start_inter < end_inter:
                intervals.append([start_inter, end_inter])

    return intervals

def union_interval(
    interval1: Sequence[float],
    interval2: Sequence[float],
) -> List[Sequence[float]]:
    """
    Compute union of 2 intervals.

    :param interval1: _description_
    :type interval1: Sequence[float]
    :param interval2: _description_
    :type interval2: Sequence[float]
    :rtype: List[Sequence[float]]
    """
    union = []
    flag = False
    intersection = intersection_intervals([interval1], [interval2])
    if (intersection == []):
        # If they are next to each other, merge
        if (interval1[1] == interval2[0] or interval2[1] == interval1[0]):
            start = min(interval1[0], interval2[0])
            end = max(interval1[1], interval2[1])
            union = [[start, end]]
        else:
            union = [interval1, interval2]
    else:
        # Merge
        start = min(interval1[0], interval2[0])
        end = max(interval1[1], interval2[1])
        union = [[start, end]]
    return union

def union_intervals(
    intervals1: List[Sequence[float]],
    intervals2: List[Sequence[float]]
) -> List[Sequence[float]]:
    """
    Compute union of list of intervals.

    By "interval" we mean a sequence (x1, x2) such that x1 <= x2.

    Assume that intervals within intervals1 and intervals2 have no
    intersection.

    :param intervals1: First list of intervals
    :type intervals1: List[Sequence[float]]
    :param intervals2: Second list of intervals
    :type intervals2: List[Sequence[float]]
    :return: Union of list of intervals
    :rtype: List[Sequence[float]]
    """
    if intervals1 == []:
        return intervals2
    elif intervals2 == []:
        return intervals1
    else:

        # Sort segments based on starting points
        sorted_segments = sorted(intervals1+intervals2, key=lambda x: x[0])

        # Initialize result with the first segment
        union = [sorted_segments[0][:]]

        for segment in sorted_segments[1:]:
            last_segment = union[-1]

            # Check for overlap
            if segment[0] <= last_segment[1]:
                last_segment[1] = max(segment[1], last_segment[1])
            else:
                union.append(segment)

        return union
