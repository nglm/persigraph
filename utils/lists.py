from typing import List, Tuple, Any, Sequence, Iterable

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
