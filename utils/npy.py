def running_mean(vect, window):
    cumsum = np.cumsum(np.insert(vect, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def unpack_2d_dimensions(
    array,
    return_2d_array: bool = True,
    assume_one_sample: bool = True,
):
    """
    Return 2d shape and array

    :param array: Array from which a 2d shape is extracted
    :type array: [type]
    :param return_2d_array: if true reshape the array, defaults to True
    :type return_2d_array: bool, optional
    :param assume_one_sample:

    If array has only one dim is it because should we assume is has
    only one sample or one value dimension?, defaults to True

    :type assume_one_sample: bool, optional
    """
    dims = array.shape
    if len(dims) < 2:
        if assume_one_sample:
            n_samples = 1       # size of the sample
            n_values = dims[0]  #dimension of one sample
        else:
            n_samples = dims[0]  # size of the sample
            n_values = 0         #dimension of one sample

        if return_2d_array:
            array = np.reshape(array, (n_samples, n_values))
    elif len(dims) == 2:
        (n_samples, n_values) = dims
    return((n_samples, n_values), array)
