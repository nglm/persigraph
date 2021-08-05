from math import exp

def sigmoid(
    x,
    range0_1 = True,
    shift=3.,
    a=6.,
    f0=0.7,
    f1=6,
):
    # Here we will get f(0) = 0 and f(1) = 1
    res = 1/(1+exp(-(a*x-shift))) + (2*x-1)/(1+exp(shift))
    # Use min and max because of precision error
    res = min(max(0, res), 1)
    # Here we will get f(0) = f0 and f(1) = f1
    if not range0_1:
        res = f1*res + f0*(1-res)
    return res

def range_rescale(
    x: float,
    x0: float = 0,
    x1: float = 1,
    f0: float = 0,
    f1: float = 1,
) -> float:
    """
    Get the [x0 x1] range into a [f0 f1] range and return x accordingly


    :param x: [description]
    :type x: float
    :param x0: [description], defaults to 0
    :type x0: float
    :param x1: [description], defaults to 1
    :type x1: float
    :param f0: [description], defaults to 0
    :type f0: float, optional
    :param f1: [description], defaults to 1
    :type f1: float, optional
    :return: [description]
    :rtype: float
    """
    if x < x0 or x > x1:
        raise ValueError('x must be within the [x0, x1] range: ',x, (x0, x1))
    a = (f1-f0) / (x1-x0)
    b = f0 - a*x0
    return a*x+b






def linear(
    x,
    range0_1 = True,
    f0=0.7,
    f1=6,
):
    # Use min and max because of precision error
    res = min(max(0, x), 1)
    # Here we will get f(0) = f0 and f(1) = f1
    if not range0_1:
        res = f1*res + f0*(1-res)
    return res
