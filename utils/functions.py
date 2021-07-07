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
