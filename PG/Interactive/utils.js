export function getParents(elem, rootElem='body', reversed=false) {
    var parents = [];
    while(
        elem.parentNode
        && elem.parentNode.nodeName.toLowerCase() != rootElem
    ) {
        elem = elem.parentNode;
        parents.push(elem);
    }
    if (reversed) {
        parents.reverse();
    }
    return parents;
}


export function sigmoid(
    x,
    {
        range0_1 = true,
        shift=3.,
        a=6.,
        f0=0.7,
        f1=6,
    } = {}
) {
    // Here we will get f(0) = 0 and f(1) = 1
    let res = 1/(1+Math.exp(-(a*x-shift))) + (2*x-1)/(1+Math.exp(shift));
    // Use Math.min and Math.max because of precision error
    res = Math.min(Math.max(0, res), 1);
    // Here we will get f(0) = f0 and f(1) = f1
    if (!range0_1) {
        res = f1*res + f0*(1-res);
    }
    return res;
}

export function range_rescale(
    x,
    {
        x0=0,
        x1=1,
        f0=0,
        f1=1
    } = {} ) {
    if ((x < x0) || (x > x1)) {
        throw 'x is outside range';
    }
    let a = (f1-f0) / (x1-x0);
    let b = f0 - a*x0;
    return a*x+b;
}

export function linear(
    x,
    {
        range0_1 = true,
        f0=0.7,
        f1=6,
    } = {} ) {
    // Use Math.min and Math.max because of precision error
    let res = Math.min(Math.max(0, x), 1)
    // Here we will get f(0) = f0 and f(1) = f1
    if (!range0_1) {
        res = f1*res + f0*(1-res);
    }
    return res;
}