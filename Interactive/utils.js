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
        parents.reverse()
    }
    return parents;
}