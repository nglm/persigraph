export function onMouseMemberAux(e, d, memberElem, interactiveGroupElem, classname) {
    let figs = document.getElementsByClassName("container-fig");
    for (let i = 0; i < figs.length; i++) {
        let groupId = document.getElementById(figs[i].id + "_input").value
        if (groupId == interactiveGroupElem.value) {
            let svgElem = document.getElementById(figs[i].id + "_svg");
            try {
                svgElem.getElementById(memberElem.id)
                    .setAttribute("class", classname);
            }
            catch(err) {}
        }
    }
}

export function onMouseClusterAux(e, d, memberElem, interactiveGroupElem, classname1, classname2) {
    let figs = document.getElementsByClassName("container-fig");
    for (let i = 0; i < figs.length; i++) {
        let groupId = document.getElementById(figs[i].id + "_input").value
        if (groupId == interactiveGroupElem.value) {
            let svgElem = document.getElementById(figs[i].id + "_svg");
            try {
                svgElem.getElementById(memberElem.id)
                    .setAttribute("class", classname1);
            }
            catch(err) {}
            for (var m of d.members) {
                let svgElem = document.getElementById(figs[i].id + "_svg");
                try {
                    svgElem.getElementById("m" + m)
                        .setAttribute("class", classname2);
                }
                catch(err) {}
            }
        }
    }
}