export function onMouseMemberAux(e, d, memberElem, interactiveGroupElem, classname) {
    // Find all figures in the document
    let figs = document.getElementsByClassName("container-fig");
    for (let i = 0; i < figs.length; i++) {
        let groupId = document.getElementById(figs[i].id + "_input").value
        if (groupId == interactiveGroupElem.value) {
            // Within the outter svg element of each fig, all ids are unique
            let svgElem = document.getElementById(figs[i].id + "_svg");
            // Change class of the member that has the same id
            try {
                svgElem.getElementById(memberElem.id)
                    .setAttribute("class", classname);
            }
            // (err is caught if this figure was actually cluster plot)
            catch(err) {}
        }
    }
}

export function onMouseClusterAux(e, d, clusterElem, interactiveGroupElem, classname1, classname2) {
    // Find all figures in the document
    let figs = document.getElementsByClassName("container-fig");
    for (let i = 0; i < figs.length; i++) {
        // Check if the current fig belongs to the same interactive group
        let groupId = document.getElementById(figs[i].id + "_input").value
        if (groupId == interactiveGroupElem.value) {
            // Within the outter svg element of each fig, all ids are unique
            let svgElem = document.getElementById(figs[i].id + "_svg");
            try {
                // Correspondance between that v-event is the v id
                id = "v" + clusterElem.id.slice(7);
                // Change class of the cluster that has the associated id
                svgElem.getElementById(id)
                    .setAttribute("class", classname1);
            }
            // (err is caught if this figure was actually spaghetti plot)
            catch(err) {}
            // Change class of all members in that cluster
            for (var m of d.members) {
                try {
                    svgElem.getElementById("m" + m)
                        .setAttribute("class", classname2);
                }
                // (err is caught if this figure was actually cluster plot)
                catch(err) {}
            }
        }
    }
}