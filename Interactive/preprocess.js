export function d3fy(data) {

    var data_xy = []; // start empty, add each element one at a time
    var xy = [];

    // Iterate over members
    for (var m = 0; m < data.members.length; m++ ) {
        xy = [];
        // Iterate over time steps
        for(var i = 0; i < data.time.length; i++ ) {
            // Initialize object and add time
            let obj = {t: data.time[i]};
            // Iterate over variables
            for(var k = 0; k < data.members[0].length; k++ ) {
                // Add each variable one by one if mutlivariate
                obj[data.var_names[k]] =  data.members[m][k][i];
            }
            xy.push(obj);
        }
        data_xy.push(xy);
    }
    return data_xy
}

export function d3fy_life_span(data) {
    // WARNING: Does not give real time step, just the index of the time
    // step

    var data_xy = []; // start empty, add each element one at a time
    var xy = [];
    console.log(data)

    // Iterate over keys (k values)
    for (const [key, value] of Object.entries(data)) {
        xy = [];
        // Iterate over time steps
        for(var i = 0; i < value.length; i++ ) {
            xy.push({t: i, k: key, life_span: value[i]});
        }
        data_xy.push(xy);
    }
    return data_xy
}