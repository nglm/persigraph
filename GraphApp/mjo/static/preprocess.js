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

    // Iterate over keys (key, values)
    for (const [key, value] of Object.entries(data)) {
        xy = [];
        // Iterate over time steps
        for (var i = 0; i < value.length; i++ ) {
            xy.push({t: i, k: key, life_span: value[i]});
        }
        data_xy.push(xy);
    }
    return data_xy
}

export function d3fy_dict_of_arrays(data) {
    // WARNING: this function expects data to be a dict of arrays, all arrays
    // having the same shape
    // WARNING: Does not give real time step, just the index of the time
    // step

    var data_xy = []; // start empty, add each element one at a time
    // Common length to all arrays in the dict of arrays
    let n = Object.values(data)[0].length;

    // Common index to each array in the dict
    for (var i = 0; i < n; i++ ){
        let obj = {t: i};
        // Create a key[i], value[i] pair for each key in dict
        for (const [key, value] of Object.entries(data)) {
            obj[key] = value[i];
        }
        data_xy.push(obj);
    }
    return data_xy
}