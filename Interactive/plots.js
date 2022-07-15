
import { d3fy, d3fy_life_span } from "./preprocess.js";

import {
    dimensions, setAxTitle, setFigTitle, setXLabel, setYLabel,
    draw_mjo_classes, draw_fig, style_ticks, get_list_colors, add_axes,
} from "./figures.js"
import { onMouseClusterAux, onMouseMemberAux } from "./interact.js";

import {range_rescale, sigmoid, linear} from "./utils.js"

    // <!-- simple dot marker definition -->
    // <marker id="dot" viewBox="0 0 10 10" refX="5" refY="5"
    //     markerWidth="5" markerHeight="5">
    // <circle cx="5" cy="5" r="5" fill="red" />
    // </marker>
    // </defs>

    // <!-- Coordinate axes with a arrowhead in both direction -->
    // <polyline points="10,10 10,90 90,90" fill="none" stroke="black"
    // marker-start="url(#arrow)" marker-end="url(#arrow)"  />

    // <!-- Data line with polymarkers -->
    // <polyline points="15,80 29,50 43,60 57,30 71,40 85,15" fill="none" stroke="grey"
    // marker-start="url(#dot)" marker-mid="url(#dot)"  marker-end="url(#dot)" />

function f_life_span(life_span, vmax, {vmin=0} = {}) {
    let rescaled = range_rescale(life_span, {x0:vmin, x1:vmax});
    return sigmoid(rescaled, {range0_1:true});
}


function f_polygon_edge(d, g, xscale, yscale, iplot) {
    let offset = 0.*xscale(g.time_axis[1]);
    let t = d.time_step;
    let mean_start = d.info_start.mean[iplot];
    let mean_end = d.info_end.mean[iplot];
    let points = offset + xscale(g.time_axis[t])+","+yscale(mean_start - d.info_start.std_inf[iplot])+" ";
    points += offset + xscale(g.time_axis[t])+","+yscale(mean_start + d.info_start.std_sup[iplot])+" ";
    points += -offset + xscale(g.time_axis[t+1])+","+yscale(mean_end + d.info_end.std_sup[iplot])+" ";
    points += -offset + xscale(g.time_axis[t+1])+","+yscale(mean_end - d.info_end.std_inf[iplot])+" ";
    return points;
}

function f_line_edge_detailed(d, g, xscale, yscale, iplot) {
    let t = d.time_step;
    let start_edge = d.info_start.mean[iplot];
    let end_edge = d.info_end.mean[iplot];
    let start_vertex = g.vertices[t][d.v_start].info.mean[iplot];
    let end_vertex = g.vertices[t+1][d.v_end].info.mean[iplot];
    let points = xscale(g.time_axis[t])+","+yscale(start_vertex)+" ";
    points += xscale(g.time_axis[t])+","+yscale(start_edge)+" ";
    points += xscale(g.time_axis[t+1])+","+yscale(end_edge)+" ";
    points += xscale(g.time_axis[t+1])+","+yscale(end_vertex);
    return points;
}

function f_line_edge(d, g, xscale, yscale, iplot) {
    let t = d.time_step;
    let start_vertex = g.vertices[t][d.v_start].info.mean[iplot];
    let end_vertex = g.vertices[t+1][d.v_end].info.mean[iplot];
    let points = xscale(g.time_axis[t])+","+yscale(start_vertex)+" ";
    points += xscale(g.time_axis[t+1])+","+yscale(end_vertex);
    return points;
}

function f_line_edge_mjo(d, g, xscale, yscale) {
    let t = d.time_step;
    let mean_start = g.vertices[t][d.v_start].info.mean;
    let mean_end = g.vertices[t+1][d.v_end].info.mean;
    let points = xscale(mean_start[0])+","+yscale(mean_start[1])+" ";
    points += xscale(mean_end[0])+","+yscale(mean_end[1]);
    return points;
}

function f_line_life_span(d, g, xscale, yscale) {
    let t = d.time_step;
}

function f_polygon_vertex(d, g, xscale, yscale, iplot) {
    let offset = 0.2*xscale(g.time_axis[1]);
    let t = d.time_step;
    let points = ""
    for (let e_num of d.e_to) {
        let e = g.edges[t-1][e_num];
        let mean_start = e.info_end.mean[iplot];
        let mean_end = d.info.mean[iplot];
        points += -offset + xscale(g.time_axis[t])+","+yscale(mean_start + e.info_end.std_sup[iplot])+" ";
        points += -offset + xscale(g.time_axis[t])+","+yscale(mean_start - e.info_end.std_inf[iplot])+" ";
        points += xscale(g.time_axis[t])+","+yscale(mean_end - d.info.std_inf[iplot])+" ";
        points += xscale(g.time_axis[t])+","+yscale(mean_end + d.info.std_sup[iplot])+" ";
    }
    for (let e_num of d.e_from) {
        let e = g.edges[t][e_num];
        let mean_start = d.info.mean[iplot];
        let mean_end = e.info_start.mean[iplot];
        points += offset + xscale(g.time_axis[t])+","+yscale(mean_end + e.info_start.std_sup[iplot])+" ";
        points += offset + xscale(g.time_axis[t])+","+yscale(mean_end - e.info_start.std_inf[iplot])+" ";
        points += xscale(g.time_axis[t])+","+yscale(mean_start - d.info.std_inf[iplot])+" ";
        points += xscale(g.time_axis[t])+","+yscale(mean_start + d.info.std_sup[iplot])+" ";
    }
    return points;
}

function f_color_vertex(d, g, colors) {
    return colors[d.info.brotherhood_size[0]];
}

function f_color_edge(d, g, colors) {
    return colors[g.vertices[d.time_step][d.v_start].info.brotherhood_size[0]];
}

function f_radius(d) {
    return (4*d.ratio_members)
}

function f_stroke_width(d) {
    return (4*d.ratio_members)
}


export async function draw_meteogram(
    filename,
    {include_k = "blank", kmax = 4, id="fig", dims = dimensions()} = {},
) {
    // Load the data and wait until it is ready
    const data =  await d3.json(filename);
    // d3 expects a very specific data format
    let data_xy = d3fy(data);
    // where we will store all our figs
    let figs = [];

    // We create a new fig for each variable
    for(var iplot = 0; iplot < data.var_names.length; iplot++ ) {

        let figElem = draw_fig(dims, id + "_" + iplot);
        let myPlot = d3.select(figElem).select("#plot-group");
        let interactiveGroupElem = document.getElementById(figElem.id + "_input");

        // Add x and y axis element
        let {x, y, xk, yk} = add_axes(
            figElem, data.time, data.members,
            {include_k:include_k, iplot:iplot}
        );

        // Add titles and labels  and style ticks
        setFigTitle(figElem, "");
        setAxTitle(figElem, "");
        setXLabel(figElem, "Time (h)");
        setYLabel(
            figElem, data.long_name[iplot] +" (" + data.units[iplot] + ")"
        );
        style_ticks(figElem);

        const myLine = d3.line()
            .x(d => x(d.t))
            .y(d => y(d[data.var_names[iplot]]));

        // This element will render the lines
        myPlot.append('g')
            .attr('id', 'members')
            .selectAll('.line')
            .data(data_xy)
            .enter()
            .append("path")  // "path" is the svg element for lines
            .classed("line", true)        // Style
            .on("mouseover", onMouseOverMember(interactiveGroupElem)) // Add listener for mouseover event
            .on("mouseout", onMouseOutMember(interactiveGroupElem))   // Add listener for mouseout event
            .attr("d", (d => myLine(d)))  // How to compute x and y
            .attr("id", ((d, i) => "m" + i));   // Member's id (for selection)

        figs.push(figElem);
    }
    return figs
}



export async function draw_mjo(
    filename,
    {id="fig", dims = dimensions()} = {},
) {

    let figElem = draw_fig(dims, id);
    let myPlot = d3.select(figElem).select("#plot-group");
    let interactiveGroupElem = document.getElementById(figElem.id + "_input");
    let vmax = 5;

    // x y scales and their range <-> domain
    var x = d3.scaleLinear().range([0, dims.plot.width]),
    y = d3.scaleLinear().range([dims.plot.height, 0]);

    x.domain([-vmax, vmax]);
    y.domain([-vmax, vmax]);

    // Load the data and wait until it is ready
    const data =  await d3.json(filename);
    let data_xy = d3fy(data);

    // This element will render the xAxis with the xLabel
    myPlot.select('#xaxis')
        .call(d3.axisBottom(x).tickSizeOuter(0));

    myPlot.select('#yaxis')
        .call(d3.axisLeft(y).tickSizeOuter(0));

    // Add titles and labels and style ticks
    setFigTitle(figElem, "");
    setAxTitle(figElem, "");
    setXLabel(figElem, "RMM1");
    setYLabel(figElem, "RMM2");
    style_ticks(figElem);

    const myLine = d3.line()
        .x(d => x(d.rmm1))
        .y(d => y(d.rmm2));

    // This element will render the lines
    myPlot.append('g')
        .attr('id', 'members')
        .selectAll('.line')
        .data(data_xy)
        .enter()
        .append("path")  // "path" is the svg element for lines
        .classed("line", true)
        .on("mouseover", onMouseOverMember(interactiveGroupElem))
        .on("mouseout", onMouseOutMember(interactiveGroupElem))
        .attr("d", (d => myLine(d)))
        .attr("id", ((d, i) => "m" + i));
    // Add mjo classes lines
    draw_mjo_classes(figElem, x, y, vmax=vmax);
    return figElem
}


export async function draw_entire_graph_meteogram(
    filename_data,
    filename_graph,
    {include_k = "yes", kmax = 4, id="fig", dims = dimensions()} = {},
) {
    // Load the graph and wait until it is ready
    const g =  await d3.json(filename_graph);
    const vertices = g.vertices.flat();
    const edges = g.edges.flat();
    const time = g.time_axis;
    const members = g.members;
    const colors = get_list_colors(g.n_clusters_range.length);

    const data =  await d3.json(filename_data);

    // where we will store all our figs
    let figs = [];

    // We create a new fig for each variable
    for(var iplot = 0; iplot < g.d; iplot++ ) {

        let figElem = draw_fig(dims, id + "_" + iplot);
        let interactiveGroupElem = document.getElementById(figElem.id + "_input");
        let myPlot = d3.select(figElem).select("#plot-group");

        // Add x and y axis element
        let {x, y, xk, yk} = add_axes(
            figElem, data.time, data.members,
            {include_k : include_k, iplot : iplot}
        );

        // Add titles and labels  and style ticks
        setFigTitle(figElem, " ");
        setAxTitle(figElem, "");
        setXLabel(figElem, "Time (h)");
        setYLabel(
            figElem, data.long_name[iplot] +" (" + data.units[iplot] + ")"
        );
        style_ticks(figElem);

        const edge_fn = d3.line()
            .x(d => x( g.time_axis(d.time_step) ))
            .y(d => y( d.info.mean[iplot] ));

        // This element will render the vertices
        myPlot.append('g')
            .attr('id', 'vertices')
            .selectAll('.vertex')
            .data(vertices)
            .enter()
            .append("circle")
            .classed("vertex", true)
            .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("cx", (d => x( g.time_axis[d.time_step] )))
            .attr("cy", (d => y( d.info.mean[iplot] )))
            .attr("r", (d => f_radius(d)) )
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)))
            .attr("fill", (d => colors[d.info.brotherhood_size[0]]))
            .attr("id", (d => "v" + d.key) );

        // This element will render the standard deviation of edges
        // myPlot.append('g')
        //     .attr('id', 'edges-std')
        //     .selectAll('.edge-std')
        //     .data(edges)
        //     .enter()
        //     .append("polygon")
        //     .classed("edge-std", true)
        //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        //     .attr("points", (d => f_polygon_edge(d, g, x, y, iplot)))
        //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
        //     .attr("fill", (d => f_color_edge(d, g, colors)))
        //     .attr("id", (d => "e-std" + d.key) );

        // // This element will render the standard deviation of vertices
        // myPlot.append('g')
        //     .attr('id', 'vertices-std')
        //     .selectAll('.vertex-std')
        //     .data(vertices)
        //     .enter()
        //     .append("polygon")
        //     .classed("vertex-std", true)
        //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        //     .attr("points", (d => f_polygon_vertex(d, g, x, y, iplot)))
        //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
        //     .attr("fill", (d => f_color_vertex(d, g, colors)))
        //     .attr("id", (d => "v-std" + d.key) );

        // This element will render the edges
        myPlot.append('g')
            .attr('id', 'edges')
            .selectAll('.edges')
            .data(edges)
            .enter()
            .append("polyline")
            .classed("edge", true)
            // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("points", (d => f_line_edge(d, g, x, y, iplot)))
            .attr("marker-start",(d => "url(graph.svg#dot) markerWidth="+f_radius(d)) )
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max) ))
            .attr("stroke", (d => f_color_edge(d, g, colors)))
            .attr("stroke-width", (d => f_stroke_width(d)))
            .attr("id", (d => "e" + d.key) );

        figs.push(figElem);
    }
    return figs
}



export async function draw_relevant_graph_meteogram(
    filename_data,
    filename_graph,
    {include_k = "yes", kmax = 4, id="fig", dims = dimensions()} = {},
) {
    // Load the graph and wait until it is ready
    const g =  await d3.json(filename_graph);
    const vertices = g.vertices.flat();
    const edges = g.edges.flat();
    const time = g.time_axis;
    const members = g.members;
    const colors = get_list_colors(g.n_clusters_range.length);

    const data =  await d3.json(filename_data);

    // where we will store all our figs
    let figs = [];

    // We create a new fig for each variable
    for(var iplot = 0; iplot < g.d; iplot++ ) {

        let figElem = draw_fig(dims, id + "_" + iplot);
        let interactiveGroupElem = document.getElementById(figElem.id + "_relevant");
        let myPlot = d3.select(figElem).select("#plot-group");

        // Add x and y axis element
        let {x, y, xk, yk} = add_axes(
            figElem, data.time, data.members,
            {include_k : include_k, kmax : kmax, iplot : iplot}
        );

        // Add titles and labels  and style ticks
        setFigTitle(figElem, " ");
        setAxTitle(figElem, "");
        setXLabel(figElem, "Time (h)");
        setYLabel(
            figElem, data.long_name[iplot] +" (" + data.units[iplot] + ")"
        );
        style_ticks(figElem);

        const edge_fn = d3.line()
            .x(d => x( g.time_axis(d.time_step) ))
            .y(d => y( d.info.mean[iplot] ));

        // This element will render the vertices
        myPlot.append('g')
            .attr('id', 'vertices')
            .selectAll('.vertex')
            .data(vertices)
            .enter()
            .append("circle")
            .classed("vertex", true)
            .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("cx", (d => x( g.time_axis[d.time_step] )))
            .attr("cy", (d => y( d.info.mean[iplot] )))
            .attr("r", (d => f_radius(d)) )
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)))
            .attr("fill", (d => colors[d.info.brotherhood_size[0]]))
            .attr("id", (d => "v" + d.key) );

        // This element will render the standard deviation of edges
        // myPlot.append('g')
        //     .attr('id', 'edges-std')
        //     .selectAll('.edge-std')
        //     .data(edges)
        //     .enter()
        //     .append("polygon")
        //     .classed("edge-std", true)
        //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        //     .attr("points", (d => f_polygon_edge(d, g, x, y, iplot)))
        //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
        //     .attr("fill", (d => f_color_edge(d, g, colors)))
        //     .attr("id", (d => "e-std" + d.key) );

        // // This element will render the standard deviation of vertices
        // myPlot.append('g')
        //     .attr('id', 'vertices-std')
        //     .selectAll('.vertex-std')
        //     .data(vertices)
        //     .enter()
        //     .append("polygon")
        //     .classed("vertex-std", true)
        //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        //     .attr("points", (d => f_polygon_vertex(d, g, x, y, iplot)))
        //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
        //     .attr("fill", (d => f_color_vertex(d, g, colors)))
        //     .attr("id", (d => "v-std" + d.key) );

        // This element will render the edges
        myPlot.append('g')
            .attr('id', 'edges')
            .selectAll('.edges')
            .data(edges)
            .enter()
            .append("polyline")
            .classed("edge", true)
            // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("points", (d => f_line_edge(d, g, x, y, iplot)))
            .attr("marker-start",(d => "url(graph.svg#dot) markerWidth="+f_radius(d)) )
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max) ))
            .attr("stroke", (d => f_color_edge(d, g, colors)))
            .attr("stroke-width", (d => f_stroke_width(d)))
            .attr("id", (d => "e" + d.key) );

        figs.push(figElem);
    }
    return figs
}


export async function draw_entire_graph_mjo(
    filename_data,
    filename_graph,
    {id="fig", dims = dimensions()} = {},
) {
    // Load the graph and wait until it is ready
    const g =  await d3.json(filename_graph);
    const vertices = g.vertices.flat();
    const edges = g.edges.flat();
    const time = g.time_axis;
    const members = g.members;
    const colors = get_list_colors(g.n_clusters_range.length);

    const data =  await d3.json(filename_data);

    const vmax = 5;

    let figElem = draw_fig(dims, id + "_mjo");
    let interactiveGroupElem = document.getElementById(figElem.id + "_input");
    let myPlot = d3.select(figElem).select("#plot-group");

    // Reminder:
    // - Range: output range that input values to map to
    // - scaleLinear: Continuous domain mapped to continuous output range
    let x = d3.scaleLinear().range([0, dims.plot.width]),
        y = d3.scaleLinear().range([dims.plot.height, 0]);

    // Reminder: domain = min/max values of input data
    x.domain([ -vmax, vmax ]);
    y.domain([ -vmax, vmax ]);

    // This element will render the xAxis with the xLabel
    myPlot.select('#xaxis')
        // Create many sub-groups for the xAxis
        .call(d3.axisBottom(x).tickSizeOuter(0));

    myPlot.select('#yaxis')
        // Create many sub-groups for the yAxis
        .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat(d => d));

    // Add titles and labels  and style ticks
    setFigTitle(figElem, " ");
    setAxTitle(figElem, "");
    setXLabel(figElem, "RMM1");
    setYLabel(figElem, "RMM2");
    style_ticks(figElem);

    // This element will render the vertices
    myPlot.append('g')
        .attr('id', 'vertices')
        .selectAll('.vertex')
        .data(vertices)
        .enter()
        .append("circle")
        .classed("vertex", true)
        .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        .attr("cx", (d => x( d.info.mean[0] )))
        .attr("cy", (d => y( d.info.mean[1] )))
        .attr("r", (d =>  f_radius(d)) )
        .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)))
        .attr("fill", (d => colors[d.info.brotherhood_size[0]]))
        .attr("id", (d => "v" + d.key) );

    // This element will render the standard deviation of edges
    // myPlot.append('g')
    //     .attr('id', 'edges-std')
    //     .selectAll('.edge-std')
    //     .data(edges)
    //     .enter()
    //     .append("polygon")
    //     .classed("edge-std", true)
    //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
    //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
    //     .attr("points", (d => f_polygon_edge(d, g, x, y, iplot)))
    //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
    //     .attr("fill", (d => f_color_edge(d, g, colors)))
    //     .attr("id", (d => "e-std" + d.key) );

    // // This element will render the standard deviation of vertices
    // myPlot.append('g')
    //     .attr('id', 'vertices-std')
    //     .selectAll('.vertex-std')
    //     .data(vertices)
    //     .enter()
    //     .append("polygon")
    //     .classed("vertex-std", true)
    //     // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
    //     // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
    //     .attr("points", (d => f_polygon_vertex(d, g, x, y, iplot)))
    //     .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
    //     .attr("fill", (d => f_color_vertex(d, g, colors)))
    //     .attr("id", (d => "v-std" + d.key) );

    // This element will render the edges
    myPlot.append('g')
        .attr('id', 'edges')
        .selectAll('.edges')
        .data(edges)
        .enter()
        .append("polyline")
        .classed("edge", true)
        // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
        // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
        .attr("points", (d => f_line_edge_mjo(d, g, x, y)))
        .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max) ))
        .attr("stroke", (d => f_color_edge(d, g, colors)))
        .attr("stroke-width", (d => f_stroke_width(d)))
        .attr("id", (d => "e" + d.key) );

    // Add mjo classes lines
    draw_mjo_classes(figElem, x, y, vmax);

    return figElem
}



export async function life_span_plot(
    filename_graph,
    {id="fig", dims = dimensions()} = {},
) {
    // Load the graph and wait until it is ready
    const g =  await d3.json(filename_graph);
    const life_spans = d3fy_life_span(g.life_span);
    const colors = get_list_colors(g.n_clusters_range.length);

    let figElem = draw_fig(dims, id);
    let myPlot = d3.select(figElem).select("#plot-group");

    let x = d3.scaleLinear().range([0, dims.plot.width]),
        y = d3.scaleLinear().range([dims.plot.height, 0]);

    x.domain([ d3.min(g.time_axis), d3.max(g.time_axis) ] );
    y.domain([0, 1]);

    myPlot.select('#xaxis')
        .call(d3.axisBottom(x).tickSizeOuter(0));

    myPlot.select('#yaxis')
        .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat(d => d));

    // Add titles and labels  and style ticks
    setFigTitle(figElem, " ");
    setAxTitle(figElem, "");
    setXLabel(figElem, "Time (h)");
    setYLabel(figElem, "Life span");
    style_ticks(figElem);

    const myLine = d3.line()
        .x(d => x(g.time_axis[d.t]))
        .y(d => y(d.life_span));

    // This element will render the life span
    myPlot.append('g')
        .attr('id', 'life-spans')
        .selectAll('.life-span')
        .data(life_spans)
        .enter()
        .append("path")
        .classed("life-span", true)
        .attr("d", (d => myLine(d)))
        .attr("stroke", (d => colors[d[0].k]))
        .attr("id", (d => "k" + d[0].k) );

    return figElem
}

//mouseover event handler function using closure
function onMouseOverMember(interactiveGroupElem, e, d) {
    return function (e, d) {
        onMouseMemberAux(e, d, this, interactiveGroupElem, 'lineSelected')
    }
}

//mouseout event handler function using closure
function onMouseOutMember(interactiveGroupElem, e, d) {
    return function (e, d) {
        onMouseMemberAux(e, d, this, interactiveGroupElem, 'line')
    }
}

//mouseover event handler function using closure
function onMouseOverCluster(interactiveGroupElem, e, d) {
    return function (e, d) {
        onMouseClusterAux(
            e, d, this, interactiveGroupElem,
            "vertexSelected", "lineSelectedbyCluster")
    }
}

//mouseout event handler function using closure
function onMouseOutCluster(interactiveGroupElem, e, d) {
    return function (e, d) {
        onMouseClusterAux(
            e, d, this, interactiveGroupElem,
            "vertex", "line")
    }
}