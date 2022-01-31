
import { d3fy } from "./preprocess.js";
import { range_rescale, sigmoid, linear } from "./utils.js";

export const COLOR_BREWER = [
    "#636363", // Grey
    "#377eb8", // Blue
    "#a65628", // Brown
    "#984ea3", // Purple
    "#e41a1c", // Red
    "#4daf4a", // Green
    "#ff7f00", // Orange
    "#f781bf", // Pink
    "#ffff33", // Yellow
];

export function get_list_colors(N){
    let i = 0;
    let colors = [];
    while (i < N) {
        for (var j = 0; j < COLOR_BREWER.length; j++ ){
            colors.push(COLOR_BREWER[j]);
            i++;
        }
    }
    return colors
}

export function dimensions({
    figWidth=1200,
    figHeight=600,
    figMarginTop=5,
    figMarginLeft=5,
    figMarginRight=5,
    figMarginBottom=5,
    labelsX=70,
    labelsY=80,
    labelsAxes=50,
    labelsFig=0,
    axesMarginTop=5,
    axesMarginLeft=5,
    axesMarginRight=5,
    axesMarginBottom=5,
    plotWidth=undefined,
    plotHeight=undefined,
    plotMarginTop=5,
    plotMarginLeft=5,
    plotMarginRight=5,
    plotMarginBottom=5,
} = {}) {

    let fig, labels, axes, plot;
    labels = { x: labelsX, y: labelsY, axes: labelsAxes, fig: labelsFig };

    // Compute dimensions based on fig size
    if (plotWidth === undefined) {
        fig = {
            width: figWidth, height: figHeight,
            margin: {
                top: figMarginTop, left: figMarginLeft,
                right: figMarginRight, bottom: figMarginBottom
            }
        };

        axes = {
            width: fig.width - (fig.margin.right + fig.margin.left),
            height: fig.height - (fig.margin.top + fig.margin.bottom + labels.fig),
            margin: {
                top: axesMarginTop, left: axesMarginLeft,
                right: axesMarginRight, bottom: axesMarginBottom
            },
        };

        plot = {
            width: axes.width - (axes.margin.right + axes.margin.left + labels.y),
            height: axes.height - (axes.margin.top + axes.margin.bottom + labels.axes + labels.x),
            margin: {
                top: plotMarginTop, left: plotMarginLeft,
                right: plotMarginRight, bottom: plotMarginBottom
            },
        };
    }
    // Compute dimensions based on plot size
    else {

        plot = {
            width: plotWidth,
            height: plotHeight,
            margin: {
                top: plotMarginTop, left: plotMarginLeft,
                right: plotMarginRight, bottom: plotMarginBottom
            },
        };

        axes = {
            width: plot.width + (plot.margin.right + plot.margin.left + labels.y),
            height: plot.height + (plot.margin.top + plot.margin.bottom + labels.x + labels.axes),
            margin: {
                top: axesMarginTop, left: axesMarginLeft,
                right: axesMarginRight, bottom: axesMarginBottom
            },
        };

        fig = {
            width: axes.width + (axes.margin.right + axes.margin.left),
            height: axes.height + (axes.margin.top + axes.margin.bottom + labels.fig),
            margin: {
                top: figMarginTop, left: figMarginLeft,
                right: figMarginRight, bottom: figMarginBottom
            }
        };
    }
    return {fig, labels, axes, plot}
}

const DIMS = dimensions();

function setInnerHTMLById(elem, id, text) {
    let e = document.getElementById(elem.id+"_svg")
    e.getElementById(id).innerHTML = text;
}

export function setFigTitle(figElem, text) {
    setInnerHTMLById(figElem, "figtitle", text);
}

export function setAxTitle(figElem, text) {
    setInnerHTMLById(figElem, "axtitle", text);
}
export function setXLabel(figElem, text) {
    setInnerHTMLById(figElem, "xlabel", text);
}

export function setYLabel(figElem, text) {
    setInnerHTMLById(figElem, "ylabel", text);
}

function f_life_span(life_span, vmax, {vmin=0} = {}) {
    let rescaled = range_rescale(life_span, {x0:vmin, x1:vmax});
    return sigmoid(rescaled, {range0_1:true});
}

function f_polygon(d, g, xscale, yscale, iplot) {
    let offset = 0.1*xscale(g.time_axis[1]);
    let t = d.time_step;
    let v_start = g.vertices[t][d.v_start];
    let v_end = g.vertices[t + 1][d.v_end];
    let mean_start = v_start.info.mean[iplot];
    let mean_end = v_end.info.mean[iplot];
    let points = offset + xscale(g.time_axis[t])+","+yscale(mean_start - v_start.info.std_inf[iplot])+" ";
    points += offset + xscale(g.time_axis[t])+","+yscale(mean_start + v_start.info.std_sup[iplot])+" ";
    points += -offset + xscale(g.time_axis[t+1])+","+yscale(mean_end + v_end.info.std_sup[iplot])+" ";
    points += -offset + xscale(g.time_axis[t+1])+","+yscale(mean_end - v_end.info.std_inf[iplot])+" ";
    return points;
}

function f_color(d, g, colors) {
    return colors[g.vertices[d.time_step][d.v_start].info.brotherhood_size[0]];
}

export function draw_fig(dims = DIMS, fig_id = 'fig') {
    /* -------------- fig architecture --------------
    fig01 (div)
    -- buttons/input
    -- svg
    ---- background
    ---- fig-group
    ------ figtitle
    ------ main
    -------- axes

    axes
    -- axes-group
    ---- axtitle
    ---- main
    ------ xlabel
    ------ ylabel
    ------ plot

    plot
    -- background
    -- plot-group
    ---- xaxis
    ---- yaxis
    ---- members
    ---- mjoClasses
    ------------------------------------------------
    Other comments:
    -- only 'document' and 'svg' elements have a '.getElementById() method
    -- Therefore ids must be unique in the entire document or within a svg element
    ------------------------------------------------*/


    // Append 'div'>'svg'>'rect' elements at the end of 'body' to contain our fig
    //
    d3.select("body")
        .append('div')
        .attr('id', fig_id)
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height)
        .classed('container-fig', true)
        .append('svg')
        .attr('id', fig_id + "_svg")
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height)
        .append('rect')                // And style it
        .attr("id", "background")
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height)
        .classed("fig", true);

    let figElem = document.getElementById(fig_id);


    // Button
    d3.select(figElem)
        .append('input')
        .attr('id', fig_id + "_input")
        .attr('type', 'number')
        .attr('value', 1)
        .attr("min", 1)
        .attr("max", 9)
        .classed("input-interactive-group", true)

    // Create our fig group
    let fig_group_width = dims.fig.width - dims.fig.margin.left - dims.fig.margin.right;
    let fig_group_height = dims.fig.height - dims.fig.margin.top - dims.fig.margin.bottom;
    let myFig = d3.select(figElem)
        .select('svg')
        .append('g')
        .attr("id", 'fig-group')
        .attr(
            "transform",
            "translate(" + dims.fig.margin.left +","+ dims.fig.margin.top + ")"
        )
        .attr('width', fig_group_width)
        .attr('height', fig_group_height);


    // Fig title
    myFig.append('text')
        .attr("id", "figtitle")
        // Use '50%' in combination with "text-anchor: middle" to center text
        .attr("x", "50%")  // percentage based on the parent container
        .attr("y", dims.labels.fig/2)
        .classed("figtitle", true);


    // Prepare the remaining part of the fig
    myFig.append('g')
        .attr('id', 'main')
        .attr(
            "transform",
            "translate(" + 0 + "," + dims.labels.fig + ")"
        )
        .attr('width', fig_group_width)
        .attr('height', fig_group_height - dims.labels.fig);

    // create a new SVG and a group inside our fig to contain axes
    // Note that SVG groups 'g' can not be styled
    let axes_group_width = dims.axes.width - dims.axes.margin.left - dims.axes.margin.right;
    let axes_group_height = dims.axes.height - dims.axes.margin.top -  dims.axes.margin.bottom;
    let myAxes = myFig.select('#main')
        .append("svg")
        .attr("id", 'axes')
        .attr('width', dims.axes.width)
        .attr('height', dims.axes.height)
        .classed('axes', true)
        .append('g')
        .attr("id", 'axes-group')
        .attr(
            "transform",
            "translate(" + dims.axes.margin.left+","+dims.axes.margin.top + ")"
        )
        .attr('width', axes_group_width)
        .attr('height', axes_group_height);

    // Axes title
    myAxes.append('text')
        .attr("id", "axtitle")
        // Use '50%' in combination with "text-anchor: middle" to center text
        .attr("x", "50%")  // percentage based on the parent container
        .attr("y", dims.labels.axes/2)
        .classed("axtitle", true);

    // Prepare the remaining part of the axes
    myAxes.append('g')
        .attr('id', 'main')
        .attr(
            "transform",
            "translate(" + 0 + "," + dims.labels.axes + ")"
        )
        .attr('width', axes_group_width)
        .attr('height', axes_group_height - dims.labels.axes);

    // xlabel
    myAxes.select('#main')
        .append('text')
        .attr('id', 'xlabel')
        .attr("x", dims.labels.y + dims.plot.width/2)
        .attr("y", dims.plot.height + 0.80*dims.labels.x )
        .classed("xlabel", true);

    // ylabel
    myAxes.select('#main')
        .append('text')
        .attr('id', 'ylabel')
        .attr("x", -dims.plot.height/2)
        .attr("y", 0.3*dims.labels.y)
        .classed("ylabel", true);

    // Add an svg element for our plot
    // Add a rect element to the group in order to style chart
    myAxes.select('#main')
        .append('svg')
        .attr("id", 'plot')
        .attr('width', axes_group_width)
        .attr('height', axes_group_height - dims.labels.axes)
        .append('rect')                // And style it
        .attr("id", "background")
        .attr('width', dims.plot.width)
        .attr('height', dims.plot.height)
        .attr(
            "transform",
            "translate(" + dims.labels.y + "," + 0 + ")"
        )
        .classed("plot", true);

    let myPlot = myAxes.select('#main')
        .select('#plot')
        .append('g')
        .attr("id", 'plot-group')
        .attr(
            "transform",
            "translate(" + dims.labels.y + "," + 0 + ")"
        );

    // This element will render the xAxis with the xLabel
    myPlot.append("g")
        .attr("id", "xaxis")
        // This transform moves (0,0) to the bottom left corner of the chart
        .attr("transform", "translate(0," + dims.plot.height + ")");

    // This element will render the yAxis with the yLabel
    myPlot.append("g")
        .attr('id', 'yaxis');

    return figElem
}

export function style_ticks(figElem) {

    d3.select(figElem)
        .select("#yaxis")
        .selectAll('.tick')       // Select all ticks
        .selectAll('text')        // Select the text associated with each tick
        .classed("ytext", true);  // Style the text

    d3.select(figElem)
        .select("#xaxis")
        .selectAll(".tick")       // Select all ticks,
        .selectAll("text")        // Select the text associated with each tick
        .classed("xtext", true);  // Style the text
}

export async function draw_meteogram(
    filename,
    dims = dimensions(),
    fig_id="fig",
) {
    // Load the data and wait until it is ready
    const myData =  await d3.json(filename);
    // d3 expects a very specific data format
    let data_xy = d3fy(myData);
    // where we will store all our figs
    let figs = [];

    // We create a new fig for each variable
    for(var iplot = 0; iplot < myData.var_names.length; iplot++ ) {

        // Find extremum to set axes limits
        const ymin = d3.min(myData.members, (d => d3.min(d[iplot])) ),
            ymax = d3.max(myData.members, (d => d3.max(d[iplot])) );

        let figElem = draw_fig(dims, fig_id + "_" + iplot);
        let myPlot = d3.select(figElem).select("#plot-group");
        let interactiveGroupElem = document.getElementById(figElem.id + "_input");

        // Reminder:
        // - Range: output range that input values to map to
        // - scaleLinear: Continuous domain mapped to continuous output range
        let x = d3.scaleLinear().range([0, dims.plot.width]),
            y = d3.scaleLinear().range([dims.plot.height, 0]);

        // Reminder: domain = min/max values of input data
        x.domain([ d3.min(myData.time), d3.max(myData.time) ] );
        y.domain([ymin, ymax]);

        // This element will render the xAxis with the xLabel
        myPlot.select('#xaxis')
            // Create many sub-groups for the xAxis
            .call(d3.axisBottom(x).tickSizeOuter(0));

        myPlot.select('#yaxis')
            // Create many sub-groups for the yAxis
            .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat(d => d));

        // Add titles and labels  and style ticks
        setFigTitle(figElem, "");
        setAxTitle(figElem, "");
        setXLabel(figElem, "Time (h)");
        setYLabel(
            figElem, myData.long_name[iplot] +" (" + myData.units[iplot] + ")"
        );
        style_ticks(figElem);

        const myLine = d3.line()
            .x(d => x(d.t))
            .y(d => y(d[myData.var_names[iplot]]));

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

function draw_mjo_classes(figElem, x, y, vmax=5) {

    // Draw classes and weak section
    const mjoClassLine = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y));

    let limit = Math.cos(Math.PI/4);

    const mjoClassCoord = [
        // bottom left - top right
        [ { x: -vmax, y: -vmax}, { x: -limit, y: -limit} ],
        [ { x: limit, y: limit}, { x: vmax, y: vmax} ],
        // vertical
        [ { x: 0, y: -vmax}, { x: 0, y: -1} ],
        [ { x: 0, y: 1}, { x: 0, y: vmax} ],
        // horizontal
        [ { x: -vmax, y: 0}, { x: -1, y: 0} ],
        [ { x: 1, y: 0}, { x: vmax, y: 0} ],
        // top left - bottom right
        [ { x: -vmax, y: vmax}, { x: -limit, y: limit} ],
        [ { x: limit, y: -limit}, { x: vmax, y: -vmax} ]
    ];

    let myPlot = d3.select(figElem).select("#plot-group");

    // Put all lines in one group
    myPlot.append('g')
        .attr('id', 'mjoClasses');

    myPlot.select('#mjoClasses')
        .selectAll('.mjoClass')
        .data(mjoClassCoord)
        .enter()
        .append("path")  // "path" is the svg element for lines
        .classed("mjoClass", true)
        .attr("d",  (d => mjoClassLine(d)));

    myPlot.select('#mjoClasses')
        .append("circle")
        .attr("cx", x(0))
        .attr("cy", y(0))
        .attr("r", (x(1)-x(0)))
        .classed("mjoClass", true)
}

export async function draw_mjo(
    filename,
    dims = dimensions(),
    fig_id="fig",
) {

    let figElem = draw_fig(dims, fig_id);
    let myPlot = d3.select(figElem).select("#plot-group");
    let interactiveGroupElem = document.getElementById(figElem.id + "_input");
    let vmax = 5;

    // x y scales and their range <-> domain
    var x = d3.scaleLinear().range([0, dims.plot.width]),
    y = d3.scaleLinear().range([dims.plot.height, 0]);

    x.domain([-vmax, vmax]);
    y.domain([-vmax, vmax]);

    // Load the data and wait until it is ready
    const myData =  await d3.json(filename);
    let data_xy = d3fy(myData);

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


export async function draw_entire_graph(
    filename_data,
    filename_graph,
    dims = dimensions(),
    fig_id="fig",
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

        // Find extremum to set axes limits
        const ymin = d3.min(g.members, (d => d3.min(d[iplot])) ),
            ymax = d3.max(g.members, (d => d3.max(d[iplot])) );

        let figElem = draw_fig(dims, fig_id + "_" + iplot);
        let interactiveGroupElem = document.getElementById(figElem.id + "_input");
        let myPlot = d3.select(figElem).select("#plot-group");

        // Reminder:
        // - Range: output range that input values to map to
        // - scaleLinear: Continuous domain mapped to continuous output range
        let x = d3.scaleLinear().range([0, dims.plot.width]),
            y = d3.scaleLinear().range([dims.plot.height, 0]);

        // Reminder: domain = min/max values of input data
        x.domain([ d3.min(g.time_axis), d3.max(g.time_axis) ] );
        y.domain([ymin, ymax]);

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
            .attr("r", (d => 10*d.ratio_members) )
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)))
            .attr("fill", (d => colors[d.info.brotherhood_size[0]]))
            .attr("id", (d => "v" + d.key) );

        // This element will render the standard deviation
        myPlot.append('g')
            .attr('id', 'edges')
            .selectAll('.edges')
            .data(edges)
            .enter()
            .append("polygon")
            .classed("edge", true)
            // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("points", (d => f_polygon(d, g, x, y, iplot)))
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/3 ))
            .attr("fill", (d => f_color(d, g, colors)))
            .attr("id", (d => "v" + d.key) );

        // This element will render the edges
        myPlot.append('g')
            .attr('id', 'edges')
            .selectAll('.edges')
            .data(edges)
            .enter()
            .append("polygon")
            .classed("edge", true)
            // .on("mouseover", onMouseOverCluster(interactiveGroupElem))
            // .on("mouseout", onMouseOutCluster(interactiveGroupElem))
            .attr("points", (d => f_polygon(d, g, x, y, iplot)))
            .attr("opacity", (d => f_life_span(d.life_span, g.life_span_max)/5 ))
            .attr("fill", (d => f_color(d, g, colors)))
            .attr("id", (d => "v" + d.key) );

        figs.push(figElem);
    }
    return figs
}

function onMouseMemberAux(e, d, memberElem, interactiveGroupElem, classname) {
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

function onMouseClusterAux(e, d, memberElem, interactiveGroupElem, classname1, classname2) {
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