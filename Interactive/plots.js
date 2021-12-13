import { d3fy } from "./preprocess.js";

export function dimensions({
    figWidth=800,
    figHeight=600,
    figMarginTop=5,
    figMarginLeft=5,
    figMarginRight=5,
    figMarginBottom=5,
    labelsX=130,
    labelsY=110,
    labelsAxes=50,
    labelsFig=50,
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

export function draw_fig(dims = DIMS, fig_id = 'fig') {


    // Append 'svg' DOM element at the end of the document to contain our fig
    d3.select("body")
        .append('svg')
        .attr('id', fig_id)
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height)
        .append('rect')                // And style it
        .attr("id", "background")
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height)
        .classed("fig", true);

    // Create our fig group
    let myFig = d3.select("body")
        .select('#'+fig_id)
        .append('g')
        .attr("id", 'fig-group')
        .attr(
            "transform",
            "translate(" + dims.fig.margin.left +","+ dims.fig.margin.top + ")"
        );

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
        .attr('width', dims.fig.width)
        .attr('height', dims.fig.height - dims.labels.fig);

    // create a new SVG and a group inside our fig to contain axes
    // Note that SVG groups 'g' can not be styled
    let myAxes = myFig.select('#main')
        .append("svg")
        .attr("id", 'axes')
        .attr('width', dims.axes.width)
        .attr('height', dims.axes.height - dims.labels.axes)
        .classed('axes', true)
        .append('g')
        .attr("id", 'axes-group')
        .attr(
            "transform",
            "translate(" + dims.axes.margin.left+","+dims.axes.margin.top + ")"
        );

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
        .attr('width', dims.axes.width)
        .attr('height', dims.axes.height - dims.labels.axes);

    // xlabel
    myAxes.select('#main')
        .append('text')
        .attr('id', 'xlabel')
        .attr("x", dims.labels.y + dims.plot.width/2)
        .attr("y", dims.plot.height + dims.labels.x/2)
        .classed("xlabel", true);

    // ylabel
    myAxes.select('#main')
        .append('text')
        .attr('id', 'ylabel')
        .attr("x", -dims.plot.height/2)
        .attr("y", dims.labels.y/2)
        .classed("ylabel", true);

    // Add an svg element for our plot
    // Add a rect element to the group in order to style chart
    myAxes.select('#main')
        .append('svg')
        .attr("id", 'plot')
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

    return {myFig, myAxes, myPlot}
}

export function style_ticks(myPlot) {
    myPlot.select('#yaxis')
        .selectAll('.tick')       // Select all ticks
        .selectAll('text')        // Select the text associated with each tick
        .classed("ytext", true);  // Style the text

    myPlot.select('#xaxis')
        .selectAll(".tick")      // Select all ticks,
        .selectAll("text")       // Select the text associated with each tick
        .classed("xtext", true);  // Style the text

    return myPlot
}

export async function draw_meteogram(
    filename,
    dims = dimensions(),
    fig_id="fig",
) {
    // "async" so that we can call 'await' inside and therefore use the data

    // Load the data and wait until it is ready
    const myData =  await d3.json(filename);
    let data_xy = d3fy(myData);

    let figs = [];

    for(var iplot = 0; iplot < myData.var_names.length; iplot++ ) {

        const ymin = d3.min(myData.members, (d => d3.min(d[iplot])) ),
            ymax = d3.max(myData.members, (d => d3.max(d[iplot])) );

        let meteogram = draw_fig(dims, fig_id + "_" + iplot);

        let myFig = meteogram.myFig;
        let myAxes = meteogram.myAxes;
        let myPlot = meteogram.myPlot;

        // Reminder:
        // - domain: min/max values of input data
        // - Range: output range that input values to map to.
        // - scaleOrdinal from discrete and map to discrete numeric output range.
        // - scaleBand like scaleOrdinal except the output range is continuous and
        // numeric (so from discrete to continuous)
        // - scaleLinear: Continuous domain mapped to continuous output range
        let x = d3.scaleLinear().range([0, dims.plot.width]),
        y = d3.scaleLinear().range([dims.plot.height, 0]);



        // Now we can specify the domain of our scales
        x.domain([ d3.min(myData.time, (d => d)), d3.max(myData.time, (d => d)) ]);
        y.domain([ymin, ymax]);

        // Add the title of the figure
        myFig.select('#figtitle')
            .text("Figure");

        // Add the title of the axes
        myAxes.select('#axtitle')
            .text(myData.filename);

        // This element will render the xAxis with the xLabel
        myPlot.select('#xaxis')
            // The next line will create many sub-groups for the xAxis
            //
            // D3’s "call(myF)" takes a selection as input and hands that
            // selection off to any function myF.
            // So selection.call(myF) is equiv to myF(selection)
            //
            // d3.axisBottom(scale) constructs a x-axis generator for the given
            // scale, with empty tick arguments, a tick size of 6 and padding of 3.
            // In this orientation, ticks are drawn below the line.
            // Note that this generator has to be called (using .call) by a
            // svg element in order to render the axis onto the HTML page
            .call(d3.axisBottom(x).tickSizeOuter(0));

        myAxes.select('#main')
            .select('#xlabel')
            .text("Time (h)");

        myPlot.select('#yaxis')
            // The next line will create many sub-groups for the yAxis
            // We can specify the tickFormat with '.tickFormat()'. Otherwise
            // raw range values are written (see xAxis above)
            .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat(d => d));

        myPlot = style_ticks(myPlot);

        myAxes.select('#main')
            .select("#ylabel")
            .text(myData.long_name[iplot] +" (" + myData.units[iplot] + ")");

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
            .classed("line", true)
            // .on("mouseover", onMouseOver) //Add listener for the mouseover event
            // .on("mouseout", onMouseOut)   //Add listener for the mouseout event
            .attr("d", (d => myLine(d)))
            .attr("id", ((d, i) => i));
            // .attr("width", x.bandwidth())
            // .transition()
            // .ease(d3.easeLinear)
            // .duration(400)
            // .delay((d, i) => (i * 50))
            // .attr("height", d => (myHeight - y(d.value)));

        figs.push({myFig, myAxes, myPlot});
    }


    return figs
}

function draw_mjo_classes(myPlot, x, y, vmax=5) {
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

    return myPlot
}

export async function draw_mjo(
    filename,
    dims = dimensions(),
    fig_id="fig",
) {

    let mjo = draw_fig(dims, fig_id);

    let myFig = mjo.myFig;
    let myAxes = mjo.myAxes;
    let myPlot = mjo.myPlot;
    let vmax = 5;

    var x = d3.scaleLinear().range([0, dims.plot.width]),
    y = d3.scaleLinear().range([dims.plot.height, 0]);

    // Load the data and wait until it is ready
    const myData =  await d3.json(filename);
    let data_xy = d3fy(myData);

    // Now we can specify the domain of our scales
    x.domain([-vmax, vmax]);
    y.domain([-vmax, vmax]);

    // Add the title of the figure
    myFig.select('#figtitle')
        .text("Figure");

    // Add the title of the axes
    myAxes.select('#axtitle')
        .text(myData.filename);

    // This element will render the xAxis with the xLabel
    myPlot.select('#xaxis')
        .call(d3.axisBottom(x).tickSizeOuter(0));

    myAxes.select('#main')
        .select('#xlabel')
        .text("RMM1");

    myPlot.select('#yaxis')
        .call(d3.axisLeft(y).tickSizeOuter(0));

    myPlot = style_ticks(myPlot);

    myAxes.select('#main')
        .select("#ylabel")
        .text('RMM2');

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
        // .on("mouseover", onMouseOver) //Add listener for the mouseover event
        // .on("mouseout", onMouseOut)   //Add listener for the mouseout event
        .attr("d", (d => myLine(d)))
        .attr("id", ((d, i) => i));
        // .attr("width", x.bandwidth())
        // .transition()
        // .ease(d3.easeLinear)
        // .duration(400)
        // .delay((d, i) => (i * 50))
        // .attr("height", d => (myHeight - y(d.value)));

    // Add mjo classes lines
    myPlot = draw_mjo_classes(myPlot, x, y, vmax=vmax);

    return {myFig, myAxes, myPlot}
}

//mouseover event handler function
function onMouseOver(e, d) {

    // The selection.transition method now takes an optional transition
    // instance which can be used to synchronize a new transition with an
    // existing transition
    //
    // In this callback function we will use this optional parameter
    // and in the next callback we will chain all the functions.
    //
    // .ease() control apparent motion in animation
    // .ease(d3.easeLinear) is actually the default configuration so
    // we could have omitted it (we will in the next callback function)
    const tr = d3.transition().duration(400).ease(d3.easeLinear);

    // Change the attributes of the DOM element ('this') associated with
    // the event fired
    // Note that the style itself has already been configured in
    // the css subclass '.bar:hover'
    // We could have changed to a completely different class also
    d3.select(this).transition(tr)
        .attr("width", (x.bandwidth() + 5))
        .attr("height", d => (myHeight - y(d.value) + 10))
        .attr("y", d => (y(d.value) - 10) );

    // Show the value corresponding to this bar
    myAxes.append("text")
        .attr('class', 'val')
        .attr('x', () => x(d.year))
        .attr('y', () => (y(d.value) - 15))
        .text(() => ('$' +d.value) );
}

//mouseout event handler function
function onMouseOut(e, d) {

    // Note that the style itself has already been configured in
    // the css subclass '.bar:hover'
    d3.select(this)
        .transition()   // initiate a transition
        .duration(400)  // specifies the duration of the transition
        .attr('width', x.bandwidth())
        .attr("y", d => y(d.value))
        .attr("height", d => (axes.height - y(d.value)));

    // removes the text value we had added during the bar selection
    d3.selectAll('.val')
        .remove()
}

