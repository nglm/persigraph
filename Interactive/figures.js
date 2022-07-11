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

export const DIMS = dimensions();

export function setInnerHTMLById(elem, id, text) {
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

export function draw_mjo_classes(figElem, x, y, vmax=5) {

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


export function add_axes(
    figElem, xvalues, yvalues, include_k=true, kmax=4, iplot=0
) {
    let myPlot = d3.select(figElem).select("#plot-group");

    const plotWidth = d3.select(figElem)
        .select("#plot")
        .select("#background")
        .attr("width");

    const plotHeight = d3.select(figElem)
        .select("#plot")
        .select("#background")
        .attr("height");

    // Find extremum to set axes limits
    const ymin = d3.min(yvalues, (d => d3.min(d[iplot])) ),
        ymax = d3.max(yvalues, (d => d3.max(d[iplot])) );

    // Default value if not include_k
    let yMinDomain = ymin;
    const plotHeightK = plotHeight / 5;

    const date_min = d3.min(xvalues),
        date_max = d3.max(xvalues);

    // Reminder:
    // - Range: output range that input values to map to
    // - scaleLinear: Continuous domain mapped to continuous output range
    let x = d3.scaleLinear().range([0, plotWidth]),
        y = d3.scaleLinear().range([plotHeight, 0]);

    let xk, yk;

    if (include_k === true) {
        // Additional scales for relevant k
        yk = d3.scaleLinear().range([plotHeightK, 0]);
        xk = d3.scaleBand().range([0, plotWidth])
            .padding(0.2);

        // Extra space for relevant k
        yMinDomain = ymin-Math.abs(ymax-ymin)/8;

        // Reminder: domain = min/max values of input data
        xk.domain([ date_min, date_max ] );
        yk.domain([0.5, kmax]);
    }


    // Reminder: domain = min/max values of input data
    x.domain([ date_min, date_max ] );
    y.domain([yMinDomain, ymax]);

    myPlot.select('#xaxis')
        // Create many sub-groups for the xAxis
        .call(d3.axisBottom(x).tickSizeOuter(0));

    myPlot.select('#yaxis')
        // Create many sub-groups for the yAxis
        .call(d3.axisLeft(y).tickSizeOuter(0).tickFormat(d => d));

    if (include_k === true) {
        myPlot.append("g")
            .attr('id', 'xaxis-k')
            .attr("transform", "translate(0, " + plotHeight + ")");

        myPlot.select('#xaxis-k')
            .call(d3.axisBottom(xk).tickValues([]));

        myPlot.append("g")
            .attr('id', 'yaxis-k')
            .attr(
                "transform",
                "translate("+(plotWidth - 12)+ ", "+ (plotHeight-plotHeightK) +")");

        myPlot.select('#yaxis-k')
            .call(d3.axisLeft(yk)
                //.ticks(kmax)
                .tickValues(d3.range(kmax, 0, -1))
                .tickFormat(d3.format("d")));
    }

    return {x, y, xk, yk}
}