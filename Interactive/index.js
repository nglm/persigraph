import { dimensions, setAxTitle } from "./figures.js";

import {
    draw_meteogram, draw_mjo, draw_entire_graph_mjo,
    draw_entire_graph_meteogram, life_span_plot
} from "./plots.js";
const data_path = "./data/";
const data_graph = "./graphs/";
const f1 = "ec.ens.2020011400.sfc.meteogram.json";
const f2 = "ec.ens.2020011500.sfc.meteogram.json";
const f3 = "ec.ens.2020011600.sfc.meteogram.json";
const f4 = "z_s2s_rmm_ecmf_prod_rt_2015030500.json";
const f4_polar = "z_s2s_rmm_ecmf_prod_rt_2015030500_polar.json";
const f5 = "z_s2s_rmm_ecmf_prod_rt_2020120300";

const dims_meteogram = dimensions({plotWidth : 800, plotHeight : 400});
const dims_mjo = dimensions({plotWidth : 600, plotHeight : 600});

// await draw_meteogram(data_path + f1, undefined, "fig01");
// await draw_meteogram(data_path + f2, undefined, "fig02");
// await draw_meteogram(data_path + f3, undefined, "fig03");


let mjo = await draw_mjo(data_path + f4, dims_mjo, "mjo");
setAxTitle(mjo, f4);
let mjo_graph = await draw_entire_graph_mjo(data_path + f4, data_graph + f4, dims_mjo, "mjo_graph");
let life_span = await life_span_plot(data_graph + f4, dims_mjo, "life_span");
d3.select("body").append('text').html('<br>');
let mjo_rmm = await draw_meteogram(data_path + f4, dims_meteogram, "mjo_rmm");
let mjo_rmm_graph = await draw_entire_graph_meteogram(data_path + f4, data_graph + f4, dims_meteogram, "mjo_rmm_graph");
let mjo_polar = await draw_meteogram(data_path + f4_polar, dims_meteogram, "mjo_polar");


// await draw_meteogram(data_path + f4, undefined, "fig01");
// await draw_mjo(data_path + f4, dims_mjo, "fig04");