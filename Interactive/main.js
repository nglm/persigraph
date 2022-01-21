import { dimensions, draw_meteogram, draw_mjo, setFigTitle, setAxTitle, draw_entire_graph } from "./plots.js";

const data_path = "./data/";
const data_graph = "./graphs/";
const f1 = "ec.ens.2020011400.sfc.meteogram.json";
const f2 = "ec.ens.2020011500.sfc.meteogram.json";
const f3 = "ec.ens.2020011600.sfc.meteogram.json";
const f4 = "z_s2s_rmm_ecmf_prod_rt_2015030500.json";
const f4_polar = "z_s2s_rmm_ecmf_prod_rt_2015030500_polar.json";
const f5 = "z_s2s_rmm_ecmf_prod_rt_2020120300";


let dims_meteogram = dimensions();
let dims_mjo = dimensions({plotWidth : 400, plotHeight : 400});

let interactiveGroup = [];

// await draw_meteogram(data_path + f1, undefined, "fig01", interactiveGroup);
// await draw_meteogram(data_path + f2, undefined, "fig02", interactiveGroup);
// await draw_meteogram(data_path + f3, undefined, "fig03", interactiveGroup);


let mjo = await draw_mjo(data_path + f4, dims_mjo, "mjo", interactiveGroup);
setAxTitle(mjo, f4);
d3.select("body").append('text').html('<br>');
let mjo_rmm = await draw_meteogram(data_path + f4, undefined, "mjo_rmm", interactiveGroup);
let mjo_rmm_graph = await draw_entire_graph(data_path + f4, data_graph + f4, undefined, "mjo_rmm_graph", interactiveGroup);
let mjo_polar = await draw_meteogram(data_path + f4_polar, undefined, "mjo_polar", interactiveGroup);


// await draw_meteogram(data_path + f4, undefined, "fig01");
// await draw_mjo(data_path + f4, dims_mjo, "fig04");