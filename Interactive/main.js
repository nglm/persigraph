import { dimensions, draw_meteogram, draw_mjo } from "./plots.js";

const data_path = "./data/";
const f1 = "ec.ens.2020011400.sfc.meteogram.json";
const f2 = "ec.ens.2020011500.sfc.meteogram.json";
const f3 = "ec.ens.2020011600.sfc.meteogram.json";
const f4 = "z_s2s_rmm_ecmf_prod_rt_2015030500.json";
const f5 = "z_s2s_rmm_ecmf_prod_rt_2020120300";


let dims_meteogram = dimensions();
let dims_mjo = dimensions({plotWidth : 400, plotHeight : 400});

let interactiveGroup = [];

// await draw_meteogram(data_path + f1, undefined, "fig01", interactiveGroup);
// await draw_meteogram(data_path + f2, undefined, "fig02", interactiveGroup);
// await draw_meteogram(data_path + f3, undefined, "fig03", interactiveGroup);

await draw_meteogram(data_path + f4, undefined, "fig01", interactiveGroup);
await draw_mjo(data_path + f4, dims_mjo, "fig04", interactiveGroup);

// await draw_meteogram(data_path + f4, undefined, "fig01");
// await draw_mjo(data_path + f4, dims_mjo, "fig04");