// ----------------------- About loading local data ---------------------------
//
// Web browsers are not happy about you loading local data.
// To get around  this, you have to run a local web server. In your terminal,
// after cd-ing to your website's document root, type:
// 'python3 -m http.server 8888'
// To close the local web server you can then use:
// - 'ps -fA | grep python' to find the PID 'yourPID'
// - 'kill yourPID'

import { dimensions, draw_meteogram, draw_mjo } from "./plots.js";

const data_path = "./data/";
const f1 = "ec.ens.2020011400.sfc.meteogram.json";
const f2 = "ec.ens.2020011500.sfc.meteogram.json";
const f3 = "ec.ens.2020011600.sfc.meteogram.json";
const f4 = "z_s2s_rmm_ecmf_prod_rt_2015030500.json";


let dims_meteogram = dimensions();
let dims_mjo = dimensions({figWidth : 600});

let meteogram01 = draw_meteogram(data_path + f1, undefined, "fig01");
let meteogram02 = draw_meteogram(data_path + f2, undefined, "fig02");
let meteogram03 = draw_meteogram(data_path + f3, undefined, "fig03");
let mjo01 = draw_mjo(data_path + f4, dims_mjo, "fig04");

