// ----------------------- About loading local data ---------------------------
//
// Web browsers are not happy about you loading local data.
// To get around  this, you have to run a local web server. In your terminal,
// after cd-ing to your website's document root, type:
// 'python3 -m http.server 8888'
// To close the local web server you can then use:
// - 'ps -fA | grep python' to find the PID 'yourPID'
// - 'kill yourPID'

import { dimensions, draw_meteogram } from "./plots.js";

let dims = dimensions();

let meteogram01 = draw_meteogram("./data/meteogram.json", undefined, "fig01");
let meteogram02 = draw_meteogram("./data/meteogram.json", undefined, "fig02");
let meteogram03 = draw_meteogram("./data/meteogram.json", undefined, "fig03");

