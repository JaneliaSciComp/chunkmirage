// The stitch page: the tiles of a BigStitcher project stitched by interest points and RANSAC,
// every step chunkmirage.stitching's Python run by the engine's Pyodide workers (engine.ts):
// interest points in each overlap, matches, RANSAC and the global fit, then the fused volume,
// which the engine serves chunk by chunk as Neuroglancer asks for it. Settings rerun what they
// change: the detection's rerun everything, the matching's and RANSAC's only the fit (the
// points are kept). Defaults and the "same in Python" command come from StitchParams.
import { Engine } from "./engine";
import type { StitchParams } from "./generated/chunkmirage";
import schema from "./generated/chunkmirage.schema.json";
import type { ViewInfo } from "./types";

const XML = "https://janelia-bigstitcher-spark.s3.amazonaws.com/Stitching/dataset.xml";
const TILE_CHUNK = [64, 128, 128];  // the tiles as served, as stored
const FUSED_CHUNK = [8, 128, 128];  // the fused volume's chunks: thin, for x-y slices
const COLOURS = ["#ff4fd8", "#45f07a"];  // neighbouring tiles, alternately
const IN = "#45f07a", OUT = "#ff5a5a";
const SEAM_ZOOM = 0.6;  // where it opens: scene units per screen pixel
const TO_METRES: Record<string, number> = { micrometer: 1e-6, um: 1e-6, nanometer: 1e-9, nm: 1e-9, millimeter: 1e-3, mm: 1e-3, meter: 1, m: 1 };
const FIELDS = schema.$defs.StitchParams.properties as Record<string, { default?: unknown; description?: string; enum?: string[] }>;
const DEFAULTS = Object.fromEntries(Object.entries(FIELDS).map(([k, v]) => [k, v.default])) as Required<StitchParams>;

type Affine = number[][];
interface Tile { name: string; setup: number; url: string; select: Record<string, number>; shape: number[][]; stage: Affine; reference: Affine | null }
interface Overlap { tiles: [number, number]; lo: number[]; hi: number[]; regions: ([number[], number[]] | null)[] }
interface Pair { tiles: [number, number]; points: [number, number]; candidates: number; inliers: number; kept: boolean; a: number[][]; b: number[][]; inlier: boolean[]; error?: { mean: number; max: number } }
interface Grid { shape: number[]; voxel: number[]; origin: number[] }
interface Found { pairs: Pair[]; corrections: Affine[]; placed: boolean[]; placements: Affine[]; grids: Grid[]; reference: number | null }

/** A control: a number slider (`log`: over its logarithm) or a choice. */
type Control = { key: keyof StitchParams; label: string; group: string }
  & ({ min: number; max: number; step: number; log?: boolean; unit?: string } | { options: [string, string][] });
const CONTROLS: Control[] = [
  { key: "channel", label: "Channel", group: "Interest points", options: [["0", "0: tissue"], ["1", "1: sparse cells"], ["2", "2: bright tracts"]] },
  { key: "sigma", label: "Blob size (sigma)", group: "Interest points", min: 1, max: 4, step: 0.1, unit: " px" },
  { key: "threshold", label: "Threshold", group: "Interest points", min: -3.3, max: -1.3, step: 0.05, log: true },
  { key: "neighbors", label: "Neighbours per descriptor", group: "Matching", min: 2, max: 6, step: 1 },
  { key: "redundancy", label: "Redundancy", group: "Matching", min: 0, max: 3, step: 1 },
  { key: "significance", label: "Significance (ratio test)", group: "Matching", min: 1, max: 5, step: 0.05 },
  { key: "model", label: "Model", group: "RANSAC", options: [["translation", "translation"], ["rigid", "rigid"], ["affine", "affine"]] },
  { key: "epsilon", label: "Max error (epsilon)", group: "RANSAC", min: 0.5, max: 20, step: 0.5, unit: " µm" },
  { key: "min_inlier_ratio", label: "Min inlier ratio", group: "RANSAC", min: 0, max: 1, step: 0.05 },
  { key: "min_inliers", label: "Min inliers", group: "RANSAC", min: 1, max: 40, step: 1 },
  { key: "iterations", label: "Iterations", group: "RANSAC", min: 1, max: 5, step: 0.25, log: true },
  { key: "blend", label: "Blending band", group: "Fusion", min: 0, max: 100, step: 5, unit: " µm" },
];
const DETECTION: (keyof StitchParams)[] = ["channel", "level", "sigma", "threshold", "margin"];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const engine = new Engine(showCounts);
const params: Required<StitchParams> = { ...DEFAULTS };
let tiles: Tile[] = [];
let points: { key: string; values: number[][][][]; overlaps: Overlap[] } | null = null;
let found: Found | null = null;
let fused = "", runs = 0, shown = false, opened = -1;  // opened: the channel whose tiles are open

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  const c = engine.counts;
  $("counts").textContent = `Chunks computed here: ${c.computed} · computing ${engine.running} · waiting ${engine.waiting}${c.failed ? ` · failed ${c.failed}` : ""}`;
}
const tileView = (t: number) => `c${params.channel}t${t}`;
const seconds = (t0: number) => `${((performance.now() - t0) / 1000).toFixed(1)} s`;

/** The project's tiles of the chosen channel, served as views (the reader reads them). */
async function openTiles() {
  const base = XML.slice(0, XML.lastIndexOf("/"));
  const xml = await (await fetch(XML)).text();
  tiles = await engine.call<Tile[]>("tiles", { xml, base, channel: params.channel });
  const views = Object.fromEntries(tiles.map((t, k) => [tileView(k), { source: t.url, select: t.select, chunk: TILE_CHUNK }]));
  if (!engine.infos[tileView(0)]) await engine.add(views);
  for (const [k, t] of tiles.entries()) t.shape = engine.infos[tileView(k)].levels.map((l) => l.shape);
  opened = params.channel;
}

/** Each overlap's interest points in both its tiles, kept until a detection setting changes. */
async function detect(): Promise<number[][][][]> {
  const key = JSON.stringify(DETECTION.map((k) => params[k]));
  if (points?.key === key) return points.values;
  const overlaps = await engine.call<Overlap[]>("overlaps", { tiles, params });
  const values = await Promise.all(overlaps.map((o) => Promise.all(o.regions.map(async (r, k) => {
    if (!r) return [];
    const t = o.tiles[k], level = engine.infos[tileView(t)].levels[params.level];
    const block = await engine.read(tileView(t), params.level, r[0], r[1]);
    return engine.call<number[][]>("points", {
      tile: tiles[t], params, dtype: engine.infos[tileView(t)].dtype, shape: r[1].map((h, a) => h - r[0][a]),
      start: r[0], voxel: level.voxel, lo: o.lo, hi: o.hi,
    }, [block]);
  }))));
  points = { key, values, overlaps };
  return values;
}

/** The fused volume of `f`'s placements, served under a new name. */
function serveFused(f: Found): string {
  const name = `fused~${++runs}`, info = engine.infos[tileView(0)];
  const placements = f.placements, own = tiles.map((t) => ({ ...t })), p = { ...params };
  const view: ViewInfo = {
    dtype: info.dtype, out: info.dtype, channels: 1, lead: 0, halo: [0, 0, 0], axes: info.axes,
    levels: f.grids.map((g) => ({ shape: g.shape, voxel: g.voxel, origin: g.origin })),
  };
  engine.serve(name, view, FUSED_CHUNK, async (level, index) => {
    const g = f.grids[level], C = FUSED_CHUNK;
    const lo = index.map((i, a) => i * C[a]), hi = lo.map((o, a) => Math.min(o + C[a], g.shape[a]));
    const args = { tiles: own, placements, level, grid: g, lo, hi };
    const regions = await engine.call<([number[], number[]] | null)[]>("regions", args);
    const blocks = await Promise.all(regions.flatMap((r, t) => (r ? [engine.read(tileView(t), level, r[0], r[1])] : [])));
    return engine.call<ArrayBuffer>("fuse", { ...args, regions, dtype: info.dtype, chunk: C, params: p }, blocks);
  });
  return name;
}

// ------------------------------------------------ the viewer
/** A placement (level-0 voxels to the scene) as Neuroglancer's matrix from a tile's voxels
 * to the viewer's coordinates, which count the fused volume's level-0 voxels. */
function matrix(a: Affine, voxel: number[]): number[][] {
  return a.map((row, r) => row.map((v) => v / voxel[r]));
}

/** Annotations of the matches, x and y only (shown on every slice): lines from tile i's
 * point to tile j's, where `place` puts them. */
function matchLayer(name: string, colour: string, inliers: boolean, place: (t: number, p: number[]) => number[], toM: number) {
  const [width, ends] = inliers ? [2, 5] : [1, 0];
  const annotations: Record<string, unknown>[] = [];
  for (const [k, pair] of (found?.pairs ?? []).entries()) {
    const [i, j] = pair.tiles;
    pair.a.forEach((a, m) => {
      if (pair.inlier[m] !== inliers || (inliers && !pair.kept)) return;
      const pa = place(i, a), pb = place(j, pair.b[m]);
      annotations.push({ type: "line", id: `${k}-${m}`, pointA: [pa[2], pa[1]], pointB: [pb[2], pb[1]] });
    });
  }
  return {
    type: "annotation", name, annotationColor: colour, annotations,
    source: { url: "local://annotations", transform: { outputDimensions: { x: [toM, "m"], y: [toM, "m"] } } },
    shader: `void main() { setColor(defaultColor()); setLineWidth(${width.toFixed(1)}); setEndpointMarkerSize(${ends.toFixed(1)}, ${ends.toFixed(1)}); }`,
  };
}

const apply = (a: Affine, p: number[]) => a.map((r) => r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]);

function viewerState() {
  const f = found!, g = f.grids[0], info = engine.infos[tileView(0)];
  const toM = TO_METRES[info.axes[2].unit] ?? 1;
  const dims = { x: [g.voxel[2] * toM, "m"], y: [g.voxel[1] * toM, "m"], z: [g.voxel[0] * toM, "m"] };
  const out = { z: dims.z, y: dims.y, x: dims.x };  // the matrices' rows, C order
  // tiles coloured as a checkerboard of their stage grid, so neighbours differ
  const xs = [...new Set(tiles.map((t) => Math.round(t.stage[2][3])))].sort((a, b) => a - b);
  const ys = [...new Set(tiles.map((t) => Math.round(t.stage[1][3])))].sort((a, b) => a - b);
  const colour = (t: Tile) => COLOURS[(xs.indexOf(Math.round(t.stage[2][3])) + ys.indexOf(Math.round(t.stage[1][3]))) % 2];
  const tileLayers = (suffix: string, place: (k: number) => Affine) => tiles.map((t, k) => ({
    type: "image", name: `${t.name}${suffix}`, blend: "additive",
    source: { url: `zarr3://${engine.url(tileView(k))}`, transform: { outputDimensions: out, matrix: matrix(place(k), g.voxel) } },
    shader: `#uicontrol invlerp normalized(range=[10, 120])\n#uicontrol vec3 colour color(default="${colour(t)}")\nvoid main() { emitRGB(colour * normalized()); }\n`,
  }));
  const stage = tileLayers(" at stage", (k) => tiles[k].stage), placed = tileLayers(" stitched", (k) => f.placements[k]);
  const atStage = (_t: number, p: number[]) => p, byFit = (t: number, p: number[]) => apply(f.corrections[t], p);
  const layers = [
    ...stage, ...placed,
    matchLayer("inliers at stage", IN, true, atStage, toM), matchLayer("rejected at stage", OUT, false, atStage, toM),
    matchLayer("inliers stitched", IN, true, byFit, toM), matchLayer("rejected stitched", OUT, false, byFit, toM),
    { type: "image", name: "fused", source: `zarr3://${engine.url(fused)}`, shader: "#uicontrol invlerp normalized(range=[10, 120])\nvoid main() { emitGrayscale(normalized()); }\n" },
  ];
  // the kept overlap nearest the middle of it all, close enough to see the tiles disagree
  const middle = [0, 1, 2].map((a) => g.origin[a] + g.voxel[a] * (g.shape[a] - 1) / 2);
  const centre = (o: Overlap) => o.lo.map((v, a) => (v + o.hi[a]) / 2);
  const far = (o: Overlap) => Math.hypot(...centre(o).map((c, a) => (a ? c - middle[a] : 0)));
  const near = points!.overlaps.filter((_, k) => f.pairs[k].kept).sort((a, b) => far(a) - far(b))[0] ?? points!.overlaps[0];
  const seam = centre(near);
  const main = document.querySelector("main")!;
  const w = (main.clientWidth || 1200) / 3 - 16, h = (main.clientHeight || 800) - 50;
  const column = (names: string[]) => ({ type: "viewer", layers: names, layout: "xy" });
  return {
    dimensions: dims, displayDimensions: ["x", "y", "z"],
    position: [2, 1, 0].map((a) => seam[a] / g.voxel[a]),
    crossSectionScale: Math.min(Math.max(g.shape[2] / w, g.shape[1] / h) * 1.05, SEAM_ZOOM / g.voxel[2]),
    crossSectionBackgroundColor: "#000000", showAxisLines: false, showDefaultAnnotations: false, layers,
    layout: { type: "row", children: [
      column([...stage.map((l) => l.name), "rejected at stage", "inliers at stage"]),
      column([...placed.map((l) => l.name), "rejected stitched", "inliers stitched"]),
      column(["fused"]),
    ] },
    selectedLayer: { visible: false },
  };
}

interface NgState { layers: { name: string; shaderControls?: unknown }[]; [k: string]: unknown }
interface NgViewer { state: { toJSON(): NgState; restoreState(s: NgState): void } }

/** Show the new result, keeping where the user looks and the contrast they set. */
function show() {
  const ng = $<HTMLIFrameElement>("ng"), state = viewerState() as unknown as NgState;
  const viewer = (ng.contentWindow as (Window & { viewer?: NgViewer }) | null)?.viewer;
  if (shown && viewer) {
    const cur = viewer.state.toJSON();
    for (const k of ["position", "crossSectionScale", "crossSectionOrientation"]) if (cur[k] !== undefined) state[k] = cur[k];
    for (const l of state.layers) {
      const old = cur.layers?.find((o) => o.name === l.name);
      if (old?.shaderControls) l.shaderControls = old.shaderControls;
    }
    viewer.state.restoreState(state);
    return;
  }
  shown = true;
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`;
  ng.hidden = false; $("captions").hidden = false; $("empty").hidden = true;
}

// ------------------------------------------------ the result
function showResult(f: Found, timing: string) {
  const kept = f.pairs.filter((p) => p.kept);
  const n = (points?.values ?? []).flat().reduce((s, v) => s + v.length, 0);
  const lonely = tiles.filter((_, k) => !f.placed[k]).map((t) => t.name);
  $("summary").innerHTML = `<b>${kept.length} of ${f.pairs.length}</b> overlaps kept, from ${n.toLocaleString()} interest points. `
    + (f.reference != null ? `The tiles sit <b>${f.reference.toFixed(2)} µm</b> (RMS) from where BigStitcher's own stitching put them. ` : "")
    + (lonely.length ? `Not joined to any other tile: ${lonely.join(", ")}. ` : "")
    + `<span class="faint">${timing}</span>`;
  const rows = f.pairs.map((p) => {
    const [i, j] = p.tiles;
    return `<tr class="${p.kept ? "" : "dropped"}"><td>${tiles[i].name.replace("tile ", "")}–${tiles[j].name.replace("tile ", "")}</td>`
      + `<td>${p.points[0]}/${p.points[1]}</td><td>${p.candidates}</td><td>${p.kept ? p.inliers : "dropped"}</td>`
      + `<td>${p.error ? p.error.mean.toFixed(2) : ""}</td></tr>`;
  });
  $("pairs").innerHTML = `<tr><th>Tiles</th><th>Points</th><th>Matches</th><th>Inliers</th><th>Error µm</th></tr>${rows.join("")}`;
  $("result").hidden = false;
}

function showCommand() {
  const q = (Object.keys(DEFAULTS) as (keyof StitchParams)[])
    .filter((k) => JSON.stringify(params[k]) !== JSON.stringify(DEFAULTS[k])).map((k) => `${k}=${params[k]}`);
  $("command").textContent = `chunkmirage serve 'stitch://${XML}${q.length ? `?${q.join("&")}` : ""}' --python-viewer`;
}

let queued = false, running: Promise<void> | null = null;
/** Run what the settings changed, once at a time; a change made meanwhile runs after. */
function run() {
  if (running) { queued = true; return; }
  running = (async () => {
    try {
      const t0 = performance.now();
      if (opened !== params.channel) await openTiles();
      const fresh = points?.key !== JSON.stringify(DETECTION.map((k) => params[k]));
      status(fresh ? "Finding interest points in every overlap…" : "Matching and RANSAC…");
      const values = await detect();
      const t1 = performance.now();
      status("Matching and RANSAC…");
      found = await engine.call<Found>("register", { tiles, points: values, params });
      fused = serveFused(found);
      showResult(found, `${fresh ? `Points ${((t1 - t0) / 1000).toFixed(1)} s, ` : ""}matches, RANSAC and fit ${seconds(t1)}.`);
      showCommand();
      show();
      status("Done. The fused volume is computed as the viewer asks for it.");
    } catch (e) {
      console.error(e);
      status(`Failed: ${(e as Error).message ?? e}`);
    } finally {
      running = null;
      if (queued) { queued = false; run(); }
    }
  })();
}
function controls() {
  let group = "";
  for (const c of CONTROLS) {
    if (c.group !== group) {
      group = c.group;
      const h = document.createElement("div");
      h.className = "label";
      h.textContent = group;
      $("controls").append(h);
    }
    const row = document.createElement("label");
    row.className = "control";
    row.title = FIELDS[c.key]?.description ?? "";
    const value = params[c.key];
    if ("options" in c) {
      row.innerHTML = `<span></span><select></select>`;
      const select = row.querySelector("select")!;
      for (const [v, text] of c.options) select.append(new Option(text, v));
      select.value = String(value);
      select.addEventListener("change", () => {
        (params as Record<string, unknown>)[c.key] = typeof DEFAULTS[c.key] === "number" ? Number(select.value) : select.value;
        run();
      });
    } else {
      row.innerHTML = `<span></span><output></output><input type="range">`;
      const input = row.querySelector("input")!, out = row.querySelector("output")!;
      const toValue = (s: string) => (c.log ? Number((10 ** Number(s)).toPrecision(2)) : Number(s));
      const integer = Number.isInteger(DEFAULTS[c.key]) && (c.log || c.step >= 1);
      Object.assign(input, { min: String(c.min), max: String(c.max), step: String(c.step), value: String(c.log ? Math.log10(Number(value)) : value) });
      const shown = () => {
        const v = integer ? Math.round(toValue(input.value)) : toValue(input.value);
        out.textContent = `${v.toLocaleString()}${c.unit ?? ""}`;
        return v;
      };
      shown();
      input.addEventListener("input", shown);
      input.addEventListener("change", () => { (params as Record<string, unknown>)[c.key] = shown(); run(); });
    }
    row.querySelector("span")!.textContent = c.label;
    $("controls").append(row);
  }
  $("controls").hidden = false;
}

async function start() {
  ($("xml") as HTMLAnchorElement).href = XML;
  const t0 = performance.now();
  await engine.start({}, status, ["scipy"]);  // the interest points' filters
  await engine.ready;
  status(`Python ready in ${seconds(t0)}. Reading the project…`);
  controls();
  showCommand();
  run();
  Object.assign(window, { engine, params, run });  // for a console, and the headless checks
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
