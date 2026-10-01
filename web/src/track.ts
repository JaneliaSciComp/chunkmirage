// The track page: a nucleus followed through a time-lapse of a stem cell colony, frame by
// frame, as the segmentation is read: chunkmirage.tracking's step, one call a frame in the
// engine's Pyodide workers, on a box around the nucleus that the page's reader reads from
// the public bucket. Neuroglancer shows the images and the nuclei straight from the bucket;
// the page keeps the nucleus followed highlighted in each frame (its id changes from frame
// to frame) and plots its volume as each frame comes in. Double-click another to follow it.
import { Engine } from "./engine";

const BASE = "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/"
  + "hipsc_fov_nuclei_timelapse_data_used_for_analysis/baseline_colonies_fov_timelapse_dataset/20200323_09_small";
const RAW = `${BASE}/raw.ome.zarr`, SEG = `${BASE}/seg.ome.zarr`;
const LEVEL = 3;              // the segmentation's level followed: 1.08 µm across, 0.75 µm deep
const MARGIN = [3, 6, 6];     // the box read around the nucleus, voxels of that level (2 to 6 µm)
const START = { t: 100, label: 148 };  // a nucleus that grows through the first day
const TO_UM: Record<string, number> = { micrometer: 1, um: 1, nanometer: 1e-3, nm: 1e-3 };

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const engine = new Engine(showCounts);

interface Found { label: number; volume: number; centroid: number[]; lo: number[]; hi: number[]; touches: boolean; overlap?: number; divided?: boolean }
interface Frame { frames: number; minutes: number }
let time: Frame = { frames: 1, minutes: 5 };
let followed = new Map<number, Found>();  // frame -> the nucleus there
let run = 0, shown = -1;
let lost: { forward?: number; back?: number } = {};

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  $("counts").textContent = `Frames followed: ${followed.size} of ${time.frames}${engine.running ? ` · reading and matching ${engine.running}` : ""}`;
}

/** The time axis (frames, minutes apart), from the store's metadata. */
async function timeAxis(): Promise<Frame> {
  const attrs = await (await fetch(`${SEG}/.zattrs`)).json(), array = await (await fetch(`${SEG}/0/.zarray`)).json();
  const m = attrs.multiscales[0], t = m.axes.findIndex((a: { name: string }) => a.name === "t");
  return { frames: array.shape[t], minutes: m.datasets[0].coordinateTransformations[0].scale[t] };
}

// ------------------------------------------------ following
const level = () => engine.sources.seg.levels[LEVEL];
const voxel = () => level().voxel.map((v) => v * (TO_UM[engine.sources.seg.axes[2].unit] ?? 1));

async function read(t: number, lo: number[], hi: number[]): Promise<ArrayBuffer> {
  return engine.read("seg", LEVEL, lo, hi, { t });
}

/** The nucleus labelled `label` at frame `t0`, followed both ways until it is lost. */
async function follow(t0: number, label: number) {
  const mine = ++run, shape = level().shape, dtype = engine.sources.seg.dtype;
  followed = new Map();
  lost = {};
  draw();
  status(`Finding nucleus ${label} in frame ${t0}…`);
  const whole = await read(t0, [0, 0, 0], shape);
  const first = await engine.call<Found | null>("track_measure", { label, start: [0, 0, 0], voxel: voxel(), dtype, shape }, [whole]);
  if (!first || mine !== run) { if (!first) status(`No nucleus ${label} in frame ${t0}.`); return; }
  followed.set(t0, first);
  highlight();
  status("Following it frame by frame, both ways…");
  const way = async (dt: 1 | -1) => {
    let record = first, t = t0;
    while (mine === run && t + dt >= 0 && t + dt < time.frames) {
      const nt = t + dt;
      let found = await stepTo(record, t, nt, MARGIN);
      if (found?.touches) found = await stepTo(record, t, nt, MARGIN.map((m) => 2 * m));  // the box cut it
      if (mine !== run) return;
      if (!found) { lost[dt > 0 ? "forward" : "back"] = nt; break; }
      found.divided = found.volume <= 0.65 * record.volume;  // chunkmirage.tracking.DIVIDED
      followed.set(nt, found);
      record = found; t = nt;
      draw();
      showCounts();
    }
  };
  await Promise.all([way(1), way(-1)]);
  if (mine === run) { status("Done: double-click another nucleus to follow it."); draw(); }
}

async function stepTo(record: Found, t: number, nt: number, margin: number[]): Promise<Found | null> {
  const shape = level().shape;
  const lo = record.lo.map((v, a) => Math.max(v - margin[a], 0)), hi = record.hi.map((v, a) => Math.min(v + margin[a], shape[a]));
  const [prev, next] = await Promise.all([read(t, lo, hi), read(nt, lo, hi)]);
  return engine.call<Found | null>("track_step", {
    label: record.label, start: lo, voxel: voxel(), dtype: engine.sources.seg.dtype, shape: hi.map((h, a) => h - lo[a]),
  }, [prev, next]);
}

// ------------------------------------------------ the graph
function draw() {
  const svg = $<SVGSVGElement>("graph"), W = 320, H = 190, L = 34, B = 18;
  const pts = [...followed.entries()].sort((a, b) => a[0] - b[0]);
  const vols = pts.map(([, f]) => f.volume), vmax = Math.max(100, ...vols) * 1.1;
  const x = (t: number) => L + ((W - L - 6) * t) / Math.max(time.frames - 1, 1);
  const y = (v: number) => H - B - ((H - B - 8) * v) / vmax;
  const hours = (time.frames * time.minutes) / 60, ticks: string[] = [];
  for (let h = 0; h <= hours; h += 12) ticks.push(`<line x1="${x((h * 60) / time.minutes)}" x2="${x((h * 60) / time.minutes)}" y1="${H - B}" y2="${H - B + 3}" stroke="#5d6675"/><text x="${x((h * 60) / time.minutes)}" y="${H - 4}" text-anchor="middle">${h} h</text>`);
  for (const v of [0, vmax / 2, vmax].map((v) => Math.round(v / 100) * 100)) ticks.push(`<text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v}</text>`);
  // the line, broken where frames are missing
  let path = "", prevT = -2;
  for (const [t, f] of pts) { path += `${t === prevT + 1 ? "L" : "M"}${x(t).toFixed(1)},${y(f.volume).toFixed(1)}`; prevT = t; }
  const divisions = pts.filter(([, f]) => f.divided).map(([t]) => `<line x1="${x(t)}" x2="${x(t)}" y1="8" y2="${H - B}" stroke="#ffd21f" stroke-dasharray="3 3"/>`);
  const now = shown >= 0 ? `<line x1="${x(shown)}" x2="${x(shown)}" y1="8" y2="${H - B}" stroke="#7ab0ff"/>` : "";
  svg.innerHTML = `<line x1="${L}" x2="${W - 6}" y1="${H - B}" y2="${H - B}" stroke="#5d6675"/><text x="4" y="12">µm³</text>`
    + ticks.join("") + divisions.join("") + now + `<path d="${path}" fill="none" stroke="#45f07a" stroke-width="1.5"/>`;
  const span = pts.length ? [pts[0][0], pts[pts.length - 1][0]] : [0, 0];
  const n = pts.filter(([, f]) => f.divided).length;
  $("summary").textContent = pts.length
    ? `${pts.length} frames, ${((span[1] - span[0]) * time.minutes / 60).toFixed(1)} h: ${Math.round(vols[0])} to ${Math.round(vols[vols.length - 1])} µm³`
      + `${n ? `, ${n} division${n > 1 ? "s" : ""} (dashed)` : ""}.`
      + `${lost.back !== undefined ? ` Lost going back at frame ${lost.back}.` : ""}${lost.forward !== undefined ? ` Lost going on at frame ${lost.forward}.` : ""}`
    : "";
}

// ------------------------------------------------ the viewer
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type NgViewer = any;
const viewer = (): NgViewer => ($<HTMLIFrameElement>("ng").contentWindow as unknown as { viewer?: NgViewer })?.viewer;

/** The frame the viewer is at. */
function frameShown(): number {
  const v = viewer();
  if (!v?.position) return -1;
  const names: string[] = v.coordinateSpace.value.names, i = names.indexOf("t");
  return i < 0 ? -1 : Math.round(v.position.value[i]);  // frame i is centred on i
}

let showing: string | null = null;  // the segment the page last showed
let updating = false;  // the page is changing the segments shown (which tells this again)
/** Show the nucleus followed in the frame on screen (its id there), and follow a new one
 * when one is double-clicked. */
function highlight() {
  const v = viewer(), layer = v?.layerManager?.getLayerByName("nuclei")?.layer;
  if (!layer || updating) return;
  const visible = layer.displayState.segmentationGroupState.value.visibleSegments;
  const ids: string[] = [...visible].map(String);
  const t = frameShown();
  const picked = ids.find((id) => id !== showing);
  if (picked && t >= 0) { showing = picked; void follow(t, Number(picked)); return; }
  const f = followed.get(t), want = f ? String(f.label) : null;
  if (want !== showing || ids.length !== (want ? 1 : 0)) {
    showing = want;
    updating = true;
    try { visible.clear(); if (want) visible.add(BigInt(want)); } finally { updating = false; }
  }
  if (t !== shown) {
    shown = t;
    draw();
    if (f && $<HTMLInputElement>("centre").checked) centreOn(f);
  }
}

/** Move the view to a nucleus's centroid (its frame's), keeping the zoom. */
function centreOn(f: Found) {
  const v = viewer(), names: string[] = v.coordinateSpace.value.names, scales: Float64Array = v.coordinateSpace.value.scales;
  const um = f.centroid.map((c, a) => c * voxel()[a]), p = Float32Array.from(v.position.value);
  for (const [a, n] of ["z", "y", "x"].entries()) { const i = names.indexOf(n); if (i >= 0) p[i] = (um[a] * 1e-6) / scales[i]; }
  v.position.value = p;
}

function viewerState(dims: Record<string, [number, string]>, centroid: number[]) {
  const vx = voxel(), toM = 1e-6;
  const raw = { x: dims.x[0] / toM, y: dims.y[0] / toM, z: dims.z[0] / toM };
  const c = centroid.map((v, a) => v * vx[a]);  // micrometres
  return {
    dimensions: dims, displayDimensions: ["x", "y", "z"],
    position: [c[2] / raw.x, c[1] / raw.y, c[0] / raw.z, START.t],  // OME-Zarr: voxel i centred on i
    crossSectionScale: 0.8, showAxisLines: false, crossSectionBackgroundColor: "#000000",  // image voxels per pixel
    layers: [
      { type: "image", name: "lamin B1", source: `zarr://${RAW}/`, localPosition: [0],
        shader: "#uicontrol invlerp normalized(range=[98, 135])\nvoid main() { emitGrayscale(normalized()); }\n" },
      { type: "segmentation", name: "nuclei", source: `zarr://${SEG}/`, localPosition: [0], segments: [String(START.label)],
        selectedAlpha: 0.45, notSelectedAlpha: 0, segmentDefaultColor: "#45f07a",
        crossSectionRenderScale: 3 },  // a coarser level: the finest's chunks are 28 MB planes
    ],
    layout: "xy", selectedLayer: { visible: false },
    velocity: { t: { velocity: 4, atBoundary: "loop", paused: true } },  // the play button: 4 frames a second
  };
}

async function start() {
  const t0 = performance.now();
  time = await timeAxis();
  await engine.start({ seg: { source: SEG, select: { t: 0, c: 0 }, chunk: [1, 256, 256] } }, status);
  status(`Data opened in ${((performance.now() - t0) / 1000).toFixed(1)} s; loading Python…`);
  const s = engine.sources.seg, toM = (TO_UM[s.axes[2].unit] ?? 1) * 1e-6;
  // the viewer's grid: the images' finest (the segmentation's is finer still), and frames
  const rawAttrs = await (await fetch(`${RAW}/.zattrs`)).json();
  const sc: number[] = rawAttrs.multiscales[0].datasets[0].coordinateTransformations[0].scale;
  const dims: Record<string, [number, string]> = { x: [sc[4] * toM, "m"], y: [sc[3] * toM, "m"], z: [sc[2] * toM, "m"], t: [time.minutes * 60, "s"] };
  await engine.ready;
  const whole = await read(START.t, [0, 0, 0], level().shape);
  const first = await engine.call<Found | null>("track_measure", { label: START.label, start: [0, 0, 0], voxel: voxel(), dtype: s.dtype, shape: level().shape }, [whole]);
  const ng = $<HTMLIFrameElement>("ng");
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(viewerState(dims, first?.centroid ?? level().shape.map((n) => n / 2))))}`;
  ng.hidden = false;
  $("empty").hidden = true;
  showing = String(START.label);
  const attach = () => {
    const v = viewer();
    if (!v?.position || !v.layerManager?.getLayerByName("nuclei")) return void setTimeout(attach, 300);
    v.position.changed.add(highlight);
    v.layerManager.getLayerByName("nuclei").layer.displayState.segmentationGroupState.value.visibleSegments.changed.add(highlight);
  };
  attach();
  $("command").textContent = `uv run python examples/track_nucleus.py --frame ${START.t} --label ${START.label}`;
  void follow(START.t, START.label);
  Object.assign(window, { engine, followed });  // for a console, and the headless checks
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
$("graph").addEventListener("click", (e) => {  // jump to the frame clicked
  const svg = $<SVGSVGElement>("graph"), r = svg.getBoundingClientRect(), L = 34;
  const t = Math.round(((((e as MouseEvent).clientX - r.left) / r.width) * 320 - L) / (320 - L - 6) * (time.frames - 1));
  const v = viewer(), i = v?.coordinateSpace.value.names.indexOf("t");
  if (v && i >= 0 && t >= 0 && t < time.frames) { const p = Float32Array.from(v.position.value); p[i] = t; v.position.value = p; }
});
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
