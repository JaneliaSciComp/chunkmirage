// The track page: nuclei followed through a time-lapse of a stem cell colony, frame by
// frame, as the segmentation is read: chunkmirage.tracking's step, one call a frame in the
// engine's Pyodide workers, on a box around each nucleus that the page's reader reads from
// the public bucket. Neuroglancer shows the images and the nuclei straight from the bucket.
// Each nucleus double-clicked is a track of its own colour, followed both ways in time; when
// one collapses into mitosis and is lost, its likely daughters are guessed (new nuclei
// appearing near where it was) and followed too, so a track is a (guessed) lineage. The page keeps the
// nuclei followed highlighted in each frame (their ids change from frame to frame) and plots
// their volumes as frames come in. Double-click a tracked nucleus again to drop its track.
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
/** One line of descent: frame -> the nucleus there; `from` the frame it split off at; `end`
 * how it ends going forward (lost, into mitosis with or without daughters found, or the
 * movie's end). */
interface Branch { frames: Map<number, Found>; from?: number; lost?: number; end?: "lost" | "mitosis" | "divided" | "movie" }
/** A nucleus double-clicked and its descendants (and, going back, its ancestors). */
interface Track { colour: string; picked: { t: number; label: number }; branches: Branch[]; dropped: boolean; running: number }
const COLOURS = ["#45f07a", "#ff4fd8", "#7ab0ff", "#ffd21f", "#ff8a3d", "#3de0ff"];
const MAX_BRANCHES = 8;  // per track: a lineage two or three divisions deep
// chunkmirage.tracking's: a collapse (DIVIDED of the largest in the dozen frames before), and
// the daughters looked for after it (GAP frames, WITHIN µm of the mother)
const DIVIDED = 0.65, GAP = 24, WITHIN = 18;
let time: Frame = { frames: 1, minutes: 5 };
const tracks: Track[] = [];
let shown = -1, picks = 0;

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  const n = tracks.reduce((s, k) => s + k.branches.reduce((b, br) => b + br.frames.size, 0), 0);
  $("counts").textContent = `Frames followed: ${n}, ${tracks.length} track${tracks.length === 1 ? "" : "s"}${engine.running ? ` · reading and matching ${engine.running}` : ""}`;
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

/** Follow the nucleus labelled `label` at frame `t0` as a new track, both ways in time,
 * and both daughters at each division (forward). */
async function follow(t0: number, label: number) {
  const shape = level().shape, dtype = engine.sources.seg.dtype;
  const track: Track = { colour: COLOURS[picks++ % COLOURS.length], picked: { t: t0, label }, branches: [], dropped: false, running: 0 };
  tracks.push(track);
  status(`Finding nucleus ${label} in frame ${t0}…`);
  const whole = await read(t0, [0, 0, 0], shape);
  const first = await engine.call<Found | null>("track_measure", { label, start: [0, 0, 0], voxel: voxel(), dtype, shape }, [whole]);
  if (!first) { drop(track); status(`No nucleus ${label} in frame ${t0}.`); return; }
  const main: Branch = { frames: new Map([[t0, first]]) };
  track.branches.push(main);
  highlight(true);
  status("Following frame by frame, both ways, and the daughters of each division…");
  await Promise.all([along(track, main, first, t0, 1), along(track, main, first, t0, -1)]);
  if (!tracks.some((k) => k.running)) status("Done: double-click a nucleus to follow it too, or a followed one to drop it.");
  draw();
}

/** One branch of a track, from `record` at frame `t`, a frame at a time in direction `dt`. */
async function along(track: Track, branch: Branch, record: Found, t: number, dt: 1 | -1): Promise<void> {
  track.running++;
  const splits: Promise<void>[] = [];
  try {
    while (!track.dropped && t + dt >= 0 && t + dt < time.frames) {
      const nt = t + dt;
      let found = await stepTo(record, t, nt, MARGIN);
      if (found?.touches) found = await stepTo(record, t, nt, MARGIN.map((m) => 2 * m));  // the box cut it
      if (track.dropped) return;
      if (!found) {
        branch.lost = nt;
        const mother = dt > 0 ? motherOf(branch, nt) : null;
        if (dt > 0) branch.end = mother ? "mitosis" : "lost";
        if (mother) splits.push(daughters(track, branch, mother, nt));  // it went into mitosis
        break;
      }
      branch.frames.set(nt, found);
      record = found; t = nt;
      draw();
      showCounts();
      if (nt === frameShown()) highlight(true);
    }
    if (dt > 0 && !branch.end && t === time.frames - 1) branch.end = "movie";
  } finally {
    track.running--;
  }
  await Promise.all(splits);
}

/** chunkmirage.tracking.mother_of: if a branch lost at frame `lost` had collapsed first, its
 * record at its largest in the dozen frames before (the nucleus that went into mitosis). */
function motherOf(branch: Branch, lost: number): Found | null {
  const recent = [...branch.frames.entries()].filter(([t]) => lost - 12 <= t && t < lost).sort((a, b) => a[0] - b[0]).map(([, f]) => f);
  if (!recent.length) return null;
  const biggest = recent.reduce((m, f) => (f.volume > m.volume ? f : m));
  return recent[recent.length - 1].volume <= DIVIDED * biggest.volume ? biggest : null;
}

/** chunkmirage.tracking.daughters: up to two new nuclei near `mother`, from frame `lost` on,
 * each followed on as a branch of its own: likely daughters, a guess from where and when
 * they appear (the segmentation does not say which nucleus a new one came from). */
async function daughters(track: Track, branch: Branch, mother: Found, lost: number) {
  const shape = level().shape, vx = voxel(), dtype = engine.sources.seg.dtype;
  const reach = vx.map((v) => Math.ceil(WITHIN / v)), c = mother.centroid.map(Math.round);
  const lo = c.map((v, a) => Math.max(v - reach[a], 0)), hi = c.map((v, a) => Math.min(v + reach[a] + 1, shape[a]));
  const from = Math.max(...branch.frames.keys());
  let found = 0;
  const followed: Promise<void>[] = [];
  for (let t = lost; t < Math.min(lost + GAP, time.frames) && found < 2 && !track.dropped; t++) {
    const [prev, next] = await Promise.all([read(t - 1, lo, hi), read(t, lo, hi)]);
    const born = await engine.call<Found[]>("track_newborns", {
      start: lo, voxel: vx, dtype, shape: hi.map((h, a) => h - lo[a]), centre: mother.centroid, within: WITHIN, min_volume: 0.1 * mother.volume,
    }, [prev, next]);
    for (const d of born) {
      if (found >= 2 || track.branches.length >= MAX_BRANCHES || track.dropped) break;
      found++;
      d.divided = true;  // a likely daughter's first frame: a dot on the graph
      branch.end = "divided";
      const child: Branch = { frames: new Map([[t, d]]), from };
      track.branches.push(child);
      draw();
      followed.push(along(track, child, d, t, 1));
    }
  }
  await Promise.all(followed);
}

function drop(track: Track) {
  track.dropped = true;
  tracks.splice(tracks.indexOf(track), 1);
  draw();
  showCounts();
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
  const all = tracks.flatMap((k) => k.branches.flatMap((b) => [...b.frames.values()].map((f) => f.volume)));
  const vmax = Math.max(100, ...all) * 1.1;
  const x = (t: number) => L + ((W - L - 6) * t) / Math.max(time.frames - 1, 1);
  const y = (v: number) => H - B - ((H - B - 8) * v) / vmax;
  const hours = (time.frames * time.minutes) / 60, parts: string[] = [];
  for (let h = 0; h <= hours; h += 12) parts.push(`<line x1="${x((h * 60) / time.minutes)}" x2="${x((h * 60) / time.minutes)}" y1="${H - B}" y2="${H - B + 3}" stroke="#5d6675"/><text x="${x((h * 60) / time.minutes)}" y="${H - 4}" text-anchor="middle">${h} h</text>`);
  for (const v of [0, vmax / 2, vmax].map((v) => Math.round(v / 100) * 100)) parts.push(`<text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v}</text>`);
  if (shown >= 0) parts.push(`<line x1="${x(shown)}" x2="${x(shown)}" y1="8" y2="${H - B}" stroke="#7ab0ff" stroke-opacity="0.6"/>`);
  const lines: string[] = [];
  for (const k of tracks) {
    for (const br of k.branches) {
      const pts = [...br.frames.entries()].sort((a, b) => a[0] - b[0]);
      // a daughter's line starts at its mother's last point before they split
      let path = "", prev = -2;
      if (br.from !== undefined && pts.length) {  // across the mitosis, from the mother's last frame
        const mother = k.branches.find((m) => m !== br && m.frames.has(br.from!));
        const m = mother?.frames.get(br.from);
        if (m) lines.push(`<path d="M${x(br.from).toFixed(1)},${y(m.volume).toFixed(1)}L${x(pts[0][0]).toFixed(1)},${y(pts[0][1].volume).toFixed(1)}" stroke="${k.colour}" stroke-dasharray="2 2" fill="none"/>`);
      }
      for (const [t, f] of pts) { path += `${t === prev + 1 ? "L" : "M"}${x(t).toFixed(1)},${y(f.volume).toFixed(1)}`; prev = t; }
      lines.push(`<path d="${path}" fill="none" stroke="${k.colour}" stroke-width="1.5"${br.from !== undefined ? ' stroke-opacity="0.8"' : ""}/>`);
      for (const [t, f] of pts) if (f.divided) lines.push(`<circle cx="${x(t)}" cy="${y(f.volume)}" r="2.5" fill="${k.colour}"/>`);
      if ((br.end === "lost" || br.end === "mitosis") && pts.length) {  // where a line ends before the movie does
        const [t, f] = pts[pts.length - 1], cx = x(t), cy = y(f.volume);
        lines.push(`<path d="M${cx - 3},${cy - 3}L${cx + 3},${cy + 3}M${cx - 3},${cy + 3}L${cx + 3},${cy - 3}" stroke="${k.colour}" stroke-width="1.5"/>`);
      }
    }
  }
  svg.innerHTML = `<line x1="${L}" x2="${W - 6}" y1="${H - B}" y2="${H - B}" stroke="#5d6675"/><text x="${L + 4}" y="14">µm³</text>` + parts.join("") + lines.join("");
  $("summary").innerHTML = tracks.map((k) => {
    const frames = k.branches.flatMap((b) => [...b.frames.keys()]), lo = Math.min(...frames), hi = Math.max(...frames);
    const ends = { lost: 0, mitosis: 0 };
    for (const b of k.branches) if (b.end === "lost" || b.end === "mitosis") ends[b.end]++;
    const how = [ends.mitosis ? `${ends.mitosis} went into mitosis but no new nuclei appeared near it (×)` : "",
      ends.lost ? `${ends.lost} lost, the segmentation missing it (×)` : ""].filter(Boolean).join("; ");
    return `<span style="color:${k.colour}">●</span> nucleus ${k.picked.label} (frame ${k.picked.t}): frames ${lo}–${hi}, ${((hi - lo) * time.minutes / 60).toFixed(1)} h`
      + (k.branches.length > 1 ? `, ${k.branches.length} nuclei (${k.branches.length - 1} likely daughters, dots)` : "")
      + (how ? `; ${how}` : "") + (k.running ? " …" : "");
  }).join("<br>");
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

let shownIds = new Set<string>();  // the segments the page last showed
let updating = false;  // the page is changing the segments shown (which tells this again)
/** The nuclei followed, in the frame on screen: track and id. */
function inFrame(t: number): { track: Track; found: Found }[] {
  return tracks.flatMap((k) => k.branches.flatMap((b) => (b.frames.has(t) ? [{ track: k, found: b.frames.get(t)! }] : [])));
}

/** Show the nuclei followed in the frame on screen (their ids there, in their tracks'
 * colours); a nucleus double-clicked starts a track, one followed and double-clicked again
 * drops its track. `force`: the tracks changed, not the viewer. */
function highlight(force: boolean | Event = false) {
  const v = viewer(), layer = v?.layerManager?.getLayerByName("nuclei")?.layer;
  if (!layer || updating) return;
  const group = layer.displayState.segmentationGroupState.value, visible = group.visibleSegments;
  const ids = new Set<string>([...visible].map(String)), t = frameShown();
  if (t !== shown) force = true;
  else if (force !== true) {  // the viewer's segments changed: what did the user do?
    const added = [...ids].filter((id) => !shownIds.has(id)), removed = [...shownIds].filter((id) => !ids.has(id));
    for (const id of removed) { const hit = inFrame(t).find((h) => String(h.found.label) === id); if (hit) drop(hit.track); }
    for (const id of added) void follow(t, Number(id));
  }
  const here = inFrame(t), want = new Set(here.map((h) => String(h.found.label)));
  updating = true;
  try {
    // the layer's segment list too: only this frame's nuclei followed, not every id ever shown
    for (const id of [...group.selectedSegments].map(String)) if (!want.has(id)) group.selectedSegments.delete(BigInt(id));
    for (const id of ids) if (!want.has(id)) visible.delete(BigInt(id));
    for (const id of want) if (!ids.has(id)) visible.add(BigInt(id));
    const colours = layer.displayState.segmentStatedColors?.value;
    if (colours) {  // ids are numbered afresh each frame: this frame's colours only
      colours.clear();
      for (const h of here) colours.set(BigInt(h.found.label), BigInt(parseInt(h.track.colour.slice(1), 16)));
    }
  } catch (e) {
    console.warn("highlighting the nuclei followed:", e);
  } finally {
    updating = false;
  }
  shownIds = want;
  if (t !== shown) {
    shown = t;
    draw();
    const last = here.filter((h) => h.track === tracks[tracks.length - 1]);
    if (last.length && $<HTMLInputElement>("centre").checked) centreOn(last[0].found);
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
        selectedAlpha: 0.45, notSelectedAlpha: 0,  // each shown in its track's colour (highlight)
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
  shownIds = new Set([String(START.label)]);
  const attach = () => {
    const v = viewer();
    if (!v?.position || !v.layerManager?.getLayerByName("nuclei")) return void setTimeout(attach, 300);
    v.position.changed.add(() => highlight());
    v.layerManager.getLayerByName("nuclei").layer.displayState.segmentationGroupState.value.visibleSegments.changed.add(() => highlight());
  };
  attach();
  $("command").textContent = `uv run python examples/track_nucleus.py --frame ${START.t} --label ${START.label}`;
  void follow(START.t, START.label);
  Object.assign(window, { engine, tracks });  // for a console, and the headless checks
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
$("graph").addEventListener("click", (e) => {  // jump to the frame clicked
  const svg = $<SVGSVGElement>("graph"), r = svg.getBoundingClientRect(), L = 34;
  const t = Math.round(((((e as MouseEvent).clientX - r.left) / r.width) * 320 - L) / (320 - L - 6) * (time.frames - 1));
  const v = viewer(), i = v?.coordinateSpace.value.names.indexOf("t");
  if (v && i >= 0 && t >= 0 && t < time.frames) { const p = Float32Array.from(v.position.value); p[i] = t; v.position.value = p; }
});
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
