// The registration page: reads the two images, finds an affine if none is given, solves the
// field on the GPU (solver.ts), and serves the before, after and field views to Neuroglancer
// through the service worker (sw.ts) and the chunk workers (chunks.ts). Form defaults and the
// "same in Python" command come from chunkmirage's RegisterParams, via its JSON Schema.
import { affineDistance, findAffine, halve, type FoundAffine } from "./affine";
import { Blocks, normalize } from "./blocks";
import { Cancelled, Claim, queue } from "./demand";
import type { RegisterParams } from "./generated/chunkmirage";
import schema from "./generated/chunkmirage.schema.json";
import { nearestLevel, openImage, percentiles, prod, readLevel, type Image, type Numbers } from "./ome";
import { gpu, solve, type Settings } from "./solver";
import type { Affine, ControlGrid, FromWorker, Later, Reply, StoreStats, ToWorker, ViewKind, Volume } from "./types";

const CHUNK = [16, 128, 128];    // chunks of the registered volume, z, y, x
const MAX_VOXELS = 1 << 22;      // finest automatic level: the whole level sits on the GPU
const MIN_SIZE = 16;             // coarsest automatic level: this many voxels on every axis
const TO_METRES: Record<string, number> = { micrometer: 1e-6, nanometer: 1e-9, millimeter: 1e-3, meter: 1, angstrom: 1e-10 };
const TO_SECONDS: Record<string, number> = { millisecond: 1e-3, second: 1, microsecond: 1e-6 };
const SHORT: Record<string, string> = { micrometer: "µm", nanometer: "nm", millimeter: "mm", meter: "m", angstrom: "Å" };
// this page's share of /virtual/ (getRandomValues: randomUUID exists on secure pages only)
const PAGE = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");

type FieldName = "fixed" | "moving" | "fixed_channel" | "moving_channel" | "affine" | "mirrored" | "levels" | "iterations" | "smooth" | "grid" | "window" | "refine" | "halo" | "block";
type Form = Record<FieldName, string>;
interface Ranges { fixed: number[]; moving: number[]; field?: number }
interface Sources { before?: string; after?: string; field?: string }
interface Session {
  fixed: Image; moving: Image; pool: Pool; views: Map<string, ViewKind>;
  blocks: Map<string, Map<number, Blocks>>;  // per view, the levels whose field is fitted where it is viewed
  before?: { id: string; key: string; url: string };  // the moving image placed by the affine alone
  sources?: Sources;  // what the viewer shows
  ranges?: Ranges; rangesKey?: string; shown?: boolean;
}
interface Channels { fixedChannel: number; movingChannel: number }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("form");
const input = (name: FieldName) => form.elements.namedItem(name) as HTMLInputElement | HTMLTextAreaElement;
// the form's values; a checkbox reads "true" when ticked, "" when not, as a link carries it
const values = () => ({ ...Object.fromEntries(new FormData(form)), mirrored: String((input("mirrored") as HTMLInputElement).checked) }) as Form;

// RegisterParams' properties: the form's defaults and lower bounds
const REGISTER = schema.$defs.RegisterParams.properties as Record<string,
  { default?: unknown; minimum?: number; exclusiveMinimum?: number; items?: { minimum?: number } }>;

let session: Session | null = null;  // images, chunk workers and published views
let solves = 0, previews = 0;
let previewing: Promise<void> = Promise.resolve(), previewTimer: ReturnType<typeof setTimeout> | undefined;
let exampleRef: { fixed: string; moving: string; affine: Affine } | null = null;  // the example's published affine
const INSECURE = window.isSecureContext ? null :
  `${location.origin} is not a secure page to this browser, so it allows neither WebGPU nor the viewer's service worker. `
  + "Open it over https, or as localhost (a forwarded port), or in Chrome add this address to "
  + "chrome://flags/#unsafely-treat-insecure-origin-as-secure and relaunch.";

function copyOnClick(link: HTMLElement, text: () => string, label: string) {
  link.addEventListener("click", async (e) => {
    e.preventDefault();
    if (!navigator.clipboard) return void window.prompt("Copy this:", text());  // insecure pages
    await navigator.clipboard.writeText(text());
    link.textContent = "Copied";
    setTimeout(() => (link.textContent = label), 1500);
  });
}

// ------------------------------------------------ the same in Python
/** A register:// query from its parameters, leaving out what matches the defaults. Values
 * are escaped only where the URL needs it: & ? # % + and spaces. */
function toQuery(p: RegisterParams): string {
  const esc = (v: string) => v.replace(/[%&+#? ]/g, encodeURIComponent);
  return Object.entries(p).flatMap(([k, v]) =>
    v == null || JSON.stringify(v) === JSON.stringify(REGISTER[k]?.default) ? []
      : [`${k}=${esc(Array.isArray(v) ? v.join(",") : String(v))}`]).join("&");
}

/** The form's registration as a chunkmirage command, served to any viewer or library. After a
 * run it names the levels the page solved on: automatic levels differ, since Python allows
 * bigger ones (a server GPU has more memory than a browser gets). */
const numbers = (text: string) => text.split(",").map((v) => Number(v.trim())).filter((v) => !Number.isNaN(v));

function showCommand(solvedLevels?: number[]) {
  const f = values();
  let affine = "auto";  // Python finds one as this page does, until this page has
  try { if (f.affine.trim()) affine = parseAffine(f.affine).flat().map((v) => +v.toPrecision(6)).join(","); } catch { /* shown once fixed */ }
  const params: RegisterParams = {
    fixed: f.fixed, affine, mirrored: f.mirrored === "true" && affine === "auto",
    fixed_channel: Number(f.fixed_channel), moving_channel: Number(f.moving_channel),
    levels: f.levels.trim() ? numbers(f.levels) : solvedLevels ?? null,
    iterations: numbers(f.iterations), smooth: Number(f.smooth), grid: Number(f.grid),
    window: numbers(f.window) as RegisterParams["window"], refine: Number(f.refine), halo: Number(f.halo),
    block: numbers(f.block) as RegisterParams["block"],
  };
  $("pyCommand").textContent = f.fixed && f.moving
    ? `chunkmirage serve 'register://${f.moving}?${toQuery(params)}'` : "chunkmirage serve 'register://<moving>?fixed=<fixed>'";
}

/** `values` for every level, refined ones included: one for all, one per solved level (the
 * refined ones take the last) or one per level, as register.py's per_level. */
function perLevel(name: string, values: number[], solved: number, all: number): number[] {
  if (![1, solved, all].includes(values.length)) {
    const counts = [...new Set([1, solved, all])].sort((a, b) => a - b).join(" or ");
    throw new Error(`${name} needs ${counts} values: one, one per level, or one per level with the refined ones`);
  }
  const v = values.length === 1 ? Array(all).fill(values[0]) : values;
  return [...v, ...Array(all - v.length).fill(v[v.length - 1])];
}

// ------------------------------------------------ the example
/** The images in data/example.json, if this copy of the page has one (fetch_example.py). */
async function loadExample() {
  const url = new URL("data/example.json", location.href);
  let ex: { fixed: string; moving: string; about: string; source: string; published_affine?: Affine; published_about?: string };
  try {
    const r = await fetch(url);
    if (!r.ok) return;
    ex = await r.json();
  } catch { return; }
  input("fixed").value = new URL(ex.fixed, url).href;
  input("moving").value = new URL(ex.moving, url).href;
  const note = $("example"), about = document.createElement("p"), link = document.createElement("a");
  link.href = ex.source; link.textContent = "source";
  about.append(`${ex.about} (`, link, ")");
  note.append(about);
  if (ex.published_affine) {  // start as stored (Register finds the affine) or from the published one
    const ref = { fixed: input("fixed").value, moving: input("moving").value, affine: ex.published_affine };
    exampleRef = ref;
    const start = document.createElement("div");
    start.className = "start";
    start.innerHTML = `<span>Start from</span>
      <label><input type="radio" name="start" value="stored" checked><span>the images as stored: Register finds the affine</span></label>
      <label><input type="radio" name="start" value="published"><span></span></label>`;
    start.querySelector("[value=published] + span")!.textContent = ex.published_about ?? "the published affine";
    const choose = (which: string) => {
      start.querySelector<HTMLInputElement>(`[value=${which}]`)!.checked = true;
      input("affine").value = which === "published" ? formatAffine(ref.affine) : "";
      showCommand();
    };
    start.addEventListener("change", (e) => { choose((e.target as HTMLInputElement).value); schedulePreview(); });
    if (query.get("start") === "published") choose("published");
    input("affine").addEventListener("input", uncheckStart);  // an affine of one's own
    note.append(start);
  }
  const own = document.createElement("p");
  own.textContent = "Or enter your own images.";
  note.append(own);
  note.hidden = false;
  $("empty").querySelector("b")!.textContent = "Loading the example…";
}

function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => { previewing = previewing.then(preview); }, 300);
}

/** The images before a solve: the moving image placed by the affine alone, shown once both
 * URLs are in and again when they, the channels or the affine change. */
async function preview() {
  const f = values();
  if (INSECURE || !f.fixed || !f.moving || $<HTMLButtonElement>("go").disabled) return;
  const p = channels(f);
  try {
    const affine = parseAffine(f.affine);
    const [fixed, moving] = await images(f);
    const s = await startServing(fixed, moving);
    const key = `${p.fixedChannel},${p.movingChannel}`;
    if (!s.ranges || s.rangesKey !== key) {  // contrast from the level the solve starts on
      const i = defaultLevels(fixed)[0], j = nearestLevel(moving, fixed.levels[i].voxel);
      const [fl, ml] = await Promise.all([readLevel(fixed, i, p.fixedChannel), readLevel(moving, j, p.movingChannel)]);
      s.ranges = { fixed: contrast(fl.data), moving: contrast(ml.data) };
      s.rangesKey = key;
    }
    const before = beforeView(s, affine);
    retain(s, [s.before!.id]);
    showViewer(s, viewerState(s, p, { before }));
    $("beforeCaption").textContent = beforeCaption(f.affine);
    $("afterCaption").textContent = "After: press Register";
    $("fieldCaption").textContent = "Field: press Register";
    $("error").hidden = true;
  } catch (e) {
    console.error(e);
    $("status").hidden = false; $("error").hidden = false; $("error").textContent = (e as Error).message ?? String(e);
  }
}

// ------------------------------------------------ chunk workers, and the service worker
type FieldRequest = Extract<FromWorker, { type: "field" }>;

class Pool {
  private pending = new Map<number, { resolve: (b: ArrayBuffer) => void; reject: (e: Error) => void }>();
  private seq = 0;
  private workers: Worker[];
  private starting = new Map<Worker, { resolve: () => void; reject: (e: Error) => void }>();

  constructor(n: number, onField: (w: Worker, m: FieldRequest) => void) {
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./chunks.ts", import.meta.url), { type: "module" });
      w.onmessage = ({ data: m }: MessageEvent<FromWorker>) => {
        if (m.type === "ready") return this.starting.get(w)?.resolve();
        if (m.type === "field") return onField(w, m);
        if (m.type === "error" && m.reqId === undefined) return this.starting.get(w)?.reject(new Error(m.message));
        const p = m.reqId === undefined ? undefined : this.pending.get(m.reqId);
        if (!p) return;
        this.pending.delete(m.reqId!);
        if (m.type === "error") p.reject(new Error(m.message)); else if (m.type === "chunk") p.resolve(m.body);
      };
      return w;
    });
  }
  setup(msg: ToWorker) {
    return Promise.all(this.workers.map((w) => new Promise<void>((resolve, reject) => {
      this.starting.set(w, { resolve, reject });
      w.onerror = (e) => reject(new Error(e.message || "worker failed to start"));
      w.postMessage(msg);
    })));
  }
  broadcast(msg: ToWorker) { for (const w of this.workers) w.postMessage(msg); }
  /** A chunk computed by a worker, and the request's id (its field requests carry it). */
  chunk(msg: Omit<Extract<ToWorker, { type: "chunk" }>, "reqId">, key: number): { reqId: number; body: Promise<ArrayBuffer> } {
    // neighbouring chunks go to one worker, which then reuses its reads
    const reqId = ++this.seq, w = this.workers[key % this.workers.length];
    return { reqId, body: new Promise((resolve, reject) => { this.pending.set(reqId, { resolve, reject }); w.postMessage({ ...msg, reqId }); }) };
  }
  terminate() { for (const w of this.workers) w.terminate(); }
}

navigator.serviceWorker?.addEventListener("message", async (e: MessageEvent<{ path: string }>) => {
  const port = e.ports[0];
  if (!port) return;
  let a: Answer;
  try { a = await answer(e.data.path); } catch (err) { a = { status: 500, body: String(err), type: "text/plain" }; }
  if (!a || !("pending" in a)) return port.postMessage(a, a && "body" in a && a.body instanceof ArrayBuffer ? [a.body] : []);
  // a chunk: the head now, the body once computed; the client may give up in between
  port.onmessage = (m: MessageEvent<{ cancel?: boolean }>) => { if (m.data?.cancel) a.claim.cancel(); };
  port.postMessage({ status: a.status, type: a.type, stream: true } satisfies Reply);
  a.pending.then(
    (body) => port.postMessage({ body } satisfies Later, [body]),
    (err) => port.postMessage({ error: String((err as Error)?.message ?? err) } satisfies Later),
  ).finally(() => port.close());
});

/** A reply now, or a chunk being computed for a request holding `claim` on the blocks it needs. */
type Answer = Reply | { status: number; type: string; pending: Promise<ArrayBuffer>; claim: Claim };
const claims = new Map<number, Claim>();  // chunk request id -> its claim, while it is computed
let cancelled = 0;  // chunk requests their client gave up on before they were answered

const asJson = (o: object): Reply => ({ status: 200, body: JSON.stringify(o), type: "application/json" });

/** A worker's request for the field over a window of a refined level's lattice: fitted
 * block by block here, on the GPU, and sent back. */
async function fieldRequest(w: Worker, m: FieldRequest) {
  const blocks = session?.blocks.get(m.id)?.get(m.level);
  const reply = (grid: ControlGrid | null, error?: string) =>
    w.postMessage({ type: "field", reqId: m.reqId, grid, error } satisfies ToWorker, grid ? [grid.values.buffer] : []);
  if (!blocks) return reply(null, `no fitted field for view ${m.id}, level ${m.level}`);
  try { reply(await blocks.window(m.lo, m.hi, claims.get(m.chunk))); } catch (e) {
    if (!(e instanceof Cancelled)) console.error(e);
    reply(null, String((e as Error).message ?? e));
  }
}

/** The service worker's store cache counters, or null without one. */
function storeStats(): Promise<StoreStats | null> {
  const sw = navigator.serviceWorker?.controller;
  if (!sw) return Promise.resolve(null);
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = (e: MessageEvent<StoreStats>) => resolve(e.data);
    setTimeout(() => resolve(null), 1000);
    sw.postMessage({ type: "store-stats" }, [ch.port2]);
  });
}

let blocksTimer: ReturnType<typeof setInterval> | undefined;
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** What the fits where the viewer looks are doing: fitting and waiting now, fitted so far per
 * level and the GPU time, and how much of what was read came from the store cache. Kept
 * current while there is work. */
let showing = false, again = false;  // one refresh at a time; calls meanwhile make one more
async function showBlocks(s: Session) {
  if (showing) { again = true; return; }
  showing = true;
  try { await refreshBlocks(s); } finally { showing = false; }
  if (again) { again = false; void showBlocks(s); }
}

async function refreshBlocks(s: Session) {
  const all = [...new Set([...s.blocks.values()].flatMap((m) => [...m.values()]))];  // views share their levels' blocks
  const n = all.reduce((k, b) => k + b.fitted, 0), secs = all.reduce((k, b) => k + b.seconds, 0);
  const levels = (m: Map<number, number>) => [...m].sort((a, b) => a[0] - b[0]).map(([l, k]) => `level ${l}: ${k}`).join(", ");
  const per = new Map(all.filter((b) => b.fitted).map((b) => [b.lattice.level, b.fitted]));
  const { running, waiting } = queue.byLevel();
  const now = queue.running + queue.queued
    ? `Fitting ${queue.running}${queue.running ? ` (${levels(running)})` : ""}, waiting ${queue.queued}${queue.queued ? ` (${levels(waiting)})` : ""}. ` : "";
  const gone = queue.dropped || cancelled ? `Dropped ${plural(queue.dropped, "block")} the viewer stopped waiting for (${plural(cancelled, "chunk")} given up on). ` : "";
  const done = n ? `Fitted ${plural(n, "block")} where you looked (${levels(per)}) in ${secs.toFixed(1)} s on the GPU.` : "Nothing fitted yet: zoom in past the solved levels.";
  const st = await storeStats();
  const reads = st && st.requests ? ` Images: ${(st.fetchedBytes / 2 ** 20).toFixed(0)} MB fetched, ${Math.round((100 * st.hits) / st.requests)}% of ${st.requests} reads from the cache.` : "";
  const busy = queue.running + queue.queued;
  $("blocks").hidden = !all.length;
  $("blocks").textContent = now + gone + done + reads;
  if (busy && !blocksTimer) blocksTimer = setInterval(() => void showBlocks(s), 1000);
  if (!busy && blocksTimer) { clearInterval(blocksTimer); blocksTimer = undefined; }
}

async function answer(path: string): Promise<Answer> {  // a request under /virtual/<page>/<view>/..., or null if not ours
  const parts = path.split("/virtual/")[1]?.split("/");
  const s = session;
  if (!parts || parts[0] !== PAGE || !s) return null;
  const [, view, ...rest] = parts;
  const notFound = { status: 404, body: "", type: "text/plain" };
  const kind = s.views.get(view);
  if (!kind) return notFound;
  if (rest.length === 1 && rest[0] === "zarr.json") return asJson(groupMetadata(s, view, kind));
  if (rest.length === 2 && rest[1] === "zarr.json") return asJson(arrayMetadata(s, Number(rest[0]), kind));
  if (rest[1] === "c") {
    const level = Number(rest[0]), idx = rest.slice(2).map(Number);
    const n = kind === "field" ? 1 : s.moving.lead, c = s.moving.names.indexOf("c");
    const channel = kind === "field" || c < 0 ? 0 : idx[c], index = idx.slice(n);
    const key = ((((level * 7 + channel) * 131 + (index[0] >> 2)) * 131 + index[1]) * 131 + index[2]) >>> 0;
    const { reqId, body } = s.pool.chunk({ type: "chunk", id: view, level, channel, index }, key);
    const claim = new Claim();
    claims.set(reqId, claim);
    claim.onCancel(() => { if (claims.delete(reqId)) { cancelled++; void showBlocks(s); } });
    const pending = body.finally(() => claims.delete(reqId));
    pending.catch(() => {});  // a cancelled chunk's failure is nobody's concern
    return { status: 200, type: "application/octet-stream", pending, claim };
  }
  return notFound;
}

// The registered volume as OME-Zarr 0.5: the moving image's time and channel axes on the
// fixed image's grid and levels; uncompressed chunks, since they are computed here. A field
// view is the displacement instead: three components (z, y, x) on the same grid.
function groupMetadata({ fixed, moving }: Session, view: string, kind: ViewKind) {
  // the leading axes: the field's components, or the moving image's time and channel
  const field = kind === "field", m0 = moving.levels[0], n = moving.lead;
  const axes = field ? [{ name: "c", type: "channel" }] : moving.axes.slice(0, n);
  const scale = field ? [1] : m0.scale.slice(0, n), shift = field ? [0] : m0.shift.slice(0, n);
  return {
    zarr_format: 3, node_type: "group",
    attributes: { ome: { version: "0.5", multiscales: [{
      name: view, axes: [...axes, ...fixed.axes.slice(-3)],
      datasets: fixed.levels.map((l, i) => ({ path: String(i), coordinateTransformations: [
        { type: "scale", scale: [...scale, ...l.voxel] }, { type: "translation", translation: [...shift, ...l.origin] },
      ] })),
    }] } },
  };
}

function arrayMetadata({ fixed, moving }: Session, level: number, kind: ViewKind) {
  const field = kind === "field";
  const lead = field ? [3] : moving.levels[0].fullShape.slice(0, moving.lead);
  return {
    zarr_format: 3, node_type: "array", shape: [...lead, ...fixed.levels[level].shape],
    data_type: field ? "float32" : moving.dtype, fill_value: 0,
    chunk_grid: { name: "regular", configuration: { chunk_shape: [...(field ? [3] : lead.map(() => 1)), ...CHUNK] } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    dimension_names: [...(field ? ["c"] : moving.names.slice(0, moving.lead)), "z", "y", "x"], attributes: {},
  };
}

async function startServing(fixed: Image, moving: Image): Promise<Session> {
  if (!navigator.serviceWorker) throw new Error("this browser has no service workers (a private window?)");
  try {
    await navigator.serviceWorker.register("sw.js", { scope: "./" });
  } catch (e) {
    const local = ["localhost", "127.0.0.1"].includes(location.hostname);
    const message = (e as Error).message;
    throw new Error(local ? message : `${message}\n\nBrowsers only run the viewer's service worker on a page whose certificate `
      + "they trust, and clicking through the warning is not enough. Either trust this server's certificate "
      + `(download it from ${location.origin}/certificate.crt; on a Mac, open it in Keychain Access and set it to Always Trust), `
      + "or open the page through localhost: forward this server's port to your computer (VS Code's Ports panel, "
      + "or ssh -L) and run serve.py with --host 127.0.0.1. Or skip certificates: run serve.py with --no-https "
      + "and, in Chrome, add its address to chrome://flags/#unsafely-treat-insecure-origin-as-secure.");
  }
  await navigator.serviceWorker.ready;
  if (session && session.fixed.url === fixed.url && session.moving.url === moving.url) return session;
  session?.pool.terminate();
  const pool = new Pool(Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 2)), fieldRequest);
  await pool.setup({
    type: "setup", moving: moving.url, chunkShape: CHUNK,
    fixedLevels: fixed.levels.map((l) => ({ shape: l.shape, voxel: l.voxel, origin: l.origin })),
  });
  session = { fixed, moving, pool, views: new Map(), blocks: new Map() };
  return session;
}

function publish(s: Session, id: string, affine: Affine, grid: ControlGrid | null, kind: ViewKind = "image", refined: Map<number, Blocks> | null = null): string {
  s.pool.broadcast({ type: "view", id, kind, affine, grid, refined: refined ? [...refined.values()].map((b) => b.lattice) : [] });
  s.views.set(id, kind);
  if (refined) s.blocks.set(id, refined);
  return `zarr3://${new URL(`virtual/${PAGE}/${id}/`, location.href).href}`;
}

/** The moving image placed by `affine` alone, published once per affine: a fresh URL for
 * the same view would have the viewer reload the whole panel for nothing. */
function beforeView(s: Session, affine: Affine): string {
  const key = JSON.stringify(affine);
  if (s.before?.key !== key) {
    const id = `before-${++previews}`;
    s.before = { id, key, url: publish(s, id, affine, null) };
  }
  return s.before.url;
}

/** Forget the views the viewer no longer shows, here and in the chunk workers. */
function retain(s: Session, keep: string[]) {
  const ids = [...s.views.keys()].filter((id) => !keep.includes(id));
  for (const id of ids) { s.views.delete(id); s.blocks.delete(id); }
  if (ids.length) s.pool.broadcast({ type: "drop", ids });
}

/** The two images, reused while the form names the same ones. */
async function images(f: Form): Promise<[Image, Image]> {
  const same = (img: Image, url: string) => img.url === url.replace(/\/+$/, "");
  if (session && same(session.fixed, f.fixed) && same(session.moving, f.moving)) return [session.fixed, session.moving];
  return Promise.all([openImage(f.fixed), openImage(f.moving)]);
}

// ------------------------------------------------ Neuroglancer (hosted on this origin, ng/)
// How far the field moved each point, as a heat map up to `scale` (physical units), or with
// `direction` its direction as colour: (z, y, x) components as (red, green, blue), grey at none.
// In 3D the brightest point along each ray is the one that moved furthest (emitIntensity; the
// image layers get theirs from their invlerp).
const MAX = { volumeRendering: "max", volumeRenderingDepthSamples: 128 };  // 3D: maximum intensity
const view3d = $<HTMLInputElement>("view3d");  // on unless the link or the visitor says
let view3dChosen = false;
const FIELD_SHADER = (scale: number) => `#uicontrol float scale slider(min=${(scale / 20).toPrecision(2)}, max=${(scale * 4).toPrecision(2)}, default=${scale.toPrecision(3)})
#uicontrol bool direction checkbox(default=false)
void main() {
  vec3 d = vec3(getDataValue(0), getDataValue(1), getDataValue(2));
  emitIntensity(clamp(length(d) / scale, 0.0, 1.0));
  if (direction) {
    emitRGB(clamp(vec3(0.5) + 0.5 * d / scale, 0.0, 1.0));
  } else {
    emitRGB(colormapJet(clamp(length(d) / scale, 0.0, 1.0)));
  }
}
`;

interface NgLayer { name: string; shaderControls?: unknown; [k: string]: unknown }
interface NgState { layers: NgLayer[]; [k: string]: unknown }
interface NgViewer { state: { toJSON(): NgState; restoreState(s: NgState): void } }

function viewerState(s: Session, p: Channels, sources: Sources): NgState {
  const { fixed, moving } = s, l0 = fixed.levels[0], ranges = s.ranges!, three = view3d.checked;
  s.sources = sources;
  const toM = TO_METRES[fixed.axes[fixed.axes.length - 1].unit ?? ""] ?? 1;
  const dims: Record<string, [number, string]> = { x: [l0.voxel[2] * toM, "m"], y: [l0.voxel[1] * toM, "m"], z: [l0.voxel[0] * toM, "m"] };
  const position = [l0.shape[2] / 2, l0.shape[1] / 2, l0.shape[0] / 2];
  const t = fixed.names.indexOf("t");
  if (t >= 0) { dims.t = [l0.scale[t] * (TO_SECONDS[fixed.axes[t].unit ?? ""] ?? 1), "s"]; position.push(0.5); }
  const shader = (colour: string) => `#uicontrol invlerp normalized\n#uicontrol vec3 colour color(default="${colour}")\nvoid main() { emitRGB(colour * normalized()); }\n`;
  const layer = (name: string, source: string, colour: string, range: number[], channel: number, img: Image): NgLayer => ({
    type: "image", name, source, blend: "additive", shader: shader(colour), shaderControls: { normalized: { range } }, ...(three ? MAX : {}),
    ...(img.names.includes("c") ? { localPosition: [channel] } : {}),
  });
  // zarr:// has the viewer detect the version (v2 or v3); the views served here are v3
  const layers = [layer("fixed", `zarr://${fixed.url}/`, "#ff4fd8", ranges.fixed, p.fixedChannel, fixed)];
  for (const name of ["before", "after"] as const) {
    const src = sources[name];
    if (src) layers.push(layer(name, src, "#45f07a", ranges.moving, p.movingChannel, moving));
  }
  if (sources.field) layers.push({
    type: "image", name: "field", shader: FIELD_SHADER(ranges.field ?? 1), ...(three ? MAX : {}),
    // its components are the shader's channels (getDataValue(0..2)): rename c' to c^
    source: { url: sources.field, transform: { outputDimensions: { "c^": [1, ""], z: dims.z, y: dims.y, x: dims.x } } },
  });
  // three columns (before, after, field), each a slice fitted to the whole x-y extent, with a
  // 3D view below when asked; the 3D views share one camera, tilted to show depth
  const main = document.querySelector("main")!;
  const w = (main.clientWidth || 1200) / 3 - 20, h = ((main.clientHeight || 800) - 90) / (three ? 2 : 1);
  const columns = [["fixed", "before"], sources.after ? ["fixed", "after"] : ["fixed"], sources.field ? ["field"] : ["fixed"]];
  const row = (layout: string) => ({ type: "row", children: columns.map((names) => ({ type: "viewer", layers: names, layout })) });
  return {
    dimensions: dims, position, displayDimensions: ["x", "y", "z"],
    crossSectionScale: Math.max(l0.shape[2] / w, l0.shape[1] / h) * 1.05,
    projectionScale: Math.max((l0.shape[2] * h) / w, l0.shape[1]) * 1.3,
    projectionOrientation: [-0.2164, 0, 0, 0.9763],  // 25 degrees about x
    crossSectionBackgroundColor: "#000000", projectionBackgroundColor: "#000000", showSlices: false, layers,
    // twice Neuroglancer's defaults: each chunk it drops costs a resample to get back
    gpuMemoryLimit: 2e9, systemMemoryLimit: 4e9,
    layout: three ? { type: "column", children: [row("xy"), row("3d")] } : row("xy"),
    selectedLayer: { layer: sources.after ? "after" : "before", visible: false },
  };
}

function showViewer(s: Session, state: NgState) {
  const ng = $<HTMLIFrameElement>("ng"), viewer = (ng.contentWindow as (Window & { viewer?: NgViewer }) | null)?.viewer;
  const keepCamera = s.shown;  // later views of the same images keep the camera
  s.shown = true;
  $("empty").hidden = true; ng.hidden = false; $("captions").hidden = false;
  if (keepCamera && viewer) {  // same origin: keep where the user is looking
    const cur = viewer.state.toJSON();
    for (const k of ["position", "crossSectionScale", "crossSectionOrientation", "projectionScale", "projectionOrientation"])
      if (cur[k] !== undefined) state[k] = cur[k];
    for (const l of state.layers) {
      const old = cur.layers?.find((o) => o.name === l.name);
      if (old?.shaderControls) l.shaderControls = old.shaderControls;  // contrast the user set
    }
    viewer.state.restoreState(state);
    return;
  }
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`;
}

// ------------------------------------------------ the run
function defaultLevels(img: Image): number[] {
  const shapes = img.levels.map((l) => l.shape);
  const fits = shapes.flatMap((s, i) => (prod(s) <= MAX_VOXELS ? [i] : []));
  const finest = fits.length ? fits[0] : shapes.length - 1;
  const coarse = shapes.flatMap((s, i) => (Math.min(...s) >= MIN_SIZE ? [i] : []));
  const coarsest = Math.max(coarse.length ? Math.max(...coarse) : finest, finest);
  return Array.from({ length: coarsest - finest + 1 }, (_, k) => coarsest - k);
}

function parseAffine(text: string): Affine {
  const v = (text || "").split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!v.length) return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
  if ((v.length !== 12 && v.length !== 16) || v.some(Number.isNaN)) throw new Error("the affine needs 12 or 16 numbers");
  return [v.slice(0, 4), v.slice(4, 8), v.slice(8, 12)];
}

function contrast(data: Numbers): number[] {  // display limits from the voxels with data, which skips padding
  const v = (data as Float32Array).filter((x) => x > 0);
  const [lo, hi] = v.length ? percentiles(v, [1, 99.8]) : [0, 1];
  return [lo, Math.max(hi, lo + 1)];
}

function channels(f: Form): Channels { return { fixedChannel: Number(f.fixed_channel), movingChannel: Number(f.moving_channel) }; }
function uncheckStart() { for (const i of document.querySelectorAll<HTMLInputElement>(".start input")) i.checked = false; }
function formatAffine(a: Affine) { return a.map((r) => r.map((v) => +v.toPrecision(6)).join(", ")).join("\n"); }  // a row per line
function beforeCaption(text: string) { return text.trim() ? "Before: placed by the affine alone" : "Before: as they are, no affine"; }

function stageRow(label: string, where: "append" | "prepend" = "append"): HTMLElement {
  const el = document.createElement("div");
  el.className = "stage";
  el.innerHTML = `<div class="row"><span>${label}</span><span class="t">waiting</span></div><div class="bar"><i></i></div><div class="sim">&nbsp;</div>`;
  $("stages")[where](el);
  return el;
}
function setRow(row: HTMLElement, sel: ".t" | ".sim", text: string) { row.querySelector(sel)!.textContent = text; }
function setBar(row: HTMLElement, fraction: number) { row.querySelector<HTMLElement>("i")!.style.width = `${100 * fraction}%`; }

async function run() {
  const go = $<HTMLButtonElement>("go");
  go.disabled = true;
  $("status").hidden = false; $("error").hidden = true; $("links").hidden = true; $("summary").textContent = "";
  const f = values(), params = new URLSearchParams(f);
  if (view3dChosen) params.set("3d", view3d.checked ? "1" : "0");
  history.replaceState(null, "", "?" + params);
  const p = channels(f);
  clearTimeout(previewTimer);
  try {
    await previewing;  // one still starting the viewer
    if (INSECURE) throw new Error(INSECURE);
    $("gpu").textContent = "Starting the GPU…";
    const gpuReady = gpu();  // made while the images are read
    gpuReady.then((g) => ($("gpu").textContent = `GPU: ${g.name || "unnamed adapter"}`), () => {});
    $("summary").textContent = "Reading the images…";
    const [fixed, moving] = await images(f);
    const unitOf = (img: Image) => img.axes[img.axes.length - 1].unit;
    if (unitOf(fixed) !== unitOf(moving)) throw new Error(`units differ: ${unitOf(fixed)} and ${unitOf(moving)}`);
    let affine = parseAffine(f.affine);
    const auto = !f.affine.trim();  // find the affine first
    const levels = f.levels ? numbers(f.levels) : defaultLevels(fixed);
    const solved = levels[levels.length - 1], refine = Number(f.refine) || 0, halo = Number(f.halo) || 0;
    if (refine > solved) throw new Error(`refine=${refine}: ${solved} level(s) lie below level ${solved}, the finest solved`);
    const iters = perLevel("iterations", numbers(f.iterations), levels.length, levels.length + refine);
    const windows = perLevel("window", numbers(f.window), levels.length, levels.length + refine);
    if (windows.some((w) => w < 1 || w % 2 === 0)) throw new Error(`window must be positive odd numbers, got ${windows}`);
    const block = numbers(f.block);
    if (block.length !== 3 || block.some((b) => !(b >= 1))) throw new Error(`block needs three sizes (z, y, x), got ${f.block}`);
    const setting = (k: number): Settings => ({ iterations: iters[k], window: windows[k], smooth: Number(f.smooth), grid: Number(f.grid) });
    const pairs = levels.map((i) => [i, nearestLevel(moving, fixed.levels[i].voxel)]);
    const tRead = performance.now();
    const [s, read] = await Promise.all([  // the chunk workers start while the levels are read
      startServing(fixed, moving),
      Promise.all(pairs.map(([i, j]) => Promise.all([readLevel(fixed, i, p.fixedChannel), readLevel(moving, j, p.movingChannel)]))),
    ]);
    const readSecs = (performance.now() - tRead) / 1000;
    const before = beforeView(s, affine);
    retain(s, [s.before!.id]);
    const ranges: Ranges = { fixed: contrast(read[0][0].data), moving: contrast(read[0][1].data) };
    s.ranges = ranges;
    s.rangesKey = `${p.fixedChannel},${p.movingChannel}`;
    showViewer(s, viewerState(s, p, { before }));
    $("beforeCaption").textContent = beforeCaption(f.affine);
    $("afterCaption").textContent = "After: solving…";
    $("fieldCaption").textContent = "Field: solving…";

    const fr = percentiles(read[0][0].data, [0.5, 99.5]), mr = percentiles(read[0][1].data, [0.5, 99.5]);
    const raw = ({ data: d, shape, voxel, origin }: typeof read[0][0]): Volume => ({ norm: Float32Array.from(d), shape, voxel, origin });
    const data: [Volume, Volume][] = read.map(([fl, ml]) => {
      let [f0, m0] = [raw(fl), raw(ml)];
      // a level too big for this GPU (the stored pyramid stops early): coarser copies of its own,
      // made as a pyramid is (the voxels' means) and then normalized
      while (prod(f0.shape) > MAX_VOXELS) [f0, m0] = [halve(f0), halve(m0)];
      return [{ ...f0, norm: normalize(f0.norm, fr) }, { ...m0, norm: normalize(m0.norm, mr) }];
    });
    const l0 = fixed.levels[0];
    const lo = l0.origin.map((o, a) => o - l0.voxel[a] / 2), hi = lo.map((v, a) => v + l0.shape[a] * l0.voxel[a]);
    $("stages").innerHTML = "";
    $("blocks").hidden = true;
    const rows = levels.map((lvl, k) => stageRow(`<b>Level ${lvl}</b> · ${data[k][0].shape.join("×")} voxels`
      + (data[k][0].shape.join() === read[k][0].shape.join() ? "" : " (halved to fit this GPU)")));
    const unit = SHORT[unitOf(fixed) ?? ""] ?? unitOf(fixed) ?? "";
    let found: FoundAffine | null = null;
    if (auto) {  // on the coarsest level; the field is then solved from it
      const row = stageRow("<b>Affine</b> · centres and axes, then a fit", "prepend");
      $("summary").textContent = `Read levels ${levels.join(", ")} in ${readSecs.toFixed(1)} s. Finding the affine…`;
      const fa = await findAffine(data[0][0], data[0][1], ({ stage, stages, iteration, iterations, similarity }) => {
        setBar(row, (stage + iteration / iterations) / stages);
        setRow(row, ".t", `fit ${stage + 1} / ${stages}`);
        setRow(row, ".sim", `similarity ${similarity?.toFixed(3)}`);
      }, f.mirrored === "true");
      affine = fa.affine;
      const ref = exampleRef && f.fixed === exampleRef.fixed && f.moving === exampleRef.moving ? exampleRef.affine : null;
      fa.distance = ref ? affineDistance(data[0][0], affine, ref) : null;
      setRow(row, ".t", `${fa.seconds.toFixed(1)} s`);
      setRow(row, ".sim", `similarity ${fa.identity.toFixed(3)} → ${fa.final.toFixed(3)}`
        + (fa.distance == null ? "" : ` · ${fa.distance.toFixed(1)} ${unit} from the published affine`));
      input("affine").value = formatAffine(affine);
      uncheckStart();  // the box now holds the found affine
      found = fa;
    }
    showCommand(levels);
    $("summary").textContent = `Read levels ${levels.join(", ")} in ${readSecs.toFixed(1)} s. Solving the field…`;
    const g = await gpuReady;
    const res = await solve(g, data, affine, [lo, hi], levels.map((_, k) => setting(k)), ({ stage, iteration, iterations, similarity }) => {
      setBar(rows[stage], iteration / iterations);
      setRow(rows[stage], ".t", `${iteration} / ${iterations}`);
      if (similarity != null) setRow(rows[stage], ".sim", `similarity ${similarity.toFixed(3)}`);
    });
    res.stages.forEach((st, k) => {
      setRow(rows[k], ".t", `${st.seconds.toFixed(1)} s`);
      setRow(rows[k], ".sim", `similarity ${st.first?.toFixed(3)} → ${st.final?.toFixed(3)}`);
    });
    // below the solved levels, each refined level's field is fitted block by block as the
    // viewer asks for chunks, from the solved field
    const refined = new Map<number, Blocks>();
    const pair = { fixed, moving, fixedChannel: p.fixedChannel, movingChannel: p.movingChannel, ranges: { fixed: fr, moving: mr } };
    for (let r = 0; r < refine; r++) {
      const i = solved - 1 - r;
      refined.set(i, new Blocks(g, pair, affine, res.grid, i, [lo, hi], block, setting(levels.length + r), halo, solved - i));
    }
    const afterId = `after-${++solves}`, fieldId = `field-${solves}`;
    const after = publish(s, afterId, affine, res.grid, "image", refined), field = publish(s, fieldId, affine, res.grid, "field", refined);
    retain(s, [s.before!.id, afterId, fieldId]);
    if (refine) { queue.onChange = () => void showBlocks(s); void showBlocks(s); }
    const sizes = new Float32Array(res.grid.values.length / 3);
    for (let i = 0; i < sizes.length; i++) sizes[i] = Math.hypot(res.grid.values[3 * i], res.grid.values[3 * i + 1], res.grid.values[3 * i + 2]);
    ranges.field = Math.max(percentiles(sizes, [99])[0], 1e-6);
    showViewer(s, viewerState(s, p, { before, after, field }));
    $("afterCaption").textContent = "After: the affine and the solved field";
    $("fieldCaption").textContent = `Field: how far the solve moved each point (up to ${ranges.field.toPrecision(3)} ${unit})`;
    $("summary").textContent = found
      ? `Found the affine in ${found.seconds.toFixed(1)} s, then solved the field in ${res.seconds.toFixed(1)} s on the GPU.`
      : `Solved in ${res.seconds.toFixed(1)} s on the GPU.`;
    const result = { gpu: g.name, levels, affine, found, stages: res.stages, solveSeconds: res.seconds, grid: { ...res.grid, values: Array.from(res.grid.values) } };
    const save = $<HTMLAnchorElement>("save");
    if (save.href.startsWith("blob:")) URL.revokeObjectURL(save.href);  // the last run's
    save.href = URL.createObjectURL(new Blob([JSON.stringify(result)], { type: "application/json" }));
    $("links").hidden = false;
  } catch (e) {
    console.error(e);
    $("error").hidden = false; $("error").textContent = (e as Error).message ?? String(e);
    $("summary").textContent = "";
  } finally {
    go.disabled = false;
  }
}

// ------------------------------------------------ start-up
for (const name of ["fixed_channel", "moving_channel", "iterations", "smooth", "grid", "window", "refine", "halo", "block"] as const) {
  const prop = REGISTER[name], el = input(name) as HTMLInputElement;
  el.value = String(prop.default);  // iterations' [100] shows as 100
  const min = prop.minimum ?? prop.exclusiveMinimum ?? prop.items?.minimum;
  if (min !== undefined) el.min = String(min);
}
const query = new URLSearchParams(location.search);
if (!query.has("fixed") && !query.has("moving")) await loadExample();
for (const [k, v] of query) {
  const el = form.elements.namedItem(k);
  if (el instanceof HTMLInputElement && el.type === "checkbox") el.checked = v === "true";
  else if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) el.value = v;
}
if (input("affine").value) {  // one row of the matrix per line
  const v = input("affine").value.split(/[\s,;]+/).filter(Boolean);
  if (v.length === 12 || v.length === 16) input("affine").value = [0, 4, 8].map((i) => v.slice(i, i + 4).join(", ")).join("\n");
}
form.addEventListener("submit", (e) => { e.preventDefault(); void run(); });
form.addEventListener("input", () => showCommand());
copyOnClick($("copy"), () => location.href, "Copy link to this run");
copyOnClick($("copyPy"), () => $("pyCommand").textContent ?? "", "Copy");
showCommand();
if (INSECURE) { $("status").hidden = false; $("error").hidden = false; $("error").textContent = INSECURE; }
else schedulePreview();  // a link shows its images; Register solves
for (const k of ["fixed", "moving", "fixed_channel", "moving_channel", "affine"] as const) input(k).addEventListener("change", schedulePreview);
input("mirrored").addEventListener("change", () => showCommand());
if (query.has("3d")) { view3d.checked = query.get("3d") === "1"; view3dChosen = true; }
view3d.addEventListener("change", () => {
  view3dChosen = true;
  if (session?.sources) showViewer(session, viewerState(session, channels(values()), session.sources));
});
