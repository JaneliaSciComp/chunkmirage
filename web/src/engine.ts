// The browser engine behind the pipeline and map pages: a demo's views (each a chunkmirage
// pipeline spec) served from this page through the service worker (sw.ts), as a chunkmirage
// server would serve them. Each chunk's input region is read by the page's one reader
// (reader.ts) and computed by a Pyodide worker (pyworker.ts) running chunkmirage's own ops;
// chunks a client gives up on before their turn are dropped, and computed ones are kept a
// while (a map client keeps none itself). The Pyodide workers start when a view first needs
// them (ops, a computed source, a mesh): a page whose views are read as stored (asread.ts)
// or computed by the page itself (`serve`) loads no Python, which is how a page of another
// project builds on the engine (docs/design.md). Two layouts of the same chunks: OME-Zarr 0.5 for
// Neuroglancer (virtual/<page>/<view>/...) and GeoZarr for map clients such as OpenLayers
// (virtual/<page>/geo/<view>/<level>/<view>/...), the zarr-conventions multiscales, proj:
// and spatial: attributes on the group. And zarr v2 with OME 0.4 (virtual/<page>/zarr2/<view>/),
// for readers that predate zarr v3's final spec, such as GDAL 3.8.
import { asRead, computedSource, needsPython, TYPED, wholeChunk } from "./asread";
import { Cancelled, Claim, Queue } from "./demand";
import schema from "./generated/chunkmirage.schema.json";
import { Kept } from "./kept";
import { route } from "./routes";
import { bytes, fitted, inputBox, levelAt, padded, prod, type Block, type Padding, type Stage, type StageAt } from "./stages";
import type { Answer, Ask, Later, MeshCall, PipelineView, Plan, Printed, Reply, SourceInfo, ToPyWorker, ToReader, ViewAxis, ViewInfo } from "./types";

export const TO_SECONDS: Record<string, number> = { s: 1, second: 1, millisecond: 1e-3, ms: 1e-3, minute: 60, hour: 3600, day: 86400 };
const OME_UNIT: Record<string, string> = { nm: "nanometer", um: "micrometer", m: "meter", s: "second" };
const CONVENTIONS = [
  { uuid: "d35379db-88df-4056-af3a-620245f8e347", name: "multiscales" },
  { uuid: "f17cb550-5864-4468-aeb7-f3180cfb622f", name: "proj:" },
  { uuid: "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4", name: "spatial:" },
];
const KEPT_BYTES = 256 * 2 ** 20;    // computed chunks kept for clients that refetch
const STAGED_BYTES = 512 * 2 ** 20;  // the outputs of a staged view's stages (stages.ts), kept for the stages after them
const V2_DTYPE: Record<string, string> = {
  uint8: "|u1", int8: "|i1", uint16: "<u2", int16: "<i2", uint32: "<u4", int32: "<i4", uint64: "<u8", int64: "<i8", float32: "<f4", float64: "<f8",
};
/** The Pyodide packages each op imports beyond numpy, by op name (chunkmirage's schema). */
const OP_PACKAGES: Record<string, string[]> = Object.fromEntries(Object.values(schema.$defs as Record<string, { properties?: { op?: { const?: string } }; "x-packages"?: string[] }>)
  .flatMap((d) => (d.properties?.op?.const && d["x-packages"] ? [[d.properties.op.const, d["x-packages"]]] : [])));
/** Pyodide workers to run: most of the cores, two left for the page and the reader. */
const workers = () => Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));

type Request<R> = R extends unknown ? Omit<R, "reqId"> : never;
type Answered = null | { status: number; body: string | ArrayBuffer; type: string; range?: string }
  | { status: number; type: string; pending: Promise<ArrayBuffer>; claim: Claim; range?: string };
/** A multi-resolution mesh's octree: its nodes (level, z, y, x each, in the data file's
 * order), each fragment's size, the quantization and the index file. */
type Octree = { nodes: Int32Array; size: number; bits: number; index: ArrayBuffer; mask: Uint8Array; level: number };
const BAND = 2;  // chunkmirage.meshes.BAND: the border a part of the coarsest level is masked with

/** A worker answering requests by reqId. */
class Rpc<Req extends { reqId: number }> {
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  private seq = 0;
  /** `onPrint` hears what Python printed, which comes unasked. */
  constructor(private worker: Worker, onPrint?: (p: Printed) => void) {
    worker.onmessage = ({ data: m }: MessageEvent<Answer>) => {
      if ("print" in m) { onPrint?.(m); return; }
      const p = this.pending.get(m.reqId);
      if (!p) return;
      this.pending.delete(m.reqId);
      if ("error" in m) p.reject(new Error(typeof m.error === "string" ? m.error : JSON.stringify(m.error))); else p.resolve(m.value as never);
    };
    worker.onerror = (e) => { for (const p of this.pending.values()) p.reject(new Error(e.message || "a worker failed")); this.pending.clear(); };
  }
  call<T>(msg: Request<Req>, transfer: Transferable[] = []): Promise<T> {
    const reqId = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(reqId, { resolve: resolve as (v: never) => void, reject });
      this.worker.postMessage({ ...msg, reqId }, transfer);
    });
  }
  /** Stop the worker; calls still waiting fail as `Cancelled`. */
  terminate() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Cancelled("the engine was stopped"));
    this.pending.clear();
  }
}

const asJson = (o: unknown) => ({ status: 200, body: JSON.stringify(o), type: "application/json" });
const notFound = { status: 404, body: "", type: "text/plain" };
export const isTime = (a: ViewAxis) => a.unit in TO_SECONDS;

/** `Engine.start`'s options. `packages`: what Python loads beyond numpy and the ops' own
 * imports (each op's schema says those). `python`: when the Pyodide workers start; "auto"
 * (the default) when a view first needs them, "always" at `start`, for a page that will
 * `call()` the package's Python and wants it loading while the data opens. */
export interface StartOptions { packages?: string[]; python?: "auto" | "always" }

export class Engine {
  readonly page = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  infos: Record<string, ViewInfo> = {};
  readonly counts = { computed: 0, dropped: 0, failed: 0 };
  /** What the views' Python printed, the last 200 lines, newest last. */
  readonly printed: Printed[] = [];
  private print(p: Printed) {
    this.printed.push(p);
    if (this.printed.length > 200) this.printed.shift();
    this.onChange();
  }
  /** The last chunk that failed: its view (as served, `name~n` after an edit) and why. */
  failure: { view: string; message: string } | null = null;
  private views: Record<string, PipelineView> = {};
  private reader: Rpc<ToReader> | null = null;
  private pool: Rpc<ToPyWorker>[] = [];  // the Pyodide workers, none until a view needs Python (`python`)
  private packages: string[] = [];        // what Python loads beyond numpy when it starts
  private warm: Promise<void> = Promise.resolve();  // Python loaded in every worker of the pool
  private turn = 0;
  private chunks: Queue<ArrayBuffer> | null = null;
  private inflight = new Map<string, Promise<ArrayBuffer>>();
  private kept = new Kept(KEPT_BYTES);
  private staged = new Kept(STAGED_BYTES);  // by stage key, not view: shared by views, kept past an unserve
  private edits = 0;
  private producers = new Map<string, (level: number, index: number[]) => Promise<ArrayBuffer>>();
  private unroute: (() => void) | null = null;  // removes this page's /virtual/ route (routes.ts)
  private staging = new Map<string, Promise<Block>>();  // stage outputs being made, by key: once each
  private status: ((s: string) => void) | null = null;  // start's, for what happens after it
  private stopped = false;

  /** `onChange` is told whenever the counts change. */
  constructor(private onChange: () => void = () => {}) {}

  /** Start the service worker and the reader, and open `views`' sources. A view that needs
   * Python (ops to run, a computed source, a mesh) starts the Pyodide workers too, and its
   * ops are planned once Python has loaded (`ready`); requests that come before wait for it,
   * so a viewer can start meanwhile. A view with no ops is planned at once and served from
   * the reader alone, so a page whose views need no Python never loads it; one that will
   * `call()` the package's Python asks for it with `python: "always"`. */
  async start(views: Record<string, PipelineView>, status: (s: string) => void, { packages = [], python = "auto" }: StartOptions = {}): Promise<void> {
    if (!window.isSecureContext) throw new Error("This page needs a secure context: open it over https, or through localhost.");
    if (!navigator.serviceWorker) throw new Error("This browser has no service workers (a private window?).");
    this.stopped = false;
    this.status = status;
    status("Starting the service worker…");
    await navigator.serviceWorker.register("sw.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    this.listen();
    this.reader = new Rpc<ToReader>(new Worker(new URL("./reader.ts", import.meta.url), { type: "module" }));
    this.chunks = new Queue<ArrayBuffer>(workers());
    this.packages = [...packages];
    this.wants(views);
    if (python === "always" || Object.values(views).some(needsPython)) {
      const n = this.python().length;
      status(`Opening the data, and loading Python (Pyodide, ${["numpy", ...this.packages].join(", ")}) and chunkmirage's ops in ${n} worker${n > 1 ? "s" : ""}…`);
    } else status("Opening the data…");
    await this.open(views);
    this.ready = this.warm.then(() => this.planOps(views));
    this.ready.catch(() => {});
  }

  /** Resolves once every view `start` was given is planned, and Python has loaded if it
   * was started. */
  ready: Promise<void> = Promise.resolve();
  /** The views' sources as opened: axes and levels, known before Python has loaded. */
  sources: Record<string, SourceInfo> = {};

  /** The Pyodide workers, started the first time a view needs Python (ops to run, a source
   * they compute, a mesh to make) or a `call()` does: a page whose views are read as stored
   * or computed by the page itself (`serve`) never loads it. Python loads with the packages
   * noted so far (`wants`); any an op imports unannounced, a worker loads when it first does. */
  private python(): Rpc<ToPyWorker>[] {
    if (this.stopped) throw new Error("the engine was stopped: start another");
    if (!this.pool.length) {
      this.status?.(`Loading Python (Pyodide, ${["numpy", ...this.packages].join(", ")}) and chunkmirage's ops…`);
      this.pool = Array.from({ length: workers() }, () => new Rpc<ToPyWorker>(new Worker(new URL("./pyworker.ts", import.meta.url), { type: "module" }), (p) => this.print(p)));
      this.warm = Promise.all(this.pool.map((w) => w.call({ type: "plan", views: {}, packages: this.packages }))).then(() => {});
      this.warm.catch(() => {});
    }
    return this.pool;
  }

  /** The next Pyodide worker, in turn. */
  private worker(): Rpc<ToPyWorker> {
    const pool = this.python();
    return pool[this.turn++ % pool.length];
  }

  /** Note what `views`' ops import beyond numpy (each op's schema says), and marching cubes
   * for a mesh, for Python to load when it starts. */
  private wants(views: Record<string, PipelineView>) {
    for (const v of Object.values(views)) {
      const needed = [...(v.ops ?? []).flatMap((o) => OP_PACKAGES[String(o.op)] ?? []), ...(v.mesh && v.mesh.kind !== "terrain" ? ["scikit-image"] : [])];  // (and scipy, for a multi-resolution index)
      for (const p of needed) if (!this.packages.includes(p)) this.packages.push(p);
    }
  }

  /** Open `views`' sources and plan their ops. Requests for a view whose plan is on its way
   * wait for `ready` (answer), so it follows this too. */
  private plan(views: Record<string, PipelineView>): Promise<void> {
    const planned = (async () => {
      this.wants(views);
      await this.open(views);
      await this.planOps(views);
    })();
    planned.catch(() => {});
    this.ready = this.ready.then(() => planned, () => planned);
    return planned;
  }

  /** Open `views`' sources, once each: read ones by the reader, computed ones described by
   * a worker (which waits for Python). */
  private async open(views: Record<string, PipelineView>): Promise<void> {
    const entries = Object.entries(views), read = entries.filter(([, s]) => !computedSource(s.source));
    const opened = read.length ? await this.reader!.call<Record<string, SourceInfo>>({ type: "open", views: Object.fromEntries(read) }) : {};
    for (const [v, s] of entries) if (computedSource(s.source)) opened[v] = await this.python()[0].call<SourceInfo>({ type: "describe", source: s.source });
    Object.assign(this.sources, opened);
    Object.assign(this.views, views);
  }

  /** Plan opened views' ops: a view read as stored (no ops) here, from its source alone; the
   * rest in every worker, whose plan says their output dtype, leading axes and halo. */
  private async planOps(views: Record<string, PipelineView>): Promise<void> {
    const specs: Extract<ToPyWorker, { type: "plan" }>["views"] = {};
    for (const [v, spec] of Object.entries(views)) {
      const s = this.sources[v];
      if (!needsPython(spec)) { this.planned(v, asRead(s)); continue; }
      const shape = [...(s.channels > 1 ? [s.channels] : []), ...s.levels[0].shape];
      specs[v] = { ops: spec.ops ?? [], shape, dtype: s.dtype, chunk: spec.chunk, voxel: s.levels[0].voxel, source: computedSource(spec.source) ? spec.source : undefined };
    }
    if (!Object.keys(specs).length) return;
    const plans = await Promise.all(this.python().map((w) => w.call<Record<string, Plan>>({ type: "plan", views: specs })));
    for (const v of Object.keys(specs)) this.planned(v, plans[0][v]);
  }

  /** A view as served: its source, and what plan `p` makes of it. */
  private planned(view: string, p: Plan) {
    this.infos[view] = { ...this.sources[view], out: p.dtype, kind: p.kind, lead: p.lead, halo: p.halo, added: p.added, controls: p.controls };
  }

  /** Serve `views` too: their sources opened and their ops planned, as `start`'s. The way
   * to open another source once started. */
  add(views: Record<string, PipelineView>): Promise<void> { return this.plan(views); }

  /** Serve a view whose chunks the page computes itself: `produce` gives chunk `index` of a
   * level as zarr v3 bytes (whole, little endian). The work is queued, shared and dropped as
   * the engine's own, and the chunks kept a while. */
  serve(view: string, info: ViewInfo, chunk: number[], produce: (level: number, index: number[]) => Promise<ArrayBuffer>) {
    this.infos[view] = info;
    this.sources[view] = info;
    this.views[view] = { source: "page://", chunk };
    this.producers.set(view, produce);
  }

  /** Serve a view the page computes in `stages` (stages.ts), the first of them from `from`,
   * an opened view, read at the level its `inputVoxelSize` names (else the one at the view
   * level's voxel size) and padded past the array's edge by `padding`, as the Python
   * pipeline pads. Each stage's output is kept under the stage's key (512 MB, least
   * recently used out), so a view served again with a later stage changed reuses the
   * earlier ones' outputs; `unserve` leaves them, `stop` drops them. The view's dtype and
   * channel axis are the last stage's; `serve` is the one-stage case with the page doing
   * the reading. */
  serveStages(view: string, info: ViewInfo, chunk: number[], from: string, stages: Stage[], padding: Padding = "edge") {
    if (!stages.length) throw new Error(`${view}: a staged view needs at least one stage`);
    if (!this.sources[from] || this.producers.has(from) || computedSource(this.views[from].source)) {
      throw new Error(`${view}: ${from} is not a view read from its source (one of start's or add's; not one the page computes)`);
    }
    if (stages.slice(1).some((st) => st.inputVoxelSize)) throw new Error(`${view}: only the first stage reads at its own voxel size (inputVoxelSize)`);
    if (stages[0].inputVoxelSize && info.levels.length > 1) {
      // the Python pipeline runs such a stage at its one level and downsamples its result for
      // the coarser ones; the engine does not yet, and the stage fed a coarser level's box on
      // its grid would return a block of another shape
      throw new Error(`${view}: a view whose first stage reads at its own voxel size has one level (the engine does not downsample it for coarser ones yet)`);
    }
    const last = stages[stages.length - 1];
    this.serve(view, { ...info, out: last.dtype, added: last.added ?? [] }, chunk, (level, index) => this.stagedChunk(view, from, stages, padding, level, index));
  }

  /** Chunk `index` of a staged view's level: the last stage's output over the chunk's box,
   * placed whole in the chunk (asread.ts). */
  private async stagedChunk(view: string, from: string, stages: Stage[], padding: Padding, level: number, index: number[]): Promise<ArrayBuffer> {
    const v = this.infos[view], l = v.levels[level], C = this.views[view].chunk;
    const lo = index.map((i, a) => i * C[a]), hi = lo.map((o, a) => Math.min(o + C[a], l.shape[a]));
    const out = await this.stage(from, stages, padding, stages.length - 1, { view, level, index, lo, hi, voxel: l.voxel });
    return wholeChunk(bytes(out.data), v.out, prod(out.shape.slice(0, -3)), hi.map((h, a) => h - lo[a]), C);
  }

  /** Stage `k`'s output over `at`'s box: kept under the stage's key if made before, else
   * made by the stage from its input, the stage before's output over the box grown by this
   * one's halo, or for the first stage the source read and padded. */
  private stage(from: string, stages: Stage[], padding: Padding, k: number, at: StageAt): Promise<Block> {
    // the key says what the output is: what the stage computes (its key), what it reads (the
    // source view, padded how) and where (the box, at the level's voxel size)
    const st = stages[k], halo = st.halo ?? [0, 0, 0];
    const key = `${st.key}|${from}|${padding}|${at.voxel.join(".")}|${at.lo.join(".")}|${at.hi.join(".")}`;
    const shape = [...(st.added ?? []), ...at.hi.map((h, a) => h - at.lo[a])], Typed = TYPED[st.dtype] ?? Float32Array;
    const made = this.staged.get(key);
    if (made) return Promise.resolve({ data: new Typed(made), shape });
    let making = this.staging.get(key);  // on its way for another view or chunk: made once
    if (!making) {
      making = (async () => {
        const grown = { ...at, lo: at.lo.map((o, a) => o - halo[a]), hi: at.hi.map((h, a) => h + halo[a]) };
        const input = k ? await this.stage(from, stages, padding, k - 1, grown) : await this.input(from, st, padding, at);
        const scaled = !k && !!st.inputVoxelSize;  // on another grid: the result is the output's shape, not shaved from the input's
        const block = fitted(await st.produce(input, at), st.dtype, shape, input.shape, scaled ? null : halo);
        this.staged.put(key, bytes(block.data));
        return block;
      })();
      this.staging.set(key, making);
      making.finally(() => this.staging.delete(key)).catch(() => {});
    }
    return making;
  }

  /** The first stage's input: `from`'s level at the stage's voxel size (`inputVoxelSize`,
   * else the view level's), over `at`'s box on that grid grown by the stage's halo, what
   * lies inside the array read and the rest padded. */
  private async input(from: string, st: Stage, padding: Padding, at: StageAt): Promise<Block> {
    const s = this.sources[from], want = st.inputVoxelSize ?? at.voxel, level = levelAt(s.levels, want);
    if (level < 0) throw new Error(`${at.view}: no level of ${from} is at voxel size ${want.join("×")}`);
    const sl = s.levels[level], r = at.voxel.map((v, a) => v / sl.voxel[a]);  // input voxels to each output voxel
    const { lo, hi } = inputBox(at.lo, at.hi, r, st.halo ?? [0, 0, 0]);
    const readLo = lo.map((o) => Math.max(o, 0)), readHi = hi.map((h, a) => Math.min(h, sl.shape[a]));
    const data = readHi.every((h, a) => h > readLo[a]) ? await this.read(from, level, readLo, readHi) : null;
    return padded(data, s.dtype, s.channels, readLo, readHi, lo, hi, padding);
  }

  /** Stop serving a view: its infos, producer and kept chunks are forgotten, chunks of it
   * waiting to be computed are dropped (one being computed finishes, and is not kept), and
   * requests for it answer 404 from now on. For a `serve()` view that is all of it; a view
   * of `start()` or `add()` keeps its source open in the reader and its plan in the workers,
   * both small, and loses its mesh's octree (a whole level's mask). A page that serves a
   * volume again under a new name, as viewers cache by URL,
   * unserves the old one so its chunks make room. Unserving a view not served is nothing. */
  unserve(view: string) {
    delete this.infos[view]; delete this.sources[view]; delete this.views[view];
    this.producers.delete(view);
    this.octrees.delete(view);
    this.kept.forget(view);
    for (const key of [...this.inflight.keys()]) if (key.startsWith(`${view}/`)) this.chunks?.drop(key, `${view} is no longer served`);
  }

  /** Stop: the reader and the Pyodide workers are terminated (a call still waiting fails as
   * `Cancelled`), chunks waiting to be computed dropped, every view forgotten and the
   * service worker's requests no longer answered. A page that opens another source then
   * starts a fresh Engine, whose URLs carry a new page id, so a viewer that cached chunks
   * by URL fetches afresh. */
  stop() {
    this.stopped = true;  // and stays so: a late call() or read() throws rather than starting workers again
    for (const key of [...this.inflight.keys()]) this.chunks?.drop(key, "the engine was stopped");
    this.reader?.terminate();
    this.reader = null;
    for (const w of this.pool) w.terminate();
    this.pool = [];
    this.warm = Promise.resolve();
    this.chunks = null;
    this.producers.clear();
    this.octrees.clear();
    this.kept.clear();
    this.staged.clear();
    this.staging.clear();
    this.infos = {}; this.sources = {}; this.views = {};
    this.ready = Promise.resolve();
    this.unroute?.();
    this.unroute = null;
  }

  /** Voxels [lo, hi) of a level of a view's source, as the reader reads them for a chunk;
   * no Python is needed. */
  read(view: string, level: number, lo: number[], hi: number[], at?: Record<string, number>): Promise<ArrayBuffer> {
    if (!this.reader) return Promise.reject(new Error(this.stopped ? "the engine was stopped" : "the engine has not started"));
    return this.reader.call<ArrayBuffer>({ type: "read", view, level, lo, hi, ...(at ? { at } : {}) });
  }

  /** A step of a page's work (chunkmirage_contrib.stitching's, or tracking's `track_*`), run by the
   * next Pyodide worker (started, with Python loading, if none runs yet). */
  call<T>(fn: string, args: Record<string, unknown>, arrays: ArrayBuffer[] = []): Promise<T> {
    return this.worker().call<T>({ type: "call", fn, args: JSON.stringify(args), arrays }, arrays);
  }

  /** View `view` with some of its spec changed (other ops, another source), served under a
   * new name, so clients fetch it afresh and the old one's chunks stay as they were; returns
   * the name. */
  async edit(view: string, changes: Partial<PipelineView>): Promise<string> {
    const base = view.split("~")[0], name = `${base}~${++this.edits}`;
    this.views[base] = { ...this.views[base], ...changes };
    await this.plan({ [name]: this.views[base] });
    return name;
  }

  /** OME-Zarr URL of a view (for Neuroglancer: `zarr3://` + it). */
  url(view: string): string { return new URL(`virtual/${this.page}/${view}/`, location.href).href; }
  /** Zarr v2 URL of a view (OME-Zarr 0.4): the same chunks. */
  zarr2Url(view: string): string { return new URL(`virtual/${this.page}/zarr2/${view}/`, location.href).href; }
  /** GeoZarr URL of a view: a group whose levels hold the view as their one band. */
  geoUrl(view: string): string { return new URL(`virtual/${this.page}/geo/${view}`, location.href).href; }

  sample(view: string, ps: number[]): Promise<number[]> {
    return this.reader!.call<number[]>({ type: "sample", view, ps });
  }

  get running(): number { return this.chunks?.running ?? 0; }
  get waiting(): number { return this.chunks ? [...this.chunks.byLevel().waiting.values()].reduce((a, b) => a + b, 0) : 0; }

  private omeGroup(view: string) {
    const v = this.infos[view], added = (v.added ?? []).map(() => 0);  // channel axes the ops add
    return {
      zarr_format: 3, node_type: "group",
      attributes: { ome: { version: "0.5", multiscales: [{
        name: view,
        axes: [...added.map(() => ({ name: "c", type: "channel" })),
          ...v.axes.map((a) => ({ name: a.name, type: isTime(a) ? "time" : "space", unit: OME_UNIT[a.unit] ?? (a.unit || undefined) }))],
        datasets: v.levels.map((l, i) => ({ path: String(i), coordinateTransformations: [
          { type: "scale", scale: [...added.map(() => 1), ...l.voxel] }, { type: "translation", translation: [...added, ...l.origin] },
        ] })),
      }] } },
    };
  }

  /** The GeoZarr group of a view: the conventions, its levels as the multiscales layout, its
   * projection (the source's own, which the client must know by `proj`) and bounding box. */
  private geoGroup(view: string, proj: string) {
    const v = this.infos[view], bbox = v.geo?.bbox;
    if (!bbox) return null;
    return {
      zarr_format: 3, node_type: "group",
      attributes: {
        zarr_conventions: CONVENTIONS,
        multiscales: { layout: v.levels.map((l, i) => ({ asset: String(i), "spatial:shape": l.shape.slice(-2) })) },
        "proj:code": proj, "spatial:dimensions": ["y", "x"], "spatial:bbox": bbox, "spatial:shape": v.levels[0].shape.slice(-2),
      },
    };
  }

  /** A level's array; in the GeoZarr layout (`map`) as the map's y, x band, the view's one
   * plane along z (a map client reads 2-D bands: with a third axis it would need the
   * group's consolidated metadata to know to select it). */
  private array(view: string, level: number, map = false) {
    const v = this.infos[view], keep = map ? -2 : 0, added = map ? [] : v.added ?? [];  // channels: chunked whole
    return {
      zarr_format: 3, node_type: "array", shape: [...added, ...v.levels[level].shape.slice(keep)], data_type: v.out, fill_value: 0,
      chunk_grid: { name: "regular", configuration: { chunk_shape: [...added, ...this.views[view].chunk.slice(keep)] } },
      chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
      codecs: [{ name: "bytes", configuration: { endian: "little" } }],
      dimension_names: [...added.map(() => "c"), ...v.axes.map((a) => a.name).slice(keep)], attributes: {},
    };
  }

  /** The level a view's mesh is made from (chunkmirage.meshes.mesh_level). */
  meshLevel(view: string): number {
    const v = this.sources[view], spec = this.views[view].mesh ?? {};
    if (spec.level !== undefined) return spec.level;
    const small = v.levels.findIndex((l) => Math.max(...l.shape.slice(-3)) <= 512);
    return small < 0 ? v.levels.length - 1 : small;
  }

  /** Chunk `index` of a view's level: its input region from the reader (clipped to the
   * level; the worker pads it at the edges as a server stage does), computed by a worker.
   * With `mesh`, a mesh of the chunk instead, one voxel more on its high sides; or with
   * `box`, of those voxels of the level. */
  private async compute(view: string, level: number, index: number[], mesh?: MeshCall, box?: [number[], number[]]): Promise<ArrayBuffer> {
    const produce = this.producers.get(view);
    if (produce) return produce(level, index);
    if (!needsPython(this.views[view])) return this.asRead(view, level, index);  // no ops: the region as read, whole
    if (mesh?.mode === "node") {  // a coarsest node: cut from the mask the octree was made from
      const o = await this.octrees.get(view);
      if (o && o.level === level) return this.fromMask(view, o, index, mesh);
    }
    const v = this.infos[view], l = v.levels[level], C = this.views[view].chunk, halo = v.halo, more = mesh ? 1 : 0;
    const outLo = box ? box[0] : index.map((i, a) => i * C[a]);
    const outHi = box ? box[1] : outLo.map((o, a) => Math.min(o + C[a] + more, l.shape[a]));
    const inLo = outLo.map((o, a) => o - halo[a]), inHi = outHi.map((o, a) => o + halo[a]);
    const lo = inLo.map((o) => Math.max(o, 0)), hi = inHi.map((o, a) => Math.min(o, l.shape[a]));
    const data = computedSource(this.views[view].source) ? null : await this.reader!.call<ArrayBuffer>({ type: "read", view, level, lo, hi });
    const lead = v.lead ? [v.channels] : [];
    return this.worker().call<ArrayBuffer>({
      type: "compute", view, level, data, readShape: [...lead, ...hi.map((h, a) => h - lo[a])],
      inLo, inHi, outLo, outHi, full: [...lead, ...l.shape], voxel: l.voxel, origin: l.origin,
      unit: v.axes[v.axes.length - 1].unit, ...(mesh ? { mesh } : {}),
    }, data ? [data] : []);
  }

  /** Chunk `index` of a level of a view read as stored: its region from the reader, placed
   * whole in the chunk (asread.ts). */
  private async asRead(view: string, level: number, index: number[]): Promise<ArrayBuffer> {
    const v = this.infos[view], l = v.levels[level], C = this.views[view].chunk;
    const lo = index.map((i, a) => i * C[a]), hi = lo.map((o, a) => Math.min(o + C[a], l.shape[a]));
    const data = await this.reader!.call<ArrayBuffer>({ type: "read", view, level, lo, hi });
    return wholeChunk(data, v.out, v.channels, hi.map((h, a) => h - lo[a]), C);
  }

  private fromMask(view: string, o: Octree, index: number[], mesh: MeshCall): Promise<ArrayBuffer> {
    const v = this.infos[view], l = v.levels[o.level], C = this.views[view].chunk, shape = l.shape;
    const lo = index.map((i, a) => i * C[a]), hi = lo.map((x, a) => Math.min(x + C[a] + 1, shape[a])), d = hi.map((h, a) => h - lo[a]);
    const block = new Uint8Array(d[0] * d[1] * d[2]);
    for (let z = 0; z < d[0]; z++) for (let y = 0; y < d[1]; y++) {
      const from = ((z + lo[0]) * shape[1] + y + lo[1]) * shape[2] + lo[2];
      block.set(o.mask.subarray(from, from + d[2]), (z * d[1] + y) * d[2]);
    }
    return this.worker().call<ArrayBuffer>({
      type: "compute", view, level: o.level, data: block.buffer, readShape: d, inLo: lo, inHi: hi, outLo: lo, outHi: hi,
      full: shape, voxel: l.voxel, origin: l.origin, unit: v.axes[v.axes.length - 1].unit, mesh: { ...mesh, raw: true, threshold: 1 },
    }, [block.buffer]);
  }

  private chunk(view: string, level: number, index: number[], mesh?: MeshCall): Answered {
    const key = `${view}/${mesh ? `mesh${level}` : level}/${index.join(".")}`, done = this.kept.get(key);
    if (done) return { status: 200, body: done.slice(0), type: "application/octet-stream" };
    const chunks = this.chunks!, claim = new Claim();
    let pending = this.inflight.get(key);
    if (pending) chunks.claim(key);  // another request waits for it too
    else {
      pending = chunks.submit(key, level, () => this.compute(view, level, index, mesh), () => { this.counts.dropped++; })
        .then((b) => { this.counts.computed++; if (this.views[view]) this.kept.put(key, b); return b; }, (e) => {  // kept unless unserved meanwhile
          if (!(e instanceof Cancelled)) { this.counts.failed++; this.failure = { view, message: String((e as Error).message ?? e) }; console.error(`chunk ${key}: ${String((e as Error).message ?? e).trim().split("\n").slice(-3).join(" / ")}`); }
          throw e;
        })
        .finally(() => { this.inflight.delete(key); this.onChange(); });
      this.inflight.set(key, pending);
    }
    claim.onCancel(() => chunks.release(key));
    const mine = pending.then((b) => b.slice(0));  // each request its own copy: replies transfer it
    mine.catch(() => {});
    this.onChange();
    return { status: 200, type: "application/octet-stream", pending: mine, claim };
  }

  /** Projection codes of the GeoZarr groups, by view (set by the map page). */
  proj: Record<string, string> = {};

  private async answer(path: string, range?: string | null): Promise<Answered> {
    const parts = path.split("/virtual/")[1]?.split("/");
    if (!parts || parts[0] !== this.page) return null;
    if (!this.chunks) return notFound;
    const named = parts[1] === "geo" || parts[1] === "zarr2" ? parts[2] : parts[1];
    if (named && !this.infos[named] && this.sources[named]) await this.ready;  // its ops are being planned
    let rest = parts.slice(1), view: string;
    if (rest[0] === "zarr2") return this.zarr2(rest[1], rest.slice(2));
    const map = rest[0] === "geo";
    if (map) {  // geo/<view>/zarr.json, geo/<view>/<level>/zarr.json, geo/<view>/<level>/<view>/...
      view = rest[1];
      if (!this.infos[view] || this.infos[view].added?.length) return notFound;  // a map reads one band
      if (rest.length === 3 && rest[2] === "zarr.json") {
        const g = this.geoGroup(view, this.proj[view.split("~")[0]] ?? "");
        return g ? asJson(g) : notFound;
      }
      if (rest.length === 4 && rest[3] === "zarr.json") return asJson({ zarr_format: 3, node_type: "group", attributes: {} });
      if (rest[3] !== view || this.infos[view].levels[0].shape.slice(0, -2).some((n) => n !== 1)) return notFound;
      rest = [rest[2], ...rest.slice(4)];  // as the OME layout's <level>/..., its keys y, x
      if (rest[1] === "c") rest.splice(2, 0, ...this.infos[view].levels[0].shape.slice(0, -2).map(() => "0"));
    } else {
      view = rest[0];
      rest = rest.slice(1);
      if (!this.infos[view]) return notFound;
      if (rest.length === 1 && rest[0] === "zarr.json") return asJson(this.omeGroup(view));
      if (rest[0] === "mesh" && this.views[view].mesh) return this.mesh(view, rest.slice(1).join("/"), range);
    }
    const level = Number(rest[0]);
    if (!this.infos[view].levels[level]) return notFound;
    if (rest.length === 2 && rest[1] === "zarr.json") return asJson(this.array(view, level, map));
    const added = map ? 0 : this.infos[view].added?.length ?? 0, key = rest.slice(2).map(Number);
    if (rest[1] !== "c" || key.length !== added + this.infos[view].levels[level].shape.length || key.slice(0, added).some((i) => i !== 0)) return notFound;
    return this.chunk(view, level, key.slice(added));  // a channel axis is one chunk: the volume's index
  }

  /** A view in the zarr v2 layout: .zgroup, .zattrs (OME 0.4 multiscales), each level's
   * .zarray and .zattrs (its dimension names), and the chunks, keyed <level>/<i>/<j>/... */
  private zarr2(view: string, rest: string[]): Answered {
    const v = this.infos[view];
    if (!v) return notFound;
    const added = v.added ?? [], names = [...added.map(() => "c"), ...v.axes.map((a) => a.name)];
    if (rest.length === 1 && rest[0] === ".zgroup") return asJson({ zarr_format: 2 });
    if (rest.length === 1 && rest[0] === ".zattrs") {
      const g = this.omeGroup(view).attributes.ome.multiscales[0];
      return asJson({ multiscales: [{ ...g, version: "0.4" }] });
    }
    const level = Number(rest[0]), l = v.levels[level];
    if (!l) return notFound;
    if (rest.length === 2 && rest[1] === ".zattrs") return asJson({ _ARRAY_DIMENSIONS: names });
    if (rest.length === 2 && rest[1] === ".zarray") {
      return asJson({
        zarr_format: 2, shape: [...added, ...l.shape], chunks: [...added, ...this.views[view].chunk], dtype: V2_DTYPE[v.out] ?? v.out,
        compressor: null, fill_value: 0, order: "C", filters: null, dimension_separator: "/",
      });
    }
    const key = rest.slice(1).map(Number);
    if (key.length !== added.length + l.shape.length || key.some((i) => !Number.isInteger(i)) || key.slice(0, added.length).some((i) => i !== 0)) return notFound;
    return this.chunk(view, level, key.slice(added.length));
  }

  private octrees = new Map<string, Promise<Octree>>();

  /** A view's multi-resolution octree, made once from its whole coarsest level: its eighths
   * masked by the workers at once (each with a border, for its surface band), joined here,
   * and the nodes and index made by one. The mask is kept: the coarsest nodes are cut from it. */
  private octree(view: string): Promise<Octree> {
    let o = this.octrees.get(view);
    if (!o) {
      const spec = this.views[view].mesh!, v = this.infos[view], level = this.meshLevel(view), shape = v.levels[level].shape;
      const parts = [...Array(8).keys()].map((k) => {
        const lo = shape.map((n, a) => ((k >> a) & 1 ? n >> 1 : 0)), hi = shape.map((n, a) => ((k >> a) & 1 ? n : n >> 1));
        const glo = lo.map((x) => Math.max(x - BAND, 0)), ghi = hi.map((x, a) => Math.min(x + BAND, shape[a]));
        return { lo, hi, glo, ghi };
      }).filter((p) => p.hi.every((h, a) => h > p.lo[a]));
      o = Promise.all(parts.map((p) => this.compute(view, level, [], {
        kind: "surface", ...spec, mode: "mask", core_lo: p.lo.map((x, a) => x - p.glo[a]), core_hi: p.hi.map((x, a) => x - p.glo[a]),
      }, [p.glo, p.ghi]))).then(async (got) => {
        const n = shape[0] * shape[1] * shape[2], mask = new Uint8Array(n), band = new Uint8Array(n);
        got.forEach((g, i) => {  // each part's rows into the whole level's: its mask, then its band
          const { lo, hi } = parts[i], d = hi.map((h, a) => h - lo[a]), size = d[0] * d[1] * d[2];
          const m = new Uint8Array(g, 0, size), b = new Uint8Array(g, size, size);
          for (let z = 0; z < d[0]; z++) for (let y = 0; y < d[1]; y++) {
            const from = (z * d[1] + y) * d[2], to = ((z + lo[0]) * shape[1] + y + lo[1]) * shape[2] + lo[2];
            mask.set(m.subarray(from, from + d[2]), to);
            band.set(b.subarray(from, from + d[2]), to);
          }
        });
        const b = await this.worker().call<ArrayBuffer>({
          type: "octree", band: band.buffer, shape, chunk: this.views[view].chunk, unit: v.axes[v.axes.length - 1].unit,
          mesh: { kind: "surface", ...spec, mode: "mask", levels: v.levels },
        }, [band.buffer]);
        const head = new DataView(b, 0, 12), count = head.getUint32(0, true);
        return {
          nodes: new Int32Array(b.slice(12, 12 + 16 * count)), size: head.getUint32(4, true), bits: head.getUint32(8, true),
          index: b.slice(12 + 16 * count), mask, level,
        };
      });
      o.catch(() => this.octrees.delete(view));
      this.octrees.set(view, o);
    }
    return o;
  }

  /** A view's mesh, as chunkmirage's mesh frontend serves it. Single resolution: info, the
   * manifest (every chunk of the mesh level a fragment), and fragments, each meshed when
   * fetched. Multi-resolution (`lods` > 1): info, the index, and the fragments' file, read
   * by range, each node meshed when its bytes are asked for. */
  private mesh(view: string, path: string, range?: string | null): Answered {
    const spec = this.views[view].mesh!;
    if ((spec.lods ?? 1) > 1) return this.multires(view, path, range);
    const level = this.meshLevel(view), l = this.infos[view].levels[level], C = this.views[view].chunk;
    if (path === "info") return asJson({ "@type": "neuroglancer_legacy_mesh" });
    const grid = l.shape.map((n, a) => Math.ceil(n / C[a]));
    if (path === "1:0") {
      const fragments: string[] = [];
      const walk = (prefix: number[]) => {
        if (prefix.length === grid.length) { fragments.push(`1:0:${prefix.join("_")}`); return; }
        for (let i = 0; i < grid[prefix.length]; i++) walk([...prefix, i]);
      };
      walk([]);
      return asJson({ fragments });
    }
    const m = /^1:0:(\d+(?:_\d+)*)$/.exec(path);
    if (!m) return notFound;
    const index = m[1].split("_").map(Number);
    if (index.length !== grid.length || index.some((i, a) => i >= grid[a])) return notFound;
    return this.chunk(view, level, index, { kind: "surface", ...spec, mode: "legacy" });
  }

  private multires(view: string, path: string, range?: string | null): Answered {
    const later = (p: Promise<ArrayBuffer>, type = "application/octet-stream", extra: { status?: number; range?: string } = {}): Answered =>
      ({ status: extra.status ?? 200, type, pending: p, claim: new Claim(), ...(extra.range ? { range: extra.range } : {}) });
    const json = (o: unknown) => new TextEncoder().encode(JSON.stringify(o)).buffer as ArrayBuffer;
    if (path === "info") {
      return later(this.octree(view).then((o) => json({
        "@type": "neuroglancer_multilod_draco", vertex_quantization_bits: o.bits,
        transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], lod_scale_multiplier: 1,
      })), "application/json");
    }
    if (path === "1.index") return later(this.octree(view).then((o) => o.index.slice(0)));
    const m = /^bytes=(\d+)-(\d+)$/.exec(range ?? "");
    if (path !== "1" || !m) return notFound;
    const start = Number(m[1]), stop = Number(m[2]) + 1, spec = this.views[view].mesh!;
    const claim = new Claim();
    const pending = this.octree(view).then(async (o) => {
      const k = Math.floor(start / o.size);
      if (stop > (k + 1) * o.size || k * 4 >= o.nodes.length) throw new Error(`bytes ${start}-${stop}: one fragment at a time`);
      const [level, ...node] = o.nodes.slice(4 * k, 4 * k + 4);
      const a = this.chunk(view, level, [...node], { kind: "surface", ...spec, mode: "node", size: o.size, bits: o.bits });
      if (!a || !("pending" in a)) return (a?.body as ArrayBuffer).slice(start - k * o.size, stop - k * o.size);
      claim.onCancel(() => a.claim.cancel());
      return (await a.pending).slice(start - k * o.size, stop - k * o.size);
    });
    pending.catch(() => {});
    return { status: 206, type: "application/octet-stream", pending, claim, range: `bytes ${start}-${stop - 1}/*` };
  }

  /** Answer the service worker's requests for this page's share of /virtual/ (routes.ts hands
   * them here; the rest of the page's routes are the page's own). */
  private listen() {
    this.unroute?.();  // started again on one instance: one route, not two answering
    this.unroute = route(`/virtual/${this.page}/`, (ask, port) => void this.reply(ask, port));
  }

  private async reply(ask: Ask, port: MessagePort) {
    let a: Answered;
    try { a = await this.answer(ask.path, ask.range); } catch (err) { a = { status: 500, body: String(err), type: "text/plain" }; }
    if (!a || !("pending" in a)) return port.postMessage(a, a && a.body instanceof ArrayBuffer ? [a.body] : []);
    // a chunk: the head now, the body once computed; the client may give up in between
    port.onmessage = (m: MessageEvent<{ cancel?: boolean }>) => { if (m.data?.cancel) a.claim.cancel(); };
    port.postMessage({ status: a.status, type: a.type, stream: true, ...(a.range ? { range: a.range } : {}) } satisfies Reply);
    a.pending.then(
      (body) => port.postMessage({ body } satisfies Later, [body]),
      (err) => port.postMessage({ error: String((err as Error)?.message ?? err) } satisfies Later),
    ).finally(() => port.close());
  }
}
