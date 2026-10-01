// The browser engine behind the pipeline and map pages: a demo's views (each a chunkmirage
// pipeline spec) served from this page through the service worker (sw.ts), as a chunkmirage
// server would serve them. Each chunk's input region is read by the page's one reader
// (reader.ts) and computed by a Pyodide worker (pyworker.ts) running chunkmirage's own ops;
// chunks a client gives up on before their turn are dropped, and computed ones are kept a
// while (a map client keeps none itself). Two layouts of the same chunks: OME-Zarr 0.5 for
// Neuroglancer (virtual/<page>/<view>/...) and GeoZarr for map clients such as OpenLayers
// (virtual/<page>/geo/<view>/<level>/<view>/...), the zarr-conventions multiscales, proj:
// and spatial: attributes on the group.
import { Cancelled, Claim, Queue } from "./demand";
import type { Answer, Later, PipelineView, Reply, SourceInfo, ToPyWorker, ToReader, ViewAxis, ViewInfo } from "./types";

export const TO_SECONDS: Record<string, number> = { s: 1, second: 1, millisecond: 1e-3, ms: 1e-3, minute: 60, hour: 3600, day: 86400 };
const OME_UNIT: Record<string, string> = { nm: "nanometer", um: "micrometer", m: "meter", s: "second" };
const CONVENTIONS = [
  { uuid: "d35379db-88df-4056-af3a-620245f8e347", name: "multiscales" },
  { uuid: "f17cb550-5864-4468-aeb7-f3180cfb622f", name: "proj:" },
  { uuid: "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4", name: "spatial:" },
];
const KEPT_BYTES = 256 * 2 ** 20;  // computed chunks kept for clients that refetch
/** Sources the Pyodide workers compute themselves (chunkmirage's own Python), nothing read. */
const computedSource = (url: string) => url.startsWith("synthetic://");

type Request<R> = R extends unknown ? Omit<R, "reqId"> : never;
type Answered = null | { status: number; body: string | ArrayBuffer; type: string }
  | { status: number; type: string; pending: Promise<ArrayBuffer>; claim: Claim };
type Plan = { dtype: string; lead: number; halo: number[] };

/** A worker answering requests by reqId. */
class Rpc<Req extends { reqId: number }> {
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  private seq = 0;
  constructor(private worker: Worker) {
    worker.onmessage = ({ data: m }: MessageEvent<Answer>) => {
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
}

const asJson = (o: unknown) => ({ status: 200, body: JSON.stringify(o), type: "application/json" });
const notFound = { status: 404, body: "", type: "text/plain" };
export const isTime = (a: ViewAxis) => a.unit in TO_SECONDS;

export class Engine {
  readonly page = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  infos: Record<string, ViewInfo> = {};
  readonly counts = { computed: 0, dropped: 0, failed: 0 };
  private views: Record<string, PipelineView> = {};
  private reader: Rpc<ToReader> | null = null;
  private pool: Rpc<ToPyWorker>[] = [];
  private turn = 0;
  private chunks: Queue<ArrayBuffer> | null = null;
  private inflight = new Map<string, Promise<ArrayBuffer>>();
  private kept = new Map<string, ArrayBuffer>();  // in use order
  private keptBytes = 0;
  private edits = 0;

  /** `onChange` is told whenever the counts change. */
  constructor(private onChange: () => void = () => {}) {}

  /** Start the service worker, the reader and `n` Pyodide workers, and plan `views`. */
  async start(views: Record<string, PipelineView>, status: (s: string) => void): Promise<void> {
    if (!window.isSecureContext) throw new Error("This page needs a secure context: open it over https, or through localhost.");
    if (!navigator.serviceWorker) throw new Error("This browser has no service workers (a private window?).");
    status("Starting the service worker…");
    await navigator.serviceWorker.register("sw.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    this.listen();
    const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));
    status(`Loading Python (Pyodide, numpy, scipy) and chunkmirage's ops in ${n} worker${n > 1 ? "s" : ""}…`);
    this.reader = new Rpc<ToReader>(new Worker(new URL("./reader.ts", import.meta.url), { type: "module" }));
    this.pool = Array.from({ length: n }, () => new Rpc<ToPyWorker>(new Worker(new URL("./pyworker.ts", import.meta.url), { type: "module" })));
    this.chunks = new Queue<ArrayBuffer>(n);
    await this.plan(views);
  }

  /** Open `views`' sources (once each: read ones by the reader, computed ones described by
   * a worker) and plan their ops in every worker. */
  private async plan(views: Record<string, PipelineView>): Promise<void> {
    const entries = Object.entries(views), read = entries.filter(([, s]) => !computedSource(s.source));
    const opened = read.length ? await this.reader!.call<Record<string, SourceInfo>>({ type: "open", views: Object.fromEntries(read) }) : {};
    for (const [v, s] of entries) if (computedSource(s.source)) opened[v] = await this.pool[0].call<SourceInfo>({ type: "describe", source: s.source });
    const specs = Object.fromEntries(entries.map(([v, spec]) => {
      const s = opened[v], shape = [...(s.channels > 1 ? [s.channels] : []), ...s.levels[0].shape];
      const source = computedSource(spec.source) ? spec.source : undefined;
      return [v, { ops: spec.ops ?? [], shape, dtype: s.dtype, chunk: spec.chunk, voxel: s.levels[0].voxel, source }];
    }));
    const plans = await Promise.all(this.pool.map((w) => w.call<Record<string, Plan>>({ type: "plan", views: specs })));
    for (const [v, s] of Object.entries(opened)) this.infos[v] = { ...s, out: plans[0][v].dtype, lead: plans[0][v].lead, halo: plans[0][v].halo };
    Object.assign(this.views, views);
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
  /** GeoZarr URL of a view: a group whose levels hold the view as their one band. */
  geoUrl(view: string): string { return new URL(`virtual/${this.page}/geo/${view}`, location.href).href; }

  sample(view: string, ps: number[]): Promise<number[]> {
    return this.reader!.call<number[]>({ type: "sample", view, ps });
  }

  get running(): number { return this.chunks?.running ?? 0; }
  get waiting(): number { return this.chunks ? [...this.chunks.byLevel().waiting.values()].reduce((a, b) => a + b, 0) : 0; }

  private omeGroup(view: string) {
    const v = this.infos[view];
    return {
      zarr_format: 3, node_type: "group",
      attributes: { ome: { version: "0.5", multiscales: [{
        name: view,
        axes: v.axes.map((a) => ({ name: a.name, type: isTime(a) ? "time" : "space", unit: OME_UNIT[a.unit] ?? (a.unit || undefined) })),
        datasets: v.levels.map((l, i) => ({ path: String(i), coordinateTransformations: [
          { type: "scale", scale: l.voxel }, { type: "translation", translation: l.origin },
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
    const v = this.infos[view], keep = map ? -2 : 0;
    return {
      zarr_format: 3, node_type: "array", shape: v.levels[level].shape.slice(keep), data_type: v.out, fill_value: 0,
      chunk_grid: { name: "regular", configuration: { chunk_shape: this.views[view].chunk.slice(keep) } },
      chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
      codecs: [{ name: "bytes", configuration: { endian: "little" } }],
      dimension_names: v.axes.map((a) => a.name).slice(keep), attributes: {},
    };
  }

  /** Chunk `index` of a view's level: its input region from the reader (clipped to the
   * level; the worker pads it at the edges as a server stage does), computed by a worker. */
  private async compute(view: string, level: number, index: number[]): Promise<ArrayBuffer> {
    const v = this.infos[view], l = v.levels[level], C = this.views[view].chunk, halo = v.halo;
    const outLo = index.map((i, a) => i * C[a]), outHi = outLo.map((o, a) => Math.min(o + C[a], l.shape[a]));
    const inLo = outLo.map((o, a) => o - halo[a]), inHi = outHi.map((o, a) => o + halo[a]);
    const lo = inLo.map((o) => Math.max(o, 0)), hi = inHi.map((o, a) => Math.min(o, l.shape[a]));
    const data = computedSource(this.views[view].source) ? null : await this.reader!.call<ArrayBuffer>({ type: "read", view, level, lo, hi });
    const lead = v.lead ? [v.channels] : [];
    const worker = this.pool[this.turn++ % this.pool.length];
    return worker.call<ArrayBuffer>({
      type: "compute", view, level, data, readShape: [...lead, ...hi.map((h, a) => h - lo[a])],
      inLo, inHi, outLo, outHi, full: [...lead, ...l.shape], voxel: l.voxel,
    }, data ? [data] : []);
  }

  private keep(key: string, body: ArrayBuffer) {
    this.kept.set(key, body);
    this.keptBytes += body.byteLength;
    while (this.keptBytes > KEPT_BYTES && this.kept.size > 1) {
      const [old] = this.kept.keys();
      this.keptBytes -= this.kept.get(old)!.byteLength;
      this.kept.delete(old);
    }
  }

  private chunk(view: string, level: number, index: number[]): Answered {
    const key = `${view}/${level}/${index.join(".")}`, done = this.kept.get(key);
    if (done) {
      this.kept.delete(key); this.kept.set(key, done);
      return { status: 200, body: done.slice(0), type: "application/octet-stream" };
    }
    const chunks = this.chunks!, claim = new Claim();
    let pending = this.inflight.get(key);
    if (pending) chunks.claim(key);  // another request waits for it too
    else {
      pending = chunks.submit(key, level, () => this.compute(view, level, index), () => { this.counts.dropped++; })
        .then((b) => { this.counts.computed++; this.keep(key, b); return b; }, (e) => {
          if (!(e instanceof Cancelled)) { this.counts.failed++; console.error(`chunk ${key}: ${String((e as Error).message ?? e).trim().split("\n").slice(-3).join(" / ")}`); }
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

  private answer(path: string): Answered {
    const parts = path.split("/virtual/")[1]?.split("/");
    if (!parts || parts[0] !== this.page) return null;
    if (!this.chunks) return notFound;
    let rest = parts.slice(1), view: string;
    const map = rest[0] === "geo";
    if (map) {  // geo/<view>/zarr.json, geo/<view>/<level>/zarr.json, geo/<view>/<level>/<view>/...
      view = rest[1];
      if (!this.infos[view]) return notFound;
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
    }
    const level = Number(rest[0]);
    if (!this.infos[view].levels[level]) return notFound;
    if (rest.length === 2 && rest[1] === "zarr.json") return asJson(this.array(view, level, map));
    if (rest[1] !== "c" || rest.length !== 2 + this.infos[view].levels[level].shape.length) return notFound;
    return this.chunk(view, level, rest.slice(2).map(Number));
  }

  private listen() {
    navigator.serviceWorker.addEventListener("message", async (e: MessageEvent<{ path: string }>) => {
      const port = e.ports[0];
      if (!port) return;
      let a: Answered;
      try { a = this.answer(e.data.path); } catch (err) { a = { status: 500, body: String(err), type: "text/plain" }; }
      if (!a || !("pending" in a)) return port.postMessage(a, a && a.body instanceof ArrayBuffer ? [a.body] : []);
      // a chunk: the head now, the body once computed; the client may give up in between
      port.onmessage = (m: MessageEvent<{ cancel?: boolean }>) => { if (m.data?.cancel) a.claim.cancel(); };
      port.postMessage({ status: a.status, type: a.type, stream: true } satisfies Reply);
      a.pending.then(
        (body) => port.postMessage({ body } satisfies Later, [body]),
        (err) => port.postMessage({ error: String((err as Error)?.message ?? err) } satisfies Later),
      ).finally(() => port.close());
    });
  }
}
