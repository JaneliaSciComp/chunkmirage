// The pipeline page: one demo of the gallery (cards.ts), its views served to Neuroglancer
// as zarr v3 from this page, through the service worker (sw.ts), as a chunkmirage server
// would serve them. Each chunk's input region is read by the page's one reader (reader.ts)
// and computed by a Pyodide worker (pyworker.ts) running chunkmirage's own ops; chunks the
// viewer gives up on before their turn are dropped.
import { CARDS, type CardLayer, type PipelineCard } from "./cards";
import { Cancelled, Claim, Queue } from "./demand";
import type { Answer, Later, Reply, SourceInfo, ToPyWorker, ToReader, ViewAxis, ViewInfo } from "./types";

const PAGE = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
const TO_METRES: Record<string, number> = { nanometer: 1e-9, nm: 1e-9, micrometer: 1e-6, um: 1e-6, millimeter: 1e-3, meter: 1, m: 1 };
const TO_SECONDS: Record<string, number> = { s: 1, second: 1, millisecond: 1e-3, ms: 1e-3, minute: 60, hour: 3600, day: 86400 };
const OME_UNIT: Record<string, string> = { nm: "nanometer", um: "micrometer", m: "meter", s: "second" };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Request<R> = R extends unknown ? Omit<R, "reqId"> : never;
type Answered = null | { status: number; body: string | ArrayBuffer; type: string }
  | { status: number; type: string; pending: Promise<ArrayBuffer>; claim: Claim };

/** A worker answering requests by reqId. */
class Rpc<Req extends { reqId: number }> {
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  private seq = 0;
  constructor(private worker: Worker) {
    worker.onmessage = ({ data: m }: MessageEvent<Answer>) => {
      const p = this.pending.get(m.reqId);
      if (!p) return;
      this.pending.delete(m.reqId);
      if ("error" in m) p.reject(new Error(m.error)); else p.resolve(m.value as never);
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

let reader: Rpc<ToReader> | null = null;
let pool: Rpc<ToPyWorker>[] = [];
let turn = 0;
let infos: Record<string, ViewInfo> = {};
let chunks: Queue<ArrayBuffer> | null = null;
const inflight = new Map<string, Promise<ArrayBuffer>>();
const counts = { computed: 0, dropped: 0, failed: 0 };

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  if (!chunks) return;
  const waiting = [...chunks.byLevel().waiting.values()].reduce((a, b) => a + b, 0);
  $("counts").textContent = `Chunks computed here: ${counts.computed} · computing ${chunks.running} · waiting ${waiting}`
    + ` · given up by the viewer before their turn: ${counts.dropped}${counts.failed ? ` · failed ${counts.failed}` : ""}`;
}

const isTime = (a: ViewAxis) => a.unit in TO_SECONDS;

function groupMetadata(view: string) {
  const v = infos[view];
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

function arrayMetadata(card: PipelineCard, view: string, level: number) {
  const v = infos[view];
  return {
    zarr_format: 3, node_type: "array", shape: v.levels[level].shape, data_type: v.out, fill_value: 0,
    chunk_grid: { name: "regular", configuration: { chunk_shape: card.views[view].chunk } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    dimension_names: v.axes.map((a) => a.name), attributes: {},
  };
}

const asJson = (o: unknown) => ({ status: 200, body: JSON.stringify(o), type: "application/json" });

/** Chunk `index` of a view's level: its input region from the reader (clipped to the
 * level; the worker pads it at the edges as a server stage does), computed by a worker. */
async function computeChunk(card: PipelineCard, view: string, level: number, index: number[]): Promise<ArrayBuffer> {
  const v = infos[view], shape = v.levels[level].shape, C = card.views[view].chunk, halo = v.halo;
  const outLo = index.map((i, a) => i * C[a]), outHi = outLo.map((o, a) => Math.min(o + C[a], shape[a]));
  const inLo = outLo.map((o, a) => o - halo[a]), inHi = outHi.map((o, a) => o + halo[a]);
  const lo = inLo.map((o) => Math.max(o, 0)), hi = inHi.map((o, a) => Math.min(o, shape[a]));
  const data = await reader!.call<ArrayBuffer>({ type: "read", view, level, lo, hi });
  const lead = v.lead ? [v.channels] : [];
  const worker = pool[turn++ % pool.length];
  return worker.call<ArrayBuffer>({
    type: "compute", view, data, readShape: [...lead, ...hi.map((h, a) => h - lo[a])],
    inLo, inHi, outLo, outHi, full: [...lead, ...shape],
  }, [data]);
}

function answer(card: PipelineCard, path: string): Answered {
  const parts = path.split("/virtual/")[1]?.split("/");
  if (!parts || parts[0] !== PAGE) return null;
  const [, view, ...rest] = parts;
  const notFound = { status: 404, body: "", type: "text/plain" };
  if (!infos[view] || !chunks) return notFound;
  if (rest.length === 1 && rest[0] === "zarr.json") return asJson(groupMetadata(view));
  const level = Number(rest[0]);
  if (!infos[view].levels[level]) return notFound;
  if (rest.length === 2 && rest[1] === "zarr.json") return asJson(arrayMetadata(card, view, level));
  if (rest[1] !== "c" || rest.length !== 5) return notFound;
  const index = rest.slice(2).map(Number), key = `${view}/${level}/${index.join(".")}`;
  const claim = new Claim();
  let pending = inflight.get(key);
  if (pending) chunks.claim(key);  // another request waits for it too
  else {
    pending = chunks.submit(key, level, () => computeChunk(card, view, level, index), () => { counts.dropped++; })
      .then((b) => { counts.computed++; return b; }, (e) => { if (!(e instanceof Cancelled)) counts.failed++; throw e; })
      .finally(() => { inflight.delete(key); showCounts(); });
    inflight.set(key, pending);
  }
  claim.onCancel(() => chunks!.release(key));
  const mine = pending.then((b) => b.slice(0));  // each request its own copy: replies transfer it
  mine.catch(() => {});
  showCounts();
  return { status: 200, type: "application/octet-stream", pending: mine, claim };
}

function listen(card: PipelineCard) {
  navigator.serviceWorker.addEventListener("message", async (e: MessageEvent<{ path: string }>) => {
    const port = e.ports[0];
    if (!port) return;
    let a: Answered;
    try { a = answer(card, e.data.path); } catch (err) { a = { status: 500, body: String(err), type: "text/plain" }; }
    if (!a || !("pending" in a)) return port.postMessage(a, a && a.body instanceof ArrayBuffer ? [a.body] : []);
    // a chunk: the head now, the body once computed; the viewer may give up in between
    port.onmessage = (m: MessageEvent<{ cancel?: boolean }>) => { if (m.data?.cancel) a.claim.cancel(); };
    port.postMessage({ status: a.status, type: a.type, stream: true } satisfies Reply);
    a.pending.then(
      (body) => port.postMessage({ body } satisfies Later, [body]),
      (err) => port.postMessage({ error: String((err as Error)?.message ?? err) } satisfies Later),
    ).finally(() => port.close());
  });
}

function shader(l: CardLayer, range: [number, number]): string {
  const c = l.colour ?? "#ffffff";
  const alpha = l.alpha ?? 1;
  return `#uicontrol invlerp normalized(range=[${range[0]}, ${range[1]}])\n#uicontrol vec3 colour color(default="${c}")\n`
    + (alpha < 1
      ? `void main() { float v = normalized(); emitRGBA(vec4(colour * v, v * ${alpha.toFixed(3)})); }\n`
      : "void main() { emitRGB(colour * normalized()); }\n");
}

/** Neuroglancer's [scale, unit] for an axis: lengths in metres, times in seconds, anything
 * else (degrees of latitude, say) unitless. */
function dimension(a: ViewAxis, voxel: number): [number, string] {
  if (a.unit in TO_METRES) return [voxel * TO_METRES[a.unit], "m"];
  if (a.unit in TO_SECONDS) return [voxel * TO_SECONDS[a.unit], "s"];
  return [voxel, ""];
}

async function viewerState(card: PipelineCard) {
  const first = infos[Object.keys(card.views)[0]], l0 = first.levels[0];
  const order = [2, 1, 0];  // shown x, y, z: the last axis across
  const names = order.map((a) => first.axes[a].name);
  const dims = Object.fromEntries(order.map((a) => [first.axes[a].name, dimension(first.axes[a], l0.voxel[a])]));
  const url = (view: string) => `zarr3://${new URL(`virtual/${PAGE}/${view}/`, location.href).href}`;
  const layers = await Promise.all(card.layers.map(async (l) => {
    const source = l.view ? url(l.view) : l.url!;
    if (l.type === "segmentation") {
      return { type: "segmentation", name: l.name, source, selectedAlpha: l.alpha ?? 0.9, ...(l.colour ? { segmentDefaultColor: l.colour } : {}) };
    }
    let range = l.range;
    if (!range && l.percentiles && l.view) range = (await reader!.call<number[]>({ type: "sample", view: l.view, ps: l.percentiles })) as [number, number];
    const glsl = l.shader ?? (range ? shader(l, range) : undefined);
    return {
      type: "image", name: l.name, source,
      ...(glsl ? { shader: glsl } : {}),
      ...(l.additive ? { blend: "additive" } : {}),
    };
  }));
  const main = document.querySelector("main")!;
  return {
    // the viewer's position is physical, in voxels: the source's origin is part of it
    dimensions: dims, position: order.map((a) => card.position[a] + 0.5 + l0.origin[a] / l0.voxel[a]),
    displayDimensions: names, crossSectionScale: card.zoom,
    ...(card.orientation ? { crossSectionOrientation: card.orientation } : {}),
    crossSectionBackgroundColor: "#000000", showAxisLines: false, layers,
    layout: card.panels.length === 1
      ? { type: "viewer", layers: card.panels[0], layout: "xy" }
      : { type: "row", children: card.panels.map((names) => ({ type: "viewer", layers: names, layout: "xy" })) },
    selectedLayer: { visible: false }, size: [main.clientWidth, main.clientHeight],
  };
}

async function start() {
  const id = new URLSearchParams(location.search).get("card") ?? "contacts";
  const card = CARDS.find((c): c is PipelineCard => c.kind === "pipeline" && c.id === id);
  if (!card) { status(`No pipeline demo called "${id}".`); return; }
  document.title = card.title;
  $("title").textContent = card.title;
  $("blurb").textContent = card.blurb;
  $("data").textContent = card.data;
  $("command").textContent = card.command;
  if (!window.isSecureContext) { status("This page needs a secure context: open it over https, or through localhost."); return; }
  if (!navigator.serviceWorker) { status("This browser has no service workers (a private window?)."); return; }
  status("Starting the service worker…");
  await navigator.serviceWorker.register("sw.js", { scope: "./" });
  await navigator.serviceWorker.ready;
  listen(card);
  const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));
  status(`Loading Python (Pyodide, numpy, scipy) and chunkmirage's ops in ${n} worker${n > 1 ? "s" : ""}…`);
  const t0 = performance.now();
  reader = new Rpc<ToReader>(new Worker(new URL("./reader.ts", import.meta.url), { type: "module" }));
  pool = Array.from({ length: n }, () => new Rpc<ToPyWorker>(new Worker(new URL("./pyworker.ts", import.meta.url), { type: "module" })));
  const opened = await reader.call<Record<string, SourceInfo>>({ type: "open", views: card.views });
  const specs = Object.fromEntries(Object.entries(card.views).map(([v, spec]) => {
    const s = opened[v], shape = [...(s.channels > 1 ? [s.channels] : []), ...s.levels[0].shape];
    return [v, { ops: spec.ops ?? [], shape, dtype: s.dtype, chunk: spec.chunk }];
  }));
  const plans = await Promise.all(pool.map((w) => w.call<Record<string, { dtype: string; lead: number; halo: number[] }>>({ type: "plan", views: specs })));
  infos = Object.fromEntries(Object.entries(opened).map(([v, s]) => [v, { ...s, out: plans[0][v].dtype, lead: plans[0][v].lead, halo: plans[0][v].halo }]));
  chunks = new Queue<ArrayBuffer>(n);
  status(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. Chunks are computed as the viewer asks for them.`);
  const ng = $<HTMLIFrameElement>("ng");
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(await viewerState(card)))}`;
  ng.hidden = false;
  $("empty").hidden = true;
  showCounts();
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
