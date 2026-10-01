// The pipeline page: one demo of the gallery (cards.ts), its views served to Neuroglancer
// as zarr v3 from this page, through the service worker (sw.ts), as a chunkmirage server
// would serve them. Each chunk is computed by a Pyodide worker (pyworker.ts) running
// chunkmirage's own ops; chunks the viewer gives up on before their turn are dropped.
import { CARDS, type CardLayer, type PipelineCard } from "./cards";
import { Cancelled, Claim, Queue } from "./demand";
import type { FromPyWorker, Later, Reply, ToPyWorker, ViewInfo } from "./types";

const PAGE = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
const TO_METRES: Record<string, number> = { nanometer: 1e-9, nm: 1e-9, micrometer: 1e-6, um: 1e-6, millimeter: 1e-3, meter: 1 };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

type Answer = null | { status: number; body: string | ArrayBuffer; type: string }
  | { status: number; type: string; pending: Promise<ArrayBuffer>; claim: Claim };

class Pool {
  private workers: Worker[] = [];
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  private seq = 0;
  private next = 0;
  constructor(n: number) {
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL("./pyworker.ts", import.meta.url), { type: "module" });
      w.onmessage = ({ data: m }: MessageEvent<FromPyWorker>) => {
        if (m.type === "ready" || !("reqId" in m) || m.reqId === undefined) return;
        const p = this.pending.get(m.reqId);
        if (!p) return;
        this.pending.delete(m.reqId);
        if (m.type === "error") p.reject(new Error(m.message));
        else p.resolve((m.type === "chunk" ? m.body : m.values) as never);
      };
      this.workers.push(w);
    }
  }
  get size() { return this.workers.length; }
  /** Every worker set up with the card's views; the views' metadata. */
  setup(views: PipelineCard["views"]): Promise<Record<string, ViewInfo>> {
    return Promise.all(this.workers.map((w) => new Promise<Record<string, ViewInfo>>((resolve, reject) => {
      const done = ({ data: m }: MessageEvent<FromPyWorker>) => {
        if (m.type === "ready") { w.removeEventListener("message", done); resolve(m.views); }
        else if (m.type === "error" && m.reqId === undefined) { w.removeEventListener("message", done); reject(new Error(m.message)); }
      };
      w.addEventListener("message", done);
      w.onerror = (e) => reject(new Error(e.message || "a worker failed to start"));
      w.postMessage({ type: "setup", views } satisfies ToPyWorker);
    }))).then((all) => all[0]);
  }
  ask<T>(msg: { type: "chunk"; view: string; level: number; index: number[] } | { type: "sample"; view: string; ps: number[] }): Promise<T> {
    const reqId = ++this.seq, w = this.workers[this.next++ % this.workers.length];
    return new Promise<T>((resolve, reject) => {
      this.pending.set(reqId, { resolve: resolve as (v: never) => void, reject });
      w.postMessage({ ...msg, reqId } as ToPyWorker);
    });
  }
}

let pool: Pool | null = null;
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

function groupMetadata(view: string) {
  const v = infos[view];
  return {
    zarr_format: 3, node_type: "group",
    attributes: { ome: { version: "0.5", multiscales: [{
      name: view, axes: ["z", "y", "x"].map((name) => ({ name, type: "space", unit: v.unit || undefined })),
      datasets: v.levels.map((l, i) => ({ path: String(i), coordinateTransformations: [
        { type: "scale", scale: l.voxel }, { type: "translation", translation: l.origin },
      ] })),
    }] } },
  };
}

function arrayMetadata(card: PipelineCard, view: string, level: number) {
  const v = infos[view];
  return {
    zarr_format: 3, node_type: "array", shape: v.levels[level].shape, data_type: v.dtype, fill_value: 0,
    chunk_grid: { name: "regular", configuration: { chunk_shape: card.views[view].chunk } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    dimension_names: ["z", "y", "x"], attributes: {},
  };
}

const asJson = (o: unknown) => ({ status: 200, body: JSON.stringify(o), type: "application/json" });

function answer(card: PipelineCard, path: string): Answer {
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
    pending = chunks.submit(key, level, () => pool!.ask<ArrayBuffer>({ type: "chunk", view, level, index }), () => { counts.dropped++; })
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
    let a: Answer;
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

async function viewerState(card: PipelineCard) {
  const first = infos[Object.keys(card.views)[0]], l0 = first.levels[0];
  const toM = TO_METRES[first.unit] ?? 1;
  const dims = { x: [l0.voxel[2] * toM, "m"], y: [l0.voxel[1] * toM, "m"], z: [l0.voxel[0] * toM, "m"] };
  const url = (view: string) => `zarr3://${new URL(`virtual/${PAGE}/${view}/`, location.href).href}`;
  const layers = await Promise.all(card.layers.map(async (l) => {
    const source = l.view ? url(l.view) : l.url!;
    if (l.type === "segmentation") return { type: "segmentation", name: l.name, source, selectedAlpha: 0.9 };
    let range = l.range;
    if (!range && l.percentiles && l.view) range = (await pool!.ask<number[]>({ type: "sample", view: l.view, ps: l.percentiles })) as [number, number];
    return {
      type: "image", name: l.name, source,
      ...(range ? { shader: shader(l, range) } : {}),
      ...(l.additive ? { blend: "additive" } : {}),
    };
  }));
  const main = document.querySelector("main")!;
  return {
    dimensions: dims, position: [card.position[2] + 0.5, card.position[1] + 0.5, card.position[0] + 0.5],
    displayDimensions: ["x", "y", "z"], crossSectionScale: card.zoom,
    crossSectionBackgroundColor: "#000000", layers,
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
  pool = new Pool(n);
  infos = await pool.setup(card.views);
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
