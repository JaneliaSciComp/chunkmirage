// The pipeline page's reader: the views' sources, opened once for the page, so each store
// chunk is decoded once however many Pyodide workers compute from it (a sea-temperature tile
// is 65 MB decoded; a copy in every worker would not fit). The page asks it for the region
// a chunk needs, all channels, and hands that to a worker.
import { percentiles, prod } from "./ome";
import { openSource, type Source } from "./sources";
import type { Answer, SourceInfo, ToReader } from "./types";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const sources = new Map<string, Source>();      // by view
const opened = new Map<string, Promise<Source>>();  // by source and select: views share one

async function open(views: Extract<ToReader, { type: "open" }>["views"]) {
  const out: Record<string, SourceInfo> = {};
  await Promise.all(Object.entries(views).map(async ([id, spec]) => {
    const key = JSON.stringify([spec.source, spec.select ?? {}]);
    if (!opened.has(key)) opened.set(key, openSource(spec.source, spec.select ?? {}));
    const src = await opened.get(key)!;
    sources.set(id, src);
    out[id] = { dtype: src.dtype, channels: src.channels, axes: src.axes, levels: src.levels.map(({ shape, voxel, origin }) => ({ shape, voxel, origin })), geo: src.geo };
  }));
  return out;
}

/** Voxels [lo, hi) of level `level` of a view's source, its channels one after another. */
async function read(id: string, level: number, lo: number[], hi: number[], at?: Record<string, number>): Promise<ArrayBuffer> {
  const src = sources.get(id)!;
  const parts = await Promise.all(Array.from({ length: src.channels }, (_, c) => src.read(level, c, lo, hi, at)));
  const each = prod(hi.map((h, a) => h - lo[a]));
  const all = new (parts[0].constructor as { new (n: number): typeof parts[0] })(each * parts.length);
  parts.forEach((p, c) => all.set(p as never, c * each));
  return all.buffer as ArrayBuffer;
}

/** Display limits of a view's source: percentiles of a full-resolution region at its centre
 * (a coarse level would average small bright things, such as spots, away). */
async function sample(id: string, ps: number[]) {
  const src = sources.get(id)!, s = src.levels[0].shape;
  const half = s.map((n, a) => Math.min(n, [32, 256, 256][a]) >> 1);
  const lo = s.map((n, a) => (n >> 1) - half[a]), hi = s.map((n, a) => (n >> 1) + half[a]);
  return percentiles((await src.read(0, 0, lo, hi)).filter((v) => !Number.isNaN(v)), ps);
}

ctx.onmessage = async ({ data: m }: MessageEvent<ToReader>) => {
  try {
    if (m.type === "open") ctx.postMessage({ reqId: m.reqId, value: await open(m.views) } satisfies Answer);
    else if (m.type === "read") { const body = await read(m.view, m.level, m.lo, m.hi, m.at); ctx.postMessage({ reqId: m.reqId, value: body } satisfies Answer, [body]); }
    else if (m.type === "sample") ctx.postMessage({ reqId: m.reqId, value: await sample(m.view, m.ps) } satisfies Answer);
  } catch (e) {
    ctx.postMessage({ reqId: m.reqId, error: String((e as Error)?.message ?? e) } satisfies Answer);
  }
};
