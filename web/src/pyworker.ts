// A pipeline worker: chunkmirage's own Python (its ops and chunkmirage.fused, the code a
// server's pipeline stage runs) in Pyodide, fed regions this worker reads from the images.
// The page sends it the views to serve; it answers each chunk with the bytes a zarr v3
// chunk holds, computed as the Python server computes it.
import pyBase from "../../src/chunkmirage/ops/base.py?raw";
import pyCombine from "../../src/chunkmirage/ops/combine.py?raw";
import pyCore from "../../src/chunkmirage/core.py?raw";
import pyFilters from "../../src/chunkmirage/ops/filters.py?raw";
import pyFused from "../../src/chunkmirage/fused.py?raw";
import pyOps from "../../src/chunkmirage/ops/__init__.py?raw";
import pyPointwise from "../../src/chunkmirage/ops/pointwise.py?raw";
import pySegment from "../../src/chunkmirage/ops/segment.py?raw";
import { percentiles, prod } from "./ome";
import { openSource, type Source } from "./sources";
import type { PipelineView, ToPyWorker, ViewInfo } from "./types";

export const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const FILES: Record<string, string> = {
  "chunkmirage/__init__.py": '"""chunkmirage\'s ops and fused stages, for the browser engine."""\n',
  "chunkmirage/core.py": pyCore, "chunkmirage/fused.py": pyFused, "chunkmirage/ops/__init__.py": pyOps,
  "chunkmirage/ops/base.py": pyBase, "chunkmirage/ops/pointwise.py": pyPointwise, "chunkmirage/ops/filters.py": pyFilters,
  "chunkmirage/ops/segment.py": pySegment, "chunkmirage/ops/combine.py": pyCombine,
};
const GLUE = `
import json
import numpy as np
from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec

VIEWS = {}

def _info(shape, dtype, chunk):
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")
    return ArrayInfo(shape=tuple(shape), dtype=np.dtype(dtype), chunk_shape=tuple(shape[: n - 3]) + tuple(chunk),
                     voxel_size=(1.0,) * n, units=("",) * n, axes=axes)

def plan(view, ops, shape, dtype, chunk):
    ops = [op_from_spec(s) for s in json.loads(ops)]
    out, lead, halo = fused.plan(_info(list(shape), dtype, list(chunk)), ops)
    VIEWS[view] = (ops, dtype, list(chunk))
    return json.dumps({"dtype": out.dtype.name, "lead": lead, "halo": list(halo), "ndim": out.ndim})

def compute(view, data, read_shape, in_lo, in_hi, out_lo, out_hi, full_shape):
    ops, dtype, chunk = VIEWS[view]
    full = tuple(full_shape)
    lead = len(full) - 3
    block = np.frombuffer(data.to_py(), dtype=np.dtype(dtype)).reshape(tuple(read_shape))
    in_box = Box((0,) * lead + tuple(in_lo), full[:lead] + tuple(in_hi))
    block = fused.pad_edge(block, in_box, full)
    out = fused.plan(_info(full, dtype, chunk), ops)[0]
    result = fused.run(ops, block, in_box, Box(tuple(out_lo), tuple(out_hi)), out)
    return np.ascontiguousarray(result).astype(result.dtype.newbyteorder("<"), copy=False).tobytes()
`;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null;
const views = new Map<string, { spec: PipelineView; src: Source; info: ViewInfo }>();

async function setup(specs: Record<string, PipelineView>) {
  const mod = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
  py = await mod.loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "scipy", "pydantic"]);
  py.FS.mkdirTree("/chunkmirage/chunkmirage/ops");
  for (const [path, text] of Object.entries(FILES)) py.FS.writeFile(`/chunkmirage/${path}`, text);
  py.runPython(`import sys; sys.path.insert(0, "/chunkmirage")\n${GLUE}`);
  const out: Record<string, ViewInfo> = {};
  for (const [id, spec] of Object.entries(specs)) {
    const src = await openSource(spec.source, spec.select ?? {});
    const l0 = src.levels[0], shape = [...(src.channels > 1 ? [src.channels] : []), ...l0.shape];
    const plan = JSON.parse(py.globals.get("plan")(id, JSON.stringify(spec.ops ?? []), shape, src.dtype, spec.chunk));
    if (plan.ndim !== 3) throw new Error(`view ${id}: a viewer shows z, y, x volumes; its ops leave ${plan.ndim} axes`);
    const info: ViewInfo = { dtype: plan.dtype, halo: plan.halo, lead: plan.lead, unit: src.unit, levels: src.levels.map((l) => ({ shape: l.shape, voxel: l.voxel, origin: l.origin })) };
    views.set(id, { spec, src, info });
    out[id] = info;
  }
  return out;
}

/** Chunk `index` of level `level` of view `id`, as zarr v3 bytes (little endian, C order). */
async function chunk(id: string, level: number, index: number[]): Promise<ArrayBuffer> {
  const v = views.get(id);
  if (!v) throw new Error(`no view ${id}`);
  const shape = v.src.levels[level].shape, C = v.spec.chunk, halo = v.info.halo;
  const outLo = index.map((i, a) => i * C[a]), outHi = outLo.map((o, a) => Math.min(o + C[a], shape[a]));
  const inLo = outLo.map((o, a) => o - halo[a]), inHi = outHi.map((o, a) => o + halo[a]);
  const lo = inLo.map((o) => Math.max(o, 0)), hi = inHi.map((o, a) => Math.min(o, shape[a]));
  const n = v.info.lead ? v.src.channels : 1;
  const parts = await Promise.all(Array.from({ length: n }, (_, c) => v.src.read(level, c, lo, hi)));
  const each = prod(hi.map((h, a) => h - lo[a]));
  const T = parts[0].constructor as { new (n: number): typeof parts[0] };
  const all = new T(each * n);
  parts.forEach((p, c) => all.set(p as never, c * each));
  const readShape = [...(v.info.lead ? [n] : []), ...hi.map((h, a) => h - lo[a])];
  const full = [...(v.info.lead ? [n] : []), ...shape];
  const bytes = py.globals.get("compute")(id, new Uint8Array(all.buffer), readShape, inLo, inHi, outLo, outHi, full);
  const out = bytes.toJs() as Uint8Array;
  bytes.destroy();
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
}

/** Display limits of view `id`'s source: percentiles of a full-resolution region at its
 * centre (a coarse level would average small bright things, such as spots, away). */
async function sample(id: string, ps: number[]) {
  const v = views.get(id)!;
  const s = v.src.levels[0].shape, half = s.map((n, a) => Math.min(n, [32, 256, 256][a]) >> 1);
  const lo = s.map((n, a) => (n >> 1) - half[a]), hi = s.map((n, a) => (n >> 1) + half[a]);
  return percentiles(await v.src.read(0, 0, lo, hi), ps);
}

ctx.onmessage = async ({ data: m }: MessageEvent<ToPyWorker>) => {
  try {
    if (m.type === "setup") ctx.postMessage({ type: "ready", views: await setup(m.views) });
    else if (m.type === "chunk") { const body = await chunk(m.view, m.level, m.index); ctx.postMessage({ type: "chunk", reqId: m.reqId, body }, [body]); }
    else if (m.type === "sample") ctx.postMessage({ type: "sample", reqId: m.reqId, values: await sample(m.view, m.ps) });
  } catch (e) {
    ctx.postMessage({ type: "error", reqId: "reqId" in m ? m.reqId : undefined, message: (e as Error)?.message ?? String(e) });
  }
};
