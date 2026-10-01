// A pipeline worker: chunkmirage's own Python (its ops and chunkmirage.fused, the code a
// server's pipeline stage runs) in Pyodide. The page has it plan the views it serves, then
// hands it each chunk's input region (read by the page's reader) and gets back the bytes a
// zarr v3 chunk holds, computed as the Python server computes it.
import pyBase from "../../src/chunkmirage/ops/base.py?raw";
import pyCombine from "../../src/chunkmirage/ops/combine.py?raw";
import pyCore from "../../src/chunkmirage/core.py?raw";
import pyFilters from "../../src/chunkmirage/ops/filters.py?raw";
import pyFused from "../../src/chunkmirage/fused.py?raw";
import pyOps from "../../src/chunkmirage/ops/__init__.py?raw";
import pyPointwise from "../../src/chunkmirage/ops/pointwise.py?raw";
import pySegment from "../../src/chunkmirage/ops/segment.py?raw";
import pyTerrain from "../../src/chunkmirage/ops/terrain.py?raw";
import pyCache from "../../src/chunkmirage/cache.py?raw";
import pySourceBase from "../../src/chunkmirage/sources/base.py?raw";
import pySynthetic from "../../src/chunkmirage/sources/synthetic.py?raw";
import type { Answer, ToPyWorker } from "./types";

export const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const FILES: Record<string, string> = {
  "chunkmirage/__init__.py": '"""chunkmirage\'s ops and fused stages, for the browser engine."""\n',
  "chunkmirage/core.py": pyCore, "chunkmirage/fused.py": pyFused, "chunkmirage/ops/__init__.py": pyOps,
  "chunkmirage/ops/base.py": pyBase, "chunkmirage/ops/pointwise.py": pyPointwise, "chunkmirage/ops/filters.py": pyFilters,
  "chunkmirage/ops/segment.py": pySegment, "chunkmirage/ops/combine.py": pyCombine, "chunkmirage/ops/terrain.py": pyTerrain,
  // the sources the worker computes itself (synthetic://), not read by the page's reader
  "chunkmirage/cache.py": pyCache, "chunkmirage/sources/__init__.py": '"""The computed sources, for the browser engine."""\n',
  "chunkmirage/sources/base.py": pySourceBase, "chunkmirage/sources/synthetic.py": pySynthetic,
};
const GLUE = `
import json
import numpy as np
from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec

VIEWS = {}
SOURCES = {}

def describe(url):
    from chunkmirage.sources.synthetic import open_synthetic
    ms = SOURCES.setdefault(url, open_synthetic(url))
    i = ms[0].info
    return json.dumps({
        "dtype": i.dtype.name, "channels": 1,
        "axes": [{"name": a, "unit": u} for a, u in zip(i.axes, i.units)],
        "levels": [{"shape": list(l.info.shape), "voxel": list(l.info.voxel_size), "origin": list(l.info.translation)} for l in ms],
    })

def _info(shape, dtype, chunk, voxel):
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")
    return ArrayInfo(shape=tuple(shape), dtype=np.dtype(dtype), chunk_shape=tuple(shape[: n - 3]) + tuple(chunk),
                     voxel_size=(1.0,) * (n - 3) + tuple(voxel), units=("",) * n, axes=axes)

def plan(view, ops, shape, dtype, chunk, voxel, source=None):
    ops = [op_from_spec(s) for s in json.loads(ops)]
    out, lead, halo = fused.plan(_info(list(shape), dtype, list(chunk), list(voxel)), ops)
    if source:
        describe(source)
    VIEWS[view] = (ops, dtype, list(chunk), source)
    return json.dumps({"dtype": out.dtype.name, "lead": lead, "halo": list(halo), "ndim": out.ndim})

def compute(view, level, data, read_shape, in_lo, in_hi, out_lo, out_hi, full_shape, voxel):
    ops, dtype, chunk, source = VIEWS[view]
    full = tuple(full_shape)
    lead = len(full) - 3
    in_box = Box((0,) * lead + tuple(in_lo), full[:lead] + tuple(in_hi))
    if data is None:  # a source computed here: the padded block, as a server stage reads it
        block = SOURCES[source][level].read_padded(in_box, edge=True)
    else:
        block = np.frombuffer(data.to_py(), dtype=np.dtype(dtype)).reshape(tuple(read_shape))
        block = fused.pad_edge(block, in_box, full)
    out = fused.plan(_info(full, dtype, chunk, list(voxel)), ops)[0]  # the level's voxels: for_level
    result = fused.run(ops, block, in_box, Box(tuple(out_lo), tuple(out_hi)), out)
    # a zarr chunk is always whole: one at the edge of the array is padded with the fill value
    result = np.pad(result, [(0, c - n) for c, n in zip(chunk, result.shape)])
    return np.ascontiguousarray(result).astype(result.dtype.newbyteorder("<"), copy=False).tobytes()
`;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null;

async function load() {
  const mod = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
  py = await mod.loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "scipy", "pydantic"]);
  for (const [path, text] of Object.entries(FILES)) {
    py.FS.mkdirTree(`/chunkmirage/${path.slice(0, path.lastIndexOf("/"))}`);
    py.FS.writeFile(`/chunkmirage/${path}`, text);
  }
  py.runPython(`import sys; sys.path.insert(0, "/chunkmirage")\n${GLUE}`);
}

/** Each view's output dtype, the leading axes its ops consume and the halo they need. */
async function plan(views: Extract<ToPyWorker, { type: "plan" }>["views"]) {
  if (!py) await load();
  const out: Record<string, { dtype: string; lead: number; halo: number[] }> = {};
  for (const [id, v] of Object.entries(views)) {
    const p = JSON.parse(py.globals.get("plan")(id, JSON.stringify(v.ops), v.shape, v.dtype, v.chunk, v.voxel, v.source));
    if (p.ndim !== 3) throw new Error(`view ${id}: a viewer shows three-axis volumes; its ops leave ${p.ndim} axes`);
    out[id] = p;
  }
  return out;
}

/** A chunk, as zarr v3 bytes (little endian, C order), from its input region. */
function compute(m: Extract<ToPyWorker, { type: "compute" }>): ArrayBuffer {
  const bytes = py.globals.get("compute")(m.view, m.level, m.data ? new Uint8Array(m.data) : undefined, m.readShape, m.inLo, m.inHi, m.outLo, m.outHi, m.full, m.voxel);
  const out = bytes.toJs() as Uint8Array;
  bytes.destroy();
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
}

ctx.onmessage = async ({ data: m }: MessageEvent<ToPyWorker>) => {
  try {
    if (m.type === "describe") { if (!py) await load(); ctx.postMessage({ reqId: m.reqId, value: JSON.parse(py.globals.get("describe")(m.source)) } satisfies Answer); }
    else if (m.type === "plan") ctx.postMessage({ reqId: m.reqId, value: await plan(m.views) } satisfies Answer);
    else if (m.type === "compute") { const body = compute(m); ctx.postMessage({ reqId: m.reqId, value: body } satisfies Answer, [body]); }
  } catch (e) {
    // Pyodide's file-system errors are objects without a message: say what they are
    const message = (e as Error)?.message ?? (e as { name?: string })?.name ?? JSON.stringify(e);
    ctx.postMessage({ reqId: m.reqId, error: String(message) } satisfies Answer);
  }
};
