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
import type { Answer, ToPyWorker } from "./types";

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

async function load() {
  const mod = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
  py = await mod.loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage(["numpy", "scipy", "pydantic"]);
  py.FS.mkdirTree("/chunkmirage/chunkmirage/ops");
  for (const [path, text] of Object.entries(FILES)) py.FS.writeFile(`/chunkmirage/${path}`, text);
  py.runPython(`import sys; sys.path.insert(0, "/chunkmirage")\n${GLUE}`);
}

/** Each view's output dtype, the leading axes its ops consume and the halo they need. */
async function plan(views: Extract<ToPyWorker, { type: "plan" }>["views"]) {
  if (!py) await load();
  const out: Record<string, { dtype: string; lead: number; halo: number[] }> = {};
  for (const [id, v] of Object.entries(views)) {
    const p = JSON.parse(py.globals.get("plan")(id, JSON.stringify(v.ops), v.shape, v.dtype, v.chunk));
    if (p.ndim !== 3) throw new Error(`view ${id}: a viewer shows three-axis volumes; its ops leave ${p.ndim} axes`);
    out[id] = p;
  }
  return out;
}

/** A chunk, as zarr v3 bytes (little endian, C order), from its input region. */
function compute(m: Extract<ToPyWorker, { type: "compute" }>): ArrayBuffer {
  const bytes = py.globals.get("compute")(m.view, new Uint8Array(m.data), m.readShape, m.inLo, m.inHi, m.outLo, m.outHi, m.full);
  const out = bytes.toJs() as Uint8Array;
  bytes.destroy();
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
}

ctx.onmessage = async ({ data: m }: MessageEvent<ToPyWorker>) => {
  try {
    if (m.type === "plan") ctx.postMessage({ reqId: m.reqId, value: await plan(m.views) } satisfies Answer);
    else if (m.type === "compute") { const body = compute(m); ctx.postMessage({ reqId: m.reqId, value: body } satisfies Answer, [body]); }
  } catch (e) {
    ctx.postMessage({ reqId: m.reqId, error: (e as Error)?.message ?? String(e) } satisfies Answer);
  }
};
