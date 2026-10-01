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
import pyMeshes from "../../src/chunkmirage/meshes.py?raw";
import pyStitching from "../../src/chunkmirage/stitching.py?raw";
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
  "chunkmirage/meshes.py": pyMeshes,  // meshes of a view, made where a fragment is fetched
  "chunkmirage/stitching.py": pyStitching,  // the stitch page's steps, one call each
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

def _info(shape, dtype, chunk, voxel, origin=None, unit=""):
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")
    return ArrayInfo(shape=tuple(shape), dtype=np.dtype(dtype), chunk_shape=tuple(shape[: n - 3]) + tuple(chunk),
                     voxel_size=(1.0,) * (n - 3) + tuple(voxel), units=(unit,) * n, axes=axes,
                     translation=(0.0,) * (n - 3) + tuple(origin or (0.0,) * 3))

def plan(view, ops, shape, dtype, chunk, voxel, source=None):
    ops = [op_from_spec(s) for s in json.loads(ops)]
    out, lead, halo = fused.plan(_info(list(shape), dtype, list(chunk), list(voxel)), ops)
    if source:
        describe(source)
    VIEWS[view] = (ops, dtype, list(chunk), source)
    return json.dumps({"dtype": out.dtype.name, "lead": lead, "halo": list(halo), "ndim": out.ndim})

def _mesh(m, result, box, shape, dtype, chunk, voxel, origin, unit):
    """legacy: a fragment (Neuroglancer's encoding). octree: from the whole coarsest level,
    the multi-resolution nodes (level, z, y, x each), header (count, fragment bytes,
    quantization bits) first, then the index. node: a node's mesh, header (vertices,
    triangles) then both as uint32, for the page to encode with Draco."""
    import struct
    from chunkmirage import meshes
    mode, levels = m.pop("mode", "legacy"), m.pop("levels", None)
    m.pop("raw", None)
    spec = meshes.MeshSpec(**{k: v for k, v in m.items() if k not in ("core_lo", "core_hi")})
    info = _info(list(shape), dtype, chunk, list(voxel), list(origin), unit)
    if mode == "mask":  # part of the coarsest level: inside or not, and the surface band, on
        # the part's core (the box the page asked for has a border of meshes.BAND around it)
        core = Box(tuple(m.pop("core_lo")), tuple(m.pop("core_hi")))
        grown = [min(a, meshes.BAND) for a in core.start]
        if any(g < meshes.BAND and b + g != a for g, a, b in zip(grown, core.start, box.start)):
            raise ValueError("a mask part needs a border of meshes.BAND voxels")
        inside = np.asarray(result) >= spec.threshold
        return np.ascontiguousarray(inside[core.slices()], np.uint8).tobytes() + meshes.surface_band(inside, core).astype(np.uint8).tobytes()
    if mode == "node":
        v, f = meshes.multires_fragment(spec, result, box, info, chunk)
        return struct.pack("<II", len(v), len(f)) + v.astype("<u4").tobytes() + f.astype("<u4").tobytes()
    return meshes.fragment(spec, result, box, info)

def octree(band, shape, mesh, chunk, unit):
    """A multi-resolution mesh's nodes (level, z, y, x each; header: their count, the
    fragments' size and the quantization bits) and index, from the whole coarsest level's
    surface band (uint8, 1 near the surface; meshes.surface_band)."""
    import struct
    from chunkmirage import meshes
    m = json.loads(mesh)
    m.pop("mode", None)
    levels = m.pop("levels")
    spec = meshes.MeshSpec(**{**m, "threshold": 1})  # the mask is inside or not
    infos = [_info(l["shape"], "uint8", list(chunk), l["voxel"], l["origin"], unit) for l in levels]
    block = np.frombuffer(band.to_py(), np.uint8).reshape(tuple(shape))
    lods = meshes.lod_levels(infos, spec)
    nodes = meshes.multires_nodes(spec, block, infos, list(chunk))
    flat = np.array([[lv, *n] for lv, ns in zip(lods, nodes) for n in ns], "<i4").reshape(-1, 4)
    head = struct.pack("<III", len(flat), meshes.FRAGMENT_BYTES, meshes.BITS)
    return head + flat.tobytes() + meshes.multires_index(nodes, infos, lods, list(chunk))

def compute(view, level, data, read_shape, in_lo, in_hi, out_lo, out_hi, full_shape, voxel, origin=None, mesh=None, unit=""):
    ops, dtype, chunk, source = VIEWS[view]
    full = tuple(full_shape)
    lead = len(full) - 3
    in_box = Box((0,) * lead + tuple(in_lo), full[:lead] + tuple(in_hi))
    if mesh is not None and json.loads(mesh).get("raw"):  # the page's own mask: no ops to run
        result = np.frombuffer(data.to_py(), np.uint8).reshape(tuple(read_shape))
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], "uint8", chunk, voxel, origin, unit)
    if data is None:  # a source computed here: the padded block, as a server stage reads it
        block = SOURCES[source][level].read_padded(in_box, edge=True)
    else:
        block = np.frombuffer(data.to_py(), dtype=np.dtype(dtype)).reshape(tuple(read_shape))
        block = fused.pad_edge(block, in_box, full)
    out = fused.plan(_info(full, dtype, chunk, list(voxel)), ops)[0]  # the level's voxels: for_level
    result = fused.run(ops, block, in_box, Box(tuple(out_lo), tuple(out_hi)), out)
    if mesh is not None:  # a mesh of these voxels, not a chunk
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], out.dtype.name, chunk, voxel, origin, unit)
    # a zarr chunk is always whole: one at the edge of the array is padded with the fill value
    result = np.pad(result, [(0, c - n) for c, n in zip(chunk, result.shape)])
    return np.ascontiguousarray(result).astype(result.dtype.newbyteorder("<"), copy=False).tobytes()
`;

const STITCH = `
def stitch(fn, args, arrays):
    """One step of chunkmirage.stitching for the stitch page: JSON in, JSON (or a chunk's
    bytes) out, arrays as raw bytes."""
    from chunkmirage import stitching as S
    a = json.loads(args)
    p = S.StitchParams(**a.get("params", {}))
    if fn == "tiles":
        return json.dumps(S.tiles_from_bdv(a["xml"], a["base"], a["channel"]))
    if fn == "overlaps":  # each overlap, and the region of each of its two tiles that holds it
        out = []
        for i, j, lo, hi in S.overlaps(a["tiles"], p.margin):
            out.append({"tiles": [i, j], "lo": lo.tolist(), "hi": hi.tolist(),
                        "regions": [S.region(a["tiles"][t], p.level, lo, hi) for t in (i, j)]})
        return json.dumps(out)
    if fn == "points":
        block = np.frombuffer(arrays[0].to_py(), np.dtype(a["dtype"])).reshape(a["shape"])
        pts = S.points_in(a["tile"], p.level, block, a["start"], a["voxel"], a["lo"], a["hi"], p)
        return json.dumps(pts.round(3).tolist())
    if fn == "register":
        found = S.register(a["tiles"], a["points"], p, a.get("fixed", 0))
        found["grids"] = S.grids(a["tiles"], found["placements"])
        found["reference"] = S.compare(a["tiles"], found["placements"])
        return json.dumps(found)
    if fn == "regions":  # where each tile holds voxels [lo, hi) of a fused level
        lo, hi = S.scene_box(a["grid"], a["lo"], a["hi"])
        return json.dumps([S.region(t, a["level"], lo, hi, pl) for t, pl in zip(a["tiles"], a["placements"])])
    if fn == "fuse":
        blocks, k = [], 0
        for r in a["regions"]:
            if r is None:
                blocks.append(None)
                continue
            shape = [b - s for s, b in zip(*r)]
            blocks.append((np.frombuffer(arrays[k].to_py(), np.dtype(a["dtype"])).reshape(shape), r[0]))
            k += 1
        out = S.fuse(a["tiles"], a["placements"], a["level"], a["grid"], a["lo"], a["hi"], blocks, p.blend, a["dtype"])
        out = np.pad(out, [(0, c - n) for c, n in zip(a["chunk"], out.shape)])  # zarr chunks are whole
        return np.ascontiguousarray(out).astype(out.dtype.newbyteorder("<"), copy=False).tobytes()
    raise ValueError(f"no stitching step {fn}")
`;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null;
let loaded: Promise<void> | null = null;

async function load(packages: string[] = []) {
  const mod = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
  py = await mod.loadPyodide({ indexURL: PYODIDE });
  // with the packages the page's ops declare; any other when first imported (withPackages)
  await py.loadPackage(["numpy", "pydantic", ...packages]);
  for (const [path, text] of Object.entries(FILES)) {
    py.FS.mkdirTree(`/chunkmirage/${path.slice(0, path.lastIndexOf("/"))}`);
    py.FS.writeFile(`/chunkmirage/${path}`, text);
  }
  py.runPython(`import sys; sys.path.insert(0, "/chunkmirage")\n${GLUE}\n${STITCH}`);
}

/** Each view's output dtype, the leading axes its ops consume and the halo they need. */
async function plan(views: Extract<ToPyWorker, { type: "plan" }>["views"], packages?: string[]) {
  if (!py) await (loaded ??= load(packages));
  const out: Record<string, { dtype: string; lead: number; halo: number[] }> = {};
  for (const [id, v] of Object.entries(views)) {
    const p = JSON.parse(py.globals.get("plan")(id, JSON.stringify(v.ops), v.shape, v.dtype, v.chunk, v.voxel, v.source));
    if (p.ndim !== 3) throw new Error(`view ${id}: a viewer shows three-axis volumes; its ops leave ${p.ndim} axes`);
    out[id] = p;
  }
  return out;
}

/** A stitching step: JSON back, or with `bytes` a chunk's bytes. */
async function stitch(m: Extract<ToPyWorker, { type: "stitch" }>): Promise<unknown> {
  if (!py) await (loaded ??= load());
  const arrays = (m.arrays ?? []).map((b) => new Uint8Array(b));
  const out = await withPackages(() => py.globals.get("stitch")(m.fn, m.args, arrays));
  if (typeof out === "string") return JSON.parse(out);
  const bytes = out.toJs() as Uint8Array;
  out.destroy();
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

/** Pyodide packages the ops import inside their functions, loaded the first time one does:
 * most demos never need scipy's 30 MB, nor a mesh scikit-image's. */
const LAZY: Record<string, string> = { scipy: "scipy", skimage: "scikit-image" };
const loading = new Map<string, Promise<void>>();

/** `call()`, again after loading what it failed to import: a package not loaded yet, or one
 * imported while it was still being installed (whose half-imported modules are dropped). */
async function withPackages<T>(call: () => T): Promise<T> {
  for (let tries = 0; ; tries++) {
    try { return call(); } catch (e) {
      const text = String((e as Error)?.message ?? "");
      const name = (/No module named '(\w+)/.exec(text) ?? /`(\w+)` install you are using seems to be broken/.exec(text))?.[1];
      if (!name || !LAZY[name] || tries > 1) throw e;
      if (!loading.has(name)) loading.set(name, py.loadPackage([LAZY[name]]));
      await loading.get(name);
      py.runPython(`import sys\nfor m in [m for m in sys.modules if m == "${name}" or m.startswith("${name}.")]: del sys.modules[m]`);
    }
  }
}

// Draco's own encoder, compiled for the web: a multi-resolution mesh's fragments are Draco
const DRACO = "https://cdn.jsdelivr.net/gh/google/draco@1.5.7/javascript/";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let draco: Promise<any> | null = null;
function loadDraco() {
  return draco ??= (async () => {
    const text = await (await fetch(`${DRACO}draco_encoder.js`)).text();
    const factory = new Function(`${text}\nreturn DracoEncoderModule;`)();
    return factory({ locateFile: (f: string) => DRACO + f });
  })();
}

/** A node's mesh (header, then uint32 vertices and triangles, from chunkmirage.meshes) as a
 * Draco fragment, its integer positions kept as they are, padded to `size` bytes. */
async function encodeNode(raw: Uint8Array, size: number, bits: number): Promise<ArrayBuffer> {
  const head = new DataView(raw.buffer, raw.byteOffset, 8), nv = head.getUint32(0, true), nf = head.getUint32(4, true);
  const out = new Uint8Array(size);
  if (!nf) return out.buffer;
  const M = await loadDraco();
  const verts = new Uint32Array(raw.slice(8, 8 + nv * 12).buffer), faces = new Uint32Array(raw.slice(8 + nv * 12).buffer);
  const encoder = new M.Encoder(), builder = new M.MeshBuilder(), mesh = new M.Mesh(), data = new M.DracoInt8Array();
  try {
    builder.AddFacesToMesh(mesh, nf, faces);
    builder.AddFloatAttributeToMesh(mesh, M.POSITION, nv, 3, new Float32Array(verts));
    encoder.SetAttributeExplicitQuantization(M.POSITION, bits, 3, [0, 0, 0], 2 ** bits - 1);  // the integers as they are
    encoder.SetSpeedOptions(3, 3);
    const n = encoder.EncodeMeshToDracoBuffer(mesh, data);
    if (n > size) throw new Error(`a mesh fragment took ${n} bytes, over ${size}`);
    for (let i = 0; i < n; i++) out[i] = data.GetValue(i);
  } finally {
    M.destroy(data); M.destroy(mesh); M.destroy(builder); M.destroy(encoder);
  }
  return out.buffer;
}

/** A chunk, as zarr v3 bytes (little endian, C order), from its input region; or with
 * `mesh`, a mesh of the region (chunkmirage.meshes; a multi-resolution node's encoded and
 * padded to `mesh.size` bytes). */
async function compute(m: Extract<ToPyWorker, { type: "compute" }>): Promise<ArrayBuffer> {
  const mesh = m.mesh ? JSON.stringify({ ...m.mesh, size: undefined, bits: undefined }) : undefined;  // the encoding's, not Python's
  const bytes = await withPackages(() => py.globals.get("compute")(m.view, m.level, m.data ? new Uint8Array(m.data) : undefined, m.readShape, m.inLo, m.inHi, m.outLo, m.outHi, m.full, m.voxel, m.origin, mesh, m.unit ?? ""));
  const out = bytes.toJs() as Uint8Array;
  bytes.destroy();
  if (m.mesh?.mode === "node") return encodeNode(out, m.mesh.size!, m.mesh.bits!);
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
}

ctx.onmessage = async ({ data: m }: MessageEvent<ToPyWorker>) => {
  try {
    if (m.type === "describe") { if (!py) await (loaded ??= load()); ctx.postMessage({ reqId: m.reqId, value: JSON.parse(py.globals.get("describe")(m.source)) } satisfies Answer); }
    else if (m.type === "plan") ctx.postMessage({ reqId: m.reqId, value: await plan(m.views, m.packages) } satisfies Answer);
    else if (m.type === "octree") {
      const out = await withPackages(() => py.globals.get("octree")(new Uint8Array(m.band), m.shape, JSON.stringify(m.mesh), m.chunk, m.unit));
      const bytes = out.toJs() as Uint8Array;
      out.destroy();
      const body = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
      ctx.postMessage({ reqId: m.reqId, value: body } satisfies Answer, [body]);
    }
    else if (m.type === "stitch") { const v = await stitch(m); ctx.postMessage({ reqId: m.reqId, value: v } satisfies Answer, v instanceof ArrayBuffer ? [v] : []); }
    else if (m.type === "compute") { const body = await compute(m); ctx.postMessage({ reqId: m.reqId, value: body } satisfies Answer, [body]); }
  } catch (e) {
    // Pyodide's file-system errors are objects without a message: say what they are
    const message = (e as Error)?.message ?? (e as { name?: string })?.name ?? JSON.stringify(e);
    ctx.postMessage({ reqId: m.reqId, error: String(message) } satisfies Answer);
  }
};
