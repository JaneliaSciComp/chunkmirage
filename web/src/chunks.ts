// A chunk worker: computes chunks of a registered volume (the moving image resampled onto
// the fixed image's grid through the affine and, for a solved view, the field), as
// chunkmirage's scene:// resampler does on a server. The page hands it Neuroglancer's
// chunk requests, relayed by the service worker, and passes the bytes back.
import { fieldAt, openImage, RegionReader, type Image, type TypedCtor } from "./ome";
import type { Affine, ControlGrid, FromWorker, Lattice, LevelGrid, ToWorker, ViewKind } from "./types";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const CACHE_BYTES = 128 * 2 ** 20;  // decoded moving-image pieces kept per worker (the fetched bytes: the service worker)

interface View { kind: ViewKind; affine: Affine; grid: ControlGrid | null; refined: Lattice[]; levels: Map<number, number> }
let moving: Image, reader: RegionReader, fixedLevels: LevelGrid[], chunkShape: number[], Typed: TypedCtor = Float32Array;
const views = new Map<string, View>();
const fields = new Map<number, { resolve: (g: ControlGrid | null) => void; reject: (e: Error) => void }>();  // asked of the page
let fieldSeq = 0;
const post = (m: FromWorker, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);

ctx.onmessage = async ({ data: m }: MessageEvent<ToWorker>) => {
  try {
    if (m.type === "setup") {
      moving = await openImage(m.moving);
      reader = new RegionReader(moving, CACHE_BYTES);
      fixedLevels = m.fixedLevels; chunkShape = m.chunkShape; Typed = reader.Typed;
      post({ type: "ready" });
    } else if (m.type === "view") {
      views.set(m.id, { kind: m.kind, affine: m.affine, grid: m.grid, refined: m.refined ?? [], levels: new Map() });
    } else if (m.type === "field") {
      const f = fields.get(m.reqId);
      fields.delete(m.reqId);
      if (m.error) f?.reject(new Error(m.error)); else f?.resolve(m.grid);
    } else if (m.type === "drop") {
      for (const id of m.ids) views.delete(id);
    } else if (m.type === "chunk") {
      const body = await chunk(m);
      post({ type: "chunk", reqId: m.reqId, body }, [body]);
    }
  } catch (e) {
    post({ type: "error", reqId: m.type === "chunk" ? m.reqId : undefined, message: String((e as Error)?.stack ?? e) });
  }
};

/** The field a chunk of `view` at `level` goes through: the solved grid, or, for a level
 * whose field is fitted where it is viewed, the window of that level's lattice the chunk
 * touches (the cells its voxel centres fall in, and the point past them), which the page
 * fits block by block on request. */
function fieldFor(id: string, view: View, level: number, fl: LevelGrid, start: number[], size: number[], chunk: number): Promise<ControlGrid | null> {
  const levels = view.refined.map((l) => l.level);
  if (!levels.length || level > Math.max(...levels)) return Promise.resolve(view.grid);
  const lat = view.refined.find((l) => l.level === Math.max(level, Math.min(...levels)))!;
  const lo = [0, 0, 0], hi = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const x0 = fl.origin[a] + start[a] * fl.voxel[a], x1 = x0 + (size[a] - 1) * fl.voxel[a];
    const g0 = (Math.min(x0, x1) - lat.origin[a]) / lat.spacing[a], g1 = (Math.max(x0, x1) - lat.origin[a]) / lat.spacing[a];
    lo[a] = Math.min(Math.max(Math.floor(g0), 0), lat.shape[a] - 2);
    hi[a] = Math.min(Math.max(Math.floor(g1) + 2, lo[a] + 2), lat.shape[a]);
  }
  return new Promise((resolve, reject) => {
    const reqId = ++fieldSeq;
    fields.set(reqId, { resolve, reject });
    post({ type: "field", reqId, chunk, id, level: lat.level, lo, hi });
  });
}

/** The field itself on the fixed grid: its three components (z, y, x), physical units. */
function fieldChunk(grid: ControlGrid, fl: LevelGrid, start: number[], size: number[]): ArrayBuffer {
  const cs = chunkShape, n = cs[0] * cs[1] * cs[2];
  const out = new Float32Array(3 * n), d = [0, 0, 0];
  for (let i = 0; i < size[0]; i++) {
    const x0 = fl.origin[0] + (start[0] + i) * fl.voxel[0];
    for (let j = 0; j < size[1]; j++) {
      const x1 = fl.origin[1] + (start[1] + j) * fl.voxel[1];
      for (let k = 0; k < size[2]; k++) {
        fieldAt(grid, x0, x1, fl.origin[2] + (start[2] + k) * fl.voxel[2], d);
        const p = (i * cs[1] + j) * cs[2] + k;
        out[p] = d[0]; out[n + p] = d[1]; out[2 * n + p] = d[2];
      }
    }
  }
  return out.buffer;
}

/** The moving level a view reads for output level `fl`: the coarsest still at least as fine as
 * one output voxel, measured through the view's mapping at the grid's centre, as
 * chunkmirage's scene._pick_level (a zoomed-out view, or an affine that shrinks, reads a
 * finer or coarser level than the voxel sizes alone suggest). */
function pickLevel(view: View, fl: LevelGrid): number {
  const m0 = moving.levels[0], d = [0, 0, 0];
  const toLevel0 = (idx: number[]) => {  // output voxel index -> moving level-0 index
    const x = idx.map((v, a) => fl.origin[a] + v * fl.voxel[a]);
    if (view.grid) { fieldAt(view.grid, x[0], x[1], x[2], d); for (let a = 0; a < 3; a++) x[a] += d[a]; }
    return view.affine.map((r, a) => (r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + r[3] - m0.origin[a]) / m0.voxel[a]);
  };
  const centre = fl.shape.map((n) => (n - 1) / 2), extent = [0, 0, 0];
  for (let c = 0; c < 3; c++) {  // one output voxel's span along each moving axis
    const plus = centre.slice(), minus = centre.slice();
    plus[c] += 0.5; minus[c] -= 0.5;
    const a = toLevel0(plus), b = toLevel0(minus);
    for (let r = 0; r < 3; r++) extent[r] += Math.abs(a[r] - b[r]);
  }
  let best = 0;
  moving.levels.forEach((l, i) => {
    if (l.voxel.every((v, a) => v / m0.voxel[a] <= Math.max(extent[a], 1) * 1.01)) best = i;
  });
  return best;
}

async function chunk({ reqId, id, level, channel, index }: { reqId: number; id: string; level: number; channel: number; index: number[] }): Promise<ArrayBuffer> {
  const view = views.get(id);
  const fl = fixedLevels[level], cs = chunkShape;
  const start = index.map((i, a) => i * cs[a]);
  const size = start.map((s, a) => Math.max(0, Math.min(cs[a], fl.shape[a] - s)));
  const out = new Typed(cs[0] * cs[1] * cs[2]);  // a whole chunk: zero past the array's edge
  if (!view || size.some((n) => n === 0)) return view?.kind === "field" ? new Float32Array(3 * out.length).buffer : out.buffer;
  const grid = await fieldFor(id, view, level, fl, start, size, reqId);
  if (view.kind === "field") return grid ? fieldChunk(grid, fl, start, size) : new Float32Array(3 * out.length).buffer;
  if (!view.levels.has(level)) view.levels.set(level, pickLevel(view, fl));
  const li = view.levels.get(level)!, ml = moving.levels[li];
  const A = view.affine, d = [0, 0, 0];
  const n = size[0] * size[1] * size[2];
  // per chunk, not shared: a worker has several chunks in flight while their regions load
  const m = new Float32Array(3 * n), inside = new Uint8Array(n);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let p = 0, any = false;
  for (let i = 0; i < size[0]; i++) {
    const x0 = fl.origin[0] + (start[0] + i) * fl.voxel[0];
    for (let j = 0; j < size[1]; j++) {
      const x1 = fl.origin[1] + (start[1] + j) * fl.voxel[1];
      for (let k = 0; k < size[2]; k++, p++) {
        const x2 = fl.origin[2] + (start[2] + k) * fl.voxel[2];
        if (grid) fieldAt(grid, x0, x1, x2, d);
        const q0 = x0 + d[0], q1 = x1 + d[1], q2 = x2 + d[2];
        let ok = true;
        for (let a = 0; a < 3; a++) {
          const r = A[a];
          const v = (r[0] * q0 + r[1] * q1 + r[2] * q2 + r[3] - ml.origin[a]) / ml.voxel[a];
          m[3 * p + a] = v;
          // a voxel owns [-0.5, 0.5) around its centre; points outside the image stay 0
          if (!(v >= -0.5 && v < ml.shape[a] - 0.5)) ok = false;
        }
        if (!ok) continue;
        inside[p] = 1; any = true;
        for (let a = 0; a < 3; a++) {
          const f = Math.floor(m[3 * p + a]);
          if (f < lo[a]) lo[a] = f;
          if (f + 1 > hi[a]) hi[a] = f + 1;
        }
      }
    }
  }
  if (!any) return out.buffer;
  const rlo = lo.map((v) => Math.max(0, v)), rhi = hi.map((v, a) => Math.min(ml.shape[a], v + 1));
  const { data: src, shape: rs } = await reader.read(li, channel, rlo, rhi);
  const round = Typed !== Float32Array && Typed !== Float64Array;
  const [tz, ty, tx] = ml.shape.map((v) => v - 1), [lz, ly, lx] = rlo, [, ry, rx] = rs;
  p = 0;
  for (let i = 0; i < size[0]; i++) for (let j = 0; j < size[1]; j++) for (let k = 0; k < size[2]; k++, p++) {
    if (!inside[p]) continue;
    // trilinear, neighbours clamped to the image (map_coordinates' mode="nearest")
    const cz = m[3 * p], cy = m[3 * p + 1], cx = m[3 * p + 2];
    const z0 = Math.floor(cz), y0 = Math.floor(cy), x0 = Math.floor(cx), fz = cz - z0, fy = cy - y0, fx = cx - x0;
    let val = 0;
    for (let q = 0; q < 8; q++) {
      const oz = q & 1, oy = (q >> 1) & 1, ox = (q >> 2) & 1;
      const w = (oz ? fz : 1 - fz) * (oy ? fy : 1 - fy) * (ox ? fx : 1 - fx);
      if (!w) continue;
      const z = Math.min(Math.max(z0 + oz, 0), tz) - lz;
      const y = Math.min(Math.max(y0 + oy, 0), ty) - ly;
      const x = Math.min(Math.max(x0 + ox, 0), tx) - lx;
      val += w * src[(z * ry + y) * rx + x];
    }
    out[(i * cs[1] + j) * cs[2] + k] = round ? Math.round(val) : val;
  }
  return out.buffer;
}
