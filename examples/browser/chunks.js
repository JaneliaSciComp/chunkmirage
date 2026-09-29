// A chunk worker: computes chunks of a registered volume (the moving image resampled onto
// the fixed image's grid through the affine and, for a solved view, the field), as
// chunkmirage's scene:// resampler does on a server. The page hands it Neuroglancer's
// chunk requests, relayed by the service worker, and passes the bytes back.
import { fieldAt, leadIndex, nearestLevel, openImage, zarr } from "./ome.js";

const BLOCK = 64;                  // moving-image blocks are read and cached this big
const CACHE_BYTES = 256 * 2 ** 20;  // decoded blocks kept per worker
const TYPED = { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, int8: Int8Array, int16: Int16Array, int32: Int32Array, float32: Float32Array, float64: Float64Array };

let moving = null, fixedLevels = null, chunkShape = null, Typed = null, blockBytes = 0;
const views = new Map();           // view id -> {affine, grid | null}
const blocks = new Map();          // block key -> Promise<{data, shape}>, in use order
let cached = 0;

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "setup") {
      moving = await openImage(m.moving);
      fixedLevels = m.fixedLevels; chunkShape = m.chunkShape;
      Typed = TYPED[moving.dtype] ?? Float32Array;
      blockBytes = BLOCK ** 3 * Typed.BYTES_PER_ELEMENT;
      self.postMessage({ type: "ready" });
    } else if (m.type === "view") {
      views.set(m.id, { affine: m.affine, grid: m.grid });
    } else if (m.type === "chunk") {
      const body = await chunk(m);
      self.postMessage({ type: "chunk", reqId: m.reqId, body }, [body]);
    }
  } catch (e) {
    self.postMessage({ type: "error", reqId: m.reqId, message: String(e?.stack ?? e) });
  }
};

async function block(li, channel, b) {
  const key = `${li}/${channel}/${b.join(",")}`;
  let hit = blocks.get(key);
  if (hit) { blocks.delete(key); blocks.set(key, hit); return hit; }
  const lvl = moving.levels[li];
  const lo = b.map((v) => v * BLOCK), hi = lo.map((v, a) => Math.min(v + BLOCK, lvl.shape[a]));
  hit = zarr.get(lvl.arr, [...leadIndex(moving, channel), ...lo.map((v, a) => zarr.slice(v, hi[a]))])
    .then((r) => ({ data: r.data, shape: hi.map((v, a) => v - lo[a]) }))
    .catch((e) => { blocks.delete(key); throw e; });  // a failed read is retried next time
  blocks.set(key, hit);
  cached += blockBytes;
  while (cached > CACHE_BYTES && blocks.size > 1) {
    blocks.delete(blocks.keys().next().value);
    cached -= blockBytes;
  }
  return hit;
}

/** The moving image's voxels [lo, hi) of level li, one channel, from cached blocks. */
async function region(li, channel, lo, hi) {
  const shape = hi.map((v, a) => v - lo[a]);
  const out = new Typed(shape[0] * shape[1] * shape[2]);
  const b0 = lo.map((v) => Math.floor(v / BLOCK)), b1 = hi.map((v) => Math.floor((v - 1) / BLOCK));
  const wanted = [];
  for (let z = b0[0]; z <= b1[0]; z++) for (let y = b0[1]; y <= b1[1]; y++) for (let x = b0[2]; x <= b1[2]; x++) wanted.push([z, y, x]);
  const got = await Promise.all(wanted.map((b) => block(li, channel, b)));
  wanted.forEach((b, n) => {
    const { data, shape: bs } = got[n];
    const start = b.map((v) => v * BLOCK);
    const from = start.map((s, a) => Math.max(s, lo[a])), to = start.map((s, a) => Math.min(s + bs[a], hi[a]));
    for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) {
      const src = ((z - start[0]) * bs[1] + (y - start[1])) * bs[2] - start[2];
      const dst = ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] - lo[2];
      out.set(data.subarray(src + from[2], src + to[2]), dst + from[2]);
    }
  });
  return { data: out, shape };
}

async function chunk({ id, level, channel, index }) {
  const view = views.get(id);
  const fl = fixedLevels[level], cs = chunkShape;
  const out = new Typed(cs[0] * cs[1] * cs[2]);  // a whole chunk: zero past the array's edge
  const start = index.map((i, a) => i * cs[a]);
  const size = start.map((s, a) => Math.max(0, Math.min(cs[a], fl.shape[a] - s)));
  if (!view || size.some((n) => n === 0)) return out.buffer;
  const li = nearestLevel(moving, fl.voxel), ml = moving.levels[li];
  const A = view.affine, d = [0, 0, 0];
  const n = size[0] * size[1] * size[2];
  const m = new Float32Array(3 * n), inside = new Uint8Array(n);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let p = 0, any = false;
  for (let i = 0; i < size[0]; i++) {
    const x0 = fl.origin[0] + (start[0] + i) * fl.voxel[0];
    for (let j = 0; j < size[1]; j++) {
      const x1 = fl.origin[1] + (start[1] + j) * fl.voxel[1];
      for (let k = 0; k < size[2]; k++, p++) {
        const x2 = fl.origin[2] + (start[2] + k) * fl.voxel[2];
        if (view.grid) fieldAt(view.grid, x0, x1, x2, d);
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
  const { data: src, shape: rs } = await region(li, channel, rlo, rhi);
  const round = Typed !== Float32Array && Typed !== Float64Array;
  const top = ml.shape.map((v) => v - 1);
  p = 0;
  for (let i = 0; i < size[0]; i++) for (let j = 0; j < size[1]; j++) for (let k = 0; k < size[2]; k++, p++) {
    if (!inside[p]) continue;
    // trilinear, neighbours clamped to the image (map_coordinates' mode="nearest")
    let val = 0;
    const c = [m[3 * p], m[3 * p + 1], m[3 * p + 2]];
    const f0 = c.map((v) => Math.floor(v)), fr = c.map((v, a) => v - f0[a]);
    for (let q = 0; q < 8; q++) {
      const o = [q & 1, (q >> 1) & 1, (q >> 2) & 1];
      const w = (o[0] ? fr[0] : 1 - fr[0]) * (o[1] ? fr[1] : 1 - fr[1]) * (o[2] ? fr[2] : 1 - fr[2]);
      if (!w) continue;
      const z = Math.min(Math.max(f0[0] + o[0], 0), top[0]) - rlo[0];
      const y = Math.min(Math.max(f0[1] + o[1], 0), top[1]) - rlo[1];
      const x = Math.min(Math.max(f0[2] + o[2], 0), top[2]) - rlo[2];
      val += w * src[(z * rs[1] + y) * rs[2] + x];
    }
    out[(i * cs[1] + j) * cs[2] + k] = round ? Math.round(val) : val;
  }
  return out.buffer;
}
