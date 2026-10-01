// OME-Zarr reading, shared by the page and its chunk workers: multiscale metadata, whole
// levels for the solve, and regions for resampling. zarrita.js reads zarr v2/v3,
// sharded or not, with the usual codecs (zstd and blosc through numcodecs).
import * as zarr from "zarrita";
import type { ControlGrid } from "./types";

export { zarr };

export type Numbers =
  | Uint8Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Int32Array | Float32Array | Float64Array;

export interface Axis {
  name: string;
  type?: string;
  unit?: string;
}

export interface ImageLevel {
  arr: zarr.Array<zarr.DataType, zarr.Readable>;
  fullShape: number[];
  scale: number[];
  shift: number[];
  shape: number[];
  voxel: number[];
  origin: number[];
}

export interface Image {
  url: string;
  axes: Axis[];
  names: string[];
  levels: ImageLevel[];
  dtype: string;
  lead: number;  // axes before z, y, x (time, channel)
}

interface Multiscale {
  axes: (string | Axis)[];
  datasets: { path: string; coordinateTransformations?: { type: string; scale?: number[]; translation?: number[] }[] }[];
}

/** Multiscale metadata of the image at `url`, and its levels' arrays. */
export async function openImage(url: string): Promise<Image> {
  const store = new zarr.FetchStore(url.replace(/\/+$/, ""));
  const root = zarr.root(store);
  const group = await zarr.open(root, { kind: "group" });
  const attrs = group.attrs as { ome?: { multiscales?: Multiscale[] }; multiscales?: Multiscale[] };
  const ms = (attrs.ome?.multiscales ?? attrs.multiscales)?.[0];
  if (!ms) throw new Error(`${url}: no OME-Zarr multiscales metadata`);
  const axes = ms.axes.map((a) => (typeof a === "string" ? { name: a } : a));
  const names = axes.map((a) => a.name);
  if (names.slice(-3).join("") !== "zyx") throw new Error(`${url}: axes ${names} must end in z, y, x`);
  const levels = await Promise.all(ms.datasets.map(async (d) => {
    let scale = names.map(() => 1), shift = names.map(() => 0);
    for (const t of d.coordinateTransformations ?? []) {  // composed in order
      if (t.type === "scale" && t.scale) { const s = t.scale; scale = scale.map((v, i) => v * s[i]); shift = shift.map((v, i) => v * s[i]); }
      if (t.type === "translation" && t.translation) { const tr = t.translation; shift = shift.map((v, i) => v + tr[i]); }
    }
    const arr = await zarr.open(root.resolve(d.path), { kind: "array" });
    return {
      arr, fullShape: arr.shape, scale, shift,
      shape: arr.shape.slice(-3), voxel: scale.slice(-3), origin: shift.slice(-3),
    };
  }));
  return { url: url.replace(/\/+$/, ""), axes, names, levels, dtype: levels[0].arr.dtype, lead: names.length - 3 };
}

/** Index of the lead axes (time, channel) to read: `channel` on c, 0 on anything else. */
export function leadIndex(img: Image, channel: number): number[] {
  return img.names.slice(0, img.lead).map((n) => (n === "c" ? channel : 0));
}

/** One channel of level `i`, all of its spatial extent. */
export async function readLevel(img: Image, i: number, channel: number) {
  const lvl = img.levels[i];
  const r = await zarr.get(lvl.arr, [...leadIndex(img, channel), null, null, null]);
  return { data: r.data as Numbers, shape: lvl.shape, voxel: lvl.voxel, origin: lvl.origin };
}

export type TypedCtor = { new (n: number): Numbers & { buffer: ArrayBuffer }; BYTES_PER_ELEMENT: number };
export const TYPED: Record<string, TypedCtor> = {
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, int8: Int8Array, int16: Int16Array, int32: Int32Array,
  float32: Float32Array, float64: Float64Array,
};
const MAX_PIECE = 1 << 22;  // voxels: a store chunk bigger than this (a shard, say) is read in 64³ pieces

interface Piece { data: Numbers; shape: number[] }

/** Regions of an image's levels, one channel at a time, assembled from pieces aligned to the
 * store's own chunks and kept (least recently used out past `maxBytes`), so each chunk is
 * decoded once however many regions overlap it: neighbouring chunks and blocks, their
 * halos, different outputs. One per image in each context (the page, each chunk worker);
 * the service worker below them keeps the fetched bytes for all of them. */
export class RegionReader {
  readonly Typed: TypedCtor;
  private pieces = new Map<string, Promise<Piece>>();  // in use order
  private bytes = 0;
  constructor(readonly img: Image, private maxBytes: number) { this.Typed = TYPED[img.dtype] ?? Float32Array; }

  private pieceShape(li: number): number[] {
    const c = this.img.levels[li].arr.chunks.slice(-3);
    return prod(c) <= MAX_PIECE ? c : [64, 64, 64];
  }

  private piece(li: number, channel: number, b: number[], P: number[]): Promise<Piece> {
    const key = `${li}/${channel}/${b.join(",")}`;
    let hit = this.pieces.get(key);
    if (hit) { this.pieces.delete(key); this.pieces.set(key, hit); return hit; }
    const lvl = this.img.levels[li];
    const lo = b.map((v, a) => v * P[a]), hi = lo.map((v, a) => Math.min(v + P[a], lvl.shape[a]));
    hit = zarr.get(lvl.arr, [...leadIndex(this.img, channel), ...lo.map((v, a) => zarr.slice(v, hi[a]))])
      .then((r) => ({ data: r.data as Numbers, shape: hi.map((v, a) => v - lo[a]) }))
      .catch((e) => { this.pieces.delete(key); throw e; });  // a failed read is retried next time
    this.pieces.set(key, hit);
    this.bytes += prod(P) * this.Typed.BYTES_PER_ELEMENT;
    while (this.bytes > this.maxBytes && this.pieces.size > 1) {
      const [oldest] = this.pieces.keys();
      this.pieces.delete(oldest);
      this.bytes -= prod(this.pieceShape(Number(oldest.split("/")[0]))) * this.Typed.BYTES_PER_ELEMENT;
    }
    return hit;
  }

  /** Voxels [lo, hi) of level `li`, one channel, and that region's own origin. */
  async read(li: number, channel: number, lo: number[], hi: number[]) {
    const lvl = this.img.levels[li], P = this.pieceShape(li), shape = hi.map((v, a) => v - lo[a]);
    const out = new this.Typed(prod(shape));
    const b0 = lo.map((v, a) => Math.floor(v / P[a])), b1 = hi.map((v, a) => Math.floor((v - 1) / P[a]));
    const wanted: number[][] = [];
    for (let z = b0[0]; z <= b1[0]; z++) for (let y = b0[1]; y <= b1[1]; y++) for (let x = b0[2]; x <= b1[2]; x++) wanted.push([z, y, x]);
    const got = await Promise.all(wanted.map((b) => this.piece(li, channel, b, P)));
    wanted.forEach((b, n) => {
      const { data, shape: bs } = got[n];
      const start = b.map((v, a) => v * P[a]);
      const from = start.map((s, a) => Math.max(s, lo[a])), to = start.map((s, a) => Math.min(s + bs[a], hi[a]));
      for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) {
        const src = ((z - start[0]) * bs[1] + (y - start[1])) * bs[2] - start[2];
        const dst = ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] - lo[2];
        out.set(data.subarray(src + from[2], src + to[2]) as ArrayLike<number>, dst + from[2]);
      }
    });
    return { data: out, shape, voxel: lvl.voxel, origin: lvl.origin.map((o, a) => o + lo[a] * lvl.voxel[a]) };
  }
}

export const prod = (a: number[]) => a.reduce((x, y) => x * y, 1);

/** numpy's (linear) percentiles, on every step-th value as chunkmirage.registration does. */
export function percentiles(data: ArrayLike<number>, ps: number[]): number[] {
  const step = Math.max(1, Math.floor(data.length / 1e6));
  const s = new Float32Array(Math.ceil(data.length / step));
  for (let i = 0, j = 0; i < data.length; i += step) s[j++] = data[i];
  s.sort();
  return ps.map((p) => {
    const x = (p / 100) * (s.length - 1), lo = Math.floor(x);
    return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (x - lo);
  });
}

/** The level of `img` whose voxel size is nearest `voxel` (by log ratio). */
export function nearestLevel(img: Image, voxel: number[]): number {
  let best = 0, bestD = Infinity;
  img.levels.forEach((l, i) => {
    const d = l.voxel.reduce((s, v, a) => s + Math.abs(Math.log(v / voxel[a])), 0);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

/** Trilinear value of a displacement grid at (x0, x1, x2), written into `out`; nearest
 * edge value beyond the grid. Called per voxel, so it allocates nothing. */
export function fieldAt(grid: ControlGrid, x0: number, x1: number, x2: number, out: number[]): number[] {
  const { shape: G, origin: o, spacing: s, values: u } = grid;
  // each axis: the grid cell (clamped to the grid) and the fraction across it
  const g0 = Math.min(Math.max((x0 - o[0]) / s[0], 0), G[0] - 1), i0 = Math.max(0, Math.min(Math.floor(g0), G[0] - 2)), f0 = g0 - i0;
  const g1 = Math.min(Math.max((x1 - o[1]) / s[1], 0), G[1] - 1), i1 = Math.max(0, Math.min(Math.floor(g1), G[1] - 2)), f1 = g1 - i1;
  const g2 = Math.min(Math.max((x2 - o[2]) / s[2], 0), G[2] - 1), i2 = Math.max(0, Math.min(Math.floor(g2), G[2] - 2)), f2 = g2 - i2;
  out[0] = out[1] = out[2] = 0;
  for (let k = 0; k < 8; k++) {
    const oz = k & 1, oy = (k >> 1) & 1, ox = (k >> 2) & 1;
    const w = (oz ? f0 : 1 - f0) * (oy ? f1 : 1 - f1) * (ox ? f2 : 1 - f2);
    if (!w) continue;
    const idx = (((i0 + oz) * G[1] + i1 + oy) * G[2] + i2 + ox) * 3;
    out[0] += w * u[idx]; out[1] += w * u[idx + 1]; out[2] += w * u[idx + 2];
  }
  return out;
}
