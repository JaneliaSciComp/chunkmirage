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

/** Trilinear value of a displacement grid at (x0, x1, x2), written into `out`. */
export function fieldAt(grid: ControlGrid, x0: number, x1: number, x2: number, out: number[]): number[] {
  const { shape: G, origin: o, spacing: s, values: u } = grid;
  const c = [(x0 - o[0]) / s[0], (x1 - o[1]) / s[1], (x2 - o[2]) / s[2]];
  const i = [0, 0, 0], f = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const g = Math.min(Math.max(c[a], 0), G[a] - 1);
    i[a] = Math.max(0, Math.min(Math.floor(g), G[a] - 2));
    f[a] = g - i[a];
  }
  out[0] = out[1] = out[2] = 0;
  for (let k = 0; k < 8; k++) {
    const oz = k & 1, oy = (k >> 1) & 1, ox = (k >> 2) & 1;
    const w = (oz ? f[0] : 1 - f[0]) * (oy ? f[1] : 1 - f[1]) * (ox ? f[2] : 1 - f[2]);
    if (!w) continue;
    const idx = (((i[0] + oz) * G[1] + i[1] + oy) * G[2] + i[2] + ox) * 3;
    out[0] += w * u[idx]; out[1] += w * u[idx + 1]; out[2] += w * u[idx + 2];
  }
  return out;
}
