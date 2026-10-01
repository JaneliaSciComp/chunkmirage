// The images a pipeline reads, in the browser, as chunkmirage's sources read them in Python:
// OME-Zarr (v2 or v3) and N5 groups, xarray-style zarr arrays (geo and climate data), and the
// stack://, flip:// and select wrappers. Each is a pyramid of three-axis levels (z, y, x, or
// an xarray array's own, such as time, lat, lon), with channels only for a stack; reads go
// through the service worker's store cache like every other fetch on the page.
import { openImage, prod, RegionReader, TYPED, zarr, type Image, type Numbers, type TypedCtor } from "./ome";
import type { ViewAxis } from "./types";

export interface SourceLevel { shape: number[]; voxel: number[]; origin: number[]; chunks: number[] }

/** Where a source's grid sits in its map projection: the bounding box of its pixels'
 * corners, [x min, y min, x max, y max] (a GeoTIFF's; for a map viewer). */
export interface Geo { bbox: number[] }

export interface Source {
  url: string;
  geo?: Geo;
  dtype: string;
  axes: ViewAxis[];   // the three axes levels have, e.g. z, y, x in nanometers
  channels: number;   // 1, or the images of a stack (its leading c axis)
  levels: SourceLevel[];
  /** Voxels [lo, hi) of level `li` (z, y, x; within the level), channel `c`; `at` picks
   * other leading axes' entries (zarr sources: {t: 12}) over those the source was opened at. */
  read(li: number, c: number, lo: number[], hi: number[], at?: Record<string, number>): Promise<Numbers>;
}

const READ_BYTES = 384 * 2 ** 20;  // decoded pieces kept per image (the page's reader holds them all)

/** Any source URL: stack://a|b, flip://image?axes=y, an OME-Zarr or N5 group, or an
 * xarray-style zarr array. `select` pins non-spatial axes, as a spec's select does ({c: 1, t: 0}). */
export async function openSource(url: string, select: Record<string, number> = {}): Promise<Source> {
  if (url.startsWith("stack://")) {
    const parts = url.slice("stack://".length).split("|").filter(Boolean);
    if (parts.length < 2) throw new Error("stack:// needs two or more images separated by '|'");
    return stack(await Promise.all(parts.map((p) => openSource(p, select))), url);
  }
  if (url.startsWith("flip://")) {
    const rest = url.slice("flip://".length), q = rest.lastIndexOf("?");
    const axes = new URLSearchParams(q < 0 ? "" : rest.slice(q + 1)).get("axes");
    if (q < 0 || !axes) throw new Error("flip:// needs the axes to mirror: flip://<image>?axes=y");
    return flip(await openSource(rest.slice(0, q), select), axes.split(",").map((a) => a.trim()), url);
  }
  if (/\.tiff?$/i.test(url.split("?")[0])) return geotiffSource(url);
  try {
    return await zarrSource(url, select);
  } catch (zarrError) {
    try { return await xarraySource(url); } catch { /* not one either */ }
    try { return await n5Source(url); } catch { throw zarrError; }
  }
}

async function zarrSource(url: string, select: Record<string, number>): Promise<Source> {
  const img = await openImage(url);
  for (const [axis, i] of Object.entries(select)) {
    if (!img.names.slice(0, img.lead).includes(axis)) throw new Error(`select ${axis}=${i}: ${url} has axes ${img.names}`);
  }
  const reader = new RegionReader(img, READ_BYTES), channel = select.c ?? 0;
  const pinned = Object.fromEntries(Object.entries(select).filter(([a]) => a !== "c"));
  return {
    url, dtype: img.dtype, axes: img.axes.slice(-3).map((a) => ({ name: a.name, unit: a.unit ?? "" })), channels: 1,
    levels: img.levels.map((l) => ({ shape: l.shape, voxel: l.voxel, origin: l.origin, chunks: l.arr.chunks.slice(-3) })),
    read: async (li, _c, lo, hi, at) => (await reader.read(li, channel, lo, hi, { ...pinned, ...at })).data,
  };
}

// ------------------------------------------------ xarray-style zarr
const SECONDS: Record<string, number> = { day: 86400, hour: 3600, minute: 60, second: 1 };

/** A CF coordinate's units as [unit, factor to it]: "days since ..." and the like become
 * seconds, degrees unitless (chunkmirage.sources.tensorstore_source._cf_unit). */
function cfUnit(units: string): [string, number] {
  const u = units.trim().toLowerCase(), m = /^(day|hour|minute|second)s?\s+since\b/.exec(u);
  if (m) return ["s", SECONDS[m[1]]];
  return u.startsWith("degree") ? ["", 1] : [units.trim(), 1];
}

const num = (v: unknown) => Number(v as number | bigint);

/** Spacing, origin and unit of the coordinate array named `dim` beside the array, where it
 * is evenly spaced and increasing (from its first two and last values); else 1, 0, "". */
async function coordinate(parent: string, dim: string, n: number): Promise<{ step: number; first: number; unit: string }> {
  const none = { step: 1, first: 0, unit: "" };
  try {
    const c = await zarr.open(zarr.root(new zarr.FetchStore(`${parent}/${dim}`)), { kind: "array" });
    if (c.shape.length !== 1 || c.shape[0] !== n || n < 2) return none;
    const head = (await zarr.get(c, [zarr.slice(0, 2)])).data as ArrayLike<unknown>;
    const tail = (await zarr.get(c, [zarr.slice(n - 1, n)])).data as ArrayLike<unknown>;
    const first = num(head[0]), step = (num(tail[0]) - first) / (n - 1);
    if (!(step > 0) || Math.abs(num(head[1]) - first - step) > 0.01 * step) return none;
    const [unit, f] = cfUnit(String((c.attrs as { units?: string }).units ?? ""));
    // to the coordinates' own precision: float32 degrees 0.01 apart are not 0.0099999998
    const sig = (v: number) => Number(v.toPrecision(c.dtype === "float32" ? 6 : 12));
    return { step: sig(step * f), first: sig(first * f), unit };
  } catch { return none; }
}

/** A zarr array as xarray writes it: its axes named by _ARRAY_DIMENSIONS (or zarr v3's
 * dimension_names), voxel size and origin from its coordinate arrays, and CF-packed
 * integers (scale_factor, add_offset) read as float32 in their units, missing ones NaN. */
async function xarraySource(url: string): Promise<Source> {
  const base = url.replace(/\/+$/, ""), parent = base.slice(0, base.lastIndexOf("/"));
  const arr = await zarr.open(zarr.root(new zarr.FetchStore(base)), { kind: "array" });
  const attrs = arr.attrs as Record<string, unknown>;
  const dims = (attrs._ARRAY_DIMENSIONS as string[] | undefined) ?? arr.dimensionNames;
  if (!dims || dims.length !== 3 || arr.shape.length !== 3) throw new Error(`${url}: not a three-axis array with dimension names`);
  const coords = await Promise.all(dims.map((d, a) => coordinate(parent, d, arr.shape[a])));
  const voxel = coords.map((c) => c.step), origin = coords.map((c) => c.first);
  const img: Image = {
    url: base, axes: dims.map((name) => ({ name })), names: dims, dtype: arr.dtype, lead: 0,
    levels: [{ arr, fullShape: arr.shape, scale: voxel, shift: origin, shape: arr.shape, voxel, origin }],
  };
  const reader = new RegionReader(img, READ_BYTES);
  const packed = "scale_factor" in attrs || "add_offset" in attrs;
  const scale = num(attrs.scale_factor ?? 1), offset = num(attrs.add_offset ?? 0);
  const missing = attrs._FillValue ?? attrs.missing_value ?? arr.fillValue;
  return {
    url, dtype: packed ? "float32" : arr.dtype, channels: 1,
    axes: dims.map((name, a) => ({ name, unit: coords[a].unit })),
    levels: [{ shape: arr.shape, voxel, origin, chunks: arr.chunks }],
    read: async (li, _c, lo, hi) => {
      const raw = (await reader.read(li, 0, lo, hi)).data;
      if (!packed) return raw;
      const out = new Float32Array(raw.length), m = missing == null ? NaN : num(missing);
      for (let i = 0; i < raw.length; i++) out[i] = raw[i] === m ? NaN : raw[i] * scale + offset;
      return out;
    },
  };
}

// ------------------------------------------------ GeoTIFF
const SAMPLE_TYPES: Record<string, string> = {
  "1/8": "uint8", "1/16": "uint16", "1/32": "uint32", "2/8": "int8", "2/16": "int16", "2/32": "int32", "3/32": "float32", "3/64": "float64",
};

/** A (cloud-optimized) GeoTIFF, as chunkmirage.sources.geotiff reads it: its pages of
 * decreasing size the levels, as z, y, x with one z; tiles read by range and decoded once
 * into the block cache; y counting down the image (minus the northing); float no-data NaN. */
async function geotiffSource(url: string): Promise<Source> {
  const { fromUrl } = await import("geotiff");
  const tif = await fromUrl(url);
  const all = await Promise.all(Array.from({ length: await tif.getImageCount() }, (_, i) => tif.getImage(i)));
  const images = all.filter((im) => !((im.fileDirectory.getValue("NewSubfileType") ?? 0) & 4));  // no masks
  const base = images[0];
  if (base.getSamplesPerPixel() !== 1) throw new Error(`${url}: ${base.getSamplesPerPixel()} samples per pixel; one is read`);
  const dtype = SAMPLE_TYPES[`${base.getSampleFormat(0)}/${base.getBitsPerSample(0)}`];
  if (!dtype) throw new Error(`${url}: samples of format ${base.getSampleFormat(0)}, ${base.getBitsPerSample(0)} bits`);
  const [sx, sy] = base.getResolution(), [ox, oy] = base.getOrigin(), W = base.getWidth(), H = base.getHeight();
  const px = Math.abs(sx), py = Math.abs(sy), nodata = base.getGDALNoData();
  const levels = images.map((im) => {
    const vx = px * W / im.getWidth(), vy = py * H / im.getHeight();
    return { shape: [1, im.getHeight(), im.getWidth()], voxel: [1, vy, vx], origin: [0, -oy + vy / 2, ox + vx / 2], chunks: [1, im.getTileHeight(), im.getTileWidth()] };
  });
  const T = TYPED[dtype], cache = new Blocks(READ_BYTES);
  const tile = async (li: number, b: number[]) => {
    const im = images[li], [, th, tw] = levels[li].chunks;
    const x0 = b[2] * tw, y0 = b[1] * th, x1 = Math.min(x0 + tw, im.getWidth()), y1 = Math.min(y0 + th, im.getHeight());
    const r = await im.readRasters({ window: [x0, y0, x1, y1], samples: [0], interleave: true });
    const data = T === Float32Array || T === Float64Array ? new T((r as ArrayLike<number>).length) : (r as unknown as Numbers);
    if (data !== (r as unknown)) {  // float: no-data as NaN
      const v = r as ArrayLike<number>;
      for (let i = 0; i < v.length; i++) data[i] = v[i] === nodata ? NaN : v[i];
    }
    return { data, shape: [1, y1 - y0, x1 - x0] };
  };
  return {
    url, dtype, channels: 1, levels,
    axes: [{ name: "z", unit: "m" }, { name: "y", unit: "m" }, { name: "x", unit: "m" }],
    geo: { bbox: [ox, oy - H * py, ox + W * px, oy] },
    read: (li, _c, lo, hi) => assemble(T, levels[li].chunks, lo, hi, (b) => cache.get(`${li}/${b.join(",")}`, () => tile(li, b))),
  };
}

// ------------------------------------------------ N5
interface N5Attrs {
  dimensions?: number[]; blockSize?: number[]; dataType?: string; compression?: { type: string };
  multiscales?: { datasets: { path: string; transform?: { axes?: string[]; scale?: number[]; translate?: number[]; units?: string[] } }[] }[];
  pixelResolution?: { dimensions: number[]; unit: string } | number[]; resolution?: number[]; units?: string[];
  downsamplingFactors?: number[]; scales?: number[][];
}

async function json<T>(url: string): Promise<T | null> {
  const r = await fetch(url);
  return r.ok ? ((await r.json()) as T) : null;
}

/** An N5 multiscale group (COSEM's multiscales, or s0, s1, ...); levels in C order. */
async function n5Source(url: string): Promise<Source> {
  const base = url.replace(/\/+$/, "");
  const root = await json<N5Attrs>(`${base}/attributes.json`);
  if (!root) throw new Error(`${url}: neither OME-Zarr nor N5`);
  let paths = root.multiscales?.[0]?.datasets.map((d) => d.path);
  if (!paths) {
    paths = [];
    for (let i = 0; ; i++) { if (!(await json<N5Attrs>(`${base}/s${i}/attributes.json`))) break; paths.push(`s${i}`); }
  }
  if (!paths.length) throw new Error(`${url}: an N5 group without levels`);
  const levels = await Promise.all(paths.map(async (p, i) => {
    const a = await json<N5Attrs>(`${base}/${p}/attributes.json`);
    if (!a?.dimensions || !a.blockSize || !a.dataType) throw new Error(`${base}/${p}: not an N5 dataset`);
    const kind = a.compression?.type ?? "raw";
    if (kind !== "gzip" && kind !== "raw") throw new Error(`${base}/${p}: ${kind} compression is not read here (gzip or raw)`);
    const t = root.multiscales?.[0]?.datasets[i]?.transform;
    let voxel: number[], origin: number[];
    if (t?.scale && t.axes) {  // COSEM: axes named, z, y, x here
      const order = ["z", "y", "x"].map((n) => t.axes!.indexOf(n));
      voxel = order.map((k) => t.scale![k]); origin = order.map((k) => t.translate?.[k] ?? 0);
    } else {
      const res = Array.isArray(a.pixelResolution) ? a.pixelResolution : a.pixelResolution?.dimensions ?? a.resolution ?? [1, 1, 1];
      const f = a.downsamplingFactors ?? root.scales?.[i] ?? [1, 1, 1];
      voxel = [2, 1, 0].map((k) => res[k] * f[k]); origin = voxel.map((v, k) => (v - [2, 1, 0].map((j) => res[j])[k]) / 2);
    }
    return { path: p, dims: [...a.dimensions].reverse(), block: [...a.blockSize].reverse(), dtype: a.dataType, gzip: kind === "gzip", voxel, origin };
  }));
  const u = root.multiscales?.[0]?.datasets[0]?.transform?.units?.[0] ?? root.units?.[0] ?? "";
  const unit = u === "nm" ? "nanometer" : u === "um" ? "micrometer" : u;
  const cache = new Blocks(READ_BYTES);
  return {
    url, dtype: levels[0].dtype, axes: ["z", "y", "x"].map((name) => ({ name, unit })), channels: 1,
    levels: levels.map((l) => ({ shape: l.dims, voxel: l.voxel, origin: l.origin, chunks: l.block })),
    read: (li, _c, lo, hi) => {
      const l = levels[li];
      return assemble(TYPED[l.dtype] ?? Float32Array, l.block, lo, hi, (b) =>
        cache.get(`${li}/${b.join(",")}`, () => n5Block(`${base}/${l.path}/${[...b].reverse().join("/")}`, l.dtype, l.gzip, l.block)));
    },
  };
}

/** One N5 block, z, y, x: its header (big endian: mode, axes, its size x first), then the
 * data, big endian too. A missing block is zeros. */
async function n5Block(url: string, dtype: string, gzip: boolean, block: number[]): Promise<{ data: Numbers; shape: number[] }> {
  const T = TYPED[dtype] ?? Float32Array;
  const r = await fetch(url);
  if (r.status === 404) return { data: new T(prod(block)), shape: block };
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const buf = new DataView(await r.arrayBuffer());
  const mode = buf.getUint16(0), n = buf.getUint16(2);
  const size = Array.from({ length: n }, (_, i) => buf.getUint32(4 + 4 * i));
  let off = 4 + 4 * n + (mode === 1 ? 4 : 0);
  let bytes: ArrayBuffer = buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + buf.byteLength);
  if (gzip) bytes = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const shape = [...size].reverse(), count = prod(shape), data = new T(count), view = new DataView(bytes);
  const w = T.BYTES_PER_ELEMENT;
  const get = bigEndian(dtype, view);
  for (let i = 0; i < count; i++) data[i] = get(i * w);
  return { data, shape };
}

function bigEndian(dtype: string, v: DataView): (o: number) => number {
  switch (dtype) {
    case "uint8": return (o) => v.getUint8(o);
    case "int8": return (o) => v.getInt8(o);
    case "uint16": return (o) => v.getUint16(o);
    case "int16": return (o) => v.getInt16(o);
    case "uint32": return (o) => v.getUint32(o);
    case "int32": return (o) => v.getInt32(o);
    case "float32": return (o) => v.getFloat32(o);
    case "float64": return (o) => v.getFloat64(o);
    default: throw new Error(`N5 data type ${dtype} is not read here`);
  }
}

/** Decoded blocks, least recently used out past `maxBytes`. */
class Blocks {
  private map = new Map<string, Promise<{ data: Numbers; shape: number[] }>>();
  private bytes = 0;
  constructor(private maxBytes: number) {}
  get(key: string, load: () => Promise<{ data: Numbers; shape: number[] }>) {
    let hit = this.map.get(key);
    if (hit) { this.map.delete(key); this.map.set(key, hit); return hit; }
    hit = load().catch((e) => { this.map.delete(key); throw e; });
    this.map.set(key, hit);
    void hit.then((b) => {
      this.bytes += b.data.byteLength;
      while (this.bytes > this.maxBytes && this.map.size > 1) {
        const [k] = this.map.keys(); const old = this.map.get(k)!; this.map.delete(k);
        void old.then((o) => { this.bytes -= o.data.byteLength; });
      }
    }, () => {});
    return hit;
  }
}

/** Region [lo, hi) assembled from blocks of `block` voxels (C order), each from `blockAt`. */
async function assemble(T: TypedCtor, block: number[], lo: number[], hi: number[],
  blockAt: (b: number[]) => Promise<{ data: Numbers; shape: number[] }>): Promise<Numbers> {
  const shape = hi.map((v, a) => v - lo[a]), out = new T(prod(shape));
  const b0 = lo.map((v, a) => Math.floor(v / block[a])), b1 = hi.map((v, a) => Math.floor((v - 1) / block[a]));
  const wanted: number[][] = [];
  for (let z = b0[0]; z <= b1[0]; z++) for (let y = b0[1]; y <= b1[1]; y++) for (let x = b0[2]; x <= b1[2]; x++) wanted.push([z, y, x]);
  const got = await Promise.all(wanted.map(blockAt));
  wanted.forEach((b, n) => {
    const { data, shape: bs } = got[n], start = b.map((v, a) => v * block[a]);
    const from = start.map((s, a) => Math.max(s, lo[a])), to = start.map((s, a) => Math.min(s + bs[a], hi[a]));
    for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) {
      const src = ((z - start[0]) * bs[1] + (y - start[1])) * bs[2] - start[2];
      const dst = ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] - lo[2];
      out.set(data.subarray(src + from[2], src + to[2]) as ArrayLike<number>, dst + from[2]);
    }
  });
  return out;
}

// ------------------------------------------------ wrappers
/** `inner` mirrored along `axes`, every level in place (chunkmirage.sources.flip). */
function flip(inner: Source, axes: string[], url: string): Source {
  const names = inner.axes.map((a) => a.name), which = axes.map((a) => names.indexOf(a));
  if (which.some((k) => k < 0)) throw new Error(`flip:// axes=${axes}: ${inner.url} has axes ${names}`);
  return {
    ...inner, url,
    read: async (li, c, lo, hi) => {
      const shape = inner.levels[li].shape;
      const mlo = lo.map((v, a) => (which.includes(a) ? shape[a] - hi[a] : v));
      const mhi = hi.map((v, a) => (which.includes(a) ? shape[a] - lo[a] : v));
      const d = await inner.read(li, c, mlo, mhi);
      const n = hi.map((v, a) => v - lo[a]), out = new (d.constructor as TypedCtor)(d.length);
      const [fz, fy, fx] = [0, 1, 2].map((a) => which.includes(a));
      for (let z = 0; z < n[0]; z++) for (let y = 0; y < n[1]; y++) {
        const sz = fz ? n[0] - 1 - z : z, sy = fy ? n[1] - 1 - y : y;
        const src = (sz * n[1] + sy) * n[2], dst = (z * n[1] + y) * n[2];
        if (fx) for (let x = 0; x < n[2]; x++) out[dst + x] = d[src + n[2] - 1 - x];
        else out.set(d.subarray(src, src + n[2]) as ArrayLike<number>, dst);
      }
      return out;
    },
  };
}

/** Images on one grid as the channels of one source (chunkmirage.sources.stack). */
function stack(parts: Source[], url: string): Source {
  const [a] = parts;
  const levels = Math.min(...parts.map((p) => p.levels.length));
  for (const p of parts.slice(1)) for (let i = 0; i < levels; i++)
    if (p.levels[i].shape.join() !== a.levels[i].shape.join()) throw new Error(`stack:// level ${i}: ${p.url} is on a different grid from ${a.url}`);
  return {
    url, dtype: a.dtype, axes: a.axes, channels: parts.length, levels: a.levels.slice(0, levels),
    read: (li, c, lo, hi) => parts[c].read(li, 0, lo, hi),
  };
}
