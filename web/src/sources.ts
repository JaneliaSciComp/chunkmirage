// The images a pipeline reads, in the browser, as chunkmirage's sources read them in Python:
// OME-Zarr (v2 or v3) and N5 groups, and the stack://, flip:// and select wrappers. Each is a
// pyramid of z, y, x levels, with channels only for a stack; reads go through the service
// worker's store cache like every other fetch on the page.
import { openImage, prod, RegionReader, TYPED, type Numbers, type TypedCtor } from "./ome";

export interface SourceLevel { shape: number[]; voxel: number[]; origin: number[]; chunks: number[] }

export interface Source {
  url: string;
  dtype: string;
  unit: string;       // of the spatial axes, e.g. "nanometer"
  channels: number;   // 1, or the images of a stack (its leading c axis)
  levels: SourceLevel[];
  /** Voxels [lo, hi) of level `li` (z, y, x; within the level), channel `c`. */
  read(li: number, c: number, lo: number[], hi: number[]): Promise<Numbers>;
}

const READ_BYTES = 96 * 2 ** 20;  // decoded pieces kept per image, per context

/** Any source URL: stack://a|b, flip://image?axes=y, an OME-Zarr or an N5 group. `select`
 * pins non-spatial axes, as a spec's select does ({c: 1, t: 0}). */
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
  try {
    return await zarrSource(url, select);
  } catch (zarrError) {
    try { return await n5Source(url); } catch { throw zarrError; }
  }
}

async function zarrSource(url: string, select: Record<string, number>): Promise<Source> {
  const img = await openImage(url);
  for (const [axis, i] of Object.entries(select)) {
    if (!img.names.slice(0, img.lead).includes(axis)) throw new Error(`select ${axis}=${i}: ${url} has axes ${img.names}`);
    if (axis !== "c" && i !== 0) throw new Error(`select ${axis}=${i}: only the first entry of ${axis} is read here`);
  }
  const reader = new RegionReader(img, READ_BYTES), channel = select.c ?? 0;
  return {
    url, dtype: img.dtype, unit: img.axes[img.axes.length - 1].unit ?? "", channels: 1,
    levels: img.levels.map((l) => ({ shape: l.shape, voxel: l.voxel, origin: l.origin, chunks: l.arr.chunks.slice(-3) })),
    read: async (li, _c, lo, hi) => (await reader.read(li, channel, lo, hi)).data,
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
  const unit = root.multiscales?.[0]?.datasets[0]?.transform?.units?.[0] ?? root.units?.[0] ?? "";
  const cache = new Blocks(READ_BYTES);
  return {
    url, dtype: levels[0].dtype, unit: unit === "nm" ? "nanometer" : unit === "um" ? "micrometer" : unit, channels: 1,
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
  const which = axes.map((a) => ["z", "y", "x"].indexOf(a));
  if (which.some((k) => k < 0)) throw new Error(`flip:// axes=${axes}: spatial axes only (z, y, x)`);
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
    url, dtype: a.dtype, unit: a.unit, channels: parts.length, levels: a.levels.slice(0, levels),
    read: (li, c, lo, hi) => parts[c].read(li, 0, lo, hi),
  };
}
