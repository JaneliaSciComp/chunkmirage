// The images a pipeline reads, in the browser, as chunkmirage's sources read them in Python:
// OME-Zarr (v2 or v3) and N5 groups, zarr arrays (funlib's, COSEM's, xarray's geo and climate
// data) and N5 arrays, Neuroglancer precomputed volumes, each placed as worldarray places it
// (placement.ts), and the stack://, flip:// and select wrappers. Each is a pyramid of three-axis levels (z, y, x, or
// an xarray array's own, such as time, lat, lon), with channels only for a stack; reads go
// through the service worker's store cache like every other fetch on the page.
import { asReals, fetchStore, image, openImage, prod, RegionReader, TYPED, zarr, type Image, type Numbers, type Settable, type TypedCtor } from "./ome";
import { multiscalesOf, placement, placeN5, placePrecomputed, placeZarr, type Attrs } from "./placement";
import { chunkName, compressedMorton, compressedSegmentation, Shards, type Fetch, type PrecomputedInfo } from "./precomputed";
import { fetchUrl } from "./urls";
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

/** Any source URL: stack://a|b, flip://image?axes=y, an OME-Zarr or N5 group, a zarr or N5
 * array, or a precomputed volume (precomputed://, as Neuroglancer writes it, or found by its
 * info); at gs:// or s3:// as well as https://, fetched as Neuroglancer fetches them
 * (urls.ts). Each is placed as worldarray places it (placement.ts). `select` pins
 * non-spatial axes, as a spec's select does ({c: 1, t: 0}). */
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
  if (url.startsWith("precomputed://")) return precomputedSource(url.slice("precomputed://".length));
  if (/\.tiff?$/i.test(url.split("?")[0])) return geotiffSource(url);
  try {
    return imageSource(url, await openImage(url), select);
  } catch (zarrError) {
    try { return await arraySource(url, select); } catch { /* not one either */ }
    try { return await n5Source(url); } catch { /* nor this */ }
    try { return await precomputedSource(url); } catch { throw zarrError; }
  }
}

/** A zarr image's levels as a source, its leading axes pinned by `select`; integers
 * CF-packed by `attrs` (scale_factor, add_offset) read as float32, missing ones NaN. */
function imageSource(url: string, img: Image, select: Record<string, number>, attrs: Attrs = {}): Source {
  for (const [axis, i] of Object.entries(select)) {
    if (!img.names.slice(0, img.lead).includes(axis)) throw new Error(`select ${axis}=${i}: ${url} has axes ${img.names}`);
  }
  const reader = new RegionReader(img, READ_BYTES), channel = select.c ?? 0;
  const pinned = Object.fromEntries(Object.entries(select).filter(([a]) => a !== "c"));
  const packed = "scale_factor" in attrs || "add_offset" in attrs;
  const scale = num(attrs.scale_factor ?? 1), offset = num(attrs.add_offset ?? 0);
  const missing = attrs._FillValue ?? attrs.missing_value ?? img.levels[0].arr.fillValue;
  return {
    url, dtype: packed ? "float32" : img.dtype, axes: img.axes.slice(-3).map((a) => ({ name: a.name, unit: a.unit ?? "" })), channels: 1,
    levels: img.levels.map((l) => ({ shape: l.shape, voxel: l.voxel, origin: l.origin, chunks: l.arr.chunks.slice(-3) })),
    read: async (li, _c, lo, hi, at) => {
      const { data } = await reader.read(li, channel, lo, hi, { ...pinned, ...at });
      if (!packed) return data;
      const raw = asReals(data), out = new Float32Array(raw.length), m = missing == null ? NaN : num(missing);
      for (let i = 0; i < raw.length; i++) out[i] = raw[i] === m ? NaN : raw[i] * scale + offset;
      return out;
    },
  };
}

// ------------------------------------------------ zarr arrays, xarray's among them
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
 * is evenly spaced and increasing (from its first two and last values); else unplaced:
 * voxels of 1 with the corner at the origin, so the first centre at 0.5. */
async function coordinate(parent: string, dim: string, n: number): Promise<{ step: number; first: number; unit: string }> {
  const none = { step: 1, first: 0.5, unit: "" };
  try {
    const c = await zarr.open(zarr.root(fetchStore(`${parent}/${dim}`)), { kind: "array" });
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

/** A zarr array of three axes or more, outside an OME-Zarr group: placed by its parent's
 * multiscales listing it, its own funlib or COSEM attributes or funlib's on its parent; an
 * array nothing places whose axes are named (xarray's _ARRAY_DIMENSIONS, zarr v3's
 * dimension_names) by its coordinate arrays beside it, CF-packed integers decoded. */
async function arraySource(url: string, select: Record<string, number>): Promise<Source> {
  const base = url.replace(/\/+$/, ""), cut = base.lastIndexOf("/"), parent = base.slice(0, cut);
  const arr = await zarr.open(zarr.root(fetchStore(base)), { kind: "array" }), n = arr.shape.length;
  if (n < 3) throw new Error(`${url}: an array of ${n} axes; three or more are read`);
  const group = await zarr.open(zarr.root(fetchStore(parent)), { kind: "group" }).then((g) => (g.attrs ?? {}) as Attrs, () => ({}));
  const attrs = (arr.attrs ?? {}) as Attrs, placed = placement(placeZarr(attrs, group, base.slice(cut + 1), n, arr.dimensionNames), n);
  if (!placed.placedBy && placed.named) {
    const coords = await Promise.all(placed.axes.map((d, a) => (a < n - 3 ? null : coordinate(parent, d, arr.shape[a]))));
    coords.forEach((c, a) => { if (c) { placed.voxel[a] = c.step; placed.origin[a] = c.first; placed.units[a] = c.unit; } });
  }
  return imageSource(url, image(base, [{ arr, placed }]), select, attrs);
}

// ------------------------------------------------ GeoTIFF
const SAMPLE_TYPES: Record<string, string> = {
  "1/8": "uint8", "1/16": "uint16", "1/32": "uint32", "2/8": "int8", "2/16": "int16", "2/32": "int32", "3/32": "float32", "3/64": "float64",
};

/** A (cloud-optimized) GeoTIFF, as chunkmirage_contrib.sources.geotiff reads it: its pages of
 * decreasing size the levels, as z, y, x with one z; tiles read by range and decoded once
 * into the block cache; y counting down the image (minus the northing); float no-data NaN. */
async function geotiffSource(url: string): Promise<Source> {
  const { fromUrl } = await import("geotiff");
  const tif = await fromUrl(fetchUrl(url));  // one object: its URL is fixed
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
async function json<T = Attrs>(url: string): Promise<T | null> {
  const r = await fetch(fetchUrl(url));
  return r.ok ? ((await r.json()) as T) : null;
}

const OME_UNIT: Record<string, string> = { nm: "nanometer", um: "micrometer", "µm": "micrometer" };

/** An N5 multiscale group (COSEM's multiscales, or s0, s1, ...) or one N5 array, of three
 * axes; levels in C order. */
async function n5Source(url: string): Promise<Source> {
  const base = url.replace(/\/+$/, ""), cut = base.lastIndexOf("/");
  const root = await json(`${base}/attributes.json`);
  if (!root) throw new Error(`${url}: neither OME-Zarr nor N5`);
  // an array on its own: its parent's attributes are its group's
  const group = root.dimensions ? (await json(`${base.slice(0, cut)}/attributes.json`)) ?? {} : root;
  let paths: string[] = root.dimensions ? [""] : multiscalesOf(root)[0]?.datasets.map((d: Attrs) => String(d.path));
  if (!paths) {
    paths = [];
    for (let i = 0; ; i++) { if (!(await json(`${base}/s${i}/attributes.json`))) break; paths.push(`s${i}`); }
  }
  if (!paths.length) throw new Error(`${url}: an N5 group without levels`);
  const levels = await Promise.all(paths.map(async (p) => {
    const dir = p ? `${base}/${p}` : base, a = p ? await json(`${dir}/attributes.json`) : root;
    if (!a?.dimensions || !a.blockSize || !a.dataType) throw new Error(`${dir}: not an N5 dataset`);
    if (a.dimensions.length !== 3) throw new Error(`${dir}: ${a.dimensions.length} axes; N5 of three is read here`);
    const kind = a.compression?.type ?? "raw";
    if (kind !== "gzip" && kind !== "raw") throw new Error(`${dir}: ${kind} compression is not read here (gzip or raw)`);
    const placed = placement(placeN5(a, group, p || base.slice(cut + 1), 3), 3);
    return { dir, dims: [...a.dimensions].reverse(), block: [...a.blockSize].reverse(), dtype: a.dataType as string, gzip: kind === "gzip", placed };
  }));
  const { axes, units } = levels[0].placed;
  const cache = new Blocks(READ_BYTES);
  return {
    url, dtype: levels[0].dtype, axes: axes.map((name, a) => ({ name, unit: OME_UNIT[units[a]] ?? units[a] })), channels: 1,
    levels: levels.map((l) => ({ shape: l.dims, voxel: l.placed.voxel, origin: l.placed.origin, chunks: l.block })),
    read: (li, _c, lo, hi) => {
      const l = levels[li];
      return assemble(TYPED[l.dtype] ?? Float32Array, l.block, lo, hi, (b) =>
        cache.get(`${li}/${b.join(",")}`, () => n5Block(`${l.dir}/${[...b].reverse().join("/")}`, l.dtype, l.gzip, l.block)));
    },
  };
}

/** One N5 block, z, y, x: its header (big endian: mode, axes, its size x first), then the
 * data, big endian too. A missing block is zeros. */
async function n5Block(url: string, dtype: string, gzip: boolean, block: number[]): Promise<{ data: Numbers; shape: number[] }> {
  const T = TYPED[dtype] ?? Float32Array;
  const r = await fetch(fetchUrl(url));
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
      (out as unknown as Settable).set(data.subarray(src + from[2], src + to[2]), dst + from[2]);  // a block is of `T`
    }
  });
  return out;
}

// ------------------------------------------------ Neuroglancer precomputed
/** A precomputed volume (chunkmirage.sources.tensorstore_source): its scales the levels,
 * x-first lists reversed to C order, voxel 0's centre half a voxel past `voxel_offset`'s
 * corner, in nanometers; channels first. Chunks unsharded (<key>/x0-x1_y0-y1_z0-z1) or
 * sharded (neuroglancer_uint64_sharded_v1, identity hash), encoded raw, as JPEG (a chunk's
 * z planes one tall image, decoded by the browser) or compressed_segmentation. A chunk not
 * stored is zeros. */
async function precomputedSource(url: string): Promise<Source> {
  const base = url.replace(/\/+$/, "");
  const info = await json<PrecomputedInfo>(`${base}/info`);
  if (!info?.scales?.length || (info["@type"] ?? "neuroglancer_multiscale_volume") !== "neuroglancer_multiscale_volume") {
    throw new Error(`${url}: not a precomputed volume`);
  }
  const dtype = info.data_type, T = TYPED[dtype], channels = info.num_channels ?? 1;
  if (!T) throw new Error(`${url}: data type ${dtype} is not read here`);
  const scales = info.scales.map((s) => {
    const encoding = s.encoding ?? "raw";
    if (!["raw", "jpeg", "compressed_segmentation"].includes(encoding)) throw new Error(`${base}/${s.key}: ${encoding} encoding is not read here`);
    const chunk = s.chunk_sizes[0], off = s.voxel_offset ?? [0, 0, 0];
    const dir = `${base}/${s.key.replace(/^\/+|\/+$/g, "")}`;
    return { s, encoding, chunk, off, dir, shards: s.sharding ? new Shards(dir, s.sharding, bytesAt) : null };
  });
  const levels = scales.map(({ s, chunk }) => {
    const { voxel, origin } = placePrecomputed(s);
    return { shape: [...s.size].reverse(), voxel: voxel!, origin: origin!, chunks: [...chunk].reverse() };
  });
  /** Chunk `b` (z, y, x in the grid) of scale `li`, all its channels (c, z, y, x). */
  const chunkAt = async (li: number, b: number[]): Promise<{ data: Numbers; shape: number[] }> => {
    const { s, encoding, chunk, off, dir, shards } = scales[li];
    const g = [...b].reverse(), lo = g.map((v, a) => v * chunk[a]), hi = lo.map((v, a) => Math.min(v + chunk[a], s.size[a]));
    const size = hi.map((v, a) => v - lo[a]), n = prod(size), shape = [...size].reverse();
    const bytes = shards
      ? await shards.chunk(compressedMorton(g, s.size.map((v, a) => Math.ceil(v / chunk[a]))))
      : await bytesAt(`${dir}/${chunkName(lo.map((v, a) => v + off[a]), hi.map((v, a) => v + off[a]))}`);
    if (!bytes) return { data: new T(n * channels), shape };
    if (encoding === "jpeg") return { data: await jpeg(bytes, size[0], size[1] * size[2], channels), shape };
    if (encoding === "compressed_segmentation") {
      if (dtype !== "uint32" && dtype !== "uint64") throw new Error(`${dir}: compressed_segmentation of ${dtype}`);
      return { data: compressedSegmentation(bytes, size, channels, s.compressed_segmentation_block_size ?? [8, 8, 8], dtype === "uint64"), shape };
    }
    if (bytes.byteLength !== n * channels * T.BYTES_PER_ELEMENT) throw new Error(`${dir}: a raw chunk of ${bytes.byteLength} bytes, not ${shape}`);
    return { data: new T(bytes), shape };
  };
  const cache = new Blocks(READ_BYTES);
  return {
    url, dtype, channels, levels,
    axes: ["z", "y", "x"].map((name) => ({ name, unit: "nanometer" })),
    read: (li, c, lo, hi) => assemble(T, levels[li].chunks, lo, hi, async (b) => {
      const { data, shape } = await cache.get(`${li}/${b.join(",")}`, () => chunkAt(li, b)), n = prod(shape);
      return { data: data.subarray(c * n, (c + 1) * n), shape };
    }),
  };
}

/** Bytes [start, end) of `url`, or all of it; null if it is not there. A server that
 * ignores the range sends all of it, and the range is cut from that. */
const bytesAt: Fetch = async (url, range) => {
  const r = await fetch(fetchUrl(url), range ? { headers: { range: `bytes=${range[0]}-${range[1] - 1}` } } : {});
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  return range && r.status === 200 ? buf.slice(range[0], range[1]) : buf;
};

/** A JPEG `width` × `height` pixels (a chunk's x by its y × z) as `channels` planes of uint8
 * (one: gray; three: red, green, blue), decoded by the browser without colour management. */
async function jpeg(bytes: ArrayBuffer, width: number, height: number, channels: number): Promise<Uint8Array> {
  const image = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  if (image.width !== width || image.height !== height) throw new Error(`a JPEG chunk ${image.width} × ${image.height}, not ${width} × ${height}`);
  const g = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true })!;
  g.drawImage(image, 0, 0); image.close();
  const rgba = g.getImageData(0, 0, width, height).data, n = width * height, out = new Uint8Array(n * channels);
  for (let c = 0; c < channels; c++) for (let i = 0; i < n; i++) out[c * n + i] = rgba[4 * i + c];
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
        else (out as unknown as Settable).set(d.subarray(src, src + n[2]), dst);  // `out` is of `d`'s kind
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
    url, dtype: a.dtype, axes: a.axes, channels: parts.length, levels: a.levels.slice(0, levels), ...(a.geo ? { geo: a.geo } : {}),
    read: (li, c, lo, hi) => parts[c].read(li, 0, lo, hi),
  };
}
