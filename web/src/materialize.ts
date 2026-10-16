// A region of a view the engine serves, written into a store as an OME-Zarr of its own
// (Engine.materialize): a box of the view computed once and kept, to open again later or to
// copy where other tools read it. The store is a directory (one the visitor picks with the
// File System Access API's showDirectoryPicker, in Chrome and Edge, or the origin's private
// file system) or memory. What is written is what the engine serves for the view
// (layout.ts), one level of it, or exactly a box of it, its chunks compressed. Here are the
// parts that need no engine, so Node's tests run them: the chunks a box covers, the files
// written, gzip, the array's chunks stitched from the view's for a box that does not start
// on chunk edges, and the loop that has the chunks made and writes them, from any producer
// (the engine's queue on the page, a fake one in the tests).
import { TYPED } from "./asread.ts";  // with its extension: Node's runner resolves no other
import { array2, array3, attrs2, chunkKey, dimensionNames, group3 } from "./layout.ts";
import type { ViewInfo } from "./types.ts";

/** Where materialize writes: files by "/"-separated path relative to the store's root. */
export interface WritableStore {
  put(path: string, bytes: Uint8Array): Promise<void>;
}

/** The names of `path`, "/" apart; empty ones (a leading, trailing or doubled "/") left out.
 * A store's paths go down from its root, so "." and ".." are refused. */
function names(path: string): string[] {
  const parts = path.split("/").filter(Boolean);
  if (parts.some((p) => p === "." || p === "..")) throw new Error(`${path}: a store's paths go down from its root, with no "." or ".."`);
  return parts;
}

/** A directory picked with showDirectoryPicker({mode: "readwrite"}) (or any
 * FileSystemDirectoryHandle, e.g. OPFS's navigator.storage.getDirectory()) as a
 * WritableStore: subdirectories made as needed (getDirectoryHandle(name, {create: true}),
 * each once however many files go in it), each file written whole with createWritable(),
 * which writes to a swap file that replaces the file when closed, so a file is never seen
 * half written. `root` (optional) is a subdirectory path under it, made as needed (e.g.
 * "mito.zarr"). */
export function directoryStore(dir: FileSystemDirectoryHandle, root?: string): WritableStore {
  const base = root ? names(root) : [];
  const dirs = new Map<string, Promise<FileSystemDirectoryHandle>>();
  const at = (parts: string[]): Promise<FileSystemDirectoryHandle> => {
    if (!parts.length) return Promise.resolve(dir);
    const key = parts.join("/");
    let made = dirs.get(key);
    if (!made) {
      made = at(parts.slice(0, -1)).then((parent) => parent.getDirectoryHandle(parts[parts.length - 1], { create: true }));
      dirs.set(key, made);
      made.catch(() => dirs.delete(key));  // tried again by the next file in it
    }
    return made;
  };
  return {
    async put(path, bytes) {
      const parts = [...base, ...names(path)];
      if (parts.length === base.length) throw new Error(`"${path}" is not a file's path`);
      const file = await (await at(parts.slice(0, -1))).getFileHandle(parts[parts.length - 1], { create: true });
      const w = await file.createWritable();
      try {
        await w.write(bytes as Uint8Array<ArrayBuffer>);
        await w.close();
      } catch (e) {
        await w.abort().catch(() => {});  // the swap file discarded, the file as it was
        throw e;
      }
    },
  };
}

/** An in-memory store (Map<string, Uint8Array>, by path without a leading "/"), for tests
 * and for zipping small results. Each file is a copy of the bytes put. */
export function memoryStore(): WritableStore & { files: Map<string, Uint8Array> } {
  const files = new Map<string, Uint8Array>();
  return {
    files,
    async put(path, bytes) {
      const parts = names(path);
      if (!parts.length) throw new Error(`"${path}" is not a file's path`);
      files.set(parts.join("/"), bytes.slice());
    },
  };
}

/** How a materialize is going: of the written array's `total` chunks, those `done` (written,
 * or skipped as empty), and of them those `written`, in `bytes` as stored; of the `needed`
 * chunks of the view the box covers, those `computed` so far. Each count only grows. With the
 * level's extent the array's chunks are the view's, so `computed` leads `done` by the chunks
 * being written; with the box's, an array chunk is done once every chunk of the view it is
 * stitched from has been computed, so `done` can trail `computed` by a slab of chunks. */
export interface MaterializeProgress { done: number; total: number; written: number; bytes: number; computed: number; needed: number }

/** `Engine.materialize`'s options. */
export interface MaterializeOptions {
  /** The view's level to write (default 0, the finest). */
  level?: number;
  /** Voxel box [lo, hi) on that level, in the level's voxel axes (z, y, x...); default the
   * whole level. Clipped to the level, and rounded out to whole voxels (to whole chunks, with
   * the level's `extent`). */
  lo?: number[]; hi?: number[];
  /** What the written array covers. "box" (the default): exactly the box, not rounded to
   * chunks: its shape hi − lo, its translation the level's moved by lo voxels, cut in the
   * view's chunk shape counted from lo. Where the box does not start on the view's chunk
   * edges, each of its chunks is stitched from the view's chunks it overlaps (up to 2 along
   * each axis), each of those computed once; the chunks at its far edge overhang it and are
   * stored whole, the fill value past the box, as zarr stores a chunk at an array's edge.
   * "level": the level's whole shape, with the level's translation, so the box keeps its
   * place in it (several boxes written into one store, say); the chunks the box touches are
   * written, rounded out to whole chunks. A box of the whole level is the same store, byte
   * for byte, either way. */
  extent?: "level" | "box";
  /** "zarr2" (OME-Zarr 0.4: .zgroup, .zattrs, .zarray, "/" as the dimension separator) or
   * "zarr3" (OME-Zarr 0.5, zarr.json); default "zarr2", which more tools read. */
  format?: "zarr3" | "zarr2";
  /** gzip through CompressionStream (zarr v2: the compressor {"id": "gzip", "level": 6};
   * zarr v3: the "gzip" codec after "bytes"), or null for none; default "gzip". */
  compression?: "gzip" | null;
  /** A chunk of nothing but zero bytes (the fill value) is not written, zarr reading a
   * missing chunk as the fill value; default true. */
  skipEmpty?: boolean;
  /** Chunks asked for at once (default: as many as the engine's queue has slots, of which
   * all but one run materialize's chunks at most, so the next is waiting while one is
   * written). */
  concurrency?: number;
  /** Aborted: no chunk is started after it, chunks waiting for their turn are given up, the
   * writes under way end, and materialize resolves with `aborted`; chunks written stay. */
  signal?: AbortSignal;
  /** Told once the metadata is written (done 0, computed 0), then after each chunk of the
   * view computed, once the array's chunks it completes are written. */
  onProgress?: (p: MaterializeProgress) => void;
  /** Attributes beside the OME multiscales in the group's attributes (the consumer's
   * provenance); `ome` (zarr v3) or `multiscales` (zarr v2) among them is the view's. */
  attributes?: Record<string, unknown>;
}

/** What a materialize wrote: of the written array's `chunks` in the box, those `written`
 * (in `bytes`, as stored) and those `skipped` as empty; `aborted` if its signal stopped it
 * first. */
export interface Materialized { chunks: number; written: number; skipped: number; bytes: number; aborted: boolean }

/** The written level's path in the store: its one dataset, "0", as an OME-Zarr's finest
 * level is, whichever level of the view it is (the scale and translation say where it lies). */
export const LEVEL_PATH = "0";
/** The gzip level written in the metadata: CompressionStream's, zlib's default (a reader
 * needs none of it to decompress). */
export const GZIP_LEVEL = 6;

/** Voxels [lo, hi) of a level of `shape`: the box clipped to the level and rounded out to
 * whole voxels; the whole level where `lo` or `hi` is left out. A box with no voxel of the
 * level is refused. */
export function voxelBox(shape: number[], lo?: number[], hi?: number[]): { lo: number[]; hi: number[] } {
  for (const [what, box] of [["lo", lo], ["hi", hi]] as const) {
    if (box && (box.length !== shape.length || box.some((x) => typeof x !== "number" || Number.isNaN(x)))) {
      throw new Error(`${what} ${JSON.stringify(box)}: one number for each of the level's ${shape.length} axes`);
    }
  }
  const inLo = shape.map((n, a) => Math.min(Math.max(lo?.[a] ?? 0, 0), n));  // the box clipped to the level
  const inHi = shape.map((n, a) => Math.min(Math.max(hi?.[a] ?? n, 0), n));
  if (inHi.some((h, a) => h <= inLo[a])) {
    throw new Error(`the box from [${(lo ?? shape.map(() => 0)).join(", ")}] to [${(hi ?? shape).join(", ")}] holds no voxel of the level, which is ${shape.join("×")}`);
  }
  return { lo: inLo.map((x) => Math.floor(x)), hi: inHi.map((x) => Math.ceil(x)) };
}

/** The chunk grid's indices [lo, hi) covering voxels [lo, hi) of a level of `shape` cut in
 * chunks of `chunk`: the box rounded out to whole chunks and clipped to the level; the
 * whole level where `lo` or `hi` is left out. A box with no voxel of the level is refused. */
export function chunkRange(shape: number[], chunk: number[], lo?: number[], hi?: number[]): { lo: number[]; hi: number[] } {
  const box = voxelBox(shape, lo, hi);
  return { lo: box.lo.map((x, a) => Math.floor(x / chunk[a])), hi: box.hi.map((x, a) => Math.ceil(x / chunk[a])) };
}

/** The chunk indices [lo, hi), in C order (the last axis fastest), or in `order`, the axes
 * from the outermost to the innermost (the last of them fastest). */
export function* chunkIndices(lo: number[], hi: number[], order: number[] = lo.map((_, a) => a)): Generator<number[]> {
  if (hi.some((h, a) => h <= lo[a])) return;
  const at = [...lo];
  for (;;) {
    yield [...at];
    let k = order.length - 1;
    while (k >= 0 && ++at[order[k]] >= hi[order[k]]) { at[order[k]] = lo[order[k]]; k--; }
    if (k < 0) return;
  }
}

/** `bytes` gzipped, through CompressionStream (the browser's, and Node's). */
export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Whether every byte of `bytes` is zero: an empty chunk (zarr's fill value, 0, for every
 * dtype; a float's -0 is not empty, its sign bit set). */
export function allZero(bytes: Uint8Array): boolean {
  let i = 0;
  if (bytes.byteOffset % 4 === 0) {  // four bytes at a time
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
    for (const w of words) if (w) return false;
    i = words.length * 4;
  }
  for (; i < bytes.length; i++) if (bytes[i]) return false;
  return true;
}

/** What a materialize writes of view `name`: which `level`, in which `format` and
 * `compression`, over which `extent`; the metadata `files` (path and bytes, the group's
 * first); the view's chunks the box covers, the chunk grid's indices [lo, hi), `needed` of
 * them; and the array written: its voxel 0 the level's voxel `at` (0 for the level's extent,
 * the box's lo for the box's), `shape` voxels on the level's axes (the level's own,
 * `levelShape`, or the box's), cut in chunks of `chunk` (the view's), those of its grid's
 * indices [first, last) written, `total` of them, each chunk the channels the ops add first,
 * `channels` values for each voxel, of `item` bytes each; and each chunk's `key` in the
 * store. With the level's extent the array's grid is the view's, [first, last) = [lo, hi). */
export interface Region {
  name: string; level: number; format: "zarr3" | "zarr2"; compression: "gzip" | null; extent: "level" | "box";
  files: [string, Uint8Array][];
  lo: number[]; hi: number[]; needed: number;
  at: number[]; shape: number[]; levelShape: number[]; chunk: number[];
  first: number[]; last: number[]; total: number;
  channels: number; item: number;
  key: (index: number[]) => string;
}

const prod = (xs: number[]) => xs.reduce((n, x) => n * x, 1);
/** The count of a grid's indices [lo, hi). */
const count = (lo: number[], hi: number[]) => prod(lo.map((l, a) => hi[a] - l));

/** What `options` asks to write of view `name` (served as `v`, chunked `chunk`): the group
 * listing the one level written as its one dataset, at LEVEL_PATH, with the level's scale and
 * translation (moved to the box's first voxel, for the box's extent); the array, the level's
 * whole shape (so a box keeps its place in it) or the box's, the chunks and dtype served,
 * compressed as asked; and the chunks of the box. */
export function region(name: string, v: ViewInfo, chunk: number[], options: MaterializeOptions = {}): Region {
  const { level = 0, format = "zarr2", compression = "gzip", extent = "box", attributes = {} } = options;
  if (!Number.isInteger(level) || !v.levels[level]) throw new Error(`${name} has no level ${level}: it has ${v.levels.length}`);
  if (format !== "zarr2" && format !== "zarr3") throw new Error(`format ${JSON.stringify(format)}: "zarr2" or "zarr3"`);
  if (compression !== "gzip" && compression !== null) throw new Error(`compression ${JSON.stringify(compression)}: "gzip" or null`);
  if (extent !== "level" && extent !== "box") throw new Error(`extent ${JSON.stringify(extent)}: "level" or "box"`);
  const levelShape = v.levels[level].shape, box = voxelBox(levelShape, options.lo, options.hi);
  const { lo, hi } = chunkRange(levelShape, chunk, box.lo, box.hi);  // the view's chunks the box needs
  const whole = extent === "level", item = TYPED[v.out]?.BYTES_PER_ELEMENT;
  if (!whole && !item) throw new Error(`${name} is ${v.out}, which a box is not cut from`);
  const at = whole ? levelShape.map(() => 0) : box.lo, shape = whole ? levelShape : box.hi.map((h, a) => h - box.lo[a]);
  const first = whole ? lo : shape.map(() => 0), last = whole ? hi : shape.map((n, a) => Math.ceil(n / chunk[a]));
  const datasets = [{ path: LEVEL_PATH, level, ...(whole ? {} : { lo: box.lo }) }], added = v.added ?? [];
  const files: [string, unknown][] = format === "zarr3"
    ? [
      ["zarr.json", group3(name, v, datasets, attributes)],
      [`${LEVEL_PATH}/zarr.json`, array3(v, chunk, level, { codecs: compression ? [{ name: "gzip", configuration: { level: GZIP_LEVEL } }] : [], shape })],
    ]
    : [
      [".zgroup", { zarr_format: 2 }],
      [".zattrs", attrs2(name, v, datasets, attributes)],
      [`${LEVEL_PATH}/.zarray`, array2(v, chunk, level, compression ? { id: "gzip", level: GZIP_LEVEL } : null, shape)],
      [`${LEVEL_PATH}/.zattrs`, { _ARRAY_DIMENSIONS: dimensionNames(v) }],
    ];
  const json = (o: unknown) => new TextEncoder().encode(`${JSON.stringify(o, null, 2)}\n`);
  return {
    name, level, format, compression, extent, files: files.map(([path, o]) => [path, json(o)]),
    lo, hi, needed: count(lo, hi),
    at, shape, levelShape, chunk, first, last, total: count(first, last),
    channels: prod(added), item: item ?? 0,
    key: (index) => chunkKey(format, LEVEL_PATH, added.length, index),
  };
}

/** The axes along which region `r`'s array chunks are stitched from two of the view's
 * chunks: those along which the array's first chunk spans a chunk edge of the view's (the
 * box starts off the view's chunk edges there, and does not lie within one of its chunks). */
export function splitAxes(r: Region): number[] {
  return r.chunk.flatMap((c, a) => {
    const from = r.at[a] + r.first[a] * c, to = r.at[a] + Math.min((r.first[a] + 1) * c, r.shape[a]);
    return Math.floor(from / c) !== Math.floor((to - 1) / c) ? [a] : [];
  });
}

/** The order region `r`'s served chunks are walked in, its axes from the outermost to the
 * innermost: first the axes it does not split (`splitAxes`), in C order, then those it
 * splits, the one with the most of the array's chunks first (C order among equals). Of the
 * orders of the axes, it holds the fewest array chunks at once (`stitcher`): a slab across
 * the split axes but the outermost, so the longest split axis is the one walked slab by
 * slab. The level's extent splits none: C order, as the view's chunks are numbered. */
export function walkOrder(r: Region): number[] {
  const split = splitAxes(r), count = (a: number) => r.last[a] - r.first[a];
  const rest = r.chunk.map((_, a) => a).filter((a) => !split.includes(a));
  return [...rest, ...split.sort((a, b) => count(b) - count(a) || a - b)];
}

/** The bytes of a chunk of region `r`'s array (or of the view's: they are cut alike). */
export const chunkBytes = (r: Region) => r.channels * prod(r.chunk) * r.item;

/** The array's chunk that `add` completes: its index in the array's grid and its bytes, null
 * for nothing but the fill value. */
export type Completed = [index: number[], bytes: Uint8Array | null];

/** Region `r`'s array chunks made of the view's chunks, as these come, in any order: `add`
 * takes one of the view's chunks (its index in the view's grid and its whole bytes) and
 * gives the array's chunks it completes. An array chunk that is one of the view's chunks as
 * it is (the same voxels, and its end the level's end where it overhangs: every chunk with
 * the level's extent, and those of a box on the view's chunk edges) is that chunk's bytes,
 * not copied. Any other is stitched in a buffer of its own, made when the first of the view's
 * chunks it overlaps that holds a value not zero comes (none, if they are all zero), each
 * one's part copied in, the rest the fill value; it is complete, and forgotten here, once all
 * of them have come. The array's chunks begun and not complete are `open`: with the view's
 * chunks added in `walkOrder`, at most a slab of them across the split axes but the
 * outermost (`splitAxes`), a row along the innermost and one more. With the box's chunks
 * n1 ≥ n2 ≥ n3 along three split axes, n2 × n3 + n3 + 1; along two, n1 ≥ n2, n2 + 1; along
 * one, 1; none if it splits none (the box starts on the view's chunk edges). A few more
 * where they come out of order. */
export function stitcher(r: Region): { add: (index: number[], bytes: Uint8Array) => Completed[]; readonly open: number } {
  const C = r.chunk, n = C.length, size = chunkBytes(r);
  const open = new Map<string, { bytes: Uint8Array | null; left: number }>();
  /** The level's voxels [from, to) of the array's chunk `o` along axis `a`, within the array. */
  const outer = (o: number, a: number) => [r.at[a] + o * C[a], r.at[a] + Math.min((o + 1) * C[a], r.shape[a])];
  /** The view's chunks that the array's chunk `o` is made of. */
  const contributors = (o: number[]) => prod(o.map((x, a) => {
    const [from, to] = outer(x, a);
    return Math.floor((to - 1) / C[a]) - Math.floor(from / C[a]) + 1;
  }));
  return {
    get open() { return open.size; },
    add(s, bytes) {
      // the array's chunks the view's chunk s overlaps, along each axis, within those written
      const lo = s.map((x, a) => Math.max(r.first[a], Math.floor((x * C[a] - r.at[a]) / C[a])));
      const hi = s.map((x, a) => Math.min(r.last[a], Math.ceil(((x + 1) * C[a] - r.at[a]) / C[a])));
      const done: Completed[] = [];
      let empty: boolean | undefined;  // whether s is all zero, once asked
      for (const o of chunkIndices(lo, hi)) {
        // along each axis: the voxels of s that land in o, from s's voxel src, to o's voxel dst
        const src: number[] = [], dst: number[] = [], len: number[] = [];
        let same = true;
        for (let a = 0; a < n; a++) {
          const [from, to] = outer(o[a], a), start = Math.max(s[a] * C[a], from), end = Math.min((s[a] + 1) * C[a], to);
          src.push(start - s[a] * C[a]); dst.push(start - from); len.push(end - start);
          // the same voxels, and where o overhangs the array, the array ends where the level does
          same &&= from === s[a] * C[a] && to === Math.min((s[a] + 1) * C[a], r.levelShape[a]);
        }
        if (len.some((l) => l <= 0)) continue;  // none of s in o's part of the array
        if (same) { done.push([o, bytes]); continue; }
        if (bytes.byteLength !== size) throw new Error(`${bytes.byteLength} bytes, not a whole chunk's ${size}`);
        const key = o.join(".");
        let part = open.get(key);
        if (!part) open.set(key, part = { bytes: null, left: contributors(o) });
        if (!(empty ??= allZero(bytes))) place(part.bytes ??= new Uint8Array(size), bytes, C, r.channels, r.item, src, dst, len);
        if (--part.left === 0) { open.delete(key); done.push([o, part.bytes]); }
      }
      return done;
    },
  };
}

/** The voxels `len` long from `src` of chunk bytes `from` copied to `dst` of `to`, both cut in
 * chunks of `chunk` with `channels` values of `item` bytes for each voxel, the channels first
 * (C order): each run along the last axis copied whole. */
function place(to: Uint8Array, from: Uint8Array, chunk: number[], channels: number, item: number, src: number[], dst: number[], len: number[]) {
  const n = chunk.length, stride = Array<number>(n);
  stride[n - 1] = item;
  for (let a = n - 2; a >= 0; a--) stride[a] = stride[a + 1] * chunk[a + 1];
  const plane = stride[0] * chunk[0], run = len[n - 1] * item, at = Array<number>(n - 1).fill(0);
  for (let c = 0; c < channels; c++) {
    for (;;) {
      let s = c * plane + src[n - 1] * item, d = c * plane + dst[n - 1] * item;
      for (let a = 0; a < n - 1; a++) { s += (src[a] + at[a]) * stride[a]; d += (dst[a] + at[a]) * stride[a]; }
      to.set(from.subarray(s, s + run), d);
      let a = n - 2;
      while (a >= 0 && ++at[a] >= len[a]) { at[a] = 0; a--; }
      if (a < 0) break;
    }
  }
}

/** Chunk `index` of the level as whole bytes, as served (zarr v3's: little endian, C order,
 * the channels first); `signal` aborts when it is no longer wanted. Asked for once for each
 * of the view's chunks a region needs. */
export type Produce = (index: number[], signal: AbortSignal) => Promise<ArrayBuffer>;

const why = (e: unknown) => String((e as Error)?.message ?? e);

/** Write region `r` into `store`: its metadata, then its chunks. The view's chunks the box
 * covers are asked of `produce` once each, in `walkOrder` (C order where the box splits no
 * chunk of the view's, as with the level's extent), `concurrency` at once (one by default);
 * each, as it comes, completes the array's chunks it
 * is the last of (`stitcher`: with the level's extent, or a box on the view's chunk edges,
 * just itself), and those are written, gzipped if `r` says so, those of nothing but zeros
 * skipped (`skipEmpty`, the default). Only the array's chunks begun and not yet complete are
 * held meanwhile (about a slab of them at most, `stitcher`), with the view's chunks on their
 * way.
 * Resolves with what was written, or on `signal`'s abort with `aborted` (no chunk asked for
 * or written after it, the ones asked for given up through `produce`'s signal and not waited
 * for, the writes under way ended). A chunk that fails to be made or written rejects it,
 * naming the chunk and why, once the writes under way have ended; the rest are given up as on
 * an abort. */
export async function writeChunks(store: WritableStore, r: Region, produce: Produce,
  options: Pick<MaterializeOptions, "skipEmpty" | "concurrency" | "signal" | "onProgress"> = {}): Promise<Materialized> {
  const { skipEmpty = true, concurrency = 1, signal, onProgress } = options;
  if (!Number.isInteger(concurrency) || concurrency < 1) throw new Error(`concurrency ${concurrency}: a whole number of chunks, 1 or more`);
  const got: Materialized = { chunks: r.total, written: 0, skipped: 0, bytes: 0, aborted: false };
  if (signal?.aborted) return { ...got, aborted: true };
  for (const [path, bytes] of r.files) {
    try { await store.put(path, bytes); } catch (e) { throw new Error(`writing ${path} of ${r.name}: ${why(e)}`); }
  }
  // stopped by the caller's abort, or by a chunk that failed
  const stop = new AbortController(), onAbort = () => stop.abort(signal?.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  const stopped = new Promise<never>((_, reject) => stop.signal.addEventListener("abort", () => reject(stop.signal.reason), { once: true }));
  stopped.catch(() => {});
  let failure: Error | null = null;
  const fail = (e: Error) => { failure ??= e; stop.abort(e); };
  let computed = 0;
  const report = () => {
    try { onProgress?.({ done: got.written + got.skipped, total: r.total, written: got.written, bytes: got.bytes, computed, needed: r.needed }); }
    catch (e) { console.error(`materialize ${r.name}: onProgress threw: ${why(e)}`); }  // the page's own trouble: the writing goes on
  };
  report();
  // a chunk of the view named as the array's chunk it is, with the level's extent; as the engine names it, with the box's
  const named = (index: number[]) => (r.extent === "level" ? `chunk ${r.key(index)} of ${r.name}` : `chunk ${r.name}/${r.level}/${index.join(".")}`);
  const stitched = stitcher(r), indices = chunkIndices(r.lo, r.hi, walkOrder(r));
  const lane = async () => {
    for (let next = indices.next(); !next.done && !stop.signal.aborted; next = indices.next()) {
      const index = next.value;
      let body: ArrayBuffer, complete: Completed[];
      try {
        body = await Promise.race([produce(index, stop.signal), stopped]);  // given up at once when stopped
      } catch (e) {
        if (!stop.signal.aborted) fail(new Error(`${named(index)}: ${why(e)}`));
        return;
      }
      if (stop.signal.aborted) return;  // made as it stopped: not written
      computed++;
      try { complete = stitched.add(index, new Uint8Array(body)); } catch (e) { fail(new Error(`${named(index)}: ${why(e)}`)); return; }
      for (const [o, bytes] of complete) {
        if (stop.signal.aborted) return;  // no write starts once stopped
        const key = r.key(o);
        if (skipEmpty && (!bytes || allZero(bytes))) { got.skipped++; continue; }
        try {
          const raw = bytes ?? new Uint8Array(chunkBytes(r)), stored = r.compression === "gzip" ? await gzip(raw) : raw;
          await store.put(key, stored);
          got.written++; got.bytes += stored.byteLength;
        } catch (e) {
          fail(new Error(`writing chunk ${key} of ${r.name}: ${why(e)}`));
          return;
        }
      }
      report();
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(concurrency, r.needed) }, lane));
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
  if (signal?.aborted) return { ...got, aborted: true };
  if (failure) throw failure;
  return got;
}
