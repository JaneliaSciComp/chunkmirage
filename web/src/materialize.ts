// A region of a view the engine serves, written into a store as an OME-Zarr of its own
// (Engine.materialize): a box of the view computed once and kept, to open again later or to
// copy where other tools read it. The store is a directory (one the visitor picks with the
// File System Access API's showDirectoryPicker, in Chrome and Edge, or the origin's private
// file system) or memory. What is written is what the engine serves for the view
// (layout.ts), one level of it, its chunks compressed. Here are the parts that need no
// engine, so Node's tests run them: the chunks a box covers, the files written, gzip, and
// the loop that has the chunks made and writes them, from any producer (the engine's queue
// on the page, a fake one in the tests).
import { array2, array3, attrs2, chunkKey, dimensionNames, group3 } from "./layout.ts";  // with its extension: Node's runner resolves no other
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

/** How a materialize is going: chunks done (written or skipped as empty) of `total`, and of
 * them `written`, in `bytes` as stored. */
export interface MaterializeProgress { done: number; total: number; written: number; bytes: number }

/** `Engine.materialize`'s options. */
export interface MaterializeOptions {
  /** The view's level to write (default 0, the finest). */
  level?: number;
  /** Voxel box [lo, hi) on that level, in the level's voxel axes (z, y, x...); default the
   * whole level. Rounded out to whole chunks, and clipped to the level. */
  lo?: number[]; hi?: number[];
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
  /** Told once the metadata is written (done 0), then after each chunk. */
  onProgress?: (p: MaterializeProgress) => void;
  /** Attributes beside the OME multiscales in the group's attributes (the consumer's
   * provenance); `ome` (zarr v3) or `multiscales` (zarr v2) among them is the view's. */
  attributes?: Record<string, unknown>;
}

/** What a materialize wrote: of the box's `chunks`, those `written` (in `bytes`, as
 * stored) and those `skipped` as empty; `aborted` if its signal stopped it first. */
export interface Materialized { chunks: number; written: number; skipped: number; bytes: number; aborted: boolean }

/** The written level's path in the store: its one dataset, "0", as an OME-Zarr's finest
 * level is, whichever level of the view it is (the scale and translation say where it lies). */
export const LEVEL_PATH = "0";
/** The gzip level written in the metadata: CompressionStream's, zlib's default (a reader
 * needs none of it to decompress). */
export const GZIP_LEVEL = 6;

/** The chunk grid's indices [lo, hi) covering voxels [lo, hi) of a level of `shape` cut in
 * chunks of `chunk`: the box rounded out to whole chunks and clipped to the level; the
 * whole level where `lo` or `hi` is left out. A box with no voxel of the level is refused. */
export function chunkRange(shape: number[], chunk: number[], lo?: number[], hi?: number[]): { lo: number[]; hi: number[] } {
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
  return { lo: inLo.map((x, a) => Math.floor(x / chunk[a])), hi: inHi.map((x, a) => Math.ceil(x / chunk[a])) };
}

/** The chunk indices [lo, hi), in C order (the last axis fastest). */
export function* chunkIndices(lo: number[], hi: number[]): Generator<number[]> {
  if (hi.some((h, a) => h <= lo[a])) return;
  const at = [...lo];
  for (;;) {
    yield [...at];
    let a = at.length - 1;
    while (a >= 0 && ++at[a] >= hi[a]) { at[a] = lo[a]; a--; }
    if (a < 0) return;
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
 * `compression`; the metadata `files` (path and bytes, the group's first), the chunk grid's
 * indices [lo, hi) the box covers, `total` chunks, and each chunk's `key` in the store. */
export interface Region {
  name: string; level: number; format: "zarr3" | "zarr2"; compression: "gzip" | null;
  files: [string, Uint8Array][];
  lo: number[]; hi: number[]; total: number;
  key: (index: number[]) => string;
}

/** What `options` asks to write of view `name` (served as `v`, chunked `chunk`): the group
 * listing the one level written as its one dataset, at LEVEL_PATH, with the level's scale and
 * translation; the level's array with its whole shape (so the box keeps its place in it),
 * the chunks and dtype served, compressed as asked; and the chunks of the box. */
export function region(name: string, v: ViewInfo, chunk: number[], options: MaterializeOptions = {}): Region {
  const { level = 0, format = "zarr2", compression = "gzip", attributes = {} } = options;
  if (!Number.isInteger(level) || !v.levels[level]) throw new Error(`${name} has no level ${level}: it has ${v.levels.length}`);
  if (format !== "zarr2" && format !== "zarr3") throw new Error(`format ${JSON.stringify(format)}: "zarr2" or "zarr3"`);
  if (compression !== "gzip" && compression !== null) throw new Error(`compression ${JSON.stringify(compression)}: "gzip" or null`);
  const { lo, hi } = chunkRange(v.levels[level].shape, chunk, options.lo, options.hi);
  const datasets = [{ path: LEVEL_PATH, level }], added = (v.added ?? []).length;
  const files: [string, unknown][] = format === "zarr3"
    ? [
      ["zarr.json", group3(name, v, datasets, attributes)],
      [`${LEVEL_PATH}/zarr.json`, array3(v, chunk, level, { codecs: compression ? [{ name: "gzip", configuration: { level: GZIP_LEVEL } }] : [] })],
    ]
    : [
      [".zgroup", { zarr_format: 2 }],
      [".zattrs", attrs2(name, v, datasets, attributes)],
      [`${LEVEL_PATH}/.zarray`, array2(v, chunk, level, compression ? { id: "gzip", level: GZIP_LEVEL } : null)],
      [`${LEVEL_PATH}/.zattrs`, { _ARRAY_DIMENSIONS: dimensionNames(v) }],
    ];
  const json = (o: unknown) => new TextEncoder().encode(`${JSON.stringify(o, null, 2)}\n`);
  return {
    name, level, format, compression, files: files.map(([path, o]) => [path, json(o)]),
    lo, hi, total: lo.reduce((n, l, a) => n * (hi[a] - l), 1),
    key: (index) => chunkKey(format, LEVEL_PATH, added, index),
  };
}

/** Chunk `index` of the level as whole bytes, as served (zarr v3's: little endian, C order,
 * the channels first); `signal` aborts when it is no longer wanted. */
export type Produce = (index: number[], signal: AbortSignal) => Promise<ArrayBuffer>;

const why = (e: unknown) => String((e as Error)?.message ?? e);

/** Write region `r` into `store`: its metadata, then its chunks, `concurrency` of them asked
 * of `produce` at once (one by default), each as it comes, gzipped if `r` says so, those of
 * nothing but zeros skipped (`skipEmpty`, the default). Resolves with what was written, or
 * on `signal`'s abort with `aborted` (no chunk started after it, the ones asked for given up
 * through `produce`'s signal and not waited for, the writes under way ended). A chunk that
 * fails to be made or written rejects it, naming the chunk and why, once the writes under
 * way have ended; the rest are given up as on an abort. */
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
  const report = () => {
    try { onProgress?.({ done: got.written + got.skipped, total: r.total, written: got.written, bytes: got.bytes }); }
    catch (e) { console.error(`materialize ${r.name}: onProgress threw: ${why(e)}`); }  // the page's own trouble: the writing goes on
  };
  report();
  const indices = chunkIndices(r.lo, r.hi);
  const lane = async () => {
    for (let next = indices.next(); !next.done && !stop.signal.aborted; next = indices.next()) {
      const index = next.value, key = r.key(index);
      let body: ArrayBuffer;
      try {
        body = await Promise.race([produce(index, stop.signal), stopped]);  // given up at once when stopped
      } catch (e) {
        if (!stop.signal.aborted) fail(new Error(`chunk ${key} of ${r.name}: ${why(e)}`));
        return;
      }
      if (stop.signal.aborted) return;  // made as it stopped: not written
      const bytes = new Uint8Array(body);
      if (skipEmpty && allZero(bytes)) { got.skipped++; report(); continue; }
      try {
        const stored = r.compression === "gzip" ? await gzip(bytes) : bytes;
        await store.put(key, stored);
        got.written++; got.bytes += stored.byteLength;
      } catch (e) {
        fail(new Error(`writing chunk ${key} of ${r.name}: ${why(e)}`));
        return;
      }
      report();
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(concurrency, r.total) }, lane));
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
  if (signal?.aborted) return { ...got, aborted: true };
  if (failure) throw failure;
  return got;
}
