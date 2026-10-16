// A region of a view written into a store (src/materialize.ts, Engine.materialize's parts
// that need no engine): the chunks a box covers, rounded out and clipped; the metadata both
// formats write, the layout the engine serves (src/layout.ts); gzip; the loop over a fake
// producer, with empty chunks skipped, progress, abort and failure; the stores; and the
// result read back by zarrita, a zarr implementation of its own, as a reader would. Then the
// same for a box written as exactly the box (extent "box"): its metadata, its chunks
// stitched from the view's (each of those made once), its edge chunks, and the slab of
// chunks held at most. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as zarr from "zarrita";
import { array3, chunkKey, group3 } from "../src/layout.ts";
import { allZero, chunkIndices, chunkRange, directoryStore, gzip, memoryStore, region, stitcher, voxelBox, writeChunks, type MaterializeOptions, type MaterializeProgress, type Produce, type WritableStore } from "../src/materialize.ts";
import type { ViewInfo } from "../src/types.ts";

const nm = (name: string) => ({ name, unit: "nm" });
/** A view of two levels, served as uint16 in chunks of 2 × 4 × 4. */
const view: ViewInfo = {
  dtype: "uint8", channels: 1, axes: [nm("z"), nm("y"), nm("x")], halo: [0, 0, 0], lead: 0, out: "uint16", added: [],
  levels: [{ shape: [5, 6, 7], voxel: [8, 8, 8], origin: [0, 0, 0] }, { shape: [3, 3, 4], voxel: [16, 16, 16], origin: [4, 4, 4] }],
};
const chunk = [2, 4, 4];
/** The same view with two channels its ops add (a gradient's, say), as float32. */
const channels: ViewInfo = { ...view, out: "float32", added: [2] };

const json = (b: Uint8Array | undefined) => JSON.parse(new TextDecoder().decode(b));
const files = (s: { files: Map<string, Uint8Array> }) => [...s.files.keys()].sort();
async function gunzip(b: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([b as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Level 0's voxel (z, y, x) as the fake producer makes it: never zero inside the level. */
const value = (z: number, y: number, x: number) => 1 + 100 * z + 10 * y + x;
/** Chunk `index` of level 0 as the engine gives it: whole, zero past the level's edge; the
 * chunks of `empty` all zero, as a model's output where there is nothing. */
function fakeChunk(index: number[], empty: string[] = []): ArrayBuffer {
  const out = new Uint16Array(chunk[0] * chunk[1] * chunk[2]), [Z, Y, X] = view.levels[0].shape;
  if (empty.includes(index.join("."))) return out.buffer;
  for (let z = 0; z < chunk[0]; z++) for (let y = 0; y < chunk[1]; y++) for (let x = 0; x < chunk[2]; x++) {
    const g = [index[0] * chunk[0] + z, index[1] * chunk[1] + y, index[2] * chunk[2] + x];
    if (g[0] < Z && g[1] < Y && g[2] < X) out[(z * chunk[1] + y) * chunk[2] + x] = value(g[0], g[1], g[2]);
  }
  return out.buffer;
}
const producer = (empty: string[] = []): Produce => async (index) => fakeChunk(index, empty);

/** `store`'s files as zarrita reads a store (keys with a leading "/"). */
const readable = (s: { files: Map<string, Uint8Array> }) => ({ async get(key: string) { return s.files.get(key.replace(/^\//, "")); } });

test("a box covers the chunks it touches, rounded out and clipped to the level; the whole level by default", () => {
  const shape = view.levels[0].shape;  // 5 × 6 × 7 in chunks of 2 × 4 × 4: a grid of 3 × 2 × 2
  assert.deepEqual(chunkRange(shape, chunk), { lo: [0, 0, 0], hi: [3, 2, 2] });
  assert.deepEqual(chunkRange(shape, chunk, [1, 3, 5], [3, 5, 6]), { lo: [0, 0, 1], hi: [2, 2, 2] });
  assert.deepEqual(chunkRange(shape, chunk, [2, 4, 4], [4, 8, 8]), { lo: [1, 1, 1], hi: [2, 2, 2] });  // on chunk edges: no more
  assert.deepEqual(chunkRange(shape, chunk, [-10, 2, 3], [100, 6, 100]), { lo: [0, 0, 0], hi: [3, 2, 2] });  // past the edges: clipped
  assert.deepEqual(chunkRange(shape, chunk, [4, 0, 0]), { lo: [2, 0, 0], hi: [3, 2, 2] });  // lo alone: to the level's end
  assert.deepEqual(chunkRange(shape, chunk, [0.5, 0, 0], [1.5, 1, 1]), { lo: [0, 0, 0], hi: [1, 1, 1] });  // fractions: rounded out too
  assert.throws(() => chunkRange(shape, chunk, [5, 0, 0]), /holds no voxel of the level, which is 5×6×7/);  // starts at the end
  assert.throws(() => chunkRange(shape, chunk, [0, 0, 0], [0, 6, 7]), /holds no voxel/);
  assert.throws(() => chunkRange(shape, chunk, [0, 0], [1, 1]), /lo \[0,0\]: one number for each of the level's 3 axes/);
  assert.throws(() => chunkRange(shape, chunk, undefined, [1, Number.NaN, 1]), /hi /);
});

test("the chunks of a range in C order, the last axis fastest", () => {
  assert.deepEqual([...chunkIndices([0, 1], [2, 3])], [[0, 1], [0, 2], [1, 1], [1, 2]]);
  assert.deepEqual([...chunkIndices([1, 1, 1], [2, 2, 2])], [[1, 1, 1]]);
  assert.deepEqual([...chunkIndices([0, 0], [0, 3])], []);
});

test("zarr v2: .zgroup, .zattrs with OME 0.4 multiscales listing the one level at 0, its .zarray gzipped and .zattrs", () => {
  const r = region("mito", view, chunk, { level: 1, attributes: { provenance: { model: "mito-v3" } } });
  assert.deepEqual(r.files.map(([p]) => p), [".zgroup", ".zattrs", "0/.zarray", "0/.zattrs"]);
  const [group, attrs, zarray, dims] = r.files.map(([, b]) => json(b));
  assert.deepEqual(group, { zarr_format: 2 });
  assert.deepEqual(attrs, {
    provenance: { model: "mito-v3" },
    multiscales: [{
      name: "mito",
      axes: [{ name: "z", type: "space", unit: "nanometer" }, { name: "y", type: "space", unit: "nanometer" }, { name: "x", type: "space", unit: "nanometer" }],
      // level 1's own scale and translation, so it lands where the view's level 1 does
      datasets: [{ path: "0", coordinateTransformations: [{ type: "scale", scale: [16, 16, 16] }, { type: "translation", translation: [4, 4, 4] }] }],
      version: "0.4",
    }],
  });
  assert.deepEqual(zarray, {
    zarr_format: 2, shape: [3, 3, 4], chunks: [2, 4, 4], dtype: "<u2", compressor: { id: "gzip", level: 6 },
    fill_value: 0, order: "C", filters: null, dimension_separator: "/",
  });
  assert.deepEqual(dims, { _ARRAY_DIMENSIONS: ["z", "y", "x"] });
  assert.deepEqual([r.lo, r.hi, r.total, r.key([1, 0, 0])], [[0, 0, 0], [2, 1, 1], 2, "0/1/0/0"]);
});

test("zarr v3: zarr.json with OME 0.5 multiscales, the level's zarr.json with the gzip codec after bytes", () => {
  const r = region("mito", view, chunk, { format: "zarr3", lo: [2, 0, 4], hi: [5, 3, 7], attributes: { provenance: 1 } });
  assert.deepEqual(r.files.map(([p]) => p), ["zarr.json", "0/zarr.json"]);
  const [group, array] = r.files.map(([, b]) => json(b));
  assert.deepEqual(group, {
    zarr_format: 3, node_type: "group", attributes: {
      provenance: 1,
      ome: { version: "0.5", multiscales: [{
        name: "mito",
        axes: [{ name: "z", type: "space", unit: "nanometer" }, { name: "y", type: "space", unit: "nanometer" }, { name: "x", type: "space", unit: "nanometer" }],
        datasets: [{ path: "0", coordinateTransformations: [{ type: "scale", scale: [8, 8, 8] }, { type: "translation", translation: [0, 0, 0] }] }],
      }] },
    },
  });
  assert.deepEqual(array, {
    zarr_format: 3, node_type: "array", shape: [5, 6, 7], data_type: "uint16", fill_value: 0,  // the whole level: the box keeps its place
    chunk_grid: { name: "regular", configuration: { chunk_shape: [2, 4, 4] } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }, { name: "gzip", configuration: { level: 6 } }],
    dimension_names: ["z", "y", "x"], attributes: {},
  });
  assert.deepEqual([r.lo, r.hi, r.total, r.key([2, 0, 1])], [[1, 0, 1], [3, 1, 2], 2, "0/c/2/0/1"]);
  // the metadata is the engine's as served, but for the one dataset and the codec
  assert.deepEqual(array.codecs.slice(0, 1), array3(view, chunk, 0).codecs);
  assert.deepEqual(group3("mito", view, [{ path: "0", level: 0 }, { path: "1", level: 1 }]).attributes.ome.multiscales[0].datasets[1].path, "1");
});

test("uncompressed when asked; a channel axis the ops add is one chunk, index 0, as the engine serves it", () => {
  const two = region("grad", channels, chunk, { compression: null });
  const zarray = json(two.files.find(([p]) => p === "0/.zarray")![1]);
  assert.deepEqual([zarray.shape, zarray.chunks, zarray.dtype, zarray.compressor], [[2, 5, 6, 7], [2, 2, 4, 4], "<f4", null]);
  assert.deepEqual(json(two.files.find(([p]) => p === "0/.zattrs")![1]), { _ARRAY_DIMENSIONS: ["c", "z", "y", "x"] });
  const ms = json(two.files.find(([p]) => p === ".zattrs")![1]).multiscales[0];
  assert.deepEqual(ms.axes[0], { name: "c", type: "channel" });
  assert.deepEqual(ms.datasets[0].coordinateTransformations, [{ type: "scale", scale: [1, 8, 8, 8] }, { type: "translation", translation: [0, 0, 0, 0] }]);
  assert.equal(two.key([1, 0, 1]), "0/0/1/0/1");
  const three = region("grad", channels, chunk, { format: "zarr3", compression: null });
  const array = json(three.files[1][1]);
  assert.deepEqual([array.shape, array.chunk_grid.configuration.chunk_shape, array.codecs.length], [[2, 5, 6, 7], [2, 2, 4, 4], 1]);
  assert.equal(three.key([1, 0, 1]), "0/c/0/1/0/1");
  assert.equal(chunkKey("zarr3", "0", 0, [3, 2, 1]), "0/c/3/2/1");
  assert.throws(() => region("grad", channels, chunk, { level: 2 }), /grad has no level 2: it has 2/);
  assert.throws(() => region("grad", channels, chunk, { format: "n5" as "zarr2" }), /format "n5"/);
});

test("the group's attributes keep the view's OME metadata whatever the extra ones are called", () => {
  const two = region("v", view, chunk, { attributes: { multiscales: "mine", note: "kept" } });
  const attrs = json(two.files[1][1]);
  assert.equal(attrs.note, "kept");
  assert.equal(attrs.multiscales[0].name, "v");
  const three = region("v", view, chunk, { format: "zarr3", attributes: { ome: "mine" } });
  assert.equal(json(three.files[0][1]).attributes.ome.version, "0.5");
});

test("gzip round trips through CompressionStream and DecompressionStream; zeros are told from the rest", async () => {
  const chunk0 = new Uint8Array(fakeChunk([0, 0, 0]));
  assert.deepEqual(await gunzip(await gzip(chunk0)), chunk0);
  const bytes = new Uint8Array(4096).fill(7, 100, 3000), packed = await gzip(bytes);  // a model's output: much of it alike
  assert.deepEqual([packed[0], packed[1]], [0x1f, 0x8b]);  // gzip's magic
  assert.ok(packed.length < bytes.length / 10, `${packed.length} bytes`);
  assert.deepEqual(await gunzip(packed), bytes);
  assert.equal(allZero(new Uint8Array(64)), true);
  assert.equal(allZero(new Uint8Array(new ArrayBuffer(67), 1, 65)), true);  // not on a word: byte by byte
  const one = new Uint8Array(67);
  one[66] = 1;
  assert.equal(allZero(one), false);  // past the last whole word
  assert.equal(allZero(new Uint8Array(new Float32Array([0, -0]).buffer)), false);  // -0 has its sign bit
});

for (const format of ["zarr2", "zarr3"] as const) {
  test(`${format}: the box's chunks written gzipped, the empty one skipped, and read back by zarrita as the view`, async () => {
    const store = memoryStore(), seen: MaterializeProgress[] = [];
    // z voxels 1 to 3 of the whole y, x: chunks z 0 and 1, the second of them empty
    const got = await writeChunks(store, region("v", view, chunk, { format, lo: [1, 0, 0], hi: [3, 6, 7] }), producer(["1.0.1"]),
      { concurrency: 3, onProgress: (p) => seen.push(p) });
    const keys = files(store).filter((f) => !f.endsWith("json") && !f.includes(".z"));
    const at = (i: string) => (format === "zarr3" ? `0/c/${i}` : `0/${i}`);
    assert.deepEqual(keys, ["0.0.0", "0.0.1", "0.1.0", "0.1.1", "1.0.0", "1.1.0", "1.1.1"].map((i) => at(i.replaceAll(".", "/"))).sort());
    const bytes = keys.reduce((n, k) => n + store.files.get(k)!.length, 0);
    assert.deepEqual(got, { chunks: 8, written: 7, skipped: 1, bytes, aborted: false });
    assert.deepEqual(await gunzip(store.files.get(at("1/1/1"))!), new Uint8Array(fakeChunk([1, 1, 1])));
    // progress: once with the metadata written, then after each chunk, done = written + skipped
    assert.deepEqual(seen[0], { done: 0, total: 8, written: 0, bytes: 0, computed: 0, needed: 8 });
    assert.deepEqual(seen.at(-1), { done: 8, total: 8, written: 7, bytes, computed: 8, needed: 8 });
    const steps = seen.slice(1).map((p, i) => [p.done - seen[i].done, p.written - seen[i].written, p.bytes - seen[i].bytes]);
    assert.deepEqual(steps.map(([d]) => d), [1, 1, 1, 1, 1, 1, 1, 1]);
    assert.deepEqual(steps.filter(([, w, b]) => w === 0 && b === 0).length, 1);  // the one skipped
    // a reader: the box's chunks as made, the skipped one and those outside the box zero
    const root = zarr.root(readable(store));
    const group = format === "zarr3" ? await zarr.open.v3(root, { kind: "group" }) : await zarr.open.v2(root, { kind: "group" });
    assert.ok(group.attrs.ome ?? group.attrs.multiscales);
    const arr = format === "zarr3" ? await zarr.open.v3(root.resolve("0"), { kind: "array" }) : await zarr.open.v2(root.resolve("0"), { kind: "array" });
    assert.deepEqual([arr.shape, arr.dtype], [[5, 6, 7], "uint16"]);
    const data = (await zarr.get(arr)).data as Uint16Array;
    const [, Y, X] = arr.shape;
    for (let z = 0; z < 5; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
      const written = z < 4 && !(z >= 2 && y < 4 && x >= 4);  // chunks z 0 and 1, but for chunk (1, 0, 1)
      assert.equal(data[(z * Y + y) * X + x], written ? value(z, y, x) : 0, `voxel ${z}, ${y}, ${x}`);
    }
  });
}

test("empty chunks are written too when asked, and uncompressed ones as they are", async () => {
  const store = memoryStore();
  const got = await writeChunks(store, region("v", view, chunk, { compression: null, lo: [2, 0, 4], hi: [4, 4, 8] }), producer(["1.0.1"]), { skipEmpty: false });
  assert.deepEqual(got, { chunks: 1, written: 1, skipped: 0, bytes: 64, aborted: false });
  assert.deepEqual(store.files.get("0/1/0/1"), new Uint8Array(32 * 2));
});

test("no more than `concurrency` chunks are asked for at once", async () => {
  let asked = 0, most = 0;
  const produce: Produce = async (index) => {
    most = Math.max(most, ++asked);
    await tick();
    asked--;
    return fakeChunk(index);
  };
  const got = await writeChunks(memoryStore(), region("v", view, chunk), produce, { concurrency: 2 });
  assert.deepEqual([got.written, most], [12, 2]);
  await assert.rejects(writeChunks(memoryStore(), region("v", view, chunk), produce, { concurrency: 0 }), /concurrency 0/);
});

test("an abort stops starting chunks, gives up those asked for and resolves at once; the chunks written stay", { timeout: 5000 }, async () => {
  const store = memoryStore(), stop = new AbortController(), asked: number[][] = [], given: AbortSignal[] = [];
  let atAbort = -1;
  // the first two chunks come at once, the rest never (a model still running, say)
  const produce: Produce = (index, signal) => {
    asked.push(index);
    given.push(signal);
    return asked.length <= 2 ? Promise.resolve(fakeChunk(index)) : new Promise<ArrayBuffer>(() => {});
  };
  const got = await writeChunks(store, region("v", view, chunk), produce, {
    concurrency: 2, signal: stop.signal, onProgress: (p) => { if (p.written === 2) { stop.abort(); atAbort = asked.length; } },
  });
  assert.deepEqual(got, { chunks: 12, written: 2, skipped: 0, bytes: got.bytes, aborted: true });
  assert.equal(asked.length, atAbort);  // none asked for after the abort
  assert.ok(atAbort > 2);               // and the one asked for when it came, never made, was not waited for
  assert.ok(given.every((s) => s.aborted));
  assert.deepEqual(files(store), [".zattrs", ".zgroup", "0/.zarray", "0/.zattrs", "0/0/0/0", "0/0/0/1"]);
  // aborted before it starts: nothing is written
  const none = memoryStore();
  assert.deepEqual(await writeChunks(none, region("v", view, chunk), producer(), { signal: AbortSignal.abort() }),
    { chunks: 12, written: 0, skipped: 0, bytes: 0, aborted: true });
  assert.equal(none.files.size, 0);
});

test("a chunk that fails rejects naming it and why, once the writes under way have ended", async () => {
  let slowWritten = false;
  const memory = memoryStore();
  const store: WritableStore = {
    async put(path, bytes) {
      if (path === "0/0/0/0") { await new Promise((r) => setTimeout(r, 30)); slowWritten = true; }
      return memory.put(path, bytes);
    },
  };
  const produce: Produce = async (index) => {
    if (index.join(".") === "0.0.1") { await tick(); throw new Error("the model ran out of GPU memory"); }
    return fakeChunk(index);
  };
  await assert.rejects(writeChunks(store, region("v", view, chunk), produce, { concurrency: 2 }), (e: Error) => {
    assert.equal(e.message, "chunk 0/0/0/1 of v: the model ran out of GPU memory");
    assert.ok(slowWritten, "the chunk being written when the other failed was written first");
    return true;
  });
  assert.ok(memory.files.has("0/0/0/0") && !memory.files.has("0/0/1/0"));  // nothing started after the failure
  const full: WritableStore = { async put(path) { if (path.startsWith("0/c/1/")) throw new Error("the disk is full"); } };
  await assert.rejects(writeChunks(full, region("v", view, chunk, { format: "zarr3" }), producer()), /^Error: writing chunk 0\/c\/1\/0\/0 of v: the disk is full$/);
  const refused: WritableStore = { async put() { throw new Error("permission revoked"); } };
  await assert.rejects(writeChunks(refused, region("v", view, chunk), producer()), /^Error: writing \.zgroup of v: permission revoked$/);
});

test("the memory store keeps a copy of each file under its path, without empty names", async () => {
  const store = memoryStore(), bytes = new Uint8Array([1, 2, 3]);
  await store.put("/a//b/c", bytes);
  bytes[0] = 9;
  assert.deepEqual([...store.files], [["a/b/c", new Uint8Array([1, 2, 3])]]);
  await assert.rejects(store.put("a/../b", bytes), /with no "\." or "\.\."/);
  await assert.rejects(store.put("/", bytes), /not a file's path/);
});

/** A FileSystemDirectoryHandle's few methods directoryStore calls, over a tree of Maps. */
interface Dir { dirs: Map<string, Dir>; files: Map<string, Uint8Array> }
function fakeDirectory(calls: string[], failWrite?: string) {
  const make = (path: string, d: Dir): FileSystemDirectoryHandle => ({
    async getDirectoryHandle(name: string, { create = false } = {}) {
      calls.push(`dir ${path}${name}`);
      await tick();  // as a real handle is: concurrent puts must not make one directory twice
      if (!d.dirs.has(name)) { assert.ok(create); d.dirs.set(name, { dirs: new Map(), files: new Map() }); }
      return make(`${path}${name}/`, d.dirs.get(name)!);
    },
    async getFileHandle(name: string) {
      return {
        async createWritable() {
          const parts: Uint8Array[] = [];
          return {
            async write(b: Uint8Array) { if (name === failWrite) throw new Error("quota exceeded"); parts.push(b.slice()); },
            async close() { d.files.set(name, new Uint8Array(parts.flatMap((p) => [...p]))); calls.push(`closed ${path}${name}`); },
            async abort() { calls.push(`aborted ${path}${name}`); },
          };
        },
      };
    },
  }) as unknown as FileSystemDirectoryHandle;
  const tree: Dir = { dirs: new Map(), files: new Map() };
  return { tree, handle: make("", tree) };
}

test("a directory as a store: the root and subdirectories made once each, each file written whole", async () => {
  const calls: string[] = [], { tree, handle } = fakeDirectory(calls);
  const store = directoryStore(handle, "runs/mito.zarr");
  await Promise.all([store.put("0/c/0/0/0", new Uint8Array([1])), store.put("0/c/0/0/1", new Uint8Array([2])), store.put("zarr.json", new Uint8Array([3]))]);
  assert.deepEqual(calls.filter((c) => c.startsWith("dir")), ["dir runs", "dir runs/mito.zarr", "dir runs/mito.zarr/0", "dir runs/mito.zarr/0/c", "dir runs/mito.zarr/0/c/0", "dir runs/mito.zarr/0/c/0/0"]);
  const zarrDir = tree.dirs.get("runs")!.dirs.get("mito.zarr")!;
  assert.deepEqual(zarrDir.files.get("zarr.json"), new Uint8Array([3]));
  assert.deepEqual([...zarrDir.dirs.get("0")!.dirs.get("c")!.dirs.get("0")!.dirs.get("0")!.files], [["0", new Uint8Array([1])], ["1", new Uint8Array([2])]]);
  // a write that fails is aborted (the swap file dropped) and rejects
  const failing = fakeDirectory(calls, "bad");
  await assert.rejects(directoryStore(failing.handle).put("bad", new Uint8Array([1])), /quota exceeded/);
  assert.ok(calls.includes("aborted bad"));
  assert.equal(failing.tree.files.size, 0);
});

// ------------------------------------------------------------- a box as exactly the box

/** Voxel (z, y, x) of channel c of level 0 as the box tests' producer makes it: never zero
 * inside the level, and no two alike. */
const voxel = (z: number, y: number, x: number, c = 0) => 1 + 10000 * c + 1000 * z + 20 * y + x;
/** `view` with a level 0 of 9 × 10 × 11 voxels: 5 × 3 × 3 chunks of 2 × 4 × 4, the last
 * along each axis partial. */
const big: ViewInfo = { ...view, levels: [{ shape: [9, 10, 11], voxel: [8, 8, 8], origin: [0, 0, 0] }, view.levels[1]] };
/** `big` with two channels its ops add, as float32. */
const bigChannels: ViewInfo = { ...big, out: "float32", added: [2] };

/** Level 0 of `v` as the engine gives its chunks: whole, zero past the level's edge, the
 * channels the ops add first; the chunks of `empty` all zero; each after `delay(index)`
 * ticks. `calls` counts the asks for each chunk, `asked` lists them in order. */
function served(v: ViewInfo, { empty = [] as string[], delay = (_: number[]) => 0 } = {}) {
  const [Z, Y, X] = v.levels[0].shape, K = (v.added ?? []).reduce((n, c) => n * c, 1), n = chunk[0] * chunk[1] * chunk[2];
  const make = (index: number[]): ArrayBuffer => {
    const out = v.out === "float32" ? new Float32Array(K * n) : new Uint16Array(K * n);
    if (empty.includes(index.join("."))) return out.buffer;
    for (let c = 0; c < K; c++) for (let z = 0; z < chunk[0]; z++) for (let y = 0; y < chunk[1]; y++) for (let x = 0; x < chunk[2]; x++) {
      const g = [index[0] * chunk[0] + z, index[1] * chunk[1] + y, index[2] * chunk[2] + x];
      if (g[0] < Z && g[1] < Y && g[2] < X) out[c * n + (z * chunk[1] + y) * chunk[2] + x] = voxel(g[0], g[1], g[2], c);
    }
    return out.buffer;
  };
  const calls = new Map<string, number>(), asked: string[] = [];
  const produce: Produce = async (index) => {
    const k = index.join(".");
    calls.set(k, (calls.get(k) ?? 0) + 1);
    asked.push(k);
    for (let i = delay(index); i > 0; i--) await tick();
    return make(index);
  };
  return { produce, make, calls, asked };
}

/** The array at "0" of `store` as zarrita reads it. */
async function readArray(store: { files: Map<string, Uint8Array> }, format: "zarr2" | "zarr3") {
  const at = zarr.root(readable(store)).resolve("0");
  const arr = format === "zarr3" ? await zarr.open.v3(at, { kind: "array" }) : await zarr.open.v2(at, { kind: "array" });
  return { shape: arr.shape, dtype: arr.dtype, data: (await zarr.get(arr)).data as Uint16Array | Float32Array };
}

/** That `data`, of `shape` (its channels first), is the view's voxels from voxel `lo` of the
 * level; zero where `zero` says (a chunk of the view left empty). */
function assertBox(data: ArrayLike<number>, shape: number[], lo: number[], zero: (g: number[]) => boolean = () => false) {
  const [Z, Y, X] = shape.slice(-3), K = shape.length > 3 ? shape[0] : 1;
  for (let c = 0; c < K; c++) for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    const g = [lo[0] + z, lo[1] + y, lo[2] + x];
    assert.equal(data[((c * Z + z) * Y + y) * X + x], zero(g) ? 0 : voxel(g[0], g[1], g[2], c), `channel ${c}, voxel ${g}`);
  }
}

/** The C-order cells of the grid [lo, hi), as "z.y.x". */
const cells = (lo: number[], hi: number[]) => [...chunkIndices(lo, hi)].map((i) => i.join("."));

test("a box's array is exactly the box: its shape hi − lo, the view's chunks counted from lo, the level's translation moved by lo voxels", () => {
  // level 1: 3 × 3 × 4 voxels of 16 nm, voxel 0's centre at 4 nm
  const two = region("mito", view, chunk, { level: 1, extent: "box", lo: [1, 0, 2], hi: [3, 2, 4] });
  const [, attrs, zarray] = two.files.map(([, b]) => json(b));
  assert.deepEqual(attrs.multiscales[0].datasets, [{ path: "0", coordinateTransformations: [
    { type: "scale", scale: [16, 16, 16] }, { type: "translation", translation: [4 + 16, 4, 4 + 2 * 16] }] }]);
  assert.deepEqual([zarray.shape, zarray.chunks], [[2, 2, 2], [2, 4, 4]]);
  // the view's chunks it needs (z 0 and 1), the array's one chunk
  assert.deepEqual([two.lo, two.hi, two.needed, two.first, two.last, two.total, two.key([0, 0, 0])], [[0, 0, 0], [2, 1, 1], 2, [0, 0, 0], [1, 1, 1], 1, "0/0/0/0"]);
  const three = region("mito", view, chunk, { format: "zarr3", extent: "box", lo: [1, 3, 2], hi: [4, 6, 7] });
  const [group, array] = three.files.map(([, b]) => json(b));
  assert.deepEqual(group.attributes.ome.multiscales[0].datasets[0].coordinateTransformations[1], { type: "translation", translation: [8, 24, 16] });
  assert.deepEqual([array.shape, array.chunk_grid.configuration.chunk_shape], [[3, 3, 5], [2, 4, 4]]);
  assert.deepEqual([three.needed, three.total, three.key([1, 0, 1])], [2 * 2 * 2, 2 * 1 * 2, "0/c/1/0/1"]);
  // clipped to the level and rounded out to whole voxels, not to chunks
  assert.deepEqual(voxelBox([5, 6, 7], [-3, 0.5, 1.5], [2.5, 100, 3.2]), { lo: [0, 0, 1], hi: [3, 6, 4] });
  assert.deepEqual(region("v", view, chunk, { extent: "box", lo: [-3, 0.5, 1.5], hi: [2.5, 100, 3.2] }).shape, [3, 6, 3]);
  assert.throws(() => region("v", view, chunk, { extent: "box", lo: [5, 0, 0] }), /holds no voxel of the level, which is 5×6×7/);
  assert.throws(() => region("v", view, chunk, { extent: "slab" as "box" }), /extent "slab": "level" or "box"/);
  // the channels the ops add: whole, first, untranslated
  const grad = region("grad", channels, chunk, { extent: "box", lo: [1, 3, 2], hi: [4, 6, 7] });
  const [, gAttrs, gArray] = grad.files.map(([, b]) => json(b));
  assert.deepEqual([gArray.shape, gArray.chunks], [[2, 3, 3, 5], [2, 2, 4, 4]]);
  assert.deepEqual(gAttrs.multiscales[0].datasets[0].coordinateTransformations[1].translation, [0, 8, 24, 16]);
  assert.equal(grad.key([1, 0, 1]), "0/0/1/0/1");
  // the level's own extent is as before: the level's whole shape and translation
  const level = region("mito", view, chunk, { level: 1, lo: [1, 0, 2], hi: [3, 2, 4] });
  assert.deepEqual(json(level.files[2][1]).shape, [3, 3, 4]);
  assert.deepEqual(json(level.files[1][1]).multiscales[0].datasets[0].coordinateTransformations[1].translation, [4, 4, 4]);
});

for (const format of ["zarr2", "zarr3"] as const) {
  test(`${format}: a box across chunk edges on every axis, each of its chunks stitched from up to 8 of the view's, each of those made once, read back by zarrita as the view over the box`, async () => {
    const lo = [1, 3, 2], hi = [8, 9, 11];  // 7 × 6 × 9: 4 × 2 × 3 chunks, of the view's 4 × 3 × 3; to the level's end along x
    const fake = served(big, { delay: (i) => (i[0] + 2 * i[1] + i[2]) % 3 });  // made out of turn
    const store = memoryStore(), seen: MaterializeProgress[] = [];
    const got = await writeChunks(store, region("v", big, chunk, { format, extent: "box", lo, hi }), fake.produce,
      { concurrency: 3, onProgress: (p) => seen.push(p) });
    // each of the view's chunks the box needs asked for once, and no other
    assert.deepEqual([...fake.calls.keys()].sort(), cells([0, 0, 0], [4, 3, 3]).sort());
    assert.ok([...fake.calls.values()].every((n) => n === 1));
    const keys = files(store).filter((f) => !f.endsWith("json") && !f.includes(".z"));
    assert.deepEqual(keys, cells([0, 0, 0], [4, 2, 3]).map((i) => (format === "zarr3" ? "0/c/" : "0/") + i.replaceAll(".", "/")).sort());
    const bytes = keys.reduce((n, k) => n + store.files.get(k)!.length, 0);
    assert.deepEqual(got, { chunks: 24, written: 24, skipped: 0, bytes, aborted: false });
    const arr = await readArray(store, format);
    assert.deepEqual([arr.shape, arr.dtype], [[7, 6, 9], "uint16"]);
    assertBox(arr.data, arr.shape, lo);
    // the far corner's chunk overhangs the box (1 of 2 along z, 2 of 4 along y, 1 of 4 along x),
    // stored whole, as zarr stores an edge chunk, zero past the box
    const corner = new Uint16Array((await gunzip(store.files.get(format === "zarr3" ? "0/c/3/1/2" : "0/3/1/2")!)).buffer);
    assert.equal(corner.length, chunk[0] * chunk[1] * chunk[2]);
    corner.forEach((v, i) => {
      const [z, y, x] = [Math.floor(i / 16), Math.floor(i / 4) % 4, i % 4];
      assert.equal(v, z < 1 && y < 2 && x < 1 ? voxel(lo[0] + 6 + z, lo[1] + 4 + y, lo[2] + 8 + x) : 0, `corner ${z}, ${y}, ${x}`);
    });
    // progress: the array's chunks done, the view's computed, each only growing
    assert.deepEqual(seen[0], { done: 0, total: 24, written: 0, bytes: 0, computed: 0, needed: 36 });
    assert.deepEqual(seen.at(-1), { done: 24, total: 24, written: 24, bytes, computed: 36, needed: 36 });
    assert.equal(seen.length, 1 + 36);  // once more for each of the view's chunks
    for (const [i, p] of seen.slice(1).entries()) {
      for (const k of ["done", "written", "bytes", "computed"] as const) assert.ok(p[k] >= seen[i][k], `${k} went back at report ${i + 1}`);
      assert.ok(p.computed >= i + 1);  // another lane's chunk may be counted before it reports
    }
  });
}

test("a box on the view's chunk edges: its chunks are the view's as they are; one ending inside a chunk crops it", async () => {
  // from chunk (1, 1, 1) to the level's end: 7 × 6 × 7, the view's chunks (1..4, 1..2, 1..2)
  const fake = served(big), store = memoryStore();
  const r = region("v", big, chunk, { extent: "box", compression: null, lo: [2, 4, 4] });
  assert.deepEqual([r.shape, r.lo, r.hi, r.total, r.needed], [[7, 6, 7], [1, 1, 1], [5, 3, 3], 16, 16]);
  assert.deepEqual(await writeChunks(store, r, fake.produce), { chunks: 16, written: 16, skipped: 0, bytes: 16 * 64, aborted: false });
  for (const o of chunkIndices([0, 0, 0], [4, 2, 2])) {
    assert.deepEqual(store.files.get(r.key(o)), new Uint8Array(fake.make(o.map((i) => i + 1))), `chunk ${o}`);  // byte for byte
  }
  assertBox((await readArray(store, "zarr2")).data, [7, 6, 7], [2, 4, 4]);
  const s = stitcher(r);
  for (const i of chunkIndices(r.lo, r.hi)) {
    assert.equal(s.add(i, new Uint8Array(fake.make(i))).length, 1);
    assert.equal(s.open, 0);  // none held: each passes through
  }
  // on chunk edges at its start, but its end inside chunks short of the level's: 3 × 3 × 2, cropped
  const short = memoryStore(), cropped = region("v", big, chunk, { extent: "box", compression: null, lo: [2, 4, 4], hi: [5, 7, 6] });
  await writeChunks(short, cropped, served(big).produce);
  const second = new Uint16Array(short.files.get("0/1/0/0")!.slice().buffer);  // z 4 of 4..5, y 4..6 of 4..7, x 4..5 of 4..7
  second.forEach((v, i) => {
    const [z, y, x] = [Math.floor(i / 16), Math.floor(i / 4) % 4, i % 4];
    assert.equal(v, z < 1 && y < 3 && x < 2 ? voxel(4 + z, 4 + y, 4 + x) : 0, `voxel ${z}, ${y}, ${x}: the view's past the box not kept`);
  });
  assertBox((await readArray(short, "zarr2")).data, [3, 3, 2], [2, 4, 4]);
});

test("a box smaller than a chunk: inside one of the view's, and across the corner of eight", async () => {
  const inside = served(view), one = memoryStore();
  const got = await writeChunks(one, region("v", view, chunk, { extent: "box", compression: null, lo: [1, 1, 1], hi: [2, 3, 2] }), inside.produce);
  assert.deepEqual([got, [...inside.calls.keys()]], [{ chunks: 1, written: 1, skipped: 0, bytes: 64, aborted: false }, ["0.0.0"]]);
  const stored = new Uint16Array(one.files.get("0/0/0/0")!.slice().buffer);  // whole, the two voxels of the box first
  assert.deepEqual([stored.length, stored[0], stored[4], stored.filter(Boolean).length], [32, voxel(1, 1, 1), voxel(1, 2, 1), 2]);
  assertBox((await readArray(one, "zarr2")).data, [1, 2, 1], [1, 1, 1]);
  const corner = served(view), eight = memoryStore();
  await writeChunks(eight, region("v", view, chunk, { format: "zarr3", extent: "box", lo: [1, 3, 3], hi: [3, 5, 5] }), corner.produce, { concurrency: 3 });
  assert.deepEqual([...corner.calls.keys()].sort(), cells([0, 0, 0], [2, 2, 2]));
  assert.ok([...corner.calls.values()].every((n) => n === 1));
  const arr = await readArray(eight, "zarr3");
  assert.deepEqual(arr.shape, [2, 2, 2]);
  assertBox(arr.data, arr.shape, [1, 3, 3]);
});

for (const format of ["zarr2", "zarr3"] as const) {
  test(`${format}: a box of the whole level is the level's extent, file for file and byte for byte`, async () => {
    for (const box of [{}, { lo: [0, 0, 0], hi: [5, 6, 7] }, { lo: [-1, -1, -1], hi: [9, 9, 9] }] as MaterializeOptions[]) {
      const level = memoryStore(), whole = memoryStore();
      await writeChunks(level, region("v", view, chunk, { format, ...box }), producer(["1.0.1"]));
      await writeChunks(whole, region("v", view, chunk, { format, extent: "box", ...box }), producer(["1.0.1"]));
      assert.deepEqual(whole.files, level.files);
    }
  });
}

test("skipEmpty: a box's chunk is left out when the view's chunks it is made of are all empty; one made of some is written, zero there", async () => {
  // 4 × 8 × 8 from (1, 1, 1): 2 × 2 × 2 chunks, the first made of the view's (0..1, 0..1, 0..1), all empty
  const empty = cells([0, 0, 0], [2, 2, 2]), zero = (g: number[]) => g[0] < 4 && g[1] < 8 && g[2] < 8;
  const opts: MaterializeOptions = { extent: "box", lo: [1, 1, 1], hi: [5, 9, 9] };
  const store = memoryStore();
  assert.deepEqual(await writeChunks(store, region("v", big, chunk, opts), served(big, { empty }).produce),
    { chunks: 8, written: 7, skipped: 1, bytes: [...store.files].filter(([k]) => /^0\/\d/.test(k)).reduce((n, [, b]) => n + b.length, 0), aborted: false });
  assert.ok(!store.files.has("0/0/0/0") && store.files.has("0/0/0/1"));
  assertBox((await readArray(store, "zarr2")).data, [4, 8, 8], [1, 1, 1], zero);
  // kept when asked: written whole, all zero
  const all = memoryStore();
  const got = await writeChunks(all, region("v", big, chunk, { ...opts, compression: null }), served(big, { empty }).produce, { skipEmpty: false });
  assert.deepEqual([got.written, got.skipped, all.files.get("0/0/0/0")], [8, 0, new Uint8Array(64)]);
  assertBox((await readArray(all, "zarr2")).data, [4, 8, 8], [1, 1, 1], zero);
});

test("a box's chunk is written once the last of the view's chunks it is made of comes; a slab of them held at most", async () => {
  // from (1, 1, 1) to the level's end: off the chunk edges on every axis, 8 × 9 × 10 in 4 × 3 × 3 chunks
  const r = region("v", big, chunk, { extent: "box", compression: null, lo: [1, 1, 1] }), events: string[] = [];
  const fake = served(big), produce: Produce = async (i, signal) => { events.push(`make ${i.join(".")}`); return fake.produce(i, signal); };
  const store: WritableStore = { async put(path) { if (/^0\/\d/.test(path)) events.push(`write ${path}`); } };
  await writeChunks(store, r, produce);
  const writes = events.filter((e) => e.startsWith("write"));
  assert.equal(writes.length, r.total);
  for (const [i, e] of events.entries()) {
    if (!e.startsWith("write")) continue;
    // the last of the view's chunks it is made of, in C order: the one each axis's last part comes from
    const o = e.slice("write 0/".length).split("/").map(Number);
    const last = o.map((x, a) => Math.floor((r.at[a] + Math.min((x + 1) * chunk[a], r.shape[a]) - 1) / chunk[a]));
    assert.equal(events.slice(0, i).filter((x) => x.startsWith("make")).at(-1), `make ${last.join(".")}`, `${e} as soon as it is complete`);
  }
  // held: off chunk edges along z, a slab of the box's chunks (3 × 3), a row (3) and one more at most
  const slab = (lo: number[]) => {
    const s = stitcher(region("v", big, chunk, { extent: "box", compression: null, lo })), r2 = region("v", big, chunk, { extent: "box", lo });
    let most = 0;
    for (const i of chunkIndices(r2.lo, r2.hi)) { s.add(i, new Uint8Array(fake.make(i))); most = Math.max(most, s.open); }
    assert.equal(s.open, 0);  // all complete at the end
    return most;
  };
  assert.equal(slab([1, 1, 1]), 3 * 3 + 3 + 1);
  assert.equal(slab([2, 1, 1]), 3 + 1);  // on edges along z: a row and one
  assert.equal(slab([2, 4, 1]), 1);      // along z and y: one
  assert.equal(slab([2, 4, 4]), 0);
  // the view's chunks all empty: nothing held in memory but the counts (no buffer made)
  const r3 = region("v", big, chunk, { extent: "box", lo: [1, 1, 1] }), s3 = stitcher(r3);
  const done = [...chunkIndices(r3.lo, r3.hi)].flatMap((i) => s3.add(i, new Uint8Array(64)));
  assert.deepEqual([done.length, done.every(([, b]) => b === null)], [r3.total, true]);
});

test("a box's abort: no chunk asked for or written after it, those asked for given up; the chunks written stay", { timeout: 5000 }, async () => {
  const store = memoryStore(), stop = new AbortController(), asked: string[] = [], given: AbortSignal[] = [], fake = served(big);
  let atAbort = -1;
  // the first 16 of the view's chunks come at once, the rest never; the box's first chunk is complete with the 14th
  const produce: Produce = (index, signal) => {
    asked.push(index.join("."));
    given.push(signal);
    return asked.length <= 16 ? Promise.resolve(fake.make(index)) : new Promise<ArrayBuffer>(() => {});
  };
  const got = await writeChunks(store, region("v", big, chunk, { extent: "box", lo: [1, 1, 1] }), produce, {
    concurrency: 2, signal: stop.signal, onProgress: (p) => { if (p.written >= 1 && atAbort < 0) { stop.abort(); atAbort = asked.length; } },
  });
  assert.deepEqual([got.chunks, got.aborted], [36, true]);
  assert.ok(got.written >= 1);
  assert.equal(asked.length, atAbort);  // none asked for after the abort
  assert.ok(given.every((s) => s.aborted));
  assert.ok(store.files.has("0/0/0/0"));
  assert.equal(files(store).filter((f) => /^0\/\d/.test(f)).length, got.written);
});

test("a box's chunk that fails rejects naming it as the engine does, once the writes under way have ended; no write starts after", async () => {
  // 4 × 5 × 6 from (1, 1, 1) of view: the view's chunk (1, 1, 1) completes four of its chunks, (2, 0, 0) fails
  let slowWritten = false;
  const memory = memoryStore();
  const store: WritableStore = {
    async put(path, bytes) {
      if (path === "0/0/0/0") { await new Promise((r) => setTimeout(r, 30)); slowWritten = true; }
      return memory.put(path, bytes);
    },
  };
  const fake = served(view);
  const produce: Produce = async (index, signal) => {
    if (index.join(".") === "2.0.0") { await tick(); throw new Error("the model ran out of GPU memory"); }
    return fake.produce(index, signal);
  };
  await assert.rejects(writeChunks(store, region("v", view, chunk, { extent: "box", lo: [1, 1, 1] }), produce, { concurrency: 2 }), (e: Error) => {
    assert.equal(e.message, "chunk v/0/2.0.0: the model ran out of GPU memory");
    assert.ok(slowWritten, "the chunk being written when the other failed was written first");
    return true;
  });
  assert.deepEqual(files(memory).filter((f) => /^0\/\d/.test(f)), ["0/0/0/0"]);  // not the three it completed with it
  // a chunk of the view that is not a whole chunk cannot be cut
  await assert.rejects(writeChunks(memoryStore(), region("v", view, chunk, { extent: "box", lo: [1, 1, 1] }), async () => new ArrayBuffer(10)),
    /^Error: chunk v\/0\/0\.0\.0: 10 bytes, not a whole chunk's 64$/);
});

for (const format of ["zarr2", "zarr3"] as const) {
  test(`${format}: a box of a view with channel axes its ops add: the channels whole and first, each stitched alike`, async () => {
    const lo = [1, 3, 2], hi = [8, 9, 11], fake = served(bigChannels), store = memoryStore();
    const got = await writeChunks(store, region("grad", bigChannels, chunk, { format, extent: "box", lo, hi }), fake.produce, { concurrency: 2 });
    assert.deepEqual([got.chunks, got.written], [24, 24]);
    assert.ok([...fake.calls.values()].every((n) => n === 1) && fake.calls.size === 36);
    assert.ok(store.files.has(format === "zarr3" ? "0/c/0/3/1/2" : "0/0/3/1/2"));
    const arr = await readArray(store, format);
    assert.deepEqual([arr.shape, arr.dtype], [[2, 7, 6, 9], "float32"]);
    assertBox(arr.data, arr.shape, lo);
  });
}
