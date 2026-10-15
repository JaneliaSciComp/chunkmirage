// A view served as read (src/asread.ts): which views need the Pyodide workers, the plan
// fused.plan gives a view with no ops (checked against Python's own `plan` in pyworker.ts),
// and the chunk a worker makes of a region with no ops to run. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { asRead, needsPython, wholeChunk } from "../src/asread.ts";
import type { SourceInfo } from "../src/types.ts";

const source: SourceInfo = {
  dtype: "uint16", channels: 1,
  axes: [{ name: "z", unit: "nm" }, { name: "y", unit: "nm" }, { name: "x", unit: "nm" }],
  levels: [{ shape: [4, 6, 8], voxel: [8, 8, 8], origin: [0, 0, 0] }],
};

test("a view read as stored needs no Python; ops, a computed source or a mesh do", () => {
  assert.equal(needsPython({ source: "https://bucket/image.zarr", chunk: [1, 256, 256] }), false);
  assert.equal(needsPython({ source: "https://bucket/image.zarr", chunk: [1, 256, 256], ops: [], select: { c: 1 } }), false);
  assert.equal(needsPython({ source: "https://bucket/image.zarr", chunk: [1, 256, 256], ops: [{ op: "threshold", level: 128 }] }), true);
  assert.equal(needsPython({ source: "synthetic://mandelbulb", chunk: [32, 32, 32] }), true);
  assert.equal(needsPython({ source: "https://bucket/dem.tif", chunk: [1, 200, 200], mesh: { kind: "terrain" } }), true);
});

test("the plan of a view with no ops is its source's, as fused.plan gives it", () => {
  // what pyworker.ts's plan(view, "[]", shape, dtype, chunk, voxel) returns for these
  assert.deepEqual(asRead(source), { dtype: "uint16", kind: null, lead: 0, halo: [0, 0, 0], ndim: 3, added: [], controls: {} });
  // channels (stack://) are one leading axis, read whole and served as a channel axis
  assert.deepEqual(asRead({ ...source, channels: 3 }), { dtype: "uint16", kind: null, lead: 1, halo: [0, 0, 0], ndim: 4, added: [3], controls: {} });
});

test("a chunk inside the array is the region as read; one at the edge is zero past it", () => {
  const read = new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8]);  // (z, y, x) = (2, 2, 2): the array ends one voxel short of the chunk along x
  assert.equal(wholeChunk(read.buffer, "uint16", 1, [2, 2, 2], [2, 2, 2]), read.buffer);
  const edge = new Uint16Array(wholeChunk(read.buffer, "uint16", 1, [2, 2, 2], [2, 2, 3]));
  assert.deepEqual([...edge], [1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0]);
  const corner = new Uint16Array(wholeChunk(read.buffer, "uint16", 1, [2, 2, 2], [3, 3, 2]));
  assert.deepEqual([...corner], [1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
});

test("the channels come first, each a whole chunk", () => {
  const read = new Float32Array([1, 2, 3, 4]);  // two channels of (1, 1, 2), one after the other
  const out = new Float32Array(wholeChunk(read.buffer, "float32", 2, [1, 1, 2], [1, 1, 3]));
  assert.deepEqual([...out], [1, 2, 0, 3, 4, 0]);
});
