// A view computed in stages on the page (src/stages.ts): the first stage's input box on its
// grid (fused.input_box's rule), the level at a voxel size, a block padded past the array's
// edge as the Python pipeline pads (edge or zero), and a stage's result fitted to its
// output box, whole or shaved by its halo. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { bytes, crop, fitted, inputBox, levelAt, padded } from "../src/stages.ts";

test("the input box is the output box on the input's grid, grown by the halo, and must be whole", () => {
  assert.deepEqual(inputBox([0, 2, 4], [2, 4, 8], [1, 1, 1], [1, 2, 3]), { lo: [-1, 0, 1], hi: [3, 6, 11] });
  // 16 nm output voxels from 8 nm input ones: two input voxels to each
  assert.deepEqual(inputBox([0, 2, 4], [2, 4, 8], [2, 2, 2], [0, 0, 0]), { lo: [0, 4, 8], hi: [4, 8, 16] });
  // 8 nm output from 16 nm input: half a voxel each, whole only for even chunks
  assert.deepEqual(inputBox([0, 2, 4], [2, 4, 8], [0.5, 0.5, 0.5], [0, 0, 0]), { lo: [0, 1, 2], hi: [1, 2, 4] });
  assert.throws(() => inputBox([0, 0, 3], [2, 2, 5], [0.5, 0.5, 0.5], [0, 0, 0]), /not whole voxels of the input/);
});

test("the level at a voxel size, to a millionth", () => {
  const levels = [{ voxel: [8, 8, 8] }, { voxel: [16, 16, 16] }, { voxel: [32, 32, 32] }];
  assert.equal(levelAt(levels, [16, 16, 16]), 1);
  assert.equal(levelAt(levels, [16.000001, 16, 16]), 1);
  assert.equal(levelAt(levels, [12, 12, 12]), -1);
});

/** A (z, y, x) = (2, 2, 3) block of uint16, read over [0, 2) × [0, 2) × [0, 3). */
const read = new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

test("a box inside the array is the region as read", () => {
  const b = padded(read.buffer, "uint16", 1, [0, 0, 0], [2, 2, 3], [0, 0, 0], [2, 2, 3], "edge");
  assert.deepEqual(b.shape, [2, 2, 3]);
  assert.equal(b.data.buffer, read.buffer);  // the bytes read, not a copy
  assert.deepEqual([...b.data], [...read]);
});

test("past the array's edge, zeros or the outermost voxels repeated", () => {
  // the box one voxel wider on every side of the (2, 2, 3) array, which is all there is to read
  const zero = padded(read.buffer, "uint16", 1, [0, 0, 0], [2, 2, 3], [-1, -1, -1], [3, 3, 4], "zero");
  assert.deepEqual(zero.shape, [4, 4, 5]);
  assert.equal(zero.data.length, 80);
  // the inner voxels are the array's, the border zero
  const at = (z: number, y: number, x: number) => zero.data[(z * 4 + y) * 5 + x];
  assert.deepEqual([at(1, 1, 1), at(1, 1, 3), at(2, 2, 3)], [1, 3, 12]);
  assert.deepEqual([at(0, 1, 1), at(1, 0, 1), at(1, 1, 0), at(1, 1, 4), at(3, 3, 4)], [0, 0, 0, 0, 0]);
  const edge = padded(read.buffer, "uint16", 1, [0, 0, 0], [2, 2, 3], [-1, -1, -1], [3, 3, 4], "edge");
  const e = (z: number, y: number, x: number) => edge.data[(z * 4 + y) * 5 + x];
  assert.deepEqual([e(1, 1, 1), e(2, 2, 3)], [1, 12]);  // the array's own
  assert.deepEqual([e(0, 0, 0), e(0, 1, 1), e(1, 0, 1), e(1, 1, 0), e(1, 1, 4), e(3, 3, 4)], [1, 1, 1, 1, 3, 12]);  // the nearest voxel inside
});

test("a box partly inside, read clipped, is placed where it lies; channels one after another", () => {
  // two channels of a (1, 1, 2) array, the box [0, 1) × [0, 1) × [-1, 3): one voxel past each end along x
  const two = new Float32Array([1, 2, 3, 4]);
  const b = padded(two.buffer, "float32", 2, [0, 0, 0], [1, 1, 2], [0, 0, -1], [1, 1, 3], "zero");
  assert.deepEqual(b.shape, [2, 1, 1, 4]);
  assert.deepEqual([...b.data], [0, 1, 2, 0, 0, 3, 4, 0]);
  const wholly = padded(null, "float32", 1, [0, 0, 0], [0, 0, 0], [5, 5, 5], [6, 6, 7], "edge");  // nothing read: zeros
  assert.deepEqual([wholly.shape, [...wholly.data]], [[1, 1, 2], [0, 0]]);
});

test("a stage's result fits its output box as it is, or shaved by its halo", () => {
  const inShape = [4, 4, 4], shape = [2, 2, 2], halo = [1, 1, 1];
  const exact = new Float32Array(8).fill(7);
  assert.deepEqual(fitted(exact, "float32", shape, inShape, halo), { data: exact, shape });
  const whole = Float32Array.from({ length: 64 }, (_, i) => i);
  const shaved = fitted({ data: whole, shape: inShape }, "float32", shape, inShape, halo);
  assert.deepEqual(shaved.shape, shape);
  assert.deepEqual([...shaved.data], [21, 22, 25, 26, 37, 38, 41, 42]);  // (1..2, 1..2, 1..2) of the 4³ block
  assert.throws(() => fitted({ data: whole, shape: inShape }, "float32", shape, inShape, null), /not its output's 2×2×2$/);  // on another grid: the output's shape only
  assert.throws(() => fitted(whole, "float32", shape, inShape, halo), /64 values for a block of 2×2×2/);  // data alone is taken as the output's shape
  assert.throws(() => fitted(new Uint8Array(8), "float32", shape, inShape, halo), /not the float32 it declares/);
  assert.throws(() => fitted({ data: new Float32Array(5), shape: [1, 1, 5] }, "float32", shape, inShape, halo), /not its output's 2×2×2 or that grown by its halo, 4×4×4$/);
});

test("added axes come first and are cropped with the block", () => {
  // two channels of a (1, 1, 4) block, halo 1 along x: the two middle voxels of each
  const block = { data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), shape: [2, 1, 1, 4] };
  assert.deepEqual([...crop(block, [0, 0, 1], [2, 1, 1, 2]).data], [2, 3, 6, 7]);
  const view = new Float32Array(block.data.buffer, 8, 2);  // a view on part of a buffer
  assert.deepEqual([...new Float32Array(bytes(view))], [3, 4]);
});
