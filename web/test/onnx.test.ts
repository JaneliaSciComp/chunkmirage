// The onnx stage (src/onnx.ts) through ONNX Runtime's WebAssembly backend, which runs in Node
// as in a browser without WebGPU: two models of a hundred bytes (test/fixtures, written by
// make_onnx.py there), their forward pass as a stage's block, the channels a stage declares
// checked against what the model wrote, a session shared by its users and released by the
// last. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { exclusive, Onnx, onnxStage } from "../src/onnx.ts";
import type { StageAt } from "../src/stages.ts";

const fixture = (name: string) => ({
  key: name,
  bytes: async () => {
    const b = await readFile(new URL(`./fixtures/${name}.onnx`, import.meta.url));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  },
});
const at: StageAt = { view: "v", level: 0, index: [0, 0, 0], lo: [0, 0, 0], hi: [2, 2, 2], voxel: [1, 1, 1] };
const block = { data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]), shape: [2, 2, 2] };
const ready = (lines: string[]) => lines.filter((l) => l.includes("session ready")).length;

test("a model's forward pass is the stage's block, on the WASM backend where there is no WebGPU", async () => {
  const lines: string[] = [], onnx = new Onnx({ log: (l) => lines.push(l) });
  const out = await onnxStage(onnx, fixture("double")).produce(block, at);
  assert.deepEqual(out.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(out.data as Float32Array), [2, 4, 6, 8, 10, 12, 14, 16]);
  assert.equal(onnx.provider, "wasm");
  assert.equal(ready(lines), 1, lines.join("\n"));
  await onnx.reset();
});

test("the channels a model writes are the stage's added axis, and must be what the stage declares", async () => {
  const onnx = new Onnx();
  const out = await onnxStage(onnx, fixture("pair"), { added: [2] }).produce(block, at);
  assert.deepEqual(out.shape, [2, 2, 2, 2]);
  assert.deepEqual(Array.from(out.data as Float32Array).slice(8), [1, 2, 3, 4, 5, 6, 7, 8]);  // the second copy
  await assert.rejects(onnxStage(onnx, fixture("pair")).produce(block, at), /wrote 2 channels, its stage declares none/);
  await assert.rejects(onnxStage(onnx, fixture("double"), { added: [2] }).produce(block, at), /wrote 1 channel, its stage declares 2/);
  await onnx.reset();
});

test("a session is shared by its users and released by the last; a release nobody retained is nothing", async () => {
  const lines: string[] = [], onnx = new Onnx({ log: (l) => lines.push(l) });
  const model = fixture("double"), stage = onnxStage(onnx, model);
  await stage.produce(block, at);
  await onnx.release(model.key);  // not retained by anyone: the session stays
  onnx.retain(model.key);
  onnx.retain(model.key);
  await onnx.release(model.key);  // one user left
  await stage.produce(block, at);
  assert.equal(ready(lines), 1);
  await onnx.release(model.key);  // the last: released
  await stage.produce(block, at);
  assert.equal(ready(lines), 2);  // built again
  await onnx.reset();
});

test("passes take turns across sessions and callers: one starts when the one before it has ended, however it ended", async () => {
  const events: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const first = exclusive(async () => { events.push("first starts"); await gate; events.push("first ends"); throw new Error("first fails"); });
  const second = exclusive(async () => { events.push("second starts"); return 2; });
  await new Promise((r) => setTimeout(r, 10));
  assert.deepEqual(events, ["first starts"]);  // the second waits, though it needs nothing of the first
  release();
  await assert.rejects(first, /first fails/);
  assert.equal(await second, 2);
  assert.deepEqual(events, ["first starts", "first ends", "second starts"]);
});

test("a session is built in the same turn: a model's first block waits for the call before it", async () => {
  const lines: string[] = [], onnx = new Onnx({ log: (l) => lines.push(l) });
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const held = exclusive(() => gate);  // another model's pass, say, still running
  const out = onnxStage(onnx, fixture("double")).produce(block, at);
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(ready(lines), 0, "not built while another call holds the turn");
  release();
  await held;
  assert.deepEqual(Array.from((await out).data as Float32Array), [2, 4, 6, 8, 10, 12, 14, 16]);
  assert.equal(ready(lines), 1);
  await onnx.reset();
});
