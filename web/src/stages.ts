// A view the page computes in stages (Engine.serveStages): the page's side of the Python
// pipeline, whose stages are each cached under their own hash. Each stage's output is kept
// under the stage's key, not the view's name, so an edit of stage k reuses what the stages
// before it made, two views sharing a stage share its outputs, and a view served again
// under a new name after an edit (viewers cache by URL) finds its earlier stages' outputs.
// The geometry is fused.py's: a stage's output block is the chunk grown by the halos of the
// stages after it; the first stage reads the source level at its voxel size
// (`inputVoxelSize`, as Op.input_voxel_size: a model that reads 8 nm voxels and writes 16 nm
// ones) over its box on that grid, grown by its halo and padded past the array's edge (the
// view's `padding`, edge or zero, as the Python pipeline's); and a stage returns its block
// whole or shaved by its halo, as an op may.
import { TYPED, type Numbers, type Settable } from "./asread.ts";  // with its extension: Node's runner (npm test) resolves no other

/** A block of voxels: `data` in C order over `shape`, any leading axes (channels) first, the
 * typed array of its dtype (asread.ts's table: a BigUint64Array for uint64, so a
 * segmentation's labels past 2^53 pass through whole). */
export interface Block { data: Numbers; shape: number[] }

/** Where a stage computes: the view and level, the chunk's index, the stage's output box
 * [lo, hi) in the level's voxels, and that level's voxel size. */
export interface StageAt { view: string; level: number; index: number[]; lo: number[]; hi: number[]; voxel: number[] }

/** One stage of a view the page computes. */
export interface Stage {
  /** What the stage computes, hashed by the page (a model's weights and parameters, say):
   * its outputs are kept under it, so stages equal in all but the view share them. */
  key: string;
  /** The input voxels it needs around its output block on every side, in its input's voxels. */
  halo?: number[];
  /** The first stage only: the source level at this voxel size is read, while the view is
   * declared at another; the view's chunks must be whole voxels on that grid. */
  inputVoxelSize?: number[];
  /** Its output's dtype (numpy's name). */
  dtype: string;
  /** The leading axes its output has before the three spatial ones (a model's channels);
   * none by default. The last stage's are the view's channel axis. */
  added?: number[];
  /** The stage's output over `at`'s box from `input`, that box grown by the stage's halo (on
   * the input's grid, for the first stage): a Block, or its data alone (then of the box's
   * shape, the added axes first), whole or shaved by the halo as an op may return it. */
  produce(input: Block, at: StageAt): Promise<Block | Numbers>;
}

/** What a stage sees past the array's edge: the outermost voxels repeated, or zeros. */
export type Padding = "edge" | "zero";

export const prod = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

/** The bytes of `data` alone (a typed array may be a view on a larger buffer). */
export function bytes(data: Numbers): ArrayBuffer {
  const b = data.buffer as ArrayBuffer;
  return data.byteOffset === 0 && data.byteLength === b.byteLength ? b : b.slice(data.byteOffset, data.byteOffset + data.byteLength);
}

/** Output box [lo, hi) on the input's grid, `r` input voxels to each output voxel (fused.py's
 * `scale`), grown by `halo`: the scaled box must be whole voxels (fused.input_box). */
export function inputBox(lo: number[], hi: number[], r: number[], halo: number[]): { lo: number[]; hi: number[] } {
  const whole = (v: number, what: string) => {
    const w = Math.round(v);
    if (Math.abs(v - w) > 1e-6) throw new Error(`${what} ${v} is not whole voxels of the input: choose chunks the voxel ratio divides`);
    return w;
  };
  return { lo: lo.map((o, a) => whole(o * r[a], "an output chunk's start") - halo[a]), hi: hi.map((h, a) => whole(h * r[a], "an output chunk's end") + halo[a]) };
}

/** The level of `levels` at voxel size `voxel` (to a millionth, relatively), or -1. */
export function levelAt(levels: { voxel: number[] }[], voxel: number[]): number {
  return levels.findIndex((l) => l.voxel.length === voxel.length && l.voxel.every((v, a) => Math.abs(v - voxel[a]) <= 1e-6 * Math.max(Math.abs(v), Math.abs(voxel[a]))));
}

/** Box [lo, hi) of a level's voxels, `channels` of them one after another (as the reader
 * gives them), from `data`, what lies inside the array, read over [readLo, readHi) (the box
 * clipped to the level); past the array's edge the outermost voxels repeated (`edge`) or
 * zeros. A box wholly outside (no data) is zeros: nothing to repeat. The block is of
 * `dtype`'s typed array, the data as read; a 64-bit integer's voxels are bigints, repeated
 * and zeroed (0n) as any. */
export function padded(data: ArrayBuffer | null, dtype: string, channels: number, readLo: number[], readHi: number[], lo: number[], hi: number[], padding: Padding): Block {
  const Typed = TYPED[dtype] ?? Float32Array, d = hi.map((h, a) => h - lo[a]), rd = readHi.map((h, a) => h - readLo[a]);
  const shape = [...(channels > 1 ? [channels] : []), ...d], read = data ? new Typed(data) : null;
  if (read && same(rd, d)) return { data: read, shape };  // all inside: as read
  const out = new Typed(channels * prod(d)), o = out as unknown as Settable;  // of `read`'s kind: its elements set and filled as they are
  if (!read) return { data: out, shape };
  const n = prod(rd), x0 = readLo[2] - lo[2], x1 = readHi[2] - lo[2];  // the inside part of a row
  for (let c = 0; c < channels; c++) for (let z = 0; z < d[0]; z++) for (let y = 0; y < d[1]; y++) {
    const sz = z + lo[0] - readLo[0], sy = y + lo[1] - readLo[1];  // the row in what was read
    const inside = sz >= 0 && sz < rd[0] && sy >= 0 && sy < rd[1];
    if (!inside && padding === "zero") continue;  // a row past the edge: zeros
    const rz = Math.min(Math.max(sz, 0), rd[0] - 1), ry = Math.min(Math.max(sy, 0), rd[1] - 1);  // or the nearest row inside
    const from = c * n + (rz * rd[1] + ry) * rd[2], to = ((c * d[0] + z) * d[1] + y) * d[2];
    o.set(read.subarray(from, from + rd[2]), to + x0);
    if (padding === "edge") { o.fill(read[from], to, to + x0); o.fill(read[from + rd[2] - 1], to + x1, to + d[2]); }
  }
  return { data: out, shape };
}

/** A stage's result as the Block of its output box, of `shape` (the stage's added axes,
 * then the box): as it is if of that shape, or shaved by `halo` if of its input's (the
 * added axes and `inShape`'s last three; not for a first stage on another grid, which
 * returns its output's shape). `dtype` is what the stage declares. */
export function fitted(result: Block | Numbers, dtype: string, shape: number[], inShape: number[], halo: number[] | null): Block {
  const block: Block = "shape" in result ? result : { data: result, shape };
  const Typed = TYPED[dtype] ?? Float32Array;
  if (!(block.data instanceof Typed)) throw new Error(`a stage returned a ${block.data.constructor.name}, not the ${dtype} it declares`);
  if (block.data.length !== prod(block.shape)) throw new Error(`a stage returned ${block.data.length} values for a block of ${block.shape.join("×")}`);
  if (same(block.shape, shape)) return block;
  const whole = [...shape.slice(0, -3), ...inShape.slice(-3)];
  if (halo && same(block.shape, whole)) return crop(block, halo, shape);
  throw new Error(`a stage returned a block of ${block.shape.join("×")}, not its output's ${shape.join("×")}${halo ? ` or that grown by its halo, ${whole.join("×")}` : ""}`);
}

/** `block` without `halo` voxels on each side of its last three axes: the `shape` inside. */
export function crop(block: Block, halo: number[], shape: number[]): Block {
  const d = block.shape.slice(-3), o = shape.slice(-3), lead = prod(shape.slice(0, -3));
  const Typed = block.data.constructor as new (n: number) => Numbers, out = new Typed(prod(shape));
  for (let c = 0; c < lead; c++) for (let z = 0; z < o[0]; z++) for (let y = 0; y < o[1]; y++) {
    const from = ((c * d[0] + z + halo[0]) * d[1] + y + halo[1]) * d[2] + halo[2], to = ((c * o[0] + z) * o[1] + y) * o[2];
    (out as unknown as Settable).set(block.data.subarray(from, from + o[2]), to);
  }
  return { data: out, shape };
}
