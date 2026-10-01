// Fields fitted where they are viewed: for a level below the solved ones, a control lattice
// `grid` voxels apart, in blocks of `block` voxels, each fitted from the solved field over
// the block plus a halo of context, coarse to fine within the block, when a chunk it covers
// is first asked for. The same as chunkmirage's register.py (refine=), so the two engines
// can be checked against each other block by block.
//
// The fits go through the page's queue of expensive work (`demand.ts`): the finest level
// and latest requests first, a few at once, and dropped if every request for one gives up.
import { halve } from "./affine";
import { Claim, queue } from "./demand";
import { fieldAt, nearestLevel, prod, RegionReader, type Image, type Numbers } from "./ome";
import { controlShape, sampled, solve, type Gpu, type Settings } from "./solver";
import type { Affine, ControlGrid, Lattice, Volume } from "./types";

const MIN_SIZE = 16;  // no coarser copy of a block with fewer voxels than this on an axis (register.MIN_SOLVE_SIZE)

/** The two images, the channels matched, and the intensity ranges the solve normalized with. */
export interface Images { fixed: Image; moving: Image; fixedChannel: number; movingChannel: number; ranges: { fixed: number[]; moving: number[] } }

const READER_BYTES = 192 * 2 ** 20;  // decoded pieces the page keeps per image for its fits

let turn: Promise<unknown> = Promise.resolve();  // the GPU fits one block at a time, across levels

const readers = new WeakMap<Image, RegionReader>();
const readerOf = (img: Image) => { let r = readers.get(img); if (!r) readers.set(img, (r = new RegionReader(img, READER_BYTES))); return r; };

/** Intensities to [0, 1] between lo and hi, clipped, as registration.normalize. */
export function normalize(d: Numbers, [lo, hi]: number[]): Float32Array {
  const o = new Float32Array(d.length), k = 1 / Math.max(hi - lo, 1e-12);
  for (let i = 0; i < d.length; i++) o[i] = Math.min(1, Math.max(0, (d[i] - lo) * k));
  return o;
}

export class Blocks {
  readonly lattice: Lattice;
  fitted = 0; seconds = 0;  // blocks fitted so far, and the GPU time they took
  private block: number[]; private context: number; private w2: number; private margin: number;
  private cache = new Map<string, Promise<Float32Array>>();
  private static count = 0;
  private id = ++Blocks.count;  // this set of blocks, in the queue's keys

  constructor(
    private g: Gpu, private img: Images, private affine: Affine, private parent: ControlGrid, level: number,
    box: number[][], block: number[], private settings: Settings, halo: number, private stages: number,
  ) {
    const fl = img.fixed.levels[level], [lo, hi] = box;
    this.lattice = { level, origin: lo, spacing: fl.voxel.map((v) => v * settings.grid), shape: controlShape(lo, hi, fl.voxel, settings.grid) };
    this.block = block.map((c) => Math.ceil(c / settings.grid));
    this.context = Math.ceil(halo / settings.grid);
    this.w2 = Math.floor(settings.window / 2);
    this.margin = halo + this.w2 + 1;  // moving voxels beyond the solved field's reach a block may move
  }

  /** Block `index`'s own lattice points [c0, c1) and those it is fitted over, [k0, k1). */
  private extent(index: number[]) {
    const n = this.lattice.shape, B = this.block;
    const c0 = index.map((v, a) => v * B[a]), c1 = c0.map((v, a) => Math.min(v + B[a], n[a]));
    const k0 = c0.map((v) => Math.max(v - this.context, 0));
    const k1 = c1.map((v, a) => Math.min(Math.max(Math.min(v + this.context, n[a]), k0[a] + 2), n[a]));  // two points per axis at least
    for (let a = 0; a < 3; a++) k0[a] = k1[a] - Math.max(k1[a] - k0[a], 2);
    return { c0, c1, k0, k1 };
  }

  /** The lattice's values over [lo, hi) (lattice indices): every block whose fit reaches it,
   * weighted 1 over its own points and less the further into its context (register.py's
   * _tent), so the field turns from one block's fit to the next's across their overlap. */
  async window(lo: number[], hi: number[], claim?: Claim): Promise<ControlGrid> {
    const { shape: n, origin, spacing } = this.lattice, B = this.block, ctx = this.context;
    const b0 = lo.map((v, a) => Math.max(Math.floor((v - ctx) / B[a]), 0));
    const b1 = hi.map((v, a) => Math.min(Math.floor((v - 1 + ctx) / B[a]), Math.ceil(n[a] / B[a]) - 1));
    const wanted: { index: number[]; c0: number[]; c1: number[]; k0: number[]; k1: number[] }[] = [];
    for (let z = b0[0]; z <= b1[0]; z++) for (let y = b0[1]; y <= b1[1]; y++) for (let x = b0[2]; x <= b1[2]; x++) {
      const e = this.extent([z, y, x]);
      if (e.k0.every((k, a) => k < hi[a]) && e.k1.every((k, a) => k > lo[a])) wanted.push({ index: [z, y, x], ...e });
    }
    const got = await Promise.all(wanted.map((b) => this.fit(b.index, claim)));
    const shape = hi.map((v, a) => v - lo[a]), acc = new Float64Array(3 * prod(shape)), total = new Float64Array(prod(shape));
    const tent = (k: number, c0: number, c1: number) => Math.max(0, 1 - Math.max(c0 - k, k - (c1 - 1), 0) / (ctx + 1));
    wanted.forEach(({ c0, c1, k0, k1 }, m) => {
      const values = got[m], ks = k1.map((v, a) => v - k0[a]);
      const from = k0.map((v, a) => Math.max(v, lo[a])), to = k1.map((v, a) => Math.min(v, hi[a]));
      for (let z = from[0]; z < to[0]; z++) {
        const wz = tent(z, c0[0], c1[0]);
        if (!wz) continue;
        for (let y = from[1]; y < to[1]; y++) {
          const wy = wz * tent(y, c0[1], c1[1]);
          if (!wy) continue;
          for (let x = from[2]; x < to[2]; x++) {
            const w = wy * tent(x, c0[2], c1[2]);
            if (!w) continue;
            const s = 3 * (((z - k0[0]) * ks[1] + (y - k0[1])) * ks[2] + (x - k0[2]));
            const o = ((z - lo[0]) * shape[1] + (y - lo[1])) * shape[2] + (x - lo[2]);
            acc[3 * o] += w * values[s]; acc[3 * o + 1] += w * values[s + 1]; acc[3 * o + 2] += w * values[s + 2];
            total[o] += w;
          }
        }
      }
    });
    const values = new Float32Array(acc.length);
    for (let o = 0; o < total.length; o++) for (let c = 0; c < 3; c++) values[3 * o + c] = acc[3 * o + c] / total[o];
    return { shape, origin: origin.map((o, a) => o + lo[a] * spacing[a]), spacing, values };
  }

  /** Block `index` fitted, for a request holding `claim` (released when that request is
   * answered or given up on: a fit waiting for no one is dropped). */
  private fit(index: number[], claim?: Claim): Promise<Float32Array> {
    const key = index.join(","), job = `${this.id}/${key}`;
    let p = this.cache.get(key);
    if (p) queue.claim(job);
    else {
      const forget = () => { if (this.cache.get(key) === p) this.cache.delete(key); };  // a failed or dropped fit is tried again when next asked for
      p = queue.submit(job, this.lattice.level, () => this.compute(index), forget).catch((e) => { forget(); throw e; });
      this.cache.set(key, p);
    }
    claim?.onCancel(() => queue.release(job));
    return p;
  }

  /** One block's values: fitted over the block plus its context against the voxels those
   * reach, coarse to fine over halved copies of them, from the solved field; the solved
   * field's own values where there is nothing to fit against (beyond an image, or too thin). */
  private async compute(index: number[]): Promise<Float32Array> {
    const { fixed, moving, fixedChannel, movingChannel, ranges } = this.img, { level, origin: lo, spacing } = this.lattice;
    const fl = fixed.levels[level], st = this.settings, { k0, k1 } = this.extent(index);
    const blo = k0.map((k, a) => lo[a] + k * spacing[a]), bhi = k1.map((k, a) => lo[a] + (k - 1) * spacing[a]);
    const shape = k1.map((v, a) => v - k0[a]);
    const fromParent = () => sampled(this.parent, shape, blo, bhi);
    if (st.iterations === 0) return fromParent();
    // the fixed voxels whose correlation windows the block's points reach
    const v0 = blo.map((b, a) => Math.max(Math.floor((b - fl.origin[a]) / fl.voxel[a]) - this.w2, 0));
    const v1 = bhi.map((b, a) => Math.min(Math.ceil((b - fl.origin[a]) / fl.voxel[a]) + this.w2 + 1, fl.shape[a]));
    if (v1.some((v, a) => v - v0[a] < st.window)) return fromParent();  // beyond the image, or too thin to correlate
    const f = await readerOf(fixed).read(level, fixedChannel, v0, v1);
    // the moving voxels those reach under the solved field and the affine, with room to move
    const j = nearestLevel(moving, fl.voxel), ml = moving.levels[j], d = [0, 0, 0];
    const m0 = [Infinity, Infinity, Infinity], m1 = [-Infinity, -Infinity, -Infinity];
    for (let z = 0; z < 9; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
      const q = [z, y, x].map((v, a) => f.origin[a] + (v / 8) * (f.shape[a] - 1) * f.voxel[a]);
      fieldAt(this.parent, q[0], q[1], q[2], d);
      const p = q.map((v, a) => v + d[a]);
      for (let a = 0; a < 3; a++) {
        const r = this.affine[a], v = (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3] - ml.origin[a]) / ml.voxel[a];
        m0[a] = Math.min(m0[a], v); m1[a] = Math.max(m1[a], v);
      }
    }
    const mlo = m0.map((v) => Math.max(Math.floor(v) - this.margin, 0)), mhi = m1.map((v, a) => Math.min(Math.ceil(v) + this.margin + 1, ml.shape[a]));
    if (mhi.some((v, a) => v <= mlo[a])) return fromParent();  // nothing of the moving image here
    const m = await readerOf(moving).read(j, movingChannel, mlo, mhi);
    // coarse to fine within the block, while a copy is worth it: halved as stored pyramids
    // are (the voxels' means), then normalized, as register.py does
    const fs: Volume[] = [{ norm: Float32Array.from(f.data), shape: f.shape, voxel: f.voxel, origin: f.origin }];
    const ms: Volume[] = [{ norm: Float32Array.from(m.data), shape: m.shape, voxel: m.voxel, origin: m.origin }];
    for (let s = 0; s < this.stages; s++) {
      if (Math.min(...fs[0].shape) < 2 * Math.max(MIN_SIZE, st.window)) break;
      fs.unshift(halve(fs[0])); ms.unshift(halve(ms[0]));
    }
    for (const v of fs) v.norm = normalize(v.norm, ranges.fixed);
    for (const v of ms) v.norm = normalize(v.norm, ranges.moving);
    const run = turn.then(() => solve(this.g, fs.map((v, k) => [v, ms[k]]), this.affine, [blo, bhi], st, () => {}, this.parent));
    turn = run.catch(() => {});
    const { grid, seconds } = await run;  // the fit's own time, not its wait for the GPU
    if (grid.shape.join() !== shape.join()) throw new Error(`block ${index}: fitted ${grid.shape} control points, expected ${shape}`);
    this.fitted++; this.seconds += seconds;
    return grid.values;  // context included: neighbouring blocks blend across it
  }
}
