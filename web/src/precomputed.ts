// Neuroglancer precomputed volumes, the parts that need no browser: a chunk's compressed
// Morton code, its bytes found in a sharded scale (neuroglancer_uint64_sharded_v1: shard
// index, then minishard index, then the chunk's byte range), and compressed_segmentation
// decoded. sources.ts reads the info and the chunks with these; the engine serves a view as
// a precomputed volume with precomputedInfo. Here, in a module with no other imports (types
// aside), so Node's tests run it.
import type { ViewInfo } from "./types";

export interface Sharding {
  "@type": string;
  preshift_bits: number;
  hash: string;
  minishard_bits: number;
  shard_bits: number;
  minishard_index_encoding?: string;
  data_encoding?: string;
}

export interface PrecomputedScale {
  key: string;
  size: number[];        // x, y, z, as are the rest
  resolution: number[];
  voxel_offset?: number[];
  chunk_sizes: number[][];
  encoding: string;
  compressed_segmentation_block_size?: number[];
  sharding?: Sharding;
}

export interface PrecomputedInfo {
  "@type"?: string;
  data_type: string;
  num_channels?: number;
  scales: PrecomputedScale[];
}

/** Bytes [start, end) of `url`, or all of it; null where there are none (404). */
export type Fetch = (url: string, range?: [number, number]) => Promise<ArrayBuffer | null>;

export async function gunzip(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

/** The chunk at grid position `pos` (x, y, z) of a grid `grid` chunks across: the bits of
 * its coordinates interleaved, x first, each axis only for as many bits as its grid needs. */
export function compressedMorton(pos: number[], grid: number[]): bigint {
  const bits = grid.map((g) => Math.ceil(Math.log2(g)));
  let code = 0n, j = 0n;
  for (let i = 0; i < Math.max(...bits); i++) {
    for (let a = 0; a < 3; a++) if (i < bits[a]) { code |= BigInt((pos[a] >> i) & 1) << j; j++; }
  }
  return code;
}

/** An unsharded chunk's file name: its voxels [lo, hi) (x, y, z, the volume's offset in). */
export const chunkName = (lo: number[], hi: number[]) => lo.map((v, a) => `${v}-${hi[a]}`).join("_");

/** For a grid `grid` chunks across (x, y, z), the axis each bit of a compressed Morton code
 * comes from, and its place among that axis's bits. */
function mortonBits(grid: number[]): { axis: number[]; rank: number[] } {
  const bits = grid.map((g) => Math.ceil(Math.log2(g))), axis: number[] = [], rank: number[] = [], seen = [0, 0, 0];
  for (let i = 0; i < Math.max(...bits); i++) for (let a = 0; a < 3; a++) if (i < bits[a]) { axis.push(a); rank.push(seen[a]++); }
  return { axis, rank };
}

/** The grid position (x, y, z) of each chunk id of a grid `grid` chunks across. */
export function mortonDecoder(grid: number[]): (id: bigint) => number[] {
  const { axis, rank } = mortonBits(grid);
  return (id) => {
    const lo = Number(id & 0xffffffffn), hi = Number(id >> 32n), pos = [0, 0, 0];
    for (let j = 0; j < axis.length; j++) pos[axis[j]] += (((j < 32 ? lo >>> j : hi >>> (j - 32)) & 1) * 2 ** rank[j]);
    return pos;
  };
}

/** The extent (x, y, z, in chunks) of the chunks whose ids agree but for their lowest `bits`
 * bits: a minishard's chunks for a scale's preshift_bits (8³ for hemibrain's 8 nm EM). */
export function mortonBlock(grid: number[], bits: number): number[] {
  const { axis } = mortonBits(grid);
  return [0, 1, 2].map((a) => 2 ** axis.slice(0, bits).filter((b) => b === a).length);
}

/** Chunks of one sharded scale (in the directory `dir`), by chunk id; each shard's index read
 * once (16 bytes a minishard), and each minishard index once, when a chunk of it is asked
 * for. */
export class Shards {
  private indexes = new Map<bigint, Promise<Float64Array | null>>();
  private minishards = new Map<string, Promise<Map<bigint, [number, number]>>>();
  private dir: string;
  private s: Sharding;
  private get: Fetch;
  constructor(dir: string, s: Sharding, get: Fetch) {  // no parameter properties: Node strips types, not those
    if (s.hash !== "identity") throw new Error(`${dir}: sharding hash ${s.hash} is not read here (identity)`);
    this.dir = dir; this.s = s; this.get = get;
  }

  /** The shard and minishard chunk `id` is in. */
  where(id: bigint): { shard: bigint; mini: number } {
    const h = id >> BigInt(this.s.preshift_bits);
    return { shard: (h >> BigInt(this.s.minishard_bits)) & ((1n << BigInt(this.s.shard_bits)) - 1n), mini: Number(h & ((1n << BigInt(this.s.minishard_bits)) - 1n)) };
  }

  private file(shard: bigint): string {
    return `${this.dir}/${shard.toString(16).padStart(Math.ceil(this.s.shard_bits / 4), "0")}.shard`;
  }

  /** Shard `shard`'s index: each minishard's index's byte range in the shard, start then end
   * (counting from the end of this index); null if the shard is not stored. */
  index(shard: bigint): Promise<Float64Array | null> {
    let got = this.indexes.get(shard);
    if (!got) {
      got = this.get(this.file(shard), [0, 16 * 2 ** this.s.minishard_bits]).then((b) => b && Float64Array.from(new BigUint64Array(b), Number));
      got.catch(() => this.indexes.delete(shard));
      this.indexes.set(shard, got);
    }
    return got;
  }

  /** Chunk `id`'s bytes, decoded of the shard's data encoding; null if not stored. */
  async chunk(id: bigint): Promise<ArrayBuffer | null> {
    const { shard, mini } = this.where(id), at = (await this.minishard(shard, mini)).get(id);
    const data = at ? await this.get(this.file(shard), at) : null;
    return data && this.s.data_encoding === "gzip" ? gunzip(data) : data;
  }

  /** The chunks of minishard `mini` of `shard` and each one's compressed size (none kept). */
  async sizes(shard: bigint, mini: number): Promise<{ ids: bigint[]; sizes: number[] }> {
    const w = await this.words(shard, mini), n = w ? w.length / 3 : 0, ids: bigint[] = [], sizes: number[] = [];
    let id = 0n;
    for (let i = 0; i < n; i++) { id += w![i]; ids.push(id); sizes.push(Number(w![2 * n + i])); }
    return { ids, sizes };
  }

  /** Minishard `mini` of `shard`: each chunk id's byte range in the shard, read once. Its index
   * is 3 × n uint64s: the ids, each the one before plus its delta; the offsets, each from the
   * end of the chunk before (offsets count from the end of the shard index); the sizes. */
  private minishard(shard: bigint, mini: number): Promise<Map<bigint, [number, number]>> {
    const key = `${shard}#${mini}`;
    let got = this.minishards.get(key);
    if (!got) {
      got = this.readMinishard(shard, mini);
      got.catch(() => this.minishards.delete(key));
      this.minishards.set(key, got);
    }
    return got;
  }

  /** Minishard `mini` of `shard`'s index, decoded; null if there is none. */
  private async words(shard: bigint, mini: number): Promise<BigUint64Array | null> {
    const index = await this.index(shard), indexEnd = 16 * 2 ** this.s.minishard_bits;
    if (!index || index[2 * mini + 1] <= index[2 * mini]) return null;  // no shard, or an empty minishard
    let raw = await this.get(this.file(shard), [indexEnd + index[2 * mini], indexEnd + index[2 * mini + 1]]);
    if (raw && this.s.minishard_index_encoding === "gzip") raw = await gunzip(raw);
    return raw && new BigUint64Array(raw);
  }

  private async readMinishard(shard: bigint, mini: number): Promise<Map<bigint, [number, number]>> {
    const out = new Map<bigint, [number, number]>(), indexEnd = 16 * 2 ** this.s.minishard_bits;
    const w = await this.words(shard, mini);
    if (!w) return out;
    const n = w.length / 3;
    let id = 0n, next = 0;
    for (let i = 0; i < n; i++) {
      id += w[i];
      const from = next + Number(w[n + i]), size = Number(w[2 * n + i]);
      out.set(id, [indexEnd + from, indexEnd + from + size]);
      next = from + size;
    }
    return out;
  }
}

/** Whether each stored minishard of a shard (its index, as Shards.index reads it) lies just
 * before its index, a gap of data before every index, as minishardBytes counts on; null if
 * the shard cannot tell (fewer than two minishards stored). */
export function dataBeforeIndexes(index: Float64Array): boolean | null {
  const n = index.length / 2;
  const stored = Array.from({ length: n }, (_, m) => m).filter((m) => index[2 * m + 1] > index[2 * m])
    .sort((a, b) => index[2 * a] - index[2 * b] || index[2 * a + 1] - index[2 * b + 1]);
  if (stored.length < 2) return null;
  let end = 0;
  for (const m of stored) {
    if (index[2 * m] <= end) return false;  // indexes back to back: the data lies elsewhere
    end = index[2 * m + 1];
  }
  return true;
}

/** Each minishard's bytes from its shard's index alone: the gap before its index, in file
 * order, where tensorstore and Neuroglancer's writers put a minishard's chunks (then its
 * index); dataBeforeIndexes checks a shard is laid out so. */
export function minishardBytes(index: Float64Array): Float64Array {
  const n = index.length / 2, out = new Float64Array(n);
  const order = Array.from({ length: n }, (_, m) => m).sort((a, b) => index[2 * a] - index[2 * b] || index[2 * a + 1] - index[2 * b + 1]);
  let end = 0;
  for (const m of order) {
    const start = index[2 * m], stop = index[2 * m + 1];
    if (stop > start) { out[m] = Math.max(0, start - end); end = stop; }
  }
  return out;
}

/** The key of a stored scale passed through, `stored/<its directory, base64url>`: the
 * service worker reads the directory back from a file's path (storedDir) and fetches the file
 * from there itself, the range asked for, as the viewer would the volume (sw.ts). */
export function storedKey(dir: string): string {
  const b = String.fromCharCode(...new TextEncoder().encode(dir));
  return `stored/${btoa(b).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")}`;
}
export function storedDir(code: string): string {
  const b = atob(code.replace(/-/g, "+").replace(/_/g, "/"));
  return new TextDecoder().decode(Uint8Array.from(b, (c) => c.charCodeAt(0)));
}

/** A view as a precomputed volume's info, its axes in nanometres. A level stored as a
 * precomputed scale keeps that scale as it is (resolution, size, voxel_offset, chunks,
 * encoding, sharding), its key its directory's (storedKey), so Neuroglancer reads its files
 * untouched, fetched where they are;
 * the others are raw scales of `chunk` (z, y, x) chunks, unsharded, key `computed/<level>`.
 * A computed level's voxel 0 must start on a whole voxel of its own (voxel_offset is
 * integral), as a level of one voxel per stored chunk does. */
export function precomputedInfo(v: ViewInfo, chunk: number[]): PrecomputedInfo & { type: string } {
  if (v.axes.some((a) => a.unit !== "nanometer")) throw new Error(`a precomputed volume is in nanometres, not ${v.axes.map((a) => a.unit)}`);
  const scales = v.levels.map((l, i): PrecomputedScale => {
    if (l.stored) return { ...l.stored.scale, key: storedKey(l.stored.dir) };
    const resolution = [...l.voxel].reverse(), size = [...l.shape].reverse();
    const offset = [...l.origin].reverse().map((o, a) => o / resolution[a] - 0.5);  // voxel 0's corner, in its own voxels
    if (offset.some((o) => Math.abs(o - Math.round(o)) > 1e-6)) throw new Error(`level ${i} starts ${offset} voxels from 0: not whole voxels, as precomputed's voxel_offset needs`);
    return { key: `computed/${i}`, size, resolution, voxel_offset: offset.map(Math.round), chunk_sizes: [[...chunk].reverse().map((c, a) => Math.min(c, size[a]))], encoding: "raw" };
  });
  return { "@type": "neuroglancer_multiscale_volume", type: v.kind === "label" ? "segmentation" : "image", data_type: v.out, num_channels: v.channels, scales };
}

/** Chunk `name` (x0-x1_y0-y1_z0-z1, as Neuroglancer asks for an unsharded chunk) of a computed
 * scale: its index on the level's grid (z, y, x), or null if it is not one of its chunks. */
export function chunkIndex(name: string, scale: PrecomputedScale): number[] | null {
  const m = /^(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$/.exec(name);
  if (!m) return null;
  const c = scale.chunk_sizes[0], off = scale.voxel_offset ?? [0, 0, 0], v = m.slice(1).map(Number);
  const index = [0, 1, 2].map((a) => (v[2 * a] - off[a]) / c[a]);
  const ok = index.every((i, a) => Number.isInteger(i) && i >= 0 && v[2 * a + 1] === off[a] + Math.min((i + 1) * c[a], scale.size[a]));
  return ok ? index.reverse() : null;
}

/** A chunk computed whole (`whole`: channels × `chunk` voxels, z, y, x, of `width` bytes each)
 * cut to the `size` (z, y, x) of the volume it covers, as a precomputed raw chunk at the
 * volume's edge is. */
export function cropChunk(whole: ArrayBuffer, width: number, channels: number, chunk: number[], size: number[]): ArrayBuffer {
  if (size.every((d, a) => d === chunk[a])) return whole;
  const from = new Uint8Array(whole), row = size[2] * width, out = new Uint8Array(channels * size[0] * size[1] * row);
  for (let c = 0; c < channels; c++) for (let z = 0; z < size[0]; z++) for (let y = 0; y < size[1]; y++) {
    const at = (((c * chunk[0] + z) * chunk[1] + y) * chunk[2]) * width;
    out.set(from.subarray(at, at + row), ((c * size[0] + z) * size[1] + y) * row);
  }
  return out.buffer;
}

/** A compressed_segmentation chunk of `size` voxels (x, y, z) and `channels`, in blocks of
 * `block` (x, y, z), as uint32 or uint64 labels in C order (c, z, y, x). Per channel: an
 * offset to it (in 32-bit words); there, two words per block, the block's lookup table's
 * offset (24 bits) and bits per voxel (8), then its packed values' offset; each voxel's
 * value indexes the table. */
export function compressedSegmentation(bytes: ArrayBuffer, size: number[], channels: number, block: number[], uint64: boolean): Uint32Array | BigUint64Array {
  const w = new Uint32Array(bytes), per = uint64 ? 2 : 1, n = size[0] * size[1] * size[2];
  const out = new Uint32Array(n * channels * per);
  const grid = size.map((s, a) => Math.ceil(s / block[a]));
  for (let c = 0; c < channels; c++) {
    const base = w[c];
    for (let gz = 0; gz < grid[2]; gz++) for (let gy = 0; gy < grid[1]; gy++) for (let gx = 0; gx < grid[0]; gx++) {
      const h = base + 2 * (gx + grid[0] * (gy + grid[1] * gz));
      const table = base + (w[h] & 0xffffff), bits = w[h] >>> 24, values = base + w[h + 1];
      const mask = bits === 32 ? 0xffffffff : (1 << bits) - 1;
      const x0 = gx * block[0], y0 = gy * block[1], z0 = gz * block[2];
      for (let z = z0; z < Math.min(z0 + block[2], size[2]); z++) for (let y = y0; y < Math.min(y0 + block[1], size[1]); y++) {
        for (let x = x0; x < Math.min(x0 + block[0], size[0]); x++) {
          const bit = (((z - z0) * block[1] + (y - y0)) * block[0] + (x - x0)) * bits;
          const value = bits ? ((w[values + (bit >>> 5)] >>> (bit & 31)) & mask) >>> 0 : 0;
          const o = (c * n + (z * size[1] + y) * size[0] + x) * per, t = table + value * per;
          out[o] = w[t];
          if (uint64) out[o + 1] = w[t + 1];
        }
      }
    }
  }
  return uint64 ? new BigUint64Array(out.buffer) : out;
}
