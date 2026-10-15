// Neuroglancer precomputed volumes, the parts that need no browser: a chunk's compressed
// Morton code, its bytes found in a sharded scale (neuroglancer_uint64_sharded_v1: shard
// index, then minishard index, then the chunk's byte range), and compressed_segmentation
// decoded. sources.ts reads the info and the chunks with these. Here, in a module with no
// other imports, so Node's tests run it.

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

/** Chunks of one sharded scale (in the directory `dir`), by chunk id; each minishard index
 * read once, from its shard's index entry (16 bytes) and then its own bytes. */
export class Shards {
  private minishards = new Map<string, Promise<Map<bigint, [number, number]>>>();
  private dir: string;
  private s: Sharding;
  private get: Fetch;
  constructor(dir: string, s: Sharding, get: Fetch) {  // no parameter properties: Node strips types, not those
    if (s.hash !== "identity") throw new Error(`${dir}: sharding hash ${s.hash} is not read here (identity)`);
    this.dir = dir; this.s = s; this.get = get;
  }

  /** Chunk `id`'s bytes, decoded of the shard's data encoding; null if not stored. */
  async chunk(id: bigint): Promise<ArrayBuffer | null> {
    const h = id >> BigInt(this.s.preshift_bits);
    const mini = Number(h & ((1n << BigInt(this.s.minishard_bits)) - 1n));
    const shard = (h >> BigInt(this.s.minishard_bits)) & ((1n << BigInt(this.s.shard_bits)) - 1n);
    const file = `${this.dir}/${shard.toString(16).padStart(Math.ceil(this.s.shard_bits / 4), "0")}.shard`;
    const key = `${file}#${mini}`;
    let index = this.minishards.get(key);
    if (!index) {
      index = this.minishard(file, mini).catch((e) => { this.minishards.delete(key); throw e; });
      this.minishards.set(key, index);
    }
    const at = (await index).get(id);
    const data = at ? await this.get(file, at) : null;
    return data && this.s.data_encoding === "gzip" ? gunzip(data) : data;
  }

  /** Minishard `mini` of `file`: each chunk id's byte range in the shard. Its index is 3 × n
   * uint64s: the ids, each the one before plus its delta; the offsets, each from the end of
   * the chunk before (offsets count from the end of the shard index); the sizes. */
  private async minishard(file: string, mini: number): Promise<Map<bigint, [number, number]>> {
    const out = new Map<bigint, [number, number]>(), indexEnd = 16 * 2 ** this.s.minishard_bits;
    const entry = await this.get(file, [16 * mini, 16 * mini + 16]);
    if (!entry) return out;  // no shard
    const v = new DataView(entry), start = Number(v.getBigUint64(0, true)), end = Number(v.getBigUint64(8, true));
    if (end <= start) return out;
    let raw = await this.get(file, [indexEnd + start, indexEnd + end]);
    if (!raw) return out;
    if (this.s.minishard_index_encoding === "gzip") raw = await gunzip(raw);
    const w = new BigUint64Array(raw), n = w.length / 3;
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
