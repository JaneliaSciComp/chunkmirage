// chunksizes://<sharded precomputed volume>?range=lo,hi, as chunkmirage_contrib.sources.chunksizes
// opens it (its docstring says more): the volume's finest level as stored, then one voxel per
// chunk of it (the chunk's compressed size, from its minishard's index) and, where that is
// exact, one per minishard (the bytes before its index, from the shard's own index: a
// minishard one block of chunks, laid out before its index), each the compressed size as a
// fraction of the raw size of the chunks it covers, lo..hi mapped onto the volume's dtype,
// black to white. Nothing is decoded: a thumbnail of the data from its indexes.
import { prod, TYPED, type Reals } from "../ome";
import { compressedMorton, dataBeforeIndexes, minishardBytes, mortonBlock, mortonDecoder, Shards, type Fetch, type PrecomputedInfo } from "../precomputed";
import type { Source, SourceLevel } from "../sources";

const TOP: Record<string, number> = { uint8: 255, uint16: 65535, uint32: 4294967295, int8: 127, int16: 32767, int32: 2147483647 };
const PIECES_KEPT = 4096;  // pieces (a minishard's chunks, a shard's minishards) kept, a few MB
const READS_OUT = 64;      // index reads in flight at once: a level's pieces queue for them

/** `get`, at most `n` reads at once, the rest waiting their turn: a slice's thousands of
 * index reads issued at once would have the browser refuse some. */
function limited(get: Fetch, n: number): Fetch {
  let out = 0;
  const waiting: (() => void)[] = [];
  return async (url, range) => {
    while (out >= n) await new Promise<void>((go) => waiting.push(go));
    out++;
    try { return await get(url, range); } finally { out--; waiting.shift()?.(); }
  };
}

export async function chunkSizesSource(url: string, open: (url: string) => Promise<Source>, get: Fetch): Promise<Source> {
  const rest = url.slice("chunksizes://".length), q = rest.lastIndexOf("?");
  const location = q < 0 ? rest : rest.slice(0, q), params = new URLSearchParams(q < 0 ? "" : rest.slice(q + 1));
  const extra = [...params.keys()].filter((k) => k !== "range");
  if (extra.length) throw new Error(`chunksizes:// takes only range=lo,hi, not ${extra}`);
  const [lo, hi] = (params.get("range") ?? "0,1").split(",").map(Number);
  if (!(hi > lo)) throw new Error(`chunksizes:// range=${params.get("range")}: the second must be larger`);
  const data = await open(location);
  const base = location.replace(/^precomputed:\/\//, "").replace(/\/+$/, "");
  const bytes = await get(`${base}/info`), s = bytes && (JSON.parse(new TextDecoder().decode(bytes)) as PrecomputedInfo).scales?.[0];
  if (!s?.sharding) throw new Error(`chunksizes:// reads a sharded precomputed volume's indexes; ${location} is not one`);
  if (data.channels !== 1) throw new Error(`chunksizes:// ${location}: one channel, not ${data.channels}`);
  if (!TYPED[data.dtype] || data.dtype.endsWith("64") && !data.dtype.startsWith("float")) throw new Error(`chunksizes:// ${location}: ${data.dtype} values`);
  const T = TYPED[data.dtype] as unknown as { new (n: number): Reals; BYTES_PER_ELEMENT: number }, top = TOP[data.dtype] ?? 1, l0 = data.levels[0];
  const chunk = s.chunk_sizes[0], grid = s.size.map((v, a) => Math.ceil(v / chunk[a]));  // x, y, z
  const shards = new Shards(`${base}/${s.key.replace(/^\/+|\/+$/g, "")}`, s.sharding, limited(get, READS_OUT)), decode = mortonDecoder(grid);
  const mini = mortonBlock(grid, s.sharding.preshift_bits);  // chunks a minishard
  const shard = mortonBlock(grid, s.sharding.preshift_bits + s.sharding.minishard_bits).map((v, a) => v / mini[a]);  // minishards a shard
  const zyx = (v: number[]) => [...v].reverse();
  /** One voxel per `per` chunks (x, y, z), read in pieces of `piece` voxels (z, y, x). */
  const level = (per: number[], piece: number[]): SourceLevel => {
    const voxel = l0.voxel.map((v, a) => v * zyx(chunk)[a] * zyx(per)[a]);
    return { shape: zyx(grid.map((g, a) => Math.ceil(g / per[a]))), voxel, origin: l0.origin.map((o, a) => o - l0.voxel[a] / 2 + voxel[a] / 2), chunks: piece };
  };
  const levels = [l0, level([1, 1, 1], zyx(mini))];
  // the minishard level where a minishard is one block of chunks (its low bits vary, and no
  // bits lie past the shard's) and a shard here lays its minishards out before their indexes
  const sh = s.sharding, bits = grid.reduce((n, g) => n + Math.ceil(Math.log2(g)), 0);
  if (sh.preshift_bits > 0 && bits <= sh.preshift_bits + sh.minishard_bits + sh.shard_bits) {
    const middle = grid.map((g, a) => Math.floor(g / 2 / mini[a]) * mini[a]);
    const index = await shards.index(shards.where(compressedMorton(middle, grid)).shard);
    if (index && dataBeforeIndexes(index)) levels.push(level(mini, zyx(shard)));
  }
  /** The value of `nbytes` compressed from the chunks at grid position `pos` (x, y, z) on,
   * `per` of them along each axis: a fraction of their raw bytes, lo..hi as black to white. */
  const value = (nbytes: number, pos: number[], per: number[]) => {
    const raw = prod(pos.map((p, a) => Math.min(Math.max(s.size[a] - p * chunk[a], 0), per[a] * chunk[a]))) * T.BYTES_PER_ELEMENT;
    const f = Math.min(Math.max((nbytes / Math.max(raw, 1) - lo) / (hi - lo), 0), 1) * top;
    return top === 1 ? f : Math.round(f);
  };
  /** Piece `b` (z, y, x) of level `li`: a minishard's chunks (level 1), a shard's minishards (2). */
  const piece = async (li: number, b: number[]): Promise<{ data: Reals; shape: number[] }> => {
    const l = levels[li], start = b.map((v, a) => v * l.chunks[a]);
    const shape = start.map((v, a) => Math.min(l.chunks[a], l.shape[a] - v)), out = new T(prod(shape));
    if (li === 1) {
      const first = zyx(start), { shard: sh, mini: m } = shards.where(compressedMorton(first, grid));
      const { ids, sizes } = await shards.sizes(sh, m);
      ids.forEach((id, i) => {
        const p = decode(id), at = [2, 1, 0].map((a, k) => p[a] - start[k]);
        if (at.every((v, k) => v >= 0 && v < shape[k])) out[(at[0] * shape[1] + at[1]) * shape[2] + at[2]] = value(sizes[i], p, [1, 1, 1]);
      });
    } else {
      const first = zyx(start).map((v, a) => v * mini[a]), { shard: sh } = shards.where(compressedMorton(first, grid));
      const index = await shards.index(sh);
      if (!index) return { data: out, shape };
      const gaps = minishardBytes(index);
      for (let z = 0; z < shape[0]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[2]; x++) {
        const pos = [start[2] + x, start[1] + y, start[0] + z].map((v, a) => v * mini[a]);
        out[(z * shape[1] + y) * shape[2] + x] = value(gaps[shards.where(compressedMorton(pos, grid)).mini], pos, mini);
      }
    }
    return { data: out, shape };
  };
  const kept = new Map<string, Promise<{ data: Reals; shape: number[] }>>();
  const cached = (li: number, b: number[]) => {
    const key = `${li}/${b}`;
    let got = kept.get(key);
    if (got) kept.delete(key);  // most recent last
    else got = piece(li, b).catch((e) => { kept.delete(key); throw e; });
    kept.set(key, got);
    if (kept.size > PIECES_KEPT) kept.delete(kept.keys().next().value!);
    return got;
  };
  return {
    url, dtype: data.dtype, axes: data.axes, channels: 1, levels,
    read: async (li, c, rlo, rhi, at) => {
      if (li === 0) return data.read(0, c, rlo, rhi, at);
      const l = levels[li], shape = rhi.map((v, a) => v - rlo[a]), out = new T(prod(shape));
      const b0 = rlo.map((v, a) => Math.floor(v / l.chunks[a])), b1 = rhi.map((v, a) => Math.floor((v - 1) / l.chunks[a]));
      const wanted: number[][] = [];
      for (let z = b0[0]; z <= b1[0]; z++) for (let y = b0[1]; y <= b1[1]; y++) for (let x = b0[2]; x <= b1[2]; x++) wanted.push([z, y, x]);
      const got = await Promise.all(wanted.map((b) => cached(li, b)));
      wanted.forEach((b, n) => {
        const { data: d, shape: bs } = got[n], start = b.map((v, a) => v * l.chunks[a]);
        const from = start.map((v, a) => Math.max(v, rlo[a])), to = start.map((v, a) => Math.min(v + bs[a], rhi[a]));
        for (let z = from[0]; z < to[0]; z++) for (let y = from[1]; y < to[1]; y++) {
          const src = ((z - start[0]) * bs[1] + (y - start[1])) * bs[2] - start[2], dst = ((z - rlo[0]) * shape[1] + (y - rlo[1])) * shape[2] - rlo[2];
          for (let x = from[2]; x < to[2]; x++) out[dst + x] = d[src + x];
        }
      });
      return out;
    },
  };
}
