// Neuroglancer precomputed, the parts Node can run (src/precomputed.ts): a chunk's compressed
// Morton code and its inverse, a chunk found in a sharded scale through its shard index and
// gzipped minishard index (each read once), chunk and minishard sizes from those indexes
// (chunksizes://), a view served as a precomputed volume (stored levels passed through,
// computed ones placed on their own voxels), and compressed_segmentation decoded, uint32 and uint64,
// with a block cut by the chunk's edge. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import { chunkIndex, compressedMorton, compressedSegmentation, cropChunk, dataBeforeIndexes, minishardBytes, mortonBlock, mortonDecoder, precomputedInfo, Shards, storedDir, storedKey, type Fetch, type PrecomputedScale } from "../src/precomputed.ts";
import type { ViewInfo } from "../src/types.ts";

test("a chunk's compressed Morton code interleaves only the bits each axis's grid needs", () => {
  // a grid 4 × 2 × 1: x takes two bits, y one, z none
  assert.equal(compressedMorton([3, 1, 0], [4, 2, 1]), 0b111n);
  assert.equal(compressedMorton([2, 0, 0], [4, 2, 1]), 0b100n);
  assert.equal(compressedMorton([0, 1, 0], [4, 2, 1]), 0b010n);
  assert.equal(compressedMorton([0, 0, 0], [1, 1, 1]), 0n);
});

test("a compressed Morton code decodes back to its grid position; a minishard's chunks are a block", () => {
  const grid = [12, 8, 4], decode = mortonDecoder(grid);  // x four bits, y three, z two
  for (const pos of [[0, 0, 0], [11, 7, 3], [5, 2, 1], [8, 0, 3]]) assert.deepEqual(decode(compressedMorton(pos, grid)), pos);
  assert.deepEqual(mortonBlock(grid, 3), [2, 2, 2]);
  assert.deepEqual(mortonBlock(grid, 6), [4, 4, 4]);
  assert.deepEqual(mortonBlock(grid, 8), [8, 8, 4]);  // z's bits run out first
});

const u64 = (...v: number[]) => Buffer.from(new BigUint64Array(v.map(BigInt)).buffer);

test("a sharded chunk is found through its shard's index and minishard index", async () => {
  // minishard_bits 1, shard_bits 1: ids 2 and 6 are both in shard 1, minishard 0
  const chunks = Buffer.from([1, 2, 3, 0, 4, 5]);  // 2 at [0, 3), a byte unused, 6 at [4, 6)
  const minishard = gzipSync(u64(2, 4, 0, 1, 3, 2));  // ids by delta, offsets from the end of the one before, sizes
  const shard = Buffer.concat([u64(6, 6 + minishard.length, 0, 0), chunks, minishard]);
  const files: Record<string, Buffer> = { "s0/1.shard": shard }, asked: string[] = [];
  const get: Fetch = async (url, range) => {
    asked.push(`${url} ${range}`);
    const f = files[url];
    if (!f) return null;
    const b = range ? f.subarray(...range) : f;
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  };
  const shards = new Shards("s0", { "@type": "neuroglancer_uint64_sharded_v1", hash: "identity", preshift_bits: 0, minishard_bits: 1, shard_bits: 1, minishard_index_encoding: "gzip" }, get);
  assert.deepEqual([...new Uint8Array((await shards.chunk(2n))!)], [1, 2, 3]);
  assert.deepEqual([...new Uint8Array((await shards.chunk(6n))!)], [4, 5]);
  assert.equal(await shards.chunk(10n), null);  // shard 1, minishard 0, not in it
  assert.equal(await shards.chunk(4n), null);   // shard 0: no file
  assert.equal(asked.filter((a) => a.startsWith("s0/1.shard")).length, 4);  // shard index and minishard once, then the chunks
  assert.deepEqual(await shards.sizes(1n, 0), { ids: [2n, 6n], sizes: [3, 2] });
  assert.deepEqual([...minishardBytes((await shards.index(1n))!)], [4 + 2, 0]);  // the gap before minishard 0's index: its chunks and the byte unused
  // each minishard's chunks before its index: the gaps are its bytes; indexes back to back: not so
  assert.equal(dataBeforeIndexes(Float64Array.from([10, 20, 30, 40, 0, 0])), true);
  assert.equal(dataBeforeIndexes(Float64Array.from([100, 110, 110, 120])), false);
  assert.equal(dataBeforeIndexes(Float64Array.from([10, 20, 0, 0])), null);  // one minishard stored: cannot tell
  assert.throws(() => new Shards("s0", { "@type": "neuroglancer_uint64_sharded_v1", hash: "murmurhash3_x86_128", preshift_bits: 0, minishard_bits: 0, shard_bits: 0 }, get), /not read here/);
});

test("compressed_segmentation decodes each block's packed values through its table", () => {
  // 3 × 2 × 1 voxels in blocks of 2 × 2 × 1: block 0 two labels at one bit each, block 1
  // (cut to one voxel across by the edge) one label at no bits
  const words = new Uint32Array([1, 5 | (1 << 24), 4, 7, 8, 0b0110, 7, 9, 5]);
  const labels = compressedSegmentation(words.buffer, [3, 2, 1], 1, [2, 2, 1], false);
  assert.ok(labels instanceof Uint32Array);
  assert.deepEqual([...labels], [7, 9, 5, 9, 7, 5]);
  // uint64: two words per table entry, low first
  const big = new Uint32Array([1, 3 | (1 << 24), 2, 0b01, 3, 0, 1, 1]);
  assert.deepEqual([...compressedSegmentation(big.buffer, [2, 1, 1], 1, [2, 1, 1], true)], [2n ** 32n + 1n, 3n]);
});

test("a view as precomputed: stored levels as stored, computed ones on their own voxels", () => {
  // hemibrain's s0, then one voxel per 64³ chunk and one per 8³ chunks (chunksizes://)
  const s0: PrecomputedScale = { key: "8.0x8.0x8.0", size: [34427, 39725, 41394], resolution: [8, 8, 8], voxel_offset: [0, 0, 0], chunk_sizes: [[64, 64, 64]], encoding: "jpeg",
    sharding: { "@type": "neuroglancer_uint64_sharded_v1", hash: "identity", preshift_bits: 9, minishard_bits: 6, shard_bits: 15, minishard_index_encoding: "gzip" } };
  const nm = { name: "", unit: "nanometer" }, v: ViewInfo = {
    dtype: "uint8", out: "uint8", channels: 1, axes: ["z", "y", "x"].map((name) => ({ ...nm, name })), halo: [0, 0, 0], lead: 0,
    levels: [
      { shape: [41394, 39725, 34427], voxel: [8, 8, 8], origin: [4, 4, 4], stored: { dir: "gs://b/em/8.0x8.0x8.0", scale: s0 } },
      { shape: [647, 621, 538], voxel: [512, 512, 512], origin: [256, 256, 256] },
      { shape: [81, 78, 68], voxel: [4096, 4096, 4096], origin: [2048, 2048, 2048] },
    ],
  };
  const info = precomputedInfo(v, [1, 128, 128]);
  assert.equal(info.data_type, "uint8"); assert.equal(info.num_channels, 1); assert.equal(info.type, "image");
  assert.deepEqual(info.scales[0], { ...s0, key: storedKey("gs://b/em/8.0x8.0x8.0") });  // as stored, its files fetched where they are
  assert.match(info.scales[0].key, /^stored\/[A-Za-z0-9_-]+$/);
  assert.equal(storedDir(info.scales[0].key.slice("stored/".length)), "gs://b/em/8.0x8.0x8.0");
  assert.deepEqual(info.scales[1], { key: "computed/1", size: [538, 621, 647], resolution: [512, 512, 512], voxel_offset: [0, 0, 0], chunk_sizes: [[128, 128, 1]], encoding: "raw" });
  assert.deepEqual(info.scales[2], { key: "computed/2", size: [68, 78, 81], resolution: [4096, 4096, 4096], voxel_offset: [0, 0, 0], chunk_sizes: [[68, 78, 1]], encoding: "raw" });
  // every level's voxel 0 starts at 0 nm, as s0's does
  for (const sc of info.scales) assert.deepEqual(sc.voxel_offset!.map((o, a) => o * sc.resolution[a]), [0, 0, 0]);
  assert.deepEqual(chunkIndex("512-538_128-256_5-6", info.scales[1]), [5, 1, 4]);  // z, y, x; cut at x's edge
  assert.equal(chunkIndex("512-640_128-256_5-6", info.scales[1]), null);         // past the edge
  assert.deepEqual(chunkIndex("0-68_0-78_80-81", info.scales[2]), [80, 0, 0]);
  const shifted = { ...v, levels: [v.levels[0], { ...v.levels[1], origin: [300, 256, 256] }] };
  assert.throws(() => precomputedInfo(shifted, [1, 128, 128]), /whole voxels/);
});

test("a chunk computed whole is cut to the volume's edge, as a precomputed raw chunk is", () => {
  const whole = Uint8Array.from({ length: 2 * 3 * 4 }, (_, i) => i);  // chunk 2 × 3 × 4 (z, y, x)
  assert.deepEqual([...new Uint8Array(cropChunk(whole.buffer, 1, 1, [2, 3, 4], [1, 2, 3]))], [0, 1, 2, 4, 5, 6]);
  assert.equal(cropChunk(whole.buffer, 1, 1, [2, 3, 4], [2, 3, 4]), whole.buffer);
});
