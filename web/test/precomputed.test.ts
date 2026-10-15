// Neuroglancer precomputed, the parts Node can run (src/precomputed.ts): a chunk's compressed
// Morton code, a chunk found in a sharded scale through its shard index and gzipped
// minishard index (each read once), and compressed_segmentation decoded, uint32 and uint64,
// with a block cut by the chunk's edge. Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import { compressedMorton, compressedSegmentation, Shards, type Fetch } from "../src/precomputed.ts";

test("a chunk's compressed Morton code interleaves only the bits each axis's grid needs", () => {
  // a grid 4 × 2 × 1: x takes two bits, y one, z none
  assert.equal(compressedMorton([3, 1, 0], [4, 2, 1]), 0b111n);
  assert.equal(compressedMorton([2, 0, 0], [4, 2, 1]), 0b100n);
  assert.equal(compressedMorton([0, 1, 0], [4, 2, 1]), 0b010n);
  assert.equal(compressedMorton([0, 0, 0], [1, 1, 1]), 0n);
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
  assert.equal(asked.filter((a) => a.startsWith("s0/1.shard")).length, 4);  // index entry and minishard once, then the chunks
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
