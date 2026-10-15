// The page's kept chunks (src/kept.ts): a byte-bounded LRU, and a view's chunks forgotten
// together when it is no longer served (Engine.unserve). Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Kept } from "../src/kept.ts";

const body = (n: number) => new ArrayBuffer(n);

test("past the budget the least recently used goes first, and one used again is young", () => {
  const kept = new Kept(10);
  kept.put("a/0/0", body(4));
  kept.put("a/0/1", body(4));
  assert.ok(kept.get("a/0/0"));  // used again: now the youngest
  kept.put("a/0/2", body(4));    // 12 bytes: the oldest, a/0/1, goes
  assert.equal(kept.get("a/0/1"), undefined);
  assert.ok(kept.get("a/0/0") && kept.get("a/0/2"));
  assert.deepEqual([kept.size, kept.bytes], [2, 8]);
});

test("one chunk larger than the budget is kept alone", () => {
  const kept = new Kept(10);
  kept.put("a/0/0", body(4));
  kept.put("a/0/1", body(16));
  assert.equal(kept.get("a/0/0"), undefined);
  assert.deepEqual([kept.size, kept.bytes], [1, 16]);
});

test("a chunk kept again under its key replaces the one before, its bytes counted once", () => {
  const kept = new Kept(100);
  kept.put("a/0/0", body(4));
  kept.put("a/0/0", body(6));
  assert.deepEqual([kept.size, kept.bytes], [1, 6]);
});

test("a view's chunks are forgotten together, and only its own", () => {
  const kept = new Kept(100);
  kept.put("model~1/0/0.0.0", body(4));
  kept.put("model~1/mesh0/0.0.0", body(4));
  kept.put("model~12/0/0.0.0", body(4));  // another view whose name starts the same
  kept.put("raw/0/0.0.0", body(4));
  assert.equal(kept.forget("model~1"), 2);
  assert.deepEqual([kept.size, kept.bytes], [2, 8]);
  assert.ok(kept.get("model~12/0/0.0.0") && kept.get("raw/0/0.0.0"));
  assert.equal(kept.forget("model~1"), 0);
  kept.clear();
  assert.deepEqual([kept.size, kept.bytes], [0, 0]);
});
