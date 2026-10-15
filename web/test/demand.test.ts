// The page's queue of block fits (src/demand.ts): the same rules as chunkmirage.demand's,
// checked the same way (tests/test_demand.py). Run by `npm test` (Node's own runner, types
// stripped), so it needs nothing installed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Cancelled, Claim, queue } from "../src/demand.ts";

const done = () => Promise.resolve(new Float32Array());

/** Fill the queue's three slots until `open` is called. */
function occupy(tag: string) {
  let open!: () => void;
  const gate = new Promise<void>((resolve) => (open = resolve));
  const jobs = [0, 1, 2].map((i) =>
    queue.submit(`${tag} ${i}`, 9, async () => { await gate; return new Float32Array(); }, () => {}),
  );
  return { open, jobs };
}

test("the finest level goes first, then the first asked for", async () => {
  const slots = occupy("order");
  const order: string[] = [];
  const ask = (key: string, level: number) =>
    queue.submit(key, level, () => { order.push(key); return done(); }, () => {});
  const asked = [ask("level 2", 2), ask("level 1", 1), ask("level 0", 0), ask("level 1, later", 1)];
  slots.open();
  await Promise.all([...slots.jobs, ...asked]);
  assert.deepEqual(order, ["level 0", "level 1", "level 1, later", "level 2"]);
});

test("work no request wants any more is dropped unrun", async () => {
  const slots = occupy("drop");
  const before = queue.dropped;
  let ran = false;
  let dropped = false;
  const fit = queue.submit("unwanted", 0, () => { ran = true; return done(); }, () => { dropped = true; });
  queue.claim("unwanted"); // a second request wants it
  queue.release("unwanted"); // one gives up: still wanted
  assert.equal(dropped, false);
  queue.release("unwanted"); // the other gives up too
  await assert.rejects(fit, Cancelled);
  slots.open();
  await Promise.all(slots.jobs);
  assert.equal(ran, false);
  assert.equal(dropped, true);
  assert.equal(queue.dropped, before + 1);
});

test("running work is kept when its request gives up", async () => {
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => (finish = resolve));
  const fit = queue.submit("running", 0, async () => { await gate; return new Float32Array([7]); }, () => {});
  queue.release("running"); // it already started: nothing to drop
  finish();
  assert.deepEqual([...(await fit)], [7]);
});

test("work that cannot be done any more is dropped unrun, whatever claims it holds", async () => {
  const slots = occupy("gone");
  let ran = false;
  let dropped = false;
  const fit = queue.submit("gone/0/0.0.0", 0, () => { ran = true; return done(); }, () => { dropped = true; });
  queue.claim("gone/0/0.0.0"); // two requests want it still
  queue.drop("gone/0/0.0.0", "gone is no longer served");
  await assert.rejects(fit, (e: Error) => e instanceof Cancelled && e.message === "cancelled: gone is no longer served");
  slots.open();
  await Promise.all(slots.jobs);
  assert.equal(ran, false);
  assert.equal(dropped, true);
});

test("dropping work that is running, or unknown, changes nothing", async () => {
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => (finish = resolve));
  const fit = queue.submit("running still", 0, async () => { await gate; return new Float32Array([3]); }, () => {});
  const before = queue.dropped;
  queue.drop("running still");
  queue.drop("never submitted");
  assert.equal(queue.dropped, before);
  finish();
  assert.deepEqual([...(await fit)], [3]);
});

test("a claim cancelled already runs what is hooked to it at once", () => {
  const claim = new Claim();
  let calls = 0;
  claim.onCancel(() => calls++);
  claim.cancel();
  claim.cancel(); // once only
  claim.onCancel(() => calls++);
  assert.equal(calls, 2);
});
