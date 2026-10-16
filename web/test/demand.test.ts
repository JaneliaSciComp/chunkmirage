// The page's queue of block fits (src/demand.ts): the same rules as chunkmirage.demand's,
// checked the same way (tests/test_demand.py). Run by `npm test` (Node's own runner, types
// stripped), so it needs nothing installed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Cancelled, Claim, Queue, queue } from "../src/demand.ts";

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

// ------------------------------------------------ background work (Engine.materialize's)
/** A job that runs until its gate opens, noting when it starts. */
function gated(started: string[], key: string) {
  let open!: () => void;
  const gate = new Promise<void>((resolve) => (open = resolve));
  return { open, run: async () => { started.push(key); await gate; return new Float32Array(); } };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

test("background work waits behind every job a client waits for, whatever its level, and takes all the slots but one", async () => {
  const q = new Queue(3), started: string[] = [];
  assert.deepEqual([q.slots, q.backgroundSlots], [3, 2]);
  const bg = [0, 1, 2].map((i) => gated(started, `background ${i}`));
  const runs = bg.map((b, i) => q.submit(`bg ${i}`, 0, b.run, () => {}, true));
  await tick();
  assert.deepEqual(started, ["background 0", "background 1"]);  // the third slot is kept for clients
  const client = gated(started, "client, level 3");
  const asked = q.submit("client", 3, client.run, () => {});
  await tick();
  assert.deepEqual(started.slice(2), ["client, level 3"]);  // at once, in the slot kept for it
  const later = gated(started, "client, later, level 5");
  const asked2 = q.submit("client later", 5, later.run, () => {});
  bg[0].open();
  await tick();
  // a slot freed: the client's job goes before the background one waiting, coarser as it is
  assert.deepEqual(started.slice(3), ["client, later, level 5"]);
  client.open();
  await tick();
  assert.deepEqual(started.slice(4), ["background 2"]);
  for (const g of [...bg, later]) g.open();
  await Promise.all([...runs, asked, asked2]);
});

test("a background claim is never released by a client giving up; given up in the background, it is dropped", async () => {
  const q = new Queue(1), started: string[] = [];
  const busy = gated(started, "busy");
  const running = q.submit("busy", 0, busy.run, () => {});
  let dropped = 0;
  const wanted = q.submit("both", 0, async () => { started.push("both"); return new Float32Array(); }, () => dropped++, true);
  q.claim("both");    // a client asks for it too
  q.release("both");  // and gives up
  busy.open();
  await wanted;
  assert.deepEqual([started, dropped], [["busy", "both"], 0]);
  await running;
  const again = gated(started, "busy again");
  const r2 = q.submit("busy again", 0, again.run, () => {});
  const unwanted = q.submit("background alone", 0, async () => new Float32Array(), () => dropped++, true);
  q.release("background alone", true);  // materialize aborted before its turn
  await assert.rejects(unwanted, Cancelled);
  assert.equal(dropped, 1);
  again.open();
  await r2;
});

test("a client asking for a background job brings it forward, and it goes back when the client gives up", async () => {
  const q = new Queue(3), started: string[] = [];
  const bg = [0, 1, 2, 3].map((i) => gated(started, `background ${i}`));
  const runs = bg.map((b, i) => q.submit(`bg ${i}`, 0, b.run, () => {}, true));
  await tick();
  assert.deepEqual(started, ["background 0", "background 1"]);
  q.claim("bg 3");  // a viewer wants the last one: it takes the slot kept for clients at once
  await tick();
  assert.deepEqual(started.slice(2), ["background 3"]);
  // a fourth waiting in the background, a client's claim on it given up: back behind the rest
  const fourth = gated(started, "background 4");
  const r4 = q.submit("bg 4", 0, fourth.run, () => {}, true);
  q.claim("bg 4");
  q.release("bg 4");
  const client = gated(started, "client");
  const asked = q.submit("client", 1, client.run, () => {});
  bg[0].open();
  await tick();
  assert.deepEqual(started.slice(3), ["client"]);
  for (const g of [...bg, fourth, client]) g.open();
  await Promise.all([...runs, r4, asked]);
});

test("with one slot, background work runs when no client waits", async () => {
  const q = new Queue(1), started: string[] = [];
  assert.equal(q.backgroundSlots, 1);
  const bg = gated(started, "background");
  const run = q.submit("bg", 0, bg.run, () => {}, true);
  await tick();
  assert.deepEqual(started, ["background"]);
  bg.open();
  await run;
});
