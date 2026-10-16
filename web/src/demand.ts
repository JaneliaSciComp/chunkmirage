// Expensive work done because a client asked for it, in the order clients want it, and dropped
// when no client waits for it any more: the page's side of chunkmirage.demand, with the same
// rules. A request holds a Claim on the work it needs. A client that gives up on a request
// aborts it; a service worker is not told of the abort, but the reply it streams is cancelled,
// so the claim is cancelled, and waiting work no request claims any more is dropped without
// running (one already running finishes, and its result is kept). Among the rest, the finest
// level goes first: a client asking for one place at two levels (a viewer showing a coarse
// placeholder while the fine chunk computes) wants the finer. Within a level the first asked
// for goes first. A few run at once, so the GPU works on one while the next ones read, and the
// rest cannot flood the network with reads.
//
// One rule more than chunkmirage.demand's: work asked for in the background (a page writing a
// region to a store, Engine.materialize), not for a client waiting on it. A background claim
// is never released by a client giving up, so the work is not dropped, but a job only
// background claims hold waits behind every other, whatever its level, and such jobs take at
// most all the slots but one (one, if there is only one): a viewer's request finds a slot free
// or the next one to free, while the background work goes on in the rest. A client asking for
// a job waiting in the background brings it forward, and it goes back if the client gives up.

const SLOTS = 3;  // jobs run at once

/** A request's hold on the work it needs: released when it is answered or its client gives
 * up on it (the service worker sees the reply cancelled), so work nobody wants is dropped. */
export class Claim {
  cancelled = false;
  private hooks: (() => void)[] = [];
  onCancel(f: () => void) { if (this.cancelled) f(); else this.hooks.push(f); }
  cancel() { if (this.cancelled) return; this.cancelled = true; for (const f of this.hooks.splice(0)) f(); }
}

/** Work that will not be done: no request wants it any more, or (`why`) it cannot be done,
 * its view no longer served or its engine stopped. */
export class Cancelled extends Error { constructor(why = "no request wants this any more") { super(`cancelled: ${why}`); } }

interface Job<T> {
  run: () => Promise<T>; level: number; seq: number;  // seq: when first asked for
  demand: number;      // requests holding a claim on it
  background: number;  // of them, background claims: a job only they hold waits behind the rest
  resolve: (v: T) => void; reject: (e: unknown) => void; onDrop: () => void;
}

/** Jobs waiting, in the order described at the top; `slots` of them run at once. */
export class Queue<T = Float32Array> {
  /** Jobs run at once. */
  readonly slots: number;
  /** Of them, jobs only background claims hold (`submit`'s `background`): all but one, so a
   * client's request finds a slot free; one, if there is only one. */
  readonly backgroundSlots: number;
  constructor(slots = SLOTS) { this.slots = slots; this.backgroundSlots = Math.max(1, slots - 1); }
  private waiting = new Map<string, Job<T>>();
  private active = new Map<string, { level: number; background: boolean }>();  // running, by key
  dropped = 0;  // waiting fits no request wanted any more, never run
  onChange: () => void = () => {};  // a fit waiting, started, done or dropped
  private seq = 0;
  get running() { return this.active.size; }
  get queued() { return this.waiting.size; }
  /** Fits running and waiting, per level. */
  byLevel(): { running: Map<number, number>; waiting: Map<number, number> } {
    const count = (levels: Iterable<number>) => { const m = new Map<number, number>(); for (const l of levels) m.set(l, (m.get(l) ?? 0) + 1); return m; };
    return { running: count([...this.active.values()].map((a) => a.level)), waiting: count([...this.waiting.values()].map((j) => j.level)) };
  }

  /** A new fit, wanted by one request (one in the background, with `background`); `onDrop`
   * runs, synchronously, if it is dropped unrun. */
  submit(key: string, level: number, run: () => Promise<T>, onDrop: () => void, background = false): Promise<T> {
    return new Promise((resolve, reject) => {
      const job: Job<T> = { run, level, resolve, reject, onDrop, seq: ++this.seq, demand: 1, background: background ? 1 : 0 };
      this.waiting.set(key, job);
      this.pump();
      this.onChange();
    });
  }

  /** Another request wants a fit (one in the background, with `background`): if it is still
   * waiting, it counts one more, and a client's claim brings a background job forward. */
  claim(key: string, background = false) {
    const job = this.waiting.get(key);
    if (!job) return;
    job.demand++;
    if (background) job.background++;
    else this.pump();  // in the background before: it may take a slot kept for clients
  }

  /** A request no longer wants a fit (`background`: one that claimed it in the background):
   * waiting and wanted by no one, it is dropped unrun; wanted in the background alone, it goes
   * back behind the rest. */
  release(key: string, background = false) {
    const job = this.waiting.get(key);
    if (!job) return;
    if (background && job.background > 0) job.background--;
    if (--job.demand > 0) return;
    this.drop(key);
  }

  /** A waiting fit dropped unrun, whatever claims it holds: `release` does this once none
   * is left, and the page when the fit cannot be done any more (`why`: its view is no
   * longer served, the engine stopped). One running finishes; one unknown is nothing. */
  drop(key: string, why?: string) {
    const job = this.waiting.get(key);
    if (!job) return;
    this.waiting.delete(key);
    this.dropped++;
    job.onDrop();
    job.reject(new Cancelled(why));
    this.onChange();
  }

  private pump() {
    while (this.active.size < this.slots && this.waiting.size) {
      let best: [string, Job<T>] | null = null;
      for (const e of this.waiting) if (!best || before(e[1], best[1])) best = e;
      const [key, job] = best!, background = inBackground(job);
      // the best in the background: so is every job waiting, and they have slots of their own
      if (background && [...this.active.values()].filter((a) => a.background).length >= this.backgroundSlots) return;
      this.waiting.delete(key);
      this.active.set(key, { level: job.level, background });
      job.run().then(job.resolve, job.reject).finally(() => { this.active.delete(key); this.pump(); this.onChange(); });
    }
  }
}

/** Whether only background claims hold a job. */
const inBackground = <T>(job: Job<T>) => job.demand === job.background;

/** Which of two waiting fits goes first: one a client waits for before one wanted in the
 * background alone; then the finer level (a client asking for one place at two levels, as a
 * viewer does to show a coarse placeholder while the fine chunk computes, wants the finer
 * one; coarser fits run when nothing finer waits), then the first asked for. */
function before<T>(a: Job<T>, b: Job<T>): boolean {
  const ba = inBackground(a), bb = inBackground(b);
  if (ba !== bb) return bb;
  return a.level !== b.level ? a.level < b.level : a.seq < b.seq;
}

export const queue = new Queue();
