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

const SLOTS = 3;  // jobs run at once

/** A request's hold on the work it needs: released when it is answered or its client gives
 * up on it (the service worker sees the reply cancelled), so work nobody wants is dropped. */
export class Claim {
  cancelled = false;
  private hooks: (() => void)[] = [];
  onCancel(f: () => void) { if (this.cancelled) f(); else this.hooks.push(f); }
  cancel() { if (this.cancelled) return; this.cancelled = true; for (const f of this.hooks.splice(0)) f(); }
}

export class Cancelled extends Error { constructor() { super("cancelled: no request wants this any more"); } }

interface Job<T> {
  run: () => Promise<T>; level: number; seq: number;  // seq: when first asked for
  demand: number;  // requests holding a claim on it
  resolve: (v: T) => void; reject: (e: unknown) => void; onDrop: () => void;
}

/** Jobs waiting, in the order described at the top; `slots` of them run at once. */
export class Queue<T = Float32Array> {
  private slots: number;
  constructor(slots = SLOTS) { this.slots = slots; }
  private waiting = new Map<string, Job<T>>();
  private active = new Map<string, number>();  // running: key -> level
  dropped = 0;  // waiting fits no request wanted any more, never run
  onChange: () => void = () => {};  // a fit waiting, started, done or dropped
  private seq = 0;
  get running() { return this.active.size; }
  get queued() { return this.waiting.size; }
  /** Fits running and waiting, per level. */
  byLevel(): { running: Map<number, number>; waiting: Map<number, number> } {
    const count = (levels: Iterable<number>) => { const m = new Map<number, number>(); for (const l of levels) m.set(l, (m.get(l) ?? 0) + 1); return m; };
    return { running: count(this.active.values()), waiting: count([...this.waiting.values()].map((j) => j.level)) };
  }

  /** A new fit, wanted by one request; `onDrop` runs, synchronously, if it is dropped unrun. */
  submit(key: string, level: number, run: () => Promise<T>, onDrop: () => void): Promise<T> {
    return new Promise((resolve, reject) => {
      const job: Job<T> = { run, level, resolve, reject, onDrop, seq: ++this.seq, demand: 1 };
      this.waiting.set(key, job);
      this.pump();
      this.onChange();
    });
  }

  /** Another request wants a fit: if it is still waiting, it counts one more. */
  claim(key: string) { const job = this.waiting.get(key); if (job) job.demand++; }

  /** A request no longer wants a fit: waiting and wanted by no one, it is dropped unrun. */
  release(key: string) {
    const job = this.waiting.get(key);
    if (!job || --job.demand > 0) return;
    this.waiting.delete(key);
    this.dropped++;
    job.onDrop();
    job.reject(new Cancelled());
    this.onChange();
  }

  private pump() {
    while (this.active.size < this.slots && this.waiting.size) {
      let best: [string, Job<T>] | null = null;
      for (const e of this.waiting) if (!best || before(e[1], best[1])) best = e;
      const [key, job] = best!;
      this.waiting.delete(key);
      this.active.set(key, job.level);
      job.run().then(job.resolve, job.reject).finally(() => { this.active.delete(key); this.pump(); this.onChange(); });
    }
  }
}

/** Which of two waiting fits goes first: the finer level (a client asking for one place at
 * two levels, as a viewer does to show a coarse placeholder while the fine chunk computes,
 * wants the finer one; coarser fits run when nothing finer waits), then the first asked for. */
function before<T>(a: Job<T>, b: Job<T>): boolean {
  return a.level !== b.level ? a.level < b.level : a.seq < b.seq;
}

export const queue = new Queue();
