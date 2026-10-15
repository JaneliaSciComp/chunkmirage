// Computed chunks kept a while on the page for clients that fetch them again (a map client
// keeps none itself; a viewer that drops a chunk and asks for it later): the page's side of
// chunkmirage.cache.LRUCache, bounded in bytes, the least recently used dropped first. A view
// no longer served would leave its chunks here, so they can be forgotten by view.

/** Chunk bodies by key (`<view>/<level>/<index>`, as engine.ts keys them). */
export class Kept {
  private bodies = new Map<string, ArrayBuffer>();  // in use order
  private budget: number;
  /** Bytes kept now. */
  bytes = 0;

  /** `budget`: bytes to keep at most; past it the least recently used go. (A plain field, not
   * a parameter property: Node's type stripping, which runs the tests, has none.) */
  constructor(budget: number) { this.budget = budget; }

  get size(): number { return this.bodies.size; }

  /** The chunk kept under `key`, counted as used now; undefined if none. */
  get(key: string): ArrayBuffer | undefined {
    const body = this.bodies.get(key);
    if (body) { this.bodies.delete(key); this.bodies.set(key, body); }
    return body;
  }

  /** Keep `body` under `key` (in place of one kept under it before), then drop the least
   * recently used until the rest fit the budget; one chunk larger than the budget is kept
   * alone. */
  put(key: string, body: ArrayBuffer) {
    this.drop(key);
    this.bodies.set(key, body);
    this.bytes += body.byteLength;
    while (this.bytes > this.budget && this.bodies.size > 1) this.drop(this.bodies.keys().next().value!);
  }

  /** Forget every chunk of `view` (the keys `<view>/...`); how many there were. */
  forget(view: string): number {
    let n = 0;
    for (const key of [...this.bodies.keys()]) if (key.startsWith(`${view}/`)) { this.drop(key); n++; }
    return n;
  }

  clear() { this.bodies.clear(); this.bytes = 0; }

  private drop(key: string) {
    const body = this.bodies.get(key);
    if (!body) return;
    this.bodies.delete(key);
    this.bytes -= body.byteLength;
  }
}
