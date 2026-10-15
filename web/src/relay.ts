// The service worker's two jobs, as a module: this site's service worker (sw.ts) is made of
// it, and so can be the service worker of another page built on the engine (engine.ts),
// which registers its own sw.js by name and relays routes of its own beside these.
//
// `relay`: a viewer (Neuroglancer, in an iframe on this origin) asks for .../virtual/... as
// if a server held the volume; the request is passed to the page that serves it over a
// MessageChannel, and the page computes the answer in its workers and replies (the page's
// end is routes.ts); a chunk's body is streamed once computed, and a client that gives up
// cancels the stream, which the page is told of, so it can drop the work. A browser tab
// cannot answer HTTP requests itself; a service worker on the same origin can, which is
// why the viewer is hosted here, not on appspot. Any request can be relayed the same way
// (an API the page answers, its event stream in pieces): the page answers the paths it
// has a route for and declines the rest, so the worker's fetch handler decides which paths
// go to a page, and the page which of them it owns.
//
// `fromStore`: every read of the images' stores, by the page, its chunk workers and the
// viewer alike, goes through one cache here: a region is read by many (overlapping blocks
// and chunks, the before and after views, the viewer's own fixed layer), and each worker
// keeping its own copy fetched the same chunks 13 times over.
import type { Ask, Later, Reply, StoreStats } from "./types";

export type { Ask, Later, Reply, StoreStats } from "./types";

const sw = self as unknown as ServiceWorkerGlobalScope;
const STORE_BYTES = 512 * 2 ** 20;  // fetched (still compressed) store chunks kept
const MAX_FETCHES = 24;             // upstream at once; more pending and Chrome refuses some

// ------------------------------------------------ the store cache
interface Stored { body: Blob; status: number; headers: [string, string][] }
const stored = new Map<string, Promise<Stored | null>>();  // in use order; null: not kept (opaque, or an error)
const sizes = new Map<string, number>();
let storedBytes = 0, fetching = 0;
const waiting: (() => void)[] = [];
/** Data requests seen, answered from the cache, fetched, and the bytes fetched. */
export const stats: StoreStats = { requests: 0, hits: 0, fetches: 0, fetchedBytes: 0 };
const KEEP = ["content-type", "content-range", "content-length"];

/** `req` (a GET of data, by its URL and range) answered from the cache, or fetched once for
 * every request that waits for it and kept, up to STORE_BYTES, the least recently used
 * dropped first; an opaque or failed response is passed on as it is and not kept. */
export async function fromStore(req: Request): Promise<Response> {
  const key = `${req.url} ${req.headers.get("range") ?? ""}`;
  stats.requests++;
  let entry = stored.get(key);
  if (entry) { stats.hits++; stored.delete(key); stored.set(key, entry); }
  else {
    entry = upstream(req).then(async (r) => {
      if (r.type === "opaque" || (!r.ok && r.status !== 404)) { forget(key); return null; }
      const body = await r.blob();
      const s: Stored = { body, status: r.status, headers: KEEP.flatMap((h) => { const v = r.headers.get(h); return v ? [[h, v] as [string, string]] : []; }) };
      stats.fetchedBytes += body.size; sizes.set(key, body.size); storedBytes += body.size;
      evict();
      return s;
    }, (e) => { forget(key); throw e; });
    stored.set(key, entry);
  }
  const s = await entry;
  return s ? new Response(s.body, { status: s.status, headers: s.headers }) : fetch(req);  // not kept: as it is
}

/** A network fetch, at most MAX_FETCHES at once. */
async function upstream(req: Request): Promise<Response> {
  while (fetching >= MAX_FETCHES) await new Promise<void>((r) => waiting.push(r));
  fetching++; stats.fetches++;
  try { return await fetch(req); } finally { fetching--; waiting.shift()?.(); }
}

function forget(key: string) {
  stored.delete(key);
  storedBytes -= sizes.get(key) ?? 0; sizes.delete(key);
}

function evict() {
  for (const key of stored.keys()) {
    if (storedBytes <= STORE_BYTES) break;
    if (sizes.has(key)) forget(key);  // settled entries only: one in flight has no size yet
  }
}

// ------------------------------------------------ the relay to the page
/** How a request is relayed. `pick`: the pages to ask, of this origin's open windows; all by
 * default (the one with a route for the path answers, the rest decline). `timeout`: how
 * long a page may take to answer before it counts as declining, two minutes by default;
 * longer for a route that may work long before its head (downloading something). */
export interface Relay { pick?: (pages: WindowClient[]) => WindowClient[]; timeout?: number }

/** The response to `req` from the page that answers its path: 503 if no open page does.
 * The request's path, range, method, headers and body (if any) are what the page is asked. */
export async function relay(req: Request, { pick = (pages) => pages, timeout = 120_000 }: Relay = {}): Promise<Response> {
  const url = new URL(req.url), body = req.method === "GET" || req.method === "HEAD" ? null : await req.arrayBuffer();
  const what: Ask = { path: url.pathname, range: req.headers.get("range"), method: req.method, headers: [...req.headers], body };
  // every page picked is asked at once: the one whose route this is answers, the rest (the
  // viewer's frame, other tabs) decline
  const pages = pick([...await sw.clients.matchAll({ type: "window", includeUncontrolled: true })]);
  const asks = pages.map((page) => ask(page, what, timeout));
  const reply = await first(asks.map((a) => a.reply));
  const owner = reply ? asks.find((a) => a.answered === reply) : undefined;
  for (const a of asks) if (a !== owner || !reply || !("stream" in reply)) a.cancel();
  if (!reply) return new Response("the page that serves this is closed", { status: 503 });
  const headers: Record<string, string> = { "content-type": reply.type, "cache-control": "no-store", ...Object.fromEntries(reply.headers ?? []) };
  if (reply.range) headers["content-range"] = reply.range;  // a part of a file (mesh fragments)
  if (!("stream" in reply)) return new Response(reply.body, { status: reply.status, headers });
  // A chunk: the head now, the body when the page has computed it (or in pieces, until done).
  // A client that stops waiting cancels this stream (a service worker sees no other sign of
  // it), and the page is told, so it can drop the work nobody wants any more.
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      owner!.later = (m: Later) => {
        if ("chunk" in m) return controller.enqueue(new Uint8Array(m.chunk));
        if ("body" in m) { controller.enqueue(new Uint8Array(m.body)); controller.close(); }
        else if ("done" in m) controller.close();
        else controller.error(new Error(m.error));
        owner!.cancel();
      };
    },
    cancel() { owner!.tell({ cancel: true }); owner!.cancel(); },
  });
  return new Response(stream, { status: reply.status, headers });
}

/** The first reply that is not null, or null once every page has declined or timed out. */
function first(replies: Promise<Reply>[]): Promise<Reply> {
  return new Promise((resolve) => {
    let left = replies.length;
    if (!left) resolve(null);
    for (const r of replies) void r.then((reply) => (reply ? resolve(reply) : --left || resolve(null)));
  });
}

/** Ask one page `what`: posted to it with a port, on which its first message is the reply
 * (or a streamed one's head; null declines), later ones a streamed body's contents
 * (`later`); `tell` tells the page its client gave up, `cancel` closes the port. A page that
 * never answers declines after `timeout` ms. */
export function ask(client: Client, what: Ask, timeout = 120_000) {
  const channel = new MessageChannel();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const a = {
    answered: null as Reply, later: (_: Later) => {},
    reply: null as unknown as Promise<Reply>,
    tell: (m: { cancel: true }) => channel.port1.postMessage(m),
    cancel: () => { clearTimeout(timer); channel.port1.close(); },
  };
  let got = false;
  a.reply = new Promise<Reply>((resolve) => {
    timer = setTimeout(() => resolve(null), timeout);  // a page that never answers
    channel.port1.onmessage = (e: MessageEvent<Reply | Later>) => {
      if (got) return a.later(e.data as Later);
      got = true; clearTimeout(timer);
      a.answered = e.data as Reply;
      resolve(a.answered);
    };
  });
  client.postMessage(what, [channel.port2]);
  return a;
}
