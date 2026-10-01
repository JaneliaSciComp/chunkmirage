// The service worker, two jobs. A viewer (Neuroglancer, in an iframe on this origin) asks for
// .../virtual/... as if a server held the registered volume; this passes each request to the
// registration page, which computes the answer in its chunk workers and replies. A browser
// tab cannot answer HTTP requests itself; a service worker on the same origin can, which is
// why the viewer is hosted here, not on appspot. And every read of the images' stores, by the
// page, its chunk workers and the viewer alike, goes through one cache here: a region is read
// by many (overlapping blocks and chunks, the before and after views, the viewer's own fixed
// layer), and each worker keeping its own copy fetched the same chunks 13 times over.
import type { Later, Reply, StoreStats } from "./types";

const sw = self as unknown as ServiceWorkerGlobalScope;
const STORE_BYTES = 512 * 2 ** 20;  // fetched (still compressed) store chunks kept
const MAX_FETCHES = 24;             // upstream at once; more pending and Chrome refuses some

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const req = event.request, url = new URL(req.url), own = url.origin === sw.location.origin;
  if (own && url.pathname.includes("/virtual/")) return event.respondWith(relay(url.pathname));
  // data a script fetches (not pages, scripts or this site's own assets and viewer)
  const asset = own && /\/(assets|ng)\//.test(url.pathname);
  if (req.method === "GET" && req.destination === "" && !asset) event.respondWith(fromStore(req));
});

sw.addEventListener("message", (event) => {
  if ((event.data as { type?: string })?.type === "store-stats") event.ports[0]?.postMessage(stats satisfies StoreStats);
});

// ------------------------------------------------ the store cache
interface Stored { body: Blob; status: number; headers: [string, string][] }
const stored = new Map<string, Promise<Stored | null>>();  // in use order; null: not kept (opaque, or an error)
const sizes = new Map<string, number>();
let storedBytes = 0, fetching = 0;
const waiting: (() => void)[] = [];
const stats: StoreStats = { requests: 0, hits: 0, fetches: 0, fetchedBytes: 0 };
const KEEP = ["content-type", "content-range", "content-length"];

async function fromStore(req: Request): Promise<Response> {
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

async function relay(path: string): Promise<Response> {
  // every open page on this site is asked at once: the one whose share of /virtual/ this is
  // answers, the rest (the viewer's frame, other tabs) never do
  const pages = await sw.clients.matchAll({ type: "window", includeUncontrolled: true });
  const asks = pages.map((page) => ask(page, path));
  const reply = await first(asks.map((a) => a.reply));
  const owner = reply ? asks.find((a) => a.answered === reply) : undefined;
  for (const a of asks) if (a !== owner || !reply || !("stream" in reply)) a.cancel();
  if (!reply) return new Response("the registration page that serves this volume is closed", { status: 503 });
  const headers = { "content-type": reply.type, "cache-control": "no-store" };
  if (!("stream" in reply)) return new Response(reply.body, { status: reply.status, headers });
  // A chunk: the head now, the body when the page has computed it. A client that stops waiting
  // cancels this stream (a service worker sees no other sign of it), and the page is told, so
  // it can drop the work nobody wants any more.
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      owner!.later = (m: Later) => {
        if ("body" in m) { controller.enqueue(new Uint8Array(m.body)); controller.close(); } else controller.error(new Error(m.error));
        owner!.cancel();
      };
    },
    cancel() { owner!.tell({ cancel: true }); owner!.cancel(); },
  });
  return new Response(body, { status: reply.status, headers });
}

/** The first reply that is not null, or null once every page has declined or timed out. */
function first(replies: Promise<Reply>[]): Promise<Reply> {
  return new Promise((resolve) => {
    let left = replies.length;
    if (!left) resolve(null);
    for (const r of replies) void r.then((reply) => (reply ? resolve(reply) : --left || resolve(null)));
  });
}

/** Ask one page for `path`: its first message is the reply (or its head), later ones a
 * streamed body's contents. */
function ask(client: Client, path: string) {
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
    timer = setTimeout(() => resolve(null), 120_000);  // a page that never answers
    channel.port1.onmessage = (e: MessageEvent<Reply | Later>) => {
      if (got) return a.later(e.data as Later);
      got = true; clearTimeout(timer);
      a.answered = e.data as Reply;
      resolve(a.answered);
    };
  });
  client.postMessage({ path }, [channel.port2]);
  return a;
}
