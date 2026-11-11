// The service worker, two jobs, both from relay.ts (a module the service worker of another
// page built on the engine imports too): a viewer's requests for .../virtual/... relayed to
// the page that serves them, and one cache of the images' stores for the page, its chunk
// workers and the viewer alike (the viewer's reads not queued behind the page's). The stored
// levels a view passes through to the viewer (.../stored/...) are fetched as it asks, not
// kept. This site's own files (pages, scripts, assets, the viewer) are the network's as
// they are.
import { storedDir } from "./precomputed";
import { fromStore, metadata, passThrough, relay, stats, type StoreStats } from "./relay";

// a stored level of a view served as precomputed: .../stored/<its directory>/<file> (precomputed.ts's storedKey)
const STORED = /\/virtual\/[^/]+\/precomputed\/[^/]+\/stored\/([A-Za-z0-9_-]+)\/(.+)$/;

const sw = self as unknown as ServiceWorkerGlobalScope;
// on a page served by web/serve.py (the page says so: pipeline.ts's logToServer), store reads
// that failed, posted to its /log
let logging = false;
function watched(url: string, answer: Promise<Response>): Promise<Response> {
  if (!logging) return answer;
  const log = (what: string) => void fetch("log", { method: "POST", body: JSON.stringify({ worker: what, url: url.slice(0, 300) }) }).catch(() => {});
  return answer.then((r) => { if (!r.ok && r.status !== 404) log(`status ${r.status}`); return r; }, (e) => { log(`failed: ${(e as Error)?.message ?? e}`); throw e; });
}

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const req = event.request, url = new URL(req.url), own = url.origin === sw.location.origin;
  const stored = own && req.method === "GET" ? STORED.exec(url.pathname) : null;
  if (stored) {
    const file = `${storedDir(stored[1])}/${stored[2]}`;
    return event.respondWith(watched(file, passThrough(file, req.headers.get("range"))));
  }
  if (own && url.pathname.includes("/virtual/")) return event.respondWith(relay(req));
  // data a script fetches (not pages, scripts or this site's own assets and viewer), and of
  // that the chunks: a store's metadata is the page's own request
  const asset = own && /\/(assets|ng)\//.test(url.pathname);
  if (req.method !== "GET" || req.destination !== "" || asset || metadata(url)) return;
  // the viewer's own reads of a store (a layer it reads itself; its frame or its workers, all
  // under ng/) share the cache with the page's but do not queue behind them
  event.respondWith(watched(req.url, (async () => {
    const client = event.clientId ? await sw.clients.get(event.clientId) : undefined;
    return fromStore(req, !(client && new URL(client.url).pathname.includes("/ng/")));
  })()));
});

sw.addEventListener("message", (event) => {
  if ((event.data as { type?: string })?.type === "log") logging = true;
  if ((event.data as { type?: string })?.type === "store-stats") event.ports[0]?.postMessage(stats satisfies StoreStats);
});
