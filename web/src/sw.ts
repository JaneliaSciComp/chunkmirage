// The service worker, two jobs, both from relay.ts (a module the service worker of another
// page built on the engine imports too): a viewer's requests for .../virtual/... relayed to
// the page that serves them, and one cache of the images' stores for the page, its chunk
// workers and the viewer alike. This site's own files (pages, scripts, assets, the viewer)
// are the network's as they are.
import { fromStore, relay, stats, type StoreStats } from "./relay";

const sw = self as unknown as ServiceWorkerGlobalScope;

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const req = event.request, url = new URL(req.url), own = url.origin === sw.location.origin;
  if (own && url.pathname.includes("/virtual/")) return event.respondWith(relay(req));
  // data a script fetches (not pages, scripts or this site's own assets and viewer)
  const asset = own && /\/(assets|ng)\//.test(url.pathname);
  if (req.method === "GET" && req.destination === "" && !asset) event.respondWith(fromStore(req));
});

sw.addEventListener("message", (event) => {
  if ((event.data as { type?: string })?.type === "store-stats") event.ports[0]?.postMessage(stats satisfies StoreStats);
});
