// The service worker: Neuroglancer (in an iframe on this origin) asks for .../virtual/...
// as if a server held the registered volume; this passes each request to the
// registration page, which computes the answer in its chunk workers and replies. Nothing
// else is intercepted. A browser tab cannot answer HTTP requests itself; a service worker
// on the same origin can, which is why Neuroglancer is hosted here, not on appspot.
import type { Reply } from "./types";

const sw = self as unknown as ServiceWorkerGlobalScope;

sw.addEventListener("install", () => void sw.skipWaiting());
sw.addEventListener("activate", (event) => event.waitUntil(sw.clients.claim()));

sw.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== sw.location.origin || !url.pathname.includes("/virtual/")) return;
  event.respondWith(relay(url.pathname));
});

async function relay(path: string): Promise<Response> {
  // every open page on this site is asked at once: the one whose share of /virtual/ this is
  // answers, the rest (the viewer's frame, other tabs) never do
  const pages = await sw.clients.matchAll({ type: "window", includeUncontrolled: true });
  const asks = pages.map((page) => ask(page, path));
  const reply = await first(asks.map((a) => a.reply));
  for (const a of asks) a.cancel();
  if (!reply) return new Response("the registration page that serves this volume is closed", { status: 503 });
  return new Response(reply.body, {
    status: reply.status,
    headers: { "content-type": reply.type, "cache-control": "no-store" },
  });
}

/** The first reply that is not null, or null once every page has declined or timed out. */
function first(replies: Promise<Reply>[]): Promise<Reply> {
  return new Promise((resolve) => {
    let left = replies.length;
    if (!left) resolve(null);
    for (const r of replies) void r.then((reply) => (reply ? resolve(reply) : --left || resolve(null)));
  });
}

function ask(client: Client, path: string) {
  const channel = new MessageChannel();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const reply = new Promise<Reply>((resolve) => {
    timer = setTimeout(() => resolve(null), 120_000);
    channel.port1.onmessage = (e: MessageEvent<Reply>) => { clearTimeout(timer); resolve(e.data); };
  });
  client.postMessage({ path }, [channel.port2]);
  return { reply, cancel: () => { clearTimeout(timer); channel.port1.close(); } };
}
