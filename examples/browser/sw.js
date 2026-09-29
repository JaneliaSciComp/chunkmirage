// The service worker: Neuroglancer (in an iframe on this origin) asks for .../virtual/...
// as if a server held the registered volume; this passes each request to the
// registration page, which computes the answer in its chunk workers and replies. Nothing
// else is intercepted. A browser tab cannot answer HTTP requests itself; a service worker
// on the same origin can, which is why Neuroglancer is hosted here, not on appspot.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || !url.pathname.includes("/virtual/")) return;
  event.respondWith(relay(url.pathname));
});

async function relay(path) {
  const pages = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  for (const page of pages) {
    if (new URL(page.url).pathname.includes("/ng/")) continue;  // the viewer itself
    const reply = await ask(page, path);
    if (reply) {
      return new Response(reply.body, {
        status: reply.status,
        headers: { "content-type": reply.type ?? "application/octet-stream", "cache-control": "no-store" },
      });
    }
  }
  return new Response("the registration page that serves this volume is closed", { status: 503 });
}

function ask(client, path) {  // the page's reply, or null if it does not serve that path
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = (e) => resolve(e.data);
    client.postMessage({ path }, [channel.port2]);
    setTimeout(() => resolve(null), 120000);
  });
}
