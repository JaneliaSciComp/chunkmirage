// The page's end of the service worker's relay (relay.ts): one listener for its asks, each
// handed to the handler of the route its path holds (a prefix, matched anywhere in the
// path, as the worker matches /virtual/ under any base), and declined (null) when no route
// on this page has it, so the worker learns at once and waits for no timeout. One listener
// for the page, not one per route: the worker takes a page's first message as its answer,
// so two listeners each declining what is not theirs would answer for each other. The
// engine routes its share of /virtual/ here; a page with routes of its own (an API its page
// answers) adds them beside it.
import type { Ask } from "./types";

/** Answers `ask` on `port`: a Reply first (null declines), then a streamed one's Laters. */
export type Handler = (ask: Ask, port: MessagePort) => void;

const routes: { prefix: string; handle: Handler }[] = [];
let listening = false;

/** Answer the asks whose path holds `prefix` with `handle`; returns what removes the route. */
export function route(prefix: string, handle: Handler): () => void {
  const r = { prefix, handle };
  routes.push(r);
  if (!listening) { listening = true; navigator.serviceWorker.addEventListener("message", dispatch); }
  return () => { const i = routes.indexOf(r); if (i >= 0) routes.splice(i, 1); };
}

function dispatch(e: MessageEvent<Ask>) {
  const port = e.ports[0];
  if (!port || typeof e.data?.path !== "string") return;  // not an ask of the relay's
  const r = routes.find((r) => e.data.path.includes(r.prefix));
  if (r) r.handle(e.data, port); else port.postMessage(null);
}
