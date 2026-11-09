// Page views counted with GoatCounter (no cookies, nothing personal), on the live site only.
// The page sends GoatCounter's counting request itself rather than loading its script, which
// would also send the query string and so the addresses of images you register (register.html)
// and the Python you write (pipeline.html, in the hash). Sent: the page's path, with the demo
// card it shows (a name from the gallery, nothing else), its title, the screen width and the
// referring site's domain. The docs pages send the same (docs/javascripts/count.js).
// https://www.goatcounter.com
// David's GoatCounter site, shared with his other projects; this site is a project page, so
// every path already starts /chunkmirage/.
const CODE = "davidackerman"; // https://davidackerman.goatcounter.com
const LIVE = "janeliascicomp.github.io";

export function countView(): void {
  if (location.hostname !== LIVE || navigator.webdriver) return; // not local runs or tests
  const card = new URLSearchParams(location.search).get("card");
  const path = location.pathname + (card && /^[a-z0-9-]{1,40}$/.test(card) ? `?card=${card}` : "");
  let ref = "";
  try {
    ref = document.referrer ? new URL(document.referrer).origin : "";
  } catch {
    /* none */
  }
  const q = new URLSearchParams({
    p: path,
    t: `chunkmirage: ${document.title}`,
    e: "false",
    s: String(screen.width),
    r: ref,
    rnd: Math.random().toString(36).slice(2, 7),
  });
  const url = `https://${CODE}.goatcounter.com/count?${q}`;
  try {
    if (navigator.sendBeacon?.(url)) return;
  } catch {
    /* fall back */
  }
  new Image().src = url;
}

countView();
