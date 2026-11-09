// Page views of the docs counted with GoatCounter (no cookies, nothing personal), on the live
// site only: the page's path, its title, the screen width and the referring site's domain, sent
// as the browser demos send theirs (web/src/count.ts). https://www.goatcounter.com
// David's GoatCounter site, shared with his other projects; every path starts /chunkmirage/.
(() => {
  if (location.hostname !== "janeliascicomp.github.io" || navigator.webdriver) return;
  let ref = "";
  try {
    ref = document.referrer ? new URL(document.referrer).origin : "";
  } catch {
    /* none */
  }
  const q = new URLSearchParams({
    p: location.pathname,
    t: `chunkmirage: ${document.title}`,
    e: "false",
    s: String(screen.width),
    r: ref,
    rnd: Math.random().toString(36).slice(2, 7),
  });
  const url = `https://davidackerman.goatcounter.com/count?${q}`;
  try {
    if (navigator.sendBeacon?.(url)) return;
  } catch {
    /* fall back */
  }
  new Image().src = url;
})();
