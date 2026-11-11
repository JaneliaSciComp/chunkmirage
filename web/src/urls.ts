// Where a store's URL is fetched from, and which of its files are its metadata. No imports,
// so the service worker (relay.ts), the page, its workers and Node's tests share it.

/** The URL a browser fetches `url` at, as Neuroglancer fetches the same URLs.
 *
 * `gs://bucket/object` goes through the Cloud Storage JSON API. The bucket's own URL
 * (storage.googleapis.com/bucket/object) answers a page with no access-control-allow-origin
 * unless the bucket's owner set CORS on it, and most have not, so Chrome drops the answer
 * ("Failed to fetch"); the API answers every origin, allows the Range header in its
 * preflight and honours ranges (206). The object's whole name is one path segment there,
 * encoded as one piece (em/x/info → em%2Fx%2Finfo), so a reader builds hierarchical
 * `gs://` paths as it likes and this makes the URL at each request; appending to a base
 * would not do. `s3://bucket/object` is the bucket's https URL, which S3 serves with CORS
 * for a public bucket. Anything else is fetched as given. */
export function fetchUrl(url: string): string {
  const gs = /^gs:\/\/([^/?]+)\/?([^?]*)(?:\?(.*))?$/.exec(url);
  if (gs) {
    const [, bucket, object, query] = gs;
    return `https://storage.googleapis.com/storage/v1/b/${bucket}/o/${encodeURIComponent(object)}?alt=media${query ? `&${query}` : ""}`;
  }
  return url.replace(/^s3:\/\/([^/]+)/, "https://$1.s3.amazonaws.com");
}

/** `fetch(url, init)`, tried again after a pause when it fails in a way that may pass: the
 * network (or the browser, short of resources with too many requests out), or a server busy
 * or limiting its rate (429, 5xx). Up to `tries` in all, about 0.25, 0.5, 1 and 2 s apart;
 * the last answer, or error, is the caller's. A viewer does not ask again for a chunk that
 * failed, so a burst of thousands of small reads (a chunksizes:// level's indexes) must not
 * fail on one refusal. */
export async function fetchRetry(url: string, init: RequestInit = {}, tries = 5): Promise<Response> {
  for (let i = 1; ; i++) {
    try {
      const r = await fetch(url, init);
      if (i >= tries || !(r.status === 429 || r.status >= 500)) return r;
    } catch (e) {
      if (i >= tries || (e as Error)?.name === "AbortError") throw e;
    }
    await new Promise((ok) => setTimeout(ok, 250 * 2 ** (i - 1) * (0.5 + Math.random())));
  }
}

const METADATA = new Set([".zattrs", ".zarray", ".zgroup", ".zmetadata", "zarr.json", "attributes.json", "info"]);

/** Whether `url` is a store's metadata (zarr's, N5's, a precomputed volume's info) rather
 * than a chunk. The path is decoded first: through the JSON API an object's whole name is
 * one encoded segment, `…%2Finfo`. */
export function isMetadata(url: URL): boolean {
  let path = url.pathname;
  try { path = decodeURIComponent(path); } catch { /* not percent-encoded as a URL is: as it came */ }
  return METADATA.has(path.slice(path.lastIndexOf("/") + 1));
}
