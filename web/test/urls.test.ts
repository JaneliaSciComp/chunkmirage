// Where a store's URL is fetched from (src/urls.ts): gs:// through the Cloud Storage JSON
// API with the object's name encoded as one piece, s3:// as the bucket's https URL, the
// rest as given; and which URLs are a store's metadata, the API's encoded names included.
// Run by `npm test`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fetchUrl, isMetadata } from "../src/urls.ts";

test("gs:// is fetched through the JSON API, the object's whole name one encoded segment", () => {
  assert.equal(fetchUrl("gs://flyem-male-cns/em/em-clahe-jpeg/info"),
    "https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/em%2Fem-clahe-jpeg%2Finfo?alt=media");
  // encoded once: +, a space and a % in a name
  assert.equal(fetchUrl("gs://b/a+b c%d/zarr.json"), "https://storage.googleapis.com/storage/v1/b/b/o/a%2Bb%20c%25d%2Fzarr.json?alt=media");
  // a query of the URL's own rides along after alt=media
  assert.equal(fetchUrl("gs://b/x/s0/0/0/0?generation=5"), "https://storage.googleapis.com/storage/v1/b/b/o/x%2Fs0%2F0%2F0%2F0?alt=media&generation=5");
});

test("s3:// is the bucket's https URL and anything else is as given", () => {
  assert.equal(fetchUrl("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/.zattrs"),
    "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/.zattrs");
  for (const u of ["https://example.org/vol.zarr/.zattrs", "http://10.0.0.5:8000/em/zarr/zarr.json", "https://storage.googleapis.com/bucket/x/info"]) {
    assert.equal(fetchUrl(u), u);
  }
});

test("a store joins a key to a gs:// base as a path, which fetchUrl then encodes whole", () => {
  // what zarrita's FetchStore does with a key: new URL(key, base + "/")
  const joined = new URL(".zarray", "gs://b/x/s0/").href;
  assert.equal(joined, "gs://b/x/s0/.zarray");
  assert.equal(fetchUrl(joined), "https://storage.googleapis.com/storage/v1/b/b/o/x%2Fs0%2F.zarray?alt=media");
});

test("metadata is told from chunks by the file's name, decoded", () => {
  for (const u of ["https://h/vol.zarr/.zattrs", "https://h/vol.zarr/s0/.zarray", "https://h/g/zarr.json", "https://h/n5/s0/attributes.json",
    "https://h/vol/info", fetchUrl("gs://b/em/x/info"), fetchUrl("gs://b/em/x/8x8x8/zarr.json")]) {
    assert.ok(isMetadata(new URL(u)), u);
  }
  for (const u of ["https://h/vol.zarr/s0/0/0/0", "https://h/vol/8x8x8/0a.shard", fetchUrl("gs://b/em/x/8x8x8/0a.shard"), "https://h/info/0.0.0", "https://h/x/%E0%A4%A"]) {
    assert.ok(!isMetadata(new URL(u)), u);
  }
});
