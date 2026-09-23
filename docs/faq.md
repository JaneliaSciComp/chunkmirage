# FAQ

## Does this work with neuroglancer-demo.appspot.com?

**With the Python server on your machine: yes, today.** The server sends
`Access-Control-Allow-Origin: *`, and Chrome and Firefox treat `http://localhost` as a
secure origin, so the https appspot page may fetch from it without mixed-content blocking.
This is exactly how example-virtual-n5 and cellmap-flow work.

Caveats:

* **Safari** blocks https-to-http fetches even for localhost. Use Chrome or Firefox.
* **Other machines on your network** (a cluster node, a colleague's laptop): the printed
  URLs already use the machine's network address, and the python viewer works over plain
  http from anywhere on the network. The appspot viewer specifically blocks
  `http://10.x.x.x` as mixed content, so either SSH-tunnel the port so the browser sees
  `localhost`, or run with `--https` and accept the self-signed certificate once per
  browser. Pass `--public-url` when behind a tunnel or proxy.
* **VSCode Remote** forwards ports automatically, so `localhost:8000` in your laptop browser
  usually just works.
* **Refetching after an edit**: Neuroglancer caches by URL, so chunkmirage puts the pipeline
  digest in the URL. The control page can push the new state to an appspot window (camera
  resets), or `--python-viewer --ng-client appspot` runs the appspot client build with
  state sync so only the edited layer refetches. See [Interactivity](concepts/interactivity.md).

**Fully in-browser compute with appspot: impossible.** The client-side idea is to run ops
on your laptop GPU with no server. But Neuroglancer only knows how to *fetch chunks from a
URL*, and a browser tab cannot answer HTTP requests. The one browser mechanism that can
fake responses is a service worker, and a service worker only intercepts fetches from pages
on **its own origin**. A worker on `chunkmirage.github.io` never sees requests made by
`neuroglancer-demo.appspot.com`. CORS is unrelated: CORS governs whether a page may *read* a
response a real server sent, and here there is no server. The fix is to host our own
Neuroglancer build on the same origin as the service worker; see [Roadmap](roadmap.md).

## Why not just use Neuroglancer shaders?

For pointwise ops on one source (threshold, windowing, colormaps, channel mixing) you
should. They run on the GPU with zero latency. chunkmirage is for ops a shader cannot do:
neighbourhoods, models, resampling, multiple sources, format bridging.

## Why Python and not Rust?

The ops are the point, and scientists write them in numpy/torch/scipy. The hot paths
(tensorstore I/O, codecs, numpy kernels, torch) release the GIL, so per-request Python
overhead is negligible. Rust would only win for a pure transcoding proxy; if that ever
matters, the format layer is the piece to port, ideally as one crate compiled to both a
Python extension and WASM. See [Design](design.md#language-and-stack).

## Why uv and not pixi?

Everything the core needs is on PyPI. cellmap-flow keeps pixi for CUDA/torch and depends on
chunkmirage as an ordinary PyPI or git dependency.

## Couldn't Claude (or a script) just do this?

Claude can write a 10-line op on the spot, which is why chunkmirage does not need a big
built-in op library or a pipeline-composer UI. What it cannot cheaply reproduce is the
reusable substrate: byte-exact N5/zarr v2/zarr v3/precomputed encodings, OME metadata, halo
bookkeeping, per-stage caching, cache-busting URLs. And it cannot replace the interactive
loop where a human looks at the volume and adjusts. The intended division of labour: the
tool stays thin and correct; Claude writes ops and drives it through the REST/MCP API.

## Where does it fall short?

* **Global ops** (connected components, watershed, meshes) need whole-volume context;
  per-chunk with a halo gives a preview whose labels disagree across chunk borders.
* **Downsampled levels of derived data**: ops currently run independently per scale level.
  Correct for thresholding, wrong for a model trained at one resolution. Fix planned.
* **Latency**: a chunk must return in well under a second. Heavy inference on CPU will not.
  Halos multiply the work (a 32-voxel halo on a 64³ chunk computes ~3× the voxels).
* **Nothing is saved** unless the disk cache lands. What you see is ephemeral.
* **Single user**: per-process cache, no auth.
