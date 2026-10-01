# Roadmap

Ordered by impact. Items move to the docs proper when they land; this page must not
describe shipped features as future work.

1. **A real inference consumer.** Port an existing live-inference server (cellmap-flow is
   the obvious candidate) onto an `Op` with `halo` and `cache=True` on a GPU node. Not
   because the tool is for that project, but because real models, real data and a real user
   are the fastest way to expose the gaps below.
2. **Multiscale semantics for ops.** Per-op declaration of valid scale levels plus a
   downsample-from-s0 mode. Today each level is processed independently, which is wrong for
   inference.
3. **Materialize-on-browse with a disk cache.** Back the cache with a real zarr on local or
   shared storage so it persists, is shared across workers, and doubles as a partially
   computed output. Add a background filler that expands outward from requested chunks: the
   viewer becomes the job scheduler. (Ordering the work by what clients ask for, and
   dropping what they stop waiting for, has shipped; see
   [caching](concepts/caching.md#order-of-work-and-requests-given-up-on). Inference ops
   would join its queues.)
4. **Lazy-compute story for non-viewer clients.** Demonstrate dask/tensorstore reading a
   served pipeline and running downstream analysis with no intermediate written.
5. **DAG pipelines, multi-source ops.** Named stages with fan-out (one model, many
   post-processors) and a `Combine` op taking another pipeline as input. Unlocks masking,
   model-vs-model disagreement views, registration overlays. (The two-input case has
   shipped in its simplest form: `stack://` serves images on one grid as channels and an
   op such as `contacts` consumes them; see
   [formats](concepts/formats.md#stack-sources-several-images-as-one-arrays-channels).)
   With it, structured sources:
   a pipeline's source as a JSON object as well as a URL, such as
   `{"register": {"moving": ..., "fixed": ..., "affine": [...]}}`, typed by the same schema
   (`chunkmirage schema`) so nested sources and long parameters need no escaping and the
   browser engine, an MCP server or an agent build specs instead of strings. The URL
   schemes (`register://`, `scene://`, ...) stay as shorthand for the command line and
   links; `register://` becomes a two-input node rather than a source.
6. **More registration inputs, live landmarks, fusion.** Readers for bigstream/EASI-FISH
   (`affine.mat` plus a deform zarr), BigWarp landmarks and ANTs/ITK fields into the same
   transform model; landmark pairs drawn in Neuroglancer, fitted to a thin-plate spline and
   served live; several scene images fused into one volume (for example through
   multiview-stitcher). Resampling through OME-Zarr 0.6 transformations has shipped as
   [`scene://` sources](concepts/formats.md#scene-sources-ome-zarr-06-transformations).
7. **Adaptive caching.** Measure stage compute time at runtime and cache automatically when
   it exceeds a threshold, removing the manual `cache` flag.
8. **MCP surface and hot-loaded ops.** `list_ops`, `set_pipeline`, `define_op` from source,
   `neuroglancer_link`, `screenshot`. Off by default, local only. (The REST API, `/api/events`
   stream, control page and python-neuroglancer viewer it would wrap already exist; see
   [Interactivity](concepts/interactivity.md).)
9. **Deployment hardening.** Bearer token on `/api/*`, shared-cache multi-worker mode.
   (`--https` with an auto-generated self-signed certificate has shipped; see the
   [CLI reference](reference/cli.md).)
10. **Own-hosted Neuroglancer with a service worker.** The zero-install browser demo with
   WebGPU ops and ONNX Runtime Web inference, sharing the JSON pipeline spec with the
   Python server. See [FAQ](faq.md#does-this-work-with-neuroglancer-demoappspotcom) for why
   it cannot target the hosted appspot viewer. A first piece exists: `web/`, TypeScript
   whose types are generated from chunkmirage's JSON Schema (`chunkmirage schema`), fits
   `register://`'s deformable registration on the viewer's GPU, reading OME-Zarr straight
   from its URLs (see the [design notes](design.md#client-side-browser-roadmap)), and
   serves the registered volume to Neuroglancer through a service worker, computed by web
   workers. The docs site hosts it with its own Neuroglancer build. What remains, in order:
    * A gallery. An index page of cards, each a JSON pipeline spec plus viewer settings,
      opened by one generic page that builds its form from the schema, embeds Neuroglancer
      and shows the same pipeline as a `chunkmirage serve` command. The registration page
      becomes a card. This is the first, smallest piece of structured sources (item 5).
    * Ops on data nobody hosts. `synthetic://` ported to TypeScript (a hash of world
      coordinates, exact at every scale), the pointwise and filter ops as WebGPU compute
      shaders, `label` on the CPU, each with a parity test against Python on seeded synthetic
      data. Cards: a filter chain, morphology on shells, the same volume served as zarr v2,
      v3, N5 and precomputed by the service worker, a 4096³ pyramid that costs nothing.
    * Real public data: OpenOrganelle's bucket allows any origin, so a card can run a live
      filter chain on jrc_hela-2 with nothing copied at deploy time.
    * A layer of where `refine` has fitted the field and how well (each block's
      correlation before and after), in both engines.
    * User code, in two tiers. A `python` op holding a block-to-block function runs in the
      page through Pyodide (numpy, scipy and scikit-image ship with it) and natively in
      Python, where the server takes it only from the command line or a spec file, never
      through the REST API. Array functions exported to ONNX run on the GPU in both engines
      (ONNX Runtime and ONNX Runtime Web).
    * A CI job that runs the page headless and diffs its chunks against Python's, so the
      engines cannot drift. Whether the page's WebGPU shaders can also serve Python
      (wgpu-py), as one implementation of the solver for both, is still to be measured
      against PyTorch.

## Untapped potential

* **Every zarr reader is a client.** dask, xarray, tensorstore, napari, Fiji and
  cellmap-analyze can open a served pipeline as a lazy array, making the server a general
  lazy compute node with a shared cache.
* **Derived datasets as specs.** A pipeline spec is a few hundred bytes; publishing a
  derived view of a public dataset costs no storage, and stateless chunks could run
  serverless.
* **Active learning loop.** Serve model uncertainty as a layer, take annotations back from
  Neuroglancer, fine-tune, invalidate the inference stage.
* **Claude in the loop.** With Neuroglancer's screenshot endpoint plus the REST API, an
  agent can render, look, adjust and repeat.
