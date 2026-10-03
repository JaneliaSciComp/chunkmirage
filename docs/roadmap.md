# Roadmap

Ordered by impact. Items move to the docs proper when they land; this page must not
describe shipped features as future work.

1. **A real inference consumer.** Port an existing live-inference server (cellmap-flow is
   the obvious candidate) onto an `Op` with `halo` and `cache=True` on a GPU node. Not
   because the tool is for that project, but because real models, real data and a real user
   are the fastest way to expose the gaps. Reading that server's code against this
   one gave the list; the rule for what goes here is that a second consumer with no notion
   of models would want it too, stated without model vocabulary. Everything else (model
   configs, device slots, warmup, weight swaps, job launching, dashboards) stays in the
   consumer, as a plugin package of ops and a source scheme, which the two entry points
   make possible with no change here.

    The five changes this needed have shipped: ops that return their block shaved by the
    halo and add leading axes ([pipelines](concepts/pipelines.md#ops); `gradient` and the
    Gulf Stream fronts demo use both), a `chunkmirage.sources` entry point
    ([formats](concepts/formats.md#your-own-schemes-sources-from-other-packages)), `kind`
    on `ArrayInfo` (`output_kind`), ops at one resolution with the pyramid downsampled from
    them ([pipelines](concepts/pipelines.md#ops-at-one-resolution)), and a token on `/api/*`
    ([API](reference/api.md)). So have the hooks an application needs to build on
    chunkmirage as an ordinary dependency: datasets built from their name on first request
    ([resolvers](reference/api.md#datasets-resolved-by-name)), routes of its own and the app
    mounted inside another ([API](reference/api.md#routes-of-your-own)), `serve --ready-file`
    for launchers ([CLI](reference/cli.md)), ops whose output depends on state outside their
    parameters ([`cache_token`](concepts/caching.md#state-outside-the-parameters)), a limit
    on how many calls of an op run at once (`slots`, [pipelines](concepts/pipelines.md#ops)),
    reading the nearest level as it is (`input_level`), zero padding, labels told apart from
    images by dtype, and remote stores read anonymously first. What is left is the
    consumer's own port: its ops as `Op` subclasses, its scripts that read the data
    themselves as a source scheme, its launcher starting `chunkmirage serve`.
2. **Materialize-on-browse with a disk cache.** Opt-in, off by default. Back the cache with
   a real zarr on local or shared storage so it persists, is shared across workers, and doubles as a partially
   computed output. Add a background filler that expands outward from requested chunks: the
   viewer becomes the job scheduler. (Ordering the work by what clients ask for, and
   dropping what they stop waiting for, has shipped; see
   [caching](concepts/caching.md#order-of-work-and-requests-given-up-on). Inference ops
   would join its queues.)
3. **Lazy-compute story for non-viewer clients.** Demonstrate dask/tensorstore reading a
   served pipeline and running downstream analysis with no intermediate written.
4. **DAG pipelines, multi-source ops.** Named stages with fan-out (one model, many
   post-processors) and a `Combine` op taking another pipeline as input. Unlocks masking,
   the difference of any two sources (two models' outputs, two conditions), registration
   overlays. (The two-input case has
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
5. **More registration inputs, live landmarks, fusion.** Readers for bigstream/EASI-FISH
   (`affine.mat` plus a deform zarr), BigWarp landmarks and ANTs/ITK fields into the same
   transform model; landmark pairs drawn in Neuroglancer, fitted to a thin-plate spline and
   served live; several scene images fused into one volume (for example through
   multiview-stitcher). Resampling through OME-Zarr 0.6 transformations has shipped as
   [`scene://` sources](concepts/formats.md#scene-sources-ome-zarr-06-transformations), and
   tiles stitched by interest points and fused as read as
   [`stitch://` sources](concepts/formats.md#stitch-sources-tiles-stitched-by-interest-points-and-ransac)
   (BigStitcher projects of OME-Zarr tiles; other loaders, multi-view registration and
   saving the result back to the project are still to do).
6. **Adaptive caching.** Measure stage compute time at runtime and cache automatically when
   it exceeds a threshold, removing the manual `cache` flag.
7. **MCP surface and hot-loaded ops.** `list_ops`, `set_pipeline`, `define_op` from source,
   `neuroglancer_link`, `screenshot`. Off by default, local only. (The REST API, `/api/events`
   stream, control page and python-neuroglancer viewer it would wrap already exist; see
   [Interactivity](concepts/interactivity.md).)
8. **Deployment hardening.** Shared-cache multi-worker mode. (`--https` with an
   auto-generated self-signed certificate, and `--token` on `/api/*`, have shipped; see the
   [CLI reference](reference/cli.md).)
9. **Own-hosted Neuroglancer with a service worker.** The zero-install browser demo with
   WebGPU ops and ONNX Runtime Web inference, sharing the JSON pipeline spec with the
   Python server. See [FAQ](faq.md#does-this-work-with-neuroglancer-demoappspotcom) for why
   it cannot target the hosted appspot viewer. A first piece exists: `web/`, TypeScript
   whose types are generated from chunkmirage's JSON Schema (`chunkmirage schema`), fits
   `register://`'s deformable registration on the viewer's GPU, reading OME-Zarr straight
   from its URLs (see the [design notes](design.md#client-side-browser-roadmap)), and
   serves the registered volume to Neuroglancer through a service worker, computed by web
   workers. The docs site hosts it with its own Neuroglancer build, and a gallery of demos
   whose pipelines run chunkmirage's own ops in the page through Pyodide (see
   [Demos](demos.md)). What remains, in order:
    * Cards built from the schema: a form for each card's parameters, so a visitor edits the
      pipeline as the control page edits a server's.
    * Ops on data nobody hosts. `synthetic://` ported to TypeScript (a hash of world
      coordinates, exact at every scale), the pointwise and filter ops as WebGPU compute
      shaders, `label` on the CPU, each with a parity test against Python on seeded synthetic
      data. Cards: a filter chain, morphology on shells, the same volume served as zarr v2,
      v3, N5 and precomputed by the service worker, a 4096³ pyramid that costs nothing.
    * A layer of where `refine` has fitted the field and how well (each block's
      correlation before and after), in both engines.
    * User code, in two tiers. A `python` op holding a block-to-block function runs in the
      page through Pyodide (as the built-in ops already do there) and natively in
      Python, where the server takes it only from the command line or a spec file, never
      through the REST API. Array functions exported to ONNX run on the GPU in both engines
      (ONNX Runtime and ONNX Runtime Web).
    * A CI job that runs the page headless and diffs its chunks against Python's, so the
      engines cannot drift. Whether the page's WebGPU shaders can also serve Python
      (wgpu-py), as one implementation of the solver for both, is still to be measured
      against PyTorch.

10. **Domain ops and sources as plugin packages.** A proposal, not a decision: the geo
    and microscopy demos' ops and sources move to `chunkmirage-geo` and
    `chunkmirage-microscopy`, registered through the entry points any plugin uses, so the
    core stays general and the plugin API is exercised by real use
    ([design](design.md#what-stays-in-the-core-a-proposal)).

## Untapped potential

* **Every zarr reader is a client.** dask, xarray, tensorstore, napari, Fiji and
  cellmap-analyze can open a served pipeline as a lazy array, making the server a general
  lazy compute node with a shared cache.
* **Derived datasets as specs.** A pipeline spec is a few hundred bytes; publishing a
  derived view of a public dataset costs no storage, and stateless chunks could run
  serverless.
* **Invalidation when outside state changes.** An op's `cache_token` and a dataset refresh
  give a stage new keys when the files or weights it depends on change, and viewers refetch
  ([caching](concepts/caching.md#state-outside-the-parameters)). Loops built on it, such as
  annotating, fine-tuning and serving the new model, belong to the applications that use
  chunkmirage (cellmap-flow's, for models), not to chunkmirage.
* **Claude in the loop.** With Neuroglancer's screenshot endpoint plus the REST API, an
  agent can render, look, adjust and repeat.
