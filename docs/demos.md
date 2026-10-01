# Demos

## In the browser, nothing to install

**[The gallery](https://yuriyzubov.github.io/chunkmirage/browser/)** lists demos that run
entirely in your browser (Chrome, Edge or Safari 26, for WebGPU and service workers). Images
are read from their public buckets and every chunk on screen is computed in the page when
the viewer asks for it; nothing is uploaded or precomputed. Each card also shows the
`chunkmirage serve` command that serves the same from Python, for Neuroglancer, Fiji,
napari or a dask script.

| demo | what is computed per chunk | data |
| ---- | -------------------------- | ---- |
| [Organelle contact sites](https://yuriyzubov.github.io/chunkmirage/browser/pipeline.html?card=contacts) | `contacts` then `label` on mitochondria and ER predictions, stacked and flipped into the EM's frame | OpenOrganelle jrc_hela-2 |
| [Single mRNA molecules](https://yuriyzubov.github.io/chunkmirage/browser/pipeline.html?card=spots) | `spots` on both FISH channels | Janelia EASI-FISH, fly central brain |
| [Fly brain templates](https://yuriyzubov.github.io/chunkmirage/browser/register.html) | the moving brain resampled through a field solved on your GPU | OME-Zarr RFC-5 examples |
| [EASI-FISH rounds](https://yuriyzubov.github.io/chunkmirage/browser/register.html?fixed=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0&moving=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0&refine=3&iterations=100,40,40,40&window=15,31,31,31) | an affine found, a field solved, finer fields fitted where you zoom | Janelia EASI-FISH, rounds 1 and 2 |

The two pipeline demos run chunkmirage's own Python in the page: Pyodide (Python compiled to
WebAssembly, with numpy and scipy) loads the package's ops and `chunkmirage.fused`, the code
a server's pipeline stage runs, in a few web workers (about 6 s, once). The page reads the
images in TypeScript (OME-Zarr through zarrita, N5 itself, `stack://` and `flip://` as the
Python sources do) and hands each worker the padded block a chunk needs; the worker returns
the chunk. `tests/test_fused.py` checks that a chunk computed this way equals the pipeline's.
On an RTX 2080 Ti workstation the contact-sites view filled in about 40 s, a spot or
contact block taking 0.05 to 0.13 s in Python; chunks the viewer gave up on before their
turn are dropped unrun, as the server drops them. A demo is an entry of `web/src/cards.ts`:
views (each a pipeline spec: source, `select`, ops, chunks) and a Neuroglancer layout.

## From Python

| script | shows | needs |
| ------ | ----- | ----- |
| `examples/contact_sites.py` | the contact-sites demo served by Python, recomputing at a prompt | `uv sync --extra all` |
| `examples/fish_spots.py` | the spots demo served by Python, thresholds at a prompt | `uv sync --extra all` |
| `examples/fly_brain_registration.py` | the fly templates through their published OME-Zarr 0.6 transformations (`scene://`) | `uv sync --extra all` |
| `examples/register_demo.py` | a deformable registration solved as the source opens, re-solved as you type settings | `uv sync --extra all --extra gpu` (or `--extra cpu`) |
| `examples/swirl_demo.py` | a procedural deformation and its field, animated on a time axis | `uv sync --extra all` |
| `examples/demo.py` | one small volume served in every format at once, with a shared cache | `uv sync --extra ops` |
| `examples/quickstart.py` | the library in thirty lines: a cached blur, a threshold, a server | `uv sync --extra ops` |

Each prints a viewer link; see [Getting started](getting-started.md) for the basics.
