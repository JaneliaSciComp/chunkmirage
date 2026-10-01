// The demos the gallery (index.html) lists. A "pipeline" demo is a set of views, each a
// chunkmirage pipeline spec (source, select, ops, chunks), served by pipeline.html from
// Pyodide workers running chunkmirage's own ops, and a Neuroglancer layout over them; a
// "link" demo is a page of its own (the registration page). Each carries the command that
// serves the same from Python.
import type { PipelineView } from "./types";

export interface CardLayer {
  name: string;
  view?: string;             // one of the card's views, served by the page
  url?: string;              // or a store the viewer reads itself, e.g. zarr://https://...
  type: "image" | "segmentation";
  colour?: string;           // image layers: tint
  range?: [number, number];  // image layers: display limits
  percentiles?: [number, number];  // or limits from a sample of the view's data
  alpha?: number;            // image layers drawn over others: opacity at full intensity
  additive?: boolean;        // image layers: add to what is below (channels of one image)
}

interface Card { id: string; title: string; blurb: string; image: string; data: string; command: string }
export interface PipelineCard extends Card {
  kind: "pipeline";
  views: Record<string, PipelineView>;
  layers: CardLayer[];
  panels: string[][];        // layer names per panel, side by side, each an x-y slice
  position: number[];        // z, y, x, full-resolution voxels of the first view
  zoom: number;              // full-resolution voxels per screen pixel
}
export interface LinkCard extends Card { kind: "link"; href: string }
export type DemoCard = PipelineCard | LinkCard;

const COSEM = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2";
const PRED = `${COSEM}/jrc_hela-2.n5/labels`;
// the N5 predictions are upside down in y relative to the EM, though nothing says so
const MITO = `flip://${PRED}/mito_pred?axes=y`, ER = `flip://${PRED}/er_pred?axes=y`;
const EFISH = "https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119";
const ROUND1 = `${EFISH}/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0`;
const ROUND2 = `${EFISH}/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0`;
const CHUNK = [16, 128, 128];

export const CARDS: DemoCard[] = [
  {
    kind: "pipeline", id: "contacts", image: "cards/contacts.jpg",
    title: "Organelle contact sites, computed where you look",
    blurb: "Where mitochondria and the ER come within 12 nm of each other in a whole HeLa cell (122 gigavoxels of FIB-SEM), from OpenOrganelle's published predictions. Each chunk on screen is computed as you pan, by chunkmirage's own contacts and label ops running in this page; nothing is precomputed.",
    data: "OpenOrganelle jrc_hela-2 (Heinrich et al., Nature 2021): EM and the COSEM mitochondria and ER predictions",
    views: {
      mito: { source: MITO, chunk: CHUNK },
      er: { source: ER, chunk: CHUNK },
      contacts: {
        source: `stack://${MITO}|${ER}`, chunk: CHUNK,
        ops: [{ op: "contacts", radius: 3, a_low: 128, b_low: 128 }, { op: "label", min_size: 50 }],
      },
    },
    layers: [
      { name: "em", url: `zarr://${COSEM}/jrc_hela-2.zarr/recon-1/em/fibsem-uint8`, type: "image" },
      { name: "mito", view: "mito", type: "image", colour: "#33e64d", range: [64, 255], alpha: 0.35 },
      { name: "er", view: "er", type: "image", colour: "#e64de6", range: [64, 255], alpha: 0.35 },
      { name: "contacts", view: "contacts", type: "segmentation" },
    ],
    panels: [["em", "mito", "er", "contacts"]],
    position: [2156, 596, 3028], zoom: 1,
    command: `P=${PRED}\nchunkmirage serve "stack://flip://$P/mito_pred?axes=y|flip://$P/er_pred?axes=y" \\\n  --op contacts:radius=3 --op label:min_size=50 --chunk 16,128,128 --python-viewer`,
  },
  {
    kind: "pipeline", id: "spots", image: "cards/spots.jpg",
    title: "Single mRNA molecules across a fly brain",
    blurb: "An EASI-FISH round of a whole fly central brain, its two FISH channels' spots found chunk by chunk as you browse (a difference of Gaussians and its local maxima): the step a pipeline usually tunes on a crop, here on the whole brain. Left, the images; right, the spots found in them.",
    data: "Janelia EASI-FISH, fly central brain, round 1 (janelia-data-examples)",
    views: {
      structure: { source: ROUND1, select: { c: 0, t: 0 }, chunk: CHUNK },
      fish1: { source: ROUND1, select: { c: 1, t: 0 }, chunk: CHUNK },
      fish2: { source: ROUND1, select: { c: 2, t: 0 }, chunk: CHUNK },
      spots1: { source: ROUND1, select: { c: 1, t: 0 }, chunk: CHUNK, ops: [{ op: "spots", threshold: 10, radius: 2 }] },
      spots2: { source: ROUND1, select: { c: 2, t: 0 }, chunk: CHUNK, ops: [{ op: "spots", threshold: 8, radius: 2 }] },
    },
    layers: [
      { name: "structure", view: "structure", type: "image", colour: "#cccccc", percentiles: [50, 99.95], additive: true },
      { name: "channel 1", view: "fish1", type: "image", colour: "#33ff4d", percentiles: [99, 99.99], additive: true },
      { name: "channel 2", view: "fish2", type: "image", colour: "#ff4dff", percentiles: [99, 99.99], additive: true },
      { name: "spots 1", view: "spots1", type: "segmentation" },
      { name: "spots 2", view: "spots2", type: "segmentation" },
    ],
    panels: [["structure", "channel 1", "channel 2"], ["channel 1", "channel 2", "spots 1", "spots 2"]],
    position: [465, 960, 960], zoom: 0.4,
    command: `chunkmirage serve '${ROUND1}' --select c=1,t=0 \\\n  --op spots:threshold=10,radius=2 --chunk 16,128,128 --python-viewer`,
  },
  {
    kind: "link", id: "register-fly", image: "cards/register-fly.jpg",
    title: "Register two fly brain templates on your GPU",
    blurb: "Not just an affine: one is found from the images, then a deformable field is solved on top of it on this computer's GPU in seconds, and the moving brain is served through both at every resolution, chunk by chunk. Before (as stored), after, and the field (how far the deformable part moved each point) side by side.",
    data: "FCWB and JRC2018F templates (OME-Zarr RFC-5 examples)",
    href: "register.html",
    command: "uv run python examples/fly_brain_registration.py",
  },
  {
    kind: "link", id: "register-efish", image: "cards/register-efish.jpg",
    title: "Align two EASI-FISH rounds, finer where you zoom",
    blurb: "Two imaging rounds of one fly brain: an affine found from the images, a deformable field solved on top of it on the GPU, and finer fields fitted block by block only where you zoom in.",
    data: "Janelia EASI-FISH, fly central brain, rounds 1 and 2 (janelia-data-examples)",
    href: `register.html?fixed=${ROUND1}&moving=${ROUND2}&refine=3&iterations=100,40,40,40&window=15,31,31,31`,
    command: `chunkmirage serve 'register://${ROUND2}?fixed=${encodeURIComponent(ROUND1)}&affine=auto&refine=3&iterations=100,40,40,40&window=15,31,31,31&show=pair' --python-viewer`,
  },
];
