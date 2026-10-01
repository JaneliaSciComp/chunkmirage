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
  type: "image" | "segmentation" | "mesh";  // mesh: the view's surface, from its mesh spec
  colour?: string;           // image layers: tint; segmentation layers: every segment's colour
  range?: [number, number];  // image layers: display limits
  percentiles?: [number, number];  // or limits from a sample of the view's data
  alpha?: number;            // image layers drawn over others: opacity at full intensity; segmentations: opacity
  additive?: boolean;        // image layers: add to what is below (channels of one image)
  shader?: string;           // image layers: a Neuroglancer shader of its own, over the above
  volume?: boolean;          // image layers: volume rendered in 3-D panels
}

interface Card { id: string; title: string; blurb: string; image: string; data: string; command: string }
export interface PipelineCard extends Card {
  kind: "pipeline";
  views: Record<string, PipelineView>;
  layers: CardLayer[];
  panels: string[][];        // layer names per panel, side by side, each an x-y slice
  layouts?: string[];        // or each its own: "xy", "xz", "yz", "3d"
  turn?: number[];           // 3-D panels: the view's rotation (a quaternion)
  position: number[];        // full-resolution voxels of the first view, along its axes
  zoom: number;              // full-resolution voxels per screen pixel
  orientation?: number[];    // the slices' rotation (a quaternion), e.g. north up for a map
  timeline?: Timeline;       // the date on screen, for an axis of days
  playback?: { axis: string; velocity: number };  // the viewer's play button: steps per second
}
/** Dates along an axis whose coordinate counts days from `start`, and what happened on some. */
export interface Timeline { axis: string; start: string; events: { from: string; to?: string; text: string }[] }
/** A layer of a map demo: a view the page computes, drawn with an OpenLayers WebGL tile
 * style (expressions over `['band', 1]` and `['var', name]`). */
export interface MapLayer { name: string; view: string; style: Record<string, unknown>; visible?: boolean }
/** A slider: a parameter of a view's op (the view is computed again, for what is on screen),
 * or a style variable of a layer (redrawn by the map at once). */
export type MapControl = { label: string; unit: string; min: number; max: number; step: number }
  & ({ view: string; op: number; param: string } | { layer: string; variable: string; value: number });
export interface MapCard extends Card {
  kind: "map";
  views: Record<string, PipelineView>;
  projection: { code: string; extent: number[] };  // the views' own, so nothing is reprojected
  layers: MapLayer[];
  controls: MapControl[];
  /** Sources to choose between for `views` (the map fits each, and goes no further out). */
  sites: { label: string; views: string[]; options: { name: string; url: string }[] };
}
export interface LinkCard extends Card { kind: "link"; href: string }
/** A demo that runs from Python only (its data cannot be read from a browser page). */
export interface PythonCard extends Card { kind: "python"; why: string }
export type DemoCard = PipelineCard | MapCard | LinkCard | PythonCard;

const COSEM = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2";
const PRED = `${COSEM}/jrc_hela-2.n5/labels`;
// the N5 predictions are upside down in y relative to the EM, though nothing says so
const MITO = `flip://${PRED}/mito_pred?axes=y`, ER = `flip://${PRED}/er_pred?axes=y`;
const EFISH = "https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119";
const ROUND1 = `${EFISH}/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0`;
const ROUND2 = `${EFISH}/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0`;
const CHUNK = [16, 128, 128];
const SOUTH_POLE = "https://astrogeo-ard.s3.us-west-2.amazonaws.com/moon";
const DEMS = `${SOUTH_POLE}/lro/lola/barker_south_pole_dems`;
const SITES: [string, string][] = [  // NASA's 5 m south-pole elevation maps: id, place
  ["Site01", "Connecting ridge (Shackleton to de Gerlache)"], ["SL3", "Connecting ridge extension"],
  ["Site04", "Shackleton rim"], ["LM1", "Shackleton rim B"], ["Site07", "Peak near Shackleton"],
  ["Site11", "de Gerlache rim"], ["SL2", "de Gerlache rim 2"], ["Site42", "de Gerlache-Kocher massif"],
  ["Site23", "Malapert massif"], ["NPD", "Malapert crater"], ["Site20v2", "Leibnitz beta plateau"],
  ["Haworth", "Haworth"], ["Shoemaker", "Shoemaker"], ["LM2", "Shoemaker rim A"], ["LM3", "Shoemaker rim B"],
  ["LM4", "Shoemaker rim C"], ["LM5", "Shoemaker rim D"], ["LM6", "Shoemaker rim E"], ["LM8", "Shoemaker rim F"],
  ["LM7", "Faustini rim"], ["DM1", "Amundsen rim"], ["NPB", "Amundsen"], ["Site06", "Nobile rim 1"],
  ["DM2", "Nobile rim 2"], ["NPA", "Cabeus exterior wall"], ["NPC", "Idel'son L crater"],
];
const RIDGE = `${DEMS}/Site01/Site01.tif`;
const TILE = [1, 256, 256];  // a map's tiles
const SIDE = 2 ** 28;  // the Mandelbulb's array: 2^28 voxels across, 21 levels (2^30 overflows Neuroglancer's stack)
const BULB = `synthetic://mandelbulb?shape=${SIDE},${SIDE},${SIDE}&voxel_size=1&unit=nm`;
const BULB_SLICE = `#uicontrol float period slider(min=4, max=200, default=48)
void main() {
  float v = toNormalized(getDataValue()) * 255.0;  // 4 x the escape iteration; 255 inside
  if (v > 254.5) { emitRGB(vec3(0.02)); return; }
  emitRGB(0.5 + 0.5 * cos(6.2832 * (v / period + vec3(0.0, 0.33, 0.67))));
}`;
const BULB_VOLUME = `#uicontrol float period slider(min=4, max=200, default=48)
#uicontrol float skin slider(min=0, max=1, default=0.5)
#uicontrol float core slider(min=0, max=0.2, default=0)
void main() {
  float v = toNormalized(getDataValue()) * 255.0;  // 4 x the escape iteration; 255 inside
  if (v > 254.5) { emitRGBA(vec4(0.98, 0.85, 0.55, core)); return; }  // inside: a faint core
  if (v < 32.0) { emitTransparent(); return; }  // well outside: clear
  // just outside the surface, where points take longest to escape: the bulb's skin
  emitRGBA(vec4(0.5 + 0.5 * cos(6.2832 * (v / period + vec3(0.0, 0.33, 0.67))), skin * (v - 32.0) / 223.0));
}`;
const MUR = "https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1/analysed_sst";
const LAND = "if (isnan(v)) { emitRGB(vec3(0.18)); return; }";  // MUR has no value on land
const KELVIN = `#uicontrol invlerp temperature(range=[298, 305])
void main() { float v = getDataValue(); ${LAND} emitRGB(colormapJet(clamp(temperature(), 0.0, 1.0))); }`;
const CHANGE = `#uicontrol invlerp change(range=[-2, 2])
void main() {
  float v = getDataValue(); ${LAND}
  float t = clamp(change(), 0.0, 1.0) * 2.0 - 1.0;
  emitRGB(t < 0.0 ? mix(vec3(1.0), vec3(0.15, 0.35, 0.85), -t) : mix(vec3(1.0), vec3(0.85, 0.2, 0.15), t));
}`;

export const CARDS: DemoCard[] = [
  {
    kind: "link", id: "register-fly", image: "cards/register-fly.jpg",
    title: "Two fly brains registered on your GPU in seconds, served at every resolution",
    blurb: "Not just an affine: one is found from the images, then a deformable field is solved on top of it on this computer's GPU in seconds, and the moving brain is served through both at every resolution, chunk by chunk. Before (as stored), after, and the field (how far the deformable part moved each point) side by side.",
    data: "FCWB and JRC2018F templates (OME-Zarr RFC-5 examples)",
    href: "register.html",
    command: "uv run python examples/fly_brain_registration.py",
  },
  {
    kind: "link", id: "register-efish", image: "cards/register-efish.jpg",
    title: "Two EASI-FISH rounds aligned: finer deformation fields fitted where you zoom",
    blurb: "Two imaging rounds of one fly brain: an affine found from the images, a deformable field solved on top of it on the GPU, and finer fields fitted block by block only where you zoom in.",
    data: "Janelia EASI-FISH, fly central brain, rounds 1 and 2 (janelia-data-examples)",
    href: `register.html?fixed=${ROUND1}&moving=${ROUND2}&refine=3&iterations=100,40,40,40&window=15,31,31,31`,
    command: `chunkmirage serve 'register://${ROUND2}?fixed=${encodeURIComponent(ROUND1)}&affine=auto&refine=3&iterations=100,40,40,40&window=15,31,31,31&show=pair' --python-viewer`,
  },
  {
    kind: "pipeline", id: "contacts", image: "cards/contacts.jpg",
    title: "Organelle contact sites in a whole cell, computed as you pan",
    blurb: "Where mitochondria and the ER come within 12 nm of each other in a whole HeLa cell (122 gigavoxels of FIB-SEM). The organelles are OpenOrganelle's published predictions thresholded at 128, where the predicted distance to their boundary crosses zero (the first step of its own segmentations): mitochondria labelled as objects in green, the ER in magenta, and the contact sites between them as objects of their own. Each chunk on screen is computed as you pan, by chunkmirage's threshold, label and contacts ops running in this page; nothing is precomputed.",
    data: "OpenOrganelle jrc_hela-2 (Heinrich et al., Nature 2021): EM and the COSEM mitochondria and ER predictions",
    views: {
      mito: { source: MITO, chunk: CHUNK, ops: [{ op: "threshold", low: 128 }, { op: "label", min_size: 50 }] },
      er: { source: ER, chunk: CHUNK, ops: [{ op: "threshold", low: 128 }] },
      contacts: {
        source: `stack://${MITO}|${ER}`, chunk: CHUNK,
        ops: [{ op: "contacts", radius: 3, a_low: 128, b_low: 128 }, { op: "label", min_size: 50 }],
      },
    },
    layers: [
      { name: "em", url: `zarr://${COSEM}/jrc_hela-2.zarr/recon-1/em/fibsem-uint8`, type: "image" },
      { name: "mito", view: "mito", type: "segmentation", colour: "#33e64d", alpha: 0.3 },
      { name: "er", view: "er", type: "segmentation", colour: "#e64de6", alpha: 0.3 },
      { name: "contacts", view: "contacts", type: "segmentation", colour: "#ffd21f", alpha: 0.9 },
    ],
    panels: [["em", "mito", "er", "contacts"]],
    position: [2156, 596, 3028], zoom: 1,
    command: `P=${PRED}\nchunkmirage serve "stack://flip://$P/mito_pred?axes=y|flip://$P/er_pred?axes=y" \\\n  --op contacts:radius=3 --op label:min_size=50 --chunk 16,128,128 --python-viewer`,
  },
  {
    kind: "pipeline", id: "spots", image: "cards/spots.jpg",
    title: "Every mRNA spot in a fly brain, found as you browse",
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
    kind: "pipeline", id: "mandelbulb", image: "cards/mandelbulb.jpg",
    title: "A 3-D fractal to zoom into forever, computed chunk by chunk",
    blurb: "The Mandelbulb, the best known 3-D fractal, as a zarr array 2^28 voxels across, 21 levels deep: 10^25 voxels that exist nowhere. Each chunk is computed when the viewer asks for it, by chunkmirage's synthetic source running in this page; zoom into the slice on the left and finer levels iterate more, so new buds keep appearing. Right, the whole bulb volume rendered from its coarse levels. Colours are escape times, the same at every level.",
    data: "Computed: the power-8 Mandelbulb (White and Nylander, 2009), chunkmirage's synthetic://mandelbulb",
    views: {
      slice: { source: BULB, chunk: [256, 1, 256] },
      volume: { source: BULB, chunk: [32, 32, 32] },
    },
    layers: [
      { name: "slice", view: "slice", type: "image", shader: BULB_SLICE },
      { name: "bulb", view: "volume", type: "image", shader: BULB_VOLUME, volume: true },
    ],
    panels: [["slice"], ["bulb"]], layouts: ["xz", "3d"], turn: [0.28, 0.2, 0.06, 0.94],
    position: [SIDE / 2, SIDE / 2, SIDE / 2], zoom: SIDE / 700,
    command: `chunkmirage serve '${BULB}' --chunk 256,1,256 --python-viewer`,
  },
  {
    kind: "pipeline", id: "mesh-bulb", image: "cards/mesh-bulb.jpg",
    title: "The Mandelbulb as a solid surface, meshed as the viewer asks for it",
    blurb: "The same 3-D fractal, now as a surface: each piece of mesh is made when Neuroglancer fetches it, by marching cubes over a chunk of chunkmirage's synthetic Mandelbulb, in this page. 64 pieces, each its chunk plus one voxel so they meet without seams; nothing is stored. Turn it in the 3-D panel.",
    data: "Computed: the power-8 Mandelbulb, synthetic://mandelbulb, its 256³ level meshed by scikit-image's marching cubes",
    views: { bulb: { source: BULB, chunk: [64, 64, 64], mesh: { threshold: 255, level: 20 } } },
    layers: [{ name: "surface", view: "bulb", type: "mesh", colour: "#f2c46d" }],
    panels: [["surface"]], layouts: ["3d"], turn: [0.28, 0.2, 0.06, 0.94],
    position: [SIDE / 2, SIDE / 2, SIDE / 2], zoom: SIDE / 700,
    command: `chunkmirage serve '${BULB}' --mesh threshold=255,level=20 --chunk 64,64,64 --python-viewer`,
  },
  {
    kind: "pipeline", id: "mesh-moon", image: "cards/mesh-moon.jpg",
    title: "Shackleton crater's rim in 3-D, meshed from its elevation as you look",
    blurb: "NASA's 5 m elevation map of the rim of Shackleton crater at the Moon's south pole, 16 km across, turned into a surface in 3-D: each piece of terrain is made when Neuroglancer fetches it, two triangles per 20 m cell, by chunkmirage in this page. Heights are real, not exaggerated: from the rim the ground falls 4.6 km into the crater.",
    data: "LOLA 5 m south-pole elevation (Barker et al.), Shackleton rim, cloud-optimized GeoTIFF (USGS Astrogeology)",
    views: { rim: { source: `${DEMS}/Site04/Site04.tif`, chunk: [1, 200, 200], mesh: { kind: "terrain", level: 2 } } },
    layers: [{ name: "terrain", view: "rim", type: "mesh", colour: "#c9c4b8" }],
    panels: [["terrain"]], layouts: ["3d"], turn: [0.5, 0, 0, 0.866],
    position: [0, 1600, 1600], zoom: 4,
    command: `chunkmirage serve '${DEMS}/Site04/Site04.tif' --mesh kind=terrain,level=2 --chunk 200,200 --python-viewer`,
  },
  {
    kind: "map", id: "moon", image: "cards/moon.jpg",
    title: "Landing ground at the Moon's south pole: slope and relief computed per map tile",
    blurb: "NASA's 5 m elevation maps of 26 places at the Moon's south pole, where Artemis astronauts may land (from the Lunar Orbiter Laser Altimeter, 16 to 30 km across). Everything on the map is computed for each tile as the map asks, by chunkmirage's hillshade and slope ops in this page: the relief, lit by a sun you can move (shading only, no cast shadows), and in green the ground flat enough to land on, under the slope you choose. Pick a site to compute another. The map is OpenLayers, reading the page's chunks as GeoZarr: no Neuroglancer here.",
    data: "LOLA 5 m south-pole elevation (Barker et al.), cloud-optimized GeoTIFFs (USGS Astrogeology, AWS Open Data)",
    views: {
      relief: { source: RIDGE, chunk: TILE, ops: [{ op: "hillshade", azimuth: 135, altitude: 10 }] },
      slope: { source: RIDGE, chunk: TILE, ops: [{ op: "slope" }] },
    },
    projection: { code: "IAU_2015:30135", extent: [-1095700, -1095700, 1095700, 1095700] },  // south polar stereographic, the Moon's sphere
    layers: [
      { name: "Relief", view: "relief", style: { color: ["array", ["/", ["band", 1], 255], ["/", ["band", 1], 255], ["/", ["band", 1], 255], ["case", [">", ["band", 1], 0], 1, 0]] } },
      {
        name: "Flat enough to land", view: "slope",
        style: { variables: { flat: 8 }, color: ["case", ["<", ["band", 1], ["var", "flat"]], ["color", 40, 220, 90, 0.55], ["color", 0, 0, 0, 0]] },
      },
    ],
    controls: [
      { label: "Sun from", unit: "°", min: 0, max: 345, step: 15, view: "relief", op: 0, param: "azimuth" },
      { label: "Sun height", unit: "°", min: 1, max: 45, step: 1, view: "relief", op: 0, param: "altitude" },
      { label: "Flat: slope under", unit: "°", min: 2, max: 20, step: 1, layer: "Flat enough to land", variable: "flat", value: 8 },
    ],
    sites: { label: "Site", views: ["relief", "slope"], options: SITES.map(([id, name]) => ({ name, url: `${DEMS}/${id}/${id}.tif` })) },
    command: `chunkmirage serve '${RIDGE}' \\\n  --op hillshade:azimuth=135,altitude=10 --chunk 256,256 --python-viewer`,
  },
  {
    kind: "pipeline", id: "hurricanes", image: "cards/hurricanes.jpg",
    title: "Hurricanes' cold wakes: each day's change across 4 trillion sea temperatures",
    blurb: "NASA's daily sea-surface temperature of all the oceans since 2002, a kilometre apart: 4 trillion values, read straight from their public store. The change from the day before is computed for each chunk as you look, by chunkmirage's diff op running in this page. On 29 August 2005 Katrina leaves a cold swath across the Gulf of Mexico; scroll on through the days to Rita's, a month later. Left, the temperature (25 to 32 °C); right, its change since the day before (±2 °C).",
    data: "MUR sea-surface temperature, v4.1 (NASA JPL; AWS Open Data)",
    views: {
      sst: { source: MUR, chunk: [1, 256, 256] },
      change: { source: MUR, chunk: [1, 256, 256], ops: [{ op: "diff", axis: 0, lag: 1 }] },
    },
    layers: [
      { name: "temperature", view: "sst", type: "image", shader: KELVIN },
      { name: "change", view: "change", type: "image", shader: CHANGE },
    ],
    panels: [["temperature"], ["change"]],
    // Katrina's track across the Gulf, inside one of MUR's 18 x 36 degree tiles, so a new
    // 5-day stretch is one download
    position: [1185, 11699, 9249], zoom: 1.2,
    orientation: [1, 0, 0, 0],  // latitude increases northward: turn the map north up
    playback: { axis: "time", velocity: 1 },  // a day a second
    timeline: {
      axis: "time", start: "2002-06-01",
      events: [
        { from: "2005-08-23", to: "2005-08-24", text: "A tropical depression over the Bahamas becomes Katrina." },
        { from: "2005-08-25", text: "Katrina crosses southern Florida, a category 1 hurricane, into the Gulf." },
        { from: "2005-08-26", to: "2005-08-27", text: "Katrina strengthens over the Gulf's warm water." },
        { from: "2005-08-28", text: "Katrina reaches category 5 in the central Gulf." },
        { from: "2005-08-29", text: "Katrina makes landfall in Louisiana, a category 3 hurricane. Behind it, the cold water it stirred up from below." },
        { from: "2005-08-30", to: "2005-09-17", text: "Katrina's cold wake warms again." },
        { from: "2005-09-18", to: "2005-09-19", text: "Rita forms near the Turks and Caicos." },
        { from: "2005-09-20", text: "Rita passes the Florida Keys into the Gulf." },
        { from: "2005-09-21", to: "2005-09-22", text: "Rita reaches category 5 in the central Gulf." },
        { from: "2005-09-23", text: "Rita heads for the Texas-Louisiana border." },
        { from: "2005-09-24", text: "Rita makes landfall at the Texas-Louisiana border, a category 3 hurricane." },
        { from: "2005-10-21", to: "2005-10-22", text: "Wilma crosses the Yucatán Peninsula into the Gulf." },
        { from: "2005-10-23", to: "2005-10-24", text: "Wilma crosses the Gulf and makes landfall in southwest Florida (24 October)." },
      ],
    },
    command: `chunkmirage serve '${MUR}' --op diff:axis=0,lag=1 --chunk 1,256,256 --python-viewer`,
  },
  {
    kind: "python", id: "solar", image: "cards/solar.jpg",
    title: "A solar flare in the running difference, frame by frame (Python only)",
    blurb: "NASA's Solar Dynamics Observatory images the sun every 6 minutes: 73 thousand 512 x 512 frames for 2014 in one array of its public machine-learning dataset. Each frame minus the one before, the running difference solar physicists watch for what changes, is computed as the viewer asks: on 10 September 2014 an X1.6 flare erupts from the centre of the disk. Left, the sun at 171 Å; right, its change in 6 minutes.",
    data: "SDO machine-learning dataset v2, AIA 171 Å, 2014 (NASA FDL; NASA's open data bucket)",
    why: "NASA's bucket does not let a browser page read it (no CORS), so this one runs from Python.",
    command: "uv run python examples/solar_flares.py",
  },
];
