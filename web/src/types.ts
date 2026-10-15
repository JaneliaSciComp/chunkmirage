// Shapes shared by the page, the solvers and the workers. Arrays are C order (z, y, x) and
// coordinates physical, as in chunkmirage.

/** Rows [A | t] of the affine from fixed to moving coordinates: three rows of four. */
export type Affine = number[][];

/** A displacement field on a control grid: (z, y, x) components interleaved, physical units. */
export interface ControlGrid {
  shape: number[];
  origin: number[];
  spacing: number[];
  values: Float32Array;
}

/** One level of an image as the solvers take it: values normalized to [0, 1]. */
export interface Volume {
  norm: Float32Array;
  shape: number[];
  voxel: number[];
  origin: number[];
}

/** A level's grid, which the chunk workers compute registered chunks on. */
export interface LevelGrid {
  shape: number[];
  voxel: number[];
  origin: number[];
}

export type ViewKind = "image" | "field";

/** The control lattice of a level whose field is fitted block by block where it is viewed. */
export interface Lattice { level: number; origin: number[]; spacing: number[]; shape: number[] }

/** How a fit is going: stage (level) of stages, iteration of iterations. */
export interface Progress { stage: number; stages: number; iteration: number; iterations: number; similarity: number | null }

/** What the page tells its chunk workers. */
export type ToWorker =
  | { type: "setup"; moving: string; chunkShape: number[]; fixedLevels: LevelGrid[] }
  | { type: "view"; id: string; kind: ViewKind; affine: Affine; grid: ControlGrid | null; refined?: Lattice[] }
  | { type: "drop"; ids: string[] }
  | { type: "chunk"; reqId: number; id: string; level: number; channel: number; index: number[] }
  | { type: "field"; reqId: number; grid: ControlGrid | null; error?: string };  // the page's answer to a worker's field request

/** What the chunk workers answer, and ask: the field of a refined level over a window of
 * its lattice, `lo` to `hi` (lattice indices), which the page fits block by block. */
export type FromWorker =
  | { type: "ready" }
  | { type: "chunk"; reqId: number; body: ArrayBuffer }
  | { type: "field"; reqId: number; chunk: number; id: string; level: number; lo: number[]; hi: number[] }  // chunk: the request it serves
  | { type: "error"; reqId?: number; message: string };

/** The service worker's store cache: data requests seen, answered from the cache (including
 * ones waiting on a fetch another request started), network fetches and their bytes. */
export interface StoreStats { requests: number; hits: number; fetches: number; fetchedBytes: number }

/** What the service worker asks a page for (relay.ts): a request's path (the URL's pathname
 * alone, what a route's prefix is matched against), its query string (`search`: the URL's,
 * with its leading "?", or "" when it has none; a page's API reads its parameters from it),
 * its range header, and, for a route that is more than a path, its method, headers and body. */
export interface Ask { path: string; search: string; range: string | null; method: string; headers: [string, string][]; body: ArrayBuffer | null }
/** The page's answer to a request the service worker relays, or null if it isn't the page's.
 * A chunk's answer comes in two parts: its head at once ({stream: true}), its body later as a
 * Later, so the service worker can stream it and tell the page ({cancel: true}) if the client
 * gives up first; a body that comes in pieces (an event stream) is Laters of `chunk` until
 * `done`. `headers`: any beyond the content type (an event stream's cache-control, say). */
export type Reply =
  | { status: number; body: string | ArrayBuffer; type: string; range?: string; headers?: [string, string][] }
  | { status: number; type: string; stream: true; range?: string; headers?: [string, string][] }
  | null;
export type Later = { body: ArrayBuffer } | { chunk: ArrayBuffer } | { done: true } | { error: string };

// ------------------------------------------------ the pipeline page, its reader and its Pyodide workers
/** One view a pipeline page serves: chunkmirage's PipelineSpec, as far as the browser goes. */
export interface PipelineView {
  source: string;                    // stack://, flip://, an OME-Zarr or N5 group, an xarray array
  select?: Record<string, number>;   // pin non-spatial axes, e.g. {c: 1, t: 0}
  ops?: Record<string, unknown>[];   // op specs, as the CLI and REST API take them
  chunk: number[];                   // output chunks, one per axis
  mesh?: { kind?: "surface" | "terrain"; level?: number; threshold?: number; exaggeration?: number; lods?: number };  // served at <view>/mesh
}
/** An axis of a view: its name (z, or time, lat, ...) and the unit of its voxel size. */
export interface ViewAxis { name: string; unit: string }
export interface ViewLevel { shape: number[]; voxel: number[]; origin: number[] }
/** A view's source as the reader opened it: three axes, and channels for a stack. */
export interface SourceInfo { dtype: string; channels: number; axes: ViewAxis[]; levels: ViewLevel[]; geo?: { bbox: number[] } }
/** A view as served: its source's axes and levels, and what its ops make of them. */
/** A view as served: its source, and what its ops make of it (output dtype, the leading axes
 * they consume and the halo they need; the leading axes they add, such as a gradient's
 * components, served as OME channel axes). */
export interface ViewInfo extends SourceInfo {
  halo: number[]; lead: number; out: string; added?: number[];
  kind?: string | null;                 // image, label or mask, if the ops say
  controls?: Record<string, Control[]>; // a python op's #uicontrol lines, by the op's index
}
/** A python op's `#uicontrol` line, as the op read it: a slider or a checkbox. */
export interface Control { name: string; type: "float" | "int" | "bool"; min: number | null; max: number | null; step: number | null; default: number | boolean }
/** What a worker makes of a view's ops: output dtype and kind, the leading axes they consume
 * and add, the halo they need, and their sliders. */
export interface Plan { dtype: string; kind: string | null; lead: number; halo: number[]; ndim: number; added: number[]; controls: Record<string, Control[]> }

/** A mesh made from a view's voxels: a legacy fragment, part of the coarsest level's mask
 * (for the multi-resolution octree), or a multi-resolution node's fragment. */
export type MeshCall = NonNullable<PipelineView["mesh"]> & {
  mode: "legacy" | "mask" | "node"; levels?: ViewLevel[]; size?: number; bits?: number;
  core_lo?: number[]; core_hi?: number[];  // mask: the part's core within the box asked for
  raw?: boolean;                           // node: the data is the coarsest level's mask, no ops to run
};
/** What the page asks the reader, which opens the views' sources once for the page. */
export type ToReader =
  | { type: "open"; reqId: number; views: Record<string, PipelineView> }
  | { type: "read"; reqId: number; view: string; level: number; lo: number[]; hi: number[]; at?: Record<string, number> }
  | { type: "sample"; reqId: number; view: string; ps: number[] };
/** What the page asks a Pyodide worker. */
export type ToPyWorker =
  | { type: "describe"; reqId: number; source: string }  // a source computed in the worker (synthetic://)
  | { type: "plan"; reqId: number; packages?: string[]; views: Record<string, { ops: Record<string, unknown>[]; shape: number[]; dtype: string; chunk: number[]; voxel: number[]; source?: string }> }
  // data: the input region read by the reader, or null for a source the worker computes
  | { type: "compute"; reqId: number; view: string; level: number; data: ArrayBuffer | null; readShape: number[]; inLo: number[]; inHi: number[]; outLo: number[]; outHi: number[]; full: number[]; voxel: number[]; origin: number[]; unit?: string; mesh?: MeshCall }
  // a multi-resolution mesh's octree and index, from its coarsest level's surface band (`mesh.levels` the view's)
  | { type: "octree"; reqId: number; band: ArrayBuffer; shape: number[]; mesh: MeshCall; chunk: number[]; unit: string }
  // a step of a page's work (chunkmirage.stitching, chunkmirage.tracking): JSON arguments, arrays as raw bytes
  | { type: "call"; reqId: number; fn: string; args: string; arrays?: ArrayBuffer[] };
/** Either's answer to request `reqId`. */
/** A line a view's Python printed while a worker computed one of its chunks (`at`: the
 * chunk's first voxel). */
export type Printed = { print: string; view: string; level: number; at: number[] };
export type Answer = { reqId: number; value: unknown } | { reqId: number; error: string } | Printed;
