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

/** The page's answer to a request the service worker relays, or null if it isn't the page's.
 * A chunk's answer comes in two parts: its head at once ({stream: true}), its body later as a
 * Later, so the service worker can stream it and tell the page ({cancel: true}) if the client
 * gives up first. */
export type Reply = { status: number; body: string | ArrayBuffer; type: string } | { status: number; type: string; stream: true } | null;
export type Later = { body: ArrayBuffer } | { error: string };

// ------------------------------------------------ the pipeline page, its reader and its Pyodide workers
/** One view a pipeline page serves: chunkmirage's PipelineSpec, as far as the browser goes. */
export interface PipelineView {
  source: string;                    // stack://, flip://, an OME-Zarr or N5 group, an xarray array
  select?: Record<string, number>;   // pin non-spatial axes, e.g. {c: 1, t: 0}
  ops?: Record<string, unknown>[];   // op specs, as the CLI and REST API take them
  chunk: number[];                   // output chunks, one per axis
}
/** An axis of a view: its name (z, or time, lat, ...) and the unit of its voxel size. */
export interface ViewAxis { name: string; unit: string }
export interface ViewLevel { shape: number[]; voxel: number[]; origin: number[] }
/** A view's source as the reader opened it: three axes, and channels for a stack. */
export interface SourceInfo { dtype: string; channels: number; axes: ViewAxis[]; levels: ViewLevel[] }
/** A view as served: its source's axes and levels, and what its ops make of them. */
export interface ViewInfo extends SourceInfo { halo: number[]; lead: number; out: string }

/** What the page asks the reader, which opens the views' sources once for the page. */
export type ToReader =
  | { type: "open"; reqId: number; views: Record<string, PipelineView> }
  | { type: "read"; reqId: number; view: string; level: number; lo: number[]; hi: number[] }
  | { type: "sample"; reqId: number; view: string; ps: number[] };
/** What the page asks a Pyodide worker. */
export type ToPyWorker =
  | { type: "plan"; reqId: number; views: Record<string, { ops: Record<string, unknown>[]; shape: number[]; dtype: string; chunk: number[] }> }
  | { type: "compute"; reqId: number; view: string; data: ArrayBuffer; readShape: number[]; inLo: number[]; inHi: number[]; outLo: number[]; outHi: number[]; full: number[] };
/** Either's answer to request `reqId`. */
export type Answer = { reqId: number; value: unknown } | { reqId: number; error: string };
