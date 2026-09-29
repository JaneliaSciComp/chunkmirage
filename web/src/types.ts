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

/** How a fit is going: stage (level) of stages, iteration of iterations. */
export interface Progress { stage: number; stages: number; iteration: number; iterations: number; similarity: number | null }

/** What the page tells its chunk workers. */
export type ToWorker =
  | { type: "setup"; moving: string; chunkShape: number[]; fixedLevels: LevelGrid[] }
  | { type: "view"; id: string; kind: ViewKind; affine: Affine; grid: ControlGrid | null }
  | { type: "drop"; ids: string[] }
  | { type: "chunk"; reqId: number; id: string; level: number; channel: number; index: number[] };

/** What the chunk workers answer. */
export type FromWorker =
  | { type: "ready" }
  | { type: "chunk"; reqId: number; body: ArrayBuffer }
  | { type: "error"; reqId?: number; message: string };

/** The page's answer to a request the service worker relays, or null if it isn't the page's. */
export type Reply = { status: number; body: string | ArrayBuffer; type: string } | null;
