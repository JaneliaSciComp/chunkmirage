// A view's OME-Zarr layouts, as the engine serves them (engine.ts's routes) and writes them
// into a store (materialize.ts), from one place so the two cannot differ: zarr v3 with
// OME-Zarr 0.5, and zarr v2 with OME-Zarr 0.4. The group's metadata, a level's array
// metadata (or a box of the level's, which materialize writes) and a chunk's key, as
// objects ready for JSON. A channel axis the view's ops add (`added`) comes first and is
// one chunk, as the routes serve it. Nothing here imports a value, so Node's tests run it.
import type { ViewAxis, ViewInfo } from "./types.ts";

export const TO_SECONDS: Record<string, number> = { s: 1, second: 1, millisecond: 1e-3, ms: 1e-3, minute: 60, hour: 3600, day: 86400 };
const OME_UNIT: Record<string, string> = { nm: "nanometer", um: "micrometer", m: "meter", s: "second" };
/** zarr v2's dtypes, by numpy's name (the engine's dtypes); little endian. */
export const V2_DTYPE: Record<string, string> = {
  uint8: "|u1", int8: "|i1", uint16: "<u2", int16: "<i2", uint32: "<u4", int32: "<i4", uint64: "<u8", int64: "<i8", float32: "<f4", float64: "<f8",
};
export const isTime = (a: ViewAxis) => a.unit in TO_SECONDS;

/** A level of a view as one of a group's datasets: its path in the group, and which of the
 * view's levels it is; `lo` (a box of the level, materialize.ts): the level's voxel that is
 * the dataset's voxel 0, on the level's own axes (none: the level's voxel 0). */
export interface Dataset { path: string; level: number; lo?: number[] }

/** A zarr v3 codec after `bytes` (a compressor). */
export interface Codec { name: string; configuration: Record<string, unknown> }

/** Where voxel `lo` of a level of view `v` has its centre (the level's origin when `lo` is
 * left out): the origin shifted by `lo` voxels, on the view's own axes. */
export const translation = (v: ViewInfo, level: number, lo?: number[]) => {
  const { origin, voxel } = v.levels[level];
  return lo ? origin.map((o, a) => o + lo[a] * voxel[a]) : origin;
};

/** The OME multiscales entry of view `name`, whose datasets are the levels `datasets` name:
 * the channel axes its ops add, then its own (space, or time by their unit); each level's
 * scale its voxel size and its translation its origin (where voxel 0's centre is), so it
 * lands where the view does; a dataset that starts at voxel `lo` of its level is translated
 * by `lo` voxels more, so its voxel 0 lands on the level's voxel `lo`. */
export function multiscale(name: string, v: ViewInfo, datasets: Dataset[]) {
  const added = (v.added ?? []).map(() => 0);
  return {
    name,
    axes: [...added.map(() => ({ name: "c", type: "channel" })),
      ...v.axes.map((a) => ({ name: a.name, type: isTime(a) ? "time" : "space", unit: OME_UNIT[a.unit] ?? (a.unit || undefined) }))],
    datasets: datasets.map(({ path, level, lo }) => ({ path, coordinateTransformations: [
      { type: "scale", scale: [...added.map(() => 1), ...v.levels[level].voxel] },
      { type: "translation", translation: [...added, ...translation(v, level, lo)] },
    ] })),
  };
}

/** A view's zarr v3 group (zarr.json): OME-Zarr 0.5's multiscales in its attributes, beside
 * any `extra` attributes (which cannot replace `ome`). */
export function group3(name: string, v: ViewInfo, datasets: Dataset[], extra: Record<string, unknown> = {}) {
  return { zarr_format: 3, node_type: "group", attributes: { ...extra, ome: { version: "0.5", multiscales: [multiscale(name, v, datasets)] } } };
}

/** A view's zarr v2 group attributes (.zattrs): OME-Zarr 0.4's multiscales, beside any
 * `extra` attributes (which cannot replace `multiscales`). The group itself is
 * `{"zarr_format": 2}` (.zgroup). */
export function attrs2(name: string, v: ViewInfo, datasets: Dataset[], extra: Record<string, unknown> = {}) {
  return { ...extra, multiscales: [{ ...multiscale(name, v, datasets), version: "0.4" }] };
}

/** The names of a level's axes: "c" for each channel axis the ops add, then the view's. */
export const dimensionNames = (v: ViewInfo) => [...(v.added ?? []).map(() => "c"), ...v.axes.map((a) => a.name)];

/** A level's zarr v3 array (its zarr.json), chunked `chunk` (the level's own axes), the
 * channel axes the ops add one chunk each; its bytes little endian, then `codecs`; `shape`
 * on the level's own axes (a box of it), the level's by default. In the GeoZarr layout
 * (`map`), the map's y, x band alone, the view's one plane along z (a map client reads 2-D
 * bands: with a third axis it would need the group's consolidated metadata to know to
 * select it). */
export function array3(v: ViewInfo, chunk: number[], level: number,
  { map = false, codecs = [], shape = v.levels[level].shape }: { map?: boolean; codecs?: Codec[]; shape?: number[] } = {}) {
  const keep = map ? -2 : 0, added = map ? [] : v.added ?? [];
  return {
    zarr_format: 3, node_type: "array", shape: [...added, ...shape.slice(keep)], data_type: v.out, fill_value: 0,
    chunk_grid: { name: "regular", configuration: { chunk_shape: [...added, ...chunk.slice(keep)] } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }, ...codecs],
    dimension_names: map ? v.axes.map((a) => a.name).slice(keep) : dimensionNames(v), attributes: {},
  };
}

/** A level's zarr v2 array (.zarray), chunked `chunk`, the channel axes the ops add one
 * chunk each, compressed by `compressor` (numcodecs' config; null: as they are); `shape` on
 * the level's own axes (a box of it), the level's by default. Its axes' names are its
 * .zattrs, `{"_ARRAY_DIMENSIONS": dimensionNames(v)}`. */
export function array2(v: ViewInfo, chunk: number[], level: number, compressor: Record<string, unknown> | null = null, shape = v.levels[level].shape) {
  const added = v.added ?? [];
  return {
    zarr_format: 2, shape: [...added, ...shape], chunks: [...added, ...chunk], dtype: V2_DTYPE[v.out] ?? v.out,
    compressor, fill_value: 0, order: "C", filters: null, dimension_separator: "/",
  };
}

/** The key of chunk `index` (on the level's own axes) of the array at `path`: zarr v3's
 * default encoding (`c/`, then the indices "/" apart), or zarr v2's with "/" as its
 * dimension separator; each of the `added` channel axes is one chunk, index 0. */
export function chunkKey(format: "zarr3" | "zarr2", path: string, added: number, index: number[]): string {
  const at = [...Array<number>(added).fill(0), ...index].join("/");
  return format === "zarr3" ? `${path}/c/${at}` : `${path}/${at}`;
}
