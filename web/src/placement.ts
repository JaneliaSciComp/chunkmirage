// Where a stored array sits, as worldarray (the Python reader chunkmirage shares) places it,
// so a volume opened in the browser lands where the server puts it: one entry per axis, C
// order, the origin voxel 0's centre. The documents are the format's own (a zarr array's
// attributes and its parent group's, N5 attributes.json); nothing here fetches, so
// web/test/metadata.test.ts runs it on worldarray's fixture of metadata layouts.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Attrs = Record<string, any>;

/** What a store says of an array, each null where it says nothing. */
export interface Found { axes: string[] | null; units: string[] | null; voxel: number[] | null; origin: number[] | null; placedBy: string | null }

/** An array's placement, every axis filled in: voxels of 1 with the corner at the origin
 * where nothing places it, the default axis names where none are given. */
export interface Placement { axes: string[]; named: boolean; units: string[]; voxel: number[]; origin: number[]; placedBy: string | null }

const DEFAULT_AXES = ["t", "c", "z", "y", "x"];

export function defaultAxes(n: number): string[] {
  return n <= DEFAULT_AXES.length ? DEFAULT_AXES.slice(DEFAULT_AXES.length - n)
    : [...Array.from({ length: n - DEFAULT_AXES.length }, (_, i) => `dim_${i}`), ...DEFAULT_AXES];
}

/** The axes named z, y or x; without any, the last three. */
export function spatialAxes(axes: string[]): number[] {
  const named = axes.flatMap((a, i) => (["z", "y", "x"].includes(a) ? [i] : []));
  return named.length ? named : axes.map((_, i) => i).slice(-3);
}

/** Voxel 0's centre from its corner: half a voxel on along the spatial axes. */
export function cornerToCentre(corner: number[], voxel: number[], axes: string[] | null): number[] {
  const sp = new Set(spatialAxes(axes ?? defaultAxes(corner.length)));
  return corner.map((c, i) => c + (sp.has(i) ? voxel[i] / 2 : 0));
}

/** One number per axis: a shorter list covers the trailing axes, the leading ones `pad`
 * (none: the length must match, as OME's must). */
function floats(values: unknown[], n: number, what: string, pad?: number): number[] {
  let v = values.map(Number);
  if (v.length < n && pad !== undefined) v = [...Array(n - v.length).fill(pad), ...v];
  if (v.length !== n) throw new Error(`${what} has ${v.length} entries for ${n} axes`);
  return v;
}

function strings(values: unknown[], n: number): string[] {
  const v = values.map((u) => (u == null ? "" : String(u)));
  return v.length < n ? [...Array(n - v.length).fill(""), ...v] : v;
}

/** Axis names, the leading ones defaulted when the list is short; null when empty or too long. */
function axisNames(names: unknown[] | undefined, n: number): string[] | null {
  if (!names?.length || names.length > n) return null;
  return [...defaultAxes(n).slice(0, n - names.length), ...names.map(String)];
}

const none = (axes: string[] | null = null): Found => ({ axes, units: null, voxel: null, origin: null, placedBy: null });

export function multiscalesOf(attrs: Attrs): Attrs[] {
  return attrs.ome?.multiscales ?? attrs.multiscales ?? [];
}

/** The multiscale and dataset listing the array `name`, by path (never by position). */
function datasetFor(group: Attrs, name: string): [Attrs, Attrs] | null {
  for (const ms of multiscalesOf(group)) {
    for (const ds of ms.datasets ?? []) if (String(ds.path ?? "").replace(/^\/+|\/+$/g, "") === name) return [ms, ds];
  }
  return null;
}

/** OME-NGFF 0.4 and 0.5: the dataset's transforms, then the multiscale's; 0.6
 * (coordinateSystems): the intrinsic system's axes, the dataset's scales and translations. */
function ome(ms: Attrs, ds: Attrs, n: number): Found {
  let transforms: Attrs[] = ds.coordinateTransformations ?? [], axes: Attrs[];
  if (ms.coordinateSystems?.length) {
    const wanted = transforms.find((t) => t.output)?.output;
    axes = (ms.coordinateSystems.find((s: Attrs) => wanted == null || s.name === wanted) ?? ms.coordinateSystems[0]).axes ?? [];
    const unpack = (ts: Attrs[]): Attrs[] => ts.flatMap((t) => (t.type === "sequence" ? unpack(t.transformations ?? []) : [t]));
    transforms = unpack(transforms);
  } else {
    axes = ms.axes ?? [];
    transforms = [...transforms, ...(ms.coordinateTransformations ?? [])];
  }
  let scale = Array(n).fill(1), shift = Array(n).fill(0);
  for (const t of transforms) {
    if (t.type === "scale") { const s = floats(t.scale, n, "OME scale"); scale = scale.map((v, i) => v * s[i]); shift = shift.map((v, i) => v * s[i]); }
    if (t.type === "translation") { const tr = floats(t.translation, n, "OME translation"); shift = shift.map((v, i) => v + tr[i]); }
  }
  const named = axes.length === n;
  return {
    axes: named ? axes.map((a) => String(typeof a === "string" ? a : a.name)) : null,
    units: named ? axes.map((a) => String((typeof a === "string" ? null : a.unit) ?? "")) : null,
    voxel: scale, origin: shift, placedBy: "ome",
  };
}

/** COSEM's transform: C order unless `ordering` is F; `translate` already a centre. */
function cosem(t: Attrs, n: number): Found {
  const order = <T>(v: T[]) => (t.ordering === "F" ? [...v].reverse() : v);
  return {
    axes: t.axes ? axisNames(order(t.axes), n) : null,
    units: t.units ? strings(order(t.units), n) : null,
    voxel: floats(order(t.scale), n, "transform.scale", 1),
    origin: floats(order(t.translate ?? []), n, "transform.translate", 0),
    placedBy: "cosem",
  };
}

const isList = (v: unknown) => Array.isArray(v);
const FUNLIB = ["voxel_size", "resolution", "offset", "units", "axis_names"];
const funlibOf = (attrs: Attrs) => Object.fromEntries(FUNLIB.filter((k) => k in attrs).map((k) => [k, attrs[k]]));

/** funlib (`voxel_size` or `resolution`, `offset` a corner, `units`, `axis_names`) or COSEM
 * (`transform`) attributes on the array, C order; an `offset` alone places it on voxels of 1. */
function arrayAttrs(attrs: Attrs, n: number): Found {
  if (attrs.transform?.scale) return cosem(attrs.transform, n);
  const axes = axisNames(attrs.axis_names, n);
  const vs = [attrs.voxel_size, attrs.resolution, attrs.scale].find((v) => v !== undefined);
  if (!isList(vs) && attrs.offset == null) return none(axes);
  const voxel = floats(isList(vs) ? vs : Array(n).fill(1), n, "voxel_size", 1);
  const units = attrs.units == null ? null : strings(isList(attrs.units) ? attrs.units : Array(n).fill(attrs.units), n);
  return { axes, units, voxel, origin: cornerToCentre(floats(attrs.offset ?? [], n, "offset", 0), voxel, axes), placedBy: "funlib" };
}

/** A zarr array: the parent group's OME multiscales listing it, else its own funlib or COSEM
 * attributes, else funlib's keys on the parent; its axes named by _ARRAY_DIMENSIONS or zarr
 * v3's dimension_names where nothing else names them. */
export function placeZarr(attrs: Attrs, group: Attrs, name: string, n: number, dimensionNames?: (string | null)[] | null): Found {
  const hit = datasetFor(group, name);
  let found = hit ? ome(hit[0], hit[1], n) : none();
  if (!found.voxel) {
    found = arrayAttrs(attrs, n);
    if (!found.voxel && ["voxel_size", "resolution", "offset"].some((k) => isList(group[k]))) {
      found = { ...arrayAttrs(funlibOf(group), n), axes: found.axes };
    }
  }
  const dims = attrs._ARRAY_DIMENSIONS ?? dimensionNames;
  if (!found.axes && dims?.length === n && dims.every(Boolean)) found.axes = dims.map(String);
  return found;
}

/** An N5 array: COSEM's transform on it, else the group's multiscales transform for it, else
 * funlib's `voxel_size` (array, else group), else `pixelResolution` or `resolution` (array,
 * else group; x first) times its `downsamplingFactors`, its corner at `offset` or, with none,
 * the origin at every level, as Neuroglancer draws N5. */
export function placeN5(attrs: Attrs, group: Attrs, name: string, n: number): Found {
  if (attrs.transform?.scale) return cosem(attrs.transform, n);
  const hit = datasetFor(group, name);
  if (hit?.[1].transform) return cosem(hit[1].transform, n);
  const axes = axisNames([...(attrs.axes ?? group.axes ?? [])].reverse(), n);
  const funlib = [attrs, group].find((a) => isList(a.voxel_size));
  if (funlib) { const f = arrayAttrs(funlibOf(funlib), n); return { ...f, axes: f.axes ?? axes }; }
  let base: unknown[] | undefined, unit: string | undefined;
  for (const a of [attrs, group]) {
    const pr = a.pixelResolution;
    if (pr !== undefined) { [base, unit] = isList(pr) ? [pr, undefined] : [pr.dimensions, pr.unit]; break; }
    if (a.resolution !== undefined) { base = a.resolution; break; }
  }
  if (!base) return none(axes);
  const f = floats([...(attrs.downsamplingFactors ?? [1])].reverse(), n, "downsamplingFactors", 1);
  const voxel = floats([...base].reverse(), n, "resolution", 1).map((v, i) => v * f[i]);
  let list = attrs.units ?? group.units;
  if (typeof list === "string") list = Array(n).fill(list);
  const sp = new Set(spatialAxes(axes ?? defaultAxes(n)));
  const units = unit === undefined && list ? strings([...list].reverse(), n) : voxel.map((_, i) => (sp.has(i) ? unit ?? "" : ""));
  const corner = attrs.offset ? floats([...attrs.offset].reverse(), n, "offset", 0) : Array(n).fill(0);
  return { axes, units, voxel, origin: cornerToCentre(corner, voxel, axes), placedBy: "n5" };
}

/** A precomputed scale, z, y, x in nanometres: `voxel_offset` is voxel 0's corner. */
export function placePrecomputed(scale: { resolution: number[]; voxel_offset?: number[] }): Found {
  const voxel = [...scale.resolution].reverse(), corner = [...(scale.voxel_offset ?? [0, 0, 0])].reverse();
  return { axes: ["z", "y", "x"], units: ["nm", "nm", "nm"], voxel, origin: corner.map((o, a) => (o + 0.5) * voxel[a]), placedBy: "precomputed" };
}

/** `found`, every axis filled in. */
export function placement(found: Found, n: number): Placement {
  const axes = found.axes ?? defaultAxes(n), voxel = found.voxel ?? Array(n).fill(1);
  return {
    axes, named: found.axes !== null, units: found.units ?? Array(n).fill(""), voxel,
    origin: found.origin ?? cornerToCentre(Array(n).fill(0), voxel, axes), placedBy: found.voxel ? found.placedBy : null,
  };
}
