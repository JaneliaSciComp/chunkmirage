// A view served as read: no ops over a source the page's reader reads. It needs no Python.
// Its plan is its source's (`asRead`: the dtype as stored, no halo, a channel axis served as
// one), and each chunk is the region the reader read placed whole in a zarr chunk
// (`wholeChunk`, zero past the array's edge), the bytes a Pyodide worker makes of it with no
// ops to run (fused.run with none, then the padding compute in pyworker.ts does). So a page
// whose views are all read as stored, or computed by the page itself (Engine.serve), never
// loads Pyodide; `needsPython` says which views do.
import type { PipelineView, Plan, SourceInfo } from "./types";

/** Sources the Pyodide workers compute themselves (chunkmirage's own Python), nothing read. */
export const computedSource = (url: string) => url.startsWith("synthetic://");

/** Whether a view's chunks need the Pyodide workers: ops to run, a source they compute, or
 * a mesh to make (marching cubes). */
export const needsPython = (spec: PipelineView) => (spec.ops?.length ?? 0) > 0 || computedSource(spec.source) || !!spec.mesh;

/** The plan of a view with no ops, as fused.plan gives one: the source's dtype, no halo and
 * no kind. A source with channels (stack://) has them as one leading axis, read whole
 * (`lead`) and served as a channel axis (`added`), as the workers plan it. */
export function asRead(source: SourceInfo): Plan {
  const channels = source.channels > 1 ? [source.channels] : [];
  return { dtype: source.dtype, kind: null, lead: channels.length, halo: [0, 0, 0], ndim: channels.length + 3, added: channels, controls: {} };
}

/** The typed arrays of the dtypes the reader gives (numpy's names); any other dtype is read
 * as float32 (ome.ts). The 64-bit integers (a segmentation's labels, past 2^53 as numpy's
 * uint64 goes) are BigUint64Array and BigInt64Array, whose elements are bigints, not
 * numbers: code over `Numbers` that does arithmetic on an element narrows to the kinds it
 * takes, or converts (`Number(v)`). Here, in a module with no other imports, so the pure
 * parts of the engine (this, stages.ts) run in Node's tests. */
export type Numbers = Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | BigUint64Array | BigInt64Array | Float32Array | Float64Array;
export type TypedCtor = { new (n: number): Numbers & { buffer: ArrayBuffer }; new (buffer: ArrayBuffer): Numbers; BYTES_PER_ELEMENT: number };
export const TYPED: Record<string, TypedCtor> = {
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, uint64: BigUint64Array, int8: Int8Array, int16: Int16Array, int32: Int32Array, int64: BigInt64Array,
  float32: Float32Array, float64: Float64Array,
};

/** The kinds of `Numbers` whose elements are numbers: all but the 64-bit integers. */
export type Reals = Exclude<Numbers, BigUint64Array | BigInt64Array>;

/** `data` with numbers for elements, for arithmetic over it (display limits, intensities to
 * register): itself, or for a 64-bit integer kind its bigints converted, exact to 2^53 and
 * the nearest double past it, so for intensities, not for ids. */
export function asReals(data: Numbers): Reals {
  return data instanceof BigUint64Array || data instanceof BigInt64Array ? Float64Array.from(data, Number) : data;
}

/** `set` and `fill` on a typed array of any kind with elements of its own kind, numbers or
 * bigints. Code that makes its output of its input's dtype (`padded` and `crop` in stages.ts,
 * the readers assembling a region from pieces) knows the two are of one kind, which
 * TypeScript cannot see across the union, where each kind's methods take its own elements
 * alone; such code casts its output to this (`as unknown as Settable`) and sets it from its
 * input as it is. */
export type Settable = { set(values: ArrayLike<number | bigint>, at: number): void; fill(v: number | bigint, from: number, to: number): void };

/** A whole zarr chunk of `chunk` voxels (the channels first, if more than one) from the
 * region the reader read: `read` voxels of each channel one after another, in `dtype`. The
 * region fills the chunk from its first voxel; a chunk at the array's edge is zero past it,
 * as a server pads it with the fill value. A chunk inside the array is the region as read. */
export function wholeChunk(data: ArrayBuffer, dtype: string, channels: number, read: number[], chunk: number[]): ArrayBuffer {
  if (read.every((d, a) => d === chunk[a])) return data;
  const width = (TYPED[dtype] ?? Float32Array).BYTES_PER_ELEMENT, bytes = new Uint8Array(data);
  const [d0, d1, d2] = read, [c0, c1, c2] = chunk, row = d2 * width;
  const out = new Uint8Array(channels * c0 * c1 * c2 * width);
  for (let c = 0; c < channels; c++) for (let z = 0; z < d0; z++) for (let y = 0; y < d1; y++) {
    const from = ((c * d0 + z) * d1 + y) * row;
    out.set(bytes.subarray(from, from + row), ((c * c0 + z) * c1 + y) * c2 * width);
  }
  return out.buffer;
}
