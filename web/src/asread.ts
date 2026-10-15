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

/** Bytes a voxel of each dtype the reader gives (ome.ts's TYPED; anything else is read as float32). */
const BYTES: Record<string, number> = { uint8: 1, int8: 1, uint16: 2, int16: 2, uint32: 4, int32: 4, float32: 4, float64: 8 };

/** A whole zarr chunk of `chunk` voxels (the channels first, if more than one) from the
 * region the reader read: `read` voxels of each channel one after another, in `dtype`. The
 * region fills the chunk from its first voxel; a chunk at the array's edge is zero past it,
 * as a server pads it with the fill value. A chunk inside the array is the region as read. */
export function wholeChunk(data: ArrayBuffer, dtype: string, channels: number, read: number[], chunk: number[]): ArrayBuffer {
  if (read.every((d, a) => d === chunk[a])) return data;
  const width = BYTES[dtype] ?? 4, bytes = new Uint8Array(data);
  const [d0, d1, d2] = read, [c0, c1, c2] = chunk, row = d2 * width;
  const out = new Uint8Array(channels * c0 * c1 * c2 * width);
  for (let c = 0; c < channels; c++) for (let z = 0; z < d0; z++) for (let y = 0; y < d1; y++) {
    const from = ((c * d0 + z) * d1 + y) * row;
    out.set(bytes.subarray(from, from + row), ((c * c0 + z) * c1 + y) * c2 * width);
  }
  return out.buffer;
}
