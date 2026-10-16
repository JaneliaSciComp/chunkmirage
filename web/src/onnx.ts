// An ONNX Runtime Web stage for a view the page computes in stages (stages.ts): a model's
// forward pass on each input block, on the GPU through ORT's WebGPU execution provider when
// the browser has one, else its WebAssembly one (the CPU, many times slower). Nothing of a
// model's meaning is here, neither what its input is normalized to nor what its output
// means: a page puts those in stages of its own before and after this one, and whatever
// describes its models (a manifest, a catalog) stays with the page. Sessions are built once
// per model and shared by the views that use it (`retain` and `release`), and one forward
// pass runs at a time on the page, whatever its session (`exclusive`). ORT's WebAssembly runtime is fetched from `wasmPaths`,
// ort/ beside the page by default, where the Vite plugin in web/vite-ort.ts serves it.
import * as ort from "onnxruntime-web/webgpu";
import { asReals, type Numbers } from "./asread.ts";  // with its extension: Node's runner (npm test) resolves no other
import { prod, type Block, type Stage } from "./stages.ts";  // with its extension: Node's runner (npm test) resolves no other

/** A model: its bytes when first needed (the page keeps them as it likes), under a key that
 * is its identity (its file's URL or hash); the tensors to feed and read, the session's
 * first of each by default. */
export interface OnnxModel { key: string; bytes: () => Promise<ArrayBuffer>; input?: string; output?: string }
export type Provider = "webgpu" | "wasm";
/** What the GPU allows, for a page choosing among a model's sizes: nulls without WebGPU. */
export interface GpuLimits { maxBufferSize: number | null; maxStorageBufferBindingSize: number | null; shaderF16: boolean }
/** A forward pass's output tensor: its values, dims and type, and the pass's time alone. */
export interface Output { data: Numbers; dims: number[]; type: string; ms: number }

interface Session { session: ort.InferenceSession; provider: Provider; input: string; output: string }

let turn: Promise<unknown> = Promise.resolve();  // the last pass started on this page's ORT

/** `pass` run after every pass started before it on this page's ORT, whatever its session or
 * Onnx; its result. ORT Web runs all its sessions in one WebAssembly instance, and its WebGPU
 * build waits on the GPU with Emscripten's Asyncify, which allows one suspended call per
 * instance: a run started while another waits breaks both. Two runs of one session never
 * resolved, and a page serving a model beside its finetuned copy, their runs free to overlap,
 * hung and then crashed while both computed. So the passes take turns, and anything else running a session
 * on the same ORT (a page's own training passes, say) takes its turn here too. On WASM alone
 * the runs are synchronous and never overlap anyway, so the turn costs nothing there. */
export function exclusive<T>(pass: () => Promise<T>): Promise<T> {
  const mine = turn.then(pass);
  turn = mine.catch(() => {});  // the next pass waits for this one, however it ends
  return mine;
}
/** `wasmPaths`: where ORT's WebAssembly runtime is served from, ort/ beside the page by
 * default (web/vite-ort.ts puts it there); `log` hears what the sessions do. */
export interface OnnxOptions { wasmPaths?: string; log?: (line: string) => void }

export class Onnx {
  private sessions = new Map<string, Promise<Session>>();
  private users = new Map<string, number>();  // views using each session
  private adapter: Promise<GPUAdapter | null> | null = null;
  private configured = false;
  /** The provider of the first session built: WebGPU, or WASM (the CPU). */
  provider: Provider | null = null;

  private options: OnnxOptions;

  constructor(options: OnnxOptions = {}) { this.options = options; }

  private log(line: string) { this.options.log?.(line); }

  /** ORT's settings, once, before the first session. */
  private configure() {
    if (this.configured) return;
    this.configured = true;
    // the runtime from where it is served: as given, else ort/ beside the page; in Node (the
    // tests) nothing is set and ORT reads it from its own package
    if (this.options.wasmPaths) ort.env.wasm.wasmPaths = this.options.wasmPaths;
    else if (typeof document !== "undefined") ort.env.wasm.wasmPaths = new URL("ort/", document.baseURI).href;
    // threads need SharedArrayBuffer, which needs cross-origin isolation (COOP/COEP headers
    // a static host may not set); without it ORT would fall back to one thread anyway
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 1;
  }

  /** The session for `model`, built from its bytes on first use: on WebGPU if the browser
   * has it and the session can be created there, else on WASM. One that failed to build is
   * forgotten, so the next block tries again. */
  session(model: OnnxModel): Promise<Session> {
    let pending = this.sessions.get(model.key);
    if (!pending) {
      pending = this.build(model).catch((e) => { this.sessions.delete(model.key); throw e; });
      this.sessions.set(model.key, pending);
    }
    return pending;
  }

  private async build(model: OnnxModel): Promise<Session> {
    this.configure();
    const bytes = new Uint8Array(await model.bytes());
    let session: ort.InferenceSession | null = null, provider: Provider = "wasm";
    if (navigator.gpu) {
      try { session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }); provider = "webgpu"; }
      catch (e) { this.log(`${model.key}: no WebGPU session (${(e as Error).message ?? e}); on WASM (the CPU) instead`); }
    } else this.log("navigator.gpu is unavailable: the model runs on WASM (the CPU), which is slow");
    session ??= await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    this.provider ??= provider;
    const input = model.input ?? session.inputNames[0], output = model.output ?? session.outputNames[0];
    if (!session.inputNames.includes(input)) throw new Error(`${model.key} has no input ${input} (it has ${session.inputNames.join(", ")})`);
    if (!session.outputNames.includes(output)) throw new Error(`${model.key} has no output ${output} (it has ${session.outputNames.join(", ")})`);
    this.log(`${model.key}: session ready on ${provider === "webgpu" ? "WebGPU" : "WASM (the CPU)"}`);
    return { session, provider, input, output };
  }

  /** One forward pass of `model` on `data` of `dims` (float32), after the session's other
   * passes on the page (`exclusive`): the output tensor, and the pass's time alone, not the
   * wait for its turn. */
  async run(model: OnnxModel, data: Float32Array, dims: number[]): Promise<Output> {
    const s = await this.session(model);
    return exclusive(async () => {
      const t0 = performance.now();
      const results = await s.session.run({ [s.input]: new ort.Tensor("float32", data, dims) });
      const out = results[s.output];
      return { data: out.data as Numbers, dims: [...out.dims], type: out.type, ms: performance.now() - t0 };
    });
  }

  /** A view starts using `key`'s session. */
  retain(key: string) { this.users.set(key, (this.users.get(key) ?? 0) + 1); }

  /** A view that retained `key`'s session is done with it: the last one releases it. A
   * release nobody retained is nothing (a session goes only by its users or `reset`), so a
   * view's stray release cannot take a session from under another's pass. */
  async release(key: string): Promise<void> {
    const users = this.users.get(key);
    if (users === undefined) return;
    if (users > 1) { this.users.set(key, users - 1); return; }
    this.users.delete(key);
    const pending = this.sessions.get(key);
    this.sessions.delete(key);
    if (pending) await released(pending);
  }

  /** Every session released and forgotten; the next block builds its session again. The
   * WebGPU device is ORT's: it makes the one its backend uses when the first session is
   * built and no other if that one is destroyed, so this recovers from a session in a bad
   * state, while a pass that hangs the GPU's queue needs the page reloaded. */
  async reset(): Promise<void> {
    const pending = [...this.sessions.values()];
    this.sessions.clear();
    this.users.clear();
    await Promise.all(pending.map(released));
  }

  /** The WebGPU adapter's limits a page chooses a model's size by; nulls without WebGPU. */
  async limits(): Promise<GpuLimits> {
    const adapter = await this.gpu();
    if (!adapter) return { maxBufferSize: null, maxStorageBufferBindingSize: null, shaderF16: false };
    return { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, shaderF16: adapter.features.has("shader-f16") };
  }

  /** The GPU as its adapter describes it, or null without WebGPU. */
  async device(): Promise<string | null> {
    const adapter = await this.gpu();
    if (!adapter) return null;
    const info = adapter.info as Partial<GPUAdapterInfo> | undefined;
    return [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" ").trim() || "WebGPU adapter";
  }

  private gpu(): Promise<GPUAdapter | null> {
    return this.adapter ??= navigator.gpu ? navigator.gpu.requestAdapter().catch(() => null) : Promise.resolve(null);
  }
}

/** A session released, not waiting for ever on one whose pass hangs (why `reset` is
 * called), nor failing on one that never got built. */
async function released(pending: Promise<Session>): Promise<void> {
  try {
    const s = await pending;
    await Promise.race([s.session.release(), new Promise<void>((r) => setTimeout(r, 3000))]);
  } catch { /* a session that failed to build has nothing to release */ }
}

/** How a model is run as a stage: the stage's `key` (the model's by default), `halo` (the
 * model's context, in input voxels), `inputVoxelSize`, the output tensor's `dtype`
 * (float32 by default; the stage fails on another) and `added`, the channel axis its output
 * has: `[C]` for a model writing C channels, none for one writing one. */
export interface OnnxStageOptions { key?: string; halo?: number[]; inputVoxelSize?: number[]; dtype?: string; added?: number[] }

/** A stage running `model` on each block: the block as a float32 tensor of dims
 * `[1, C, z, y, x]` (C the product of the block's leading axes, 1 if none), the output
 * tensor as the block the model wrote, `[C, z, y, x]` with a batch of one dropped (rank 5),
 * as it is (rank 4) or `[z, y, x]` (rank 3), then the stage's `added` axes or none: a model
 * writing one channel serves it as a plain volume unless `added: [1]` says otherwise. */
export function onnxStage(onnx: Onnx, model: OnnxModel, options: OnnxStageOptions = {}): Stage {
  const dtype = options.dtype ?? "float32", added = options.added ?? [];
  return {
    key: options.key ?? model.key, halo: options.halo, inputVoxelSize: options.inputVoxelSize, dtype, added,
    async produce(input: Block): Promise<Block> {
      const lead = prod(input.shape.slice(0, -3));
      const data = input.data instanceof Float32Array ? input.data : Float32Array.from(asReals(input.data));  // a 64-bit integer block's bigints as numbers
      const out = await onnx.run(model, data, [1, lead, ...input.shape.slice(-3)]);
      if (out.type !== dtype) throw new Error(`${model.key} wrote ${out.type}, not the ${dtype} its stage declares`);
      let shape = out.dims;
      if (shape.length === 5) {
        if (shape[0] !== 1) throw new Error(`${model.key} wrote a batch of ${shape[0]}`);
        shape = shape.slice(1);
      }
      if (shape.length < 3 || shape.length > 4) throw new Error(`${model.key} wrote a tensor of rank ${out.dims.length} (${out.dims.join("×")}), not a volume`);
      const channels = shape.length === 4 ? shape[0] : 1, spatial = shape.slice(-3);
      if (added.length ? prod(added) !== channels : channels !== 1) {
        throw new Error(`${model.key} wrote ${channels} channel${channels === 1 ? "" : "s"}, its stage declares ${added.length ? added.join("×") : "none"}`);
      }
      return { data: out.data, shape: [...added, ...spatial] };
    },
  };
}
