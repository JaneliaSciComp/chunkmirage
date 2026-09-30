// The deformable registration solve on the GPU, in WebGPU compute shaders: the same method,
// defaults and conventions as chunkmirage.registration (the PyTorch solver behind
// register://), so the two can be checked against each other. The field u lives in the
// fixed image's physical space on a control grid; the moving image is sampled at
// affine(p + u(p)). Gradients are written out by hand: the closed-form gradient of the
// squared local correlation (box sums of three per-window factors), then the chain rule
// through trilinear sampling; WGSL has no float atomics, so each control point gathers
// its voxels' gradients rather than voxels scattering into control points.
import type { RegisterParams } from "./generated/chunkmirage";
import schema from "./generated/chunkmirage.schema.json";
import { fieldAt, prod } from "./ome";
import type { Affine, ControlGrid, Progress, Volume } from "./types";

const WG = 256;
const FLAT = 1e-4;  // windows with less variance than this carry no signal (registration.FLAT)
const WINDOW: number = schema.$defs.RegisterParams.properties.window.default;  // correlation window, voxels
const STEP = 0.5;   // Adam learning rate, voxels of each level (registration.Settings.step)

/** The page solves with one iteration count for every level. */
export type Settings = Pick<Required<RegisterParams>, "smooth" | "grid"> & { iterations: number };
export interface Stage { shape: number[]; grid: number[]; seconds: number; first: number | null; final: number | null }
export interface Gpu { device: GPUDevice; pipelines: Record<Kernel, GPUComputePipeline>; name: string }
type Kernel = keyof typeof KERNELS;

const COMMON = /* wgsl */ `
struct Params {
  fshape: vec4<u32>, mshape: vec4<u32>, gshape: vec4<u32>,   // (z, y, x, count)
  vf: vec4<f32>, tf: vec4<f32>, vm: vec4<f32>, tm: vec4<f32>,  // voxel sizes and origins
  lo: vec4<f32>, sp: vec4<f32>,                                // control grid origin, spacing
  a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>,                 // affine rows [A | t]
  misc: vec4<f32>,  // flat, 1/N, lambda, window half-width
  reg: vec4<f32>,   // 2 / (differences_a * spacing_a^2), per axis
  adam: vec4<f32>,  // lr, 1/(1-b1^t), 1/(1-b2^t)
};
@group(0) @binding(0) var<uniform> P: Params;
fn voxel(p: u32) -> vec3<u32> {
  let W = P.fshape.z; let H = P.fshape.y;
  return vec3<u32>(p / (W * H), (p / W) % H, p % W);
}
fn phys(v: vec3<u32>) -> vec3<f32> { return P.tf.xyz + vec3<f32>(v) * P.vf.xyz; }
fn to_moving(q: vec3<f32>) -> vec3<f32> {
  let m = vec3<f32>(dot(P.a0.xyz, q) + P.a0.w, dot(P.a1.xyz, q) + P.a1.w, dot(P.a2.xyz, q) + P.a2.w);
  return (m - P.tm.xyz) / P.vm.xyz;
}
fn window_count(p: u32) -> f32 {
  let h = i32(P.misc.w);
  let c = vec3<i32>(voxel(p)); let n = vec3<i32>(P.fshape.xyz);
  let e = min(c + vec3<i32>(h), n - vec3<i32>(1)) - max(c - vec3<i32>(h), vec3<i32>(0)) + vec3<i32>(1);
  return f32(e.x * e.y * e.z);
}
`;
const ENTRY = (body: string) => `
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let p = (wid.y * nwg.x + wid.x) * ${WG}u + li;
  ${body}
}`;

const KERNELS = {
  // Moving image at affine(p + u(p)) (0 beyond it), and dJ/du's per-voxel factor.
  warp: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> J: array<f32>;
@group(0) @binding(4) var<storage, read_write> C: array<f32>;
fn at_u(q: vec3<u32>) -> vec3<f32> {
  let i = ((q.x * P.gshape.y + q.y) * P.gshape.z + q.z) * 3u;
  return vec3<f32>(U[i], U[i + 1u], U[i + 2u]);
}
fn field(g: vec3<f32>) -> vec3<f32> {
  let top = vec3<f32>(P.gshape.xyz) - 1.0;
  let g0 = min(floor(g), max(top - 1.0, vec3<f32>(0.0)));
  let f = g - g0; let i = vec3<u32>(g0);
  var d = vec3<f32>(0.0);
  for (var c = 0u; c < 8u; c++) {
    let o = vec3<u32>(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
    let w = select(1.0 - f.x, f.x, o.x == 1u) * select(1.0 - f.y, f.y, o.y == 1u) * select(1.0 - f.z, f.z, o.z == 1u);
    d += w * at_u(min(i + o, P.gshape.xyz - vec3<u32>(1u)));
  }
  return d;
}
fn mov(q: vec3<i32>) -> f32 {
  let n = vec3<i32>(P.mshape.xyz);
  if (any(q < vec3<i32>(0)) || any(q >= n)) { return 0.0; }
  return M[(u32(q.x) * P.mshape.y + u32(q.y)) * P.mshape.z + u32(q.z)];
}` + ENTRY(/* wgsl */ `
  if (p >= P.fshape.w) { return; }
  let x = phys(voxel(p));
  let g = clamp((x - P.lo.xyz) / P.sp.xyz, vec3<f32>(0.0), vec3<f32>(P.gshape.xyz) - 1.0);
  let m = to_moving(x + field(g));
  let m0 = floor(m); let f = m - m0; let i0 = vec3<i32>(m0);
  var val = 0.0; var gr = vec3<f32>(0.0);
  for (var c = 0u; c < 8u; c++) {
    let o = vec3<u32>(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
    let s = mov(i0 + vec3<i32>(o));
    let w = vec3<f32>(select(1.0 - f.x, f.x, o.x == 1u), select(1.0 - f.y, f.y, o.y == 1u), select(1.0 - f.z, f.z, o.z == 1u));
    let sg = vec3<f32>(select(-1.0, 1.0, o.x == 1u), select(-1.0, 1.0, o.y == 1u), select(-1.0, 1.0, o.z == 1u));
    val += w.x * w.y * w.z * s;
    gr += s * vec3<f32>(sg.x * w.y * w.z, w.x * sg.y * w.z, w.x * w.y * sg.z);
  }
  J[p] = val;
  let gm = gr / P.vm.xyz;  // dJ/d(moving index) -> dJ/d(displacement): A^T gm
  C[3u * p] = P.a0.x * gm.x + P.a1.x * gm.y + P.a2.x * gm.z;
  C[3u * p + 1u] = P.a0.y * gm.x + P.a1.y * gm.y + P.a2.y * gm.z;
  C[3u * p + 2u] = P.a0.z * gm.x + P.a1.z * gm.y + P.a2.z * gm.z;`),

  // Per level, before fitting: fixed image, its square, and "not covered by the moving
  // image under the affine" (1) as three channels, to be box-summed.
  prep: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;` + ENTRY(/* wgsl */ `
  let N = P.fshape.w;
  if (p >= N) { return; }
  let m = to_moving(phys(voxel(p)));
  let top = max(vec3<f32>(P.mshape.xyz) - 1.0, vec3<f32>(1.0));
  let covered = all(m >= vec3<f32>(0.0)) && all(m <= top);
  S[p] = F[p]; S[N + p] = F[p] * F[p]; S[2u * N + p] = select(1.0, 0.0, covered);`),

  statics: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> STAT: array<f32>;` + ENTRY(/* wgsl */ `
  let N = P.fshape.w;
  if (p >= N) { return; }
  let n = window_count(p);
  let mi = S[p] / n;
  STAT[p] = mi;
  STAT[N + p] = S[N + p] / n - mi * mi;
  STAT[2u * N + p] = select(0.0, 1.0, S[2u * N + p] < 0.5);  // whole window covered`),

  products: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> J: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;` + ENTRY(/* wgsl */ `
  let N = P.fshape.w;
  if (p >= N) { return; }
  let j = J[p];
  S[p] = j; S[N + p] = j * j; S[2u * N + p] = F[p] * j;`),

  // Box sum of three channels along one axis, the window clipped at the volume's faces: one
  // thread per line, keeping a running sum (the voxel entering the window added, the one
  // leaving it subtracted), so the cost does not grow with the window.
  box: COMMON + /* wgsl */ `
struct Box { axis: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(1) var<uniform> B: Box;
@group(0) @binding(2) var<storage, read> IN: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
fn at(q: u32) -> vec3<f32> { let N = P.fshape.w; return vec3<f32>(IN[q], IN[N + q], IN[2u * N + q]); }` + ENTRY(/* wgsl */ `
  let N = P.fshape.w; let Z = P.fshape.x; let Y = P.fshape.y; let X = P.fshape.z;
  var n = X; var lines = Z * Y; var stride = 1u; var start = p * X;       // along x: line (z, y)
  if (B.axis == 0u) { n = Z; lines = Y * X; stride = Y * X; start = p; }  // along z: line (y, x)
  if (B.axis == 1u) { n = Y; lines = Z * X; stride = X; start = (p / X) * (Y * X) + (p % X); }  // along y
  if (p >= lines) { return; }
  let h = u32(P.misc.w);
  var s = vec3<f32>(0.0);
  for (var t = 0u; t <= min(h, n - 1u); t++) { s += at(start + t * stride); }
  for (var i = 0u; i < n; i++) {
    let q = start + i * stride;
    OUT[q] = s.x; OUT[N + q] = s.y; OUT[2u * N + q] = s.z;
    if (i + h + 1u < n) { s += at(start + (i + h + 1u) * stride); }
    if (i >= h) { s -= at(start + (i - h) * stride); }
  }`),

  // Squared local correlation and the per-window factors of its gradient.
  cc: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> STAT: array<f32>;
@group(0) @binding(3) var<storage, read_write> AB: array<f32>;
@group(0) @binding(4) var<storage, read_write> CC: array<f32>;` + ENTRY(/* wgsl */ `
  let N = P.fshape.w;
  if (p >= N) { return; }
  let n = window_count(p);
  let mj = S[p] / n; let vj = S[N + p] / n - mj * mj; let ij = S[2u * N + p] / n;
  let mi = STAT[p]; let vi = STAT[N + p];
  let cov = ij - mi * mj;
  var a = 0.0; var b = 0.0; var cc = 0.0;
  if (vi > P.misc.x && vj > P.misc.x && STAT[2u * N + p] > 0.5) {
    let r = cov / (vi * vj);
    cc = cov * r; a = 2.0 * r / n; b = 2.0 * cov * r / (vj * n);
  }
  AB[p] = a; AB[N + p] = b; AB[2u * N + p] = a * mi - b * mj;
  CC[p] = cc;`),

  // d(loss)/d(displacement) at every voxel: the windows through it, then the chain rule.
  grad: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> J: array<f32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
@group(0) @binding(5) var<storage, read_write> GD: array<f32>;` + ENTRY(/* wgsl */ `
  let N = P.fshape.w;
  if (p >= N) { return; }
  let dj = -(F[p] * S[p] - J[p] * S[N + p] - S[2u * N + p]) * P.misc.y;
  GD[3u * p] = dj * C[3u * p]; GD[3u * p + 1u] = dj * C[3u * p + 1u]; GD[3u * p + 2u] = dj * C[3u * p + 2u];`),

  // Gradient at each control point: its voxels' gradients weighted by their trilinear
  // weights (gathered, since WGSL has no float atomics), plus the smoothness term.
  gather: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> GD: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> GU: array<f32>;
fn u_at(q: vec3<u32>, k: u32) -> f32 { return U[((q.x * P.gshape.y + q.y) * P.gshape.z + q.z) * 3u + k]; }` + ENTRY(/* wgsl */ `
  if (p >= P.gshape.w) { return; }
  let Gx = P.gshape.z; let Gy = P.gshape.y;
  let c = vec3<u32>(p / (Gx * Gy), (p / Gx) % Gy, p % Gx);
  let cf = vec3<f32>(c);
  let xc = P.lo.xyz + cf * P.sp.xyz;
  let top = vec3<f32>(P.gshape.xyz) - 1.0;
  let lo = vec3<i32>(max(ceil((xc - P.sp.xyz - P.tf.xyz) / P.vf.xyz), vec3<f32>(0.0)));
  let hi = vec3<i32>(min(floor((xc + P.sp.xyz - P.tf.xyz) / P.vf.xyz), vec3<f32>(P.fshape.xyz) - 1.0));
  var acc = vec3<f32>(0.0);
  for (var i = lo.x; i <= hi.x; i++) {
    for (var j = lo.y; j <= hi.y; j++) {
      for (var k = lo.z; k <= hi.z; k++) {
        let v = vec3<u32>(u32(i), u32(j), u32(k));
        let g = clamp((phys(v) - P.lo.xyz) / P.sp.xyz, vec3<f32>(0.0), top);
        let d = abs(g - cf);
        if (all(d < vec3<f32>(1.0))) {
          let w = (1.0 - d.x) * (1.0 - d.y) * (1.0 - d.z);
          let q = (v.x * P.fshape.y + v.y) * P.fshape.z + v.z;
          acc += w * vec3<f32>(GD[3u * q], GD[3u * q + 1u], GD[3u * q + 2u]);
        }
      }
    }
  }
  for (var k = 0u; k < 3u; k++) {
    var r = 0.0;
    let u0 = u_at(c, k);
    for (var a = 0u; a < 3u; a++) {
      var e = vec3<u32>(0u); e[a] = 1u;
      if (c[a] > 0u) { r += P.reg[a] * (u0 - u_at(c - e, k)); }
      if (c[a] + 1u < P.gshape[a]) { r -= P.reg[a] * (u_at(c + e, k) - u0); }
    }
    GU[3u * p + k] = acc[k] + P.misc.z * r;
  }`),

  adam: COMMON + /* wgsl */ `
@group(0) @binding(1) var<storage, read> GU: array<f32>;
@group(0) @binding(2) var<storage, read_write> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> MOM: array<f32>;
@group(0) @binding(4) var<storage, read_write> VEL: array<f32>;` + ENTRY(/* wgsl */ `
  if (p >= 3u * P.gshape.w) { return; }
  let g = GU[p];
  let m = 0.9 * MOM[p] + 0.1 * g; let v = 0.999 * VEL[p] + 0.001 * g * g;
  MOM[p] = m; VEL[p] = v;
  U[p] -= P.adam.x * (m * P.adam.y) / (sqrt(v * P.adam.z) + 1e-8);`),
};

let opened: Promise<Gpu> | null = null;

/** The GPU and its compiled kernels: made once per page, the kernels compiled in parallel. */
export function gpu(): Promise<Gpu> {
  opened ??= open().catch((e) => { opened = null; throw e; });  // a failed start is retried next time
  return opened;
}

async function open(): Promise<Gpu> {
  if (!navigator.gpu) throw new Error("this browser has no WebGPU (chrome://gpu shows why; on Linux Chrome may need chrome://flags/#enable-unsafe-webgpu)");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const compiled = await Promise.all((Object.entries(KERNELS) as [Kernel, string][]).map(async ([name, code]) => {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) if (m.type === "error") throw new Error(`${name}: ${m.message} (line ${m.lineNum})`);
    return [name, await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } })] as const;
  }));
  const info = adapter.info;
  return {
    device, pipelines: Object.fromEntries(compiled) as Record<Kernel, GPUComputePipeline>,
    name: [info.vendor, info.architecture, info.description].filter(Boolean).join(" "),
  };
}

function controlShape(lo: number[], hi: number[], voxel: number[], grid: number): number[] {
  return lo.map((l, a) => Math.max(2, Math.ceil((hi[a] - l) / (grid * voxel[a])) + 1));
}

/** The field on a finer control grid spanning the same box: trilinear, as registration's
 * F.interpolate(align_corners=True). */
function refine(u: Float32Array, from: number[], to: number[], lo: number[], hi: number[]): Float32Array {
  const spacing = (n: number[]) => lo.map((l, a) => (hi[a] - l) / (n[a] - 1));
  const old: ControlGrid = { shape: from, origin: lo, spacing: spacing(from), values: u };
  const sp = spacing(to), out = new Float32Array(3 * prod(to)), d = [0, 0, 0];
  for (let z = 0, p = 0; z < to[0]; z++) for (let y = 0; y < to[1]; y++) for (let x = 0; x < to[2]; x++, p += 3) {
    fieldAt(old, lo[0] + z * sp[0], lo[1] + y * sp[1], lo[2] + x * sp[2], d);
    out[p] = d[0]; out[p + 1] = d[1]; out[p + 2] = d[2];
  }
  return out;
}

type Buffers = Record<"P" | "F" | "M" | "U" | "J" | "C" | "S" | "T" | "AB" | "STAT" | "CC" | "GD" | "GU" | "MOM" | "VEL", GPUBuffer>;

class Level {
  g: Gpu; N: number; G: number; lines: number[]; spacing: number[]; buffers: Buffers; axes: GPUBuffer[];
  params: ArrayBuffer; lr: number;
  groups: Record<"prep" | "statics" | "warp" | "products" | "cc" | "grad" | "gather" | "adam", GPUBindGroup>
    & Record<"boxS" | "boxAB", GPUBindGroup[]>;

  constructor(g: Gpu, fixed: Volume, moving: Volume, affine: Affine, box: number[][], gshape: number[], settings: Settings) {
    const { device } = g;
    this.g = g;
    const N = prod(fixed.shape), G = prod(gshape), [Z, Y, X] = fixed.shape;
    this.N = N; this.G = G;
    this.lines = [Y * X, Z * X, Z * Y];  // lines along z, y, x: the box kernel's threads
    const lo = box[0], hi = box[1];
    this.spacing = lo.map((l, a) => (hi[a] - l) / (gshape[a] - 1));
    const buf = (n: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) =>
      device.createBuffer({ size: Math.max(16, n * 4), usage });
    this.buffers = {
      P: device.createBuffer({ size: 16 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      F: buf(N), M: buf(prod(moving.shape)), U: buf(3 * G), J: buf(N), C: buf(3 * N),
      S: buf(3 * N), T: buf(3 * N), AB: buf(3 * N), STAT: buf(3 * N), CC: buf(N), GD: buf(3 * N),
      GU: buf(3 * G), MOM: buf(3 * G), VEL: buf(3 * G),
    };
    this.axes = [0, 1, 2].map((a) => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([a, 0, 0, 0]));
      return b;
    });
    const B = this.buffers;
    device.queue.writeBuffer(B.F, 0, fixed.norm as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(B.M, 0, moving.norm as Float32Array<ArrayBuffer>);
    // Params
    const p = new ArrayBuffer(16 * 16), u32 = new Uint32Array(p), f32 = new Float32Array(p);
    u32.set([...fixed.shape, N], 0); u32.set([...moving.shape, prod(moving.shape)], 4); u32.set([...gshape, G], 8);
    f32.set(fixed.voxel, 12); f32.set(fixed.origin, 16); f32.set(moving.voxel, 20); f32.set(moving.origin, 24);
    f32.set(lo, 28); f32.set(this.spacing, 32);
    f32.set(affine[0], 36); f32.set(affine[1], 40); f32.set(affine[2], 44);
    f32.set([FLAT, 1 / N, settings.smooth, Math.floor(WINDOW / 2)], 48);
    const diffs = [0, 1, 2].map((a) => 3 * (gshape[a] - 1) * prod(gshape.filter((_, b) => b !== a)));
    f32.set([0, 1, 2].map((a) => (diffs[a] ? 2 / (diffs[a] * this.spacing[a] ** 2) : 0)), 52);
    this.params = p;
    this.lr = STEP * (fixed.voxel.reduce((x, y) => x + y) / 3);
    device.queue.writeBuffer(B.P, 0, p);
    const bind = (name: Kernel, ...names: (keyof Buffers | GPUBuffer)[]) => device.createBindGroup({
      layout: g.pipelines[name].getBindGroupLayout(0),
      entries: names.map((n, i) => ({ binding: i, resource: { buffer: typeof n === "string" ? B[n] : n } })),
    });
    const boxes = (a: GPUBuffer, b: GPUBuffer, c: GPUBuffer, d: GPUBuffer) => [bind("box", "P", this.axes[0], a, b), bind("box", "P", this.axes[1], b, c), bind("box", "P", this.axes[2], c, d)];
    this.groups = {
      prep: bind("prep", "P", "F", "S"), statics: bind("statics", "P", "T", "STAT"),
      boxS: boxes(B.S, B.T, B.S, B.T),  // S's three channels into T: the prep's, then each step's
      warp: bind("warp", "P", "U", "M", "J", "C"), products: bind("products", "P", "F", "J", "S"),
      cc: bind("cc", "P", "T", "STAT", "AB", "CC"),
      boxAB: boxes(B.AB, B.S, B.T, B.S), grad: bind("grad", "P", "F", "J", "S", "C", "GD"),
      gather: bind("gather", "P", "GD", "U", "GU"), adam: bind("adam", "P", "GU", "U", "MOM", "VEL"),
    };
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    this.dispatch(pass, "prep", N); this.boxes(pass, "boxS"); this.dispatch(pass, "statics", N);
    pass.end(); device.queue.submit([enc.finish()]);
  }

  dispatch(pass: GPUComputePassEncoder, name: Kernel, n: number, group = this.groups[name as keyof Level["groups"]] as GPUBindGroup) {
    const blocks = Math.ceil(n / WG), x = Math.min(blocks, 65535);
    pass.setPipeline(this.g.pipelines[name]); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, Math.ceil(blocks / x));
  }

  boxes(pass: GPUComputePassEncoder, key: "boxS" | "boxAB") { this.groups[key].forEach((grp, a) => this.dispatch(pass, "box", this.lines[a], grp)); }

  setField(u: Float32Array) { this.g.device.queue.writeBuffer(this.buffers.U, 0, u as Float32Array<ArrayBuffer>); }

  step(t: number) {  // one Adam iteration
    const f32 = new Float32Array(this.params);
    f32.set([this.lr, 1 / (1 - 0.9 ** t), 1 / (1 - 0.999 ** t), 0], 56);
    this.g.device.queue.writeBuffer(this.buffers.P, 0, this.params);
    const enc = this.g.device.createCommandEncoder(), pass = enc.beginComputePass();
    this.dispatch(pass, "warp", this.N); this.dispatch(pass, "products", this.N); this.boxes(pass, "boxS");
    this.dispatch(pass, "cc", this.N); this.boxes(pass, "boxAB"); this.dispatch(pass, "grad", this.N);
    this.dispatch(pass, "gather", this.G); this.dispatch(pass, "adam", 3 * this.G);
    pass.end(); this.g.device.queue.submit([enc.finish()]);
  }

  async read(name: keyof Buffers, n: number): Promise<Float32Array> {
    const { device } = this.g;
    const staging = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buffers[name], 0, staging, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }

  async similarity() { const cc = await this.read("CC", this.N); let s = 0; for (const v of cc) s += v; return s / this.N; }

  destroy() { for (const b of [...Object.values(this.buffers), ...this.axes]) b.destroy(); }
}

/**
 * Fit the field level by level. `data` is [[fixed, moving], ...] coarse to fine, each
 * {norm, shape, voxel, origin}; `box` is the fixed image's physical extent [lo, hi].
 * `onProgress({stage, stages, iteration, iterations, similarity})` is called as it goes.
 */
export async function solve(
  g: Gpu, data: [Volume, Volume][], affine: Affine, box: number[][], settings: Settings,
  onProgress: (p: Progress) => void = () => {},
): Promise<{ grid: ControlGrid; stages: Stage[]; seconds: number }> {
  const [lo, hi] = box;
  const t0 = performance.now();
  let gshape = controlShape(lo, hi, data[0][0].voxel, settings.grid);
  let u: Float32Array = new Float32Array(3 * prod(gshape));
  const stages: Stage[] = [];
  let last: Level | null = null;
  for (const [s, [fl, ml]] of data.entries()) {
    const shape = controlShape(lo, hi, fl.voxel, settings.grid);
    if (shape.join() !== gshape.join()) { u = refine(u, gshape, shape, lo, hi); gshape = shape; }
    last?.destroy();
    const lvl = new Level(g, fl, ml, affine, box, gshape, settings);
    lvl.setField(u);
    const ts = performance.now();
    let first: number | null = null, sim: number | null = null;
    for (let t = 1; t <= settings.iterations; t++) {
      lvl.step(t);
      if (t === 1) sim = first = await lvl.similarity();
      else if (t % 10 === 0 || t === settings.iterations) await g.device.queue.onSubmittedWorkDone();
      if (t === settings.iterations) sim = await lvl.similarity();
      onProgress({ stage: s, stages: data.length, iteration: t, iterations: settings.iterations, similarity: sim });
    }
    u = await lvl.read("U", 3 * lvl.G);
    stages.push({ shape: fl.shape, grid: gshape, seconds: (performance.now() - ts) / 1000, first, final: sim });
    last = lvl;
  }
  if (!last) throw new Error("no levels to solve on");
  last.destroy();
  return { grid: { shape: gshape, origin: lo, spacing: last.spacing, values: u }, stages, seconds: (performance.now() - t0) / 1000 };
}
