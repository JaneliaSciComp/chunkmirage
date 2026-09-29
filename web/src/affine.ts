// The affine to start a registration from when none is given. The images' centres and
// principal axes are matched first (their intensity moments), which leaves each axis's sign
// open: of the orientations that do not mirror the image, the one that correlates best is
// kept. Then the 12 numbers are fitted by gradient ascent on the normalized cross-correlation,
// coarse to fine. A few hundred thousand voxels are enough, so this runs on the CPU.
// Images are {norm, shape, voxel, origin} (C order z, y, x, physical units), as the page
// hands the field solve; the affine maps fixed to moving coordinates, rows [A | t].
import { percentiles, prod } from "./ome";
import type { Affine, Progress, Volume } from "./types";

interface Scratch { mv: Float32Array; g: Float32Array }  // the fit's per-voxel values and gradients

export interface FoundAffine {
  affine: Affine;
  seconds: number;
  identity: number;  // correlation with no affine,
  moments: number;   // after matching the moments,
  final: number;     // and after the fit
  distance?: number | null;  // from a reference affine, when the page has one
}

const MAX_FIT_VOXELS = 300_000;  // finest stage of the fit
const MIN_FIT_SIZE = 12;         // coarsest stage: at least this many voxels on every axis
const ITERATIONS = [150, 100];   // per stage, coarse to fine
const LR_MATRIX = 2e-3;          // Adam steps: matrix entries, and the centre's image in voxels
const LR_CENTRE = 0.4;

const last = <T>(a: T[]): T => a[a.length - 1];
const pause = () => new Promise<void>((r) => setTimeout(r, 0));  // let the page draw progress

/** The affine as rows [A | t], and the correlation (on the fit's finest copy) with no
 * affine, after the moments, and after the fit. */
export async function findAffine(fixed: Volume, moving: Volume, onProgress: (p: Progress) => void = () => {}): Promise<FoundAffine> {
  const t0 = performance.now();
  const fs = [fixed], ms = [moving];
  // stages: the finest copy within MAX_FIT_VOXELS, after one coarser if that is big enough
  while (prod(last(fs).shape) > MAX_FIT_VOXELS) { fs.push(halve(last(fs))); ms.push(halve(last(ms))); }
  const stages = [fs.length - 1], coarser = halve(last(fs));
  if (Math.min(...coarser.shape) >= MIN_FIT_SIZE) { fs.push(coarser); ms.push(halve(last(ms))); stages.unshift(fs.length - 1); }

  const F0 = fs[stages[0]], M0 = ms[stages[0]], Ff = fs[last(stages)], Mf = ms[last(stages)];
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], zero = [0, 0, 0];
  const [cf, Cf] = moments(F0), [cm, Cm] = moments(M0);
  const [ef, Vf] = eigh(Cf), [em, Vm] = eigh(Cm);
  let best: { v: number; A: number[][] } | null = null;
  for (const s of [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]) {
    // A = Vm sqrt(em) S / sqrt(ef) Vf^T: takes the fixed image's second moments to the moving one's
    const D = [0, 1, 2].map((k) => (Math.sqrt(em[k] / ef[k]) * s[k]));
    const A = [0, 1, 2].map((r) => [0, 1, 2].map((c) => Vm[r][0] * D[0] * Vf[c][0] + Vm[r][1] * D[1] * Vf[c][1] + Vm[r][2] * D[2] * Vf[c][2]));
    if (det(A) <= 0) continue;  // a mirror image
    const v = ncc(F0, M0, A, cm, cf).value;  // the fixed centre goes to the moving centre
    if (!best || v > best.v) best = { v, A };
  }
  if (!best) throw new Error("no orientation of the principal axes fits");  // four always qualify
  const scores = { identity: ncc(Ff, Mf, identity, zero, zero).value, moments: ncc(Ff, Mf, best.A, cm, cf).value };
  let A = best.A, u = cm, final = best.v;  // u: where the fixed centre goes
  for (const [k, s] of stages.entries()) {
    const F = fs[s], M = ms[s], iters = ITERATIONS[stages.length === 1 ? 0 : k];
    const lrU = LR_CENTRE * (F.voxel[0] + F.voxel[1] + F.voxel[2]) / 3;
    const adam = new Adam(12), n = F.norm.length;
    const scratch = { mv: new Float32Array(n), g: new Float32Array(3 * n) };
    for (let it = 0; it < iters; it++) {
      const r = ncc(F, M, A, u, cf, scratch);
      final = r.value;
      const step = adam.step([...r.gA!.flat(), ...r.gu!]);  // ascent
      A = A.map((row, i) => row.map((a, j) => a + LR_MATRIX * step[3 * i + j]));
      u = u.map((x, i) => x + lrU * step[9 + i]);
      if (it % 5 === 4 || it === iters - 1) {
        onProgress({ stage: k, stages: stages.length, iteration: it + 1, iterations: iters, similarity: final });
        await pause();
      }
    }
  }
  final = ncc(Ff, Mf, A, u, cf).value;
  const t = u.map((x, i) => x - (A[i][0] * cf[0] + A[i][1] * cf[1] + A[i][2] * cf[2]));  // q = A p + t
  return {
    affine: A.map((row, i) => [...row, t[i]]), seconds: (performance.now() - t0) / 1000,
    ...scores, final,
  };
}

/** Mean distance, over the fixed image's tissue (a coarse copy of it), between where two
 * affines put its voxels: how far apart they are, in physical units. */
export function affineDistance(fixed: Volume, a: Affine, b: Affine): number {
  let F = fixed;
  while (prod(F.shape) > MAX_FIT_VOXELS) F = halve(F);
  const [nz, ny, nx] = F.shape;
  let sum = 0, n = 0;
  for (let z = 0, i = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) {
    if (F.norm[i] <= 0.05) continue;
    const p = [F.origin[0] + z * F.voxel[0], F.origin[1] + y * F.voxel[1], F.origin[2] + x * F.voxel[2]];
    let d2 = 0;
    for (let r = 0; r < 3; r++) {
      const d = (a[r][0] - b[r][0]) * p[0] + (a[r][1] - b[r][1]) * p[1] + (a[r][2] - b[r][2]) * p[2] + a[r][3] - b[r][3];
      d2 += d * d;
    }
    sum += Math.sqrt(d2); n++;
  }
  return n ? sum / n : 0;
}

// ------------------------------------------------ pieces
function halve(img: Volume): Volume {  // 2x2x2 means; the grid's origin moves to the first block's centre
  const [nz, ny, nx] = img.shape, s = [nz, ny, nx].map((n) => Math.max(1, n >> 1));
  const out = new Float32Array(prod(s)), d = img.norm;
  for (let z = 0, o = 0; z < s[0]; z++) for (let y = 0; y < s[1]; y++) for (let x = 0; x < s[2]; x++, o++) {
    let sum = 0, k = 0;
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const zz = 2 * z + dz, yy = 2 * y + dy, xx = 2 * x + dx;
      if (zz < nz && yy < ny && xx < nx) { sum += d[(zz * ny + yy) * nx + xx]; k++; }
    }
    out[o] = sum / k;
  }
  return {
    norm: out, shape: s, voxel: img.voxel.map((v, a) => (img.shape[a] > 1 ? 2 * v : v)),
    origin: img.origin.map((o, a) => (img.shape[a] > 1 ? o + img.voxel[a] / 2 : o)),
  };
}

function moments(img: Volume): [number[], number[][]] {  // intensity-weighted centre and covariance of the tissue
  const { norm: data, shape: [nz, ny, nx], voxel, origin } = img;
  const nonzero = data.filter((v) => v > 0);
  const floor = nonzero.length ? percentiles(nonzero, [20])[0] : 0;  // not background
  let w = 0;
  const m = [0, 0, 0], S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let z = 0, i = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, i++) {
    const v = data[i];
    if (v <= floor) continue;
    const p = [origin[0] + z * voxel[0], origin[1] + y * voxel[1], origin[2] + x * voxel[2]];
    w += v;
    for (let a = 0; a < 3; a++) { m[a] += v * p[a]; for (let b = 0; b < 3; b++) S[a][b] += v * p[a] * p[b]; }
  }
  const c = m.map((x) => x / w);
  return [c, S.map((row, a) => row.map((s, b) => s / w - c[a] * c[b]))];
}

function eigh(S: number[][]): [number[], number[][]] {  // eigenvalues and eigenvectors (columns) of a symmetric 3x3, by Jacobi
  const a = S.map((r) => [...r]), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-20 * (a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2)) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < 3; k++) {  // a = J^T a J, V = V J
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = cs * akp - sn * akq; a[k][q] = sn * akp + cs * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = cs * apk - sn * aqk; a[q][k] = sn * apk + cs * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = cs * vkp - sn * vkq; V[k][q] = sn * vkp + cs * vkq;
      }
    }
  }
  // ascending, so both images' axes pair up by extent
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return [order.map((i) => Math.max(a[i][i], 1e-12)), V.map((row) => order.map((i) => row[i]))];
}

const det = (A: number[][]) => A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
  - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

/** Normalized cross-correlation of the fixed image and the moving one sampled at
 * q = A (p - c) + u (trilinear, 0 beyond it), and given `scratch` its gradient in A and u. */
function ncc(F: Volume, M: Volume, A: number[][], u: number[], c: number[], scratch?: Scratch): { value: number; gA?: number[][]; gu?: number[] } {
  const [nz, ny, nx] = F.shape, [mz, my, mx] = M.shape, n = F.norm.length;
  const grad = scratch !== undefined, mv = scratch?.mv, g = scratch?.g;
  let sf = 0, sm = 0, sff = 0, smm = 0, sfm = 0;
  for (let z = 0, i = 0; z < nz; z++) {
    const pz = F.origin[0] + z * F.voxel[0] - c[0];
    for (let y = 0; y < ny; y++) {
      const py = F.origin[1] + y * F.voxel[1] - c[1];
      for (let x = 0; x < nx; x++, i++) {
        const px = F.origin[2] + x * F.voxel[2] - c[2];
        const iz = (A[0][0] * pz + A[0][1] * py + A[0][2] * px + u[0] - M.origin[0]) / M.voxel[0];
        const iy = (A[1][0] * pz + A[1][1] * py + A[1][2] * px + u[1] - M.origin[1]) / M.voxel[1];
        const ix = (A[2][0] * pz + A[2][1] * py + A[2][2] * px + u[2] - M.origin[2]) / M.voxel[2];
        const z0 = Math.floor(iz), y0 = Math.floor(iy), x0 = Math.floor(ix);
        let v = 0, gz = 0, gy = 0, gx = 0;
        if (z0 >= -1 && y0 >= -1 && x0 >= -1 && z0 < mz && y0 < my && x0 < mx) {
          const fz = iz - z0, fy = iy - y0, fx = ix - x0;
          for (let dz = 0; dz < 2; dz++) {
            const zz = z0 + dz;
            if (zz < 0 || zz >= mz) continue;
            const wz = dz ? fz : 1 - fz, sz = dz ? 1 : -1;
            for (let dy = 0; dy < 2; dy++) {
              const yy = y0 + dy;
              if (yy < 0 || yy >= my) continue;
              const wy = dy ? fy : 1 - fy, sy = dy ? 1 : -1;
              for (let dx = 0; dx < 2; dx++) {
                const xx = x0 + dx;
                if (xx < 0 || xx >= mx) continue;
                const wx = dx ? fx : 1 - fx, sx = dx ? 1 : -1, m = M.norm[(zz * my + yy) * mx + xx];
                v += wz * wy * wx * m;
                if (grad) { gz += sz * wy * wx * m; gy += wz * sy * wx * m; gx += wz * wy * sx * m; }
              }
            }
          }
        }
        if (mv && g) { mv[i] = v; g[3 * i] = gz / M.voxel[0]; g[3 * i + 1] = gy / M.voxel[1]; g[3 * i + 2] = gx / M.voxel[2]; }
        const f = F.norm[i];
        sf += f; sm += v; sff += f * f; smm += v * v; sfm += f * v;
      }
    }
  }
  const saa = sff - (sf * sf) / n, sbb = smm - (sm * sm) / n, sab = sfm - (sf * sm) / n;
  const norm = Math.sqrt(Math.max(saa * sbb, 1e-24)), value = sab / norm;
  if (!mv || !g) return { value };
  // dNCC/dm_i = a_i / norm - NCC b_i / sbb, a and b the centred images; then through q
  const fbar = sf / n, mbar = sm / n, k1 = 1 / norm, k2 = value / Math.max(sbb, 1e-24);
  const gA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], gu = [0, 0, 0];
  for (let z = 0, i = 0; z < nz; z++) {
    const pz = F.origin[0] + z * F.voxel[0] - c[0];
    for (let y = 0; y < ny; y++) {
      const py = F.origin[1] + y * F.voxel[1] - c[1];
      for (let x = 0; x < nx; x++, i++) {
        const px = F.origin[2] + x * F.voxel[2] - c[2];
        const w = (F.norm[i] - fbar) * k1 - (mv[i] - mbar) * k2;
        if (w === 0) continue;
        for (let r = 0; r < 3; r++) {
          const d = w * g[3 * i + r];
          gA[r][0] += d * pz; gA[r][1] += d * py; gA[r][2] += d * px; gu[r] += d;
        }
      }
    }
  }
  return { value, gA, gu };
}

class Adam {  // step() returns the update direction for a gradient to climb
  m: Float64Array; v: Float64Array; t = 0;
  constructor(n: number) { this.m = new Float64Array(n); this.v = new Float64Array(n); }
  step(g: number[]): number[] {
    this.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    return g.map((x, i) => {
      this.m[i] = b1 * this.m[i] + (1 - b1) * x;
      this.v[i] = b2 * this.v[i] + (1 - b2) * x * x;
      return (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + 1e-12);
    });
  }
}
