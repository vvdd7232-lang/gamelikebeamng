// World: heightfield terrain, a closed circuit road, a concrete testing pad and
// static colliders (oriented boxes).  Everything is baked once at start-up so
// the physics can do cheap lookups.

import { mulberry32, clamp, lerp, smoothstep } from '../core/math.js';

export const MATERIALS = [
  { name: 'asphalt', mu: 1.0, color: [0.115, 0.118, 0.125], rough: 0.82, dust: [0.20, 0.20, 0.21] },
  { name: 'grass', mu: 0.60, color: [0.135, 0.175, 0.085], rough: 0.95, dust: [0.32, 0.33, 0.19] },
  { name: 'dirt', mu: 0.66, color: [0.245, 0.195, 0.145], rough: 0.9, dust: [0.42, 0.34, 0.24] },
  { name: 'concrete', mu: 0.95, color: [0.32, 0.32, 0.325], rough: 0.78, dust: [0.5, 0.5, 0.5] },
];

const MAT_ASPHALT = 0;
const MAT_GRASS = 1;
const MAT_DIRT = 2;
const MAT_CONCRETE = 3;
const _surfN = [0, 1, 0];

// closed circuit control points (x, z), roughly 1.15 km
const TRACK_CTRL = [
  [-10, -170], [95, -180], [165, -135], [185, -55], [155, 25],
  [95, 70], [55, 125], [-25, 165], [-110, 155], [-165, 95],
  [-180, 10], [-160, -85], [-95, -150],
];
const TRACK_WIDTH = 11.0;

// handling / crash-test pad
export const PAD = { x: 265, z: 285, r: 62, y: 6.0 };

// value noise ---------------------------------------------------------------
function makeNoise(seed) {
  const rnd = mulberry32(seed);
  const perm = new Uint8Array(512);
  const vals = new Float32Array(256);
  for (let i = 0; i < 256; i++) { perm[i] = i; vals[i] = rnd() * 2 - 1; }
  for (let i = 255; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (let i = 0; i < 256; i++) perm[i + 256] = perm[i];
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  return function noise2(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const u = fade(xf), v = fade(yf);
    const x0 = xi & 255, y0 = yi & 255, x1 = (xi + 1) & 255, y1 = (yi + 1) & 255;
    const a = vals[perm[x0 + perm[y0]] & 255];
    const b = vals[perm[x1 + perm[y0]] & 255];
    const c = vals[perm[x0 + perm[y1]] & 255];
    const d = vals[perm[x1 + perm[y1]] & 255];
    return lerp(lerp(a, b, u), lerp(c, d, u), v);
  };
}

function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

export class World {
  constructor(seed = 1337) {
    this.seed = seed;
    this.noise = makeNoise(seed);
    this.fbm = (x, z, oct = 4, lac = 2.1, gain = 0.5) => {
      let a = 1, f = 1, sum = 0, norm = 0;
      for (let i = 0; i < oct; i++) {
        sum += a * this.noise(x * f, z * f);
        norm += a; a *= gain; f *= lac;
      }
      return sum / norm;
    };

    this.halfSize = 430;
    this.cell = 1.4;
    this.gridN = Math.round((this.halfSize * 2) / this.cell) + 1;
    this.heights = new Float32Array(this.gridN * this.gridN);
    this.matIds = new Uint8Array(this.gridN * this.gridN);

    this.buildTrack();
    this.bakeHeights();
    this.buildColliders();
  }

  // ------------------------------------------------------------- base terrain
  baseHeight(x, z) {
    let h = 0;
    h += this.fbm(x * 0.0016, z * 0.0016, 4) * 34.0;
    h += this.fbm(x * 0.0065 + 11.3, z * 0.0065 - 4.1, 3) * 7.0;
    h += this.fbm(x * 0.028 - 2.7, z * 0.028 + 9.4, 2) * 0.75;
    // keep the middle of the map fairly flat (drivable)
    const r = Math.hypot(x, z);
    const flat = smoothstep(clamp((r - 150) / 230, 0, 1));
    return h * (0.35 + 0.65 * flat) + 1.0;
  }

  // ------------------------------------------------------------------- track
  buildTrack() {
    const ctrl = TRACK_CTRL;
    const n = ctrl.length;
    const raw = [];
    const stepsPerSeg = 60;
    for (let i = 0; i < n; i++) {
      const p0 = ctrl[(i - 1 + n) % n], p1 = ctrl[i], p2 = ctrl[(i + 1) % n], p3 = ctrl[(i + 2) % n];
      for (let s = 0; s < stepsPerSeg; s++) {
        const t = s / stepsPerSeg;
        raw.push([
          catmullRom(p0[0], p1[0], p2[0], p3[0], t),
          catmullRom(p0[1], p1[1], p2[1], p3[1], t),
        ]);
      }
    }
    // resample to ~1.5 m spacing
    const pts = [];
    let carry = 0;
    for (let i = 0; i < raw.length; i++) {
      const a = raw[i], b = raw[(i + 1) % raw.length];
      const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const steps = Math.max(1, Math.round(d / 1.5));
      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
      }
      carry += d;
    }
    this.trackLength = carry;
    // smooth the raw surface height along the track to get a nice road profile
    const baseH = pts.map((p) => this.baseHeight(p[0], p[1]));
    const win = 26;
    const smoothH = new Float32Array(pts.length);
    for (let i = 0; i < pts.length; i++) {
      let sum = 0;
      for (let k = -win; k <= win; k++) sum += baseH[(i + k + pts.length) % pts.length];
      smoothH[i] = sum / (win * 2 + 1);
    }
    this.road = {
      pts,
      height: smoothH,
      width: TRACK_WIDTH,
      tangents: pts.map((p, i) => {
        const a = pts[(i - 1 + pts.length) % pts.length];
        const b = pts[(i + 1) % pts.length];
        const dx = b[0] - a[0], dz = b[1] - a[1];
        const l = Math.hypot(dx, dz) || 1;
        return [dx / l, dz / l];
      }),
    };
    // spatial hash for nearest-sample queries
    this.hashCell = 12;
    this.hash = new Map();
    const key = (cx, cz) => cx * 8192 + cz;
    pts.forEach((p, i) => {
      const cx = Math.floor(p[0] / this.hashCell), cz = Math.floor(p[1] / this.hashCell);
      const k = key(cx, cz);
      let arr = this.hash.get(k);
      if (!arr) { arr = []; this.hash.set(k, arr); }
      arr.push(i);
    });
    this.hashKey = key;
  }

  /** nearest point on the road centre line -> {dist, height, index} or null */
  nearestRoad(x, z) {
    const cx = Math.floor(x / this.hashCell), cz = Math.floor(z / this.hashCell);
    let best = Infinity, bestIdx = -1;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dz = -1; dz <= 1; dz++) {
        const arr = this.hash.get(this.hashKey(cx + dx, cz + dz));
        if (!arr) continue;
        for (const i of arr) {
          const p = this.road.pts[i];
          const d = (p[0] - x) * (p[0] - x) + (p[1] - z) * (p[1] - z);
          if (d < best) { best = d; bestIdx = i; }
        }
      }
    }
    if (bestIdx < 0) return null;
    const d = Math.sqrt(best);
    // refine with the neighbours (gives a smooth distance field)
    let h = this.road.height[bestIdx];
    const n = this.road.pts.length;
    if (d < 40) {
      const a = this.road.pts[(bestIdx - 1 + n) % n];
      const b = this.road.pts[(bestIdx + 1) % n];
      const seg = [b[0] - a[0], b[1] - a[1]];
      const len2 = seg[0] * seg[0] + seg[1] * seg[1] || 1;
      const t = clamp(((x - a[0]) * seg[0] + (z - a[1]) * seg[1]) / len2, 0, 1);
      const hb = this.road.height[(bestIdx + 1) % n];
      const ha = this.road.height[(bestIdx - 1 + n) % n];
      h = lerp(lerp(ha, this.road.height[bestIdx], 0.5 + t * 0.5), hb, Math.max(0, t - 0.5) * 2) * 0 + h;
    }
    return { dist: d, height: h, index: bestIdx };
  }

  surfaceHeight(x, z) {
    let h = this.baseHeight(x, z);
    h += this.fbm(x * 0.11 + 5.1, z * 0.11 - 3.3, 2) * 0.09;

    const padD = Math.hypot(x - PAD.x, z - PAD.z);
    const padBlend = 1 - smoothstep(clamp((padD - PAD.r) / 34, 0, 1));
    if (padBlend > 0) h = lerp(h, PAD.y, padBlend);

    const road = this.nearestRoad(x, z);
    if (road) {
      const w = 1 - smoothstep(clamp((road.dist - this.road.width * 0.55) / 22, 0, 1));
      if (w > 0) {
        // road is banked slightly towards the outside and raised a touch
        const camber = clamp(road.dist / (this.road.width * 0.55), 0, 1) * 0.10;
        h = lerp(h, road.height + camber + 0.03, w * (1 - padBlend));
      }
    }
    return h;
  }

  materialAt(x, z) {
    const road = this.nearestRoad(x, z);
    if (Math.hypot(x - PAD.x, z - PAD.z) < PAD.r) return MAT_CONCRETE;
    if (road) {
      if (road.dist < this.road.width * 0.5 + 0.6) return MAT_ASPHALT;
      if (road.dist < this.road.width * 0.5 + 5.5) return MAT_DIRT;
    }
    return MAT_GRASS;
  }

  bakeHeights() {
    const N = this.gridN, c = this.cell, h0 = -this.halfSize;
    for (let j = 0; j < N; j++) {
      const z = h0 + j * c;
      for (let i = 0; i < N; i++) {
        const x = h0 + i * c;
        const h = this.surfaceHeight(x, z);
        this.heights[j * N + i] = h;
        this.matIds[j * N + i] = this.materialAt(x, z);
      }
    }
  }

  /** bilinear height lookup (world space) */
  heightAt(x, z) {
    const N = this.gridN, c = this.cell, h0 = -this.halfSize;
    let fx = (x - h0) / c, fz = (z - h0) / c;
    fx = clamp(fx, 0, N - 1.001);
    fz = clamp(fz, 0, N - 1.001);
    const i = fx | 0, j = fz | 0;
    const tx = fx - i, tz = fz - j;
    const H = this.heights;
    const h00 = H[j * N + i], h10 = H[j * N + i + 1];
    const h01 = H[(j + 1) * N + i], h11 = H[(j + 1) * N + i + 1];
    return lerp(lerp(h00, h10, tx), lerp(h01, h11, tx), tz);
  }

  normalAt(x, z, out = [0, 1, 0]) {
    const e = 0.7;
    const hl = this.heightAt(x - e, z), hr = this.heightAt(x + e, z);
    const hd = this.heightAt(x, z - e), hu = this.heightAt(x, z + e);
    const nx = hl - hr, nz = hd - hu, ny = 2 * e;
    const l = Math.hypot(nx, ny, nz) || 1;
    out[0] = nx / l; out[1] = ny / l; out[2] = nz / l;
    return out;
  }

  materialIdAt(x, z) {
    const N = this.gridN, c = this.cell, h0 = -this.halfSize;
    const i = clamp(((x - h0) / c) | 0, 0, N - 1);
    const j = clamp(((z - h0) / c) | 0, 0, N - 1);
    return this.matIds[j * N + i];
  }

  // -------------------------------------------------------------- colliders
  addBox(cx, cy, cz, hx, hy, hz, yaw = 0, pitch = 0, roll = 0, kind = 'box') {
    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    // basis: rotate around Y (yaw), then X (pitch), then Z (roll)
    let m = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    // yaw
    let a = [[cyaw, 0, syaw], [0, 1, 0], [-syaw, 0, cyaw]];
    let b = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
    let c = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
    const mul = (A, B) => {
      const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
        R[i][j] = s;
      }
      return R;
    };
    const R = mul(mul(a, b), c);
    // columns are the local axes in world space
    const axes = [
      [R[0][0], R[1][0], R[2][0]],
      [R[0][1], R[1][1], R[2][1]],
      [R[0][2], R[1][2], R[2][2]],
    ];
    const box = { center: [cx, cy, cz], half: [hx, hy, hz], axes, kind };
    this.colliders.push(box);
    return box;
  }

  buildColliders() {
    this.colliders = [];
    const g = (x, z) => this.heightAt(x, z);
    // --- crash test walls near the pad
    this.addBox(PAD.x - 40, PAD.y + 1.0, PAD.z + 30, 14, 1.0, 0.45, 0.0);
    this.addBox(PAD.x + 20, PAD.y + 1.0, PAD.z - 42, 0.45, 1.0, 16, 0.0);
    // --- ramps (pitch is negative so the car drives up towards +z)
    this.addBox(PAD.x - 10, PAD.y + 0.75, PAD.z + 20, 6, 0.16, 9, 0.0, -0.16);
    this.addBox(PAD.x + 25, PAD.y + 0.9, PAD.z + 18, 5, 0.16, 8, 0.35, -0.20);
    this.addBox(PAD.x - 26, PAD.y + 1.2, PAD.z - 12, 4.5, 0.16, 7.5, -0.5, -0.26);
    // --- big concrete blocks / barriers on the track
    this.addBox(150, g(150, -10) + 0.6, -10, 1.2, 0.6, 4.5, 0.4);
    this.addBox(-120, g(-120, 120) + 0.6, 120, 1.2, 0.6, 5.0, -0.7);
    // --- kerbs / banking objects are visual only; keep physics simple
  }

  /** closest point on a box to p (world) -> {inside, point, normal, depth} */
  static boxContact(box, p, radius) {
    const d = [p[0] - box.center[0], p[1] - box.center[1], p[2] - box.center[2]];
    const local = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      local[i] = d[0] * box.axes[i][0] + d[1] * box.axes[i][1] + d[2] * box.axes[i][2];
    }
    const clamped = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      clamped[i] = clamp(local[i], -box.half[i], box.half[i]);
    }
    const off = [local[0] - clamped[0], local[1] - clamped[1], local[2] - clamped[2]];
    const dist = Math.hypot(off[0], off[1], off[2]);
    const outside = dist > 1e-6;
    let n;
    let depth;
    if (outside) {
      if (dist >= radius) return null;
      // normal points from box towards sphere
      const nl = [off[0] / dist, off[1] / dist, off[2] / dist];
      n = [
        nl[0] * box.axes[0][0] + nl[1] * box.axes[1][0] + nl[2] * box.axes[2][0],
        nl[0] * box.axes[0][1] + nl[1] * box.axes[1][1] + nl[2] * box.axes[2][1],
        nl[0] * box.axes[0][2] + nl[1] * box.axes[1][2] + nl[2] * box.axes[2][2],
      ];
      depth = radius - dist;
    } else {
      // centre inside the box: push out along the shallowest axis
      let mi = 0, best = Infinity;
      for (let i = 0; i < 3; i++) {
        const pen = box.half[i] - Math.abs(local[i]);
        if (pen < best) { best = pen; mi = i; }
      }
      const s = local[mi] < 0 ? -1 : 1;
      n = [
        box.axes[mi][0] * s, box.axes[mi][1] * s, box.axes[mi][2] * s,
      ];
      depth = best + radius;
    }
    const surf = [
      box.center[0] + box.axes[0][0] * clamped[0] + box.axes[1][0] * clamped[1] + box.axes[2][0] * clamped[2],
      box.center[1] + box.axes[0][1] * clamped[0] + box.axes[1][1] * clamped[1] + box.axes[2][1] * clamped[2],
      box.center[2] + box.axes[0][2] * clamped[0] + box.axes[1][2] * clamped[1] + box.axes[2][2] * clamped[2],
    ];
    return { normal: n, depth, point: surf };
  }

  materialMu(x, z) {
    return MATERIALS[this.materialIdAt(x, z)].mu;
  }

  /** ground height + normal including the top faces of static boxes */
  surfaceAt(x, z, out) {
    let h = this.heightAt(x, z);
    let nx = 0, ny = 1, nz = 0;
    const nrm = this.normalAt(x, z, _surfN);
    nx = nrm[0]; ny = nrm[1]; nz = nrm[2];
    for (const box of this.colliders) {
      const up = box.axes[1];
      if (Math.abs(up[1]) < 0.2) continue;
      const px = x - box.center[0];
      const pz = z - box.center[2];
      const y = box.center[1] + (box.half[1] - up[0] * px - up[2] * pz) / up[1];
      if (y <= h) continue;
      const dy = y - box.center[1];
      const l0 = px * box.axes[0][0] + dy * box.axes[0][1] + pz * box.axes[0][2];
      const l2 = px * box.axes[2][0] + dy * box.axes[2][1] + pz * box.axes[2][2];
      if (Math.abs(l0) <= box.half[0] && Math.abs(l2) <= box.half[2]) {
        h = y;
        nx = up[0]; ny = up[1]; nz = up[2];
        out.material = 3;
      }
    }
    out.height = h;
    out.normal[0] = nx; out.normal[1] = ny; out.normal[2] = nz;
    if (out.material === undefined) out.material = this.materialIdAt(x, z);
    return out;
  }

  /** spawn transform on the road: {position:[x,y,z], yaw} */
  spawnOnTrack(t = 0.0) {
    const i = Math.floor(t * this.road.pts.length) % this.road.pts.length;
    const p = this.road.pts[i];
    const tan = this.road.tangents[i];
    const yaw = Math.atan2(tan[0], tan[1]);
    return {
      position: [p[0], this.heightAt(p[0], p[1]) + 0.05, p[1]],
      yaw,
      index: i,
    };
  }
}
