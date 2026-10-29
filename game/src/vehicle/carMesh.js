// Turns the loaded glb into GPU geometry:
//  * body: interleaved vertices with soft-body skinning weights (4 joints)
//  * wheels: rigid, shared between cars
import { CAR } from './cage.js';

const MAX_JOINTS = 4;

// surface kinds, kept in sync with MESH_FS
const KIND_PAINT = 0;
const KIND_LIGHT = 1;
const KIND_GLASS = 2;
const KIND_CARPAINT = 3;
// The greenhouse is the 'glass' material.  'glass.front' is a glossy layer the
// modeller laid over the windscreen *and* down the body sides, so it is only
// treated as glass above the belt line -- below that it is body trim.
const GLASS_MATERIALS = new Set(['glass', 'glass.dark']);
const GLASS_ABOVE_Y = 0.92;

/** nearest cage nodes for a local-space point (k nearest) */
function nearestWeights(nodes, x, y, z, k) {
  const best = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const dx = n[0] - x, dy = n[1] - y, dz = n[2] - z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (best.length < k) {
      best.push([d2, i]);
      best.sort((a, b) => a[0] - b[0]);
    } else if (d2 < best[best.length - 1][0]) {
      best[best.length - 1] = [d2, i];
      best.sort((a, b) => a[0] - b[0]);
    }
  }
  // inverse distance weights
  const w = [];
  let sum = 0;
  for (const [d2, i] of best) {
    const v = 1 / (Math.sqrt(d2) + 0.08);
    w.push([i, v]);
    sum += v;
  }
  return { idx: best.map((b) => b[1]), w: w.map(([i, v]) => v / sum) };
}

export function buildCarGeometry(glb, cageNodes) {
  const STRIDE = 20; // pos3 nrm3 col3 mat3 joints4 weights4
  const bodies = [];
  const wheels = [];
  const wheelGeom = [];

  // The model ships a few glossy "glass" materials that also cover body panels.
  // A triangle is a window only if *all* of it sits above the belt line; mixing
  // kinds per vertex tears the surface into a speckled mess.
  const triKinds = (prim, mat) => {
    const pos = prim.positions;
    const idx = prim.indices;
    const np = pos.length / 3;
    const isGlassMat = GLASS_MATERIALS.has(mat.name) || mat.name === 'glass.front';
    if (!isGlassMat || !idx) return null;
    const out = new Uint8Array(np);
    const belt = mat.name === 'glass.front' ? GLASS_ABOVE_Y : 0.9;
    for (let t = 0; t + 2 < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      const above = pos[a * 3 + 1] > belt && pos[b * 3 + 1] > belt && pos[c * 3 + 1] > belt;
      if (above) { out[a] = 1; out[b] = 1; out[c] = 1; }
    }
    return out;
  };

  for (const node of glb.nodes) {
    const isBody = node.name === 'Body';
    for (const prim of node.primitives) {
      const pos = prim.positions;
      const nor = prim.normals;
      const idx = prim.indices;
      const mat = prim.material || { color: [0.5, 0.5, 0.5], metallic: 0.2, roughness: 0.6, emissive: [0, 0, 0], alpha: 1 };
      const np = pos.length / 3;
      const out = new Float32Array(np * STRIDE);
      const glassVert = isBody ? triKinds(prim, mat) : null;
      for (let v = 0; v < np; v++) {
        const x = pos[v * 3], y = pos[v * 3 + 1], z = pos[v * 3 + 2];
        const o = v * STRIDE;
        out[o] = x; out[o + 1] = y; out[o + 2] = z;
        const nx = nor ? nor[v * 3] : 0, ny = nor ? nor[v * 3 + 1] : 1, nz = nor ? nor[v * 3 + 2] : 0;
        out[o + 3] = nx; out[o + 4] = ny; out[o + 5] = nz;
        let cr = mat.color[0], cg = mat.color[1], cb = mat.color[2];
        if (isBody && mat.name === 'Tesla') { cr = 0.62; cg = 0.07; cb = 0.07; }
        out[o + 6] = cr; out[o + 7] = cg; out[o + 8] = cb;
        out[o + 9] = mat.metallic;
        out[o + 10] = mat.roughness;
        // aMat.z: what the shader is looking at (must match MESH_FS)
        const emis = (mat.emissive[0] + mat.emissive[1] + mat.emissive[2]) > 0.01;
        const glass = glassVert ? glassVert[v] === 1
          : (GLASS_MATERIALS.has(mat.name) || (mat.name === 'glass.front' && y > GLASS_ABOVE_Y));
        out[o + 11] = emis ? KIND_LIGHT
          : glass ? KIND_GLASS
          : (isBody && mat.name === 'Tesla') ? KIND_CARPAINT
          : KIND_PAINT;
        if (isBody) {
          const sw = nearestWeights(cageNodes, x, y, z, MAX_JOINTS);
          for (let j = 0; j < MAX_JOINTS; j++) {
            out[o + 12 + j] = sw.idx[j] !== undefined ? sw.idx[j] : 0;
            out[o + 16 + j] = sw.w[j] !== undefined ? sw.w[j] : 0;
          }
        } else {
          for (let j = 0; j < MAX_JOINTS; j++) { out[o + 12 + j] = 0; out[o + 16 + j] = 0; }
        }
      }
      const rows = {
        data: out,
        stride: STRIDE,
        indices: idx ? idx : null,
        vertexCount: np,
        material: mat,
        name: node.name,
        emissive: (mat.emissive[0] + mat.emissive[1] + mat.emissive[2]) > 0.01,
      };
      if (isBody) bodies.push(rows);
      else {
        let g = wheelGeom.find((w) => w.name === node.name);
        if (!g) { g = { name: node.name, prims: [] }; wheelGeom.push(g); }
        g.prims.push(rows);
      }
    }
  }
  return { bodies, wheels: wheelGeom };
}

/** attach/detach positions of the wheels (car space) */
export function wheelOffsets(glb) {
  const out = {};
  for (const n of glb.nodes) {
    if (n.name.startsWith('Wheel_')) out[n.name.slice(6)] = n.translation;
  }
  return out;
}
