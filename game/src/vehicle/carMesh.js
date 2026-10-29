// Turns the loaded glb into GPU geometry.
//
// The car is authored as a *set of parts* -- shell, doors, hood, trunk, bumpers,
// interior, calipers, wheels -- each with its own pivot.  That is exactly what a
// damage system wants: a door is a thing that hangs on a hinge and can be ripped
// off, not a few thousand anonymous triangles welded to the body.
//
//   bodies[]  skinned primitives, positioned in *car space* (pivot applied) so
//             one skinning path covers every attached part
//   parts[]   the logical pieces: name, role, pivot, hinge axis, mass, the
//             cage nodes that hold them on, and the index range of `bodies`
//   wheels[]  rigid meshes, kept local to their own centre
//   calipers[] rigid meshes in car space, riding the hub
import { CAR } from './cage.js';

const MAX_JOINTS = 4;

// surface kinds, kept in sync with MESH_FS
const KIND_PAINT = 0;
const KIND_LIGHT = 1;
const KIND_GLASS = 2;
const KIND_CARPAINT = 3;

// Everything that is a window rather than body trim.  The Volvo asset names its
// glass materials per lamp / window, and the model is split by part, so a
// triangle is a window only when *all three* corners sit above the belt line --
// mixing kinds per vertex tears the surface into a speckled mess.
const GLASS_MATERIALS = new Set([
  'Glass', 'Glass_wavey', 'heaxagon glass', 'Glass Taillight', 'Black Glass',
  'Glass headlights', 'GlassRunninglight', 'Translucent_Glass', 'Translucent_Glass.001',
]);
const GLASS_TAIL_MATERIALS = new Set(['Glass Taillight', 'GlassRunninglight']);
const BELT_Y = 0.94;

const PAINT_MATERIALS = new Set(['Car Paint']);

// roles: how a part behaves
const ROLE_BODY = 'body';
const ROLE_PANEL = 'panel';
const ROLE_WHEEL = 'wheel';
const ROLE_CALIPER = 'caliper';
const ROLE_STATIC = 'static';

// A hinge is described in car space: the pivot comes from the model (the
// artist left every panel's origin sitting on its hinge line), the axis is
// implied by what the panel is.
const HINGES = {
  Hood: { axis: [1, 0, 0], side: 1, open: -0.85, max: 1.35, k: 260, damp: 16, mass: 16 },
  Trunk: { axis: [1, 0, 0], side: 1, open: 0.75, max: -1.20, k: 230, damp: 15, mass: 15 },
  Door_FL: { axis: [0, 1, 0], side: 1, open: -1.15, max: -1.55, k: 420, damp: 26, mass: 24 },
  Door_FR: { axis: [0, 1, 0], side: -1, open: 1.15, max: 1.55, k: 420, damp: 26, mass: 24 },
  Door_RL: { axis: [0, 1, 0], side: 1, open: -1.10, max: -1.45, k: 380, damp: 24, mass: 20 },
  Door_RR: { axis: [0, 1, 0], side: -1, open: 1.10, max: 1.45, k: 380, damp: 24, mass: 20 },
  // bumpers do not swing on a hinge; they hang off the shell and tear away
  Bumper_Front: { axis: [1, 0, 0], side: 1, open: 0.55, max: 0.85, k: 900, damp: 40, mass: 14, rigid: true },
  Bumper_Rear: { axis: [1, 0, 0], side: -1, open: 0.55, max: 0.85, k: 900, damp: 40, mass: 12, rigid: true },
};

const ROLE_BY_NAME = (name) => {
  if (name === 'Body') return ROLE_BODY;
  if (name === 'Interior' || name === 'SteeringWheel') return ROLE_STATIC;
  if (name.startsWith('Wheel_')) return ROLE_WHEEL;
  if (name.startsWith('BrakeCaliper_')) return ROLE_CALIPER;
  if (HINGES[name]) return ROLE_PANEL;
  return ROLE_STATIC;
};

/** uniform grid over the cage nodes -- a linear scan per vertex is ~100x
 *  slower and this runs over a third of a million vertices at load. */
function makeNodeGrid(nodes, cell = 0.45) {
  const map = new Map();
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const key = `${Math.floor(n[0] / cell)},${Math.floor(n[1] / cell)},${Math.floor(n[2] / cell)}`;
    let bucket = map.get(key);
    if (!bucket) { bucket = []; map.set(key, bucket); }
    bucket.push(i);
  }
  return { cell, map, nodes };
}

function nearestIndices(grid, x, y, z, k) {
  const { cell, map, nodes } = grid;
  const out = [];
  const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
  for (let r = 0; r <= 6 && out.length < k; r++) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dz = -r; dz <= r; dz++) {
          // only the shell of the box is new
          if (r > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
          const bucket = map.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!bucket) continue;
          for (const i of bucket) {
            const n = nodes[i];
            const ddx = n[0] - x, ddy = n[1] - y, ddz = n[2] - z;
            out.push([ddx * ddx + ddy * ddy + ddz * ddz, i]);
          }
        }
      }
    }
  }
  out.sort((a, b) => a[0] - b[0]);
  return out.slice(0, k);
}

/** nearest cage nodes for a car-space point, as skinning weights */
function weightsAt(grid, x, y, z, k) {
  const best = nearestIndices(grid, x, y, z, k);
  const idx = [];
  const w = [];
  let sum = 0;
  for (const [d2, i] of best) {
    const v = 1 / (Math.sqrt(d2) + 0.08);
    idx.push(i);
    w.push(v);
    sum += v;
  }
  if (!sum) return { idx: [0, 0, 0, 0], w: [1, 0, 0, 0] };
  return { idx, w: w.map((v) => v / sum) };
}

export function buildCarGeometry(glb, cageNodes) {
  const STRIDE = 20; // pos3 nrm3 col3 mat3 joints4 weights4
  const grid = makeNodeGrid(cageNodes);
  const bodies = [];
  const parts = [];
  const wheelGeom = [];
  const calipers = [];

  // The model ships glossy materials that cover windows *and* trim.  Classify
  // per triangle, not per vertex.
  const triKinds = (prim, mat) => {
    const pos = prim.positions;
    const idx = prim.indices;
    const np = pos.length / 3;
    if (!GLASS_MATERIALS.has(mat.name) || !idx) return null;
    const out = new Uint8Array(np);
    for (let t = 0; t + 2 < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      const above = pos[a * 3 + 1] > BELT_Y && pos[b * 3 + 1] > BELT_Y && pos[c * 3 + 1] > BELT_Y;
      if (above) { out[a] = 1; out[b] = 1; out[c] = 1; }
    }
    return out;
  };

  for (const node of glb.nodes) {
    const role = ROLE_BY_NAME(node.name);
    if (!node.primitives || !node.primitives.length) continue;
    const pivot = node.translation ? node.translation.slice() : [0, 0, 0];
    // everything except the wheels is skinned to the cage -- including the
    // brake calipers, which then follow the deformed arch instead of floating
    const skinned = role !== ROLE_WHEEL;
    const hinge = HINGES[node.name] || null;
    const part = (role === ROLE_PANEL || role === ROLE_STATIC) ? {
      name: node.name,
      role,
      pivot,
      hinge: hinge ? { axis: hinge.axis, open: hinge.open, max: hinge.max, k: hinge.k, damp: hinge.damp } : null,
      mass: hinge ? hinge.mass : 6,
      firstPrim: bodies.length,
      prims: 0,
      area: 0,
      com: [0, 0, 0],
      nodes: [],
      weights: [],
    } : null;

    let areaSum = 0;
    const comSum = [0, 0, 0];

    for (const prim of node.primitives) {
      const pos = prim.positions;
      const nor = prim.normals;
      const idx = prim.indices;
      const mat = prim.material || { color: [0.5, 0.5, 0.5], metallic: 0.2, roughness: 0.6, emissive: [0, 0, 0], alpha: 1 };
      const np = pos.length / 3;
      const out = new Float32Array(np * STRIDE);
      const glassVert = skinned ? triKinds(prim, mat) : null;
      const paint = PAINT_MATERIALS.has(mat.name);
      const px = skinned ? pivot[0] : 0;
      const py = skinned ? pivot[1] : 0;
      const pz = skinned ? pivot[2] : 0;
      for (let v = 0; v < np; v++) {
        const x = pos[v * 3] + px, y = pos[v * 3 + 1] + py, z = pos[v * 3 + 2] + pz;
        const o = v * STRIDE;
        out[o] = x; out[o + 1] = y; out[o + 2] = z;
        const nx = nor ? nor[v * 3] : 0, ny = nor ? nor[v * 3 + 1] : 1, nz = nor ? nor[v * 3 + 2] : 0;
        out[o + 3] = nx; out[o + 4] = ny; out[o + 5] = nz;
        let cr = mat.color[0], cg = mat.color[1], cb = mat.color[2];
        if (paint) { cr = 0.40; cg = 0.045; cb = 0.055; }
        out[o + 6] = cr; out[o + 7] = cg; out[o + 8] = cb;
        out[o + 9] = mat.metallic;
        out[o + 10] = mat.roughness;
        const emis = (mat.emissive[0] + mat.emissive[1] + mat.emissive[2]) > 0.01;
        const glass = glassVert ? glassVert[v] === 1
          : (GLASS_MATERIALS.has(mat.name) && y > BELT_Y && !GLASS_TAIL_MATERIALS.has(mat.name));
        out[o + 11] = emis ? KIND_LIGHT
          : glass ? KIND_GLASS
          : paint ? KIND_CARPAINT
          : KIND_PAINT;
        if (skinned) {
          const sw = weightsAt(grid, x, y, z, MAX_JOINTS);
          for (let j = 0; j < MAX_JOINTS; j++) {
            out[o + 12 + j] = sw.idx[j] !== undefined ? sw.idx[j] : 0;
            out[o + 16 + j] = sw.w[j] !== undefined ? sw.w[j] : 0;
          }
        } else {
          for (let j = 0; j < MAX_JOINTS; j++) { out[o + 12 + j] = 0; out[o + 16 + j] = 0; }
        }
      }
      // surface area + centroid, used for the mass and the centre of mass of a
      // panel (it decides how hard gravity pulls a hinged door open)
      if (skinned && part) {
        const step = idx ? idx.length / 3 : np / 3;
        for (let t = 0; t < step; t++) {
          const a = idx ? idx[t * 3] : t * 3;
          const b = idx ? idx[t * 3 + 1] : t * 3 + 1;
          const c = idx ? idx[t * 3 + 2] : t * 3 + 2;
          const ax = pos[a * 3] + px, ay = pos[a * 3 + 1] + py, az = pos[a * 3 + 2] + pz;
          const bx = pos[b * 3] + px, by = pos[b * 3 + 1] + py, bz = pos[b * 3 + 2] + pz;
          const cx2 = pos[c * 3] + px, cy2 = pos[c * 3 + 1] + py, cz2 = pos[c * 3 + 2] + pz;
          const ux = bx - ax, uy = by - ay, uz = bz - az;
          const vx = cx2 - ax, vy = cy2 - ay, vz = cz2 - az;
          const cxx = uy * vz - uz * vy, cyy = uz * vx - ux * vz, czz = ux * vy - uy * vx;
          const ar = 0.5 * Math.hypot(cxx, cyy, czz);
          areaSum += ar;
          comSum[0] += ar * (ax + bx + cx2) / 3;
          comSum[1] += ar * (ay + by + cy2) / 3;
          comSum[2] += ar * (az + bz + cz2) / 3;
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
      if (skinned) {
        bodies.push(rows);
        if (part) part.prims++;
      } else if (role === ROLE_WHEEL) {
        let g = wheelGeom.find((w) => w.name === node.name);
        if (!g) { g = { name: node.name, role, pivot, prims: [] }; wheelGeom.push(g); }
        g.prims.push(rows);
      } else if (role === ROLE_CALIPER) {
        let g = calipers.find((w) => w.name === node.name);
        if (!g) { g = { name: node.name, role, pivot, wheel: node.name.slice(13), prims: [] }; calipers.push(g); }
        g.prims.push(rows);
      }
    }

    if (part) {
      if (areaSum > 0) {
        part.area = areaSum;
        part.com = [comSum[0] / areaSum, comSum[1] / areaSum, comSum[2] / areaSum];
      } else {
        part.com = pivot.slice();
      }
      if (role === ROLE_PANEL && !part.hinge) part.mass = Math.max(4, areaSum * 6);
      if (part.hinge) part.mass = Math.max(4, Math.min(part.mass, Math.max(4, areaSum * 8)));
      // the cage nodes that hold this panel on: sampled along its hinge line
      // (or, for a part with no hinge, around its centre of mass)
      const anchors = [];
      if (part.hinge) {
        const a = part.hinge.axis;
        // two ends of the hinge plus the middle
        const span = Math.max(0.35, Math.min(1.3, Math.sqrt(Math.max(areaSum, 0.2)) * 0.5));
        for (const t of [-1, 0, 1]) {
          anchors.push([
            pivot[0] + (a[0] ? t * span : 0),
            pivot[1] + (a[1] ? t * span : 0),
            pivot[2] + (a[2] ? t * span : 0),
          ]);
        }
        anchors.push(part.com);
      } else {
        anchors.push(part.com, pivot);
      }
      const nodeSet = new Set();
      for (const p of anchors) {
        for (const [d2, i] of nearestIndices(grid, p[0], p[1], p[2], 4)) { void d2; nodeSet.add(i); }
      }
      part.nodes = [...nodeSet];
      // and the cage nodes the whole panel is stitched to (drives its shading
      // and how badly it is bent)
      const bodyNodes = new Set();
      for (const p of [part.com, pivot]) {
        for (const [d2, i] of nearestIndices(grid, p[0], p[1], p[2], 6)) { void d2; bodyNodes.add(i); }
      }
      part.bodyNodes = [...bodyNodes];
      parts.push(part);
    }
  }

  return { bodies, parts, wheels: wheelGeom, calipers };
}

/** attach positions of the wheels (car space) */
export function wheelOffsets(glb) {
  const out = {};
  for (const n of glb.nodes) {
    if (n.name.startsWith('Wheel_')) out[n.name.slice(6)] = n.translation;
  }
  return out;
}

export const PART_ROLES = { ROLE_BODY, ROLE_PANEL, ROLE_WHEEL, ROLE_CALIPER, ROLE_STATIC };
