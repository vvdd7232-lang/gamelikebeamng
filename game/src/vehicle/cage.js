// Builds the soft-body node cage + beams from the generated cage data.
// Shared by the physics and by the mesh skinning so node indices always match.

import { CAGE } from './tesla_nodes.js';

// Geometry is taken straight from the model: the wheel centre sits at local
// y = 0.3599 and the tyre radius is 0.3576, the underbody at y = 0.0997.
const WHEEL_CENTRE_Y = 0.3599;

export const CAR = {
  wheelRadius: 0.3576,
  wheelWidth: 0.28,
  halfTrackFront: 0.8061,
  halfTrackRear: 0.8403,
  axleFront: 1.5696,
  axleRear: -1.4137,
  // cage is lifted slightly so the floor has ground clearance at ride height
  cageLift: 0.0,
  suspRest: 0.13,
  suspTravel: 0.14,
  // attach point (top of the strut) in car space
  hubY: WHEEL_CENTRE_Y + 0.13,
  length: 4.99,
  width: 2.19,
  height: 1.44,
};
// distance from the car origin down to the tyre contact patch at rest load
CAR.staticComp = 0.044;
CAR.restOffset = CAR.hubY - CAR.suspRest + CAR.staticComp - CAR.wheelRadius;

export const WHEEL_DEFS = [
  { name: 'FL', x: CAR.halfTrackFront, z: CAR.axleFront, front: true, left: true },
  { name: 'FR', x: -CAR.halfTrackFront, z: CAR.axleFront, front: true, left: false },
  { name: 'RL', x: CAR.halfTrackRear, z: CAR.axleRear, front: false, left: true },
  { name: 'RR', x: -CAR.halfTrackRear, z: CAR.axleRear, front: false, left: false },
];

const BEAM = {
  ringK: 2.6e5,
  longK: 2.2e5,
  diagK: 1.5e5,
  ringDamp: 900,
  longDamp: 800,
  diagDamp: 600,
  yieldStrain: 0.045,
  breakStrain: 0.26,
  breakForce: 160000,
};

export function buildCage() {
  const rings = CAGE.rings;
  const sectors = rings[0].length;
  const nr = rings.length;
  const nodes = [];
  const ringsIdx = [];
  for (let r = 0; r < nr; r++) {
    const idx = [];
    for (let j = 0; j < sectors; j++) {
      const p = rings[r][j];
      idx.push(nodes.length);
      nodes.push([p[0], p[1] + CAR.cageLift, p[2]]);
    }
    ringsIdx.push(idx);
  }
  const n = nodes.length;
  const pos = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = nodes[i][0];
    pos[i * 3 + 1] = nodes[i][1];
    pos[i * 3 + 2] = nodes[i][2];
  }
  let cx = 0, cy = 0, cz = 0;
  for (const p of nodes) { cx += p[0]; cy += p[1]; cz += p[2]; }
  const centroid = [cx / n, cy / n, cz / n];

  const beams = [];
  const seen = new Set();
  const key = (a, b) => (a < b ? a * 4096 + b : b * 4096 + a);
  const addBeam = (a, b, k, damp) => {
    if (a === b) return;
    const kk = key(a, b);
    if (seen.has(kk)) return;
    const d = Math.hypot(
      pos[a * 3] - pos[b * 3],
      pos[a * 3 + 1] - pos[b * 3 + 1],
      pos[a * 3 + 2] - pos[b * 3 + 2],
    );
    if (d < 0.05) return;
    seen.add(kk);
    // longer beams are softer, short ones stiff -> similar buckling behaviour
    const scale = Math.min(1.5, Math.max(0.5, 0.55 / Math.max(d, 0.1)));
    beams.push({
      a, b, rest: d,
      k: k * scale, damp: damp * scale,
      yieldStrain: BEAM.yieldStrain,
      breakStrain: BEAM.breakStrain,
      breakForce: BEAM.breakForce * Math.min(1.5, scale),
      len: d,
    });
  };

  // ring beams (around each section) -- these carry the "shell" loads
  for (let r = 0; r < nr; r++) {
    const ring = ringsIdx[r];
    for (let j = 0; j < sectors; j++) {
      addBeam(ring[j], ring[(j + 1) % sectors], BEAM.ringK, BEAM.ringDamp);
      // shear braces inside the section (skip 3) keep the ring from flopping
      addBeam(ring[j], ring[(j + 3) % sectors], BEAM.ringK * 0.30, BEAM.ringDamp * 0.4);
    }
  }
  // longitudinal beams
  for (let r = 0; r < nr - 1; r++) {
    const A = ringsIdx[r], B = ringsIdx[r + 1];
    for (let j = 0; j < sectors; j++) {
      addBeam(A[j], B[j], BEAM.longK, BEAM.longDamp);
      // diagonals -- this is what makes the tube a torsion box
      addBeam(A[j], B[(j + 1) % sectors], BEAM.diagK, BEAM.diagDamp);
      addBeam(A[(j + 1) % sectors], B[j], BEAM.diagK, BEAM.diagDamp);
    }
  }
  // a few long rails for overall bending stiffness
  for (let r = 0; r < nr - 2; r++) {
    const A = ringsIdx[r], C = ringsIdx[r + 2];
    for (let j = 0; j < sectors; j += 2) addBeam(A[j], C[j], BEAM.longK * 0.22, BEAM.longDamp * 0.3);
  }

  // node groups for fitting the body basis
  const group = { front: [], rear: [], up: [], down: [] };
  for (let i = 0; i < n; i++) {
    const z = pos[i * 3 + 2], y = pos[i * 3 + 1];
    if (z > 1.35) group.front.push(i);
    if (z < -1.35) group.rear.push(i);
    if (y > 1.05) group.up.push(i);
    if (y < 0.32) group.down.push(i);
  }

  // per-node neighbours used to estimate local rotation
  const neighbours = [];
  for (let i = 0; i < n; i++) {
    const nb = beams.filter((b) => b.a === i || b.b === i)
      .map((b) => ({ other: b.a === i ? b.b : b.a, len: b.len }));
    neighbours.push(nb);
  }

  return { nodes, pos, count: n, beams, centroid, group, neighbours, rings: ringsIdx, sectors, ringCount: nr };
}
