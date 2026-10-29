// Builds the soft-body node cage + beams from the generated cage data.
// Shared by the physics and by the mesh skinning so node indices always match.

import { CAGE } from './car_nodes.js';

// Geometry is taken straight from the model (Volvo S90): the wheel centres sit
// at y = 0.359 and the tyre radius is 0.36, the shell is 5.00 m long and 2.01 m
// wide, and the roof tops out at 1.43 m.
const WHEEL_CENTRE_Y = 0.359;

export const CAR = {
  wheelRadius: 0.36,
  wheelWidth: 0.25,
  halfTrackFront: 0.816,
  halfTrackRear: 0.8135,
  axleFront: 1.61,
  axleRear: -1.33,
  // cage is lifted slightly so the floor has ground clearance at ride height
  cageLift: 0.0,
  suspRest: 0.13,
  suspTravel: 0.14,
  // attach point (top of the strut) in car space
  hubY: WHEEL_CENTRE_Y + 0.13,
  length: 5.0,
  width: 2.012,
  height: 1.43,
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

// Beam stiffness.  These used to be ~8x softer, which made the whole body fold
// like a rag: a 100 km/h hit crushed the car from 4.7 m to 2.9 m because the
// beams could not carry the deceleration into the rest of the structure.  Real
// car structures are far stiffer than a 1.5 t car needs to sag 5 cm, and the
// solver is implicit (unconditionally stable), so the stiffness is free.
const BEAM = {
  ringK: 2.1e6,
  longK: 1.76e6,
  diagK: 1.2e6,
  ringDamp: 2600,
  longDamp: 2300,
  diagDamp: 1700,
  yieldStrain: 0.045,
  breakStrain: 0.26,
  // scaled with the stiffness so a beam still tears at the same *strain*
  breakForce: 1.28e6,
};

/**
 * Crush zones.
 *
 * A real car is not a uniform steel shell: the front and rear ends are designed
 * to fold up and eat the impact energy, while the passenger cell around the
 * seats is a stiff safety cage that must keep its shape.  Giving every beam the
 * same yield/break limits (what this used to do) meant a 100 km/h hit turned the
 * whole car -- roof included -- into the same amorphous ball.
 *
 *  - `k`:           stiffness multiplier
 *  - `yield`:       plastic yield strain multiplier (tension / compression)
 *  - `break`:       strain at which the beam tears (tension / compression)
 *  - `force`:       force at which the beam tears
 *  - `plastic`:     how fast the permanent set accumulates (crumple zones fold,
 *                   the cage mostly springs back)
 *  - `damageLoss`:  how much stiffness a fully damaged beam keeps
 */
const ZONES = {
  crumple: {
    // yield sits above the strain gravity puts into the front overhang, so a
    // parked car never slowly "yields"; the crush limits below still let a hit
    // fold the clip right up
    k: 0.5, yield: [1.45, 1.15], break: [1.90, 2.40], force: 80.0, plastic: 2.4, damageLoss: 0.6,
    // the clip folds up like a concertina and stays attached to the car; it
    // only tears off in a really violent hit
    setC: 0.78, setT: 0.70,
  },
  cabin: {
    k: 1.12, yield: [2.10, 1.70], break: [1.55, 1.30], force: 2.40, plastic: 0.45, damageLoss: 0.32,
    setC: 0.20, setT: 0.40,
  },
  // longitudinal beams running from a crumple zone into the cage: they are
  // the hinges the front end folds around, so they sit between the two
  boundary: {
    k: 0.75, yield: [1.0, 0.85], break: [1.35, 1.55], force: 1.30, plastic: 1.0, damageLoss: 0.42,
    setC: 0.40, setT: 0.55,
  },
};

/** which crush zone a beam midpoint belongs to */
function zoneOf(z, y) {
  // the roof rails over the passenger cell are always part of the cage
  if (y > 1.00) return 'cabin';
  if (z > 1.32) return 'crumple';
  if (z < -1.58) return 'crumple';
  if (z > 0.92 || z < -1.18) return 'boundary';
  return 'cabin';
}

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
    const mz = (pos[a * 3 + 2] + pos[b * 3 + 2]) * 0.5;
    const my = (pos[a * 3 + 1] + pos[b * 3 + 1]) * 0.5;
    const zoneName = zoneOf(mz, my);
    const zone = ZONES[zoneName];
    beams.push({
      a, b, rest: d,
      k: k * scale * zone.k, damp: damp * scale,
      // tension / compression limits are different: sheet metal folds much
      // sooner than it tears
      yieldStrain: BEAM.yieldStrain * zone.yield[0],
      yieldStrainC: BEAM.yieldStrain * zone.yield[1],
      breakStrain: BEAM.breakStrain * zone.break[0],
      breakStrainC: BEAM.breakStrain * zone.break[1],
      breakForce: BEAM.breakForce * Math.min(1.5, scale) * zone.force,
      breakForceC: BEAM.breakForce * Math.min(1.5, scale) * zone.force * 1.1,
      plasticRate: zone.plastic,
      damageLoss: zone.damageLoss,
      setMaxC: zone.setC,
      setMaxT: zone.setT,
      zone: zoneName,
      len: d,
      // joint of the cage, used to measure driver/passenger cell integrity
      cabin: (mz > -1.05 && mz < 0.85) || my > 0.98,
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
