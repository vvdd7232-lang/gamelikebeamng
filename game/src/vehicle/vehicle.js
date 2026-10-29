// The car: soft-body cage + raycast-style suspension + tyre model + electric
// drivetrain.  Modelled on BeamNG: nothing about the car is rigid, the wheels
// push the deformable cage through spring/damper suspension, and impacts
// plastically deform (or break) the beams.

import { SoftBody } from '../physics/softbody.js';
import { buildCage, WHEEL_DEFS, CAR } from './cage.js';
import { clamp, lerp } from '../core/math.js';
import { MATERIALS } from '../world/terrain.js';

const SUB_DT = 1 / 360;

export const DEFAULT_SPEC = {
  mass: 1520,             // structural mass of the cage
  wheelMass: 38,
  suspKFront: 86000,
  suspKRear: 79000,
  suspDampBump: 3400,
  suspDampRebound: 5200,
  arbFront: 14000,
  arbRear: 9000,
  maxSteerDeg: 33,
  steerSpeed: 3.2,
  powerKW: 410,
  maxWheelForce: 13500,
  brakeForce: 34000,
  brakeBias: 0.62,
  gripAsphalt: 1.28,
  gripGrass: 0.72,
  tyrePeakSlip: 0.13,
  downforce: 0.55,
  dragArea: 0.72,
  rollingResistance: 9.0,
  damageFromBeams: true,
  beamScale: 1.0,
};

const _tmp = [0, 0, 0];
const _tmp2 = [0, 0, 0];
const _tmp3 = [0, 0, 0];

export class Vehicle {
  constructor(world, opts = {}) {
    this.world = world;
    this.spec = { ...DEFAULT_SPEC, ...(opts.spec || {}) };
    this.name = opts.name || 'Tesla';
    this.color = opts.color || [0.62, 0.06, 0.06];
    this.isPlayer = !!opts.isPlayer;

    const spawn = opts.spawn || world.spawnOnTrack(0);
    this.spawnPos = spawn.position.slice();
    this.spawnYaw = spawn.yaw;
    // place the car so the tyres just touch the ground under each wheel
    {
      let groundMax = -Infinity;
      const cy = Math.cos(spawn.yaw), sy = Math.sin(spawn.yaw);
      for (const d of [-1.57, 1.57, -1.41, 0.84, -0.84]) {
        for (const lx of [0.8, -0.8]) {
          const wx = spawn.position[0] + lx * cy + d * sy;
          const wz = spawn.position[2] - lx * sy + d * cy;
          groundMax = Math.max(groundMax, world.heightAt(wx, wz));
        }
      }
      this.spawnPos[1] = groundMax - CAR.restOffset;
      this.spawnPos[0] = spawn.position[0];
      this.spawnPos[2] = spawn.position[2];
    }

    this.cage = buildCage();
    const n = this.cage.count;
    this.nodeCount = n;

    const cosY = Math.cos(spawn.yaw), sinY = Math.sin(spawn.yaw);
    const toWorld = (x, y, z) => [
      spawn.position[0] + x * cosY + z * sinY,
      spawn.position[1] + y,
      spawn.position[2] - x * sinY + z * cosY,
    ];
    this.toWorld = toWorld;

    // rest positions in world space
    const restWorld = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = toWorld(this.cage.pos[i * 3], this.cage.pos[i * 3 + 1], this.cage.pos[i * 3 + 2]);
      restWorld[i * 3] = p[0];
      restWorld[i * 3 + 1] = p[1];
      restWorld[i * 3 + 2] = p[2];
    }
    this.restWorld = restWorld;
    this.R0 = [
      cosY, 0, sinY,
      0, 1, 0,
      -sinY, 0, cosY,
    ];
    let rcx = 0, rcy = 0, rcz = 0;
    for (let i = 0; i < n; i++) { rcx += restWorld[i * 3]; rcy += restWorld[i * 3 + 1]; rcz += restWorld[i * 3 + 2]; }
    this.restCentroid = [rcx / n, rcy / n, rcz / n];
    this.R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    this.rotQuat = [1, 0, 0, 0];
    this.Rinv = [1, 0, 0, 0, 1, 0, 0, 0, 1];

    const invMass = new Float64Array(n);
    const per = this.spec.mass / n;
    for (let i = 0; i < n; i++) invMass[i] = 1 / per;
    const beams = this.cage.beams.map((b) => ({ ...b, k: b.k * this.spec.beamScale }));
    this.body = new SoftBody(restWorld, invMass, beams);

    // per-node rotation estimation triples
    this.nodeTris = this.buildNodeTriangles();

    // reference directions (rest world) used to fit the body rotation
    {
      const g = this.cage.group;
      const meanRest = (list) => {
        let x = 0, y = 0, z = 0;
        for (const i of list) { x += restWorld[i * 3]; y += restWorld[i * 3 + 1]; z += restWorld[i * 3 + 2]; }
        const l = list.length || 1;
        return [x / l, y / l, z / l];
      };
      const f = meanRest(g.front), r = meanRest(g.rear);
      const u = meanRest(g.up), d = meanRest(g.down);
      const r1 = [f[0] - r[0], f[1] - r[1], f[2] - r[2]];
      const r2 = [u[0] - d[0], u[1] - d[1], u[2] - d[2]];
      const r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
      ];
      this.restRefs = [r1, r2, r3];
      this.restRefsInv = inv3(r1, r2, r3);
    }

    // wheels
    this.wheels = WHEEL_DEFS.map((d, i) => ({
      ...d, index: i,
      steer: 0, steerTarget: 0, spin: 0, spinRate: 0,
      compression: 0, contact: false, surfaceY: 0,
      pos: [0, 0, 0], contactPoint: [0, 0, 0], normal: [0, 1, 0],
      load: 0, slip: 0, skid: 0, detached: false, angularVel: 0,
    }));

    // cached node neighbourhoods for each wheel (attach / contact weights)
    this.cageToWorldInit = (lx, ly, lz) => {
      const wx = spawn.position[0] + lx * cosY + lz * sinY + 0;
      const wy = spawn.position[1] + ly;
      const wz = spawn.position[2] - lx * sinY + lz * cosY;
      return [wx, wy, wz];
    };
    for (const w of this.wheels) {
      const attLocal = [w.x, CAR.hubY, w.z];
      const conLocal = [w.x, CAR.hubY - CAR.suspRest - CAR.wheelRadius * 0.9, w.z];
      w.nodes = nearestNodes(this.cage.pos, this.nodeCount, attLocal, 9);
      w.wAttach = nodeWeights(this.cage.pos, attLocal, w.nodes);
      w.wContact = nodeWeights(this.cage.pos, conLocal, w.nodes);
    }

    this.nodeTex = new Float32Array(n * 4 * 4);
    this.basis = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
    this.center = [0, 0, 0];
    this.velocity = [0, 0, 0];
    this.speed = 0;
    this.forwardSpeed = 0;
    this.yawRate = 0;
    this.wheelSpinFast = 0;
    this.damage = 0;
    this.wheelDamage = 0;
    this.airborne = false;
    this.throttle = 0;
    this.brake = 0;
    this.handbrake = 0;
    this.steer = 0;
    this.detachedWheels = [];
    this.motorRPM = 0;
    this.drivePower = 0;
    this.gForce = [0, 0, 0];
    this.accelSmooth = [0, 0, 0];
    this.surfaceMat = 0;
    this.impactFlash = 0;
    this.lastImpact = 0;
    this.startBraking = 0;
    this.updateBasis();
    this.updateWheelVisuals(0);
    this.updateNodeTex();
  }

  // ------------------------------------------------------------- body basis
  /** Global rigid fit (Horn's quaternion method) of the deformed cage onto its
      rest shape.  Using every node keeps local crumpling from leaking into the
      body orientation (which would make the wheels feed energy back). */
  updateBasis() {
    const { pos } = this.body;
    const n = this.nodeCount;
    const rest = this.restWorld;
    const rc = this.restCentroid;

    // centroids
    let cx = 0, cy = 0, cz = 0, rx = 0, ry = 0, rz = 0;
    for (let i = 0; i < n; i++) {
      cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2];
      rx += rest[i * 3]; ry += rest[i * 3 + 1]; rz += rest[i * 3 + 2];
    }
    cx /= n; cy /= n; cz /= n;
    rx /= n; ry /= n; rz /= n;
    this.center[0] = cx; this.center[1] = cy; this.center[2] = cz;

    // cross covariance  H = sum (p - pc) * (r - rc)^T
    let sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;
    for (let i = 0; i < n; i++) {
      const px = pos[i * 3] - cx, py = pos[i * 3 + 1] - cy, pz = pos[i * 3 + 2] - cz;
      const qx = rest[i * 3] - rx, qy = rest[i * 3 + 1] - ry, qz = rest[i * 3 + 2] - rz;
      sxx += px * qx; sxy += px * qy; sxz += px * qz;
      syx += py * qx; syy += py * qy; syz += py * qz;
      szx += pz * qx; szy += pz * qy; szz += pz * qz;
    }
    // 4x4 symmetric matrix whose top eigenvector is the rotation quaternion
    const N = _N; // reused buffer
    N[0] = sxx + syy + szz; N[1] = syz - szy; N[2] = szx - sxz; N[3] = sxy - syx;
    N[4] = N[1]; N[5] = sxx - syy - szz; N[6] = sxy + syx; N[7] = szx + sxz;
    N[8] = N[2]; N[9] = N[6]; N[10] = -sxx + syy - szz; N[11] = syz + szy;
    N[12] = N[3]; N[13] = N[7]; N[14] = N[11]; N[15] = -sxx - syy + szz;

    // power iteration, warm started from the previous frame
    let q = this.rotQuat;
    let qn = _q;
    for (let iter = 0; iter < 6; iter++) {
      for (let i = 0; i < 4; i++) {
        qn[i] = N[i * 4] * q[0] + N[i * 4 + 1] * q[1] + N[i * 4 + 2] * q[2] + N[i * 4 + 3] * q[3];
      }
      const l = Math.hypot(qn[0], qn[1], qn[2], qn[3]);
      if (l < 1e-12) break;
      qn[0] /= l; qn[1] /= l; qn[2] /= l; qn[3] /= l;
      q[0] = qn[0]; q[1] = qn[1]; q[2] = qn[2]; q[3] = qn[3];
    }
    if (Math.hypot(q[0], q[1], q[2], q[3]) < 0.5) { q[0] = 0; q[1] = 0; q[2] = 0; q[3] = 1; }
    // Horn's quaternion maps rest->current; the matrix below is built from the
    // conjugate so that it maps rest->current directly (verified by unit test)
    const w = q[0], qx = -q[1], qy = -q[2], qz = -q[3];
    const R = this.R;
    R[0] = 1 - 2 * (qy * qy + qz * qz);
    R[1] = 2 * (qx * qy - qz * w);
    R[2] = 2 * (qx * qz + qy * w);
    R[3] = 2 * (qx * qy + qz * w);
    R[4] = 1 - 2 * (qx * qx + qz * qz);
    R[5] = 2 * (qy * qz - qx * w);
    R[6] = 2 * (qx * qz - qy * w);
    R[7] = 2 * (qy * qz + qx * w);
    R[8] = 1 - 2 * (qx * qx + qy * qy);

    const R0 = this.R0;
    for (let i = 0; i < 3; i++) {
      this.basis.x[i] = R[i * 3] * R0[0] + R[i * 3 + 1] * R0[3] + R[i * 3 + 2] * R0[6];
      this.basis.y[i] = R[i * 3] * R0[1] + R[i * 3 + 1] * R0[4] + R[i * 3 + 2] * R0[7];
      this.basis.z[i] = R[i * 3] * R0[2] + R[i * 3 + 1] * R0[5] + R[i * 3 + 2] * R0[8];
    }
    norm3(this.basis.z);
    const dp = this.basis.y[0] * this.basis.z[0] + this.basis.y[1] * this.basis.z[1] + this.basis.y[2] * this.basis.z[2];
    for (let i = 0; i < 3; i++) this.basis.y[i] -= this.basis.z[i] * dp;
    norm3(this.basis.y);
    cross3(this.basis.x, this.basis.y, this.basis.z);

    this.body.velocity(this.velocity);
    this.speed = Math.hypot(this.velocity[0], this.velocity[1], this.velocity[2]);
    this.forwardSpeed = this.velocity[0] * this.basis.z[0] + this.velocity[1] * this.basis.z[1] + this.velocity[2] * this.basis.z[2];
  }

  /** car-local point -> world, following the deformed body */
  toWorldPoint(local) {
    const R0 = this.R0, R = this.R;
    const p = this.spawnPos;
    const rx = R0[0] * local[0] + R0[1] * local[1] + R0[2] * local[2] + p[0];
    const ry = R0[3] * local[0] + R0[4] * local[1] + R0[5] * local[2] + p[1];
    const rz = R0[6] * local[0] + R0[7] * local[1] + R0[8] * local[2] + p[2];
    const c = this.restCentroid;
    const dx = rx - c[0], dy = ry - c[1], dz = rz - c[2];
    return [
      this.center[0] + R[0] * dx + R[1] * dy + R[2] * dz,
      this.center[1] + R[3] * dx + R[4] * dy + R[5] * dz,
      this.center[2] + R[6] * dx + R[7] * dy + R[8] * dz,
    ];
  }

  // ------------------------------------------------------------- simulation
  step(dt, input) {
    if (this.destroyed) return;
    this.throttle = input.throttle;
    this.brake = input.brake;
    this.handbrake = input.handbrake;
    this.steer = input.steer;

    const steps = Math.min(9, Math.max(1, Math.round(dt / SUB_DT)));
    const h = dt / steps;
    const velPrev = this.velocity.slice();
    for (let s = 0; s < steps; s++) {
      this.updateBasis();
      this.stepWheels(h);
      this.applyAero(h);
      this.body.step(h, this.world, this.world.colliders);
    }
    this.updateBasis();

    // smoothing for gauges / camera
    const invDt = dt > 0 ? 1 / dt : 0;
    this.accelSmooth[0] = lerp(this.accelSmooth[0], (this.velocity[0] - velPrev[0]) * invDt, 0.2);
    this.accelSmooth[1] = lerp(this.accelSmooth[1], (this.velocity[1] - velPrev[1]) * invDt, 0.2);
    this.accelSmooth[2] = lerp(this.accelSmooth[2], (this.velocity[2] - velPrev[2]) * invDt, 0.2);
    this.gForce[0] = this.accelSmooth[0] / 9.81;
    this.gForce[1] = this.accelSmooth[1] / 9.81;
    this.gForce[2] = this.accelSmooth[2] / 9.81;

    // yaw rate around the body up axis
    const w = this.bodyAngularRate();
    this.yawRate = w;

    this.updateWheelVisuals(dt);
    this.updateNodeTex();
    this.updateDamage(dt);
    this.impactFlash = Math.max(0, this.impactFlash - dt * 2.5);
  }

  applyAero(dt) {
    const s = this.speed;
    if (s < 0.4) return;
    const rho = 1.2;
    const drag = -0.5 * rho * this.spec.dragArea * s;
    const fx = this.velocity[0] * drag;
    const fy = this.velocity[1] * drag;
    const fz = this.velocity[2] * drag;
    this.body.applyForceNear(this.center[0], this.center[1] + 0.3, this.center[2], fx, fy, fz, 8, 3.0);
    // downforce split between the axles
    const df = 0.5 * rho * this.spec.downforce * s * s;
    const front = this.toWorldPoint([0, 0.3, CAR.axleFront * 0.8]);
    const rear = this.toWorldPoint([0, 0.3, CAR.axleRear * 0.9]);
    this.body.applyForceNear(front[0], front[1], front[2], 0, -df * 0.45, 0, 4, 1.4);
    this.body.applyForceNear(rear[0], rear[1], rear[2], 0, -df * 0.55, 0, 4, 1.4);
  }

  bodyAngularRate() {
    // rough yaw rate from front/rear lateral velocities
    const f = this.toWorldPoint([0, 0.5, CAR.axleFront]);
    const r = this.toWorldPoint([0, 0.5, CAR.axleRear]);
    this.body.velNear(f[0], f[1], f[2], _tmp);
    this.body.velNear(r[0], r[1], r[2], _tmp2);
    const dx = f[0] - r[0], dz = f[2] - r[2];
    const len = Math.hypot(dx, dz) || 1;
    const lx = dx / len, lz = dz / len;
    // lateral component (perpendicular in the ground plane)
    const vf = _tmp[0] * -lz + _tmp[2] * lx;
    const vr = _tmp2[0] * -lz + _tmp2[2] * lx;
    return (vf - vr) / len;
  }

  updateWheelSteer(dt) {
    const speedFactor = 1 - clamp(Math.abs(this.forwardSpeed) / 55, 0, 0.72);
    const maxSteer = this.spec.maxSteerDeg * Math.PI / 180 * speedFactor;
    for (const w of this.wheels) {
      w.steerTarget = w.front ? this.steer * maxSteer : 0;
      const delta = w.steerTarget - w.steer;
      const maxDelta = this.spec.steerSpeed * dt;
      w.steer += clamp(delta, -maxDelta, maxDelta);
    }
  }

  stepWheels(dt) {
    this.updateWheelSteer(dt);
    const spec = this.spec;
    const up = this.basis.y;
    const surf = _surf;
    surf.material = undefined;

    // drivetrain: AWD, 40/60 split, electric torque curve
    const v = Math.abs(this.forwardSpeed);
    const maxForce = Math.min(spec.maxWheelForce, (spec.powerKW * 1000) / Math.max(v, 6));
    const damageFactor = 1 - clamp(this.damageRatio() * 0.75, 0, 0.75);
    const drive = this.throttle * maxForce * damageFactor;
    const brakeTotal = this.brake * spec.brakeForce;

    let anyContact = false;
    for (const w of this.wheels) {
      if (w.detached) continue;
      const attachLocal = [w.x, CAR.hubY, w.z];
      const a = this.toWorldPoint(attachLocal);
      w.attach = a;
      // ---- suspension: distance from the attach point down to the surface
      this.world.surfaceAt(a[0], a[2], surf);
      const groundY = surf.height;
      const upy = Math.max(0.25, up[1]);
      const surfaceDist = (a[1] - groundY) / upy;
      const L = CAR.suspRest;
      const R = CAR.wheelRadius;
      let comp = clamp(L + R - surfaceDist, 0, CAR.suspTravel + R * 0.85);
      w.contact = comp > 0.001;
      w.compression = comp;
      w.surfaceY = groundY;
      w.normal[0] = surf.normal[0]; w.normal[1] = surf.normal[1]; w.normal[2] = surf.normal[2];
      this.surfaceMat = surf.material;

      // wheel centre
      const wc = [
        a[0] - up[0] * (L - comp),
        a[1] - up[1] * (L - comp),
        a[2] - up[2] * (L - comp),
      ];
      w.pos[0] = wc[0]; w.pos[1] = wc[1]; w.pos[2] = wc[2];

      const k = w.front ? spec.suspKFront : spec.suspKRear;
      let springF = 0;
      // compression rate from the suspension travel itself, then low-passed.
      // Driving it straight off the cage node velocities feeds every cage
      // vibration mode back into the spring, which makes the car chatter and
      // spin out above ~130 km/h.
      if (w.compPrev === undefined) w.compPrev = comp;
      const rawRate = dt > 0 ? (comp - w.compPrev) / dt : 0;
      w.compPrev = comp;
      w.compRate = lerp(w.compRate || 0, rawRate, clamp(dt / 0.035, 0, 1));
      if (w.contact) {
        anyContact = true;
        const compVel = w.compRate;              // positive = compressing
        const damp = compVel > 0 ? spec.suspDampBump : spec.suspDampRebound;
        springF = k * comp + damp * compVel;
        if (comp > CAR.suspTravel) {
          springF += (comp - CAR.suspTravel) * k * 5.0;
        }
        springF = clamp(springF, 0, 6.5 * (spec.mass * 9.81 / 4));
      }
      w.load = springF;

      // apply the vertical suspension force at the attach point
      if (springF > 0) {
        this.body.applyForceNodes(w.nodes, w.wAttach,
          up[0] * springF, up[1] * springF, up[2] * springF);
      }

      // ---- tyre forces
      const contact = [
        wc[0] - up[0] * R * 0.96,
        wc[1] - up[1] * R * 0.96,
        wc[2] - up[2] * R * 0.96,
      ];
      w.contactPoint[0] = contact[0];
      w.contactPoint[1] = contact[1];
      w.contactPoint[2] = contact[2];

      const n = surf.normal;
      // wheel forward direction (body forward rotated by the steer angle)
      const cs = Math.cos(w.steer), sn = Math.sin(w.steer);
      const bz = this.basis.z, bx = this.basis.x;
      let fwx = bz[0] * cs + bx[0] * sn;
      let fwy = bz[1] * cs + bx[1] * sn;
      let fwz = bz[2] * cs + bx[2] * sn;
      // project onto the contact plane
      const dn = fwx * n[0] + fwy * n[1] + fwz * n[2];
      fwx -= n[0] * dn; fwy -= n[1] * dn; fwz -= n[2] * dn;
      let fl = Math.hypot(fwx, fwy, fwz);
      if (fl < 1e-4) { fwx = bz[0]; fwy = bz[1]; fwz = bz[2]; fl = 1; }
      fwx /= fl; fwy /= fl; fwz /= fl;
      // left vector on the contact plane
      const lx = n[1] * fwz - n[2] * fwy;
      const ly = n[2] * fwx - n[0] * fwz;
      const lz = n[0] * fwy - n[1] * fwx;

      // velocity at the contact patch
      this.body.velNodes(w.nodes, w.wContact, _tmp);
      const vLong = _tmp[0] * fwx + _tmp[1] * fwy + _tmp[2] * fwz;
      const vLat = _tmp[0] * lx + _tmp[1] * ly + _tmp[2] * lz;

      if (!w.contact) {
        w.spinRate = lerp(w.spinRate, vLong / R, 0.1);
        w.skid = 0;
        w.slip = 0;
        continue;
      }

      const mat = MATERIALS[surf.material !== undefined ? surf.material : 1];
      const gripBase = mat.name === 'asphalt' || mat.name === 'concrete'
        ? spec.gripAsphalt : spec.gripGrass;
      const mu = gripBase * mat.mu * 1.0;
      const Fz = springF;
      const maxF = mu * Fz;
      w.maxF = maxF;

      // longitudinal request
      const axleShare = w.front ? 0.4 : 0.6;
      let flong = drive * axleShare * 0.5;
      const brakeF = (w.front ? spec.brakeBias : 1 - spec.brakeBias) * brakeTotal * 0.5;
      const hb = (!w.front ? this.handbrake * spec.brakeForce * 0.5 : 0);
      const braking = brakeF + hb;
      if (braking > 0) {
        // brakes may slow the wheel down but never drive it backwards: the
        // requested force is faded out as the patch speed approaches zero
        const fade = Math.min(braking, Math.abs(vLong) * 6000 + 60);
        flong -= Math.sign(vLong || 1) * fade;
      }
      flong -= vLong * spec.rollingResistance;
      if (flong > maxF) flong = maxF;
      if (flong < -maxF) flong = -maxF;

      // lateral request with a peak-slip curve
      const slipAngle = Math.atan2(vLat, Math.max(Math.abs(vLong), 2.2));
      const peak = spec.tyrePeakSlip;
      const ratio = Math.abs(slipAngle) / peak;
      const curve = ratio < 1 ? ratio : Math.max(0.72, 1 - (ratio - 1) * 0.16);
      let flat = -Math.sign(slipAngle) * curve * maxF;

      // friction ellipse
      const rem = Math.sqrt(Math.max(0, maxF * maxF - flong * flong));
      if (Math.abs(flat) > rem) flat = Math.sign(flat) * rem;

      w.slip = Math.abs(slipAngle);
      w.skid = clamp((Math.abs(slipAngle) - 0.09) * 5 + Math.abs(flong) / (maxF + 1) * 0.35, 0, 1);
      w.spinRate = vLong / R;
      if (Math.abs(flong) >= maxF * 0.92 && Math.abs(this.throttle) > 0.5) {
        // wheelspin
        w.spinRate += Math.sign(flong) * 14;
        w.skid = Math.max(w.skid, 0.7);
      }

      // apply at the contact patch
      const fx = fwx * flong + lx * flat;
      const fy = fwy * flong + ly * flat;
      const fz = fwz * flong + lz * flat;
      this.body.applyForceNodes(w.nodes, w.wContact, fx, fy, fz);
    }
    this.airborne = !anyContact;
  }

  updateWheelVisuals(dt) {
    for (const w of this.wheels) {
      if (w.detached) continue;
      w.spin += w.spinRate * dt;
      if (w.spin > Math.PI * 2) w.spin -= Math.PI * 2;
      if (w.spin < 0) w.spin += Math.PI * 2;
    }
  }

  // ---------------------------------------------------------------- damage
  buildNodeTriangles() {
    const { pos, beams } = this.cage;
    const n = this.nodeCount;
    const nb = Array.from({ length: n }, () => []);
    for (const b of beams) {
      const d = b.rest;
      nb[b.a].push([b.b, d]);
      nb[b.b].push([b.a, d]);
    }
    const tris = [];
    for (let i = 0; i < n; i++) {
      const list = nb[i].slice().sort((p, q) => p[1] - q[1]);
      let found = null;
      for (let a = 0; a < list.length && !found; a++) {
        for (let b = a + 1; b < list.length && !found; b++) {
          for (let c = b + 1; c < list.length && !found; c++) {
            const r0 = [pos[list[a][0] * 3] - pos[i * 3], pos[list[a][0] * 3 + 1] - pos[i * 3 + 1], pos[list[a][0] * 3 + 2] - pos[i * 3 + 2]];
            const r1 = [pos[list[b][0] * 3] - pos[i * 3], pos[list[b][0] * 3 + 1] - pos[i * 3 + 1], pos[list[b][0] * 3 + 2] - pos[i * 3 + 2]];
            const r2 = [pos[list[c][0] * 3] - pos[i * 3], pos[list[c][0] * 3 + 1] - pos[i * 3 + 1], pos[list[c][0] * 3 + 2] - pos[i * 3 + 2]];
            const det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1])
              - r1[0] * (r0[1] * r2[2] - r0[2] * r2[1])
              + r2[0] * (r0[1] * r1[2] - r0[2] * r1[1]);
            if (Math.abs(det) > 0.004) {
              found = { idx: [list[a][0], list[b][0], list[c][0]], inv: inv3(r0, r1, r2) };
            }
          }
        }
      }
      tris.push(found);
    }
    return tris;
  }

  updateNodeTex() {
    const n = this.nodeCount;
    const gq = this.rotQuat;
    const tex = this.nodeTex;
    const pos = this.body.pos;
    const rest = this.restWorld;
    const nb = this.cage.neighbours;
    for (let i = 0; i < n; i++) {
      const o = i * 16;
      tex[o] = pos[i * 3] - rest[i * 3];
      tex[o + 1] = pos[i * 3 + 1] - rest[i * 3 + 1];
      tex[o + 2] = pos[i * 3 + 2] - rest[i * 3 + 2];
      tex[o + 3] = 1;

      // least squares rotation of this node's neighbourhood (Horn, warm started)
      const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
      const rx = rest[i * 3], ry = rest[i * 3 + 1], rz = rest[i * 3 + 2];
      let sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;
      const list = nb[i];
      for (let k = 0; k < list.length; k++) {
        const j = list[k].other;
        const ax = pos[j * 3] - px, ay = pos[j * 3 + 1] - py, az = pos[j * 3 + 2] - pz;
        const bx = rest[j * 3] - rx, by = rest[j * 3 + 1] - ry, bz = rest[j * 3 + 2] - rz;
        sxx += ax * bx; sxy += ax * by; sxz += ax * bz;
        syx += ay * bx; syy += ay * by; syz += ay * bz;
        szx += az * bx; szy += az * by; szz += az * bz;
      }
      const N = _N;
      N[0] = sxx + syy + szz; N[1] = syz - szy; N[2] = szx - sxz; N[3] = sxy - syx;
      N[4] = N[1]; N[5] = sxx - syy - szz; N[6] = sxy + syx; N[7] = szx + sxz;
      N[8] = N[2]; N[9] = N[6]; N[10] = -sxx + syy - szz; N[11] = syz + szy;
      N[12] = N[3]; N[13] = N[7]; N[14] = N[11]; N[15] = -sxx - syy + szz;

      // start from the global body rotation: converges in 2-3 power iterations
      // and stays temporally coherent (no per-node eigenvector flipping)
      let w = gq[0], qx = gq[1], qy = gq[2], qz = gq[3];
      for (let it = 0; it < 3; it++) {
        const nw = N[0] * w + N[1] * qx + N[2] * qy + N[3] * qz;
        const nx = N[4] * w + N[5] * qx + N[6] * qy + N[7] * qz;
        const ny = N[8] * w + N[9] * qx + N[10] * qy + N[11] * qz;
        const nz = N[12] * w + N[13] * qx + N[14] * qy + N[15] * qz;
        const l = Math.hypot(nw, nx, ny, nz);
        if (l < 1e-12) break;
        w = nw / l; qx = nx / l; qy = ny / l; qz = nz / l;
      }
      // conjugate -> maps rest offsets to current offsets
      const cw = w, cx = -qx, cy = -qy, cz = -qz;
      // column-major rotation columns
      tex[o + 4] = 1 - 2 * (cy * cy + cz * cz);   // col0.x
      tex[o + 5] = 2 * (cx * cy + cz * cw);       // col0.y
      tex[o + 6] = 2 * (cx * cz - cy * cw);       // col0.z
      tex[o + 7] = 0;
      tex[o + 8] = 2 * (cx * cy - cz * cw);       // col1.x
      tex[o + 9] = 1 - 2 * (cx * cx + cz * cz);   // col1.y
      tex[o + 10] = 2 * (cy * cz + cx * cw);      // col1.z
      tex[o + 11] = 0;
      tex[o + 12] = 2 * (cx * cz + cy * cw);      // col2.x
      tex[o + 13] = 2 * (cy * cz - cx * cw);      // col2.y
      tex[o + 14] = 1 - 2 * (cx * cx + cy * cy);  // col2.z
      tex[o + 15] = 0;
    }
  }

  damageRatio() {
    const b = this.body;
    let sum = 0;
    for (let i = 0; i < b.nb; i++) sum += b.bdamage[i];
    return sum / b.nb;
  }

  updateDamage(dt) {
    const prev = this.damage;
    this.damage = this.damageRatio();
    if (this.damage > prev + 0.0015) this.impactFlash = 1;
    const impact = this.body.maxImpact;
    if (impact > this.lastImpact * 1.5 && impact > 40) this.lastImpact = impact;
    this.lastImpact *= 0.995;

    // wheel loss when the surrounding structure is destroyed
    for (const w of this.wheels) {
      if (w.detached) continue;
      const hub = this.nearestNodeIndex(w);
      if (w._hubBeams === undefined) {
        w._hubBeams = [];
        for (let b = 0; b < this.body.nb; b++) {
          if (this.body.ba[b] === hub || this.body.bb[b] === hub) w._hubBeams.push(b);
        }
      }
      let broken = 0;
      for (const b of w._hubBeams) if (this.body.bbroken[b]) broken++;
      const frac = w._hubBeams.length ? broken / w._hubBeams.length : 0;
      const disp = Math.hypot(
        this.body.pos[hub * 3] - this.restWorld[hub * 3],
        this.body.pos[hub * 3 + 1] - this.restWorld[hub * 3 + 1],
        this.body.pos[hub * 3 + 2] - this.restWorld[hub * 3 + 2],
      );
      w.rimDamage = Math.max(frac, clamp((disp - 0.28) * 2.2, 0, 1));
      if (frac > 0.62 || disp > 0.75) {
        w.pendingDetach = true;
      }
    }
    this.wheelDamage = this.wheels.reduce((s, w) => s + (w.rimDamage || 0), 0) / this.wheels.length;
  }

  /** collect wheels that should become free bodies this frame */
  takeDetachedWheels() {
    const out = [];
    for (const w of this.wheels) {
      if (w.pendingDetach && !w.detached) {
        w.pendingDetach = false;
        const p = this.detachWheel(w);
        if (p) out.push(p);
      }
    }
    return out;
  }

  nearestNodeIndex(w) {
    if (w._nodeIdx !== undefined) return w._nodeIdx;
    const local = [w.x, CAR.hubY, w.z];
    let best = 0, bestD = Infinity;
    for (let i = 0; i < this.nodeCount; i++) {
      const d = Math.hypot(
        this.cage.pos[i * 3] - local[0],
        this.cage.pos[i * 3 + 1] - local[1],
        this.cage.pos[i * 3 + 2] - local[2],
      );
      if (d < bestD) { bestD = d; best = i; }
    }
    w._nodeIdx = best;
    return best;
  }

  /** detach a wheel -> returns a free rigid body for the props system */
  detachWheel(w) {
    if (w.detached) return null;
    w.detached = true;
    this.body.velNear(w.pos[0], w.pos[1], w.pos[2], _tmp);
    return {
      kind: 'wheel',
      radius: CAR.wheelRadius,
      half: [0.13, CAR.wheelRadius, CAR.wheelRadius],
      pos: [w.pos[0], w.pos[1], w.pos[2]],
      vel: [_tmp[0], _tmp[1], _tmp[2]],
      angVel: [w.spinRate * 0.6, 0.5, 0],
      mass: this.spec.wheelMass,
      spin: w.spin,
      quat: [0, 0, 0, 1],
    };
  }

  reset() {
    const cosY = Math.cos(this.spawnYaw), sinY = Math.sin(this.spawnYaw);
    this.body.resetTo(this.restWorld, null);
    for (const w of this.wheels) {
      w.steer = 0; w.spin = 0; w.spinRate = 0; w.compression = 0;
      w.detached = false; w.load = 0;
    }
    this.damage = 0;
    this.rotQuat[0] = 1; this.rotQuat[1] = 0; this.rotQuat[2] = 0; this.rotQuat[3] = 0;
    this.center[0] = this.spawnPos[0];
    this.center[1] = this.spawnPos[1];
    this.center[2] = this.spawnPos[2];
    this.updateBasis();
    this.updateNodeTex();
  }
}

function nearestNodes(localPos, n, localPoint, k) {
  const tmp = [];
  for (let i = 0; i < n; i++) {
    const dx = localPos[i * 3] - localPoint[0];
    const dy = localPos[i * 3 + 1] - localPoint[1];
    const dz = localPos[i * 3 + 2] - localPoint[2];
    tmp.push([dx * dx + dy * dy + dz * dz, i]);
  }
  tmp.sort((a, b) => a[0] - b[0]);
  return tmp.slice(0, k).map((e) => e[1]);
}

function nodeWeights(localPos, localPoint, idx) {
  // weights from the distance of each node's rest position to the point
  const d = [];
  let sum = 0;
  for (const i of idx) {
    const dx = localPos[i * 3] - localPoint[0];
    const dy = localPos[i * 3 + 1] - localPoint[1];
    const dz = localPos[i * 3 + 2] - localPoint[2];
    const v = 1 / (dx * dx + dy * dy + dz * dz + 0.12);
    d.push(v);
    sum += v;
  }
  return d.map((v) => v / sum);
}

const _surf = { height: 0, normal: [0, 1, 0], material: 1 };
const _N = new Float64Array(16);
const _q = [0, 0, 0, 1];

function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  v[0] /= l; v[1] /= l; v[2] /= l;
  return v;
}
function cross3(out, a, b) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

function inv3(r0, r1, r2) {
  // columns r0, r1, r2 -> inverse (row-major 3x3)
  const a = r0[0], b = r1[0], c = r2[0];
  const d = r0[1], e = r1[1], f = r2[1];
  const g = r0[2], h = r1[2], i = r2[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const id = det !== 0 ? 1 / det : 0;
  return [
    (e * i - f * h) * id, (c * h - b * i) * id, (b * f - c * e) * id,
    (f * g - d * i) * id, (a * i - c * g) * id, (c * d - a * f) * id,
    (d * h - e * g) * id, (b * g - a * h) * id, (a * e - b * d) * id,
  ];
}
