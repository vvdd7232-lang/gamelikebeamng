// The car: soft-body cage + raycast-style suspension + tyre model + electric
// drivetrain.  Modelled on BeamNG: nothing about the car is rigid, the wheels
// push the deformable cage through spring/damper suspension, and impacts
// plastically deform (or break) the beams.

import { SoftBody } from '../physics/softbody.js';
import { buildCage, WHEEL_DEFS, CAR } from './cage.js';
import { CAGE } from './car_nodes.js';
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

    // ------------------------------------------------------------ body panels
    // Doors, the hood, the trunk and the bumpers are real parts hanging on their
    // own hinges (the pivots come from the model).  They crumple with the shell
    // because they are skinned to the cage, but they also swing, sag and get
    // torn off -- see updatePanels / detachPanel.
    this.panels = (CAGE.parts || []).filter((p) => p.role === 'panel').map((p, i) => ({
      index: i,
      name: p.name,
      pivot: p.pivot,
      com: p.com,
      mass: p.mass,
      area: p.area,
      hinge: p.hinge,
      nodes: p.nodes,
      bodyNodes: p.bodyNodes,
      nodeBeams: [],
      angle: 0,            // current rotation about the hinge axis (rad)
      omega: 0,            // angular velocity
      rest: 0,             // permanent bend of the hinge / mount
      loose: 0,            // 1 = nothing holds it on any more
      bend: 0,             // permanent twist of the hinge: a door that was hit
                           // sits ajar by this much for the rest of its life
      lastHit: 0,
      detached: false,
      pos: [0, 0, 0], vel: [0, 0, 0], quat: [0, 0, 0, 1], angVel: [0, 0, 0],
      frozen: false,
    }));
    for (const p of this.panels) {
      const beams = [];
      for (let bi = 0; bi < this.body.nb; bi++) {
        if (p.nodes.includes(this.body.ba[bi]) || p.nodes.includes(this.body.bb[bi])) beams.push(bi);
      }
      p.nodeBeams = beams;
      // rest-space pivot in world coordinates (the shader hinges the skinned
      // mesh about this point)
      const toRestWorld = (v) => [
        this.R0[0] * v[0] + this.R0[1] * v[1] + this.R0[2] * v[2] + this.spawnPos[0],
        this.R0[3] * v[0] + this.R0[4] * v[1] + this.R0[5] * v[2] + this.spawnPos[1],
        this.R0[6] * v[0] + this.R0[7] * v[1] + this.R0[8] * v[2] + this.spawnPos[2],
      ];
      p.pivotRest = toRestWorld(p.pivot);
      p.comRest = toRestWorld(p.com);
      const r = [p.com[0] - p.pivot[0], p.com[1] - p.pivot[1], p.com[2] - p.pivot[2]];
      const a = p.hinge.axis;
      const along = r[0] * a[0] + r[1] * a[1] + r[2] * a[2];
      p.r = r;
      // moment of inertia about the hinge line.  A hinged panel swings about a
      // line next to it (parallel-axis term); a bumper that hangs off its own
      // centre only spins in place, so it gets a flat-plate estimate instead.
      const lever = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - along * along;
      p.inertia = p.hinge.rigid
        ? Math.max(0.6, p.mass * Math.max(0.25, p.area || 1) / 12)
        : Math.max(0.4, p.mass * lever);
    }
    this.detachedPanels = [];
    this.panelClock = 0;

    this.nodeTex = new Float32Array(n * 4 * 4);
    this.basis = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
    this.center = [0, 0, 0];
    this.velocity = [0, 0, 0];
    this.speed = 0;
    this.forwardSpeed = 0;
    this.yawRate = 0;
    this.wheelSpinFast = 0;

    // ---------------------------------------------------------- damage state
    // `damage` is the share of the shell that has yielded / torn (0..1), the
    // zones feed the HUD silhouette, the sub-damage values drive the car:
    // a smashed nose cuts power, a bent front corner pulls the steering, a
    // damaged wheel loses grip and rides on a flat.
    this.damage = 0;
    this.structural = 1;
    this.cabinIntegrity = 1;
    this.damageZones = new Float64Array(6);   // FL FR ML MR RL RR
    this.zoneBeams = [[], [], [], [], [], []];
    this.cabinBeams = [];
    this.engineDamage = 0;
    this.steeringDamage = 0;
    this.steerPull = 0;
    this.powerScale = 1;
    this.lightsOut = 0;
    this.impacts = [];            // drained by the effects system
    this.detachedProps = [];      // wheels torn off, handed to the effects
    this.lastImpactSpeed = 0;     // km/h of the last big hit
    this.impactFlash = 0;
    this.impactCooldown = 0;
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
    this.lastImpact = 0;
    this.startBraking = 0;
    this.buildBeamZones();
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

    this.panelClock = (this.panelClock || 0) + dt;
    const steps = Math.min(9, Math.max(1, Math.round(dt / SUB_DT)));
    const h = dt / steps;
    const velPrev = this.velocity.slice();
    for (let s = 0; s < steps; s++) {
      this.updateBasis();
      this.stepWheels(h);
      this.applyAero(h);
      this.body.step(h, this.world, this.world.colliders);
      this.drainImpacts();
      this.updateDetachedWheels(h);
      this.updateDetachedPanels(h);
      this.updatePanels(h);
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
    // a bent front corner throws away steering lock and drags the car sideways
    const steerHealth = 1 - 0.55 * this.steeringDamage;
    const maxSteer = this.spec.maxSteerDeg * Math.PI / 180 * speedFactor * steerHealth;
    for (const w of this.wheels) {
      if (w.detached) { w.steerTarget = 0; continue; }
      const cmd = this.steer * steerHealth + this.steerPull;
      w.steerTarget = w.front ? cmd * maxSteer : 0;
      const delta = w.steerTarget - w.steer;
      const maxDelta = this.spec.steerSpeed * dt * steerHealth;
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
    // power comes from the front of the car: a smashed nose means a smashed
    // motor bay, so the throttle fades out instead of the old "lose 10% of
    // everything" rule
    const damageFactor = this.powerScale !== undefined ? this.powerScale : 1;
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

      const rim = w.rimDamage || 0;
      const k = (w.front ? spec.suspKFront : spec.suspKRear) * (1 - 0.5 * rim);
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
      // a bent rim or a flat tyre gives up part of the grip
      const mu = gripBase * mat.mu * (1 - 0.4 * rim);
      const Fz = springF;
      const maxF = mu * Fz;
      w.maxF = maxF;

      // longitudinal request
      const axleShare = w.front ? 0.4 : 0.6;
      let flong = drive * axleShare * 0.5;
      const brakeF = (w.front ? spec.brakeBias : 1 - spec.brakeBias) * brakeTotal * 0.5 * (1 - 0.7 * rim);
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
      // unused .w of the position row: damage of this node, drives the crumpled
      // paint, the cracked glass and the shot-out lights in the shader
      tex[o + 3] = this.body.nodeDamage[i];

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
      // unused .w of the last rotation row: how badly this node is folded
      tex[o + 15] = this.body.nodeCrumple[i];
    }
  }

  damageRatio() {
    const b = this.body;
    let sum = 0;
    for (let i = 0; i < b.nb; i++) sum += b.bdamage[i];
    return sum / b.nb;
  }

  /**
   * Bucket every beam into the six panels the HUD draws and into the set of
   * beams that make up the passenger safety cell.  Done once, in car space.
   */
  buildBeamZones() {
    const b = this.body;
    const local = this.cage.pos;
    for (let i = 0; i < b.nb; i++) {
      const a = b.ba[i], c = b.bb[i];
      const mz = (local[a * 3 + 2] + local[c * 3 + 2]) * 0.5;
      const mx = (local[a * 3] + local[c * 3]) * 0.5;
      const my = (local[a * 3 + 1] + local[c * 3 + 1]) * 0.5;
      const row = mz > 1.02 ? 0 : (mz < -1.05 ? 4 : 2);
      const side = mx > 0 ? 0 : 1;               // +x is the left side
      this.zoneBeams[row + side].push(i);
      if (my > 0.98 || (mz > -1.05 && mz < 0.85)) this.cabinBeams.push(i);
    }
  }

  /**
   * How badly a panel is hurt: the mean damage of the worst quarter of its
   * beams.  A plain average over a whole panel is useless for reading damage --
   * a collision folds a handful of beams completely while the rest of the panel
   * stays untouched, and the average then says "3% damaged" about a destroyed
   * nose.
   */
  beamDamage(list) {
    const bd = this.body.bdamage;
    const n = list.length;
    if (!n) return 0;
    const k = Math.max(1, Math.round(n * 0.25));
    const scratch = this._damageScratch || (this._damageScratch = new Float64Array(4096));
    const buf = k <= scratch.length ? scratch : new Float64Array(k);
    buf.fill(-1, 0, k);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = bd[list[i]];
      sum += d;
      // keep the k largest values in a small insertion buffer
      if (d <= buf[k - 1]) continue;
      let j = k - 1;
      while (j > 0 && buf[j - 1] < d) { buf[j] = buf[j - 1]; j--; }
      buf[j] = d;
    }
    let top = 0;
    for (let i = 0; i < k; i++) top += buf[i] > 0 ? buf[i] : 0;
    return clamp(top / k * 0.8 + (sum / n) * 0.2, 0, 1);
  }

  /** four stiff cage nodes next to a wheel hub: the reference frame used to
      measure how far the wheel has been pushed out of its arch */
  wheelAnchor(w) {
    if (w._anchorNodes) return;
    const hub = this.nearestNodeIndex(w);
    const local = this.cage.pos;
    const hx = local[hub * 3], hy = local[hub * 3 + 1], hz = local[hub * 3 + 2];
    const cabin = new Set();
    for (const i of this.cabinBeams) { cabin.add(this.body.ba[i]); cabin.add(this.body.bb[i]); }
    const cand = [];
    for (const i of cabin) {
      const dx = local[i * 3] - hx, dy = local[i * 3 + 1] - hy, dz = local[i * 3 + 2] - hz;
      cand.push([dx * dx + dy * dy + dz * dz, i]);
    }
    cand.sort((a, b) => a[0] - b[0]);
    const nodes = cand.slice(0, 5).map((e) => e[1]);
    let ax = 0, ay = 0, az = 0;
    for (const i of nodes) {
      ax += this.restWorld[i * 3]; ay += this.restWorld[i * 3 + 1]; az += this.restWorld[i * 3 + 2];
    }
    const n = nodes.length || 1;
    w._anchorNodes = nodes;
    w._anchorRest = [ax / n, ay / n, az / n];
  }

  /**
   * Doors, hood, trunk and bumpers as bodies on hinges.
   *
   * Every panel has one rotational DOF about its own hinge line, torqued by
   * gravity in the *body* frame (so a car on its roof has its doors fall open),
   * by the relative wind when the car is moving, and by a latch spring that
   * holds it shut until the mount around it has been destroyed.  Past that the
   * panel flops, and past the next threshold it comes off and becomes a loose
   * body that still tumbles on the ground.
   */
  updatePanels(dt) {
    if (!this.panels.length) return;
    const R = this.R;
    const g = 9.81;
    // gravity in the body frame: g_body = R^T * (0, -g, 0)
    const gbx = -g * R[3], gby = -g * R[4], gbz = -g * R[5];
    // relative wind in the body frame: the air the car is driving into
    const vx = this.velocity[0], vy = this.velocity[1], vz = this.velocity[2];
    const vbx = R[0] * vx + R[3] * vy + R[6] * vz;
    const vby = R[1] * vx + R[4] * vy + R[7] * vz;
    const vbz = R[2] * vx + R[5] * vy + R[8] * vz;
    const wind = Math.hypot(vbx, vby, vbz);

    for (const p of this.panels) {
      if (p.detached) continue;

      // ---- how much of the mount is left.  This latches: a panel never
      // re-hangs itself once the metal around its hinge has been destroyed.
      let dsum = 0;
      for (const i of p.nodes) dsum += this.body.nodeDamage[i];
      const dmg = dsum / Math.max(1, p.nodes.length);
      // how destroyed the panel itself is.  The hinge can survive while the
      // panel it holds is folded in half -- a hood does not stay on the car just
      // because the cowl it was bolted to is intact.
      let ssum = 0;
      for (const i of p.bodyNodes) ssum += this.body.nodeDamage[i];
      const skinDmg = ssum / Math.max(1, p.bodyNodes.length);
      let torn = 0;
      for (const bi of p.nodeBeams) if (this.body.bbroken[bi]) torn++;
      const tear = p.nodeBeams.length ? torn / p.nodeBeams.length : 0;
      // a bolted-on bumper lets go sooner than a door on a hinge
      const byMount = p.hinge.rigid
        ? (dmg - 0.15) * 1.9 + tear * 0.9
        : (dmg - 0.20) * 1.45 + tear * 0.85;
      const bySkin = Math.max(0, skinDmg - 0.62) * 1.7;
      const looseTarget = clamp(Math.max(byMount, bySkin), 0, 1);
      if (looseTarget > p.loose) p.loose = looseTarget;
      p.lastHit = Math.max(0, p.lastHit - dt * 0.6);
      p.mount = dmg;
      p.skinDamage = skinDmg;
      p.tearFrac = tear;

      // ---- a bumper is welded on: it does not swing, it droops a little and
      // then falls off.  Modelling it as a body on a hinge turned a knock into
      // the whole nose levering around a point at the car's origin.
      if (p.hinge.rigid) {
        p.angle = p.hinge.open * 0.22 * p.loose * p.loose;
        p.omega = 0;
        p.rest = p.angle;
        if (p.loose > 0.72 || tear > 0.45 ||
            (p.loose > 0.55 && p.lastHit > 0.35)) {
          this.detachPanel(p);
        }
        continue;
      }

      const a = p.hinge.axis;
      const r = p.r;

      // ---- torque about the hinge line.  While the latch holds it carries the
      // panel's weight (a parked car's hood does not sag); once the mount is
      // gone gravity is what makes a door hang open, or drop when the car ends
      // up on its roof.
      let torque = p.mass * (a[0] * (r[1] * gbz - r[2] * gby)
        + a[1] * (r[2] * gbx - r[0] * gbz)
        + a[2] * (r[0] * gby - r[1] * gbx));

      // ---- a panel that is even slightly ajar is a sail
      if (wind > 1.0 && (p.loose > 0.25 || Math.abs(p.angle) > 0.04)) {
        const sailArea = Math.min(1.8, 0.10 * p.mass + 0.25);
        const fw = 0.5 * 1.2 * 1.1 * sailArea * wind;
        const fxw = -vbx * fw, fyw = -vby * fw, fzw = -vbz * fw;
        torque += (0.30 + 0.70 * p.loose) *
          (a[0] * (r[1] * fzw - r[2] * fyw)
            + a[1] * (r[2] * fxw - r[0] * fzw)
            + a[2] * (r[0] * fyw - r[1] * fxw));
      }

      // ---- the latch: overwhelming while intact, gone once the mount is dead
      const hold = Math.pow(clamp(1 - p.loose / 0.62, 0, 1), 2.0);
      const stiffness = p.hinge.k * hold;
      const damping = p.hinge.damp * (0.16 + 0.84 * hold);
      torque *= 1 - hold;
      // a hit bends the hinge and leaves the panel permanently ajar; a dead
      // mount lets it hang further open.  Both are absolute values (derived
      // from state, never accumulated frame to frame) -- adding to the previous
      // angle made every door creep all the way to its stop.
      const restTarget = p.hinge.open * (0.45 * p.bend + 0.16 * p.loose * p.loose);
      torque -= stiffness * (p.angle - restTarget);

      // ---- hinge stops: closed (against the shell) and fully open
      const sign = p.hinge.open < 0 ? -1 : 1;
      const openLimit = sign * Math.abs(p.hinge.max);
      const closedLimit = -sign * 0.06;

      const sub = 2, h = dt / sub;
      for (let k = 0; k < sub; k++) {
        const alpha = (torque - damping * p.omega) / p.inertia;
        p.omega += alpha * h;
        p.angle += p.omega * h;
        if (sign > 0) {
          if (p.angle > openLimit) {
            p.angle = openLimit;
            if (p.omega > 0) {
              if (p.omega > 3.5) p.loose = Math.min(1, p.loose + p.omega * 0.012);
              p.omega *= -0.22;
            }
          } else if (p.angle < closedLimit) {
            p.angle = closedLimit;
            if (p.omega < 0) p.omega *= -0.2;
          }
        } else {
          if (p.angle < openLimit) {
            p.angle = openLimit;
            if (p.omega < 0) {
              if (-p.omega > 3.5) p.loose = Math.min(1, p.loose - p.omega * 0.012);
              p.omega *= -0.22;
            }
          } else if (p.angle > closedLimit) {
            p.angle = closedLimit;
            if (p.omega > 0) p.omega *= -0.2;
          }
        }
      }
      // remember the bend so the damage is permanent even if the spring re-centres
      if (p.loose > 0.45 || p.bend > 0.05) p.rest = clamp(restTarget, Math.min(0, openLimit), Math.max(0, openLimit));

      // ---- torn off?  The panel lets go when the mount around the hinge has
      // failed, or when it is already hanging by a thread and gets whipped.
      const fastSwing = Math.abs(p.omega) > 2.6 && p.loose > 0.72;
      const mountGone = tear > 0.80 && dmg > 0.55;
      const hanging = p.loose > 0.90 && (tear > 0.30 || p.lastHit > 0.40);
      // folded in half: what is left of it is scrap metal, not a hood
      const destroyed = skinDmg > 0.82 && p.loose > 0.52;
      if (fastSwing || mountGone || hanging || destroyed) this.detachPanel(p);
    }
  }

  /** a hit near a panel: convert the impact into angular momentum about its hinge */
  panelKick(x, y, z, power, normal, hard) {
    if (!this.panels.length || power < 0.05) return;
    this.panelClock = (this.panelClock || 0);
    const R = this.R;
    const c = this.center;
    const rc = this.restCentroid;
    for (const p of this.panels) {
      if (p.detached) continue;
      // world position of the panel's centre of mass (rigid prediction)
      const rx = p.comRest[0] - rc[0], ry = p.comRest[1] - rc[1], rz = p.comRest[2] - rc[2];
      const cx = c[0] + R[0] * rx + R[1] * ry + R[2] * rz;
      const cy = c[1] + R[3] * rx + R[4] * ry + R[5] * rz;
      const cz = c[2] + R[6] * rx + R[7] * ry + R[8] * rz;
      const dx = x - cx, dy = y - cy, dz = z - cz;
      // how far a blow can reach: a door takes a hit within its own frame,
      // not from anywhere on the car
      const reach = 0.55 + Math.sqrt(Math.max(p.inertia, 0.2)) * 0.42;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > reach) continue;
      // one blow per impact, not one per solver substep
      if (p._kickAt !== undefined && this.panelClock - p._kickAt < 0.12) continue;
      p._kickAt = this.panelClock;
      p.lastHit = Math.max(p.lastHit, power * (1 - dist / reach));
      const falloff = (1 - dist / reach) ** 1.6;
      // a whack dents the mount a little; destroying it is the cage's job
      if (power > 0.5) {
        p.loose = Math.min(1, p.loose + power * 0.02 * falloff * (hard ? 1.5 : 1));
      }
      // and it twists the hinge: the panel never quite shuts again
      if (!p.hinge.rigid) {
        p.bend = Math.min(1, p.bend + power * falloff * 0.55 * (hard ? 1.2 : 0.7));
      }
      if (p.hinge.rigid) continue;
      const a = p.hinge.axis;
      const ax = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
      const ay = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
      const az = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];
      // direction of the blow: the contact normal, or "away from the hit"
      let fx, fy, fz;
      if (normal) {
        fx = normal[0]; fy = normal[1]; fz = normal[2];
      } else {
        const l = dist || 1;
        fx = -dx / l; fy = -dy / l; fz = -dz / l;
      }
      // lever arm from the hinge pivot to the contact
      const lx = x - p.pivotRest[0], ly = y - p.pivotRest[1], lz = z - p.pivotRest[2];
      const tx = ly * fz - lz * fy;
      const ty = lz * fx - lx * fz;
      const tz = lx * fy - ly * fx;
      const axisTorque = ax * tx + ay * ty + az * tz;
      // power is 0..1; scale to a believable angular kick for the panel's inertia
      const kick = (axisTorque * power * falloff * 9.0) / Math.max(0.4, p.inertia);
      p.omega += clamp(kick, -26, 26);
    }
  }

  /** damage bookkeeping: panels, safety cell, wheels, drivetrain, lights */
  updateDamage(dt) {
    const b = this.body;
    const prev = this.damage;
    const impact = b.maxImpact;
    if (impact > this.lastImpact * 1.5 && impact > 40) this.lastImpact = impact;
    this.lastImpact *= 0.995;

    // ---- panels for the HUD silhouette
    let zmax = 0, zsum = 0;
    for (let z = 0; z < 6; z++) {
      const d = this.beamDamage(this.zoneBeams[z]);
      this.damageZones[z] = d;
      zmax = Math.max(zmax, d);
      zsum += d;
    }
    // shell damage: the worst panel counts most, so a destroyed nose reads as a
    // wrecked car even while the rest of the shell is still straight
    this.shellDamage = clamp(zmax * 0.45 + (zsum / 6) * 0.55, 0, 1);
    this.cabinIntegrity = 1 - clamp(this.beamDamage(this.cabinBeams) * 1.35, 0, 1);

    // ---- drivetrain / steering consequences
    const frontL = this.damageZones[0], frontR = this.damageZones[1];
    const front = (frontL + frontR) * 0.5;
    this.engineDamage = Math.max(this.engineDamage * (1 - dt * 0.02), clamp((front - 0.16) * 2.1, 0, 1));
    this.powerScale = clamp(1 - this.engineDamage * 0.92, 0.08, 1);
    this.steeringDamage = clamp((Math.max(frontL, frontR) - 0.1) * 1.5, 0, 1);
    // a bent corner drags the car towards the damaged side
    this.steerPull = (frontL - frontR) * 0.22;
    this.lightsOut = clamp((frontL + frontR) * 1.4, 0, 0.92);

    // ---- wheels: rim damage from the local cage frame (a crushed nose must
    // not count as a bent rear wheel), plus torn strut towers
    const R = this.R;
    const cx = this.center[0], cy = this.center[1], cz = this.center[2];
    const rc = this.restCentroid;
    for (const w of this.wheels) {
      if (w.detached) { w.rimDamage = 1; continue; }
      const hub = this.nearestNodeIndex(w);
      this.wheelAnchor(w);
      const nodes = w._anchorNodes;
      let ax = 0, ay = 0, az = 0;
      for (const i of nodes) {
        ax += b.pos[i * 3]; ay += b.pos[i * 3 + 1]; az += b.pos[i * 3 + 2];
      }
      const n = nodes.length || 1;
      ax /= n; ay /= n; az /= n;
      const ar = w._anchorRest;
      // rigid prediction of the anchor, then correct with what it actually did
      const rax = cx + R[0] * (ar[0] - rc[0]) + R[1] * (ar[1] - rc[1]) + R[2] * (ar[2] - rc[2]);
      const ray = cy + R[3] * (ar[0] - rc[0]) + R[4] * (ar[1] - rc[1]) + R[5] * (ar[2] - rc[2]);
      const raz = cz + R[6] * (ar[0] - rc[0]) + R[7] * (ar[1] - rc[1]) + R[8] * (ar[2] - rc[2]);
      const corr = [ax - rax, ay - ray, az - raz];
      const hx = this.restWorld[hub * 3], hy = this.restWorld[hub * 3 + 1], hz = this.restWorld[hub * 3 + 2];
      const ex = cx + R[0] * (hx - rc[0]) + R[1] * (hy - rc[1]) + R[2] * (hz - rc[2]) + corr[0];
      const ey = cy + R[3] * (hx - rc[0]) + R[4] * (hy - rc[1]) + R[5] * (hz - rc[2]) + corr[1];
      const ez = cz + R[6] * (hx - rc[0]) + R[7] * (hy - rc[1]) + R[8] * (hz - rc[2]) + corr[2];
      const disp = Math.hypot(b.pos[hub * 3] - ex, b.pos[hub * 3 + 1] - ey, b.pos[hub * 3 + 2] - ez);

      if (w._hubBeams === undefined) {
        w._hubBeams = [];
        for (let bi = 0; bi < b.nb; bi++) {
          if (b.ba[bi] === hub || b.bb[bi] === hub) w._hubBeams.push(bi);
        }
      }
      let torn = 0;
      for (const bi of w._hubBeams) if (b.bbroken[bi]) torn++;
      const frac = w._hubBeams.length ? torn / w._hubBeams.length : 0;

      // 26 cm of relative hub movement = the wheel has been pushed out of its
      // arch; suspension compliance alone is a few centimetres
      const rim = Math.max(frac * 1.1, clamp((disp - 0.26) * 2.4, 0, 1));
      w.rimDamage = Math.max((w.rimDamage || 0) * 0.999, clamp(rim, 0, 1));
      w.hubDisp = disp;
      // a wheel comes off when the arch around it is destroyed: pushed out of
      // place, or with most of its hub beams torn
      if (frac > 0.50 || (disp > 0.42 && rim > 0.8)) {
        // tear it off right here: the physics has to be self-contained, the
        // render loop only *reads* the result (and spawns the dust)
        const prop = this.detachWheel(w);
        if (prop) this.detachedProps.push(prop);
      }
    }
    this.wheelDamage = this.wheels.reduce((s, w) => s + (w.rimDamage || 0), 0) / this.wheels.length;
    this.detachedCount = this.wheels.filter((w) => w.detached).length;

    // headline severity for the player: shell + lost wheels + a broken safety
    // cell + torn-off body panels.  A car with its hood, its bumper and a door
    // lying on the road is a wreck even if the paint on the other side is fine.
    const lostWheels = this.detachedCount;
    const lostPanels = this.detachedPanels ? this.detachedPanels.length : 0;
    this.damage = clamp(
      this.shellDamage * 0.72
      + lostWheels * 0.10
      + (1 - this.cabinIntegrity) * 1.10
      + this.wheelDamage * 0.10
      + lostPanels * 0.055,
      0, 1);
    this.structural = 1 - this.shellDamage;
    if (this.damage > prev + 0.002 && this.impactFlash < 0.35) this.impactFlash = 0.35;

    this.impactTimer = Math.max(0, (this.impactTimer || 0) - dt);
  }

  /** tear a panel off: it becomes a loose body that keeps tumbling */
  detachPanel(p) {
    if (p.detached) return null;
    p.detached = true;
    const R = this.R;
    const c = this.center;
    const rc = this.restCentroid;
    const rx = p.comRest[0] - rc[0], ry = p.comRest[1] - rc[1], rz = p.comRest[2] - rc[2];
    p.pos[0] = c[0] + R[0] * rx + R[1] * ry + R[2] * rz;
    p.pos[1] = c[1] + R[3] * rx + R[4] * ry + R[5] * rz;
    p.pos[2] = c[2] + R[6] * rx + R[7] * ry + R[8] * rz;
    this.body.velNear(p.pos[0], p.pos[1], p.pos[2], _tmp);
    p.vel = [_tmp[0], Math.max(_tmp[1], 0.2) + 0.8, _tmp[2]];
    // the swing it had at the moment it let go, plus a tumble
    const a = p.hinge.axis;
    const ax = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
    const ay = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
    const az = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];
    const w = clamp(p.omega, -18, 18);
    p.angVel = [ax * w + (Math.random() - 0.5) * 3,
      ay * w + (Math.random() - 0.5) * 3,
      az * w + (Math.random() - 0.5) * 3];
    p.quat = [0, 0, 0, 1];
    p.radius = 0.45 + Math.sqrt(Math.max(p.inertia, 0.2)) * 0.35;
    this.detachedPanels.push(p);
    this.impacts.push({
      x: p.pos[0], y: p.pos[1], z: p.pos[2], power: 0.42,
      normal: [0, 1, 0], tear: true, speed: 0, panel: p.name,
    });
    return p;
  }

  /** loose panels: gravity, bouncing, tumbling, coming to rest */
  updateDetachedPanels(dt) {
    if (!this.detachedPanels.length) return;
    for (const p of this.detachedPanels) {
      p.vel[1] -= 9.81 * dt;
      p.pos[0] += p.vel[0] * dt;
      p.pos[1] += p.vel[1] * dt;
      p.pos[2] += p.vel[2] * dt;
      const R = p.radius;
      const ground = this.world.heightAt(p.pos[0], p.pos[2]) + R * 0.62;
      if (p.pos[1] < ground) {
        p.pos[1] = ground;
        if (p.vel[1] < 0) p.vel[1] = -p.vel[1] * 0.24;
        if (p.vel[1] < 0.55) p.vel[1] = 0;
        const f = Math.max(0, 1 - 3.0 * dt);
        p.vel[0] *= f; p.vel[2] *= f;
        const d = Math.max(0, 1 - 4.0 * dt);
        p.angVel[0] *= d; p.angVel[1] *= d; p.angVel[2] *= d;
        if (Math.hypot(p.vel[0], p.vel[2]) < 0.2 && Math.hypot(p.angVel[0], p.angVel[1], p.angVel[2]) < 0.4) {
          p.vel[0] = 0; p.vel[2] = 0;
          p.angVel[0] = 0; p.angVel[1] = 0; p.angVel[2] = 0;
        }
      }
      const [qx, qy, qz, qw] = p.quat;
      const [ax, ay, az] = p.angVel;
      const h = dt * 0.5;
      const nx = qx + h * (ax * qw + ay * qz - az * qy);
      const ny = qy + h * (ay * qw + az * qx - ax * qz);
      const nz = qz + h * (az * qw + ax * qy - ay * qx);
      const nw = qw + h * (-ax * qx - ay * qy - az * qz);
      const l = Math.hypot(nx, ny, nz, nw) || 1;
      p.quat[0] = nx / l; p.quat[1] = ny / l; p.quat[2] = nz / l; p.quat[3] = nw / l;
    }
  }

  /** pull the impact events out of the solver right after a substep */
  drainImpacts() {
    const b = this.body;
    if (b.contactEvent) {
      const e = b.contactEvent;
      this.impacts.push({
        x: e.x, y: e.y, z: e.z, power: e.power, normal: e.normal,
        hard: !!e.hard, tear: false, speed: e.power * 22,
      });
      if (e.power > 0.35) {
        this.lastImpactSpeed = e.power * 22 * 3.6;
        this.impactTimer = 1.4;
      }
      // only a solid hit knocks a panel loose -- scraping the underbody over a
      // kerb should not swing the doors
      if (e.power > 0.28 && (Math.abs(e.normal[1]) < 0.75 || e.power > 0.6)) {
        this.panelKick(e.x, e.y, e.z, e.power, e.normal, true);
      }
      b.contactEvent = null;
    }
    for (const ev of b.events) {
      this.impacts.push({ x: ev.x, y: ev.y, z: ev.z, power: ev.power * 0.8, tear: true, speed: 0 });
      if (ev.power > 0.5) this.panelKick(ev.x, ev.y, ev.z, ev.power * 0.8, null, false);
    }
    if (b.events.length) b.events.length = 0;
    if (this.impacts.length > 64) this.impacts.splice(0, this.impacts.length - 64);
  }

  /** panels torn off since the last call (for dust / debris effects) */
  takeDetachedPanels() {
    // `detachedPanels` is the persistent list the renderer draws; this only
    // hands out the ones that left since the last call
    const fresh = this.detachedPanels.filter((p) => !p._announced);
    for (const p of fresh) p._announced = true;
    return fresh;
  }

  /** wheels torn off since the last call (for dust / debris effects) */
  takeDetachedWheels() {
    const out = this.detachedProps;
    this.detachedProps = [];
    return out;
  }

  /**
   * A wheel that came off keeps existing: it flies, bounces, rolls and finally
   * comes to rest instead of just vanishing.  Cheap rigid body -- a sphere for
   * the tyre plus rolling friction -- but it is the difference between "my
   * wheel fell off" and "my wheel was deleted".
   */
  updateDetachedWheels(dt) {
    const R = CAR.wheelRadius;
    for (const w of this.wheels) {
      if (!w.detached || !w.vel) continue;
      w.vel[1] -= 9.81 * dt;
      w.pos[0] += w.vel[0] * dt;
      w.pos[1] += w.vel[1] * dt;
      w.pos[2] += w.vel[2] * dt;
      const ground = this.world.heightAt(w.pos[0], w.pos[2]) + R * 0.92;
      let onGround = false;
      if (w.pos[1] < ground) {
        w.pos[1] = ground;
        onGround = true;
        if (w.vel[1] < 0) w.vel[1] = -w.vel[1] * 0.32;
        if (w.vel[1] < 0.6) w.vel[1] = 0;
        // rolling + a bit of sideways scrub
        const f = Math.max(0, 1 - 2.6 * dt);
        w.vel[0] *= f; w.vel[2] *= f;
        if (Math.hypot(w.vel[0], w.vel[2]) < 0.25) { w.vel[0] = 0; w.vel[2] = 0; }
      }
      // tumble: integrate the quaternion, and let it settle flat on the ground
      const spinSpeed = Math.hypot(w.vel[0], w.vel[2]);
      w.spin += (w.spinRate || 0) * dt;
      const av = w.angVel;
      if (av) {
        const damp = onGround ? Math.max(0, 1 - 3.4 * dt) : 1;
        av[0] *= damp; av[1] *= damp; av[2] *= damp;
        const [ax, ay, az] = av;
        const [qx, qy, qz, qw] = w.quat;
        const h = dt * 0.5;
        const nx = qx + h * (ax * qw + ay * qz - az * qy);
        const ny = qy + h * (ay * qw + az * qx - ax * qz);
        const nz = qz + h * (az * qw + ax * qy - ay * qx);
        const nw = qw + h * (-ax * qx - ay * qy - az * qz);
        const l = Math.hypot(nx, ny, nz, nw) || 1;
        w.quat[0] = nx / l; w.quat[1] = ny / l; w.quat[2] = nz / l; w.quat[3] = nw / l;
      }
      if (onGround && spinSpeed < 0.4 && av) {
        // stand the wheel up: rotate the quaternion towards "axle horizontal"
        const target = [0, 0, 0, 1];
        for (let i = 0; i < 4; i++) w.quat[i] += (target[i] - w.quat[i]) * Math.min(1, dt * 1.6);
        const l = Math.hypot(w.quat[0], w.quat[1], w.quat[2], w.quat[3]) || 1;
        for (let i = 0; i < 4; i++) w.quat[i] /= l;
      }
    }
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
    w.rimDamage = 1;
    this.body.velNear(w.pos[0], w.pos[1], w.pos[2], _tmp);
    // a torn-off wheel leaves with the car's momentum plus whatever the impact
    // gave it; keep it alive so it can fly, bounce and roll away
    w.vel = [_tmp[0], Math.max(_tmp[1], 0.4) + 1.1, _tmp[2]];
    w.angVel = [
      (Math.random() - 0.5) * 9,
      (Math.random() - 0.5) * 9,
      (Math.random() - 0.5) * 9,
    ];
    w.quat = [0, 0, 0, 1];
    w.flat = 1;
    return {
      kind: 'wheel',
      radius: CAR.wheelRadius,
      half: [0.13, CAR.wheelRadius, CAR.wheelRadius],
      pos: [w.pos[0], w.pos[1], w.pos[2]],
      vel: w.vel.slice(),
      angVel: w.angVel.slice(),
      mass: this.spec.wheelMass,
      spin: w.spin,
      quat: [0, 0, 0, 1],
    };
  }

  reset() {
    this.body.resetTo(this.restWorld, null);
    for (const w of this.wheels) {
      w.steer = 0; w.spin = 0; w.spinRate = 0; w.compression = 0;
      w.detached = false; w.load = 0; w.pendingDetach = false;
      w.rimDamage = 0; w.flat = 0; w.hubDisp = 0;
      w.quat = [0, 0, 0, 1]; w.angVel = null; w.vel = null;
    }
    this.damage = 0;
    this.structural = 1;
    this.cabinIntegrity = 1;
    this.damageZones.fill(0);
    this.engineDamage = 0;
    this.steeringDamage = 0;
    this.steerPull = 0;
    this.powerScale = 1;
    this.lightsOut = 0;
    this.lastImpactSpeed = 0;
    this.impactTimer = 0;
    this.impactFlash = 0;
    this.impacts.length = 0;
    this.detachedProps.length = 0;
    this.wheelDamage = 0;
    this.detachedCount = 0;
    for (const p of this.panels) {
      p.angle = 0; p.omega = 0; p.rest = 0; p.loose = 0; p.bend = 0;
      p.detached = false; p.frozen = false; p.lastHit = 0; p._announced = false;
      p.pos = [0, 0, 0]; p.vel = [0, 0, 0]; p.quat = [0, 0, 0, 1]; p.angVel = [0, 0, 0];
    }
    this.detachedPanels.length = 0;
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
