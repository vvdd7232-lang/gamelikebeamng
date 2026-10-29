// Validates the skinning transform used by MESH_VS / DEPTH_VS:
//   1. at spawn (bind pose) the mesh must be a rigid placement of the glb
//   2. after a rigid transform of the soft body the skinned result must be the
//      same rigid transform of the previous skinned result (equivariance)
//   3. local rotation matrices D must stay orthonormal under deformation
//   4. wheels must sit on the ground at tyre radius after settling
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { parseGlbForNode } from './glb_node.mjs';
import fs from 'fs';

const glbBuffer = new Uint8Array(fs.readFileSync(new URL('../game/assets/tesla.glb', import.meta.url)));
const glb = parseGlbForNode(glbBuffer);
const world = new World(1337);
const car = new Vehicle(world, { spawn: world.spawnOnTrack(0.02) });

const geom = buildCarGeometry(glb, car.cage.nodes);
const bodyPrim = geom.bodies[0];
const STRIDE = 20;
const SAMPLE_STEP = 53;

// CPU mirror of the vertex shader.
function skinAll(vehicle) {
  const R0 = vehicle.R0, p = vehicle.spawnPos;
  const tex = vehicle.nodeTex, rest = vehicle.restWorld, pos = vehicle.body.pos;
  const out = [];
  for (let v = 0; v < bodyPrim.vertexCount; v += SAMPLE_STEP) {
    const o = v * STRIDE;
    const px = bodyPrim.data[o], py = bodyPrim.data[o + 1], pz = bodyPrim.data[o + 2];
    let ax = 0, ay = 0, az = 0;
    for (let k = 0; k < 4; k++) {
      const j = bodyPrim.data[o + 12 + k];
      const w = bodyPrim.data[o + 16 + k];
      if (w <= 0) continue;
      const t = j * 16;
      // bind pose of this vertex in world space
      const bx = R0[0] * px + R0[1] * py + R0[2] * pz + p[0];
      const by = R0[3] * px + R0[4] * py + R0[5] * pz + p[1];
      const bz = R0[6] * px + R0[7] * py + R0[8] * pz + p[2];
      const dx = bx - rest[j * 3], dy = by - rest[j * 3 + 1], dz = bz - rest[j * 3 + 2];
      // D columns as stored by updateNodeTex
      const c0x = tex[t + 4], c0y = tex[t + 5], c0z = tex[t + 6];
      const c1x = tex[t + 8], c1y = tex[t + 9], c1z = tex[t + 10];
      const c2x = tex[t + 12], c2y = tex[t + 13], c2z = tex[t + 14];
      ax += w * (pos[j * 3] + c0x * dx + c1x * dy + c2x * dz);
      ay += w * (pos[j * 3 + 1] + c0y * dx + c1y * dy + c2y * dz);
      az += w * (pos[j * 3 + 2] + c0z * dx + c1z * dy + c2z * dz);
    }
    out.push([ax, ay, az]);
  }
  return out;
}

// 1) bind pose: skinned == rigid placement of the glb
car.updateNodeTex();
const bindSkinned = skinAll(car);
let maxBindErr = 0;
{
  const R0 = car.R0, p = car.spawnPos;
  let i = 0;
  for (let v = 0; v < bodyPrim.vertexCount; v += SAMPLE_STEP, i++) {
    const o = v * STRIDE;
    const px = bodyPrim.data[o], py = bodyPrim.data[o + 1], pz = bodyPrim.data[o + 2];
    const ex = R0[0] * px + R0[1] * py + R0[2] * pz + p[0];
    const ey = R0[3] * px + R0[4] * py + R0[5] * pz + p[1];
    const ez = R0[6] * px + R0[7] * py + R0[8] * pz + p[2];
    const s = bindSkinned[i];
    maxBindErr = Math.max(maxBindErr, Math.hypot(s[0] - ex, s[1] - ey, s[2] - ez));
  }
}
console.log('1) bind pose error vs glb rigid placement:', maxBindErr.toExponential(2), 'm');

// 2) settle, then rigid-transform the body and compare
for (let i = 0; i < 360; i++) car.step(1 / 60, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
car.updateNodeTex();
const s1 = skinAll(car);

let maxOrtho = 0;
for (let j = 0; j < car.nodeCount; j++) {
  const t = j * 16;
  const c0 = [car.nodeTex[t + 4], car.nodeTex[t + 5], car.nodeTex[t + 6]];
  const c1 = [car.nodeTex[t + 8], car.nodeTex[t + 9], car.nodeTex[t + 10]];
  const c2 = [car.nodeTex[t + 12], car.nodeTex[t + 13], car.nodeTex[t + 14]];
  const n0 = Math.hypot(...c0), n1 = Math.hypot(...c1), n2 = Math.hypot(...c2);
  const d01 = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
  maxOrtho = Math.max(maxOrtho, Math.abs(n0 - 1), Math.abs(n1 - 1), Math.abs(n2 - 1), Math.abs(d01));
}
console.log('2) local rotation matrices: max |‖col‖-1| / |col·col|:', maxOrtho.toExponential(2));

const backup = Float64Array.from(car.body.pos);
const rc = car.restCentroid;
let yawAcc = 0, pitchAcc = 0;
let maxEqui = 0;
let prev = s1;
for (let frame = 0; frame < 24; frame++) {
  // rotate by one realistic per-frame increment (0.35 rad/frame would be a
  // 2000 deg/s roll; 0.04 rad/frame is already an extreme crash)
  const yaw = 0.04, pitch = 0.015;
  yawAcc += yaw; pitchAcc += pitch;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const Rm = [cy, sy * sp, sy * cp, 0, cp, -sp, -sy, cy * sp, cy * cp];
  for (let i = 0; i < car.nodeCount; i++) {
    const x = car.body.pos[i * 3] - rc[0], y = car.body.pos[i * 3 + 1] - rc[1], z = car.body.pos[i * 3 + 2] - rc[2];
    car.body.pos[i * 3] = rc[0] + Rm[0] * x + Rm[1] * y + Rm[2] * z;
    car.body.pos[i * 3 + 1] = rc[1] + Rm[3] * x + Rm[4] * y + Rm[5] * z;
    car.body.pos[i * 3 + 2] = rc[2] + Rm[6] * x + Rm[7] * y + Rm[8] * z;
  }
  car.updateBasis();
  car.updateNodeTex();
  const cur = skinAll(car);
  let err = 0;
  for (let i = 0; i < cur.length; i++) {
    const v = [prev[i][0] - rc[0], prev[i][1] - rc[1], prev[i][2] - rc[2]];
    const e = [
      rc[0] + Rm[0] * v[0] + Rm[1] * v[1] + Rm[2] * v[2],
      rc[1] + Rm[3] * v[0] + Rm[4] * v[1] + Rm[5] * v[2],
      rc[2] + Rm[6] * v[0] + Rm[7] * v[1] + Rm[8] * v[2],
    ];
    err = Math.max(err, Math.hypot(cur[i][0] - e[0], cur[i][1] - e[1], cur[i][2] - e[2]));
  }
  if (err > maxEqui) maxEqui = err;
  prev = cur;
}
console.log('3) skin equivariance, per-frame rigid steps (0.04 rad):', maxEqui.toExponential(2), 'm');
console.log('   total imposed rotation:', (Math.hypot(yawAcc, pitchAcc) * 57.3).toFixed(0), 'deg');

// restore and check wheels
for (let i = 0; i < car.nodeCount; i++) {
  car.body.pos[i * 3] = backup[i * 3];
  car.body.pos[i * 3 + 1] = backup[i * 3 + 1];
  car.body.pos[i * 3 + 2] = backup[i * 3 + 2];
}
car.updateBasis();
console.log('4) wheels after settle:');
let wheelOk = true;
for (const w of car.wheels) {
  const g = world.heightAt(w.pos[0], w.pos[2]);
  const h = w.pos[1] - g;
  if (Math.abs(h - 0.358) > 0.03) wheelOk = false;
  console.log(`   ${w.name} centre=(${w.pos.map((v) => v.toFixed(2)).join(',')}) ` +
    `comp=${w.compression.toFixed(3)} height=${h.toFixed(3)} contact=${w.contact}`);
}
const ok = maxBindErr < 1e-4 && maxOrtho < 1e-2 && maxEqui < 1e-2 && wheelOk;
console.log(ok ? 'SKINNING OK' : 'SKINNING FAILED');
process.exit(ok ? 0 : 1);
