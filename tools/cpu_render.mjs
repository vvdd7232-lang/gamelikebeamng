// CPU-side verification renderer input: runs a crash scenario and dumps the
// *skinned* car triangles (exactly what MESH_VS would produce on the GPU) plus
// the wheel meshes, so tools/raster.py can rasterise them into a PNG.
import fs from 'fs';
import { World, MATERIALS } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { Camera } from '../game/src/game/camera.js';

const glbBuffer = new Uint8Array(fs.readFileSync(new URL('../game/assets/volvo.glb', import.meta.url)));
const { parseGlbForNode } = await import('../tests/glb_node.mjs');
const glb = parseGlbForNode(glbBuffer);

const scenario = process.argv[2] || 'crash';
const world = new World(1337);
const probe = new Vehicle(world, { spawn: world.spawnOnTrack(0.0) });
const geomRaw = buildCarGeometry(glb, probe.cage.nodes);
// the renderer keys wheels by node name; mirror that here
const wheelGeom = {};
for (const w of geomRaw.wheels) wheelGeom[w.name] = w.prims;
const geom = { bodies: geomRaw.bodies, wheels: wheelGeom };
let spawn = world.spawnOnTrack(0.02);
if (scenario === 'pole' || scenario === 'wall') {
  // aim the car straight at a static obstacle (tree trunk / barrier)
  const start = world.spawnOnTrack(0.02).position;
  let best = null, bestD = Infinity;
  for (const c of world.colliders) {
    const d = Math.hypot(c.center[0] - start[0], c.center[2] - start[2]);
    if (d > 40 && d < bestD) { bestD = d; best = c; }
  }
  // direction obstacle -> start, so the car is placed in open country facing it
  const dx = best.center[0] - start[0], dz = best.center[2] - start[2];
  const dl = Math.hypot(dx, dz);
  const ux = dx / dl, uz = dz / dl;
  const back = scenario === 'wall' ? 26 : 0;
  const px = best.center[0] - ux * back, pz = best.center[2] - uz * back;
  spawn = { position: [px, world.heightAt(px, pz) + 0.75, pz], yaw: Math.atan2(ux, uz) };
  console.log(`aiming at obstacle ${bestD.toFixed(1)} m away, spawning ${back} m in front of it`);
}
const car = new Vehicle(world, { spawn, isPlayer: true });
const camera = new Camera();
camera.dist = Number(process.argv[4] || 6.4);
camera.pitch = Number(process.argv[5] || 0.16);
camera.yaw = Number(process.argv[3] || 0.85);

function drive(seconds, input, onStep) {
  const dt = 1 / 60;
  const n = Math.round(seconds / dt);
  for (let i = 0; i < n; i++) {
    car.step(dt, typeof input === 'function' ? input(i * dt) : input);
    if (onStep) onStep(i * dt);
  }
}

if (scenario === 'rest') {
  drive(1.5, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
} else if (scenario === 'crash') {
  // full throttle down the road for 4 s, then slam a hard left into the trees
  drive(3.2, { throttle: 1, brake: 0, steer: 0.02, handbrake: 0 });
  drive(1.2, (t) => ({ throttle: 1, brake: 0, steer: t < 0.45 ? 1 : -0.15, handbrake: 0 }));
  drive(1.4, { throttle: 0, brake: 0.4, steer: 0, handbrake: 0 });
} else if (scenario === 'pole') {
  // accelerate into the obstacle until it is damaged, then get off the throttle
  for (let i = 0; i < 60 * 20; i++) {
    car.step(1 / 60, { throttle: 1, brake: 0, steer: 0, handbrake: 0 });
    if (car.damage > 0.01) break;
  }
  drive(1.2, { throttle: 0, brake: 0.6, steer: 0, handbrake: 0 });
} else if (scenario === 'wall') {
  // launch straight into the obstacle at a set speed
  const b = car.basis.z;
  const v = Number(process.env.IMPACT_KMH || 100) / 3.6;
  for (let i = 0; i < car.body.pos.length / 3; i++) {
    car.body.vel[i * 3] = b[0] * v;
    car.body.vel[i * 3 + 1] = b[1] * v;
    car.body.vel[i * 3 + 2] = b[2] * v;
  }
  for (let i = 0; i < 60 * 6; i++) {
    car.step(1 / 60, { throttle: 1, brake: 0, steer: 0, handbrake: 0 });
    if (car.body.maxImpact > 200) break;
  }
  drive(1.6, { throttle: 0, brake: 1, steer: 0, handbrake: 0 });
} else if (scenario === 'roll') {
  drive(2.6, { throttle: 1, brake: 0, steer: 0, handbrake: 0 });
  drive(2.2, { throttle: 1, brake: 0, steer: 1, handbrake: 0 });
} else {
  throw new Error('unknown scenario ' + scenario);
}

camera.snap(car, world);
camera.update(0.016, car, { mouse: { dx: 0, dy: 0, wheel: 0, dragging: false } }, world);

// ---------------------------------------------------------------- skinning
const STRIDE = 20;
car.updateNodeTex();
const tex = car.nodeTex;
const rest = car.restWorld;
const pos = car.body.pos;
const R0 = car.R0, P0 = car.spawnPos;

function skinVertex(data, v, out) {
  const o = v * STRIDE;
  const px = data[o], py = data[o + 1], pz = data[o + 2];
  // bind pose in world space
  const bx = R0[0] * px + R0[1] * py + R0[2] * pz + P0[0];
  const by = R0[3] * px + R0[4] * py + R0[5] * pz + P0[1];
  const bz = R0[6] * px + R0[7] * py + R0[8] * pz + P0[2];
  // bind normal (rotation only)
  const nbx = R0[0] * data[o + 3] + R0[1] * data[o + 4] + R0[2] * data[o + 5];
  const nby = R0[3] * data[o + 3] + R0[4] * data[o + 4] + R0[5] * data[o + 5];
  const nbz = R0[6] * data[o + 3] + R0[7] * data[o + 4] + R0[8] * data[o + 5];
  let ax = 0, ay = 0, az = 0, nx = 0, ny = 0, nz = 0;
  for (let k = 0; k < 4; k++) {
    const j = data[o + 12 + k];
    const w = data[o + 16 + k];
    if (w <= 0) continue;
    const t = j * 16;
    const dx = bx - rest[j * 3], dy = by - rest[j * 3 + 1], dz = bz - rest[j * 3 + 2];
    const c0x = tex[t + 4], c0y = tex[t + 5], c0z = tex[t + 6];
    const c1x = tex[t + 8], c1y = tex[t + 9], c1z = tex[t + 10];
    const c2x = tex[t + 12], c2y = tex[t + 13], c2z = tex[t + 14];
    ax += w * (pos[j * 3] + c0x * dx + c1x * dy + c2x * dz);
    ay += w * (pos[j * 3 + 1] + c0y * dx + c1y * dy + c2y * dz);
    az += w * (pos[j * 3 + 2] + c0z * dx + c1z * dy + c2z * dz);
    nx += w * (c0x * nbx + c1x * nby + c2x * nbz);
    ny += w * (c0y * nbx + c1y * nby + c2y * nbz);
    nz += w * (c0z * nbx + c1z * nby + c2z * nbz);
  }
  out[0] = ax; out[1] = ay; out[2] = az;
  out[3] = nx; out[4] = ny; out[5] = nz;
  out[6] = data[o + 6]; out[7] = data[o + 7]; out[8] = data[o + 8];
}

const triPos = [];
const triNrm = [];
const triCol = [];
const tmp = new Float32Array(9);
for (const body of geom.bodies) {
  const data = body.data;
  const idx = body.indices;
  const triCount = idx.length / 3;
  for (let t = 0; t < triCount; t++) {
    for (let c = 0; c < 3; c++) {
      skinVertex(data, idx[t * 3 + c], tmp);
      triPos.push(tmp[0], tmp[1], tmp[2]);
      triNrm.push(tmp[3], tmp[4], tmp[5]);
      triCol.push(tmp[6], tmp[7], tmp[8]);
    }
  }
}

// wheels: rigid, placed like the renderer does
for (const w of car.wheels) {
  if (w.detached) continue;
  const prims = geom.wheels['Wheel_' + w.name];
  if (!prims) continue;
  const b = car.basis;
  const cs = Math.cos(w.steer), sn = Math.sin(w.steer);
  const fwd = b.z, left = b.x;
  const f2 = [fwd[0] * cs + left[0] * sn, fwd[1] * cs + left[1] * sn, fwd[2] * cs + left[2] * sn];
  const l2 = [left[0] * cs - fwd[0] * sn, left[1] * cs - fwd[1] * sn, left[2] * cs - fwd[2] * sn];
  const u2 = [b.y[0], b.y[1], b.y[2]];
  for (const prim of prims) {
    const d = prim.data, pi = prim.indices;
    const nv = prim.vertexCount;
    const wpos = new Float32Array(nv * 3);
    const wnrm = new Float32Array(nv * 3);
    for (let v = 0; v < nv; v++) {
      const o = v * STRIDE;
      const x = d[o], y = d[o + 1], z = d[o + 2];
      wpos[v * 3] = w.pos[0] + l2[0] * x + u2[0] * y + f2[0] * z;
      wpos[v * 3 + 1] = w.pos[1] + l2[1] * x + u2[1] * y + f2[1] * z;
      wpos[v * 3 + 2] = w.pos[2] + l2[2] * x + u2[2] * y + f2[2] * z;
      const nx = d[o + 3], ny = d[o + 4], nz = d[o + 5];
      wnrm[v * 3] = l2[0] * nx + u2[0] * ny + f2[0] * nz;
      wnrm[v * 3 + 1] = l2[1] * nx + u2[1] * ny + f2[1] * nz;
      wnrm[v * 3 + 2] = l2[2] * nx + u2[2] * ny + f2[2] * nz;
    }
    let wmn = [1e9, 1e9, 1e9], wmx = [-1e9, -1e9, -1e9];
    for (let v = 0; v < nv; v++) for (let k = 0; k < 3; k++) { wmn[k] = Math.min(wmn[k], wpos[v * 3 + k]); wmx[k] = Math.max(wmx[k], wpos[v * 3 + k]); }
    if (process.env.WHEEL_DEBUG) console.log(`  ${w.name}/${prim.name}: ${wmn.map((v) => v.toFixed(2))} .. ${wmx.map((v) => v.toFixed(2))}`);
    const cn = pi.length / 3;
    // wheels are dense; keep every other triangle, it is invisible at 1280x720
    for (let t = 0; t < cn; t += 2) {
      for (let c = 0; c < 3; c++) {
        const v = pi[t * 3 + c];
        triPos.push(wpos[v * 3], wpos[v * 3 + 1], wpos[v * 3 + 2]);
        triNrm.push(wnrm[v * 3], wnrm[v * 3 + 1], wnrm[v * 3 + 2]);
        triCol.push(d[v * STRIDE + 6], d[v * STRIDE + 7], d[v * STRIDE + 8]);
      }
    }
  }
}

// ground: a small patch of terrain around the car (same height function)
const cx = car.center[0], cz = car.center[2];
const R = 26, STEP = 1.0;
const groundTris = [];
for (let gx = -R; gx < R; gx += STEP) {
  for (let gz = -R; gz < R; gz += STEP) {
    const quad = [[cx + gx, cz + gz], [cx + gx + STEP, cz + gz], [cx + gx + STEP, cz + gz + STEP], [cx + gx, cz + gz + STEP]];
    const pts = quad.map(([x, z]) => [x, world.heightAt(x, z), z]);
    const tris = [[0, 1, 2], [0, 2, 3]];
    for (const tri of tris) {
      for (const k of tri) {
        triPos.push(pts[k][0], pts[k][1], pts[k][2]);
        const n = world.normalAt(pts[k][0], pts[k][2]);
        triNrm.push(n[0], n[1], n[2]);
        const mat = MATERIALS[world.materialIdAt(pts[k][0], pts[k][2])];
        triCol.push(mat.color[0], mat.color[1], mat.color[2]);
      }
    }
  }
}
void groundTris;

const out = '/tmp/scene';
const nTri = triPos.length / 9;
const buf = new Float32Array(nTri * 27);
for (let t = 0; t < nTri; t++) {
  for (let c = 0; c < 3; c++) {
    const o = t * 27 + c * 9;
    const s = t * 9 + c * 3;
    buf[o] = triPos[s]; buf[o + 1] = triPos[s + 1]; buf[o + 2] = triPos[s + 2];
    buf[o + 3] = triNrm[s]; buf[o + 4] = triNrm[s + 1]; buf[o + 5] = triNrm[s + 2];
    buf[o + 6] = triCol[s]; buf[o + 7] = triCol[s + 1]; buf[o + 8] = triCol[s + 2];
  }
}
fs.writeFileSync(out + '.bin', Buffer.from(buf.buffer));
fs.writeFileSync(out + '.json', JSON.stringify({
  triCount: nTri,
  camera: { pos: camera.position, target: camera.target, fov: camera.fov },
  light: [-0.4, 0.82, 0.4],
  car: {
    speed: car.forwardSpeed * 3.6,
    damage: car.damage,
    detached: car.wheels.filter((w) => w.detached).map((w) => w.name),
    center: car.center,
    up: car.basis.y,
  },
}, null, 2));
console.log(`scenario ${scenario}: ${nTri} triangles, ${(car.forwardSpeed * 3.6).toFixed(1)} km/h, ` +
  `damage ${car.damage.toFixed(3)}, detached ${car.wheels.filter((w) => w.detached).length}, ` +
  `up ${car.basis.y.map((v) => v.toFixed(2)).join(',')}`);
