// Verifies the CPU -> GPU node data path.
//
// The skinned shaders fetch node data with
//     texelFetch(uNodeTex, ivec2(j, row), 0)
// from a 320 x 5 RGBA32F texture, i.e. node j must sit in **column j**.  The
// upload is one flat float array, so a packing mistake (writing the 5 texels of
// a node consecutively instead of per column, or uploading D transposed) is
// invisible to the JS-side smoke test but scrambles every vertex on the GPU.
//
// This test packs the nodes exactly like the renderer does, reads them back the
// way the shader does, runs the same skinning formula as MESH_VS and compares it
// with the physically expected result.
import fs from 'fs';
import { MAX_NODES, packNodeData } from '../game/src/render/renderer.js';
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { parseGlbForNode } from './glb_node.mjs';

const glb = parseGlbForNode(new Uint8Array(fs.readFileSync(new URL('../game/assets/volvo.glb', import.meta.url))));
const world = new World(1337);
const car = new Vehicle(world, { spawn: world.spawnOnTrack(0.02), isPlayer: true });

// settle, then bend the body so both positions and rotations are non-trivial
for (let i = 0; i < 240; i++) car.step(1 / 60, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
for (let i = 0; i < 45; i++) car.step(1 / 60, { throttle: 0.6, brake: 0, steer: 0.4, handbrake: 0 });
car.updateNodeTex();
for (let i = 0; i < car.body.nb; i += 7) car.body.bbroken[i] = 0;

const geom = buildCarGeometry(glb, car.cage.nodes);
const prim = geom.bodies[0];
const data = prim.data;
const STRIDE = 20;

const packed = new Float32Array(MAX_NODES * 5 * 4);
const packedCount = packNodeData(car, packed);
console.log('nodes packed:', packedCount, 'texture', MAX_NODES, 'x 5');

// --- structural checks: node j must occupy column j of every row
let structErr = 0;
for (const j of [0, 1, 7, 100, car.nodeCount - 1]) {
  const col0 = [car.nodeTex[j * 16 + 4], car.nodeTex[j * 16 + 5], car.nodeTex[j * 16 + 6]];
  const px = packed[(1 * MAX_NODES + j) * 4];
  const py = packed[(1 * MAX_NODES + j) * 4 + 1];
  const pz = packed[(1 * MAX_NODES + j) * 4 + 2];
  structErr = Math.max(structErr, Math.abs(px - col0[0]), Math.abs(py - col0[1]), Math.abs(pz - col0[2]));
  structErr = Math.max(structErr, Math.abs(packed[j * 4] - car.body.pos[j * 3]));
  structErr = Math.max(structErr, Math.abs(packed[(4 * MAX_NODES + j) * 4] - car.restWorld[j * 3]));
}
console.log('off-by-layout error (float32 rounding only):', structErr.toExponential(2));

// --- the shader, emulated on the packed buffer (texelFetch semantics)
const fetchTexel = (j, row) => {
  const o = (row * MAX_NODES + j) * 4;
  return [packed[o], packed[o + 1], packed[o + 2]];
};
function skinFromTexture(vertex) {
  const o = vertex * STRIDE;
  const aPos = [data[o], data[o + 1], data[o + 2]];
  const R0 = car.R0, P0 = car.spawnPos;
  const bind = [
    R0[0] * aPos[0] + R0[1] * aPos[1] + R0[2] * aPos[2] + P0[0],
    R0[3] * aPos[0] + R0[4] * aPos[1] + R0[5] * aPos[2] + P0[1],
    R0[6] * aPos[0] + R0[7] * aPos[1] + R0[8] * aPos[2] + P0[2],
  ];
  let ax = 0, ay = 0, az = 0;
  for (let k = 0; k < 4; k++) {
    const j = data[o + 12 + k];
    const w = data[o + 16 + k];
    if (w <= 0) continue;
    const nodePos = fetchTexel(j, 0);
    const c0 = fetchTexel(j, 1), c1 = fetchTexel(j, 2), c2 = fetchTexel(j, 3);
    const restW = fetchTexel(j, 4);
    // D * (bind - restW) with D built from columns, as mat3(c0, c1, c2) does
    const d = [bind[0] - restW[0], bind[1] - restW[1], bind[2] - restW[2]];
    ax += w * (nodePos[0] + c0[0] * d[0] + c1[0] * d[1] + c2[0] * d[2]);
    ay += w * (nodePos[1] + c0[1] * d[0] + c1[1] * d[1] + c2[1] * d[2]);
    az += w * (nodePos[2] + c0[2] * d[0] + c1[2] * d[1] + c2[2] * d[2]);
  }
  return [ax, ay, az];
}

// --- same formula straight from the physics state (the reference)
function skinReference(vertex) {
  const o = vertex * STRIDE;
  const aPos = [data[o], data[o + 1], data[o + 2]];
  const R0 = car.R0, P0 = car.spawnPos;
  const bind = [
    R0[0] * aPos[0] + R0[1] * aPos[1] + R0[2] * aPos[2] + P0[0],
    R0[3] * aPos[0] + R0[4] * aPos[1] + R0[5] * aPos[2] + P0[1],
    R0[6] * aPos[0] + R0[7] * aPos[1] + R0[8] * aPos[2] + P0[2],
  ];
  let ax = 0, ay = 0, az = 0;
  for (let k = 0; k < 4; k++) {
    const j = data[o + 12 + k];
    const w = data[o + 16 + k];
    if (w <= 0) continue;
    const t = j * 16;
    const nodePos = [car.body.pos[j * 3], car.body.pos[j * 3 + 1], car.body.pos[j * 3 + 2]];
    const c0 = [car.nodeTex[t + 4], car.nodeTex[t + 5], car.nodeTex[t + 6]];
    const c1 = [car.nodeTex[t + 8], car.nodeTex[t + 9], car.nodeTex[t + 10]];
    const c2 = [car.nodeTex[t + 12], car.nodeTex[t + 13], car.nodeTex[t + 14]];
    const restW = [car.restWorld[j * 3], car.restWorld[j * 3 + 1], car.restWorld[j * 3 + 2]];
    const d = [bind[0] - restW[0], bind[1] - restW[1], bind[2] - restW[2]];
    ax += w * (nodePos[0] + c0[0] * d[0] + c1[0] * d[1] + c2[0] * d[2]);
    ay += w * (nodePos[1] + c0[1] * d[0] + c1[1] * d[1] + c2[1] * d[2]);
    az += w * (nodePos[2] + c0[2] * d[0] + c1[2] * d[1] + c2[2] * d[2]);
  }
  return [ax, ay, az];
}

let maxErr = 0;
let samples = 0;
for (let v = 0; v < prim.vertexCount; v += 37) {
  const a = skinFromTexture(v);
  const b = skinReference(v);
  maxErr = Math.max(maxErr, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  samples++;
}
console.log(`skinning through the packed texture vs physics state: ${samples} vertices, max error ${maxErr.toExponential(2)} m`);

// --- and the deformation must actually be non-trivial, otherwise the test is vacuous
let maxDisp = 0;
for (let i = 0; i < car.nodeCount; i++) {
  maxDisp = Math.max(maxDisp, Math.hypot(
    car.nodeTex[i * 16], car.nodeTex[i * 16 + 1], car.nodeTex[i * 16 + 2]));
}
console.log('largest node displacement in this pose:', maxDisp.toFixed(3), 'm');

// tolerances are float32 rounding, not formula differences
const ok = structErr < 1e-4 && maxErr < 1e-4 && maxDisp > 0.01;
console.log(ok ? 'NODE TEXTURE OK' : 'NODE TEXTURE FAILED');
process.exit(ok ? 0 : 1);
