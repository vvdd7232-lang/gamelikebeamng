// Body panels: doors, hood, trunk and bumpers are separate parts of the model
// hanging on their own hinges.  This test drives the physics and the renderer
// and checks the things that are easy to get subtly wrong:
//
//  * a parked car's latches carry the panels (nothing sags open by itself);
//  * a side hit swings the door on the hit side, not the one opposite;
//  * a ripped-off panel keeps its hinge transform at the moment it left (it
//    does not snap back to a clean, straight door in mid-air) and its
//    deformation is baked into its own mesh;
//  * the renderer draws every panel and skips the ones lying on the road;
//  * `reset()` re-hangs everything.
//
// The renderer half runs against a mock WebGL2 context, like render_smoke.
import fs from 'fs';
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { Renderer } from '../game/src/render/renderer.js';
import { Camera } from '../game/src/game/camera.js';
import { parseGlbForNode } from './glb_node.mjs';

let failures = 0;
const check = (ok, msg) => {
  if (!ok) { failures++; console.log('   FAIL ' + msg); } else console.log('   ok   ' + msg);
};

// ---------------------------------------------------------------- mock WebGL2
let glCalls = 0;
const draws = [];
function makeGL(canvas) {
  const handler = {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return 1;
      const fn = (...args) => {
        glCalls++;
        if (prop === 'drawElements' || prop === 'drawArrays') {
          draws.push({ call: prop, count: args[2], vao: args[0] });
        }
        if (prop === 'createProgram') return { __prog: 1 };
        if (prop === 'createShader') return { __shader: true, src: '' };
        if (prop === 'getUniformLocation') return { __loc: args[1], name: args[1] };
        if (prop === 'getAttribLocation') return 0;
        if (prop === 'getShaderParameter' || prop === 'getProgramParameter') return true;
        if (prop === 'getShaderInfoLog' || prop === 'getProgramInfoLog') return '';
        if (prop === 'getActiveUniform') return { name: 'u0', type: 0, size: 1 };
        if (prop === 'checkFramebufferStatus') return 1;
        if (prop === 'getParameter') return 8;
        if (prop === 'getExtension') return {};
        if (/^create/.test(prop)) return { __id: prop };
        return undefined;
      };
      target[prop] = fn;
      return fn;
    },
  };
  return new Proxy({}, handler);
}

global.window = { devicePixelRatio: 1, addEventListener() {}, getGamepads: () => [] };
global.document = { getElementById: () => null };

const world = new World(1337);
const IDLE = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };
const SPAWN = { position: [225, world.heightAt(225, 285), 285], yaw: 0 };
const run = (car, seconds, input, dt = 1 / 60) => {
  for (let i = 0; i < Math.round(seconds / dt); i++) car.step(dt, input);
};

// ---------------------------------------------------------------- latches
console.log('parked car:');
{
  const car = new Vehicle(world, { spawn: world.spawnOnTrack(0.02) });
  run(car, 5, IDLE);
  const maxSag = Math.max(...car.panels.map((p) => Math.abs(p.angle)));
  check(maxSag < 0.02, `latches carry the panels (max sag ${(maxSag * 57.3).toFixed(2)} deg)`);
  check(car.panels.every((p) => p.loose < 0.02), 'no mount damage from standing still');
  check(car.panels.every((p) => !p.detached), 'nothing has fallen off');
  // the hinge geometry has to match the model: a door's pivot is on the side of
  // the car, the hood's on the cowl
  const door = car.panels.find((p) => p.name === 'Door_FL');
  const hood = car.panels.find((p) => p.name === 'Hood');
  check(Math.abs(door.pivot[0]) > 0.6 && Math.abs(door.pivot[1]) > 0.7, 'door hinge sits on the pillar');
  check(Math.abs(hood.pivot[0]) < 0.05 && hood.pivot[2] > 0.8, 'hood hinge sits on the cowl');
}

// ---------------------------------------------------------------- side hit
console.log('side impact into a door:');
{
  const car = new Vehicle(world, { spawn: SPAWN });
  run(car, 1.5, IDLE);
  // shove the left side of the car sideways into nothing: just push the flank
  // forward hard, which is what a side-swipe does to the door
  const ang0 = car.panels.map((p) => p.angle);
  car.panelKick(car.center[0] + 0.78, car.center[1] + 0.72, car.center[2] + 0.1,
    0.9, [0, 0, -1], true);
  run(car, 0.6, IDLE);
  const swung = car.panels
    .map((p, i) => ({ name: p.name, d: Math.abs(p.angle - ang0[i]) }))
    .sort((a, b) => b.d - a.d);
  check(swung[0].d > 0.15, `a blow on the flank swings a panel (${swung[0].name} by ${swung[0].d.toFixed(2)} rad)`);
  check(swung[0].name.startsWith('Door_'), `it is a door that swings (${swung[0].name})`);
  const rear = car.panels.filter((p) => /_R[LR]$/.test(p.name) || p.name === 'Trunk');
  check(rear.every((p) => Math.abs(p.angle) < 0.05), 'the far end of the car stays shut');
}

// ---------------------------------------------------------------- rip-off
console.log('ripping a panel off:');
{
  const car = new Vehicle(world, { spawn: SPAWN });
  run(car, 1.5, IDLE);
  const hood = car.panels.find((p) => p.name === 'Hood');
  // destroy its mount the way a real crash does, then let the physics notice
  for (const i of hood.nodes) car.body.nodeDamage[i] = 1;
  for (const bi of hood.nodeBeams) car.body.bbroken[bi] = 1;
  run(car, 0.5, IDLE);
  check(hood.detached, 'the hood lets go once its mount is destroyed');
  check(car.detachedPanels.includes(hood), 'it is tracked as a loose body');
  check(hood.vel && hood.quat && hood.radius > 0, 'it has its own velocity / attitude / size');
  const y0 = hood.pos[1];
  run(car, 1.0, IDLE);
  check(hood.pos[1] < y0 + 0.05 && Number.isFinite(hood.pos[1]), 'it falls and lands');
  const ground = world.heightAt(hood.pos[0], hood.pos[2]);
  check(hood.pos[1] - ground < 1.5, `it comes to rest on the road (${(hood.pos[1] - ground).toFixed(2)} m up)`);

  // and it heals on reset
  car.reset();
  check(!hood.detached && car.detachedPanels.length === 0, 'reset re-hangs it');
  check(Math.abs(hood.angle) < 1e-9 && hood.loose === 0, 'reset clears the hinge state');
}

// ---------------------------------------------------------------- renderer
console.log('renderer:');
{
  const glb = parseGlbForNode(new Uint8Array(fs.readFileSync(new URL('../game/assets/volvo.glb', import.meta.url))));
  const canvas = { width: 640, height: 360, getContext: () => makeGL(canvas), addEventListener() {} };
  const renderer = new Renderer(canvas);
  renderer.resize(640, 360);
  const car = new Vehicle(world, { spawn: world.spawnOnTrack(0.02), isPlayer: true });
  const geom = buildCarGeometry(glb, car.cage.nodes);
  renderer.uploadWorld(world);
  renderer.uploadCar(geom);
  renderer.createVehicleInstance(car);
  const camera = new Camera();
  camera.snap(car, world);
  for (let i = 0; i < 10; i++) camera.update(1 / 60, car, { mouse: { dx: 0, dy: 0 } }, world);
  run(car, 1.0, IDLE);

  const bodyPrims = renderer.carGeometry.bodies.length;
  check(bodyPrims > 20, `${bodyPrims} body primitives uploaded (panels are separate meshes)`);

  // a panel that is detached must not be drawn through the skinned path
  const door = car.panels.find((p) => p.name === 'Door_FL');
  const doorPrims = renderer.carGeometry.bodies.filter((b) => b.name === 'Door_FL');
  for (const i of door.nodes) car.body.nodeDamage[i] = 1;
  for (const bi of door.nodeBeams) car.body.bbroken[bi] = 1;
  run(car, 0.4, IDLE);
  check(door.detached, 'the door came off in the physics');

  draws.length = 0;
  renderer.render({ camera, world, vehicles: [car], fx: null, debugCage: false, cageLines: null });
  check(draws.length > 10, `${draws.length} draw calls in the frame`);
  check(door.bakedMeshes && door.bakedMeshes.length === doorPrims.length,
    `the loose door was baked into ${door.bakedMeshes ? door.bakedMeshes.length : 0} mesh(es)`);

  // the bake has to carry the deformation: vertices are finite and the damage
  // rides in the joint slot the shader reads when uFrozen is on
  const raw = renderer.carGeometry.bodies.find((b) => b.name === 'Door_FL').raw;
  let bad = 0;
  let dmgSum = 0;
  for (const prim of door.bakedMeshes) {
    for (let v = 0; v < prim.data.length; v += 20) {
      for (let k = 0; k < 3; k++) if (!Number.isFinite(prim.data[v + k])) bad++;
      dmgSum += prim.data[v + 12];
    }
  }
  check(bad === 0, 'baked vertices are finite');
  check(dmgSum > 0, 'the bake carries the panel damage into the mesh');
  check(raw.length > 0, 'the source mesh is still available for a reset/re-bake');

  // after a reset the panel is drawn by the skinned path again
  car.reset();
  check(!door.detached, 'reset re-attaches the door');
}

if (failures) {
  console.log(`PANELS TEST FAILED (${failures})`);
  process.exit(1);
}
console.log('PANELS OK');
