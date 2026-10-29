// Headless smoke test: runs the whole game loop (world + cars + renderer) against
// a mock WebGL2 context to catch JS integration errors.  Shader compilation is
// verified separately with glslang (see tools/check_shaders.mjs).
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { Effects } from '../game/src/fx/effects.js';
import { Camera } from '../game/src/game/camera.js';
import { AIDriver } from '../game/src/game/ai.js';
import { Renderer } from '../game/src/render/renderer.js';
import fs from 'fs';

// ---------------------------------------------------------------- mock WebGL2
const glCalls = new Map();
// program -> its shader sources, so uniform/attribute requests can be checked
// against the code that is actually compiled (catches typos that a mock would
// silently swallow and that a real driver reports as a null location).
const programSources = new Map();
const missingUniforms = new Set();
const missingAttribs = new Set();
let current = null;
function makeGL(canvas) {
  const handler = {
    get(target, prop) {
      if (prop in target) return target[prop];
      // numeric constants
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return 1;
      const fn = (...args) => {
        glCalls.set(prop, (glCalls.get(prop) || 0) + 1);
        if (prop === 'createProgram') { const p = { __prog: programSources.size }; programSources.set(p, []); return p; }
        if (prop === 'createShader') return { __shader: true, src: '' };
        if (prop === 'shaderSource') { args[0].src = String(args[1]); return undefined; }
        if (prop === 'attachShader') { programSources.get(args[0]).push(args[1].src); return undefined; }
        if (prop === 'getUniformLocation') {
          const srcs = programSources.get(args[0]) || [];
          const name = args[1].replace(/\[0\]$/, '');
          const decl = new RegExp('uniform[^;]*\\b' + name + '\\b');
          if (!srcs.some((s) => decl.test(s))) missingUniforms.add(name);
          return { __loc: args[1] };
        }
        if (prop === 'getAttribLocation') {
          const srcs = programSources.get(args[0]) || [];
          const decl = new RegExp('in[^;]*\\b' + args[1] + '\\b');
          if (!srcs.some((s) => decl.test(s))) missingAttribs.add(args[1]);
          return 0;
        }
        if (prop === 'getShaderParameter' || prop === 'getProgramParameter') return true;
        if (prop === 'getShaderInfoLog' || prop === 'getProgramInfoLog') return '';
        if (prop === 'getActiveUniform') {
          const n = target.__uniformNames ? target.__uniformNames[target.__uniformIdx++] : 'u0';
          return { name: n, type: 0, size: 1 };
        }
        if (prop === 'getProgramParameter' ) return 1;
        if (prop === 'checkFramebufferStatus') return 1;
        if (prop === 'createBuffer' || prop === 'createTexture' || prop === 'createVertexArray'
          || prop === 'createProgram' || prop === 'createShader' || prop === 'createFramebuffer') return { __id: prop };
        if (prop === 'getParameter') return 8;
        if (prop === 'getExtension') return {};
        if (prop === 'getUniformLocation') return { __loc: args[1] };
        return undefined;
      };
      target[prop] = fn;
      return fn;
    },
  };
  const base = {
    canvas,
    ACTIVE_UNIFORMS: 35718,
    getProgramParameter: (p, which) => (which === 35718 ? 0 : 1),
  };
  return new Proxy(base, handler);
}

global.window = { devicePixelRatio: 1, addEventListener() {}, getGamepads: () => [] };
global.document = { getElementById: () => null };
global.performance = global.performance || { now: () => Date.now() };

const canvas = { width: 1280, height: 720, getContext: () => makeGL(canvas), addEventListener() {} };

const glbBuffer = new Uint8Array(fs.readFileSync(new URL('../game/assets/tesla.glb', import.meta.url)));
const { parseGlbForNode } = await import('./glb_node.mjs');
const glb = parseGlbForNode(glbBuffer);

const world = new World(1337);
const probe = new Vehicle(world, { spawn: world.spawnOnTrack(0) });
const geom = buildCarGeometry(glb, probe.cage.nodes);

const renderer = new Renderer(canvas);
renderer.resize(1280, 720);
renderer.uploadWorld(world);
renderer.uploadCar(geom);

const cars = [];
const player = new Vehicle(world, { spawn: world.spawnOnTrack(0.015), isPlayer: true });
cars.push(player);
renderer.createVehicleInstance(player);
const ai = new Vehicle(world, { spawn: world.spawnOnTrack(0.005) });
cars.push(ai);
renderer.createVehicleInstance(ai);
const aiDriver = new AIDriver(ai, world, 0.9);

const effects = new Effects(world);
const camera = new Camera();
camera.snap(player, world);

const fakeInput = {
  mouse: { dx: 0, dy: 0, wheel: 0, dragging: false },
};
const dt = 1 / 60;
let frames = 0;
for (let i = 0; i < 240; i++) {
  const inp = { throttle: i > 60 ? 1 : 0, brake: 0, steer: i > 150 ? 0.4 : 0, handbrake: 0 };
  player.step(dt, inp);
  aiDriver.update(dt);
  ai.step(dt, ai.lastInput);
  effects.carEffects(player, dt, player.speed);
  effects.carEffects(ai, dt, ai.speed);
  effects.update(dt);
  camera.update(dt, player, fakeInput, world);
  renderer.render({ camera, world, vehicles: cars, fx: effects, debugCage: true, cageLines: { data: new Float32Array(400), count: 10 } });
  frames++;
}
console.log('rendered', frames, 'frames ok');
console.log('player speed', (player.forwardSpeed * 3.6).toFixed(1), 'km/h, damage', player.damage.toFixed(3));
console.log('ai speed', (ai.forwardSpeed * 3.6).toFixed(1), 'km/h');
console.log('particles', effects.particles.count, 'skid verts', effects.skidCount);
if (missingUniforms.size) console.log('  MISSING uniforms:', [...missingUniforms].join(', '));
if (missingAttribs.size) console.log('  MISSING attributes:', [...missingAttribs].join(', '));
if (!missingUniforms.size && !missingAttribs.size) console.log('  all uniform/attribute names resolve in their programs');
const expected = ['drawElements', 'drawArrays', 'bufferSubData', 'uniformMatrix4fv', 'texSubImage2D'];
for (const e of expected) console.log('  gl.' + e, glCalls.get(e) || 0);
if (missingUniforms.size || missingAttribs.size) process.exit(1);
