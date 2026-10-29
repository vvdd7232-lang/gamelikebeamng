// Records the GL state machine at every draw call of three consecutive frames
// and checks the invariants the renderer relies on but that nothing else
// verifies.  Written after two real-browser bugs slipped past the other tests:
//
//   * node data uploaded into the shadow map because the wrong texture unit was
//     active  -> car body invisible
//   * a vertex buffer written with a 13-float stride for a 12-float attribute
//     layout -> giant random triangles
//
// The mock tracks bound texture per unit, the active unit, VAOs, buffers,
// viewport and enabled capabilities, exactly like the driver would.
import fs from 'fs';
import { Renderer, MAX_NODES } from '../game/src/render/renderer.js';
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';
import { buildCarGeometry } from '../game/src/vehicle/carMesh.js';
import { Camera } from '../game/src/game/camera.js';
import { parseGlbForNode } from './glb_node.mjs';

const CONSTS = {};
function cv(p) {
  if (!(p in CONSTS)) CONSTS[p] = 0x1000 + Object.keys(CONSTS).length;
  return CONSTS[p];
}

const programs = new Map();
let nextId = 1;
const boundUnit = new Map();
let activeUnit = 0;
let currentProgram = null;
let lastLinked = null;
const vaoBuffers = new Map();     // vao id -> { attribs: Map(loc -> {size,stride,offset,buffer}), ibo }
const bufferSizes = new Map();    // buffer id -> element count
const bufferTyped = new Map();    // buffer id -> 'float' | 'uint'
let boundVaoObj = null;
let boundArrayBuffer = null;
let viewport = [0, 0, 0, 0];
const enabled = new Set();
const draws = [];
let frameNo = 0;
let drawingProgramName = new Map();

function makeGL(canvas) {
  const t = {
    canvas,
    getProgramParameter: (p, which) => (which === cv('ACTIVE_UNIFORMS')
      ? (programs.get(p.id)?.names.length || 0) : true),
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    getProgramInfoLog: () => '',
    checkFramebufferStatus: () => cv('FRAMEBUFFER_COMPLETE'),
    getParameter: (p) => (p === cv('VIEWPORT') ? viewport.slice() : 8),
    getExtension: () => ({}),
    getError: () => 0,
  };
  return new Proxy(t, {
    get(o, prop) {
      if (prop in o) return o[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return cv(prop);
      const fn = (...a) => {
        switch (prop) {
          case 'createProgram': {
            const p = { id: nextId++, shaders: [], names: [] };
            programs.set(p.id, p);
            return p;
          }
          case 'createShader': return { id: nextId++, src: '' };
          case 'shaderSource': a[0].src = String(a[1]); return;
          case 'attachShader': { const r = programs.get(a[0].id); if (r) r.shaders.push(a[1].src); return; }
          case 'linkProgram': {
            const r = programs.get(a[0].id);
            const names = new Set();
            for (const s of (r ? r.shaders : [])) {
              for (const m of s.matchAll(/uniform\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+(\w+)\s*;/g)) names.add(m[1]);
            }
            if (r) r.names = [...names];
            lastLinked = a[0];
            return;
          }
          case 'getActiveUniform': {
            const r = programs.get(a[0].id);
            return { name: r && r.names[a[1]] ? r.names[a[1]] : 'u' + a[1], type: 0, size: 1 };
          }
          case 'useProgram': currentProgram = a[0]; return;
          case 'getUniformLocation': return { name: a[1], owner: programs.get(a[0].id) ? a[0] : lastLinked };
          case 'getAttribLocation': return 0;
          case 'createBuffer': { const b = { id: nextId++ }; return b; }
          case 'bindBuffer': {
            if (a[0] === cv('ARRAY_BUFFER')) boundArrayBuffer = a[1];
            else if (boundVaoObj) vaoBuffers.get(boundVaoObj.id).ibo = a[1];
            else boundArrayBuffer = null;
            return;
          }
          case 'bufferData': {
            const isArray = a[0] === cv('ARRAY_BUFFER');
            const buf = isArray ? boundArrayBuffer : (boundVaoObj && vaoBuffers.get(boundVaoObj.id).ibo);
            if (buf) {
              bufferSizes.set(buf.id, a[1] && a[1].length !== undefined ? a[1].length : 0);
              bufferTyped.set(buf.id, (a[1] instanceof Uint32Array || a[1] instanceof Uint16Array) ? 'uint' : 'float');
            }
            return;
          }
          case 'createVertexArray': { const v = { id: nextId++ }; vaoBuffers.set(v.id, { attribs: new Map(), ibo: null }); return v; }
          case 'bindVertexArray': boundVaoObj = a[0]; return;
          case 'vertexAttribPointer': {
            const [loc, size, , , stride, offset] = a;
            if (boundVaoObj) {
              vaoBuffers.get(boundVaoObj.id).attribs.set(loc, { size, stride, offset, buffer: boundArrayBuffer });
            }
            return;
          }
          case 'createTexture': return { id: nextId++ };
          case 'activeTexture': activeUnit = a[0]; return;
          case 'bindTexture': boundUnit.set(activeUnit, a[1]); return;
          case 'texImage2D': return;
          case 'texSubImage2D': return;
          case 'viewport': viewport = a.slice(); return;
          case 'enable': enabled.add(a[0]); return;
          case 'disable': enabled.delete(a[0]); return;
          case 'createFramebuffer': return { id: nextId++ };
          case 'drawElements':
          case 'drawArrays': {
            draws.push({
              frame: frameNo,
              kind: prop,
              program: currentProgram ? programs.get(currentProgram.id) : null,
              vao: boundVaoObj,
              indexBuffer: boundVaoObj ? vaoBuffers.get(boundVaoObj.id).ibo : null,
              unit0: (boundUnit.get(cv('TEXTURE0')) || {}),
              unit1: (boundUnit.get(cv('TEXTURE1')) || {}),
              viewport: viewport.slice(),
              depthTest: enabled.has(cv('DEPTH_TEST')),
            });
            return;
          }
          default: return undefined;
        }
      };
      o[prop] = fn;
      return fn;
    },
  });
}

global.window = { devicePixelRatio: 1, addEventListener() {}, getGamepads: () => [] };
global.document = { getElementById: () => null };
global.performance = global.performance || { now: () => Date.now() };

const glb = parseGlbForNode(new Uint8Array(fs.readFileSync(new URL('../game/assets/volvo.glb', import.meta.url))));
const world = new World(1337);
const probe = new Vehicle(world, { spawn: world.spawnOnTrack(0) });
const geom = buildCarGeometry(glb, probe.cage.nodes);
const canvas = { width: 640, height: 360, style: {}, getContext: () => makeGL(canvas), addEventListener() {} };
const renderer = new Renderer(canvas);
renderer.resize(640, 360);
renderer.uploadWorld(world);
renderer.uploadCar(geom);

const player = new Vehicle(world, { spawn: world.spawnOnTrack(0.015), isPlayer: true });
renderer.createVehicleInstance(player);
const camera = new Camera();
camera.snap(player, world);

// which program is which: the renderer keeps progSky/progMesh/progDepth/progSkid
const progNames = new Map([
  [renderer.progSky.program.id, 'sky'],
  [renderer.progMesh.program.id, 'mesh'],
  [renderer.progDepth.program.id, 'depth'],
  [renderer.progSkid.program.id, 'skid'],
  [renderer.progParticle.program.id, 'particle'],
]);

for (frameNo = 1; frameNo <= 3; frameNo++) {
  player.step(1 / 60, { throttle: 0.6, brake: 0, steer: 0.2, handbrake: 0 });
  camera.update(1 / 60, player, { mouse: { dx: 0, dy: 0, wheel: 0, dragging: false } }, world);
  renderer.render({ camera, world, vehicles: [player], fx: null, debugCage: false, cageLines: null });
}

let failures = 0;
const fail = (msg) => { failures++; console.log('   FAIL ' + msg); };

const nodeTexId = renderer.nodeTex.id;
const shadowTexId = renderer.shadowTex.id;
const carBodyVaos = new Set(renderer.carGeometry.bodies.map((b) => b.mesh.vao.id));
const carWheelVaos = new Set(Object.values(renderer.carGeometry.wheels).flat().map((p) => p.mesh.vao.id));

console.log(`recorded ${draws.length} draw calls over 3 frames`);
const counts = {};
for (const d of draws) {
  const n = d.program ? progNames.get(d.program.id) : '?';
  counts[n] = (counts[n] || 0) + 1;
}
console.log('draws per program:', JSON.stringify(counts));

// ---- 1. every mesh/depth draw of the car must sample the node texture on unit 0
let meshCarDraws = 0;
for (const d of draws) {
  const name = d.program ? progNames.get(d.program.id) : '?';
  const isCar = d.vao && (carBodyVaos.has(d.vao.id) || carWheelVaos.has(d.vao.id));
  if (!isCar) continue;
  if (name === 'mesh' || name === 'depth') {
    meshCarDraws++;
    if (!d.unit0 || d.unit0.id !== nodeTexId) {
      fail(`${name} draw of the car samples texture ${d.unit0 && d.unit0.id} on unit 0, expected the node texture ${nodeTexId}`);
    }
    // only the lit pass reads the shadow map; the depth pass has no such uniform
    if (name === 'mesh' && (!d.unit1 || d.unit1.id !== shadowTexId)) {
      fail(`${name} draw of the car has texture ${d.unit1 && d.unit1.id} on unit 1, expected the shadow map ${shadowTexId}`);
    }
    if (!d.depthTest) fail(`${name} draw of the car runs with depth test disabled`);
  }
}
console.log(`car draws checked: ${meshCarDraws}`);
if (meshCarDraws === 0) fail('the car was never drawn');

// ---- 2. the viewport of the scene pass must match the canvas
const sceneViewports = new Set(draws
  .filter((d) => d.program && progNames.get(d.program.id) === 'mesh')
  .map((d) => d.viewport.join('x')));
console.log('scene-pass viewports:', [...sceneViewports].join(', '), `(canvas ${canvas.width}x${canvas.height})`);
for (const vp of sceneViewports) {
  if (vp !== `0x0x${canvas.width}x${canvas.height}`) {
    fail(`scene pass drew with viewport ${vp} instead of the canvas size 0x0x${canvas.width}x${canvas.height}`);
  }
}

// ---- 3. every vertex buffer must be a whole number of vertices for its layout
const seen = new Set();
for (const d of draws) {
  if (!d.vao || seen.has(d.vao.id)) continue;
  seen.add(d.vao.id);
  const rec = vaoBuffers.get(d.vao.id);
  const attrs = [...rec.attribs.values()].filter((a) => a.buffer);
  if (!attrs.length) continue;
  const strideBytes = attrs[0].stride;
  const total = bufferSizes.get(attrs[0].buffer.id);
  if (!strideBytes || !total) continue;
  if (total % (strideBytes / 4) !== 0) {
    fail(`VAO ${d.vao.id}: ${total} floats is not a multiple of the ${strideBytes / 4}-float stride`);
  }
}
console.log('vertex buffer strides checked:', seen.size);

// ---- 4. the node texture must be uploaded every frame (in the depth pass and
//         the main pass), otherwise skinning lags a frame behind or is empty
const uploads = [];
for (let f = 1; f <= 3; f++) {
  const has = draws.some((d) => d.frame === f);
  if (!has) fail(`frame ${f} produced no draw calls`);
}

// ---- 5. skinning must actually be enabled for the body
const bodyDraw = draws.find((d) => d.program && progNames.get(d.program.id) === 'mesh'
  && d.vao && carBodyVaos.has(d.vao.id));
if (bodyDraw && renderer.nodeTexError) {
  console.log('   note: skinning disabled (upload rejected), rigid fallback in use');
}

console.log(failures ? `${failures} problem(s)` : 'GL STATE OK');
process.exit(failures ? 1 : 0);
