// Regression test for a WebGL state bug that made the car body invisible:
//
// uploadNodeTexture() used to call gl.bindTexture() on whatever texture unit
// happened to be active.  The main render pass leaves TEXTURE1 (the shadow map)
// active, so from the second frame on the node data was uploaded *into the
// shadow map* and the real node texture kept its zeroed initial content.  Every
// skinned vertex then collapsed to a single point: the car body disappeared
// while the rigid wheels kept rendering.
//
// The mock context below tracks the texture object bound to every unit, so the
// invariant "the node data lands in the texture the draw samples" can be
// checked directly.
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
const textures = new Map();     // id -> { uploads: [], boundUnits: Set }
const boundToUnit = new Map();  // unit -> texture object
let activeUnit = 0;
let currentProgram = null;
let lastLinked = null;
const uploads = [];             // { unit, texture, w, h, frame }
const draws = [];               // { unit0Texture, skinned }
let frameNo = 0;

function makeGL(canvas) {
  const t = {
    canvas,
    getProgramParameter: (p, which) => (which === cv('ACTIVE_UNIFORMS')
      ? (programs.get(p.id)?.names.length || 0) : true),
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    getProgramInfoLog: () => '',
    checkFramebufferStatus: () => cv('FRAMEBUFFER_COMPLETE'),
    getParameter: () => 8,
    getExtension: () => ({}),
    getError: () => 0,
  };
  return new Proxy(t, {
    get(o, prop) {
      if (prop in o) return o[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return cv(prop);
      const fn = (...a) => {
        if (prop === 'createProgram') { const p = { id: nextId++, shaders: [], names: [] }; programs.set(p.id, p); return p; }
        if (prop === 'createShader') return { id: nextId++, src: '' };
        if (prop === 'shaderSource') { a[0].src = String(a[1]); return; }
        if (prop === 'attachShader') { const r = programs.get(a[0].id); if (r) r.shaders.push(a[1].src); return; }
        if (prop === 'linkProgram') {
          const r = programs.get(a[0].id);
          const names = new Set();
          for (const s of (r ? r.shaders : [])) {
            for (const m of s.matchAll(/uniform\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+(\w+)\s*;/g)) names.add(m[1]);
          }
          if (r) r.names = [...names];
          lastLinked = a[0];
          return;
        }
        if (prop === 'getActiveUniform') {
          const r = programs.get(a[0].id);
          return { name: r && r.names[a[1]] ? r.names[a[1]] : 'u' + a[1], type: 0, size: 1 };
        }
        if (prop === 'useProgram') { currentProgram = a[0]; return; }
        if (prop === 'getUniformLocation') return { name: a[1], owner: programs.get(a[0].id) ? a[0] : lastLinked };
        if (prop === 'getAttribLocation') return 0;
        if (prop === 'createTexture') { const tex = { id: nextId++ }; textures.set(tex.id, { tex, uploads: [] }); return tex; }
        if (prop === 'activeTexture') { activeUnit = a[0]; return; }
        if (prop === 'bindTexture') { boundToUnit.set(activeUnit, a[1]); return; }
        if (prop === 'texImage2D') {
          const tex = boundToUnit.get(activeUnit);
          if (tex) textures.get(tex.id).allocated = [a[3], a[4]];
          return;
        }
        if (prop === 'texSubImage2D') {
          // texSubImage2D(target, level, x, y, w, h, format, type, data)
          const tex = boundToUnit.get(activeUnit);
          uploads.push({
            frame: frameNo, unit: activeUnit, texture: tex ? tex.id : null,
            w: a[4], h: a[5], length: a[8] ? a[8].length : 0,
          });
          if (tex) textures.get(tex.id).uploads.push(frameNo);
          return;
        }
        if (prop === 'drawElements' || prop === 'drawArrays') {
          draws.push({
            frame: frameNo,
            kind: prop,
            skinned: currentProgram ? undefined : undefined,
            unit0: (boundToUnit.get(cv('TEXTURE0')) || {}).id,
          });
          return;
        }
        if (prop === 'getUniformLocation' || prop === 'getAttribLocation') return 0;
        return undefined;
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

// three frames: the bug only appears once a previous frame has left TEXTURE1 bound
const nodeTexObject = renderer.nodeTex;
let failures = 0;
for (frameNo = 1; frameNo <= 3; frameNo++) {
  player.step(1 / 60, { throttle: 0.5, brake: 0, steer: 0, handbrake: 0 });
  camera.update(1 / 60, player, { mouse: { dx: 0, dy: 0, wheel: 0, dragging: false } }, world);
  renderer.render({ camera, world, vehicles: [player], fx: null, debugCage: false, cageLines: null });

  const thisFrame = uploads.filter((u) => u.frame === frameNo);
  const nodeUploads = thisFrame.filter((u) => u.w === MAX_NODES && u.h === 5 && u.length === MAX_NODES * 5 * 4);
  const tex0 = cv('TEXTURE0');
  const onUnit0 = nodeUploads.filter((u) => u.unit === tex0);

  console.log(`frame ${frameNo}: ${thisFrame.length} texture uploads, ` +
    `${nodeUploads.length} of them are node data (320x5), ${onUnit0.length} on TEXTURE0`);

  if (nodeUploads.length === 0) {
    failures++;
    console.log('   FAIL: the node data was never uploaded this frame');
  }
  const intoNodeTexture = nodeUploads.filter((u) => u.texture === nodeTexObject.id);
  if (intoNodeTexture.length === 0) {
    failures++;
    console.log(`   FAIL: node data went into texture(s) ` +
      `${[...new Set(nodeUploads.map((u) => u.texture))].join(',')} instead of the node texture ${nodeTexObject.id}`);
  }
  if (onUnit0.length !== nodeUploads.length) {
    failures++;
    console.log(`   FAIL: ${nodeUploads.length - onUnit0.length} node upload(s) happened on a unit other ` +
      `than TEXTURE0 (units ${[...new Set(nodeUploads.map((u) => u.unit))].join(',')}) -- ` +
      'bindTexture is relative to the active unit, so the data goes into whatever texture sits there');
  }
  const wrong = nodeUploads.filter((u) => u.texture !== nodeTexObject.id);
  if (wrong.length) {
    failures++;
    console.log(`   FAIL: ${wrong.length} upload(s) of node data landed in a different texture ` +
      `(units ${[...new Set(wrong.map((u) => u.unit))].join(',')})`);
  }
}

// the node texture must hold data by now, and the position row must be non-zero
const posRow = player.nodeTex.slice(0, 4);
const anyNonZero = player.nodeTex.some((v, i) => i % 16 === 0 && v !== 0);
console.log('node texture first node position offset:', Array.from(posRow).map((v) => v.toFixed(3)).join(','));
if (!anyNonZero) {
  failures++;
  console.log('   FAIL: no node has a non-zero displacement');
}

console.log(failures ? `${failures} problem(s)` : 'NODE TEXTURE UNIT OK');
process.exit(failures ? 1 : 0);
