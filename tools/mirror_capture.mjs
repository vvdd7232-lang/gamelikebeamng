// Records one real frame of the renderer (all draw calls, their vertex buffers,
// attribute layouts, index buffers and uniforms) so that tools/mirror_shade.py
// can replay it on the CPU.
//
// Why: the game cannot be opened in a browser inside this sandbox, and every
// GPU-side bug found so far (varying types, node texture layout, vertex stride)
// was invisible to the JS smoke test.  Replaying the *actual* recorded draw
// data through the *actual* shader math gives a faithful picture without a GPU.
//
//   node tools/mirror_capture.mjs [outDir]
import fs from 'fs';
import path from 'path';

const outDir = process.argv[2] || '/tmp/mirror';
fs.mkdirSync(outDir, { recursive: true });

// ---------------------------------------------------------------- mock WebGL2
const glCalls = [];
const programs = new Map();
let nextId = 1;
const buffers = new Map();     // id -> { target, data: Float32Array|Uint32Array }
const textures = new Map();    // id -> { data }
const vaos = new Map();        // id -> { attribs: Map<loc, {buffer, size, stride, offset, type}>, ibo }
let boundArrayBuffer = null;
let boundElementBuffer = null;
let boundVao = null;
let boundTexture = null;
let currentProgram = null;
let lastLinked = null;
let viewport = [0, 0, 1280, 720];
const uniformState = new Map(); // program id -> Map(name -> value)

function constVal(prop) {
  // stable fake enum values; only used for comparisons inside the game code
  if (!(prop in CONSTS)) CONSTS[prop] = 0x1000 + Object.keys(CONSTS).length;
  return CONSTS[prop];
}
const CONSTS = {};

function makeGL(canvas) {
  const target = {
    canvas,
    drawingBufferWidth: 1280,
    drawingBufferHeight: 720,
    getProgramParameter: (p, which) => (which === constVal('ACTIVE_UNIFORMS')
      ? (programs.get(p.id)?.names.length || 0)
      : true),
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    getProgramInfoLog: () => '',
    checkFramebufferStatus: () => constVal('FRAMEBUFFER_COMPLETE'),
    getParameter: (p) => (p === constVal('VIEWPORT') ? viewport.slice() : 8),
    getExtension: () => ({}),
  };
  return new Proxy(target, {
    get(t, prop) {
      if (prop in t) return t[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return constVal(prop);
      const fn = (...args) => {
        if (prop === 'createProgram') {
          const p = { id: nextId++, shaders: [], names: [] };
          programs.set(p.id, p);
          uniformState.set(p.id, new Map());
          return p;
        }
        if (prop === 'createShader') return { id: nextId++, src: '' };
        if (prop === 'shaderSource') { args[0].src = String(args[1]); return; }
        if (prop === 'attachShader') {
          const rec = programs.get(args[0].id);
          if (rec) rec.shaders.push(args[1].src);
          return;
        }
        if (prop === 'linkProgram') {
          const rec = programs.get(args[0].id);
          // uniforms are only active if used; the game reads them back by name,
          // so a superset of declared names is enough for the mirror renderer
          const names = new Set();
          for (const src of (rec ? rec.shaders : [])) {
            for (const m of src.matchAll(/uniform\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+(\w+)\s*;/g)) names.add(m[1]);
          }
          if (rec) rec.names = [...names];
          lastLinked = args[0];
          return;
        }
        if (prop === 'getActiveUniform') {
          const rec = programs.get(args[0].id);
          const name = rec && rec.names[args[1]] ? rec.names[args[1]] : 'u' + args[1];
          return { name, type: 0, size: 1 };
        }
        if (prop === 'useProgram') { currentProgram = args[0]; }
        if (prop === 'getUniformLocation') {
          // the game asks for locations right after linking, before useProgram
          const owner = programs.get(args[0].id) ? args[0] : lastLinked;
          return { program: owner, name: args[1], owner };
        }
        if (prop === 'getAttribLocation') return 0;
        if (prop === 'createBuffer') { const b = { id: nextId++ }; buffers.set(b.id, { b, data: null, target: null }); return b; }
        if (prop === 'bindBuffer') {
          const targetId = args[0];
          if (targetId === constVal('ARRAY_BUFFER')) {
            boundArrayBuffer = args[1];
          } else if (boundVao) {
            // the element buffer binding is part of the VAO's state
            vaos.get(boundVao.id).ibo = args[1];
            boundElementBuffer = args[1];
          } else {
            boundElementBuffer = args[1];
          }
          return;
        }
        if (prop === 'bufferData') {
          // gl.bufferData(target, data, usage) -> the bound buffer receives it
          const isArray = args[0] === constVal('ARRAY_BUFFER');
          const buf = isArray ? boundArrayBuffer : boundElementBuffer;
          if (buf && buffers.has(buf.id)) {
            buffers.get(buf.id).data = args[1];
            buffers.get(buf.id).isIndex = !isArray;
          }
          return;
        }
        if (prop === 'createVertexArray') { const v = { id: nextId++ }; vaos.set(v.id, { v, attribs: new Map(), ibo: null }); return v; }
        if (prop === 'bindVertexArray') { boundVao = args[0]; return; }
        if (prop === 'vertexAttribPointer') {
          const [loc, size, type, norm, stride, offset] = args;
          if (boundVao) vaos.get(boundVao.id).attribs.set(loc, { buffer: boundArrayBuffer, size, stride, offset, type });
          return;
        }
        if (prop === 'enableVertexAttribArray') {
          if (boundVao && !vaos.get(boundVao.id).attribs.has(args[0])) vaos.get(boundVao.id).attribs.set(args[0], { pending: true });
          return;
        }
        if (prop === 'createTexture') { const tex = { id: nextId++ }; textures.set(tex.id, { data: null, w: 0, h: 0 }); return tex; }
        if (prop === 'bindTexture') { boundTexture = args[1]; return; }
        if (prop === 'texImage2D') return;
        if (prop === 'texSubImage2D') {
          // texSubImage2D(target, level, x, y, w, h, format, type, data)
          const t = textures.get(boundTexture.id);
          t.data = args[8]; t.w = args[4]; t.h = args[5];
          return;
        }
        if (prop === 'viewport') { viewport = args.slice(); return; }
        if (prop === 'createFramebuffer') return { id: nextId++ };
        if (prop === 'drawElements' || prop === 'drawArrays') {
          glCalls.push({
            kind: prop, args: args.slice(),
            program: currentProgram,
            vao: boundVao,
            ibo: (boundVao && vaos.get(boundVao.id).ibo) || boundElementBuffer,
            uniforms: currentProgram ? new Map(uniformState.get(currentProgram.id)) : new Map(),
            seq: glCalls.length,
          });
          return;
        }
        if (/^(uniform|uniformMatrix)/.test(prop)) {
          const [loc, ...rest] = args;
          const owner = (loc && loc.owner) || currentProgram;
          if (!loc || !owner) return;
          const value = prop.startsWith('uniformMatrix')
            ? Float32Array.from(rest[1])
            : (typeof rest[0] === 'number' ? Float32Array.from(rest) : Float32Array.from(rest[0]));
          uniformState.get(owner.id).set(loc.name, { setter: prop, value });
          return;
        }
        return undefined;
      };
      t[prop] = fn;
      return fn;
    },
  });
}

// ---------------------------------------------------------------- scaffolding
global.window = { devicePixelRatio: 1, addEventListener() {}, getGamepads: () => [] };
global.document = { getElementById: () => null };
global.performance = global.performance || { now: () => Date.now() };

const { Renderer } = await import('../game/src/render/renderer.js');
const { World } = await import('../game/src/world/terrain.js');
const { Vehicle } = await import('../game/src/vehicle/vehicle.js');
const { buildCarGeometry } = await import('../game/src/vehicle/carMesh.js');
const { Effects } = await import('../game/src/fx/effects.js');
const { Camera } = await import('../game/src/game/camera.js');
const { parseGlbForNode } = await import('../tests/glb_node.mjs');

const glb = parseGlbForNode(new Uint8Array(fs.readFileSync(new URL('../game/assets/volvo.glb', import.meta.url))));
const canvas = { width: 1280, height: 720, getContext: () => makeGL(canvas), addEventListener() {} };
const renderer = new Renderer(canvas);
renderer.resize(1280, 720);

const world = new World(1337);
const probe = new Vehicle(world, { spawn: world.spawnOnTrack(0) });
const geom = buildCarGeometry(glb, probe.cage.nodes);
renderer.uploadWorld(world);
renderer.uploadCar(geom);

// SCENARIO=crash throws the car into the barrier on the test pad at
// IMPACT=<km/h>; the default scenario just drives down the road.
const scenario = process.env.SCENARIO || (process.env.IDLE === '1' ? 'idle' : 'drive');
const CRASH_SPAWN = { position: [225, world.heightAt(225, 285), 285], yaw: 0 }; // facing the pad barrier at z = 315
const player = new Vehicle(world, {
  spawn: scenario === 'crash' ? CRASH_SPAWN : world.spawnOnTrack(0.015),
  isPlayer: true,
});
renderer.createVehicleInstance(player);
const camera = new Camera();
camera.dist = Number(process.env.CAM_DIST || 7.5);
camera.pitch = Number(process.env.CAM_PITCH || 0.18);
camera.yaw = Number(process.env.CAM_YAW || 0.9);
const effects = new Effects(world);

const INPUT_IDLE = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };
let steps = 1;
if (scenario === 'crash') {
  // settle on the pad, launch at the barrier, then let the wreck come to rest
  for (let i = 0; i < 90; i++) player.step(1 / 60, INPUT_IDLE);
  const v = Number(process.env.IMPACT || 60) / 3.6;
  for (let i = 0; i < player.nodeCount; i++) {
    player.body.vel[i * 3] = 0;
    player.body.vel[i * 3 + 1] = 0;
    player.body.vel[i * 3 + 2] = v;
  }
  const crashSeconds = Number(process.env.CRASH_T || 2.6);
  steps = Math.round(crashSeconds * 60);
  for (let i = 0; i < steps; i++) {
    player.step(1 / 60, INPUT_IDLE);
    for (const d of player.takeDetachedWheels()) { void d; }
    effects.carEffects(player, 1 / 60, player.speed);
    effects.update(1 / 60);
  }
  camera.dist = Number(process.env.CAM_DIST || 6.5);
} else {
  const seconds = Number(process.env.DRIVE || 2.5);
  const idle = scenario === 'idle';         // no throttle: reproduce a "parked" frame
  steps = Math.round(seconds * 60);
  for (let i = 0; i < steps; i++) {
    const throttle = idle ? 0 : 1;
    player.step(1 / 60, { throttle, brake: 0, steer: i > steps * 0.7 ? 0.5 : 0, handbrake: 0 });
    for (const d of player.takeDetachedWheels()) { void d; }
    effects.carEffects(player, 1 / 60, player.speed);
    effects.update(1 / 60);
  }
}
if (process.env.DETACH === '1') { for (const w of player.wheels) player.detachWheel(w); }
camera.snap(player, world);
for (let i = 0; i < 30; i++) {
  camera.update(1 / 60, player, { mouse: { dx: 0, dy: 0, wheel: 0, dragging: false } }, world);
}

glCalls.length = 0;
renderer.render({ camera, world, vehicles: [player], fx: effects, debugCage: false, cageLines: null });
if (process.env.DEBUG) {
  console.log('raw draw calls:', glCalls.length, 'vaos:', vaos.size);
  let shown = 0;
  for (const v of vaos.values()) {
    if (shown++ > 2) break;
    console.log('  vao', v.v.id, [...v.attribs.entries()].map(([l, a]) => l + ':' + (a.buffer ? ('buf' + a.buffer.id + ':' + (buffers.get(a.buffer.id)?.data ? buffers.get(a.buffer.id).data.length : 'nodata')) : 'none')).join(' '));
  }
}

// ---------------------------------------------------------------- dump
const written = new Map();
function dumpBuffer(bufObj, tag) {
  const rec = bufObj && buffers.get(bufObj.id);
  // a plain number means gl.bufferData(target, size, usage): an empty allocation
  if (!rec || !rec.data || !rec.data.buffer) return null;
  // keyed by buffer identity: two different buffers may coincidentally have the
  // same length, and collapsing them silently corrupts the replay
  if (written.has(rec)) return written.get(rec);
  const arr = rec.data;
  const isIndex = arr instanceof Uint32Array || arr instanceof Uint16Array;
  const file = `buf_${written.size}_${isIndex ? 'index' : 'vertex'}.bin`;
  fs.writeFileSync(path.join(outDir, file), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
  const meta = { file, count: arr.length, kind: isIndex ? (arr instanceof Uint16Array ? 'u16' : 'u32') : 'f32' };
  written.set(rec, meta);
  return meta;
}

const draws = [];
for (const call of glCalls) {
  if (!call.vao) { if (process.env.DEBUG) console.log('  draw without vao', call.kind); continue; }
  const vao = vaos.get(call.vao.id);
  const attribs = {};
  let vertexBuffer = null;
  for (const [loc, a] of vao.attribs) {
    if (!a.buffer) continue;
    vertexBuffer = a.buffer;
    attribs[loc] = { size: a.size, stride: a.stride, offset: a.offset };
  }
  if (!vertexBuffer) continue;
  const uniforms = {};
  for (const [name, u] of call.uniforms) {
    if (name === 'uNodeTex') continue;
    uniforms[name] = { setter: u.setter, value: Array.from(u.value) };
  }
  // the node texture is whatever was last uploaded
  let nodeTex = null;
  for (const t of textures.values()) {
    if (t.data && t.data.length === 320 * 5 * 4) nodeTex = t.data;
  }
  const vb = dumpBuffer(vertexBuffer, 'v');
  const ib = call.kind === 'drawElements' ? dumpBuffer(call.ibo, 'i') : null;
  if (!vb) { if (process.env.DEBUG) console.log('  draw without vertex buffer', call.kind); continue; }
  draws.push({
    kind: call.kind,
    count: call.args[1],
    offset: call.kind === 'drawElements' ? call.args[3] : call.args[2],
    mode: call.args[0],
    vertex: vb,
    index: ib,
    attribs,
    uniforms,
    nodeTex: nodeTex ? (() => {
      const file = 'node_tex.bin';
      fs.writeFileSync(path.join(outDir, file), Buffer.from(nodeTex.buffer, nodeTex.byteOffset, nodeTex.byteLength));
      return { file, w: 320, h: 5 };
    })() : null,
  });
}

fs.writeFileSync(path.join(outDir, 'frame.json'), JSON.stringify({
  viewport,
  draws,
  car: {
    speedKmh: player.forwardSpeed * 3.6,
    damage: player.damage,
    detached: player.wheels.map((w) => w.detached),
    panels: (player.panels || []).map((p) => `${p.name}:${p.detached ? 'OFF' : p.angle.toFixed(2)}`).join(' '),
  },
}, null, 1));
console.log(`captured ${draws.length} draw calls -> ${outDir}`);
console.log('car', (player.forwardSpeed * 3.6).toFixed(1), 'km/h, damage', player.damage.toFixed(3));
