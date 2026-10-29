// Cross-check between the GLSL sources and the JS that drives them.
//
// 1. varying types must match between the vertex and fragment stage of a
//    program (this is what produced
//    "Varying vMat is not linkable between attached shaders" in a browser);
// 2. every uniform the renderer sets with gl.uniform* must be declared with a
//    compatible type in at least one shader -- a wrong setter is an
//    INVALID_OPERATION at runtime and the uniform silently stays 0;
// 3. (informational) uniforms that nothing ever sets.
import fs from 'fs';
import path from 'path';
import * as S from '../game/src/render/shaders.js';

const ROOT = path.resolve(new URL('..', import.meta.url).pathname);

const PROGRAMS = [
  ['sky', S.SKY_VS, S.SKY_FS],
  ['mesh', S.MESH_VS, S.MESH_FS],
  ['depth', S.DEPTH_VS, S.DEPTH_FS],
  ['skid', S.SKID_VS, S.SKID_FS],
  ['particle', S.PARTICLE_VS, S.PARTICLE_FS],
];

function strip(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/^\s*#version[^\n]*/m, ' ');
}

/** `in|out <type> <name>;` interface declarations */
function interfaces(src) {
  const out = new Map();
  const re = /^\s*(?:layout\s*\([^)]*\)\s*)?(in|out)\s+((?:lowp|mediump|highp)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*;/gm;
  for (const m of strip(src).matchAll(re)) {
    const name = m[4];
    if (!out.has(name)) out.set(name, { kind: m[1], type: m[3] + (m[5] || '') });
  }
  return out;
}

/** `<type> <name>;` uniform declarations, with array suffixes kept */
function uniforms(src) {
  const out = new Map();
  const re = /^\s*uniform\s+((?:lowp|mediump|highp)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*;/gm;
  for (const m of strip(src).matchAll(re)) {
    const name = m[3];
    if (!out.has(name)) out.set(name, m[2] + (m[4] || ''));
  }
  return out;
}

let failures = 0;
const fail = (msg) => { failures++; console.log('   FAIL ' + msg); };

// ---------------------------------------------------------------- 1. varyings
console.log('varyings:');
for (const [label, vsSrc, fsSrc] of PROGRAMS) {
  const vs = interfaces(vsSrc);
  const fs = interfaces(fsSrc);
  let ok = true;
  for (const [name, decl] of fs) {
    if (decl.kind !== 'in') continue;
    const v = vs.get(name);
    if (!v || v.kind !== 'out') {
      ok = false;
      fail(`${label}: fragment input '${name}' has no matching vertex output`);
      continue;
    }
    if (v.type !== decl.type) {
      ok = false;
      fail(`${label}: '${name}' vertex out ${v.type} vs fragment in ${decl.type}`);
    }
  }
  if (ok) {
    const list = [...fs].filter(([, d]) => d.kind === 'in')
      .map(([n, d]) => `${n}:${d.type}`).join(', ') || 'none';
    console.log(`   ok   ${label.padEnd(9)} ${list}`);
  }
}

// ---------------------------------------------------------------- 2. uniforms
// collect declarations from every shader (a name may be declared in several)
const declared = new Map();
for (const [, vs, fs] of PROGRAMS) {
  for (const src of [vs, fs]) {
    for (const [name, type] of uniforms(src)) {
      if (!declared.has(name)) declared.set(name, new Set());
      declared.get(name).add(type);
    }
  }
}

// setter -> declared type it is allowed to touch
const SETTERS = {
  uniform1f: ['float'],
  uniform1i: ['int', 'bool', 'sampler2D', 'sampler2DShadow', 'samplerCube'],
  uniform2f: ['vec2'],
  uniform3f: ['vec3'],
  uniform4f: ['vec4'],
  uniform1fv: ['float'],
  uniform2fv: ['vec2'],
  uniform3fv: ['vec3'],
  uniform4fv: ['vec4'],
  uniformMatrix3fv: ['mat3'],
  uniformMatrix4fv: ['mat4'],
};

const jsFiles = [];
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.js') && !p.includes('/render/shaders.js')) jsFiles.push(p);
  }
})(path.join(ROOT, 'game/src'));

// gl.uniform3fv(u.uName, ...) where `u` is an alias of `something.uniforms`
const callRe = /gl\.(uniform[\w]+)\(\s*([^,]+?)\s*[,)]/g;
const aliasRe = /(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*[^;]*?\.uniforms\b/g;
const setNames = new Set();
let checked = 0;
for (const file of jsFiles) {
  const src = fs.readFileSync(file, 'utf8');
  const aliases = new Set([...src.matchAll(aliasRe)].map((m) => m[1]));
  const referenced = [];
  for (const m of src.matchAll(callRe)) {
    const setter = m[1];
    const arg = m[2].trim();
    let name = null;
    let mm = arg.match(/\.uniforms\.([A-Za-z_]\w*)$/);
    if (mm) name = mm[1];
    else {
      mm = arg.match(/^([A-Za-z_]\w*)\.([A-Za-z_]\w*)$/);
      if (mm && aliases.has(mm[1])) name = mm[2];
    }
    if (!name) {
      const bare = arg.match(/^\[\s*([A-Za-z_]\w*)/);
      if (bare) referenced.push(bare[1]);
      continue;
    }
    setNames.add(name);
    const expect = SETTERS[setter];
    if (!expect) continue;
    const types = declared.get(name);
    if (!types) {
      fail(`${path.relative(ROOT, file)}: ${setter}(${name}) but no shader declares it`);
      continue;
    }
    checked++;
    const ok = [...types].some((t) => expect.includes(t.replace(/\[\d+\]$/, '')));
    if (!ok) {
      fail(`${path.relative(ROOT, file)}: ${setter}(${name}) but shader declares ${[...types].join('/')}`);
    }
  }
}
console.log(`uniform setters checked: ${checked}`);

// informational: declared but never set by name
const unset = [...declared.keys()].filter((n) => !setNames.has(n));
if (unset.length) console.log('   note: not set by name (may be set via a variable):', unset.join(', '));

console.log(failures ? `${failures} problem(s)` : 'shader/JS contract OK');
process.exit(failures ? 1 : 0);
