// DOM-free check of the HUD: every element id it looks up must exist in
// game/index.html, and update() must survive the whole damage range (a
// division-by-zero or a missing element there is invisible until the car is
// actually wrecked in a browser).
import fs from 'fs';
import path from 'path';

const ROOT = path.resolve(new URL('..', import.meta.url).pathname);
const html = fs.readFileSync(path.join(ROOT, 'game/index.html'), 'utf8');

// --- the ids index.html provides
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));

// --- a tiny element stub: classList / style / textContent / setAttribute
class El {
  constructor(id) {
    this.id = id;
    this.textContent = '';
    this._classes = new Set();
    this.style = {};
    this.attrs = {};
    this.classList = {
      add: (c) => this._classes.add(c),
      remove: (c) => this._classes.delete(c),
      toggle: (c, on) => { if (on === undefined ? !this._classes.has(c) : on) this._classes.add(c); else this._classes.delete(c); },
      contains: (c) => this._classes.has(c),
    };
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
}

const els = new Map();
const missing = [];
global.document = {
  getElementById(id) {
    if (!htmlIds.has(id)) { missing.push(id); return null; }
    if (!els.has(id)) els.set(id, new El(id));
    return els.get(id);
  },
};
global.window = global.window || { addEventListener() {} };

const { HUD } = await import('../game/src/game/hud.js');

let failures = 0;
const fail = (m) => { failures++; console.log('   FAIL ' + m); };

const hud = new HUD();
if (missing.length) fail('elements referenced but not in index.html: ' + [...new Set(missing)].join(', '));
else console.log('   ok   every HUD element id resolves against index.html');

const pristine = {
  dt: 1 / 60,
  speed: 62,
  power: 0.4,
  brake: 0,
  damage: 0,
  zones: [0, 0, 0, 0, 0, 0],
  cabin: 1,
  powerScale: 1,
  wheelDamage: 0,
  wheelRim: [0, 0, 0, 0],
  impact: 0,
  gears: 1,
  fps: 60,
  airborne: false,
  gForce: [0.1, 1, 0.2],
  slowmo: false,
  paused: false,
  cars: 2,
  mode: 'chase',
  wheelStates: [false, false, false, false],
  detached: 0,
  cage: false,
};

// full range of states, including the "wrecked" edge cases
hud.update(pristine);
const dmgText = els.get('damage-val').textContent;
if (dmgText !== '0%') fail(`pristine damage shows "${dmgText}" instead of 0%`);

hud.update({
  ...pristine,
  damage: 0.98,
  zones: [0.95, 0.8, 0.3, 0.3, 0.1, 0.05],
  cabin: 0.42,
  powerScale: 0.08,
  wheelRim: [1, 1, 0.6, 0.1],
  wheelStates: [true, true, false, false],
  detached: 2,
  impact: 103,
});
const wrecked = els.get('damage-val').textContent;
if (wrecked !== '98%') fail(`wrecked damage shows "${wrecked}" instead of 98%`);
if (els.get('wheels-val').textContent !== '2/4') fail('wheel counter wrong: ' + els.get('wheels-val').textContent);
if (!String(els.get('wheel-damage').textContent).includes('2 WHEELS LOST')) fail('wheel-loss line missing: ' + els.get('wheel-damage').textContent);
if (els.get('engine-val').textContent !== '8%') fail('power readout wrong: ' + els.get('engine-val').textContent);
for (const z of [0, 1, 2, 3, 4, 5]) {
  const el = els.get('zn-' + z);
  if (!el.style.fill || !el.style.fill.startsWith('rgb')) fail(`zone ${z} has no colour`);
}
if (String(els.get('impact-kmh').textContent) !== '103') fail('impact banner did not update: ' + els.get('impact-kmh').textContent);

// NaN / missing optional fields must not throw or print "NaN"
hud.update({ ...pristine, damage: NaN, zones: undefined, wheelRim: undefined, cabin: undefined });
if (/NaN/.test(els.get('damage-val').textContent)) fail('NaN leaked into the damage readout');

const speedText = els.get('speed').textContent;
if (speedText !== '62') fail('speed readout: ' + speedText);

if (failures) {
  console.log('HUD TEST FAILED');
  process.exit(1);
}
console.log('HUD OK');
