// Entry point: boots the world, builds cars, runs the loop.
import { World, PAD } from './world/terrain.js';
import { Vehicle } from './vehicle/vehicle.js';
import { buildCarGeometry } from './vehicle/carMesh.js';
import { Effects } from './fx/effects.js';
import { Renderer } from './render/renderer.js';
import { Camera } from './game/camera.js';
import { Input } from './game/input.js';
import { loadGlb } from './render/loader.js';
import { HUD } from './game/hud.js';
import { AIDriver } from './game/ai.js';

const state = {
  paused: false,
  timeScale: 1,
  slowmo: false,
  debugCage: false,
  showHelp: true,
  cameraMode: 0,   // 0 chase | 1 orbit | 2 hood
};

const CAMERA_MODES = ['chase', 'orbit', 'hood'];

async function main() {
  const canvas = document.getElementById('gl');
  const hud = new HUD();
  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');

  let renderer;
  try {
    renderer = new Renderer(canvas);
  } catch (err) {
    loadingText.innerHTML = `<b>WebGL2 unavailable</b><br>${err.message}`;
    return;
  }

  loadingText.textContent = 'loading car model…';
  const glb = await loadGlb('assets/tesla.glb');

  loadingText.textContent = 'building terrain…';
  await new Promise((r) => setTimeout(r, 16));
  const world = new World(1337);

  loadingText.textContent = 'building physics cage…';
  const probe = new Vehicle(world, { spawn: world.spawnOnTrack(0.0) });
  const geom = buildCarGeometry(glb, probe.cage.nodes);

  loadingText.textContent = 'uploading geometry…';
  await new Promise((r) => setTimeout(r, 16));
  renderer.uploadWorld(world);
  renderer.uploadCar(geom);

  const effects = new Effects(world);
  const camera = new Camera();
  const input = new Input(canvas);

  const cars = [];
  const startT = 0.015;
  const player = new Vehicle(world, {
    spawn: world.spawnOnTrack(startT),
    isPlayer: true,
    name: 'You',
    spec: {},
  });
  cars.push(player);
  renderer.createVehicleInstance(player);

  // two AI cars starting behind the player
  const aiDefs = [
    { t: startT - 0.012, name: 'Kurt', color: [0.10, 0.32, 0.62], skill: 0.90 },

  ];
  const ais = [];
  for (const def of aiDefs) {
    const v = new Vehicle(world, {
      spawn: world.spawnOnTrack(def.t < 0 ? def.t + 1 : def.t),
      name: def.name,
    });
    v.color = def.color;
    cars.push(v);
    renderer.createVehicleInstance(v);
    ais.push(new AIDriver(v, world, def.skill));
  }
  probe.reset();

  camera.snap(player, world);

  const clock = { last: performance.now(), acc: 0, frames: 0, fpsTime: 0, fps: 60 };
  let cageLines = null;
  const cageVerts = new Float32Array(20000 * 4);

  function resetCar(v = player, usePad = false) {
    if (usePad) {
      // respawn on the pad
      const yaw = 0;
      const spawn = { position: [PAD.x, world.heightAt(PAD.x, PAD.z), PAD.z + 20], yaw };
      const nv = new Vehicle(world, { spawn, isPlayer: v.isPlayer, name: v.name });
      // swap state in place so the renderer instance stays valid
      Object.assign(v, nv);
    } else {
      const spawn = world.spawnOnTrack(nearestT(world, v.center));
      const nv = new Vehicle(world, { spawn, isPlayer: v.isPlayer, name: v.name });
      Object.assign(v, nv);
    }
    effects.clearFor(v.name);
    camera.snap(v, world);
  }

  function nearestT(w, pos) {
    let best = 0, bestD = Infinity;
    const pts = w.road.pts;
    const step = Math.max(1, Math.floor(pts.length / 400));
    for (let i = 0; i < pts.length; i += step) {
      const d = (pts[i][0] - pos[0]) ** 2 + (pts[i][1] - pos[2]) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best / pts.length;
  }

  function updateCageLines(v) {
    const b = v.body;
    let n = 0;
    for (let i = 0; i < b.nb && n < 19990; i++) {
      if (b.bbroken[i]) continue;
      const a = b.ba[i] * 3, c = b.bb[i] * 3;
      // x = damage of the beam, renderer tints stressed beams hot
      const dmg = b.bdmg[i];
      for (const o of [a, c]) {
        cageVerts[n * 4] = b.pos[o]; cageVerts[n * 4 + 1] = b.pos[o + 1]; cageVerts[n * 4 + 2] = b.pos[o + 2];
        cageVerts[n * 4 + 3] = dmg;
        n++;
      }
    }
    cageLines = { data: cageVerts, count: n };
  }

  function frame(now) {
    const dtRaw = Math.min((now - clock.last) / 1000, 0.1);
    clock.last = now;
    clock.fpsTime += dtRaw;
    clock.frames++;
    if (clock.fpsTime > 0.4) {
      clock.fps = clock.frames / clock.fpsTime;
      clock.frames = 0; clock.fpsTime = 0;
    }

    const inp = input.update(dtRaw);

    // global toggles
    if (input.consume('KeyR')) {
      for (const v of cars) resetCar(v, false);
      effects.reset();
    }
    if (input.consume('KeyP')) resetCar(player, true);
    if (input.consume('KeyC')) {
      state.cameraMode = (state.cameraMode + 1) % CAMERA_MODES.length;
      camera.mode = CAMERA_MODES[state.cameraMode];
      camera.yaw = 0;
      camera.dist = camera.mode === 'hood' ? 8.2 : 8.2;
      camera.autoOrbit = 0;
      hud.toast(`camera: ${camera.mode}`);
    }
    if (input.consume('KeyG')) state.debugCage = !state.debugCage;
    if (input.consume('KeyH')) hud.toggleHelp();
    if (input.consume('KeyT')) state.slowmo = !state.slowmo;
    if (input.consume('KeyF')) { state.paused = !state.paused; }

    const timeScale = state.paused ? 0 : (state.slowmo ? 0.3 : 1);
    const dt = dtRaw * timeScale;

    // physics
    if (dt > 0) {
      player.step(dt, inp);
      for (const ai of ais) ai.update(dt);
      for (const v of cars) {
        if (!v.isPlayer) v.step(dt, v.lastInput || { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
        const detached = v.takeDetachedWheels();
        for (const d of detached) {
          for (let i = 0; i < 12; i++) {
            effects.spawnParticle(d.pos[0], d.pos[1], d.pos[2],
              (Math.random() - 0.5) * 3, Math.random() * 2, (Math.random() - 0.5) * 3,
              0.5 + Math.random(), 0.4, [0.5, 0.5, 0.5], -3, 1);
          }
        }
      }
      effects.carEffects(player, dt, player.speed);
      for (const v of cars) if (!v.isPlayer) effects.carEffects(v, dt, v.speed);
      effects.update(dt);
      if (state.debugCage) updateCageLines(player);
    }

    // camera shake on impacts
    if (player.impactFlash > 0.7) camera.shake = Math.min(1.2, camera.shake + 0.45);
    camera.update(Math.max(dtRaw, 0.0001), player, input, world);

    renderer.setEnv({});
    renderer.render({
      camera, world, vehicles: cars, fx: effects,
      debugCage: state.debugCage, cageLines,
    });

    hud.update({
      speed: Math.abs(player.forwardSpeed) * 3.6,
      power: player.throttle,
      brake: player.brake,
      damage: player.damage,
      wheelDamage: player.wheelDamage,
      gears: 1,
      fps: clock.fps,
      airborne: player.airborne,
      gForce: player.gForce,
      slowmo: state.slowmo,
      paused: state.paused,
      cars: cars.length,
      mode: camera.mode,
      wheelStates: player.wheels.map((w) => w.detached),
      detached: player.wheels.filter((w) => w.detached).length,
      cage: state.debugCage,
    });

    input.endFrame();
    requestAnimationFrame(frame);
  }

  loading.classList.add('hidden');
  requestAnimationFrame(frame);
  window.__game = { world, cars, player, renderer, state, effects, camera };
}

main().catch((err) => {
  console.error(err);
  const t = document.getElementById('loading-text');
  if (t) t.innerHTML = `<b>error</b><br><pre>${String(err.message || err)}</pre>`;
});
