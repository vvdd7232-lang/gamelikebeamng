// Headless physics sanity tests (no DOM / WebGL needed).
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';

const t0 = Date.now();
const world = new World(1337);
console.log('world baked in', Date.now() - t0, 'ms');

function makeCar(spawn) {
  return new Vehicle(world, { spawn: spawn || world.spawnOnTrack(0.02), isPlayer: true });
}

function run(car, seconds, input, dt = 1 / 60, cb) {
  const steps = Math.round(seconds / dt);
  for (let i = 0; i < steps; i++) {
    car.step(dt, typeof input === 'function' ? input(i * dt) : input);
    if (cb) cb(i * dt);
  }
}

const car = makeCar();
console.log('nodes', car.nodeCount, 'beams', car.body.nb,
  'mass', car.body.totalMass().toFixed(1));
console.log('start center', car.center.map(v => v.toFixed(3)));
console.log('start wheel comps', car.wheels.map(w => w.compression.toFixed(3)));

// ---- settle
const tSettle = Date.now();
run(car, 3, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
console.log('settle 3s took', Date.now() - tSettle, 'ms');
console.log('after settle: center', car.center.map(v => v.toFixed(3)),
  'speed', car.speed.toFixed(3));
console.log('  wheels: comp', car.wheels.map(w => w.compression.toFixed(3)),
  'load', car.wheels.map(w => w.load.toFixed(0)));
console.log('  damage', car.damage.toFixed(4), 'up', car.basis.y.map(v => v.toFixed(3)));

// ---- acceleration 0-100 km/h
let t100 = null;
const v0 = car.speed;
run(car, 8, { throttle: 1, brake: 0, steer: 0, handbrake: 0 }, 1 / 60, (t) => {
  if (t100 === null && car.forwardSpeed * 3.6 >= 100) t100 = t;
});
console.log('0-100 km/h:', t100 ? t100.toFixed(2) + 's' : 'not reached',
  'speed now', (car.forwardSpeed * 3.6).toFixed(1), 'km/h');
console.log('  damage after accel', car.damage.toFixed(4),
  'up', car.basis.y.map(v => v.toFixed(2)).join(','));

// ---- braking
const vBefore = car.forwardSpeed;
run(car, 2.0, { throttle: 0, brake: 1, steer: 0, handbrake: 0 });
console.log('brake 2s:', (vBefore * 3.6).toFixed(1), '->', (car.forwardSpeed * 3.6).toFixed(1), 'km/h');

// ---- cornering
run(car, 2.5, { throttle: 0.35, brake: 0, steer: 1, handbrake: 0 });
console.log('corner: speed', car.speed.toFixed(1), 'yawRate', car.yawRate.toFixed(2),
  'roll(up)', car.basis.y.map(v => v.toFixed(2)).join(','));

// ---- crash into the concrete barrier on the test pad at 100 km/h
// (the pad has a 30 m run-up in front of the wall, so the cage is launched the
// way the in-game crash test key does it)
const car2 = new Vehicle(world, {
  spawn: { position: [225, world.heightAt(225, 285), 285], yaw: 0 },
});
run(car2, 1.5, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
const vCrash = 100 / 3.6;
for (let i = 0; i < car2.nodeCount; i++) {
  car2.body.vel[i * 3] = 0;
  car2.body.vel[i * 3 + 1] = 0;
  car2.body.vel[i * 3 + 2] = vCrash;   // +z is straight at the barrier
}
const tImpact = Date.now();
let maxDmg = 0;
run(car2, 2.5, { throttle: 0.5, brake: 0, steer: 0, handbrake: 0 }, 1 / 60, () => {
  maxDmg = Math.max(maxDmg, car2.damage);
});
const bodyLen = () => {
  const R = car2.R, rc = car2.restCentroid, c = car2.center;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < car2.nodeCount; i++) {
    const px = car2.body.pos[i * 3] - c[0];
    const py = car2.body.pos[i * 3 + 1] - c[1];
    const pz = car2.body.pos[i * 3 + 2] - c[2];
    const lz = R[2] * px + R[5] * py + R[8] * pz + rc[2];
    mn = Math.min(mn, lz); mx = Math.max(mx, lz);
  }
  return mx - mn;
};
console.log('crash 100 km/h: damage', car2.damage.toFixed(3), 'peak', maxDmg.toFixed(3),
  'broken beams', car2.body.brokenCount, 'of', car2.body.nb, 'compute ms', Date.now() - tImpact);
console.log('  body length', bodyLen().toFixed(2), 'm (was 4.72)',
  'front panel damage', car2.damageZones[0].toFixed(2), '/', car2.damageZones[1].toFixed(2));
console.log('  wheels detached?', car2.takeDetachedWheels().length,
  'cabin', (car2.cabinIntegrity * 100).toFixed(0) + '%',
  'power', (car2.powerScale * 100).toFixed(0) + '%');
console.log('  car still has mass', car2.body.totalMass().toFixed(0));
console.log('  up axis', car2.basis.y.map(v => v.toFixed(2)).join(','));

// ---- performance
const perf = makeCar();
const tp = Date.now();
run(perf, 5, { throttle: 0.8, brake: 0, steer: 0.3, handbrake: 0 });
const ms = Date.now() - tp;
console.log('perf: 5s of driving took', ms, 'ms =>', (5000 / ms).toFixed(2) + 'x realtime');
