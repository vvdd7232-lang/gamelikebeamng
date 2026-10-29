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

// ---- crash into a wall at speed
const car2 = makeCar();
const wall = world.colliders[0];
run(car2, 1.5, { throttle: 0, brake: 0, steer: 0, handbrake: 0 });
// teleport-ish: drive towards the wall using a straight line by pointing the body
car2.body.pos.forEach((_, i) => {});
const tImpact = Date.now();
// launch the car at the wall by setting velocities
const dir = [wall.center[0] - car2.center[0], 0, wall.center[2] - car2.center[2]];
const dl = Math.hypot(dir[0], dir[2]);
for (let i = 0; i < car2.nodeCount; i++) {
  car2.body.vel[i * 3] = dir[0] / dl * 24;
  car2.body.vel[i * 3 + 2] = dir[2] / dl * 24;
}
let maxDmg = 0;
run(car2, 2.0, { throttle: 0.5, brake: 0, steer: 0, handbrake: 0 }, 1 / 60, () => {
  maxDmg = Math.max(maxDmg, car2.damage);
});
console.log('crash: damage', car2.damage.toFixed(3), 'broken beams', car2.body.brokenCount,
  'of', car2.body.nb, 'compute ms', Date.now() - tImpact);
console.log('  wheels detached?', car2.takeDetachedWheels().length);
console.log('  car still has mass', car2.body.totalMass().toFixed(0));
console.log('  up axis', car2.basis.y.map(v => v.toFixed(2)).join(','));

// ---- performance
const perf = makeCar();
const tp = Date.now();
run(perf, 5, { throttle: 0.8, brake: 0, steer: 0.3, handbrake: 0 });
const ms = Date.now() - tp;
console.log('perf: 5s of driving took', ms, 'ms =>', (5000 / ms).toFixed(2) + 'x realtime');
