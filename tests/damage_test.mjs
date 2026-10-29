// Damage model regression test: the things that used to be wrong.
//
//  * parking, driving, braking and cornering must not damage anything;
//  * a real hit must fold the *crumple zones* while the safety cell survives;
//  * the car must not come apart into a cloud of loose vertices at speed;
//  * wheels only come off in a hard hit, and when they do they keep existing;
//  * a smashed nose must cut power and bend the steering, not just paint a
//    number on the HUD.
import { World } from '../game/src/world/terrain.js';
import { Vehicle } from '../game/src/vehicle/vehicle.js';

const world = new World(1337);
const IDLE = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };
const SPAWN = { position: [225, world.heightAt(225, 285), 285], yaw: 0 };  // run-up on the test pad

let failures = 0;
const check = (ok, msg) => { if (!ok) { failures++; console.log('   FAIL ' + msg); } else console.log('   ok   ' + msg); };

function run(car, seconds, input, dt = 1 / 60) {
  const steps = Math.round(seconds / dt);
  for (let i = 0; i < steps; i++) car.step(dt, input);
}

/** crash a car into the pad barrier at `kmh`; returns the car */
function crash(kmh, settle = 2.5) {
  const car = new Vehicle(world, { spawn: SPAWN });
  run(car, 1.5, IDLE);
  const v = kmh / 3.6;
  for (let i = 0; i < car.nodeCount; i++) {
    car.body.vel[i * 3] = 0;
    car.body.vel[i * 3 + 1] = 0;
    car.body.vel[i * 3 + 2] = v;
  }
  run(car, settle, IDLE);
  return car;
}

function bodyFrameExtent(car) {
  const R = car.R, rc = car.restCentroid, c = car.center;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < car.nodeCount; i++) {
    const px = car.body.pos[i * 3] - c[0];
    const py = car.body.pos[i * 3 + 1] - c[1];
    const pz = car.body.pos[i * 3 + 2] - c[2];
    const lz = R[2] * px + R[5] * py + R[8] * pz + rc[2];
    mn = Math.min(mn, lz); mx = Math.max(mx, lz);
  }
  return mx - mn;
}

// ---------------------------------------------------------------- benign use
console.log('normal driving:');
{
  const car = new Vehicle(world, { spawn: world.spawnOnTrack(0.02), isPlayer: true });
  run(car, 3, IDLE);
  const startZones = Array.from(car.damageZones);
  run(car, 1.5, { throttle: 1, brake: 0, steer: 0, handbrake: 0 });
  run(car, 4, { throttle: 1, brake: 0, steer: 0, handbrake: 0 });
  const topSpeed = Math.abs(car.forwardSpeed) * 3.6;
  const roadDamage = car.damage;
  run(car, 2, { throttle: 0, brake: 1, steer: 0, handbrake: 0 });
  run(car, 2, { throttle: 0.7, brake: 0, steer: 1, handbrake: 0 });
  run(car, 2, { throttle: 1, brake: 0, steer: -1, handbrake: 1 });
  // a few big kerb hits at speed
  for (let i = 0; i < 240; i++) {
    car.step(1 / 60, { throttle: 1, brake: 0, steer: Math.sin(i * 0.05), handbrake: 0 });
  }
  check(topSpeed > 110, `reaches ${topSpeed.toFixed(0)} km/h`);
  check(roadDamage < 0.01, `driving flat out on the road leaves the car intact (damage ${(roadDamage * 100).toFixed(2)}%)`);
  check(car.damage < 0.05, `braking, cornering and kerb abuse stays cosmetic (damage ${(car.damage * 100).toFixed(1)}%)`);
  check(car.wheels.every((w) => !w.detached), 'no wheel falls off while driving');
  check(startZones.every((v) => v < 0.01), 'panels start undamaged');
  check(car.powerScale > 0.95, 'power stays at 100%');
}

// ---------------------------------------------------------------- light tap
console.log('light tap (15 km/h):');
{
  const car = crash(15);
  check(car.damage < 0.05, `barely marks the paint (damage ${(car.damage * 100).toFixed(1)}%)`);
  check(car.wheels.every((w) => !w.detached), 'no wheels lost');
  check(car.cabinIntegrity > 0.99, 'safety cell untouched');
}

// ---------------------------------------------------------------- mild crash
console.log('50 km/h into a concrete barrier:');
{
  const car = crash(50);
  check(car.shellDamage > 0.05, `front end yields (shell ${(car.shellDamage * 100).toFixed(0)}%)`);
  check(car.damageZones[0] > car.damageZones[4] * 2, 'damage is concentrated in the front panels');
  check(car.cabinIntegrity > 0.9, `safety cell keeps its shape (${(car.cabinIntegrity * 100).toFixed(0)}%)`);
  check(bodyFrameExtent(car) > 4.0, `body is still a car, not a pancake (${bodyFrameExtent(car).toFixed(2)} m)`);
}

// ---------------------------------------------------------------- hard crash
console.log('100 km/h into a concrete barrier:');
{
  const car = crash(100);
  const len = bodyFrameExtent(car);
  check(car.damage > 0.35, `reads as a wreck (severity ${(car.damage * 100).toFixed(0)}%)`);
  check(len < 4.2 && len > 3.0, `crumple zone absorbs the hit (${len.toFixed(2)} m, was 4.72 m)`);
  check(car.body.brokenCount > 5 && car.body.brokenCount < car.body.nb * 0.2,
    `some beams tear but the shell holds (${car.body.brokenCount}/${car.body.nb})`);
  check(car.cabinIntegrity > 0.5, `cabin survives (${(car.cabinIntegrity * 100).toFixed(0)}%)`);
  check(car.powerScale < 0.95, `crushed nose cuts power (${(car.powerScale * 100).toFixed(0)}%)`);
  const detached = car.wheels.filter((w) => w.detached);
  check(detached.length > 0, `${detached.length} wheel(s) torn off`);
  if (detached.length) {
    const w = detached[0];
    check(Array.isArray(w.vel) && Array.isArray(w.quat), 'a torn-off wheel keeps its own body state');
    const y0 = w.pos[1];
    run(car, 1.0, IDLE);
    check(Number.isFinite(w.pos[1]) && Math.abs(w.pos[1] - y0) < 5, 'the loose wheel keeps flying / settling, not teleporting');
  }
  // damage is permanent: it must not heal as the car sits still
  const before = car.damage;
  run(car, 3, IDLE);
  check(car.damage >= before - 0.01, 'damage is permanent');
}

// ---------------------------------------------------------------- repair
console.log('repair (R):');
{
  const car = crash(100);
  car.reset();
  run(car, 2, IDLE);
  check(car.damage < 0.01, 'reset clears the damage');
  check(car.wheels.every((w) => !w.detached), 'reset puts the wheels back');
  check(car.body.brokenCount === 0, 'reset welds every beam');
  check(car.powerScale > 0.99 && car.cabinIntegrity > 0.99, 'reset clears the derived damage');
}

// ---------------------------------------------------------------- soft body
console.log('no self-destruction:');
{
  const car = new Vehicle(world, { spawn: SPAWN });
  run(car, 1.5, IDLE);
  // drop it from 12 m onto the concrete
  for (let i = 0; i < car.nodeCount; i++) car.body.pos[i * 3 + 1] += 12;
  run(car, 3, IDLE);
  const len = bodyFrameExtent(car);
  check(len > 3.4, `a 12 m drop leaves a recognisable car (${len.toFixed(2)} m)`);
  check(Number.isFinite(car.center[1]) && car.center[1] > 0, 'car ends up on the ground, not at infinity');
}

if (failures) {
  console.log(`DAMAGE TEST FAILED (${failures})`);
  process.exit(1);
}
console.log('DAMAGE OK');
