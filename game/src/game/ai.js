// Very small waypoint follower used by the traffic cars.
export class AIDriver {
  constructor(vehicle, world, skill = 0.9) {
    this.v = vehicle;
    this.world = world;
    this.skill = skill;
    this.index = this.findStart();
    this.stuck = 0;
    this.input = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };
    this.steerSmooth = 0;
  }

  findStart() {
    const pts = this.world.road.pts;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const d = (pts[i][0] - this.v.center[0]) ** 2 + (pts[i][1] - this.v.center[2]) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  update(dt) {
    const v = this.v;
    const pts = this.world.road.pts;
    const n = pts.length;
    // advance the look-ahead index
    let guard = 0;
    while (guard++ < 40) {
      const p = pts[this.index % n];
      const d = Math.hypot(p[0] - v.center[0], p[1] - v.center[2]);
      if (d < 9) this.index++;
      else break;
    }
    const look = 8 + Math.abs(v.forwardSpeed) * 0.55;
    let targetIdx = this.index;
    let acc = 0;
    for (let k = 0; k < 60; k++) {
      const a = pts[(this.index + k) % n];
      const b = pts[(this.index + k + 1) % n];
      acc += Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (acc > look) { targetIdx = (this.index + k) % n; break; }
    }
    const target = pts[targetIdx];
    // steering: angle between the car forward and the target in the ground plane
    const fwd = v.basis.z;
    const toT = [target[0] - v.center[0], 0, target[1] - v.center[2]];
    const fl = Math.hypot(fwd[0], fwd[2]) || 1;
    const tl = Math.hypot(toT[0], toT[2]) || 1;
    const cross = (fwd[0] / fl) * (toT[2] / tl) - (fwd[2] / fl) * (toT[0] / tl);
    const dot = (fwd[0] / fl) * (toT[0] / tl) + (fwd[2] / fl) * (toT[2] / tl);
    const ang = Math.atan2(cross, dot);
    const steerTarget = Math.max(-1, Math.min(1, ang * 2.4));
    const rate = 4.0;
    this.steerSmooth += Math.max(-rate * dt, Math.min(rate * dt, steerTarget - this.steerSmooth));

    // speed control: look at the curvature ahead
    const a = pts[(this.index + 18) % n];
    const b = pts[(this.index + 40) % n];
    const c = pts[(this.index + 62) % n];
    const ang1 = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const ang2 = Math.atan2(c[1] - b[1], c[0] - b[0]);
    let diff = ang2 - ang1;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    const curvature = Math.abs(diff) / 22;
    const maxSpeed = Math.min(52, 12 + 1 / Math.max(curvature, 0.002) * 0.28) * this.skill;
    const speed = Math.abs(v.forwardSpeed);
    let throttle = 0;
    let brake = 0;
    if (speed < maxSpeed) throttle = Math.min(1, (maxSpeed - speed) / 6 + 0.25);
    else brake = Math.min(1, (speed - maxSpeed) / 5);

    // recover when stuck or flipped
    if (speed < 0.6) this.stuck += dt; else this.stuck = 0;
    if (this.stuck > 3.5 || v.basis.y[1] < 0.4) {
      this.stuck = 0;
      const spawn = this.world.spawnOnTrack((this.index % n) / n);
      const NV = v.constructor;
      const nv = new NV(this.world, { spawn, name: v.name, isPlayer: false });
      Object.assign(v, nv);
      this.index = this.findStart();
    }

    this.input.throttle = throttle;
    this.input.brake = brake;
    this.input.steer = this.steerSmooth;
    this.input.handbrake = 0;
    v.lastInput = this.input;
  }
}
