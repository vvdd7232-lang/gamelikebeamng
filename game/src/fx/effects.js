// Visual effects: dust/smoke/sparks particles and persistent skid marks.
import { rand, clamp } from '../core/math.js';
import { MATERIALS } from '../world/terrain.js';

const MAX_PARTICLES = 3000;
const MAX_SKID_VERTS = 40000;

export class Effects {
  constructor(world) {
    this.world = world;
    this.parts = [];
    this.particleData = new Float32Array(MAX_PARTICLES * 7);
    this.particles = { data: this.particleData, count: 0 };

    this.skidVerts = new Float32Array(MAX_SKID_VERTS * 4);
    this.skidCount = 0;
    this.skid = { data: this.skidVerts, count: 0 };
    this.skidHead = 0;
    this.skidMarks = [];
    this.lastSkid = new Map();
    this.smokeTimer = 0;
  }

  reset() {
    this.parts.length = 0;
    this.skidCount = 0;
    this.skidHead = 0;
    this.lastSkid.clear();
    this.skidVerts.fill(0);
    this.skid.count = 0;
    this.particles.count = 0;
  }

  /** forget the skid continuity of one car (after a respawn) */
  clearFor(name) {
    for (const key of this.lastSkid.keys()) {
      if (key.startsWith(name + ':')) this.lastSkid.delete(key);
    }
  }

  spawnParticle(x, y, z, vx, vy, vz, life, size, color, gravity = -2.2, drag = 1.4) {
    if (this.parts.length >= MAX_PARTICLES) return;
    this.parts.push({ x, y, z, vx, vy, vz, life, maxLife: life, size, color, gravity, drag });
  }

  // ------------------------------------------------------------- car effects
  /**
   * Everything that comes off a car when it hits something: sparks off bare
   * metal, a dust cloud, glass shards, chunks of bodywork, smoke from a torn
   * engine bay and the odd lick of flame.  Driven by the exact contact points
   * the solver reports (vehicle.impacts), not by a guess.
   */
  impactBurst(vehicle, imp) {
    const p = imp.power;
    const n = imp.normal || [0, 1, 0];
    const px = imp.x, py = imp.y, pz = imp.z;
    const speed = Math.max(4, vehicle.speed || 0);

    // --- sparks: hot, fast, short lived, flying along the surface
    const sparkCount = Math.round(4 + p * 30);
    for (let i = 0; i < sparkCount; i++) {
      const s = 0.6 + p * 2.2;
      this.spawnParticle(
        px + rand(-0.12, 0.12), py + rand(-0.06, 0.2), pz + rand(-0.12, 0.12),
        (n[0] + rand(-1, 1)) * speed * 0.25 * s,
        (Math.abs(n[1]) * 0.5 + rand(0.2, 1.4)) * speed * 0.18 * s,
        (n[2] + rand(-1, 1)) * speed * 0.25 * s,
        rand(0.18, 0.6), rand(0.05, 0.13),
        [1.0, rand(0.65, 0.9), rand(0.15, 0.45)], -11.0, 0.5,
      );
    }
    // --- dust / gravel kicked out of the impact
    const dustCount = Math.round(2 + p * 12);
    for (let i = 0; i < dustCount; i++) {
      this.spawnParticle(
        px + rand(-0.4, 0.4), py + rand(0.0, 0.5), pz + rand(-0.4, 0.4),
        rand(-2.2, 2.2), rand(0.5, 2.6), rand(-2.2, 2.2),
        rand(0.8, 2.2), rand(0.3, 0.9),
        [0.52, 0.5, 0.47], -0.9, 1.5,
      );
    }
    // --- glass shards + bright flakes on harder hits
    if (p > 0.35 || imp.tear) {
      const shards = Math.round(p * 14);
      for (let i = 0; i < shards; i++) {
        this.spawnParticle(
          px + rand(-0.5, 0.5), py + rand(-0.1, 0.6), pz + rand(-0.5, 0.5),
          rand(-3.4, 3.4), rand(0.4, 3.2), rand(-3.4, 3.4),
          rand(0.5, 1.3), rand(0.04, 0.1),
          [0.86, 0.92, 1.0], -9.0, 0.5,
        );
      }
      // panels / paint chips (dark flakes tumbling away)
      for (let i = 0; i < Math.round(p * 10); i++) {
        this.spawnParticle(
          px + rand(-0.6, 0.6), py + rand(0.0, 0.7), pz + rand(-0.6, 0.6),
          rand(-2.6, 2.6), rand(0.6, 2.8), rand(-2.6, 2.6),
          rand(0.6, 1.6), rand(0.06, 0.16),
          [0.25, 0.06, 0.06], -9.0, 0.6,
        );
      }
    }
    // --- smoke column on a heavy hit (the crumple zone venting)
    if (p > 0.55) {
      for (let i = 0; i < 6; i++) {
        this.spawnParticle(
          px + rand(-0.5, 0.5), py + rand(0.1, 0.8), pz + rand(-0.5, 0.5),
          rand(-0.7, 0.7), rand(0.8, 2.2), rand(-0.7, 0.7),
          rand(1.4, 3.0), rand(0.5, 1.2),
          [0.24, 0.24, 0.25], -0.3, 1.0,
        );
      }
    }
  }

  carEffects(vehicle, dt, speed) {
    // drain the impacts the solver reported this frame
    if (vehicle.impacts && vehicle.impacts.length) {
      for (const imp of vehicle.impacts) {
        if (imp.power > 0.12) this.impactBurst(vehicle, imp);
      }
      vehicle.impacts.length = 0;
    }
    this.engineEffects(vehicle, dt);
    for (const w of vehicle.wheels) {
      if (w.detached) continue;
      const skid = w.skid || 0;
      const mat = MATERIALS[vehicle.surfaceMat !== undefined ? vehicle.surfaceMat : 1];
      const onHard = mat.name === 'asphalt' || mat.name === 'concrete';
      // skid marks
      if (w.contact && skid > 0.12 && speed > 1.2) {
        this.addSkidSegment(vehicle, w, skid, onHard);
      }
      // dust / smoke
      const dustAmount = (w.contact ? skid : 0) * (onHard ? 0.5 : 1.6) + (w.contact ? 0 : 0);
      if (dustAmount > 0.25 && Math.random() < dustAmount * dt * 34) {
        const c = onHard ? [0.55, 0.55, 0.58] : mat.dust;
        const n = onHard ? 1 : 2;
        for (let i = 0; i < n; i++) {
          this.spawnParticle(
            w.contactPoint[0] + rand(-0.25, 0.25),
            w.contactPoint[1] + rand(0.02, 0.18),
            w.contactPoint[2] + rand(-0.25, 0.25),
            rand(-1.1, 1.1) - vehicle.velocity[0] * 0.10,
            rand(0.5, 1.9),
            rand(-1.1, 1.1) - vehicle.velocity[2] * 0.10,
            rand(0.7, 1.9),
            rand(0.35, 0.95),
            [c[0] * 1.05, c[1] * 1.05, c[2] * 1.05],
            onHard ? -0.4 : -1.2, 1.1,
          );
        }
      }
      // smoke off a flat tyre / bent rim
      if ((w.rimDamage || 0) > 0.55 && w.contact && speed > 2 && Math.random() < dt * 14) {
        this.spawnParticle(
          w.contactPoint[0] + rand(-0.15, 0.15), w.contactPoint[1] + 0.1,
          w.contactPoint[2] + rand(-0.15, 0.15),
          rand(-0.4, 0.4), rand(0.4, 1.3), rand(-0.4, 0.4),
          rand(0.5, 1.4), rand(0.25, 0.6), [0.42, 0.42, 0.43], -0.4, 1.5,
        );
      }
      // sparks when a wheel is rim-damaged or the body is grinding
      if (w.rimDamage > 0.5 && w.contact && speed > 3 && Math.random() < dt * 30) {
        this.spawnParticle(
          w.contactPoint[0], w.contactPoint[1] + 0.05, w.contactPoint[2],
          rand(-2.5, 2.5), rand(0.6, 2.4), rand(-2.5, 2.5),
          rand(0.25, 0.6), rand(0.06, 0.14), [1.0, 0.75, 0.28], -9.0, 0.4,
        );
      }
    }
    // body scraping the ground
    const b = vehicle.body;
    if (vehicle.damage > 0.12 && speed > 5) {
      for (let i = 0; i < vehicle.nodeCount; i++) {
        const y = b.pos[i * 3 + 1];
        if (y - this.world.heightAt(b.pos[i * 3], b.pos[i * 3 + 2]) < 0.12 && Math.random() < dt * 6) {
          this.spawnParticle(
            b.pos[i * 3], y + 0.03, b.pos[i * 3 + 2],
            rand(-1.5, 1.5), rand(0.4, 1.6), rand(-1.5, 1.5),
            rand(0.3, 0.9), rand(0.15, 0.4), [1.0, 0.8, 0.35], -7, 0.6,
          );
        }
      }
    }
    // impact flash burst
    if (vehicle.impactFlash > 0.55 && !vehicle._impactFx) {
      vehicle._impactFx = true;
      const p = vehicle.center;
      for (let i = 0; i < 16; i++) {
        this.spawnParticle(
          p[0] + rand(-0.6, 0.6), p[1] + rand(-0.3, 0.5), p[2] + rand(-0.6, 0.6),
          rand(-5, 5), rand(0.5, 4), rand(-5, 5),
          rand(0.2, 0.7), rand(0.08, 0.2), [1.0, 0.85, 0.5], -8, 0.8,
        );
      }
      // sparks + debris cloud
      for (let i = 0; i < 10; i++) {
        this.spawnParticle(
          p[0] + rand(-1.0, 1.0), p[1] + rand(-0.4, 0.8), p[2] + rand(-1.0, 1.0),
          rand(-1.4, 1.4), rand(0.4, 2.4), rand(-1.4, 1.4),
          rand(0.6, 1.6), rand(0.4, 1.0), [0.6, 0.6, 0.62], -1.0, 1.3,
        );
      }
    } else if (vehicle.impactFlash < 0.2) {
      vehicle._impactFx = false;
    }
  }

  /** smoke, and eventually fire, pouring out of a wrecked front end */
  engineEffects(vehicle, dt) {
    const dmg = vehicle.engineDamage || 0;
    if (dmg < 0.32) return;
    const bay = vehicle.toWorldPoint([0, 0.62, 1.75]);
    const heat = clamp((dmg - 0.32) / 0.6, 0, 1);
    // smoke: continuous while the car is alive
    if (Math.random() < dt * (6 + heat * 26)) {
      const dark = 0.11 + heat * 0.06;
      this.spawnParticle(
        bay[0] + rand(-0.3, 0.3), bay[1] + rand(0.0, 0.25), bay[2] + rand(-0.3, 0.3),
        rand(-0.6, 0.6), rand(0.9, 2.4), rand(-0.6, 0.6),
        rand(1.2, 2.8), rand(0.4, 1.0), [dark, dark, dark * 1.05], 0.2, 0.9,
      );
    }
    // fire only when the bay is really destroyed
    if (heat > 0.45 && Math.random() < dt * 22) {
      for (let i = 0; i < 2; i++) {
        this.spawnParticle(
          bay[0] + rand(-0.22, 0.22), bay[1] + rand(-0.05, 0.2), bay[2] + rand(-0.22, 0.22),
          rand(-0.5, 0.5), rand(1.4, 3.4), rand(-0.5, 0.5),
          rand(0.22, 0.6), rand(0.16, 0.42),
          [1.0, rand(0.35, 0.7), 0.12], 1.2, 1.2,
        );
      }
    }
  }

  addSkidSegment(vehicle, w, intensity, onHard) {
    const key = vehicle.wheels.indexOf(w) + ':' + (vehicle.isPlayer ? 'p' : vehicle.name);
    const prev = this.lastSkid.get(key);
    const p = w.contactPoint;
    const now = { x: p[0], y: p[1] + 0.03, z: p[2], t: performance.now() };
    if (!prev) { this.lastSkid.set(key, now); return; }
    const dx = now.x - prev.x, dz = now.z - prev.z;
    const dist = Math.hypot(dx, dz);
    if (dist < 0.22) return;
    if (dist > 3.5) { this.lastSkid.set(key, now); return; }
    const halfW = 0.13;
    const nx = -dz / (dist || 1) * halfW, nz = dx / (dist || 1) * halfW;
    const alpha = clamp(intensity * 0.9, 0, 1) * (onHard ? 1 : 0.75);
    // two triangles
    const v = this.skidVerts;
    const push = (x, y, z, a) => {
      let head = this.skidHead * 4;
      if (head >= v.length) { this.skidHead = 0; head = 0; this.skidCount = 0; }
      v[head] = x; v[head + 1] = y; v[head + 2] = z; v[head + 3] = a;
      this.skidHead++;
      this.skidCount = Math.max(this.skidCount, this.skidHead);
    };
    push(prev.x - nx, prev.y, prev.z - nz, alpha);
    push(prev.x + nx, prev.y, prev.z + nz, alpha);
    push(now.x + nx, now.y, now.z + nz, alpha);
    push(prev.x - nx, prev.y, prev.z - nz, alpha);
    push(now.x + nx, now.y, now.z + nz, alpha);
    push(now.x - nx, now.y, now.z - nz, alpha);
    this.skid.count = this.skidCount;
    this.lastSkid.set(key, now);
  }

  // ----------------------------------------------------------------- update
  update(dt) {
    const parts = this.parts;
    let write = 0;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      p.life -= dt;
      if (p.life <= 0) continue;
      p.vy += p.gravity * dt;
      const damp = Math.exp(-p.drag * dt);
      p.vx *= damp; p.vy *= damp; p.vz *= damp;
      p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
      const ground = this.world.heightAt(p.x, p.z);
      if (p.y < ground + 0.02) { p.y = ground + 0.02; p.vy *= -0.12; p.vx *= 0.7; p.vz *= 0.7; }
      parts[write++] = p;
    }
    parts.length = write;

    const data = this.particleData;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      const o = i * 7;
      const t = p.life / p.maxLife;
      data[o] = p.x; data[o + 1] = p.y; data[o + 2] = p.z;
      data[o + 3] = p.color[0]; data[o + 4] = p.color[1]; data[o + 5] = p.color[2];
      data[o + 6] = p.size * (1.5 - t * 0.6) * (0.35 + t * 0.65);
    }
    this.particles.count = parts.length;
    this.skid.count = this.skidCount;
  }
}
