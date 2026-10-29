// Soft-body physics: a node/beam network (the "crash cage") integrated with
// semi-implicit Euler.  Beams are spring-dampers with plastic yield and
// breakage -- that is what gives BeamNG-style crumpling.

export class SoftBody {
  /**
   * @param {Float64Array} restPos  rest positions (3n) in world space
   * @param {Float64Array} invMass  inverse mass per node
   * @param {Array} beams  [{a, b, rest, k, damp, yieldStrain, breakStrain, breakForce}]
   */
  constructor(restPos, invMass, beams) {
    const n = invMass.length;
    this.n = n;
    this.rest = restPos;
    this.pos = new Float64Array(restPos);
    this.vel = new Float64Array(3 * n);
    this.force = new Float64Array(3 * n);
    this.invMass = invMass;
    this.mass = new Float64Array(n);
    for (let i = 0; i < n; i++) this.mass[i] = invMass[i] > 0 ? 1 / invMass[i] : 1e9;

    this.nb = beams.length;
    this.ba = new Int32Array(this.nb);
    this.bb = new Int32Array(this.nb);
    this.brest = new Float64Array(this.nb);
    this.brest0 = new Float64Array(this.nb);   // pristine rest length
    this.bk = new Float64Array(this.nb);
    this.bk0 = new Float64Array(this.nb);
    this.bdmg = new Float64Array(this.nb);
    this.bdamp = new Float64Array(this.nb);
    this.bdamp0 = new Float64Array(this.nb);
    this.byield = new Float64Array(this.nb);   // tension
    this.byieldC = new Float64Array(this.nb);  // compression (folds earlier)
    this.bbreak = new Float64Array(this.nb);
    this.bbreakC = new Float64Array(this.nb);
    this.bbreakF = new Float64Array(this.nb);
    this.bbreakFC = new Float64Array(this.nb);
    this.bplastic = new Float64Array(this.nb);
    this.bloss = new Float64Array(this.nb);
    // how far a beam may be permanently crushed / stretched before the folded
    // metal jams up; without this limit the front clip keeps flowing backwards
    // until it is inside the cabin
    this.bsetMaxC = new Float64Array(this.nb);
    this.bsetMaxT = new Float64Array(this.nb);
    this.bjammed = new Uint8Array(this.nb);
    this.bkc = new Float64Array(this.nb);   // stiffness multiplier in compression
    this.bstrain = new Float64Array(this.nb);
    this.bbroken = new Uint8Array(this.nb);    // torn: still hangs by a strip
    this.bdamage = new Float64Array(this.nb);
    beams.forEach((b, i) => {
      this.ba[i] = b.a;
      this.bb[i] = b.b;
      this.brest[i] = b.rest;
      this.brest0[i] = b.rest;
      this.bk[i] = b.k;
      this.bk0[i] = b.k;
      this.bdamp[i] = b.damp;
      this.bdamp0[i] = b.damp;
      this.byield[i] = b.yieldStrain;
      this.byieldC[i] = b.yieldStrainC !== undefined ? b.yieldStrainC : b.yieldStrain * 0.8;
      this.bbreak[i] = b.breakStrain;
      this.bbreakC[i] = b.breakStrainC !== undefined ? b.breakStrainC : b.breakStrain;
      this.bbreakF[i] = b.breakForce;
      this.bbreakFC[i] = b.breakForceC !== undefined ? b.breakForceC : b.breakForce * 1.1;
      this.bplastic[i] = b.plasticRate !== undefined ? b.plasticRate : 1;
      this.bloss[i] = b.damageLoss !== undefined ? b.damageLoss : 0.45;
      this.bsetMaxC[i] = b.setMaxC !== undefined ? b.setMaxC : 0.55;
      this.bsetMaxT[i] = b.setMaxT !== undefined ? b.setMaxT : 0.85;
      this.bkc[i] = b.kCompress !== undefined ? b.kCompress : 1;
    });

    // per-node damage + permanent-set, used by the renderer for the crumpled
    // paint, the glass cracks and the dent pattern
    this.nodeDamage = new Float32Array(n);
    this.nodeCrumple = new Float32Array(n);

    this.brokenCount = 0;
    this.totalDamage = 0; // 0..1 (fraction of beams yielded/broken, weighted)
    this.maxImpact = 0;
    this.nodeRadius = 0.032;
    this.gravity = -9.81;
    this.dampingRate = 0.45;   // 1/s, velocity decay
    this.plasticRate = 2.6;    // permanent-set speed (1/s)
    this.plasticCap = 0.5;     // max strain absorbed per substep
    this.damageRate = 22;      // damage accumulation per unit overstrain
    this.tetherStiffness = 0.02; // a torn beam keeps hanging by a thin strip
    this.airDrag = 0.0009;
    this.contactImpulse = 0;
    this.centerVelocity = [0, 0, 0];
    this.noGround = false;
    // impact events produced during the last step (drained by the vehicle):
    // { x, y, z, power } power in [0..1] relative to "sure to hurt"
    this.events = [];
    this.contactEvent = null;
  }

  resetTo(restPos, transform) {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      let x = restPos[3 * i], y = restPos[3 * i + 1], z = restPos[3 * i + 2];
      if (transform) {
        const p = transform(x, y, z);
        x = p[0]; y = p[1]; z = p[2];
      }
      this.pos[3 * i] = x; this.pos[3 * i + 1] = y; this.pos[3 * i + 2] = z;
      this.vel[3 * i] = 0; this.vel[3 * i + 1] = 0; this.vel[3 * i + 2] = 0;
    }
    for (let i = 0; i < this.nb; i++) {
      this.bbroken[i] = 0;
      this.bdamage[i] = 0;
      this.bdmg[i] = 0;
      this.bk[i] = this.bk0[i];
      this.bdamp[i] = this.bdamp0[i];
      this.brest[i] = this.brest0[i];
    }
    this.nodeDamage.fill(0);
    this.nodeCrumple.fill(0);
    this.bjammed.fill(0);
    this.brokenCount = 0;
    this.totalDamage = 0;
    this.maxImpact = 0;
    this.events.length = 0;
    this.contactEvent = null;
  }

  addForce(i, fx, fy, fz) {
    this.force[3 * i] += fx;
    this.force[3 * i + 1] += fy;
    this.force[3 * i + 2] += fz;
  }

  /** integrate one substep; world supplies height/normal/collider queries */
  step(dt, world, colliders) {
    const { pos, vel, force, mass, invMass } = this;
    const n = this.n;
    // events are per-substep; the vehicle drains them right after this call
    if (this.events.length > 64) this.events.length = 64;
    this.contactEvent = null;

    // ---- gravity + air drag
    for (let i = 0; i < n; i++) {
      force[3 * i + 1] += mass[i] * this.gravity;
    }

    // ---- beams
    const {
      ba, bb, brest, bk, bdamp, byield, byieldC, bbreak, bbreakC, bbreakF, bbreakFC,
      bstrain, bbroken, bplastic, bloss, brest0, bkc,
    } = this;
    let brokenNow = 0;
    let damageSum = 0;
    for (let b = 0; b < this.nb; b++) {
      const a = ba[b] * 3, c = bb[b] * 3;
      let dx = pos[c] - pos[a];
      let dy = pos[c + 1] - pos[a + 1];
      let dz = pos[c + 2] - pos[a + 2];
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-6) continue;
      const inv = 1 / len;
      dx *= inv; dy *= inv; dz *= inv;
      const rest = brest[b];
      const strain = (len - rest) / rest;
      bstrain[b] = strain;

      const relV = (vel[c] - vel[a]) * dx + (vel[c + 1] - vel[a + 1]) * dy + (vel[c + 2] - vel[a + 2]) * dz;
      const k = bk[b];
      const elastic = k * (len - rest);      // >0 = stretched

      if (!bbroken[b]) {
        const tension = strain > 0;
        const yieldS = tension ? byield[b] : byieldC[b];
        const absStrain = Math.abs(strain);
        // ---- plastic deformation: permanent set + damage (time scaled, no runaway)
        if (absStrain > yieldS) {
          const over = absStrain - yieldS;
          // plastic flow: the beam takes a permanent set proportional to how far
          // past yield it is, but is capped so one frame cannot straighten it out.
          // Crumple zones flow fast (they fold), the safety cage barely at all.
          const flow = Math.min(over, this.plasticCap) * this.plasticRate * bplastic[b] * dt;
          let nextRest = brest[b] + Math.sign(strain) * flow * rest;
          // a folded beam bottoms out: past its crush limit the metal jams and
          // the joint gets *stiffer* instead of collapsing further into the car
          const setC = Math.max(0, (brest0[b] - nextRest) / brest0[b]);
          const setT = Math.max(0, (nextRest - brest0[b]) / brest0[b]);
          if (setC > this.bsetMaxC[b] || setT > this.bsetMaxT[b]) {
            nextRest = strain > 0
              ? brest0[b] * (1 + this.bsetMaxT[b])
              : brest0[b] * (1 - this.bsetMaxC[b]);
            this.bjammed[b] = 1;
          }
          brest[b] = nextRest;
          const softens = tension ? 1 : 1.35;
          // damage comes from two things: the strain rate that is doing the
          // work, and the permanent set the beam has already taken (a beam that
          // is 30% shorter is *damaged*, even if the hit was over in 50 ms)
          const set = Math.abs(nextRest - brest0[b]) / brest0[b];
          this.bdmg[b] = Math.min(1, Math.max(
            this.bdmg[b] + over * this.damageRate * softens * dt,
            set * 1.9,
          ));
          // per-beam damage is what the panels / HUD / shader read; it lags the
          // peak so a beam that has been folded stays "damaged" for good
          this.bdamage[b] = Math.max(this.bdamage[b], this.bdmg[b]);
          bk[b] = this.bk0[b] * (1 - bloss[b] * this.bdmg[b]) * (1 + (this.bjammed[b] ? 0.8 : 0));
          damageSum += Math.min(1, over * 8);
        }
        // ---- breakage: sheet metal tears; what is left of the joint still
        // hangs on by a thin strip instead of vanishing (that is what used to
        // turn the car into a cloud of unconnected vertices on a big hit)
        const breakS = tension ? bbreak[b] : bbreakC[b];
        const breakF = tension ? bbreakF[b] : bbreakFC[b];
        if (absStrain > breakS || Math.abs(elastic) > breakF) {
          bbroken[b] = 1;
          brokenNow++;
          this.bdamage[b] = 1;
          this.bdmg[b] = 1;
          const imp = Math.abs(elastic) * Math.min(absStrain, 1.0);
          if (imp > this.maxImpact) this.maxImpact = imp;
          if (this.events.length < 24) {
            const mx = (pos[a] + pos[c]) * 0.5, my = (pos[a + 1] + pos[c + 1]) * 0.5, mz = (pos[a + 2] + pos[c + 2]) * 0.5;
            this.events.push({ x: mx, y: my, z: mz, power: Math.min(1, absStrain * 1.6), tear: true });
          }
          // tether: slack at the current length, a fraction of the stiffness
          brest[b] = len;
          bk[b] = this.bk0[b] * this.tetherStiffness;
          bdamp[b] = this.bdamp0[b] * 2.5;
        } else if (absStrain > yieldS) {
          const imp = Math.abs(elastic) * (absStrain - yieldS);
          if (imp > this.maxImpact) this.maxImpact = imp;
        }
      }

      // ---- implicit spring + damper along the beam axis.
      // Backward Euler on the pair's axial DOF: unconditionally stable, so the
      // cage can be stiff without exploding.  Applied as a velocity impulse.
      // A buckled beam pushing back is much softer than a stretched one, which
      // is what folds a crumple zone into an accordion.
      const kNow = strain < 0 ? bk[b] * bkc[b] : bk[b];
      const ma = mass[ba[b]], mb = mass[bb[b]];
      const mEff = (ma * mb) / (ma + mb);
      const alpha = (kNow * dt * dt) / mEff;
      const beta = (bdamp[b] * dt) / mEff;
      const vNew = (relV - (kNow * dt * (len - rest)) / mEff) / (1 + beta + alpha);
      const j = (vNew - relV) * mEff;
      if (j !== 0) {
        vel[a] -= (dx * j) / ma; vel[a + 1] -= (dy * j) / ma; vel[a + 2] -= (dz * j) / ma;
        vel[c] += (dx * j) / mb; vel[c + 1] += (dy * j) / mb; vel[c + 2] += (dz * j) / mb;
      }
      // expose the local state of the shell to the renderer (crumpled paint,
      // crease lines, cracked glass).  Both are "highest ever", like real damage.
      const dmg = this.bdamage[b];
      const crumple = Math.min(1, Math.abs(brest[b] - brest0[b]) / (brest0[b] * 0.30));
      if (dmg > 0.01 || crumple > 0.01) {
        const ia = ba[b], ic = bb[b];
        if (dmg > this.nodeDamage[ia]) this.nodeDamage[ia] = dmg;
        if (dmg > this.nodeDamage[ic]) this.nodeDamage[ic] = dmg;
        if (crumple > this.nodeCrumple[ia]) this.nodeCrumple[ia] = crumple;
        if (crumple > this.nodeCrumple[ic]) this.nodeCrumple[ic] = crumple;
      }
    }
    if (brokenNow) this.brokenCount += brokenNow;
    this.damageSum = damageSum;

    // ---- ground + static collider contacts
    const r = this.nodeRadius;
    let contactSum = 0;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const x = pos[i3], y = pos[i3 + 1], z = pos[i3 + 2];
      // terrain
      const h = world.heightAt(x, z);
      if (!this.noGround && y - r < h) {
        const nrm = world.normalAt(x, z, _n);
        const depth = h + r - y;
        pos[i3 + 1] += depth;
        const vn = vel[i3] * nrm[0] + vel[i3 + 1] * nrm[1] + vel[i3 + 2] * nrm[2];
        if (vn < -3.5 && (!this.contactEvent || -vn > this.contactEvent.power)) {
          this.contactEvent = { x, y: h, z, power: Math.min(1, -vn / 22), normal: [nrm[0], nrm[1], nrm[2]] };
        }
        if (vn < 0) {
          const rest = 0.08;
          const jn = -(1 + rest) * vn;
          vel[i3] += nrm[0] * jn;
          vel[i3 + 1] += nrm[1] * jn;
          vel[i3 + 2] += nrm[2] * jn;
          contactSum += -vn;
          // Coulomb friction: the tangential impulse may never exceed
          // mu * (normal impulse).  (Removing all tangential velocity -- what
          // this used to do -- is an infinite brake that tears the cage apart.)
          const vdot = vel[i3] * nrm[0] + vel[i3 + 1] * nrm[1] + vel[i3 + 2] * nrm[2];
          const tx = vel[i3] - nrm[0] * vdot;
          const ty = vel[i3 + 1] - nrm[1] * vdot;
          const tz = vel[i3 + 2] - nrm[2] * vdot;
          const tl = Math.hypot(tx, ty, tz);
          if (tl > 1e-5) {
            const mu = world.materialMu(x, z) * 0.9;
            const maxDv = Math.min(tl, mu * jn);
            const k = maxDv / tl;
            vel[i3] -= tx * k;
            vel[i3 + 1] -= ty * k;
            vel[i3 + 2] -= tz * k;
          }
        }
      }
      // static boxes
      if (colliders) {
        for (let ci = 0; ci < colliders.length; ci++) {
          const c = _boxContact(colliders[ci], pos, i3, r);
          if (!c) continue;
          pos[i3] += c.normal[0] * c.depth;
          pos[i3 + 1] += c.normal[1] * c.depth;
          pos[i3 + 2] += c.normal[2] * c.depth;
          const vn = vel[i3] * c.normal[0] + vel[i3 + 1] * c.normal[1] + vel[i3 + 2] * c.normal[2];
          if (vn < -3.5 && (!this.contactEvent || -vn * 1.25 > this.contactEvent.power)) {
            this.contactEvent = {
              x: pos[i3] - c.normal[0] * c.depth,
              y: pos[i3 + 1] - c.normal[1] * c.depth,
              z: pos[i3 + 2] - c.normal[2] * c.depth,
              power: Math.min(1, -vn / 22),
              normal: [c.normal[0], c.normal[1], c.normal[2]],
              hard: true,
            };
          }
          if (vn < 0) {
            const jn = -(1 + 0.05) * vn;
            vel[i3] += c.normal[0] * jn;
            vel[i3 + 1] += c.normal[1] * jn;
            vel[i3 + 2] += c.normal[2] * jn;
            contactSum += -vn;
            // sliding friction against the obstacle, same Coulomb limit
            const vdot = vel[i3] * c.normal[0] + vel[i3 + 1] * c.normal[1] + vel[i3 + 2] * c.normal[2];
            const tx = vel[i3] - c.normal[0] * vdot;
            const ty = vel[i3 + 1] - c.normal[1] * vdot;
            const tz = vel[i3 + 2] - c.normal[2] * vdot;
            const tl = Math.hypot(tx, ty, tz);
            if (tl > 1e-5) {
              const k = Math.min(1, (0.45 * jn) / tl);
              vel[i3] -= tx * k;
              vel[i3 + 1] -= ty * k;
              vel[i3 + 2] -= tz * k;
            }
          }
        }
      }
    }
    this.contactImpulse = this.contactImpulse * 0.9 + contactSum * 0.1;

    // ---- integrate
    // The damping below must act on *deformation* only.  Damping the absolute
    // velocity would be a body-wide air brake of dampingRate * mass (~10 kN at
    // 60 km/h), so subtract the rigid translation first and add it back after.
    const drag = this.airDrag;
    const dampFactor = Math.exp(-this.dampingRate * dt);
    let cvx = 0, cvy = 0, cvz = 0, msum = 0;
    for (let i = 0; i < n; i++) {
      const im = invMass[i];
      if (im === 0) continue;
      const m = mass[i];
      cvx += vel[i * 3] * m; cvy += vel[i * 3 + 1] * m; cvz += vel[i * 3 + 2] * m;
      msum += m;
    }
    if (msum > 0) { cvx /= msum; cvy /= msum; cvz /= msum; }
    this.centerVelocity[0] = cvx; this.centerVelocity[1] = cvy; this.centerVelocity[2] = cvz;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const im = invMass[i];
      if (im === 0) continue;
      const rx = vel[i3] - cvx, ry = vel[i3 + 1] - cvy, rz = vel[i3 + 2] - cvz;
      const sp = Math.hypot(rx, ry, rz);
      const air = 1 - Math.min(0.3, drag * sp * dt);
      const d = air * dampFactor;
      // rigid translation passes through untouched, only the wobble is damped
      vel[i3] = cvx + (rx + force[i3] * im * dt) * d;
      vel[i3 + 1] = cvy + (ry + force[i3 + 1] * im * dt) * d;
      vel[i3 + 2] = cvz + (rz + force[i3 + 2] * im * dt) * d;
      pos[i3] += vel[i3] * dt;
      pos[i3 + 1] += vel[i3 + 1] * dt;
      pos[i3 + 2] += vel[i3 + 2] * dt;
      force[i3] = 0; force[i3 + 1] = 0; force[i3 + 2] = 0;
    }
  }

  nodeVelocity(i, out) {
    out[0] = this.vel[i * 3];
    out[1] = this.vel[i * 3 + 1];
    out[2] = this.vel[i * 3 + 2];
    return out;
  }

  /** average velocity near a world point, weighted by 1/dist^2 */
  velNear(x, y, z, out) {
    const { pos, vel, n } = this;
    let wsum = 0;
    out[0] = 0; out[1] = 0; out[2] = 0;
    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3] - x, dy = pos[i * 3 + 1] - y, dz = pos[i * 3 + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz + 0.35;
      const w = 1 / ((d2 + 0.25) * (d2 + 0.25));
      wsum += w;
      out[0] += vel[i * 3] * w;
      out[1] += vel[i * 3 + 1] * w;
      out[2] += vel[i * 3 + 2] * w;
    }
    if (wsum > 0) { out[0] /= wsum; out[1] /= wsum; out[2] /= wsum; }
    return out;
  }

  /** weighted force over a cached node list (very fast, no search) */
  applyForceNodes(idx, w, fx, fy, fz) {
    const f = this.force;
    for (let i = 0; i < idx.length; i++) {
      const o = idx[i] * 3, ww = w[i];
      f[o] += fx * ww;
      f[o + 1] += fy * ww;
      f[o + 2] += fz * ww;
    }
  }

  /** weighted mean velocity of a cached node list */
  velNodes(idx, w, out) {
    const v = this.vel;
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < idx.length; i++) {
      const o = idx[i] * 3, ww = w[i];
      x += v[o] * ww; y += v[o + 1] * ww; z += v[o + 2] * ww;
    }
    out[0] = x; out[1] = y; out[2] = z;
    return out;
  }

  /** apply a world force near a point, spread over the k nearest nodes */
  applyForceNear(x, y, z, fx, fy, fz, k = 4, radius = 1.2) {
    const { pos, n } = this;
    let best = [];
    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3] - x, dy = pos[i * 3 + 1] - y, dz = pos[i * 3 + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > radius * radius) continue;
      best.push([d2, i]);
    }
    if (best.length === 0) return;
    best.sort((a, b) => a[0] - b[0]);
    best = best.slice(0, k);
    let wsum = 0;
    for (const e of best) wsum += 1 / (e[0] + 0.18);
    for (const e of best) {
      const w = (1 / (e[0] + 0.18)) / wsum;
      this.addForce(e[1], fx * w, fy * w, fz * w);
    }
  }

  centroid(out) {
    const { pos, n } = this;
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < n; i++) { x += pos[i * 3]; y += pos[i * 3 + 1]; z += pos[i * 3 + 2]; }
    out[0] = x / n; out[1] = y / n; out[2] = z / n;
    return out;
  }

  velocity(out) {
    const { vel, n } = this;
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < n; i++) { x += vel[i * 3]; y += vel[i * 3 + 1]; z += vel[i * 3 + 2]; }
    out[0] = x / n; out[1] = y / n; out[2] = z / n;
    return out;
  }

  totalMass() {
    let m = 0;
    for (let i = 0; i < this.n; i++) m += this.mass[i];
    return m;
  }
}

const _n = [0, 1, 0];
function _boxContact(box, pos, i3, r) {
  const d = [pos[i3] - box.center[0], pos[i3 + 1] - box.center[1], pos[i3 + 2] - box.center[2]];
  const local = [
    d[0] * box.axes[0][0] + d[1] * box.axes[0][1] + d[2] * box.axes[0][2],
    d[0] * box.axes[1][0] + d[1] * box.axes[1][1] + d[2] * box.axes[1][2],
    d[0] * box.axes[2][0] + d[1] * box.axes[2][1] + d[2] * box.axes[2][2],
  ];
  const cl = [
    Math.max(-box.half[0], Math.min(box.half[0], local[0])),
    Math.max(-box.half[1], Math.min(box.half[1], local[1])),
    Math.max(-box.half[2], Math.min(box.half[2], local[2])),
  ];
  const ox = local[0] - cl[0], oy = local[1] - cl[1], oz = local[2] - cl[2];
  const dist = Math.hypot(ox, oy, oz);
  let nl;
  let depth;
  if (dist > 1e-6) {
    if (dist >= r) return null;
    nl = [ox / dist, oy / dist, oz / dist];
    depth = r - dist;
  } else {
    let mi = 0, best = Infinity;
    for (let i = 0; i < 3; i++) {
      const pen = box.half[i] - Math.abs(local[i]);
      if (pen < best) { best = pen; mi = i; }
    }
    const s = local[mi] < 0 ? -1 : 1;
    nl = [0, 0, 0];
    nl[mi] = s;
    depth = best + r;
  }
  return {
    normal: [
      nl[0] * box.axes[0][0] + nl[1] * box.axes[1][0] + nl[2] * box.axes[2][0],
      nl[0] * box.axes[0][1] + nl[1] * box.axes[1][1] + nl[2] * box.axes[2][1],
      nl[0] * box.axes[0][2] + nl[1] * box.axes[1][2] + nl[2] * box.axes[2][2],
    ],
    depth,
  };
}
