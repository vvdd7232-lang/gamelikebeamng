// Minimal DOM HUD: speedometer, live damage map, help panel, toasts.

/** green -> amber -> red -> dead */
function damageColor(v) {
  // v = 0 (pristine) .. 1 (destroyed)
  if (v < 0.34) {
    const t = v / 0.34;
    return `rgb(${Math.round(79 + (224 - 79) * t)}, ${Math.round(185 + (161 - 185) * t)}, ${Math.round(107 + (60 - 107) * t)})`;
  }
  const t = Math.min(1, (v - 0.34) / 0.66);
  return `rgb(${Math.round(224 + (140 - 224) * t)}, ${Math.round(161 + (38 - 161) * t)}, ${Math.round(60 + (32 - 60) * t)})`;
}

export class HUD {
  constructor() {
    this.el = {
      speed: document.getElementById('speed'),
      gear: document.getElementById('gear'),
      power: document.getElementById('power-bar'),
      brake: document.getElementById('brake-bar'),
      damage: document.getElementById('damage'),
      damageVal: document.getElementById('damage-val'),
      cabinVal: document.getElementById('cabin-val'),
      engineVal: document.getElementById('engine-val'),
      wheelsVal: document.getElementById('wheels-val'),
      wheelDamage: document.getElementById('wheel-damage'),
      fps: document.getElementById('fps'),
      status: document.getElementById('status'),
      help: document.getElementById('help'),
      slowmo: document.getElementById('slowmo'),
      gmeter: document.getElementById('gmeter'),
      zones: [0, 1, 2, 3, 4, 5].map((i) => document.getElementById('zn-' + i)),
      wheelRects: {
        FL: document.getElementById('wh-FL'),
        FR: document.getElementById('wh-FR'),
        RL: document.getElementById('wh-RL'),
        RR: document.getElementById('wh-RR'),
      },
      banner: document.getElementById('impact-banner'),
      bannerKmh: document.getElementById('impact-kmh'),
    };
    this.lastSpeed = -1;
    this.lastZones = [-1, -1, -1, -1, -1, -1];
    this.lastWheels = ['', '', '', ''];
    this.helpVisible = true;
    this.toastTimer = 0;
    this.toastEl = document.getElementById('toast');
    this.hitTimer = 0;
    this.bannerTimer = 0;
  }

  toggleHelp() {
    this.helpVisible = !this.helpVisible;
    if (this.el.help) this.el.help.classList.toggle('hidden', !this.helpVisible);
  }

  toast(text, ms = 1800) {
    if (!this.toastEl) return;
    this.toastEl.textContent = text;
    this.toastEl.classList.remove('hidden');
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => this.toastEl.classList.add('hidden'), ms);
  }

  /** colour a stat value: green when healthy, red when wrecked */
  setStat(el, value, invert = false) {
    if (!el) return;
    if (!Number.isFinite(value)) value = 0;
    el.textContent = Math.round(value) + '%';
    const bad = invert ? value : 100 - value;
    el.className = bad > 60 ? 'bad' : bad > 25 ? 'warn' : 'good';
  }

  update(d) {
    const speed = Math.round(d.speed);
    if (speed !== this.lastSpeed) {
      if (this.el.speed) this.el.speed.textContent = String(speed);
      this.lastSpeed = speed;
    }
    if (this.el.gear) {
      this.el.gear.textContent = d.airborne ? 'AIR' : (d.speed < 1 ? 'N' : 'D');
    }
    if (this.el.power) this.el.power.style.width = (d.power * 100).toFixed(0) + '%';
    if (this.el.brake) this.el.brake.style.width = (d.brake * 100).toFixed(0) + '%';

    // ---- damage map
    const pct = Math.round(Math.min(1, Number.isFinite(d.damage) ? d.damage : 0) * 100);
    this.setStat(this.el.damageVal, pct, true);
    this.setStat(this.el.cabinVal, Math.round((d.cabin !== undefined ? d.cabin : 1) * 100));
    this.setStat(this.el.engineVal, Math.round((d.powerScale !== undefined ? d.powerScale : 1) * 100));
    if (this.el.wheelsVal) {
      const lost = d.detached || 0;
      this.el.wheelsVal.textContent = `${4 - lost}/4`;
      this.el.wheelsVal.className = lost >= 2 ? 'bad' : lost ? 'warn' : 'good';
    }
    if (this.el.zones && d.zones) {
      for (let i = 0; i < 6; i++) {
        const z = Math.min(1, d.zones[i] * 2.4);
        if (Math.abs(z - this.lastZones[i]) < 0.01) continue;
        this.lastZones[i] = z;
        const el = this.el.zones[i];
        if (!el) continue;
        el.style.fill = damageColor(z);
        el.style.opacity = String(0.55 + 0.4 * (1 - z));
      }
    }
    // wheels on the silhouette
    if (d.wheelStates && this.el.wheelRects) {
      const names = ['FL', 'FR', 'RL', 'RR'];
      for (let i = 0; i < 4; i++) {
        const lost = d.wheelStates[i];
        const rim = d.wheelRim ? d.wheelRim[i] : 0;
        const cls = lost ? 'lost' : rim > 0.45 ? 'bent' : '';
        if (cls === this.lastWheels[i]) continue;
        this.lastWheels[i] = cls;
        const el = this.el.wheelRects[names[i]];
        if (el) el.setAttribute('class', cls);
      }
    }
    if (this.el.wheelDamage) {
      this.el.wheelDamage.textContent = d.detached > 0
        ? `${d.detached} WHEEL${d.detached > 1 ? 'S' : ''} LOST`
        : (d.wheelRim && Math.max(...d.wheelRim) > 0.45 ? 'WHEEL DAMAGE' : '');
    }

    // ---- impact feedback: flash the panel, show how hard the hit was
    if (d.impact) {
      this.hitTimer = 0.55;
      this.bannerTimer = 1.6;
      if (this.el.bannerKmh) this.el.bannerKmh.textContent = Math.round(d.impact);
    }
    this.hitTimer = Math.max(0, this.hitTimer - (d.dt || 0.016));
    this.bannerTimer = Math.max(0, this.bannerTimer - (d.dt || 0.016));
    if (this.el.damage) this.el.damage.classList.toggle('hit', this.hitTimer > 0);
    if (this.el.banner) this.el.banner.classList.toggle('hidden', this.bannerTimer <= 0);

    if (this.el.fps) this.el.fps.textContent = d.fps.toFixed(0);
    if (this.el.slowmo) this.el.slowmo.classList.toggle('hidden', !d.slowmo);
    if (this.el.status) {
      this.el.status.textContent = d.paused ? 'PAUSED' : '';
      this.el.status.classList.toggle('hidden', !d.paused);
    }
    if (this.el.gmeter) {
      const g = Math.hypot(d.gForce[0], d.gForce[2]);
      this.el.gmeter.textContent = g.toFixed(1) + ' g';
    }
  }
}
