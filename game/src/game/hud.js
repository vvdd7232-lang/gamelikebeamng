// Minimal DOM HUD: speedometer, damage, help panel, toasts.
export class HUD {
  constructor() {
    this.el = {
      speed: document.getElementById('speed'),
      gear: document.getElementById('gear'),
      power: document.getElementById('power-bar'),
      brake: document.getElementById('brake-bar'),
      damage: document.getElementById('damage-val'),
      damageBar: document.getElementById('damage-bar'),
      wheelDamage: document.getElementById('wheel-damage'),
      fps: document.getElementById('fps'),
      status: document.getElementById('status'),
      help: document.getElementById('help'),
      hint: document.getElementById('hint'),
      slowmo: document.getElementById('slowmo'),
      gmeter: document.getElementById('gmeter'),
      wheels: document.getElementById('wheel-status'),
    };
    this.lastSpeed = -1;
    this.helpVisible = true;
    this.toastTimer = 0;
    this.toastEl = document.getElementById('toast');
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

  update(d) {
    const speed = Math.round(d.speed);
    if (speed !== this.lastSpeed) {
      this.el.speed.textContent = String(speed);
      this.lastSpeed = speed;
    }
    if (this.el.gear) {
      this.el.gear.textContent = d.airborne ? 'AIR' : (d.speed < 1 ? 'N' : 'D');
    }
    if (this.el.power) this.el.power.style.width = (d.power * 100).toFixed(0) + '%';
    if (this.el.brake) this.el.brake.style.width = (d.brake * 100).toFixed(0) + '%';
    const dmg = Math.round(Math.min(1, d.damage * 4.2) * 100);
    if (this.el.damage) this.el.damage.textContent = dmg + '%';
    if (this.el.damageBar) {
      this.el.damageBar.style.width = Math.min(100, dmg) + '%';
      this.el.damageBar.style.background = dmg > 60 ? '#e34b3c' : dmg > 25 ? '#e0a13c' : '#4fb96b';
    }
    if (this.el.wheelDamage) {
      this.el.wheelDamage.textContent = d.detached > 0 ? `${d.detached} WHEEL${d.detached > 1 ? 'S' : ''} LOST` : '';
    }
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
    if (this.el.wheels) {
      const names = ['FL', 'FR', 'RL', 'RR'];
      this.el.wheels.innerHTML = names.map((n, i) => {
        const lost = d.wheelStates ? d.wheelStates[i] : false;
        return `<span class="wh ${lost ? 'lost' : ''}">${n}</span>`;
      }).join('');
    }
  }
}
