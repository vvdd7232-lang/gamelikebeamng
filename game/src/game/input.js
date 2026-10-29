// Keyboard + gamepad input mapped to a sane "car input" model.
export class Input {
  constructor(canvas) {
    this.keys = new Set();
    this.state = {
      throttle: 0, brake: 0, steer: 0, handbrake: 0,
      reset: false, camera: false, slowmo: false, respawn: false, cage: false,
      up: 0, down: 0, left: 0, right: 0,
    };
    this.pressed = new Set();
    this.mouse = { dx: 0, dy: 0, wheel: 0, dragging: false, down: false };
    this.gamepad = null;
    this.steerSmooth = 0;

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      this.keys.add(e.code);
      this.pressed.add(e.code);
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());

    canvas.addEventListener('mousedown', (e) => {
      this.mouse.down = true;
      this.mouse.dragging = true;
      this.mouse.lastX = e.clientX;
      this.mouse.lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { this.mouse.down = false; this.mouse.dragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (this.mouse.dragging) {
        this.mouse.dx += e.clientX - this.mouse.lastX;
        this.mouse.dy += e.clientY - this.mouse.lastY;
        this.mouse.lastX = e.clientX;
        this.mouse.lastY = e.clientY;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      this.mouse.wheel += Math.sign(e.deltaY);
      e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    window.addEventListener('gamepadconnected', () => { this.gamepadName = true; });
  }

  pollGamepad() {
    if (!navigator.getGamepads) return null;
    const pads = navigator.getGamepads();
    for (const p of pads) {
      if (p && p.connected) return p;
    }
    return null;
  }

  update(dt) {
    const k = this.keys;
    const s = this.state;
    let throttle = (k.has('KeyW') || k.has('ArrowUp')) ? 1 : 0;
    let brake = (k.has('KeyS') || k.has('ArrowDown')) ? 1 : 0;
    let steerL = (k.has('KeyA') || k.has('ArrowLeft')) ? 1 : 0;
    let steerR = (k.has('KeyD') || k.has('ArrowRight')) ? 1 : 0;
    let handbrake = k.has('Space') ? 1 : 0;

    const pad = this.pollGamepad();
    if (pad) {
      const ax = pad.axes[0] || 0;
      if (Math.abs(ax) > 0.12) { steerL = Math.max(steerL, -ax); steerR = Math.max(steerR, ax); }
      const rt = pad.buttons[7] ? pad.buttons[7].value : 0;
      const lt = pad.buttons[6] ? pad.buttons[6].value : 0;
      throttle = Math.max(throttle, rt);
      brake = Math.max(brake, lt);
      if (pad.buttons[0] && pad.buttons[0].pressed) handbrake = 1;
      if (pad.buttons[3] && pad.buttons[3].pressed) this.pressed.add('KeyR');
      if (pad.buttons[2] && pad.buttons[2].pressed) this.pressed.add('KeyC');
    }

    // steering: smooth towards the target, faster return to centre
    const target = steerL - steerR;      // +1 = left
    const rate = Math.abs(target) > 0.01 ? 3.6 : 6.5;
    const delta = target - this.steerSmooth;
    const maxStep = rate * dt;
    this.steerSmooth += Math.max(-maxStep, Math.min(maxStep, delta));

    s.throttle = throttle;
    s.brake = brake;
    s.steer = this.steerSmooth;
    s.handbrake = handbrake;
    return s;
  }

  consume(code) {
    if (this.pressed.has(code)) { this.pressed.delete(code); return true; }
    return false;
  }

  endFrame() {
    this.pressed.clear();
    this.mouse.dx = 0;
    this.mouse.dy = 0;
    this.mouse.wheel = 0;
  }
}
