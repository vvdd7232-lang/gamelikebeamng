// Chase / orbit / hood / cinematic camera.
import { mat4, clamp, lerp, v3 } from '../core/math.js';

export class Camera {
  constructor() {
    this.view = mat4.create();
    this.position = [0, 5, 10];
    this.target = [0, 0, 0];
    this.fov = 62;
    this.mode = 'chase';   // chase | orbit | hood | free
    this.yaw = 0;
    this.pitch = 0.22;
    this.dist = 8.2;
    this.smoothPos = [0, 5, 10];
    this.smoothTarget = [0, 0, 0];
    this.shake = 0;
    this.cinematic = 0;
    this.autoOrbit = 0;
  }

  update(dt, vehicle, input, world) {
    const c = vehicle.center;
    if (input.mouse.wheel) this.dist = clamp(this.dist + input.mouse.wheel * 0.9, 3.2, 26);
    if (input.mouse.dragging) {
      this.yaw -= input.mouse.dx * 0.005;
      this.pitch = clamp(this.pitch + input.mouse.dy * 0.0035, -0.35, 1.15);
    }
    if (this.mode === 'chase' || this.mode === 'orbit') {
      if (!input.mouse.dragging && Math.abs(vehicle.forwardSpeed) > 3) {
        // let the camera trail behind the car
        this.yaw = lerp(this.yaw, 0, clamp(dt * 1.8, 0, 1));
      }
      this.autoOrbit += dt * (this.mode === 'orbit' ? 0.35 : 0);
    }

    const basis = vehicle.basis;
    const carYaw = Math.atan2(basis.z[0], basis.z[2]);
    const yaw = carYaw + Math.PI + this.yaw + this.autoOrbit;
    const pitch = this.pitch;
    const dist = this.dist * (1 + clamp(vehicle.speed / 90, 0, 0.35));
    let px = c[0] + Math.sin(yaw) * Math.cos(pitch) * dist;
    let py = c[1] + Math.sin(pitch) * dist + 0.6;
    let pz = c[2] + Math.cos(yaw) * Math.cos(pitch) * dist;

    if (this.mode === 'hood') {
      const fwd = 0.35;
      px = c[0] + basis.z[0] * fwd + basis.y[0] * 0.72;
      py = c[1] + basis.z[1] * fwd + basis.y[1] * 0.72;
      pz = c[2] + basis.z[2] * fwd + basis.y[2] * 0.72;
    }

    // keep the camera above the ground
    const ground = world.heightAt(px, pz) + 0.8;
    if (py < ground) py = ground;

    const k = this.mode === 'hood' ? 1 : clamp(dt * 9, 0, 1);
    this.smoothPos[0] = lerp(this.smoothPos[0], px, k);
    this.smoothPos[1] = lerp(this.smoothPos[1], py, k);
    this.smoothPos[2] = lerp(this.smoothPos[2], pz, k);
    const look = this.mode === 'hood' ? 24 : 1.5;
    this.smoothTarget[0] = lerp(this.smoothTarget[0], c[0] + basis.z[0] * look, clamp(dt * 12, 0, 1));
    this.smoothTarget[1] = lerp(this.smoothTarget[1], c[1] + basis.z[1] * look + 0.35, clamp(dt * 12, 0, 1));
    this.smoothTarget[2] = lerp(this.smoothTarget[2], c[2] + basis.z[2] * look, clamp(dt * 12, 0, 1));

    // impact shake
    this.shake = Math.max(0, this.shake - dt * 2.2);
    const shakeAmt = this.shake * 0.35;
    const sx = (Math.random() - 0.5) * shakeAmt;
    const sy = (Math.random() - 0.5) * shakeAmt;
    this.position[0] = this.smoothPos[0] + sx;
    this.position[1] = this.smoothPos[1] + sy;
    this.position[2] = this.smoothPos[2];
    this.target[0] = this.smoothTarget[0];
    this.target[1] = this.smoothTarget[1];
    this.target[2] = this.smoothTarget[2];
    mat4.lookAt(this.view, this.position, this.target, [0, 1, 0]);

    this.fov = lerp(this.fov, 60 + clamp(Math.abs(vehicle.forwardSpeed) * 0.30, 0, 16), clamp(dt * 2, 0, 1));
  }

  snap(vehicle, world) {
    const c = vehicle.center;
    const basis = vehicle.basis;
    const carYaw = Math.atan2(basis.z[0], basis.z[2]);
    const yaw = carYaw + Math.PI + this.yaw;
    this.smoothPos[0] = c[0] + Math.sin(yaw) * Math.cos(this.pitch) * this.dist;
    this.smoothPos[1] = c[1] + Math.sin(this.pitch) * this.dist + 0.6;
    this.smoothPos[2] = c[2] + Math.cos(yaw) * Math.cos(this.pitch) * this.dist;
    this.smoothTarget[0] = c[0];
    this.smoothTarget[1] = c[1] + 0.3;
    this.smoothTarget[2] = c[2];
    void world;
  }
}
