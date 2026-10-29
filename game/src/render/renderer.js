// WebGL2 renderer: shadow map, sky, terrain, props, soft-body skinned cars,
// particles and skid marks.  Deliberately dependency free.
import { createContext, createProgram, createMesh, createTexture, createFramebuffer, fullscreenTriangle } from './gl.js';
import * as S from './shaders.js';
import { mat4, v3, clamp } from '../core/math.js';
import { MATERIALS } from '../world/terrain.js';

const SHADOW_SIZE = 2048;
export const MAX_NODES = 320;

/**
 * Pack the soft-body nodes into the GPU node texture.
 *
 * The texture is MAX_NODES wide and 5 tall, so **node j lives in column j** and
 * the shaders read it with `texelFetch(uNodeTex, ivec2(j, row), 0)`:
 *
 *   row 0 -> world position of the node
 *   row 1 -> column 0 of the node's local rotation D
 *   row 2 -> column 1
 *   row 3 -> column 2
 *   row 4 -> rest (bind) world position
 *
 * `gl.texSubImage2D` fills row-major, hence the `row * maxNodes + i` indexing;
 * writing the nodes as 5 consecutive texels instead (row-major per node) would
 * silently scramble every vertex on the GPU.
 */
export function packNodeData(vehicle, out, maxNodes = MAX_NODES) {
  const n = Math.min(vehicle.nodeCount, maxNodes);
  const pos = vehicle.body.pos;
  const rest = vehicle.restWorld;
  const tex = vehicle.nodeTex;          // 16 floats per node: columns of D at +4, +8, +12
  const nodeDamage = vehicle.body.nodeDamage;
  const nodeCrumple = vehicle.body.nodeCrumple;
  for (let i = 0; i < n; i++) {
    const t = i * 16;
    let o = i * 4;
    // .w of the position row = damage of this node (read by MESH_VS)
    out[o] = pos[i * 3]; out[o + 1] = pos[i * 3 + 1]; out[o + 2] = pos[i * 3 + 2];
    out[o + 3] = nodeDamage ? nodeDamage[i] : 0;
    o = (maxNodes + i) * 4;
    out[o] = tex[t + 4]; out[o + 1] = tex[t + 5]; out[o + 2] = tex[t + 6]; out[o + 3] = 0;
    o = (maxNodes * 2 + i) * 4;
    out[o] = tex[t + 8]; out[o + 1] = tex[t + 9]; out[o + 2] = tex[t + 10]; out[o + 3] = 0;
    o = (maxNodes * 3 + i) * 4;
    // .w of the third rotation column = crumple of this node
    out[o] = tex[t + 12]; out[o + 1] = tex[t + 13]; out[o + 2] = tex[t + 14];
    out[o + 3] = nodeCrumple ? nodeCrumple[i] : 0;
    o = (maxNodes * 4 + i) * 4;
    out[o] = rest[i * 3]; out[o + 1] = rest[i * 3 + 1]; out[o + 2] = rest[i * 3 + 2]; out[o + 3] = 1;
  }
  return n;
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = createContext(canvas);
    if (!this.gl) throw new Error('WebGL2 is not available in this browser.');
    const gl = this.gl;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderScale = 1;      // reduced automatically on slow GPUs
    this.width = canvas.width;
    this.height = canvas.height;

    this.identity3 = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    this.progSky = createProgram(gl, S.SKY_VS, S.SKY_FS, 'sky');
    this.progMesh = createProgram(gl, S.MESH_VS, S.MESH_FS, 'mesh');
    this.progDepth = createProgram(gl, S.DEPTH_VS, S.DEPTH_FS, 'depth');
    this.progSkid = createProgram(gl, S.SKID_VS, S.SKID_FS, 'skid');
    this.progParticle = createProgram(gl, S.PARTICLE_VS, S.PARTICLE_FS, 'particle');

    this.skyVao = fullscreenTriangle(gl);
    this.viewProj = mat4.create();
    this.lightViewProj = mat4.create();
    this.shadowMat = mat4.create();
    this.tmpMat = mat4.create();
    this.model = mat4.create();
    this.normalMat = new Float32Array(9);

    this.env = {
      sunDir: [0.42, 0.72, 0.30],
      sunColor: [1.0, 0.92, 0.78],
      skyColor: [0.42, 0.62, 0.92],
      groundColor: [0.34, 0.30, 0.24],
      fogColor: [0.68, 0.75, 0.85],
      fogDensity: 0.0022,
    };
    normalize(this.env.sunDir);

    this.initShadowTarget();
    this.initNodeTexture();
    this.quadIbo = null;
    this.skid = null;
    this.particles = null;
    this.terrain = null;
    this.props = null;
    this.carGeometry = null;
    this.vehicleInstances = new Map();
  }

  initShadowTarget() {
    const gl = this.gl;
    this.shadowTex = createTexture(gl, {
      width: SHADOW_SIZE, height: SHADOW_SIZE,
      internalFormat: gl.DEPTH_COMPONENT24,
      format: gl.DEPTH_COMPONENT,
      type: gl.UNSIGNED_INT,
      filter: gl.NEAREST, depth: true, compare: true,
    });
    this.shadowFb = createFramebuffer(gl, null, this.shadowTex);
    if (!this.shadowFb.ok) console.warn('shadow framebuffer incomplete');
  }

  initNodeTexture() {
    const gl = this.gl;
    // 5 texels per node: position, R0, R1, R2, rest position
    this.nodeTexWidth = MAX_NODES;
    this.nodeTexHeight = 5;
    this.nodeTexData = new Float32Array(MAX_NODES * 5 * 4);
    this.nodeTex = createTexture(gl, {
      width: MAX_NODES, height: 5,
      internalFormat: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,
      filter: gl.NEAREST,
    });
  }

  /**
   * Size the drawing buffer for a CSS size and a device pixel ratio.  The
   * canvas must get real width/height attributes: without them the browser
   * keeps the 300x150 default and the CSS stretches that tiny image over the
   * whole window, which is why everything looked like blurred blocks.
   */
  resize(cssWidth, cssHeight, dpr = window.devicePixelRatio || 1) {
    // renderScale is lowered automatically when the GPU cannot keep up
    const scale = Math.min(Math.max(dpr, 1), 2) * (this.renderScale || 1);
    const w = Math.max(64, Math.round(cssWidth * scale));
    const h = Math.max(64, Math.round(cssHeight * scale));
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (this.canvas.style) {
      this.canvas.style.width = cssWidth + 'px';
      this.canvas.style.height = cssHeight + 'px';
    }
    this.width = w;
    this.height = h;
    this.dpr = scale;
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    if (this.gl) this.gl.viewport(0, 0, w, h);
  }

  /** fit the canvas to its CSS box; returns true when the size changed */
  fitToDisplay() {
    const rect = this.canvas.getBoundingClientRect
      ? this.canvas.getBoundingClientRect()
      : { width: window.innerWidth, height: window.innerHeight };
    const cssW = Math.max(1, Math.round(rect.width || window.innerWidth));
    const cssH = Math.max(1, Math.round(rect.height || window.innerHeight));
    const prevW = this.width;
    const prevH = this.height;
    this.resize(cssW, cssH);
    return this.width !== prevW || this.height !== prevH;
  }

  // ------------------------------------------------------------------ world
  uploadWorld(world) {
    const gl = this.gl;
    const N = world.gridN;
    const c = world.cell;
    const h0 = -world.halfSize;
    // pos3 nrm3 col3 mat3 (metal, rough, emis) -- keep in sync with the
    // vertex attribute layout below (4 attributes x 3 floats = 12)
    const STRIDE = 12;
    const verts = new Float32Array(N * N * STRIDE);
    const nrm = [0, 1, 0];
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const idx = j * N + i;
        const x = h0 + i * c;
        const z = h0 + j * c;
        const y = world.heights[idx];
        world.normalAt(x, z, nrm);
        const mat = MATERIALS[world.matIds[idx]];
        const o = idx * STRIDE;
        verts[o] = x; verts[o + 1] = y; verts[o + 2] = z;
        verts[o + 3] = nrm[0]; verts[o + 4] = nrm[1]; verts[o + 5] = nrm[2];
        // subtle colour variation so large surfaces do not look flat
        const n1 = world.fbm(x * 0.02, z * 0.02, 2);
        const n2 = world.fbm(x * 0.19 + 4.0, z * 0.19 - 2.0, 2);
        const shade = 0.82 + n1 * 0.28 + n2 * 0.12;
        verts[o + 6] = mat.color[0] * shade;
        verts[o + 7] = mat.color[1] * shade * (1 + n2 * 0.25);
        verts[o + 8] = mat.color[2] * shade;
        verts[o + 9] = mat.metal || 0.0;
        verts[o + 10] = mat.rough;
        verts[o + 11] = 0.0;
      }
    }
    const indices = new Uint32Array((N - 1) * (N - 1) * 6);
    let k = 0;
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const a = j * N + i, b = a + 1, cc = a + N, d = cc + 1;
        indices[k++] = a; indices[k++] = cc; indices[k++] = b;
        indices[k++] = b; indices[k++] = cc; indices[k++] = d;
      }
    }
    if (this.terrain) this.terrain.dispose();
    this.terrain = createMesh(gl,
      [{ loc: 0, size: 3 }, { loc: 1, size: 3 }, { loc: 2, size: 3 }, { loc: 3, size: 3 }],
      verts, indices, null, 'terrain');

    this.uploadProps(world);
  }

  uploadProps(world) {
    const gl = this.gl;
    const pos = [];
    const col = [];
    const nrm = [];
    const mat = [];
    const idx = [];

    const pushQuad = (p0, p1, p2, p3, n, color, m) => {
      const base = pos.length / 3;
      for (const p of [p0, p1, p2, p3]) { pos.push(p[0], p[1], p[2]); nrm.push(n[0], n[1], n[2]); col.push(color[0], color[1], color[2]); mat.push(m[0], m[1], 0); }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };
    const pushBox = (cx, cy, cz, hx, hy, hz, color, m, yaw = 0) => {
      const co = Math.cos(yaw), si = Math.sin(yaw);
      const rot = (x, y, z) => [cx + x * co + z * si, cy + y, cz - x * si + z * co];
      const corners = [
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
      ].map((p) => rot(p[0], p[1], p[2]));
      const faces = [
        [4, 5, 6, 7, [0, 0, 1]], [1, 0, 3, 2, [0, 0, -1]],
        [5, 1, 2, 6, [1, 0, 0]], [0, 4, 7, 3, [-1, 0, 0]],
        [7, 6, 2, 3, [0, 1, 0]], [0, 1, 5, 4, [0, -1, 0]],
      ];
      for (const f of faces) {
        const nn = [f[4][0] * co + f[4][2] * si, f[4][1], -f[4][0] * si + f[4][2] * co];
        pushQuad(corners[f[0]], corners[f[1]], corners[f[2]], corners[f[3]], nn, color, m);
      }
    };

    // ---- road markings along the circuit
    const road = world.road;
    const n = road.pts.length;
    for (let i = 0; i < n; i += 6) {
      const p = road.pts[i];
      const t = road.tangents[i];
      const lx = -t[1], lz = t[0];
      for (const off of [0]) {
        const cx = p[0] + lx * off, cz = p[1] + lz * off;
        const y = world.heightAt(cx, cz) + 0.035;
        const half = 0.09, len = 1.7;
        const a = [cx - lx * half - t[0] * len, y, cz - lz * half - t[1] * len];
        const b = [cx + lx * half - t[0] * len, y, cz + lz * half - t[1] * len];
        const c = [cx + lx * half + t[0] * len, y, cz + lz * half + t[1] * len];
        const d = [cx - lx * half + t[0] * len, y, cz - lz * half + t[1] * len];
        pushQuad(a, b, c, d, [0, 1, 0], [0.62, 0.62, 0.6], [0, 0.7, 0]);
      }
    }

    // ---- barriers along the outside of the road
    for (let i = 0; i < n; i += 26) {
      const p = road.pts[i];
      const t = road.tangents[i];
      const lx = -t[1], lz = t[0];
      const side = -1;
      const off = (road.width * 0.5 + 3.2) * side;
      const cx = p[0] + lx * off, cz = p[1] + lz * off;
      const y = world.heightAt(cx, cz);
      const yaw = Math.atan2(t[0], t[1]);
      pushBox(cx, y + 0.45, cz, 0.28, 0.45, 2.0, [0.62, 0.62, 0.64], [0.6, 0.5], yaw);
      if (i % 52 === 0) {
        pushBox(cx, y + 0.95, cz, 0.3, 0.06, 2.0, [0.85, 0.2, 0.15], [0.1, 0.6], yaw);
      }
    }

    // ---- trees / rocks scattered over the map (deterministic)
    let seed = 99173;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < 420; i++) {
      const x = (rnd() * 2 - 1) * (world.halfSize - 40);
      const z = (rnd() * 2 - 1) * (world.halfSize - 40);
      const near = world.nearestRoad(x, z);
      if (near && near.dist < 26) continue;
      if (Math.hypot(x - 265, z - 285) < 95) continue;
      if (Math.hypot(x, z) < 150) continue;
      const y = world.heightAt(x, z);
      const h = 4.5 + rnd() * 6.5;
      const r = 1.1 + rnd() * 1.4;
      const trunk = [0.19, 0.13, 0.09];
      pushBox(x, y + h * 0.22, z, 0.20, h * 0.24, 0.20, trunk, [0, 0.95]);
      // canopy as a few stacked cones (pyramids)
      const leaf = [0.075 + rnd() * 0.05, 0.16 + rnd() * 0.07, 0.06];
      for (let k = 0; k < 2; k++) {
        const cr = r * (1 - k * 0.35);
        const cy = y + h * (0.42 + k * 0.28);
        const ch = h * 0.4 * (1 - k * 0.2);
        const seg = 6;
        const base = pos.length / 3;
        for (let s = 0; s < seg; s++) {
          const a0 = (s / seg) * Math.PI * 2;
          const a1 = ((s + 1) / seg) * Math.PI * 2;
          const p0 = [x + Math.cos(a0) * cr, cy, z + Math.sin(a0) * cr];
          const p1 = [x + Math.cos(a1) * cr, cy, z + Math.sin(a1) * cr];
          const apex = [x, cy + ch, z];
          for (const p of [p0, p1, apex]) {
            pos.push(p[0], p[1], p[2]);
            const nn = v3.norm([0, 0, 0], [(p[0] - x) / cr, 0.55, (p[2] - z) / cr]);
            nrm.push(nn[0], nn[1], nn[2]);
            col.push(leaf[0], leaf[1], leaf[2]);
            mat.push(0, 0.9, 0);
          }
          idx.push(base + s * 3, base + s * 3 + 1, base + s * 3 + 2);
          void p0; void p1;
        }
      }
    }
    // ---- rocks
    for (let i = 0; i < 90; i++) {
      const x = (rnd() * 2 - 1) * (world.halfSize - 60);
      const z = (rnd() * 2 - 1) * (world.halfSize - 60);
      const near = world.nearestRoad(x, z);
      if (near && near.dist < 14) continue;
      if (Math.hypot(x - 265, z - 285) < 90) continue;
      const y = world.heightAt(x, z);
      const s = 0.5 + rnd() * 1.6;
      const g = 0.28 + rnd() * 0.2;
      pushBox(x, y + s * 0.35, z, s, s * 0.7, s * 0.9, [g, g * 0.98, g * 0.95], [0, 0.85], rnd() * 3);
    }

    // ---- concrete blocks / ramps (matching the physics colliders)
    for (const box of world.colliders) {
      const isRamp = Math.abs(box.axes[1][1]) < 0.995;
      const colr = isRamp ? [0.42, 0.4, 0.36] : [0.55, 0.55, 0.54];
      const R = box.axes;
      // build the 8 corners from the axes
      const corners = [];
      for (let sx = -1; sx <= 1; sx += 2) {
        for (let sy = -1; sy <= 1; sy += 2) {
          for (let sz = -1; sz <= 1; sz += 2) {
            corners.push([
              box.center[0] + R[0][0] * box.half[0] * sx + R[1][0] * box.half[1] * sy + R[2][0] * box.half[2] * sz,
              box.center[1] + R[0][1] * box.half[0] * sx + R[1][1] * box.half[1] * sy + R[2][1] * box.half[2] * sz,
              box.center[2] + R[0][2] * box.half[0] * sx + R[1][2] * box.half[1] * sy + R[2][2] * box.half[2] * sz,
            ]);
          }
        }
      }
      // corner order: index = ((sx>0)<<2)|((sy>0)<<1)|(sz>0)
      const C = (a, b, cc) => corners[(a << 2) | (b << 1) | cc];
      const faces = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [-1, 0, 0]],
        [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, -1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, -1, 0]],
      ];
      for (const f of faces) {
        const nn = [
          R[0][0] * f[4][0] + R[1][0] * f[4][1] + R[2][0] * f[4][2],
          R[0][1] * f[4][0] + R[1][1] * f[4][1] + R[2][1] * f[4][2],
          R[0][2] * f[4][0] + R[1][2] * f[4][1] + R[2][2] * f[4][2],
        ];
        pushQuad(C(...f[0]), C(...f[1]), C(...f[2]), C(...f[3]), nn, colr, [0, 0.8, 0]);
      }
    }

    const STRIDE = 12; // same layout as the terrain: 4 x vec3
    const verts = new Float32Array(pos.length / 3 * STRIDE);
    for (let i = 0; i < pos.length / 3; i++) {
      const o = i * STRIDE;
      verts[o] = pos[i * 3]; verts[o + 1] = pos[i * 3 + 1]; verts[o + 2] = pos[i * 3 + 2];
      verts[o + 3] = nrm[i * 3]; verts[o + 4] = nrm[i * 3 + 1]; verts[o + 5] = nrm[i * 3 + 2];
      verts[o + 6] = col[i * 3]; verts[o + 7] = col[i * 3 + 1]; verts[o + 8] = col[i * 3 + 2];
      verts[o + 9] = mat[i * 3]; verts[o + 10] = mat[i * 3 + 1]; verts[o + 11] = 0;
    }
    if (this.props) this.props.dispose();
    this.props = createMesh(gl,
      [{ loc: 0, size: 3 }, { loc: 1, size: 3 }, { loc: 2, size: 3 }, { loc: 3, size: 3 }],
      verts, new Uint32Array(idx), null, 'props');
  }

  // -------------------------------------------------------------------- cars
  uploadCar(geom) {
    const gl = this.gl;
    const layout = [
      { loc: 0, size: 3 }, { loc: 1, size: 3 }, { loc: 2, size: 3 },
      { loc: 3, size: 3 }, { loc: 4, size: 4 }, { loc: 5, size: 4 },
    ];
    this.carGeometry = { bodies: [], wheels: {}, layout: [] };
    for (const prim of geom.bodies) {
      this.carGeometry.bodies.push({
        mesh: createMesh(gl, layout, prim.data, prim.indices || null, gl.TRIANGLES, 'body:' + prim.name),
        emissive: prim.emissive,
        name: prim.name,
        // kept so a ripped-off panel can be baked into a stand-alone mesh (with
        // its crumple frozen in) the moment it leaves the car
        raw: prim.data,
        stride: prim.stride || 20,
        vertexCount: prim.vertexCount,
        indices: prim.indices || null,
      });
    }
    this.carGeometry.layout = layout;
    for (const w of geom.wheels) {
      const prims = [];
      for (const p of w.prims) {
        prims.push({
          mesh: createMesh(gl, layout, p.data, p.indices || null, gl.TRIANGLES, 'wheel:' + w.name),
          emissive: p.emissive,
        });
      }
      this.carGeometry.wheels[w.name] = prims;
    }
  }

  createVehicleInstance(vehicle) {
    const inst = {
      nodeData: new Float32Array(MAX_NODES * 5 * 4),
      bindRot: new Float32Array(9),
      bindPos: [0, 0, 0],
      wheelMats: new Map(),
    };
    const R0 = vehicle.R0;
    // mat3 from R0 (row major) -> column major for the shader
    inst.bindRot[0] = R0[0]; inst.bindRot[1] = R0[3]; inst.bindRot[2] = R0[6];
    inst.bindRot[3] = R0[1]; inst.bindRot[4] = R0[4]; inst.bindRot[5] = R0[7];
    inst.bindRot[6] = R0[2]; inst.bindRot[7] = R0[5]; inst.bindRot[8] = R0[8];
    inst.bindPos = vehicle.spawnPos.slice();
    for (const w of vehicle.wheels) {
      inst.wheelMats.set(w.name, mat4.create());
    }
    inst.partRot = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    inst.frozen = new Map();
    inst.panels = new Map();
    for (const p of vehicle.panels || []) inst.panels.set(p.name, p);
    this.vehicleInstances.set(vehicle, inst);
    return inst;
  }

  updateVehicleNodes(vehicle, inst) {
    // A respawn (R / P / crash test) swaps the car's spawn pose in place.  The
    // bind transform lives in the renderer instance, so it has to be refreshed
    // or the skinned body would keep drawing at the previous spawn point.
    const p = vehicle.spawnPos;
    if (inst.bindPos[0] !== p[0] || inst.bindPos[1] !== p[1] || inst.bindPos[2] !== p[2]) {
      inst.bindPos = p.slice();
      const R0 = vehicle.R0;
      inst.bindRot[0] = R0[0]; inst.bindRot[1] = R0[3]; inst.bindRot[2] = R0[6];
      inst.bindRot[3] = R0[1]; inst.bindRot[4] = R0[4]; inst.bindRot[5] = R0[7];
      inst.bindRot[6] = R0[2]; inst.bindRot[7] = R0[5]; inst.bindRot[8] = R0[8];
    }
    if (inst.panels.size !== (vehicle.panels || []).length) {
      inst.panels = new Map();
      for (const p of vehicle.panels || []) inst.panels.set(p.name, p);
    }
    packNodeData(vehicle, inst.nodeData);
    inst.dirty = true;
  }

  uploadNodeTexture(vehicle, inst) {
    const gl = this.gl;
    // Bind on the node-texture unit explicitly.  Relying on the active unit was
    // a real bug: the main pass leaves TEXTURE1 (the shadow map) active, so the
    // node data was uploaded *into the shadow map* and the real node texture
    // kept its zero initial content -- every skinned vertex then collapsed to a
    // point and the car body was invisible.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.nodeTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_NODES, 5, gl.RGBA, gl.FLOAT, inst.nodeData);
    if (this.nodeTexError === undefined) {
      // one-shot sanity check that the GPUs accepts an RGBA32F float upload.
      // Drain pending errors first so an unrelated earlier one is not blamed.
      let guard = 0;
      while (gl.getError() !== 0 && guard++ < 8) { /* drain */ }
      gl.bindTexture(gl.TEXTURE_2D, this.nodeTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_NODES, 5, gl.RGBA, gl.FLOAT, inst.nodeData);
      const err = gl.getError();
      this.nodeTexError = err || 0;
      if (err) {
        console.warn('soft-body node texture upload rejected (0x' + err.toString(16) +
          '), falling back to rigid body rendering');
      }
    }
  }

  /**
   * Rigid placement of the car mesh for the current pose.
   *
   * M = T(center) * R * T(-restCentroid) * bindTransform, which is exactly what
   * the skinned path produces when every node rotation equals the body
   * rotation.  Used as a fallback when the float node texture is unusable, so a
   * broken skinning path degrades to a rigid-but-visible car instead of none.
   */
  rigidBodyMatrix(vehicle, out) {
    const R = vehicle.R;
    const R0 = vehicle.R0;
    const c = vehicle.center;
    const rc = vehicle.restCentroid;
    const p0 = vehicle.spawnPos;
    const m = out || mat4.create();
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        let v = 0;
        for (let k = 0; k < 3; k++) v += R[row * 3 + k] * R0[k * 3 + col];
        m[col * 4 + row] = v;
      }
      let t = c[row] - rc[0] * R[row * 3] - rc[1] * R[row * 3 + 1] - rc[2] * R[row * 3 + 2];
      for (let k = 0; k < 3; k++) t += R[row * 3 + k] * p0[k];
      m[12 + row] = t;
    }
    m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
    return m;
  }

  // ------------------------------------------------------------- part hinges
  /** world position of a hinge pivot for the current pose */
  panelPivot(vehicle, p, out) {
    const R = vehicle.R;
    const rc = vehicle.restCentroid;
    const dx = p.pivotRest[0] - rc[0], dy = p.pivotRest[1] - rc[1], dz = p.pivotRest[2] - rc[2];
    const c = vehicle.center;
    out[0] = c[0] + R[0] * dx + R[1] * dy + R[2] * dz;
    out[1] = c[1] + R[3] * dx + R[4] * dy + R[5] * dz;
    out[2] = c[2] + R[6] * dx + R[7] * dy + R[8] * dz;
    return out;
  }

  /** rotation (column-major mat3) of a hinged panel for the current pose */
  panelRotation(vehicle, p, out) {
    const R = vehicle.R;
    const a = p.hinge.axis;
    // the hinge axis in world space: the body rotation applied to the car-space axis
    const ax = R[0] * a[0] + R[1] * a[1] + R[2] * a[2];
    const ay = R[3] * a[0] + R[4] * a[1] + R[5] * a[2];
    const az = R[6] * a[0] + R[7] * a[1] + R[8] * a[2];
    const th = p.angle || 0;
    const c = Math.cos(th), s = Math.sin(th), t = 1 - c;
    // Rodrigues, column-major for the shader
    out[0] = c + ax * ax * t;
    out[1] = ay * ax * t + az * s;
    out[2] = az * ax * t - ay * s;
    out[3] = ax * ay * t - az * s;
    out[4] = c + ay * ay * t;
    out[5] = az * ay * t + ax * s;
    out[6] = ax * az * t + ay * s;
    out[7] = ay * az * t - ax * s;
    out[8] = c + az * az * t;
    return out;
  }

  /**
   * Bake a panel that has just been ripped off into its own mesh.
   *
   * The panel keeps the crumple it had at that instant (skinning is evaluated
   * here on the CPU and written into the vertex buffer, with the per-vertex
   * damage riding in the free joint/weight slots), so it flies away looking
   * like a mangled door instead of snapping back to a clean, straight one.
   */
  bakePanel(vehicle, inst, p) {
    const gl = this.gl;
    if (p.bakedMeshes && p.bakedMeshes.length) {
      for (const m of p.bakedMeshes) {
        if (m.mesh && m.mesh.vao) gl.deleteVertexArray(m.mesh.vao);
      }
      p.bakedMeshes = null;
    }
    const layout = this.carGeometry.layout;
    const out = [];
    const pivot = [0, 0, 0];
    this.panelPivot(vehicle, p, pivot);
    const rot = new Float32Array(9);
    this.panelRotation(vehicle, p, rot);
    // part transform: p' = pivot + R * (p - pivot)
    let cx = 0, cy = 0, cz = 0, total = 0;
    for (const prim of this.carGeometry.bodies) {
      if (prim.name !== p.name) continue;
      const src = prim.raw;
      const stride = prim.stride;
      const nv = src.length / stride;
      const baked = new Float32Array(src);
      for (let v = 0; v < nv; v++) {
        const o = v * stride;
        // ---- skin one vertex exactly like MESH_VS does
        let px = 0, py = 0, pz = 0;
        let dmg = 0, crumple = 0;
        const d0 = [0, 0, 0], d1 = [0, 0, 0], d2 = [0, 0, 0];
        const bind = [
          inst.bindRot[0] * src[o] + inst.bindRot[3] * src[o + 1] + inst.bindRot[6] * src[o + 2] + inst.bindPos[0],
          inst.bindRot[1] * src[o] + inst.bindRot[4] * src[o + 1] + inst.bindRot[7] * src[o + 2] + inst.bindPos[1],
          inst.bindRot[2] * src[o] + inst.bindRot[5] * src[o + 1] + inst.bindRot[8] * src[o + 2] + inst.bindPos[2],
        ];
        for (let j = 0; j < 4; j++) {
          const w = src[o + 16 + j];
          if (w <= 0) continue;
          const node = src[o + 12 + j] | 0;
          const base = MAX_NODES * 4;
          const r0 = node * 4, r1 = base + node * 4, r2 = base * 2 + node * 4, r3 = base * 3 + node * 4;
          const r4 = base * 4 + node * 4;
          const npx = inst.nodeData[r0], npy = inst.nodeData[r0 + 1], npz = inst.nodeData[r0 + 2];
          const c0x = inst.nodeData[r1], c0y = inst.nodeData[r1 + 1], c0z = inst.nodeData[r1 + 2];
          const c1x = inst.nodeData[r2], c1y = inst.nodeData[r2 + 1], c1z = inst.nodeData[r2 + 2];
          const c2x = inst.nodeData[r3], c2y = inst.nodeData[r3 + 1], c2z = inst.nodeData[r3 + 2];
          const rx = bind[0] - inst.nodeData[r4], ry = bind[1] - inst.nodeData[r4 + 1], rz = bind[2] - inst.nodeData[r4 + 2];
          const sx = c0x * rx + c1x * ry + c2x * rz;
          const sy = c0y * rx + c1y * ry + c2y * rz;
          const sz = c0z * rx + c1z * ry + c2z * rz;
          px += w * (npx + sx); py += w * (npy + sy); pz += w * (npz + sz);
          d0[0] += w * c0x; d0[1] += w * c0y; d0[2] += w * c0z;
          d1[0] += w * c1x; d1[1] += w * c1y; d1[2] += w * c1z;
          d2[0] += w * c2x; d2[1] += w * c2y; d2[2] += w * c2z;
          dmg += w * inst.nodeData[r0 + 3];
          crumple += w * inst.nodeData[r3 + 3];
        }
        // hinge rotation about the pivot
        const rxp = px - pivot[0], ryp = py - pivot[1], rzp = pz - pivot[2];
        px = pivot[0] + rot[0] * rxp + rot[3] * ryp + rot[6] * rzp;
        py = pivot[1] + rot[1] * rxp + rot[4] * ryp + rot[7] * rzp;
        pz = pivot[2] + rot[2] * rxp + rot[5] * ryp + rot[8] * rzp;
        // normal: (accD * bindRot) * n, then the same hinge rotation
        const bx = inst.bindRot[0] * src[o + 3] + inst.bindRot[3] * src[o + 4] + inst.bindRot[6] * src[o + 5];
        const by = inst.bindRot[1] * src[o + 3] + inst.bindRot[4] * src[o + 4] + inst.bindRot[7] * src[o + 5];
        const bz = inst.bindRot[2] * src[o + 3] + inst.bindRot[5] * src[o + 4] + inst.bindRot[8] * src[o + 5];
        let nx = d0[0] * bx + d1[0] * by + d2[0] * bz;
        let ny = d0[1] * bx + d1[1] * by + d2[1] * bz;
        let nz = d0[2] * bx + d1[2] * by + d2[2] * bz;
        const hx = rot[0] * nx + rot[3] * ny + rot[6] * nz;
        const hy = rot[1] * nx + rot[4] * ny + rot[7] * nz;
        const hz = rot[2] * nx + rot[5] * ny + rot[8] * nz;
        const nl = Math.hypot(hx, hy, hz) || 1;
        baked[o] = px; baked[o + 1] = py; baked[o + 2] = pz;
        baked[o + 3] = hx / nl; baked[o + 4] = hy / nl; baked[o + 5] = hz / nl;
        // junk the skinning weights: a loose panel is drawn rigidly
        baked[o + 12] = clamp(dmg, 0, 1);      // aJoints.x carries the damage now
        baked[o + 13] = clamp(crumple, 0, 1);
        baked[o + 14] = 0; baked[o + 15] = 0;
        baked[o + 16] = 0; baked[o + 17] = 0; baked[o + 18] = 0; baked[o + 19] = 0;
        cx += px; cy += py; cz += pz; total++;
      }
      out.push({
        mesh: createMesh(gl, layout, baked, prim.indices || null, gl.TRIANGLES, 'loose:' + p.name),
        emissive: prim.emissive,
        indices: prim.indices || null,
        data: baked,
      });
    }
    if (!total) return null;
    p.frozen = true;
    p.bakedCentroid = [cx / total, cy / total, cz / total];
    p.bakedMeshes = out;
    return out;
  }

  /** the model matrix of a panel that is no longer attached to the car */
  loosePanelMatrix(p, out) {
    const m = out || mat4.create();
    quatToMat4(p.quat || [0, 0, 0, 1], m);
    const c0 = p.bakedCentroid || [0, 0, 0];
    // rotate about the baked centroid, then move it to where the body is now
    for (let i = 0; i < 3; i++) {
      m[12 + i] = p.pos[i] - (m[i] * c0[0] + m[4 + i] * c0[1] + m[8 + i] * c0[2]);
    }
    return m;
  }

  updateWheelMatrices(vehicle, inst) {
    const basis = vehicle.basis;
    for (const w of vehicle.wheels) {
      const m = inst.wheelMats.get(w.name);
      if (!m) continue;
      if (w.detached) {
        // a wheel that came off keeps flying: place it by its own free-body
        // state (position + tumbling quaternion) instead of hiding it
        quatToMat4(w.quat || [0, 0, 0, 1], m);
        const c = Math.cos(w.spin || 0), sn = Math.sin(w.spin || 0);
        const r0 = [m[0], m[1], m[2]], r1 = [m[4], m[5], m[6]], r2 = [m[8], m[9], m[10]];
        for (let k = 0; k < 3; k++) {
          m[4 + k] = r1[k] * c + r2[k] * sn;
          m[8 + k] = r2[k] * c - r1[k] * sn;
        }
        m[12] = w.pos[0]; m[13] = w.pos[1]; m[14] = w.pos[2];
        continue;
      }
      const fwd = [basis.z[0], basis.z[1], basis.z[2]];
      const left = [basis.x[0], basis.x[1], basis.x[2]];
      const up = [basis.y[0], basis.y[1], basis.y[2]];
      // steer: rotate around up
      const cs = Math.cos(w.steer), sn = Math.sin(w.steer);
      const f2 = [fwd[0] * cs + left[0] * sn, fwd[1] * cs + left[1] * sn, fwd[2] * cs + left[2] * sn];
      const l2 = [left[0] * cs - fwd[0] * sn, left[1] * cs - fwd[1] * sn, left[2] * cs - fwd[2] * sn];
      // spin around the wheel axle (local X = left direction)
      const spin = w.spin || 0;
      // axle axis = local x; the wheel mesh spins about its x axis
      const c2 = Math.cos(spin), s2 = Math.sin(spin);
      // final basis columns: X = axle (left), Y = up', Z = forward'
      const Y0 = [up[0], up[1], up[2]];
      const Z0 = f2;
      // rotate Y0/Z0 about the axle (l2) by spin
      const rotAbout = (v, axis, c, s) => {
        const d = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
        return [
          v[0] * c + (axis[1] * v[2] - axis[2] * v[1]) * s + axis[0] * d * (1 - c),
          v[1] * c + (axis[2] * v[0] - axis[0] * v[2]) * s + axis[1] * d * (1 - c),
          v[2] * c + (axis[0] * v[1] - axis[1] * v[0]) * s + axis[2] * d * (1 - c),
        ];
      };
      const Y1 = rotAbout(Y0, l2, c2, s2);
      const Z1 = rotAbout(Z0, l2, c2, s2);
      // a bent rim / flat tyre squats: squash the wheel along the body up axis
      const flat = 1 - 0.22 * clamp(w.rimDamage || 0, 0, 1) * (w.contact ? 1 : 0.35);
      const Y2 = [Y1[0] * flat, Y1[1] * flat, Y1[2] * flat];
      m[0] = l2[0]; m[1] = l2[1]; m[2] = l2[2]; m[3] = 0;
      m[4] = Y2[0]; m[5] = Y2[1]; m[6] = Y2[2]; m[7] = 0;
      m[8] = Z1[0]; m[9] = Z1[1]; m[10] = Z1[2]; m[11] = 0;
      m[12] = w.pos[0] - Y1[0] * (1 - flat) * 0.32;
      m[13] = w.pos[1] - Y1[1] * (1 - flat) * 0.32;
      m[14] = w.pos[2] - Y1[2] * (1 - flat) * 0.32;
      m[15] = 1;
    }
  }

  // --------------------------------------------------------------- particles
  initParticles(max = 4000) {
    const gl = this.gl;
    this.particleMax = max;
    this.particleData = new Float32Array(max * 7); // pos3, color3, size
    this.particleVao = gl.createVertexArray();
    gl.bindVertexArray(this.particleVao);
    this.particleVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.particleData.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);
    gl.bindVertexArray(null);
  }

  initSkid(max = 6000) {
    const gl = this.gl;
    this.skidMax = max;
    this.skidData = new Float32Array(max * 4); // pos3, alpha
    this.skidVao = gl.createVertexArray();
    gl.bindVertexArray(this.skidVao);
    this.skidVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skidVbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.skidData.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 12);
    gl.bindVertexArray(null);
  }

  // -------------------------------------------------------------------- draw
  setEnv(env) { Object.assign(this.env, env); }

  render(state) {
    const gl = this.gl;
    const { camera, world, vehicles, fx } = state;
    const aspect = this.width / this.height;
    const near = 0.25;
    const far = 900;
    mat4.perspective(this.tmpMat, camera.fov * Math.PI / 180, aspect, near, far);
    mat4.multiply(this.viewProj, this.tmpMat, camera.view);

    // ---------------- shadow pass
    const sun = this.env.sunDir;
    const focus = camera.target;
    const eye = [focus[0] + sun[0] * 160, focus[1] + sun[1] * 160, focus[2] + sun[2] * 160];
    const lightView = mat4.lookAt(mat4.create(), eye, focus, [0, 1, 0]);
    const extent = 78;
    const lightProj = mat4.ortho(mat4.create(), -extent, extent, -extent, extent, 20, 360);
    mat4.multiply(this.lightViewProj, lightProj, lightView);
    // texture matrix for the shadow lookup
    const bias = new Float32Array([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0.5, 0.5, 0.5, 1]);
    mat4.multiply(this.shadowMat, bias, this.lightViewProj);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFb.fb);
    gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.useProgram(this.progDepth.program);
    const uDepth = this.progDepth.uniforms;
    gl.uniformMatrix4fv(uDepth.uLightViewProj, false, this.lightViewProj);
    gl.uniform1i(uDepth.uSkinned, 0);
    mat4.identity(this.model);
    gl.uniformMatrix4fv(uDepth.uModel, false, this.model);
    gl.bindVertexArray(this.terrain.vao);
    gl.drawElements(this.terrain.mode, this.terrain.count, this.terrain.indexType, 0);
    if (this.props) {
      gl.bindVertexArray(this.props.vao);
      gl.drawElements(this.props.mode, this.props.count, this.props.indexType, 0);
    }
    if (this.carGeometry) {
      gl.uniform1i(uDepth.uSkinned, 1);
      const pivotTmp = [0, 0, 0];
      for (const vehicle of vehicles) {
        const inst = this.vehicleInstances.get(vehicle);
        if (!inst) continue;
        this.updateVehicleNodes(vehicle, inst);
        this.uploadNodeTexture(vehicle, inst);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.nodeTex);
        gl.uniform1i(uDepth.uNodeTex, 0);
        gl.uniform1i(uDepth.uNodeCount, vehicle.nodeCount);
        gl.uniform3f(uDepth.uPartOffset, 0, 0, 0);
        for (const prim of this.carGeometry.bodies) {
          const panel = inst.panels.get(prim.name);
          if (panel) {
            if (panel.detached) continue;
            this.panelPivot(vehicle, panel, pivotTmp);
            this.panelRotation(vehicle, panel, inst.partRot);
            gl.uniformMatrix3fv(uDepth.uPartRot, false, inst.partRot);
            gl.uniform3f(uDepth.uPartPivot, pivotTmp[0], pivotTmp[1], pivotTmp[2]);
          } else {
            gl.uniformMatrix3fv(uDepth.uPartRot, false, this.identity3);
            gl.uniform3f(uDepth.uPartPivot, 0, 0, 0);
          }
          gl.bindVertexArray(prim.mesh.vao);
          gl.drawElements(prim.mesh.mode, prim.mesh.count, prim.mesh.indexType, 0);
        }
      }
      gl.uniformMatrix3fv(uDepth.uPartRot, false, this.identity3);
      gl.uniform3f(uDepth.uPartPivot, 0, 0, 0);
      gl.uniform3f(uDepth.uPartOffset, 0, 0, 0);
    }

    // ---------------- main pass
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0.55, 0.65, 0.78, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // sky
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.useProgram(this.progSky.program);
    const invVP = mat4.invert(mat4.create(), this.viewProj);
    gl.uniformMatrix4fv(this.progSky.uniforms.uInvViewProj, false, invVP);
    gl.uniform3fv(this.progSky.uniforms.uCamPos, camera.position);
    gl.uniform3fv(this.progSky.uniforms.uSunDir, this.env.sunDir);
    gl.uniform3fv(this.progSky.uniforms.uSunColor, this.env.sunColor);
    gl.uniform3fv(this.progSky.uniforms.uSkyColor, this.env.skyColor);
    gl.uniform3fv(this.progSky.uniforms.uGroundColor, this.env.groundColor);
    gl.bindVertexArray(this.skyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);

    // scene
    gl.useProgram(this.progMesh.program);
    const u = this.progMesh.uniforms;
    gl.uniformMatrix4fv(u.uViewProj, false, this.viewProj);
    gl.uniformMatrix4fv(u.uShadowMat, false, this.shadowMat);
    gl.uniform3fv(u.uCamPos, camera.position);
    gl.uniform3fv(u.uSunDir, this.env.sunDir);
    gl.uniform3fv(u.uSunColor, this.env.sunColor);
    gl.uniform3fv(u.uSkyColor, this.env.skyColor);
    gl.uniform3fv(u.uGroundColor, this.env.groundColor);
    gl.uniform3fv(u.uFogColor, this.env.fogColor);
    gl.uniform1f(u.uFogDensity, this.env.fogDensity);
    gl.uniform1i(u.uShadows, 1);
    gl.uniform1f(u.uAlpha, 1.0);
    gl.uniform3f(u.uEmissive, 0, 0, 0);
    if (u.uTint) gl.uniform3f(u.uTint, 1, 1, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.uniform1i(u.uShadowMap, 1);
    mat4.identity(this.model);
    gl.uniformMatrix4fv(u.uModel, false, this.model);
    this.normalMat.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    gl.uniformMatrix3fv(u.uNormalMat, false, this.normalMat);
    gl.uniform1i(u.uSkinned, 0);

    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.bindVertexArray(this.terrain.vao);
    gl.drawElements(this.terrain.mode, this.terrain.count, this.terrain.indexType, 0);
    if (this.props) {
      gl.bindVertexArray(this.props.vao);
      gl.drawElements(this.props.mode, this.props.count, this.props.indexType, 0);
    }

    // cars
    if (this.carGeometry) {
      // if the node texture cannot be used on this GPU, fall back to drawing the
      // car rigidly (it will not deform, but it will be visible)
      const skinnable = !this.nodeTexError;
      gl.uniform1i(u.uSkinned, skinnable ? 1 : 0);
      for (const vehicle of vehicles) {
        const inst = this.vehicleInstances.get(vehicle);
        if (!inst) continue;
        this.updateVehicleNodes(vehicle, inst);
        this.updateWheelMatrices(vehicle, inst);
        this.uploadNodeTexture(vehicle, inst);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.nodeTex);
        gl.uniform1i(u.uNodeTex, 0);
        // the shadow map lives on unit 1 and must stay bound there
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
        gl.uniform1i(u.uNodeCount, vehicle.nodeCount);
        if (u.uTint) gl.uniform3fv(u.uTint, vehicle.color || [1, 1, 1]);
        if (!skinnable) gl.uniformMatrix4fv(u.uModel, false, this.rigidBodyMatrix(vehicle));
        gl.uniformMatrix3fv(u.uBindRot, false, inst.bindRot);
        gl.uniform3fv(u.uBindPos, inst.bindPos);

        // body.  Panels (doors, hood, trunk, bumpers) carry their own hinge
        // rotation; a panel that has been ripped off is not drawn here at all --
        // it comes back later as a loose body with its crumple baked in.
        gl.disable(gl.CULL_FACE);
        gl.uniform1i(u.uFrozen, 0);
        const identityRot = this.identity3;
        const pivotTmp = [0, 0, 0];
        for (const prim of this.carGeometry.bodies) {
          if (prim.emissive) continue;
          const panel = inst.panels.get(prim.name);
          if (panel) {
            if (panel.detached) continue;
            this.panelPivot(vehicle, panel, pivotTmp);
            this.panelRotation(vehicle, panel, inst.partRot);
            gl.uniformMatrix3fv(u.uPartRot, false, inst.partRot);
            gl.uniform3f(u.uPartPivot, pivotTmp[0], pivotTmp[1], pivotTmp[2]);
            gl.uniform3f(u.uPartOffset, 0, 0, 0);
          } else {
            gl.uniformMatrix3fv(u.uPartRot, false, identityRot);
            gl.uniform3f(u.uPartPivot, 0, 0, 0);
            gl.uniform3f(u.uPartOffset, 0, 0, 0);
          }
          gl.bindVertexArray(prim.mesh.vao);
          gl.drawElements(prim.mesh.mode, prim.mesh.count, prim.mesh.indexType, 0);
        }
        // lights
        // smashed lamp clusters go dark (the shader also fades them locally)
        const lights = 1 - 0.75 * (vehicle.lightsOut || 0);
        gl.uniform3f(u.uEmissive, 0.9 * lights, 0.85 * lights, 0.72 * lights);
        for (const prim of this.carGeometry.bodies) {
          if (!prim.emissive) continue;
          const panel = inst.panels.get(prim.name);
          if (panel) {
            if (panel.detached) continue;
            this.panelPivot(vehicle, panel, pivotTmp);
            this.panelRotation(vehicle, panel, inst.partRot);
            gl.uniformMatrix3fv(u.uPartRot, false, inst.partRot);
            gl.uniform3f(u.uPartPivot, pivotTmp[0], pivotTmp[1], pivotTmp[2]);
          } else {
            gl.uniformMatrix3fv(u.uPartRot, false, identityRot);
            gl.uniform3f(u.uPartPivot, 0, 0, 0);
          }
          gl.bindVertexArray(prim.mesh.vao);
          gl.drawElements(prim.mesh.mode, prim.mesh.count, prim.mesh.indexType, 0);
        }
        gl.uniform3f(u.uEmissive, 0, 0, 0);
        gl.uniformMatrix3fv(u.uPartRot, false, identityRot);
        gl.uniform3f(u.uPartPivot, 0, 0, 0);
        gl.uniform3f(u.uPartOffset, 0, 0, 0);
        gl.uniform1i(u.uSkinned, 0);

        // ---- panels that came off: rigid bodies with their damage frozen in
        if (vehicle.detachedPanels && vehicle.detachedPanels.length) {
          gl.uniform1i(u.uFrozen, 1);
          for (const p of vehicle.detachedPanels) {
            if (!p.bakedMeshes) this.bakePanel(vehicle, inst, p);
            if (!p.bakedMeshes) continue;
            this.loosePanelMatrix(p, this.model);
            gl.uniformMatrix4fv(u.uModel, false, this.model);
            for (const baked of p.bakedMeshes) {
              gl.bindVertexArray(baked.mesh.vao);
              gl.drawElements(baked.mesh.mode, baked.mesh.count, baked.mesh.indexType, 0);
            }
          }
          gl.uniform1i(u.uFrozen, 0);
          gl.uniformMatrix4fv(u.uModel, false, this.model);
          gl.uniform1i(u.uSkinned, skinnable ? 1 : 0);
        }

        // wheels (rigid)
        gl.enable(gl.CULL_FACE);
        for (const w of vehicle.wheels) {
          const wm = inst.wheelMats.get(w.name);
          if (!wm) continue;
          gl.uniformMatrix4fv(u.uModel, false, wm);
          const prims = this.carGeometry.wheels['Wheel_' + w.name];
          if (!prims) continue;
          for (const prim of prims) {
            gl.bindVertexArray(prim.mesh.vao);
            gl.drawElements(prim.mesh.mode, prim.mesh.count, prim.mesh.indexType, 0);
          }
        }
        gl.uniformMatrix4fv(u.uModel, false, this.model);
        gl.uniform1i(u.uSkinned, skinnable ? 1 : 0);
      }
      gl.uniform1i(u.uSkinned, 0);
    }

    // ---------------- fx: skid marks (blended, no depth write)
    if (fx && fx.skid && fx.skid.count > 0) {
      if (!this.skid) this.initSkid();
      const gl2 = gl;
      gl2.useProgram(this.progSkid.program);
      gl2.uniformMatrix4fv(this.progSkid.uniforms.uViewProj, false, this.viewProj);
      gl2.bindVertexArray(this.skidVao);
      gl2.bindBuffer(gl2.ARRAY_BUFFER, this.skidVbo);
      gl2.bufferSubData(gl2.ARRAY_BUFFER, 0, fx.skid.data.subarray(0, fx.skid.count * 4));
      gl2.enable(gl2.BLEND);
      gl2.blendFunc(gl2.SRC_ALPHA, gl2.ONE_MINUS_SRC_ALPHA);
      gl2.depthMask(false);
      gl2.disable(gl2.CULL_FACE);
      gl2.drawArrays(gl2.TRIANGLES, 0, fx.skid.count);
      gl2.depthMask(true);
      gl2.disable(gl2.BLEND);
    }

    // ---------------- fx: particles
    if (fx && fx.particles && fx.particles.count > 0) {
      if (!this.particles) this.initParticles();
      gl.useProgram(this.progParticle.program);
      gl.uniformMatrix4fv(this.progParticle.uniforms.uViewProj, false, this.viewProj);
      gl.uniform3fv(this.progParticle.uniforms.uCamPos, camera.position);
      gl.uniform1f(this.progParticle.uniforms.uPixelScale, this.height * 0.5);
      gl.bindVertexArray(this.particleVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.particleVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, fx.particles.data.subarray(0, fx.particles.count * 7));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.drawArrays(gl.POINTS, 0, fx.particles.count);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(null);
    }

    // ---------------- debug: deformed cage
    if (state.debugCage && state.cageLines) {
      gl.useProgram(this.progSkid.program);
      gl.uniformMatrix4fv(this.progSkid.uniforms.uViewProj, false, this.viewProj);
      if (!this.cageVao) {
        this.cageVao = gl.createVertexArray();
        this.cageVbo = gl.createBuffer();
        gl.bindVertexArray(this.cageVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cageVbo);
        gl.bufferData(gl.ARRAY_BUFFER, 3 * 4 * 40000, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 12);
      }
      const lines = state.cageLines;
      gl.bindVertexArray(this.cageVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cageVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, lines.data.subarray(0, lines.count * 4));
      gl.drawArrays(gl.LINES, 0, lines.count);
    }
    gl.bindVertexArray(null);
  }
}

function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  v[0] /= l; v[1] /= l; v[2] /= l;
}

/** rigid transform from a unit quaternion (x, y, z, w) -- used for loose wheels */
export function quatToMat4(q, out) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const m = out || mat4.create();
  m[0] = 1 - 2 * (yy + zz); m[1] = 2 * (xy + wz); m[2] = 2 * (xz - wy); m[3] = 0;
  m[4] = 2 * (xy - wz); m[5] = 1 - 2 * (xx + zz); m[6] = 2 * (yz + wx); m[7] = 0;
  m[8] = 2 * (xz + wy); m[9] = 2 * (yz - wx); m[10] = 1 - 2 * (xx + yy); m[11] = 0;
  m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
  return m;
}
