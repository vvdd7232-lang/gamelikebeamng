// Tiny WebGL2 helper layer.

export function createContext(canvas) {
  const opts = {
    alpha: false,
    antialias: true,
    depth: true,
    stencil: false,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false,
    failIfMajorPerformanceCaveat: false,
  };
  const gl = canvas.getContext('webgl2', opts);
  if (!gl) return null;
  gl.getExtension('EXT_color_buffer_float');
  gl.getExtension('OES_texture_float_linear');
  return gl;
}

export function compileShader(gl, type, src, name) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    const numbered = src.split('\n').map((l, i) => String(i + 1).padStart(3) + '| ' + l).join('\n');
    throw new Error(`shader compile failed (${name}):\n${log}\n${numbered}`);
  }
  return sh;
}

export function createProgram(gl, vsSrc, fsSrc, name = 'program') {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc, name + '.vert');
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc, name + '.frag');
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`program link failed (${name}): ${gl.getProgramInfoLog(p)}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  const uniforms = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    const base = info.name.replace(/\[0\]$/, '');
    uniforms[base] = gl.getUniformLocation(p, info.name);
  }
  return { program: p, uniforms, name };
}

export function createMesh(gl, layout, data, indices, mode, name = 'mesh') {
  // layout: [{loc, size, stride?, offset?, type?}]; data: interleaved typed array
  const strideFloats = layout.reduce((s, a) => s + a.size, 0);
  // The interleaved buffer must hold exactly `strideFloats` per vertex; a
  // mismatch silently shifts every attribute by a few floats and turns the
  // geometry into giant random triangles (which is exactly how the terrain
  // looked before this check existed).
  if (data.length % strideFloats !== 0) {
    throw new Error(
      `mesh "${name}": ${data.length} floats is not a multiple of the ${strideFloats}-float ` +
      `vertex stride implied by the attribute layout`);
  }
  const vertexCount = data.length / strideFloats;
  if (indices && indices.length) {
    let maxIndex = 0;
    for (let i = 0; i < indices.length; i++) if (indices[i] > maxIndex) maxIndex = indices[i];
    if (maxIndex >= vertexCount) {
      throw new Error(`mesh "${name}": index ${maxIndex} out of range for ${vertexCount} vertices`);
    }
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const stride = strideFloats * 4;
  let off = 0;
  for (const a of layout) {
    gl.enableVertexAttribArray(a.loc);
    gl.vertexAttribPointer(a.loc, a.size, a.type || gl.FLOAT, false, stride, off);
    off += a.size * 4;
  }
  let ibo = null;
  let count = 0;
  let indexType = gl.UNSIGNED_INT;
  if (indices) {
    ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    count = indices.length;
    indexType = indices instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;
  } else {
    count = vertexCount;
  }
  gl.bindVertexArray(null);
  return {
    name, vao, vbo, ibo, count, indexType, vertexCount, strideFloats,
    mode: mode || gl.TRIANGLES,
    dispose() {
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vbo);
      if (ibo) gl.deleteBuffer(ibo);
    },
  };
}

export function createTexture(gl, opts) {
  const {
    width, height, internalFormat = gl.RGBA8, format = gl.RGBA,
    type = gl.UNSIGNED_BYTE, filter = gl.LINEAR, wrap = gl.CLAMP_TO_EDGE,
    data = null, depth = false, compare = false,
  } = opts;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  if (compare) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
  }
  if (depth) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }
  return tex;
}

export function createFramebuffer(gl, colorTex, depthTex) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  if (colorTex) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);
  if (depthTex) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
  if (!colorTex) {
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
  }
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fb, status, ok: status === gl.FRAMEBUFFER_COMPLETE };
}

export function fullscreenTriangle(gl) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}
