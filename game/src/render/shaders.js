// GLSL ES 3.00 shader sources.

export const COMMON_LIGHT = /* glsl */`
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform int uShadows;
uniform highp sampler2DShadow uShadowMap;
uniform mat4 uShadowMat;

vec3 skyRadiance(vec3 d) {
  float y = clamp(d.y, -1.0, 1.0);
  float t = pow(clamp(y * 0.5 + 0.5, 0.0, 1.0), 1.35);
  vec3 col = mix(uSkyColor * 0.45, uSkyColor, t);
  col = mix(col, uGroundColor * 0.5, clamp(-y * 2.0, 0.0, 1.0));
  float sd = max(dot(d, uSunDir), 0.0);
  col += uSunColor * pow(sd, 900.0) * 6.0;
  col += uSunColor * pow(sd, 14.0) * 0.35;
  return col;
}

float shadowFactor(vec3 world) {
  if (uShadows == 0) return 1.0;
  vec4 sc = uShadowMat * vec4(world, 1.0);
  vec3 proj = sc.xyz / sc.w * 0.5 + 0.5;
  if (proj.x < 0.005 || proj.x > 0.995 || proj.y < 0.005 || proj.y > 0.995 || proj.z > 1.0) {
    return 1.0;
  }
  float bias = 0.0012;
  float sum = 0.0;
  vec2 texel = vec2(1.0 / 2048.0);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      sum += texture(uShadowMap, vec3(proj.xy + vec2(float(x), float(y)) * texel, proj.z - bias));
    }
  }
  return sum / 9.0;
}
`;

export const SKY_VS = /* glsl */`#version 300 es
layout(location = 0) in vec2 aPos;
out vec2 vNdc;
void main() {
  vNdc = aPos;
  gl_Position = vec4(aPos, 0.9999, 1.0);
}
`;

export const SKY_FS = /* glsl */`#version 300 es
precision highp float;
in vec2 vNdc;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
out vec4 fragColor;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
}

void main() {
  vec4 p = uInvViewProj * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(p.xyz / p.w - uCamPos);

  float h = clamp(dir.y, -1.0, 1.0);
  float t = pow(clamp(h * 0.5 + 0.5, 0.0, 1.0), 1.35);
  vec3 col = mix(uSkyColor * 0.42, uSkyColor, t);
  col = mix(col, uGroundColor * 0.55, clamp(-h * 2.2, 0.0, 1.0));

  float sd = max(dot(dir, uSunDir), 0.0);
  col += uSunColor * pow(sd, 1200.0) * 8.0;
  col += uSunColor * pow(sd, 20.0) * 0.30;

  // clouds on the upper hemisphere
  if (dir.y > 0.015) {
    vec2 uv = dir.xz / (dir.y + 0.12) * 0.9;
    float c = fbm(uv * 1.6 + vec2(uCamPos.x, uCamPos.z) * 0.0015);
    c = smoothstep(0.52, 0.95, c);
    float fade = smoothstep(0.0, 0.22, dir.y);
    col = mix(col, mix(vec3(0.95, 0.96, 1.0), uSunColor, 0.25), c * 0.75 * fade);
  }

  col = 1.0 - exp(-col * 1.35);
  fragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
`;

export const MESH_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aMat;
layout(location = 4) in vec4 aJoints;
layout(location = 5) in vec4 aWeights;

uniform mat4 uViewProj;
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform highp sampler2D uNodeTex;
uniform int uSkinned;
uniform int uNodeCount;
uniform mat3 uBindRot;
uniform vec3 uBindPos;

out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
out vec3 vMat;

void main() {
  vec3 p = aPos;
  vec3 n = aNormal;
  if (uSkinned == 1) {
    vec3 bind = uBindRot * aPos + uBindPos;   // world position in the bind pose
    vec3 acc = vec3(0.0);
    vec3 accN = vec3(0.0);
    mat3 accD = mat3(0.0);
    for (int i = 0; i < 4; i++) {
      int j = int(aJoints[i] + 0.5);
      float w = aWeights[i];
      if (w <= 0.0) continue;
      if (j < 0 || j >= uNodeCount) continue;
      vec3 nodePos = texelFetch(uNodeTex, ivec2(j, 0), 0).xyz;
      mat3 D = mat3(texelFetch(uNodeTex, ivec2(j, 1), 0).xyz,
                    texelFetch(uNodeTex, ivec2(j, 2), 0).xyz,
                    texelFetch(uNodeTex, ivec2(j, 3), 0).xyz);
      vec3 restW = texelFetch(uNodeTex, ivec2(j, 4), 0).xyz;
      acc += w * (nodePos + D * (bind - restW));
      accD += D * w;
    }
    p = acc;
    n = (accD * uBindRot) * aNormal;
    gl_Position = uViewProj * vec4(p, 1.0);
  } else {
    vec4 wp = uModel * vec4(p, 1.0);
    p = wp.xyz;
    n = uNormalMat * n;
    gl_Position = uViewProj * wp;
  }
  vWorld = p;
  vNormal = n;
  vColor = aColor;
  vMat = aMat;
}
`;

export const MESH_FS = /* glsl */`#version 300 es
precision highp float;
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in vec3 vMat;   // x = metallic, y = roughness, z = emissive flag (must match MESH_VS)
out vec4 fragColor;

uniform float uAlpha;
uniform vec3 uEmissive;
${COMMON_LIGHT}

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCamPos - vWorld);
  if (dot(N, V) < 0.0) N = -N;
  float metal = clamp(vMat.x, 0.0, 1.0);
  float rough = clamp(vMat.y, 0.04, 1.0);
  vec3 base = vColor;

  float ndl = max(dot(N, uSunDir), 0.0);
  float sh = shadowFactor(vWorld);

  vec3 albedo = base * (1.0 - metal);
  vec3 F0 = mix(vec3(0.045), base, metal);

  vec3 H = normalize(uSunDir + V);
  float a = rough * rough;
  float ndh = max(dot(N, H), 0.0);
  float dd = ndh * ndh * (a * a - 1.0) + 1.0;
  float D = (a * a) / (3.14159 * dd * dd + 1e-5);
  float ndv = max(dot(N, V), 1e-4);
  float k = a * 0.5;
  float G = (ndl / (ndl * (1.0 - k) + k + 1e-4)) * (ndv / (ndv * (1.0 - k) + k + 1e-4));
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
  vec3 spec = F * D * G / (4.0 * ndv * max(ndl, 1e-4)) * ndl;
  spec = min(spec, vec3(24.0));

  vec3 R = reflect(-V, N);
  vec3 env = skyRadiance(R);
  vec3 envDiff = skyRadiance(N) * 0.55;

  vec3 ambient = albedo * envDiff + albedo * uGroundColor * 0.12 * (1.0 - N.y * 0.5);
  vec3 direct = albedo * uSunColor * ndl * sh * 1.9;
  vec3 specDirect = spec * uSunColor * sh;

  float fres = pow(1.0 - ndv, 5.0);
  vec3 specEnv = env * (F0 + fres * (1.0 - rough)) * (0.35 + 0.65 * pow(1.0 - rough, 2.0));

  vec3 col = ambient + direct + specDirect + specEnv * mix(0.35, 1.0, sh) + uEmissive;

  // fog
  float dist = length(uCamPos - vWorld);
  float fog = 1.0 - exp(-pow(dist * uFogDensity, 1.7));
  col = mix(col, uFogColor, clamp(fog, 0.0, 1.0));

  col = aces(col);
  fragColor = vec4(pow(col, vec3(1.0 / 2.2)), uAlpha);
}
`;

export const DEPTH_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aMat;
layout(location = 4) in vec4 aJoints;
layout(location = 5) in vec4 aWeights;

uniform mat4 uLightViewProj;
uniform mat4 uModel;
uniform highp sampler2D uNodeTex;
uniform int uSkinned;
uniform int uNodeCount;
uniform mat3 uBindRot;
uniform vec3 uBindPos;

void main() {
  if (uSkinned == 1) {
    vec3 bind = uBindRot * aPos + uBindPos;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 4; i++) {
      int j = int(aJoints[i] + 0.5);
      float w = aWeights[i];
      if (w <= 0.0) continue;
      if (j < 0 || j >= uNodeCount) continue;
      vec3 nodePos = texelFetch(uNodeTex, ivec2(j, 0), 0).xyz;
      mat3 D = mat3(texelFetch(uNodeTex, ivec2(j, 1), 0).xyz,
                    texelFetch(uNodeTex, ivec2(j, 2), 0).xyz,
                    texelFetch(uNodeTex, ivec2(j, 3), 0).xyz);
      vec3 restW = texelFetch(uNodeTex, ivec2(j, 4), 0).xyz;
      acc += w * (nodePos + D * (bind - restW));
    }
    gl_Position = uLightViewProj * vec4(acc, 1.0);
  } else {
    gl_Position = uLightViewProj * uModel * vec4(aPos, 1.0);
  }
}
`;

export const DEPTH_FS = /* glsl */`#version 300 es
precision highp float;
void main() {}
`;

export const SKID_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aAlpha;
uniform mat4 uViewProj;
out float vAlpha;
void main() {
  vAlpha = aAlpha;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}
`;

export const SKID_FS = /* glsl */`#version 300 es
precision highp float;
in float vAlpha;
out vec4 fragColor;
void main() {
  if (vAlpha <= 0.001) discard;
  fragColor = vec4(0.03, 0.03, 0.035, clamp(vAlpha, 0.0, 1.0) * 0.85);
}
`;

export const PARTICLE_VS = /* glsl */`#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;
uniform mat4 uViewProj;
uniform vec3 uCamPos;
uniform float uPixelScale;
out vec3 vColor;
out float vFade;
void main() {
  vec4 cp = uViewProj * vec4(aPos, 1.0);
  gl_Position = cp;
  gl_PointSize = clamp(aSize * uPixelScale / max(cp.w, 0.1), 1.0, 64.0);
  vColor = aColor;
  vFade = clamp(1.0 - cp.w / 120.0, 0.0, 1.0);
}
`;

export const PARTICLE_FS = /* glsl */`#version 300 es
precision highp float;
in vec3 vColor;
in float vFade;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = dot(d, d);
  if (r > 0.25) discard;
  float a = smoothstep(0.25, 0.02, r) * vFade;
  fragColor = vec4(vColor, a);
}
`;
