"""Replays a frame captured by tools/mirror_capture.mjs on the CPU.

It reads the *actual* vertex/index buffers the renderer uploaded, decodes them
with the *actual* attribute strides/offsets, executes the vertex-shader math
(skinning included) and the fragment shading of MESH_FS, then rasterises with a
z-buffer.  That makes GPU-side mistakes (wrong stride, wrong variable packing,
transposed matrices, bad node texture layout) visible as a picture instead of a
guess.

    node tools/mirror_capture.mjs /tmp/mirror
    .venv/bin/python tools/mirror_shade.py /tmp/mirror /tmp/frame.png
"""
import json
import os
import sys

import numpy as np
from PIL import Image

# render size (the capture happens at 1280x720); MIRROR_SCALE=0.66 halves the
# pixel count, handy while iterating on the shaders
_SCALE = float(os.environ.get('MIRROR_SCALE', '1'))
W, H = int(960 * _SCALE), int(540 * _SCALE)


# --------------------------------------------------------------------- helpers
def mat4(a):
    """column-major 16 floats -> 4x4 numpy matrix"""
    m = np.asarray(a, dtype=np.float64).reshape(4, 4, order='F')
    return m


def mat3(a):
    return np.asarray(a, dtype=np.float64).reshape(3, 3, order='F')


def u(call, name, default=None):
    v = call['uniforms'].get(name)
    if v is None:
        return default
    val = v['value']
    if len(val) == 1:
        return val[0]
    return np.asarray(val, dtype=np.float64)


def norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


# ------------------------------------------------------------------------ sky
def sky_image(call):
    inv = mat4(u(call, 'uInvViewProj'))
    cam = u(call, 'uCamPos')
    sun = u(call, 'uSunDir')
    sun_c = u(call, 'uSunColor')
    sky_c = u(call, 'uSkyColor')
    ground_c = u(call, 'uGroundColor')

    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = 1 - (np.arange(H) + 0.5) / H * 2
    X, Y = np.meshgrid(xs, ys)
    p = np.stack([X, Y, np.ones_like(X), np.ones_like(X)], axis=-1) @ inv.T
    world = p[..., :3] / p[..., 3:4]
    d = norm(world - cam)

    h = np.clip(d[..., 1], -1, 1)
    t = np.clip(h * 0.5 + 0.5, 0, 1) ** 1.35
    col = (sky_c[None, None, :] * 0.42) * (1 - t[..., None]) + sky_c[None, None, :] * t[..., None]
    gmix = np.clip(-h * 2.2, 0, 1)[..., None]
    col = col * (1 - gmix) + (ground_c[None, None, :] * 0.55) * gmix
    sd = np.clip(d @ sun, 0, 1)
    col = col + sun_c[None, None, :] * (sd ** 1200)[..., None] * 8.0
    col = col + sun_c[None, None, :] * (sd ** 20)[..., None] * 0.30

    # clouds (cheap value noise)
    upper = d[..., 1] > 0.015
    if upper.any():
        uv = d[..., [0, 2]] / (d[..., 1:2] + 0.12) * 0.9
        uv = uv * 1.6 + np.array([cam[0], cam[2]]) * 0.0015
        n = _fbm(uv)
        c = np.clip((n - 0.52) / 0.43, 0, 1)
        c = c * c * (3 - 2 * c)
        fade = np.clip(d[..., 1] / 0.22, 0, 1)
        cloud_col = np.array([0.95, 0.96, 1.0]) * 0.75 + sun_c * 0.25
        k = (c * 0.75 * fade * upper)[..., None]
        col = col * (1 - k) + cloud_col[None, None, :] * k

    col = 1 - np.exp(-col * 1.35)
    return np.clip(col, 0, 1) ** (1 / 2.2)


def _hash(px):
    v = np.sin(px[..., 0] * 127.1 + px[..., 1] * 311.7) * 43758.5453
    return v - np.floor(v)


def _noise(p):
    i = np.floor(p)
    f = p - i
    uu = f * f * (3 - 2 * f)
    a = _hash(i)
    b = _hash(i + np.array([1, 0]))
    c = _hash(i + np.array([0, 1]))
    d = _hash(i + np.array([1, 1]))
    return (a * (1 - uu[..., 0]) + b * uu[..., 0]) * (1 - uu[..., 1]) + \
           (c * (1 - uu[..., 0]) + d * uu[..., 0]) * uu[..., 1]


def _fbm(p):
    v = 0.0
    a = 0.5
    q = p.copy()
    for _ in range(4):
        v = v + a * _noise(q)
        q = q * 2.03
        a *= 0.5
    return v


# ------------------------------------------------------------------- geometry
def decode(vertex, attrib):
    """interleaved buffer -> float32 (n, stride_floats)"""
    arr = vertex
    stride = attrib['stride'] or (attrib['size'] * 4)
    return arr, stride


def attribute(buf, attrib, size=None):
    """one interleaved attribute -> (n, size) view, using the recorded stride"""
    stride = (attrib['stride'] or (attrib['size'] * 4)) // 4
    off = attrib['offset'] // 4
    size = size or attrib['size']
    # the last vertex only needs its own attribute in range, not a full stride
    n = (buf.size - off - size) // stride + 1
    block = buf[off:off + n * stride]
    return np.lib.stride_tricks.as_strided(
        block, shape=(n, size), strides=(stride * 4, 4), writeable=False)


def _fhash(x, y, z):
    v = np.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return v - np.floor(v)


def vnoise(p):
    """3D value noise, the same construction as dNoise() in MESH_FS"""
    i = np.floor(p)
    f = p - i
    f = f * f * (3.0 - 2.0 * f)
    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[(dx, dy, dz)] = _fhash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    x0 = c[(0, 0, 0)] * (1 - f[:, 0]) + c[(1, 0, 0)] * f[:, 0]
    x1 = c[(0, 1, 0)] * (1 - f[:, 0]) + c[(1, 1, 0)] * f[:, 0]
    x2 = c[(0, 0, 1)] * (1 - f[:, 0]) + c[(1, 0, 1)] * f[:, 0]
    x3 = c[(0, 1, 1)] * (1 - f[:, 0]) + c[(1, 1, 1)] * f[:, 0]
    y0 = x0 * (1 - f[:, 1]) + x1 * f[:, 1]
    y1 = x2 * (1 - f[:, 1]) + x3 * f[:, 1]
    return y0 * (1 - f[:, 2]) + y1 * f[:, 2]


def vfbm(p, octaves=4):
    v = np.zeros(len(p))
    a = 0.5
    q = p.copy()
    for _ in range(octaves):
        v += a * vnoise(q)
        q = q * 2.03
        a *= 0.5
    return v


def node_texels(node_tex, w, row, j):
    o = (row * w + j) * 4
    return node_tex[o:o + 3]


def shade_mesh(call, mesh_cache, node_tex, background, depth, color):
    vbuf = mesh_cache[call['vertex']['file']]
    vp = mat4(u(call, 'uViewProj'))
    model = mat4(u(call, 'uModel'))
    skinned = int(u(call, 'uSkinned', 0))
    a = call['attribs']

    pos = attribute(vbuf, a['0'], 3).astype(np.float64)
    nrm = attribute(vbuf, a['1'], 3).astype(np.float64)
    col = attribute(vbuf, a['2'], 3).astype(np.float64)
    mat = attribute(vbuf, a['3'], 3).astype(np.float64)

    if skinned:
        joints = attribute(vbuf, a['4'], 4).astype(np.int64)
        weights = attribute(vbuf, a['5'], 4).astype(np.float64)
        node_count = int(u(call, 'uNodeCount', 0))
        bind_rot = mat3(u(call, 'uBindRot'))
        bind_pos = u(call, 'uBindPos')
        w = node_tex['w']
        tex = node_tex['data']
        damage = np.zeros(len(pos))
        crumple = np.zeros(len(pos))

        # bind pose positions (world space at spawn)
        bind = pos @ bind_rot.T + bind_pos[None, :]
        acc = np.zeros_like(bind)
        accD = np.zeros((len(pos), 3, 3))
        for i in range(4):
            j = np.clip(joints[:, i], 0, max(node_count - 1, 0))
            wgt = np.where(joints[:, i] < node_count, weights[:, i], 0.0)[:, None]
            node_pos = np.stack([tex[(0 * w + j) * 4 + 0], tex[(0 * w + j) * 4 + 1], tex[(0 * w + j) * 4 + 2]], axis=-1)
            c0 = np.stack([tex[(1 * w + j) * 4 + 0], tex[(1 * w + j) * 4 + 1], tex[(1 * w + j) * 4 + 2]], axis=-1)
            c1 = np.stack([tex[(2 * w + j) * 4 + 0], tex[(2 * w + j) * 4 + 1], tex[(2 * w + j) * 4 + 2]], axis=-1)
            c2 = np.stack([tex[(3 * w + j) * 4 + 0], tex[(3 * w + j) * 4 + 1], tex[(3 * w + j) * 4 + 2]], axis=-1)
            rest = np.stack([tex[(4 * w + j) * 4 + 0], tex[(4 * w + j) * 4 + 1], tex[(4 * w + j) * 4 + 2]], axis=-1)
            d = bind - rest
            # D * d == c0*d.x + c1*d.y + c2*d.z
            offset = c0 * d[:, 0:1] + c1 * d[:, 1:2] + c2 * d[:, 2:3]
            acc += wgt * (node_pos + offset)
            # mat3(c0, c1, c2) in GLSL: the fetched texels are the *columns*
            D = np.stack([c0, c1, c2], axis=-1)         # (n, 3 rows, 3 cols)
            accD += D * wgt[:, :, None]
            # spare .w channels: node damage (row 0) and crumple (row 3)
            damage += wgt[:, 0] * tex[(0 * w + j) * 4 + 3]
            crumple += wgt[:, 0] * tex[(3 * w + j) * 4 + 3]
        world_pos = acc
        # normal: (accD * uBindRot) * aNormal
        nmat = np.einsum('nij,jk->nik', accD, bind_rot)
        world_nrm = np.einsum('nij,nj->ni', nmat, nrm)
    else:
        world_pos = pos @ model[:3, :3].T + model[:3, 3]
        world_nrm = nrm @ model[:3, :3].T
        damage = np.zeros(len(pos))
        crumple = np.zeros(len(pos))

    # project
    hom = np.concatenate([world_pos, np.ones((len(world_pos), 1))], axis=1)
    clip = hom @ vp.T
    w = clip[:, 3]
    valid = w > 1e-4
    ndc = np.zeros((len(clip), 3))
    ndc[valid] = clip[valid, :3] / w[valid, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * W
    sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
    sz = ndc[:, 2]

    if call['kind'] == 'drawElements':
        ibuf = mesh_cache[call['index']['file']]
        first = call['offset'] // 4 if call['index']['kind'] == 'u32' else call['offset'] // 2
        count = call['count']
        idx = ibuf[first:first + count].astype(np.int64)
        idx = idx[:len(idx) // 3 * 3].reshape(-1, 3)
    else:
        idx = np.arange(call['offset'], call['offset'] + call['count'], dtype=np.int64)
        idx = idx[:len(idx) // 3 * 3].reshape(-1, 3)

    tri_ok = valid[idx].all(axis=1)
    # cheap frustum reject with the projected bounding box
    x = sx[idx]
    y = sy[idx]
    tri_ok &= (x.max(axis=1) > -50) & (x.min(axis=1) < W + 50)
    tri_ok &= (y.max(axis=1) > -50) & (y.min(axis=1) < H + 50)
    tri_ok &= (sz[idx] < 1.0).all(axis=1)
    tri = idx[tri_ok]
    # shattered panes fall out (MESH_FS discards them): drop those triangles
    if damage is not None and len(tri):
        dmg_t = np.clip(damage[tri].mean(axis=1), 0, 1)
        fold_t = np.clip(crumple[tri].mean(axis=1), 0, 1)
        kind_t = np.rint(mat[tri].mean(axis=1)[:, 2]).astype(np.int32)
        gain_t = np.clip(dmg_t * 0.75 + fold_t * 0.65, 0, 1)
        centre = world_pos[tri].mean(axis=1)
        hole_noise = vfbm(centre * 11.0 + 31.0)
        holes = np.clip((gain_t - 0.40) / 0.15, 0, 1) * np.clip((hole_noise - 0.50) / 0.08, 0, 1)
        keep = ~((kind_t == 2) & (holes > 0.5))
        if not keep.all():
            print(f'    {int((~keep).sum())} shattered glass triangles removed')
            tri = tri[keep]
    print(f'    {len(tri)} / {len(idx)} triangles kept')

    shade_pbr(call, tri, sx, sy, sz, world_pos, world_nrm, col, mat, background, depth, color,
              damage, crumple)


def shade_pbr(call, tri, sx, sy, sz, wpos, wnrm, col, mat, background, depth, color,
              damage=None, crumple=None):
    cam = u(call, 'uCamPos')
    sun = u(call, 'uSunDir')
    sun_c = u(call, 'uSunColor')
    sky_c = u(call, 'uSkyColor')
    ground_c = u(call, 'uGroundColor')
    fog_c = u(call, 'uFogColor')
    fog_d = u(call, 'uFogDensity', 0.0)
    emissive = u(call, 'uEmissive')
    if emissive is None or np.ndim(emissive) == 0:
        emissive = np.zeros(3)
    alpha = float(u(call, 'uAlpha', 1.0))

    # per-triangle geometry
    p = wpos[tri]                          # (t, 3, 3)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    fn = norm(np.cross(e1, e2))

    vn = norm(wnrm[tri].mean(axis=1))
    flip = np.where((vn * fn).sum(axis=1) < 0, -1.0, 1.0)[:, None]
    vn = vn * flip
    base = col[tri].mean(axis=1).copy()
    mm = mat[tri].mean(axis=1)
    metal = np.clip(mm[:, 0], 0, 1).copy()
    rough = np.clip(mm[:, 1], 0.04, 1).copy()
    kind = np.rint(mat[tri].mean(axis=1)[:, 2]).astype(np.int32)
    # windows are tinted glass (see MESH_FS)
    gl = kind == 2
    metal[gl] = 0.0
    rough[gl] = 0.18
    base = np.where(gl[:, None], base * 0.30, base)

    # ---------------------------------------------------------------- damage
    dmg = np.zeros(len(tri)) if damage is None else np.clip(damage[tri].mean(axis=1), 0, 1)
    fold = np.zeros(len(tri)) if crumple is None else np.clip(crumple[tri].mean(axis=1), 0, 1)
    tint = u(call, 'uTint', np.array([1.0, 1.0, 1.0]))
    if np.ndim(tint) == 0:
        tint = np.array([1.0, 1.0, 1.0])
    base = np.where((kind == 3)[:, None], base * tint[None, :], base)
    gain = np.clip(dmg * 0.75 + fold * 0.65, 0, 1)
    emissive_scale = np.ones(len(tri))
    ctr = p.mean(axis=1)
    glass = kind == 2
    lit = kind == 1
    if (gain > 0.005).any():
        gp = ctr * 7.0
        n1 = vfbm(gp)
        web = None
        for scale, width in ((1.0, 0.055), (2.7, 0.035)):
            n = vfbm(ctr * 7.0 * scale + 11.0) if scale > 1 else n1
            line = 1.0 - np.clip(np.abs(n - 0.5) / width, 0, 1)
            web = line if web is None else np.maximum(web, line * 0.8)
        web *= np.clip((gain - 0.18) / 0.52, 0, 1)
        # cracked glass: milky white with a spider web of bright fractures
        gmask = glass[:, None]
        milk = np.clip((gain - 0.35) * 0.45, 0, 0.30)
        base = np.where(gmask, base * (1 - milk[:, None])
                        + np.array([0.52, 0.57, 0.61])[None, :] * milk[:, None], base)
        wv = (web * 0.55)[:, None]
        base = np.where(gmask, base * (1 - wv) + np.array([0.80, 0.85, 0.88])[None, :] * wv, base)
        # painted panels: chips to bare aluminium, lifted-paint lips, scratches
        chip = vfbm(ctr * 14.0 + 3.0) * 0.72 + vfbm(ctr * 46.0) * 0.28
        edge = 0.58 - gain * 0.22
        chip_mask = np.clip((chip - edge) / 0.06, 0, 1) * np.clip(gain * 1.15, 0, 0.76)
        lip = (np.clip((chip - (edge - 0.07)) / 0.07, 0, 1)
               * (1 - np.clip((chip - edge) / 0.03, 0, 1)) * np.clip(gain, 0, 1))
        bare = np.array([0.27, 0.285, 0.30])
        pm = (~glass)[:, None]
        base = np.where(pm, base * (1 - chip_mask[:, None]) + bare[None, :] * chip_mask[:, None], base)
        base = np.where(pm, base * (1 - (lip * 0.55)[:, None]), base)
        metal = np.where(pm[:, 0], metal * (1 - chip_mask) + 0.52 * chip_mask, metal)
        rough = np.where(pm[:, 0], rough * (1 - chip_mask) + 0.62 * chip_mask, rough)
        streak = vfbm(np.stack([ctr[:, 0] * 90.0, ctr[:, 1] * 3.0, ctr[:, 2] * 90.0], axis=1) + 7.0)
        scratch = np.clip((streak - 0.62) / 0.08, 0, 1) * np.clip(gain * 1.2, 0, 0.85)
        base = np.where(pm, base * (1 - (scratch * 0.6)[:, None])
                        + np.array([0.42, 0.43, 0.44])[None, :] * (scratch * 0.6)[:, None], base)
        rough = np.where(pm[:, 0], np.maximum(rough, 0.55 * scratch), rough)
        soot = np.clip((vfbm(ctr * 3.1) * 0.55 + gain - 0.42) / 0.58, 0, 1)
        base = np.where(pm, base * (1 - (soot * 0.75)[:, None] * 0.62), base)
        rough = np.where(pm[:, 0], np.maximum(rough, 0.95 * soot * 0.6), rough)
        if lit.any():
            emissive_scale = np.ones(len(tri))
            emissive_scale[lit] = 1.0 - 0.92 * np.clip(gain[lit] * 1.3, 0, 1)
    # cracked glass scatters light: kill the sharp highlight where the web is
    if glass.any() and (gain > 0.05).any():
        rough = np.where(glass, np.maximum(rough, gain * 0.66), rough)


    centroid = p.mean(axis=1)
    V = norm(cam[None, :] - centroid)
    N = vn
    albedo = base * (1 - metal[:, None])
    ndl = np.clip((N * sun).sum(axis=1), 0, 1)
    tint = sky_c * 0.55 + ground_c * 0.2
    ambient = albedo * tint[None, :]
    direct = albedo * sun_c[None, :] * ndl[:, None] * 1.9
    # crude environment reflection (skyRadiance in the shader): a bright sky
    # above, the ground below, with a fresnel weight -- without it every
    # surface looks like flat matte paint and glass reads as milk
    upness = np.clip(N[:, 1] * 0.5 + 0.5, 0, 1)
    env = (sky_c[None, :] * (0.35 + 0.45 * upness)[:, None]
           + ground_c[None, :] * (0.55 - 0.45 * upness)[:, None] * 0.55)
    ndv_env = np.clip((N * V).sum(axis=1), 1e-4, 1)
    fres = (1.0 - ndv_env) ** 5
    f0 = 0.045 * (1 - metal)[:, None] + base * metal[:, None]
    eweight = np.clip(0.30 + 0.55 * (1 - rough) ** 2, 0, 1)
    envspec = env * (f0 + fres[:, None] * (1 - rough)[:, None]) * eweight[:, None]
    envspec[glass] *= 1.9
    spec = np.zeros_like(albedo)
    Hh = norm(sun[None, :] + V)
    ndh = np.clip((N * Hh).sum(axis=1), 0, 1)
    a = rough ** 2
    dd = ndh ** 2 * (a ** 2 - 1) + 1
    D = (a ** 2) / (3.14159 * dd ** 2 + 1e-5)
    ndv = np.clip((N * V).sum(axis=1), 1e-4, 1)
    k = a * 0.5
    G = (ndl / (ndl * (1 - k) + k + 1e-4)) * (ndv / (ndv * (1 - k) + k + 1e-4))
    F0 = 0.045 * (1 - metal)[:, None] + base * metal[:, None]
    F = F0 + (1 - F0) * (1 - np.clip((Hh * V).sum(axis=1), 0, 1))[:, None] ** 5
    spec = np.minimum(F * (D * G / (4 * ndv * np.maximum(ndl, 1e-4)))[:, None] * ndl[:, None], 24)
    if gl.any():
        spec[gl] *= 0.55
    rgb = ambient + direct + spec * sun_c[None, :] + envspec + emissive[None, :] * emissive_scale[:, None]

    # fog
    dist = np.linalg.norm(cam[None, :] - centroid, axis=1)
    fog = np.clip(1 - np.exp(-np.power(dist * fog_d, 1.7)), 0, 1)
    rgb = rgb * (1 - fog[:, None]) + fog_c[None, :] * fog[:, None]
    rgb = aces(rgb) ** (1 / 2.2)

    # ------------------------------------------------------------- rasterise
    X = sx[tri]
    Y = sy[tri]
    Z = sz[tri]
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    drawable = np.abs(area) > 1e-9
    order = np.nonzero(drawable)[0]

    minx = np.clip(np.floor(X.min(axis=1)), 0, W - 1).astype(np.int32)
    maxx = np.clip(np.ceil(X.max(axis=1)), 0, W - 1).astype(np.int32)
    miny = np.clip(np.floor(Y.min(axis=1)), 0, H - 1).astype(np.int32)
    maxy = np.clip(np.ceil(Y.max(axis=1)), 0, H - 1).astype(np.int32)
    # skip sliver triangles larger than a few pixels after clipping? no: keep all
    for i in order:
        x0, x1, x2 = X[i]
        y0, y1, y2 = Y[i]
        if maxx[i] < minx[i] or maxy[i] < miny[i]:
            continue
        xs = np.arange(minx[i], maxx[i] + 1) + 0.5
        ys = np.arange(miny[i], maxy[i] + 1) + 0.5
        PX, PY = np.meshgrid(xs, ys)
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        a_ = ((y1 - y2) * (PX - x2) + (x2 - x1) * (PY - y2)) / den
        b_ = ((y2 - y0) * (PX - x2) + (x0 - x2) * (PY - y2)) / den
        c_ = 1 - a_ - b_
        inside = (a_ >= -1e-6) & (b_ >= -1e-6) & (c_ >= -1e-6)
        if not inside.any():
            continue
        z = a_ * Z[i, 0] + b_ * Z[i, 1] + c_ * Z[i, 2]
        sub = depth[miny[i]:maxy[i] + 1, minx[i]:maxx[i] + 1]
        inside &= z < sub
        if not inside.any():
            continue
        sub[inside] = z[inside]
        colsub = color[miny[i]:maxy[i] + 1, minx[i]:maxx[i] + 1]
        if alpha < 1.0:
            blend = rgb[i] * alpha + colsub * (1 - alpha)
            colsub[inside] = blend[inside]
        else:
            colsub[inside] = rgb[i]


def main(capture_dir, out_path):
    frame = json.load(open(os.path.join(capture_dir, 'frame.json')))
    cache = {}

    def load(name):
        if name not in cache:
            cache[name] = np.fromfile(os.path.join(capture_dir, name))
        return cache[name]

    # typed views: vertex buffers are float32, index buffers u32
    mesh_cache = {}

    def buf(entry):
        if entry['file'] not in mesh_cache:
            raw = np.fromfile(os.path.join(capture_dir, entry['file']), dtype=np.float32)
            if entry['kind'] == 'u32':
                raw = raw.view(np.uint32)
            elif entry['kind'] == 'u16':
                raw = raw.view(np.uint16)
            mesh_cache[entry['file']] = raw
        return mesh_cache[entry['file']]

    node = None
    for call in frame['draws']:
        if call.get('nodeTex'):
            node = {
                'data': np.fromfile(os.path.join(capture_dir, call['nodeTex']['file']), dtype=np.float32),
                'w': call['nodeTex']['w'],
            }
            break
    if node is None:
        node = {'data': np.zeros(320 * 5 * 4, dtype=np.float32), 'w': 320}

    if os.environ.get('ZERO_NODES') == '1':
        print('!! simulating a rejected node-texture upload (all texels zero)')
        node = {'data': np.zeros_like(node['data']), 'w': node['w']}

    # sky first (it is the background)
    sky_call = next((c for c in frame['draws'] if 'uInvViewProj' in c['uniforms']), None)
    color = sky_image(sky_call) if sky_call else np.zeros((H, W, 3)) + 0.5
    depth = np.full((H, W), np.inf, dtype=np.float64)

    # meshes: sorted far to near is not needed (z-buffer), but draw bigger ones first
    mesh_calls = []
    for call in frame['draws']:
        if 'uInvViewProj' in call['uniforms']:
            continue           # sky
        if 'uLightViewProj' in call['uniforms']:
            continue           # shadow pass
        if 'uViewProj' not in call['uniforms']:
            continue           # particles / skid
        if 'uPixelScale' in call['uniforms']:
            continue           # particles
        mesh_calls.append(call)

    # preload every referenced buffer once
    for call in mesh_calls:
        buf(call['vertex'])
        if call.get('index'):
            buf(call['index'])

    print(f'replaying {len(mesh_calls)} mesh draw calls')
    for i, call in enumerate(mesh_calls):
        kind = 'skinned' if int(u(call, 'uSkinned', 0)) else 'rigid'
        print(f'  [{i}] {kind} {call["kind"]} verts={call["vertex"]["count"]} indices={call["count"]}')
        shade_mesh(call, mesh_cache, node, color, depth, color)

    Image.fromarray((np.clip(color, 0, 1) * 255).astype(np.uint8)).save(out_path)
    print('wrote', out_path)
    print('car:', frame.get('car'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/mirror',
         sys.argv[2] if len(sys.argv) > 2 else '/tmp/frame.png')
