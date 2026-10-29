"""Software rasteriser used only for verifying the WebGL output headlessly.

Reads /tmp/scene.bin (written by tools/cpu_render.mjs: skinned car triangles +
wheels + a terrain patch, already in world space) and writes a PNG.

    /home/user/.venv/bin/python tools/raster.py /tmp/scene.bin /tmp/scene.png
"""
import json
import struct
import sys

import numpy as np
from PIL import Image

W, H = 1280, 720


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def look_at(eye, target, up):
    f = normalize(np.asarray(target) - np.asarray(eye))
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ np.asarray(eye)
    return m


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def main(bin_path, out_path):
    meta = json.load(open(bin_path.replace('.bin', '.json')))
    raw = np.fromfile(bin_path, dtype=np.float32)
    n = meta['triCount']
    tri = raw.reshape(n, 3, 9)  # per vertex: pos3 nrm3 col3
    pos = tri[:, :, 0:3].astype(np.float64)
    nrm = tri[:, :, 3:6].astype(np.float64)
    col = tri[:, :, 6:9].astype(np.float64)

    cam = meta['camera']
    eye = np.asarray(cam['pos'], dtype=np.float64)
    target = np.asarray(cam['target'], dtype=np.float64)
    view = look_at(eye, target, np.array([0.0, 1.0, 0.0]))
    proj = perspective(cam['fov'], W / H, 0.25, 600.0)
    vp = proj @ view

    flat = pos.reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    clip = hom @ vp.T
    w = clip[:, 3]
    ok = w > 0.05
    ndc = np.zeros((clip.shape[0], 3))
    ndc[ok] = clip[ok, :3] / w[ok, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * W
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    sz = ndc[:, 2]
    sx = sx.reshape(n, 3)
    sy = sy.reshape(n, 3)
    sz = sz.reshape(n, 3)
    vis = ok.reshape(n, 3).all(axis=1)
    # clip triangles whose centroid is off screen
    cx = sx.mean(axis=1)
    cy = sy.mean(axis=1)
    vis &= (cx > -400) & (cx < W + 400) & (cy > -400) & (cy < H + 400)
    vis &= (sz < 1.0).all(axis=1)

    # face normals from the flat (non smooth) geometry for shading
    e1 = pos[:, 1] - pos[:, 0]
    e2 = pos[:, 2] - pos[:, 0]
    fn = np.cross(e1, e2)
    # backface cull in screen space (y flipped)
    area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
            - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
    front = area < 0
    # no backface culling: normals are already oriented towards the viewer,
    # the z-buffer handles occluded geometry, and this keeps ground quads that
    # were wound either way
    keep = np.where(vis)[0]
    print(f'{n} triangles, {len(keep)} after clip/cull')

    light = np.asarray(meta['light'], dtype=np.float64)
    light = normalize(light)
    vn = nrm.mean(axis=1)
    vn_len = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, np.where(vn_len == 0, 1, vn_len))
    face_n = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-6)
    # orient the smooth normal towards the viewer side of the face
    flip = np.where((vn * face_n).sum(axis=1) < 0, -1.0, 1.0)[:, None]
    vn = vn * flip
    ndotl = np.clip(vn @ light, 0, 1)
    ambient = 0.34
    diffuse = 0.85 * ndotl
    shade = (ambient + diffuse)[:, None]
    # a fake sky gradient fill light from above
    fill = (0.10 * np.clip(vn[:, 1], 0, 1))[:, None]
    rgb = np.clip(col.mean(axis=1) * (shade + fill), 0, 1)

    color = np.zeros((H, W, 3), dtype=np.float32)
    # sky background
    yy = np.linspace(0, 1, H)[:, None]
    top = np.array([0.36, 0.55, 0.78])
    bottom = np.array([0.72, 0.80, 0.86])
    color[:] = (top * (1 - yy[..., None]) + bottom * yy[..., None])
    depth = np.full((H, W), np.inf, dtype=np.float32)

    order = keep
    for i in order:
        x0, x1, x2 = sx[i]
        y0, y1, y2 = sy[i]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        px = np.arange(minx, maxx + 1) + 0.5
        py = np.arange(miny, maxy + 1) + 0.5
        PX, PY = np.meshgrid(px, py)
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        a = ((y1 - y2) * (PX - x2) + (x2 - x1) * (PY - y2)) / denom
        b = ((y2 - y0) * (PX - x2) + (x0 - x2) * (PY - y2)) / denom
        c = 1 - a - b
        mask = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        if not mask.any():
            continue
        z = a * sz[i, 0] + b * sz[i, 1] + c * sz[i, 2]
        sub_depth = depth[miny:maxy + 1, minx:maxx + 1]
        mask &= z < sub_depth
        if not mask.any():
            continue
        sub_depth[mask] = z[mask]
        sub_color = color[miny:maxy + 1, minx:maxx + 1]
        sub_color[mask] = rgb[i]

    Image.fromarray((np.clip(color, 0, 1) ** (1 / 1.05) * 255).astype(np.uint8)).save(out_path)
    print('wrote', out_path)
    print('car:', meta.get('car'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/scene.bin',
         sys.argv[2] if len(sys.argv) > 2 else '/tmp/scene.png')
