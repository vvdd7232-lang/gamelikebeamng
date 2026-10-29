#!/usr/bin/env python3
"""
gen_cage.py -- derive the soft-body node cage from the car's shell meshes.

Cage is a tube: N stations along the car (front = +z) with a fixed number of
angular sectors around each section.  For every sector the outermost shell
vertex in a thin slice is taken, so the cage sits on the sheet metal.  The
result is a clean symmetric truss (rings + longitudinal beams + diagonals)
which is far more stable than a hand-wired point cloud.

Every shell panel counts when measuring a section -- with the multi-part model
the doors and the hood are separate meshes, and using only "Body" would leave
holes in the silhouette exactly where a crash needs nodes.

Writes game/src/vehicle/car_nodes.js plus a PNG debug plot of the cage.

Usage:  python3 tools/gen_cage.py [game/assets/volvo.glb]
"""
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# meshes that make up the outer silhouette, and get a cage node ring
SHELL = ["Body", "Hood", "Trunk", "Door_FL", "Door_FR", "Door_RL", "Door_RR",
         "Bumper_Front", "Bumper_Rear"]

# z stations (car space, front = +z) -- denser at the ends, where the car has
# to fold, coarser in the middle where the safety cell must stay a box
STATIONS = [2.44, 2.24, 2.02, 1.78, 1.50, 1.14, 0.72, 0.30,
            -0.12, -0.54, -0.96, -1.36, -1.72, -2.06, -2.38]
SECTORS = 16
SLAB = 0.075


def load_glb_mesh(path):
    d = open(path, 'rb').read()
    off = 12
    js = bins = None
    while off < len(d):
        ln, ty = struct.unpack_from('<II', d, off)
        off += 8
        chunk = d[off:off + ln]
        off += ln
        if ty == 0x4E4F534A:
            js = json.loads(chunk)
        else:
            bins = chunk
    prims = []
    for node in js['nodes']:
        if 'mesh' not in node:
            continue
        tr = node.get('translation', [0, 0, 0])
        for p in js['meshes'][node['mesh']]['primitives']:
            a = js['accessors'][p['attributes']['POSITION']]
            bv = js['bufferViews'][a['bufferView']]
            arr = np.frombuffer(bins, dtype='<f4', count=a['count'] * 3,
                                offset=bv.get('byteOffset', 0)).reshape(-1, 3)
            idx = None
            if 'indices' in p:
                ia = js['accessors'][p['indices']]
                ibv = js['bufferViews'][ia['bufferView']]
                idx = np.frombuffer(bins, dtype='<u4', count=ia['count'],
                                    offset=ibv.get('byteOffset', 0)).reshape(-1, 3)
            prims.append((node['name'], arr + np.array(tr, dtype=np.float32), idx))
    return prims, js


def slice_section(verts, zc):
    m = np.abs(verts[:, 2] - zc) < SLAB
    sl = verts[m]
    if len(sl) < 40:
        m = np.abs(verts[:, 2] - zc) < SLAB * 3
        sl = verts[m]
    return sl


def section_ring(sl, k=SECTORS):
    """outermost point per angular sector around the section centroid"""
    cx = float(np.median(sl[:, 0]))
    cy = float(np.median(sl[:, 1]))
    rel = sl[:, :2] - np.array([cx, cy])
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 1])
    rs = []
    for i in range(k):
        a0 = -math.pi + 2 * math.pi * i / k
        a1 = a0 + 2 * math.pi / k
        sel = (ang >= a0) & (ang < a1)
        if not sel.any():
            rs.append(None)
            continue
        idx = np.argmax(rad[sel])
        rr = rad[sel][idx]
        aa = math.atan2(rel[sel][idx][1], rel[sel][idx][0])
        rs.append((cx + math.cos(aa) * rr, cy + math.sin(aa) * rr))
    n = len(rs)
    for i in range(n):
        if rs[i] is None:
            prev = next((rs[(i - j) % n] for j in range(1, n) if rs[(i - j) % n]), None)
            nxt = next((rs[(i + j) % n] for j in range(1, n) if rs[(i + j) % n]), None)
            if prev and nxt:
                rs[i] = ((prev[0] + nxt[0]) * 0.5, (prev[1] + nxt[1]) * 0.5)
            else:
                rs[i] = (cx, cy + 0.4)
    # enforce left/right mirror symmetry
    out = [None] * n
    for i in range(n):
        a = -math.pi + 2 * math.pi * (i + 0.5) / n
        am = math.pi - a
        am = (am + math.pi) % (2 * math.pi) - math.pi
        j = int(round((am + math.pi) / (2 * math.pi) * n - 0.5)) % n
        px, py = rs[i]
        qx, qy = rs[j]
        if abs(px) >= abs(qx):
            x, y = abs(px), py
        else:
            x, y = abs(qx), qy
        out[i] = [round(x if px >= 0 else -x, 4), round(y, 4)]
        out[j] = [round(-(x if px >= 0 else -x), 4), round(y, 4)]
    out = [[max(-1.2, min(1.2, p[0])), p[1]] for p in out]
    return out, (cx, cy)


# ---------------------------------------------------------------- parts
# Hinges live where the artist put each panel's origin: the door origins sit on
# the A/B pillars, the hood origin on the cowl, the trunk origin on its front
# edge.  `open` is the angle gravity settles a freed panel towards, `max` the
# hard stop of the hinge.
HINGES = {
    'Hood':       {'axis': [1, 0, 0], 'open': -0.85, 'max': 1.35, 'k': 260, 'damp': 16, 'mass': 16},
    'Trunk':      {'axis': [1, 0, 0], 'open': 0.75, 'max': -1.20, 'k': 230, 'damp': 15, 'mass': 15},
    'Door_FL':    {'axis': [0, 1, 0], 'open': -1.15, 'max': -1.55, 'k': 420, 'damp': 26, 'mass': 24},
    'Door_FR':    {'axis': [0, 1, 0], 'open': 1.15, 'max': 1.55, 'k': 420, 'damp': 26, 'mass': 24},
    'Door_RL':    {'axis': [0, 1, 0], 'open': -1.10, 'max': -1.45, 'k': 380, 'damp': 24, 'mass': 20},
    'Door_RR':    {'axis': [0, 1, 0], 'open': 1.10, 'max': 1.45, 'k': 380, 'damp': 24, 'mass': 20},
    'Bumper_Front': {'axis': [1, 0, 0], 'open': 0.55, 'max': 0.85, 'k': 900, 'damp': 40, 'mass': 14, 'rigid': True},
    'Bumper_Rear':  {'axis': [1, 0, 0], 'open': 0.55, 'max': 0.85, 'k': 900, 'damp': 40, 'mass': 12, 'rigid': True},
}
STATIC = ['Interior', 'SteeringWheel']


def build_parts(prims, js, rings):
    """Panel table: pivots, mass, centre of mass and the cage nodes that hold
    each part on.  The game simulates a panel as a body on a hinge, so all of
    this has to come from the model rather than be guessed."""
    nodes = np.array([[p[0], p[1], p[2]] for ring in rings for p in ring], dtype=np.float64)
    tr = {n['name']: n.get('translation', [0, 0, 0]) for n in js['nodes']}
    # the glb splits a part into one primitive per material: merge them back
    merged = {}
    for name, arr, idx in prims:
        if name not in merged:
            merged[name] = [[], []]
        base = sum(len(v) for v in merged[name][0])
        merged[name][0].append(arr)
        if idx is not None:
            merged[name][1].append(idx.astype(np.int64) + base)
    out = []
    for name, (chunks, idx_chunks) in merged.items():
        if name.startswith('Wheel_') or name.startswith('BrakeCaliper_'):
            continue
        hinge = HINGES.get(name)
        role = 'panel' if hinge else ('body' if name == 'Body' else 'static')
        if role == 'static' and name not in STATIC:
            continue
        pivot = [round(float(v), 4) for v in tr.get(name, [0, 0, 0])]
        p = np.concatenate(chunks).astype(np.float64)   # already in car space
        idx = np.concatenate(idx_chunks) if idx_chunks else None
        if idx is not None and len(idx):
            a = p[idx[:, 0]]
            b = p[idx[:, 1]]
            c = p[idx[:, 2]]
        else:
            nt = len(p) // 3
            a, b, c = p[:nt * 3:3], p[1:nt * 3:3], p[2:nt * 3:3]
        nrm = np.cross(b - a, c - a)
        tri_area = 0.5 * np.linalg.norm(nrm, axis=1)
        area = float(np.sum(tri_area)) if len(a) else 0.0
        if area > 0:
            com = ((a + b + c) / 3.0 * tri_area[:, None]).sum(axis=0) / area
        else:
            com = np.array(pivot)
        com = [round(float(v), 4) for v in com]
        entry = {
            'name': name,
            'role': role,
            'pivot': pivot,
            'com': com,
            'area': round(area, 3),
        }
        if hinge:
            entry['hinge'] = dict(hinge)
            entry['mass'] = max(4, min(hinge['mass'], max(4, area * 8)))
            if name.startswith('Bumper'):
                # a bumper is welded on, not hinged: it hangs off its own centre
                # so that a knock makes it droop instead of levering the whole
                # nose around some imaginary pivot at the car origin
                entry['pivot'] = com
                pivot = com
                anchors = [com]
            else:
                ax = hinge['axis']
                span = max(0.35, min(1.3, math.sqrt(max(area, 0.2)) * 0.5))
                # the mount is the hinge line: those are the nodes that hold it on
                anchors = [[pivot[i] + (ax[i] * t * span if ax[i] else 0) for i in range(3)]
                           for t in (-1, 0, 1)]
        else:
            entry['mass'] = 6
            anchors = [com]
        # cage nodes that hold the part on (nearest to the hinge line / centre)
        near = set()
        for an in anchors:
            d = np.linalg.norm(nodes - np.array(an), axis=1)
            near.update(int(i) for i in np.argsort(d)[:4])
        entry['nodes'] = sorted(near)
        near2 = set()
        for an in [com, pivot]:
            d = np.linalg.norm(nodes - np.array(an), axis=1)
            near2.update(int(i) for i in np.argsort(d)[:6])
        entry['bodyNodes'] = sorted(near2)
        out.append(entry)
        print('  part %-14s %-6s pivot %s com %s area %5.2f mass %4.1f' %
              (name, role, pivot, com, area, entry['mass']))
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'game/assets/volvo.glb')
    prims, js = load_glb_mesh(src)
    shell = np.concatenate([a for n, a, _ in prims if n in SHELL])
    print('shell verts', len(shell))
    lo = shell.min(axis=0)
    hi = shell.max(axis=0)
    print('bounds x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]' % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    rings = []
    stations = []
    for zc in STATIONS:
        sl = slice_section(shell, zc)
        ring, c = section_ring(sl)
        rings.append([[p[0], p[1], zc] for p in ring])
        stations.append({'z': zc, 'centroid': [round(c[0], 3), round(c[1], 3)], 'points': len(ring)})
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        print('z=%6.3f  x[%.2f..%.2f] y[%.2f..%.2f]  (%d verts)' %
              (zc, min(xs), max(xs), min(ys), max(ys), len(sl)))

    out = {
        'stations': stations,
        'rings': rings,
        'meta': {
            'sectors': SECTORS,
            'source': os.path.basename(src),
            'bounds': {'min': [round(float(v), 4) for v in lo],
                       'max': [round(float(v), 4) for v in hi]},
        },
    }
    parts = build_parts(prims, js, rings)
    out['parts'] = parts
    dst = os.path.join(ROOT, 'game/src/vehicle/car_nodes.js')
    with open(dst, 'w') as fh:
        fh.write('// AUTO-GENERATED by tools/gen_cage.py -- do not edit by hand.\n')
        fh.write('// Structured soft-body cage: rings (sector points) x stations.\n')
        fh.write('export const CAGE = ')
        json.dump(out, fh, separators=(',', ':'))
        fh.write(';\n')
    print('wrote', dst, 'nodes', len(rings) * SECTORS)

    # ---- debug plot
    W, H = 1200, 760
    img = Image.new('RGB', (W, H), (18, 22, 28))
    dr = ImageDraw.Draw(img)
    scale = 130.0

    def side(p):
        return (W * 0.5 - p[2] * scale, 300 - p[1] * scale)

    def top(p):
        return (W * 0.5 - p[2] * scale, 580 - p[0] * scale)

    for name, arr, _ in prims:
        if name not in SHELL:
            continue
        col = (55, 65, 75) if name == 'Body' else (70, 60, 50)
        for v in arr[::7]:
            dr.point(side(v), fill=col)
            dr.point(top(v), fill=col)
    for ring in rings:
        for p in ring:
            x, y = side(p)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(90, 230, 150))
            x2, y2 = top(p)
            dr.ellipse([x2 - 3, y2 - 3, x2 + 3, y2 + 3], fill=(245, 190, 80))
        pts_top = [top(p) for p in ring]
        dr.line(pts_top + [pts_top[0]], fill=(120, 90, 40))
        pts_side = [side(p) for p in ring]
        dr.line(pts_side + [pts_side[0]], fill=(50, 120, 80))
    for i in range(len(rings) - 1):
        for j in range(SECTORS):
            dr.line([side(rings[i][j]), side(rings[i + 1][j])], fill=(40, 90, 60))
    # wheels
    for name, wx, wz, r in (('FL', 0.816, 1.61, 0.36), ('FR', -0.813, 1.611, 0.36),
                            ('RL', 0.816, -1.33, 0.36), ('RR', -0.813, -1.33, 0.36)):
        x, y = side((0, 0.36, wz))
        dr.ellipse([x - r * scale, y - r * scale, x + r * scale, y + r * scale], outline=(230, 90, 90))
        x2, y2 = top((wx, 0, wz))
        dr.rectangle([x2 - 0.13 * scale, y2 - 0.36 * scale,
                      x2 + 0.13 * scale, y2 + 0.36 * scale], outline=(230, 90, 90))
    dr.text((10, 10), 'side view: green = cage nodes, red = wheels', fill=(200, 220, 240))
    dr.text((10, 350), 'top view: orange = cage nodes', fill=(200, 220, 240))
    img.save('/tmp/cage.png')
    print('wrote /tmp/cage.png')


if __name__ == '__main__':
    main()
