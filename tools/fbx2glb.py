#!/usr/bin/env python3
"""
fbx2glb.py -- minimal FBX (7.x binary) -> glTF 2.0 (.glb) converter.

Only needs what our car asset uses: geometry (vertices / polygon indices /
normals / per-polygon material ids) and Phong materials.  Node hierarchy is
flattened and re-emitted as a fixed, known-good layout:

    root
      Body          mesh (one primitive per material)
      Wheel_FL/FR/RL/RR   mesh + translation

Axes are converted from the FBX scene (Z-up, front = -X, as authored in
Blender) to glTF (Y-up, front = +Z) with a single rotation:

    (x, y, z) -> (-y, z, -x)

Usage:  python3 tools/fbx2glb.py Tesla_Model_S_game.fbx game/assets/tesla.glb
"""
import json
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbx_dump import parse  # noqa: E402

# --- our asset's known layout -------------------------------------------------
MESH_ORDER = ["Body", "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"]

# per material: pbr tweaks applied on top of the FBX diffuse colour
MATERIAL_OVERRIDES = {
    # name: (metallic, roughness, opacity, emissiveStrength, colorMultiplier)
    "Tesla":                  (0.55, 0.30, 1.0, 0.0),   # body paint
    "Tesla.red":              (0.55, 0.30, 1.0, 0.0),
    "glass":                  (0.10, 0.06, 0.42, 0.0),
    "glass.dark":             (0.10, 0.06, 0.55, 0.0),
    "glass.front":            (0.10, 0.05, 0.38, 0.0),
    "mirror":                 (0.95, 0.05, 1.0, 0.0),
    "headlights.emissions.led": (0.0, 0.25, 1.0, 2.5),
    "headight.emissions.led": (0.0, 0.25, 1.0, 2.5),
    "headights.emissions.led": (0.0, 0.25, 1.0, 2.5),
    "headights.emissions":    (0.0, 0.25, 1.0, 2.0),
    "backights.red":          (0.0, 0.30, 1.0, 1.4),
    "Tyres":                  (0.0, 0.85, 1.0, 0.0),
    "Metal-brake_discs":      (0.9, 0.35, 1.0, 0.0),
    "Metal-dark":             (0.85, 0.40, 1.0, 0.0),
    "Metal-gray":             (0.85, 0.35, 1.0, 0.0),
    "Metal-gray-warmer":      (0.80, 0.40, 1.0, 0.0),
    "plastic.black":          (0.0, 0.65, 1.0, 0.0),
}

# A few colour fixes (the source FBX relies on missing texture maps, so the
# diffuse values are mostly placeholders).
MATERIAL_COLORS = {
    "Tesla":        (0.62, 0.07, 0.07),   # metallic red paint
    "Tesla.red":    (0.05, 0.05, 0.05),   # wheel centre caps
    "glass":        (0.35, 0.40, 0.45),
    "glass.dark":   (0.10, 0.11, 0.13),
    "glass.front":  (0.55, 0.62, 0.70),
    "mirror":       (0.95, 0.95, 1.00),
    "Tyres":        (0.035, 0.035, 0.038),
    "Metal-brake_discs": (0.35, 0.35, 0.37),
    "Metal-dark":   (0.09, 0.09, 0.10),
    "Metal-gray":   (0.42, 0.44, 0.47),
    "Metal-gray-warmer": (0.50, 0.47, 0.43),
    "plastic.black": (0.03, 0.03, 0.035),
    "headlight_reflective": (0.85, 0.87, 0.90),
    "backights.red": (0.55, 0.03, 0.03),
    "backights.grey": (0.20, 0.20, 0.21),
    "backights.grey.001": (0.75, 0.75, 0.76),
    "backights.off": (0.12, 0.12, 0.12),
    "floor":        (0.12, 0.12, 0.12),
    "grid":         (0.10, 0.10, 0.10),
    "red":          (0.45, 0.02, 0.02),
    "white":        (0.88, 0.88, 0.88),
    "indicator":    (0.90, 0.45, 0.10),
    "FrontPlaqueImage": (0.85, 0.85, 0.85),
    "headights.emissions.led": (1.0, 0.98, 0.92),
    "headights.emissions": (0.95, 0.95, 0.90),
}


def rot(v):
    """(x, y, z) -> (-y, z, -x); Z-up/-X-front  =>  Y-up/+Z-front."""
    return (-v[1], v[2], -v[0])


def read_geometry(geom):
    """-> dict(vertices, normals, mat_index, materials) fully triangulated."""
    children = {c.name: c for c in geom.children}
    verts = children["Vertices"].props[0]
    pvi = children["PolygonVertexIndex"].props[0]

    normals = None
    nidx = None
    ln = children.get("LayerElementNormal")
    if ln is not None:
        lnc = {c.name: c for c in ln.children}
        normals = lnc["Normals"].props[0]
        if "NormalsIndex" in lnc:
            nidx = lnc["NormalsIndex"].props[0]

    mat_ids = None
    lm = children.get("LayerElementMaterial")
    if lm is not None:
        lmc = {c.name: c for c in lm.children}
        mat_ids = lmc["Materials"].props[0]

    # split into polygons
    polys = []
    cur = []
    for i, p in enumerate(pvi):
        if p < 0:
            cur.append((~p, i))
            polys.append(cur)
            cur = []
        else:
            cur.append((p, i))

    tris = []  # (a_vert, a_corner, b_vert, b_corner, c_vert, c_corner, poly_id)
    for pid, poly in enumerate(polys):
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1], pid))

    return verts, normals, nidx, mat_ids, tris


def build_gltf(fbx_path):
    nodes = parse(fbx_path)
    objs = [n for n in nodes if n.name == "Objects"][0]

    id_to_name = {}
    for o in objs.children:
        nm = o.props[1].split("\x00")[0] if len(o.props) > 1 and isinstance(o.props[1], str) else ""
        id_to_name[o.props[0]] = nm

    geoms = {}
    for o in objs.children:
        if o.name == "Geometry":
            geoms[o.props[1].split("\x00")[0]] = o

    # material order as seen on the Body model (matches the per-polygon indices)
    conn = [n for n in nodes if n.name == "Connections"][0]
    body_mats = []
    for c in conn.children:
        if c.props[0] == "OO" and id_to_name.get(c.props[2]) == "Body":
            nm = id_to_name.get(c.props[1])
            if nm and c.props[1] in id_to_name and any(
                    m.props[0] == c.props[1] and m.name == "Material" for m in objs.children):
                body_mats.append(nm)

    # ------------------------------------------------------------------ gather meshes
    bin_parts = []
    buffer_views = []
    accessors = []
    gl_meshes = []
    gl_nodes = []
    gl_materials = []
    mat_index = {}

    def add_mat(name):
        if name in mat_index:
            return mat_index[name]
        metal, rough, opacity, emissive = MATERIAL_OVERRIDES.get(name, (0.1, 0.5, 1.0, 0.0))
        col = MATERIAL_COLORS.get(name)
        if col is None:
            src = next((m for m in objs.children
                        if m.name == "Material" and m.props[1].split("\x00")[0] == name), None)
            col = (0.5, 0.5, 0.5)
            if src is not None:
                for c in src.children:
                    if c.name == "Properties70":
                        for p in c.children:
                            if p.props[0] == "DiffuseColor":
                                col = tuple(p.props[4:7])
        m = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [round(col[0], 4), round(col[1], 4), round(col[2], 4), opacity],
                "metallicFactor": metal,
                "roughnessFactor": rough,
            },
            "doubleSided": False,
        }
        if opacity < 1.0:
            m["alphaMode"] = "BLEND"
        if emissive > 0.0:
            m["emissiveFactor"] = [round(min(1.0, col[0] * emissive), 4),
                                   round(min(1.0, col[1] * emissive), 4),
                                   round(min(1.0, col[2] * emissive), 4)]
        gl_materials.append(m)
        mat_index[name] = len(gl_materials) - 1
        return mat_index[name]

    def add_data(blob, target):
        while len(blob) % 4:
            blob += b"\x00"
        offset = sum(len(b) for b in bin_parts)
        bin_parts.append(blob)
        buffer_views.append({"buffer": 0, "byteOffset": offset,
                             "byteLength": len(blob), "target": target})
        return len(buffer_views) - 1

    for name in MESH_ORDER:
        geom = geoms[name]
        verts, normals, nidx, mat_ids, tris = read_geometry(geom)
        nmat = len(body_mats) if name == "Body" else 1
        groups = {i: [] for i in range(nmat)}
        for a, b, c, pid in tris:
            mi = mat_ids[pid] if mat_ids else 0
            if name != "Body":
                # wheels reuse the Body material list order (exported by the same DCC pass)
                pass
            groups.setdefault(mi, []).append((a, b, c))

        primitives = []
        wheel_mats = None
        if name != "Body":
            # wheels have their own material order -- collect from the model connection
            wheel_mats = []
            for c in conn.children:
                if c.props[0] == "OO" and id_to_name.get(c.props[2]) == name:
                    nm = id_to_name.get(c.props[1])
                    if nm and any(m.props[0] == c.props[1] and m.name == "Material" for m in objs.children):
                        wheel_mats.append(nm)

        for mi, tri_list in sorted(groups.items()):
            if not tri_list:
                continue
            pos, nor, idx = [], [], []
            cache = {}
            for t in tri_list:
                for corner in t:
                    vi, corner_i = corner
                    x, y, z = verts[vi * 3:vi * 3 + 3]
                    p = rot((x, y, z))
                    if normals is not None:
                        ni = nidx[corner_i] if nidx is not None else vi
                        nx, ny, nz = normals[ni * 3: ni * 3 + 3]
                        nn = rot((nx, ny, nz))
                    else:
                        nn = (0.0, 1.0, 0.0)
                    key = (round(p[0], 4), round(p[1], 4), round(p[2], 4),
                           round(nn[0], 2), round(nn[1], 2), round(nn[2], 2))
                    got = cache.get(key)
                    if got is None:
                        got = len(pos) // 3
                        cache[key] = got
                        pos.extend(p)
                        nor.extend(nn)
                    idx.append(got)

            pos_bin = struct.pack("<%df" % len(pos), *pos)
            nor_bin = struct.pack("<%df" % len(nor), *nor)
            idx_bin = struct.pack("<%dI" % len(idx), *idx)
            pv = add_data(pos_bin, 34962)
            nv = add_data(nor_bin, 34962)
            iv = add_data(idx_bin, 34963)
            xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
            accessors.append({"bufferView": pv, "componentType": 5126, "count": len(pos) // 3,
                              "type": "VEC3",
                              "min": [min(xs), min(ys), min(zs)],
                              "max": [max(xs), max(ys), max(zs)]})
            a_pos = len(accessors) - 1
            accessors.append({"bufferView": nv, "componentType": 5126,
                              "count": len(nor) // 3, "type": "VEC3"})
            a_nor = len(accessors) - 1
            accessors.append({"bufferView": iv, "componentType": 5125,
                              "count": len(idx), "type": "SCALAR"})
            a_idx = len(accessors) - 1

            if name == "Body":
                mname = body_mats[mi]
            else:
                mname = wheel_mats[mi] if wheel_mats and mi < len(wheel_mats) else "Metal-gray"
            primitives.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nor},
                               "indices": a_idx, "material": add_mat(mname),
                               "material_name": mname})

        gl_meshes.append({"name": name, "primitives": [
            {k: v for k, v in p.items() if k != "material_name"} for p in primitives]})
        gl_nodes.append({"name": name, "mesh": len(gl_meshes) - 1})
        print("  %-10s %d primitives" % (name, len(primitives)))

    # wheel node translations (already in glTF axes)
    WHEEL_LOCAL = {
        "Wheel_FL": (-1.56961190700531, -0.806113600730896, 0.3598611652851105),
        "Wheel_FR": (-1.56961190700531, 0.806113600730896, 0.3598611652851105),
        "Wheel_RL": (1.4137461185455322, -0.8403064608573914, 0.35986122488975525),
        "Wheel_RR": (1.4137461185455322, 0.8403064608573914, 0.35986122488975525),
    }
    children = [0]
    for i, nm in enumerate(MESH_ORDER[1:], start=1):
        gl_nodes[i]["translation"] = list(rot(WHEEL_LOCAL[nm]))
        children.append(i)
    gl_nodes.insert(0, {"name": "Tesla", "children": children})

    bin_blob = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "fbx2glb.py (gamelikebeamng)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": gl_nodes,
        "meshes": gl_meshes,
        "materials": gl_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf8")
    while len(json_blob) % 4:
        json_blob += b" "

    total = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_blob), 0x4E4F534A) + json_blob
    out += struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
    return bytes(out)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Tesla_Model_S_game.fbx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "game/assets/tesla.glb"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("converting", src, "->", dst)
    data = build_gltf(src)
    with open(dst, "wb") as fh:
        fh.write(data)
    print("wrote %s (%.2f MB)" % (dst, len(data) / 1e6))
