#!/usr/bin/env python3
"""
fbx2glb_parts.py -- FBX (7.x binary) -> glTF 2.0 (.glb) for the multi-part car.

Unlike tools/fbx2glb.py (written around the Tesla, where the whole shell is one
"Body" mesh), this converter keeps *every* model as its own node and carries the
node pivot through to glTF.  That pivot is what the game uses as a hinge: the
Volvo asset is authored with the door / hood / trunk origins sitting on their
hinge lines, so a panel can be swung open around its own translation.

Axes: the source is authored Z-up with the car pointing down -Y (Blender).
glTF (and this game) wants Y-up with the car pointing at +Z, so:

    (x, y, z) -> (x, z, -y)

Usage:  python3 tools/fbx2glb_parts.py Volvo_S90_optimized.fbx game/assets/volvo.glb
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbx_dump import parse  # noqa: E402

# ---------------------------------------------------------------- materials
# The asset's textures are external files that are not in the repository, so
# every material is described by hand instead of being read from the FBX.
# name: (metallic, roughness, opacity, emissive_strength)
MATERIAL_OVERRIDES = {
    "Car Paint":            (0.60, 0.26, 1.0, 0.0),
    "rubber":               (0.0, 0.90, 1.0, 0.0),
    "Tire.001":             (0.0, 0.92, 1.0, 0.0),
    "Plastic":              (0.0, 0.70, 1.0, 0.0),
    "Plastic.001":          (0.0, 0.75, 1.0, 0.0),
    "Metal Black Rough":    (0.85, 0.55, 1.0, 0.0),
    "Metal Rough":          (0.92, 0.34, 1.0, 0.0),
    "Metal Rough.001":      (0.92, 0.38, 1.0, 0.0),
    "metal_rough.002":      (0.90, 0.40, 1.0, 0.0),
    "Chrome":               (1.0, 0.06, 1.0, 0.0),
    "chrome.001":           (1.0, 0.08, 1.0, 0.0),
    "Black Glossy":         (0.35, 0.10, 1.0, 0.0),
    "Glass":                (0.10, 0.05, 0.38, 0.0),
    "Glass Taillight":      (0.05, 0.10, 0.55, 0.0),
    "Glass headlights":     (0.10, 0.05, 0.35, 0.0),
    "GlassRunninglight":    (0.10, 0.08, 0.55, 1.6),
    "Glass_wavey":          (0.10, 0.05, 0.40, 0.0),
    "heaxagon glass":       (0.10, 0.06, 0.45, 0.0),
    "Black Glass":          (0.10, 0.06, 0.45, 0.0),
    "Translucent_Glass":    (0.10, 0.06, 0.55, 0.0),
    "Translucent_Glass.001": (0.10, 0.06, 0.55, 0.0),
    "Headlight Insides":    (0.55, 0.18, 1.0, 0.0),
    "emission":             (0.0, 0.30, 1.0, 2.2),
    "Taillight":            (0.0, 0.20, 1.0, 1.6),
    "Turn signal(taillight)": (0.0, 0.25, 1.0, 1.1),
    "turnsignal":           (0.0, 0.25, 1.0, 0.6),
    "Metal Turnsignal":     (0.60, 0.30, 1.0, 0.0),
    "Shell":                (0.0, 0.55, 1.0, 0.0),
    "Rims.001":             (0.95, 0.22, 1.0, 0.0),
    "disc.001":             (0.90, 0.35, 1.0, 0.0),
    "caliper":              (0.20, 0.35, 1.0, 0.0),
    "Dashboard":            (0.0, 0.62, 1.0, 0.0),
    "DoorPanelFront":       (0.0, 0.66, 1.0, 0.0),
    "DoorPanelRear":        (0.0, 0.66, 1.0, 0.0),
    "Front Seat":           (0.0, 0.70, 1.0, 0.0),
    "Rear Seats":           (0.0, 0.70, 1.0, 0.0),
    "Steeringwheel":        (0.05, 0.55, 1.0, 0.0),
    "CenterConsole":        (0.0, 0.50, 1.0, 0.0),
    "Speedometer":          (0.0, 0.20, 1.0, 1.2),
    "infotainmentScreen":   (0.0, 0.12, 1.0, 0.9),
    "crystal":              (0.10, 0.08, 1.0, 0.0),
    "tailpipes":            (0.95, 0.25, 1.0, 0.0),
    "yellow Glossy":        (0.1, 0.20, 1.0, 0.0),
    "blue Glossy":          (0.1, 0.20, 1.0, 0.0),
    "null":                 (0.0, 0.70, 1.0, 0.0),
    "null.002":             (0.0, 0.70, 1.0, 0.0),
}

MATERIAL_COLORS = {
    "Car Paint":            (0.42, 0.045, 0.055),   # metallic dark red
    "rubber":               (0.028, 0.028, 0.030),
    "Tire.001":             (0.030, 0.030, 0.032),
    "Plastic":              (0.035, 0.035, 0.038),
    "Plastic.001":          (0.045, 0.045, 0.048),
    "Metal Black Rough":    (0.055, 0.055, 0.060),
    "Metal Rough":          (0.55, 0.56, 0.59),
    "Metal Rough.001":      (0.48, 0.49, 0.52),
    "metal_rough.002":      (0.52, 0.53, 0.56),
    "Chrome":               (0.92, 0.93, 0.95),
    "chrome.001":           (0.90, 0.91, 0.94),
    "Black Glossy":         (0.020, 0.020, 0.024),
    "Glass":                (0.16, 0.20, 0.24),
    "Glass Taillight":      (0.35, 0.05, 0.06),
    "Glass headlights":     (0.55, 0.60, 0.66),
    "GlassRunninglight":    (0.85, 0.84, 0.78),
    "Glass_wavey":          (0.20, 0.24, 0.28),
    "heaxagon glass":       (0.30, 0.34, 0.38),
    "Black Glass":          (0.030, 0.032, 0.038),
    "Translucent_Glass":    (0.42, 0.46, 0.50),
    "Translucent_Glass.001": (0.42, 0.46, 0.50),
    "Headlight Insides":    (0.80, 0.82, 0.86),
    "emission":             (1.0, 0.97, 0.90),
    "Taillight":            (0.62, 0.035, 0.035),
    "Turn signal(taillight)": (0.80, 0.30, 0.06),
    "turnsignal":           (0.85, 0.40, 0.10),
    "Metal Turnsignal":     (0.50, 0.50, 0.52),
    "Shell":                (0.045, 0.045, 0.050),
    "Rims.001":             (0.62, 0.64, 0.67),
    "disc.001":             (0.42, 0.42, 0.44),
    "caliper":              (0.22, 0.23, 0.25),
    "Dashboard":            (0.10, 0.10, 0.11),
    "DoorPanelFront":       (0.11, 0.10, 0.10),
    "DoorPanelRear":        (0.11, 0.10, 0.10),
    "Front Seat":           (0.13, 0.12, 0.12),
    "Rear Seats":           (0.12, 0.11, 0.11),
    "Steeringwheel":        (0.08, 0.08, 0.09),
    "CenterConsole":        (0.09, 0.09, 0.10),
    "Speedometer":          (0.75, 0.82, 0.90),
    "infotainmentScreen":   (0.10, 0.16, 0.22),
    "crystal":              (0.30, 0.34, 0.40),
    "tailpipes":            (0.72, 0.73, 0.75),
    "yellow Glossy":        (0.70, 0.50, 0.06),
    "blue Glossy":          (0.06, 0.14, 0.55),
    "null":                 (0.05, 0.05, 0.05),
    "null.002":             (0.05, 0.05, 0.05),
}

# Which meshes belong to which role in the game.
WHEELS = ["Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"]
CALIPERS = {"BrakeCaliper_FL": "FL", "BrakeCaliper_FR": "FR",
            "BrakeCaliper_RL": "RL", "BrakeCaliper_RR": "RR"}
PANELS = ["Hood", "Trunk", "Door_FL", "Door_FR", "Door_RL", "Door_RR",
          "Bumper_Front", "Bumper_Rear"]


def rot(v):
    """FBX (Z-up, front = -Y) -> glTF (Y-up, front = +Z)."""
    return (v[0], v[2], -v[1])


def read_geometry(geom):
    children = {c.name: c for c in geom.children}
    verts = children["Vertices"].props[0]
    pvi = children["PolygonVertexIndex"].props[0]

    normals = None
    nidx = None
    nbyvert = False
    ln = children.get("LayerElementNormal")
    if ln is not None:
        lnc = {c.name: c for c in ln.children}
        normals = lnc["Normals"].props[0]
        if "NormalsIndex" in lnc:
            nidx = lnc["NormalsIndex"].props[0]
        # Blender exports some meshes with per-vertex normals and others with
        # per-polygon-vertex ones; the index array means something different in
        # each case, and reading it as the wrong one crashes past its end.
        nbyvert = lnc.get("MappingInformationType") is not None and \
            lnc["MappingInformationType"].props[0] == "ByVertice"

    polys = []
    cur = []
    for i, p in enumerate(pvi):
        if p < 0:
            cur.append((~p, i))
            polys.append(cur)
            cur = []
        else:
            cur.append((p, i))

    mat_ids = None
    lm = children.get("LayerElementMaterial")
    if lm is not None:
        lmc = {c.name: c for c in lm.children}
        mat_ids = lmc["Materials"].props[0]
        # "AllSame" stores a single material id for the whole mesh
        if lmc.get("MappingInformationType") is not None and \
                lmc["MappingInformationType"].props[0] == "AllSame":
            mat_ids = [mat_ids[0]] * len(polys)

    tris = []
    for pid, poly in enumerate(polys):
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1], pid))
    return verts, normals, nidx, nbyvert, mat_ids, tris


def build(fbx_path, glb_path):
    nodes = parse(fbx_path)
    objs = [n for n in nodes if n.name == "Objects"][0]
    conns = [n for n in nodes if n.name == "Connections"][0]

    models = [o for o in objs.children if o.name == "Model"]
    geoms = {o.props[0]: o for o in objs.children if o.name == "Geometry"}
    mats = {o.props[0]: o.props[1].split("\x00")[0] for o in objs.children if o.name == "Material"}
    id_name = {o.props[0]: (o.props[1].split("\x00")[0] if len(o.props) > 1 and isinstance(o.props[1], str) else "")
               for o in objs.children}

    # model -> geometry, and model -> material list (connection order)
    model_geom = {}
    model_mats = {m.props[0]: [] for m in models}
    for c in conns.children:
        if c.props[0] != "OO":
            continue
        src, dst = c.props[1], c.props[2]
        if src in geoms and dst in model_geom:
            continue
        if src in geoms and dst in model_mats:
            model_geom[dst] = geoms[src]
        elif src in mats and dst in model_mats:
            model_mats[dst].append(mats[src])

    def pivot_of(model):
        for c in model.children:
            if c.name == "Properties70":
                for p in c.children:
                    if p.props[0] == "Lcl Translation":
                        return p.props[4:7]
        return [0.0, 0.0, 0.0]

    bin_parts, buffer_views, accessors, gl_meshes, gl_nodes, gl_materials = [], [], [], [], [], []
    mat_index = {}

    def add_mat(name):
        if name in mat_index:
            return mat_index[name]
        metal, rough, opacity, emissive = MATERIAL_OVERRIDES.get(name, (0.2, 0.5, 1.0, 0.0))
        col = MATERIAL_COLORS.get(name, (0.5, 0.5, 0.5))
        m = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [round(col[0], 4), round(col[1], 4), round(col[2], 4), opacity],
                "metallicFactor": metal,
                "roughnessFactor": rough,
            },
            "doubleSided": True,
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

    total_tris = 0
    for model in models:
        name = model.props[1].split("\x00")[0]
        geom = model_geom.get(model.props[0])
        if geom is None:
            gl_nodes.append({"name": name, "translation": list(rot(pivot_of(model)))})
            print("  %-18s (empty)" % name)
            continue
        verts, normals, nidx, nbyvert, mat_ids, tris = read_geometry(geom)
        mnames = model_mats.get(model.props[0]) or ["null"]

        groups = {}
        for a, b, c, pid in tris:
            mi = mat_ids[pid] if mat_ids else 0
            groups.setdefault(mi, []).append((a, b, c))

        primitives = []
        for mi, tri_list in sorted(groups.items()):
            pos, nor, idx = [], [], []
            cache = {}
            for t in tri_list:
                for vi, corner_i in t:
                    x, y, z = verts[vi * 3:vi * 3 + 3]
                    p = rot((x, y, z))
                    if normals is not None:
                        ni = nidx[vi if nbyvert else corner_i] if nidx is not None else vi
                        nn = rot((normals[ni * 3], normals[ni * 3 + 1], normals[ni * 3 + 2]))
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

            pv = add_data(struct.pack("<%df" % len(pos), *pos), 34962)
            nv = add_data(struct.pack("<%df" % len(nor), *nor), 34962)
            iv = add_data(struct.pack("<%dI" % len(idx), *idx), 34963)
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
            mname = mnames[mi] if mi < len(mnames) else mnames[0]
            primitives.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nor},
                               "indices": a_idx, "material": add_mat(mname)})
            total_tris += len(tri_list)

        gl_meshes.append({"name": name, "primitives": primitives})
        gl_nodes.append({"name": name, "mesh": len(gl_meshes) - 1,
                         "translation": list(rot(pivot_of(model)))})
        print("  %-18s %2d prims  %6d tris  pivot %s" %
              (name, len(primitives), sum(len(t) for t in groups.values()),
               [round(v, 3) for v in rot(pivot_of(model))]))

    # roles go into extras so the game does not have to hard-code name lists
    for n in gl_nodes:
        n.setdefault("extras", {})
        if n["name"] in WHEELS:
            n["extras"]["role"] = "wheel"
            n["extras"]["wheel"] = n["name"][6:]
        elif n["name"] in CALIPERS:
            n["extras"]["role"] = "caliper"
            n["extras"]["wheel"] = CALIPERS[n["name"]]
        elif n["name"] in PANELS:
            n["extras"]["role"] = "panel"
        elif n["name"] == "Body":
            n["extras"]["role"] = "body"
        else:
            n["extras"]["role"] = "static"

    root = {"name": "Volvo_S90", "children": list(range(len(gl_nodes)))}
    gl_nodes.append(root)
    gltf = {
        "asset": {"version": "2.0", "generator": "fbx2glb_parts.py (gamelikebeamng)"},
        "scene": 0,
        "scenes": [{"nodes": [len(gl_nodes) - 1]}],
        "nodes": gl_nodes,
        "meshes": gl_meshes,
        "materials": gl_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": sum(len(b) for b in bin_parts)}],
    }
    blob = b"".join(bin_parts)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf8")
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(blob)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(blob), 0x004E4942) + blob
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    with open(glb_path, "wb") as fh:
        fh.write(bytes(out))
    print("wrote %s  %.1f MB  %d tris  %d materials  %d nodes" %
          (glb_path, len(out) / 1e6, total_tris, len(gl_materials), len(gl_nodes)))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Volvo_S90_optimized.fbx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "game/assets/volvo.glb"
    build(src, dst)
