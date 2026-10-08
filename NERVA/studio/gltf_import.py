"""Extract reachable props (world-space AABBs) from a glTF/GLB environment.

Only the JSON is read: glTF requires POSITION accessors to carry min/max, so
bounding boxes come straight from the file with no mesh decoding. The client
renders the actual meshes itself; this gives the AGENTS a semantic scene.
glTF is Y-up meters, the same as the engine (Blender's exporter converts Z-up).
"""
import json
import math
import re
import struct

SKIP_NAME = re.compile(r"(floor|ground|wall|ceiling|roof|sky|plane|terrain|light|camera|collision|navmesh)", re.I)
FIXED_NAME = re.compile(r"(table|desk|counter|shelf|bench|sofa|couch|bed|cabinet|wardrobe|fridge|stove|sink|door|window|stairs)", re.I)
SURFACE_NAME = re.compile(r"(table|desk|counter|shelf|bench|sofa|couch|bed|cabinet)", re.I)
MAX_NODES = 5000


class GltfError(ValueError):
    pass


def parse(data: bytes):
    if len(data) < 12:
        raise GltfError("file too small to be glTF")
    if data[:4] == b"glTF":
        version, length = struct.unpack_from("<II", data, 4)
        if version != 2:
            raise GltfError(f"unsupported GLB version {version}")
        off = 12
        while off + 8 <= min(len(data), length):
            clen, ctype = struct.unpack_from("<II", data, off)
            if ctype == 0x4E4F534A:                     # 'JSON'
                try:
                    return json.loads(data[off + 8: off + 8 + clen].decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as e:
                    raise GltfError(f"bad GLB JSON chunk: {e}") from None
            off += 8 + clen + (-clen % 4)
        raise GltfError("GLB has no JSON chunk")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise GltfError("not a glTF/GLB file") from None


# 4x4 column-major helpers (lists of 16)
def _mul(a, b):
    return [sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4)]


def _trs(node):
    if "matrix" in node:
        m = node["matrix"]
        if len(m) != 16:
            raise GltfError("node matrix must have 16 numbers")
        return [float(v) for v in m]
    t = node.get("translation", [0, 0, 0]); q = node.get("rotation", [0, 0, 0, 1]); s = node.get("scale", [1, 1, 1])
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    r = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
         2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
         2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
         0, 0, 0, 1]
    for c in range(3):
        for rr in range(3):
            r[c * 4 + rr] *= s[c]
    r[12], r[13], r[14] = t
    return r


def _xform(m, p):
    return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
            m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]]


def props_from_gltf(doc, max_dim=6.0, min_dim=0.02):
    """-> (props, skipped, bounds). props: dicts for World.add_prop."""
    nodes, meshes, accs = doc.get("nodes", []), doc.get("meshes", []), doc.get("accessors", [])
    if len(nodes) > MAX_NODES:
        raise GltfError(f"too many nodes ({len(nodes)} > {MAX_NODES})")
    scenes = doc.get("scenes", [])
    roots = scenes[doc.get("scene", 0)].get("nodes", []) if scenes else list(range(len(nodes)))
    ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    found, skipped, seen = [], [], set()
    stack = [(r, ident) for r in roots]
    while stack:
        i, parent = stack.pop()
        if i in seen or not (0 <= i < len(nodes)):
            continue
        seen.add(i)
        n = nodes[i]
        world = _mul(parent, _trs(n))
        for c in n.get("children", []):
            stack.append((c, world))
        if "mesh" not in n:
            continue
        mesh = meshes[n["mesh"]]
        lo, hi = [math.inf] * 3, [-math.inf] * 3
        for prim in mesh.get("primitives", []):
            ai = prim.get("attributes", {}).get("POSITION")
            if ai is None or "min" not in accs[ai] or "max" not in accs[ai]:
                continue
            amin, amax = accs[ai]["min"], accs[ai]["max"]
            for cx in (amin[0], amax[0]):
                for cy in (amin[1], amax[1]):
                    for cz in (amin[2], amax[2]):
                        p = _xform(world, (cx, cy, cz))
                        for k in range(3):
                            lo[k], hi[k] = min(lo[k], p[k]), max(hi[k], p[k])
        name = n.get("name") or mesh.get("name") or f"mesh_{i}"
        if lo[0] == math.inf:
            skipped.append((name, "no POSITION bounds")); continue
        size = [hi[k] - lo[k] for k in range(3)]
        extras = n.get("extras") or {}
        forced = extras.get("nebula") == "prop"
        if not forced and (SKIP_NAME.search(name) or max(size) > max_dim):
            skipped.append((name, "structure/too large")); continue
        if max(size) < min_dim:
            skipped.append((name, "too small")); continue
        tags = [str(t) for t in extras.get("nebula_tags", [])] if isinstance(extras.get("nebula_tags"), list) else []
        if FIXED_NAME.search(name) and "fixed" not in tags:
            tags.append("fixed")
        if SURFACE_NAME.search(name) and "surface" not in tags:
            tags.append("surface")
        found.append({"name": name, "pos": [(lo[k] + hi[k]) / 2 for k in range(3)], "size": size,
                      "tags": tags, "kind": "mesh"})
    if found:
        bounds = {"min": [min(p["pos"][k] - p["size"][k] / 2 for p in found) for k in range(3)],
                  "max": [max(p["pos"][k] + p["size"][k] / 2 for p in found) for k in range(3)]}
    else:
        bounds = None
    return found, skipped, bounds


def import_bytes(data):
    return props_from_gltf(parse(data))
