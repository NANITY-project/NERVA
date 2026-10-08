"""Run inside Blender: builds a small kitchen, exports GLB, and writes ground-truth world AABBs
(converted to Y-up) as JSON next to it."""
import json, sys, bpy
from mathutils import Vector, Euler
out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)

def cube(name, loc, scale, rot=(0, 0, 0), parent=None, props=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object; o.name = name; o.scale = scale; o.rotation_euler = Euler(rot)
    if parent: o.parent = parent; o.matrix_parent_inverse = parent.matrix_world.inverted()
    for k, v in (props or {}).items(): o[k] = v
    return o

table = cube("Table", (2.0, -1.0, 0.4), (1.2, 0.6, 0.8))
cup = cube("Cup", (2.1, -1.0, 0.85), (0.08, 0.08, 0.1))
chair = cube("Chair", (0.0, 0.0, 0.25), (0.45, 0.45, 0.5), rot=(0, 0, 0.5))
crate = cube("Crate", (-2.0, 1.0, 0.2), (0.4, 0.4, 0.4), parent=table)           # child of the table: parent transform must apply
tagged = cube("Lantern", (0.0, 2.0, 0.15), (0.15, 0.15, 0.3), props={"nebula_tags": ["light", "fragile"]})
floor = cube("Floor", (0, 0, -0.05), (10, 10, 0.1))
wall = cube("Wall_North", (0, 5, 1.5), (10, 0.2, 3))
bpy.context.view_layer.update()

truth = {}
for o in bpy.data.objects:
    cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = [min(c[i] for c in cs) for i in range(3)]; hi = [max(c[i] for c in cs) for i in range(3)]
    # Blender (x, y, z-up) -> glTF (x, z, -y): min/max swap for the y/z axes
    truth[o.name] = {"min": [lo[0], lo[2], -hi[1]], "max": [hi[0], hi[2], -lo[1]]}
bpy.ops.export_scene.gltf(filepath=out + ".glb", export_format='GLB', export_extras=True, export_apply=True)
json.dump(truth, open(out + ".truth.json", "w"))
print("EXPORTED", out)
