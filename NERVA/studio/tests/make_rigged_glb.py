"""Run inside Blender: a Mixamo-named humanoid (segmented-box robot) with real skinning -> GLB.
Blender -Y is 'front', which the glTF exporter maps to +Z, so the model faces +Z in the file."""
import sys, bpy
from mathutils import Vector
out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add(enter_editmode=True); rig = bpy.context.active_object; rig.name = "Armature"
arm = rig.data
for b in list(arm.edit_bones): arm.edit_bones.remove(b)
P = "mixamorig:"
spec = [  # name, parent, head, tail
 ("Hips", None, (0, 0, 0.95), (0, 0, 1.05)), ("Spine", "Hips", (0, 0, 1.05), (0, 0, 1.30)), ("Neck", "Spine", (0, 0, 1.30), (0, 0, 1.42)), ("Head", "Neck", (0, 0, 1.42), (0, 0, 1.68)),
]
for s, x in (("Left", 1), ("Right", -1)):
    spec += [(f"{s}Shoulder", "Spine", (x * 0.04, 0, 1.38), (x * 0.20, 0, 1.38)),
             (f"{s}Arm", f"{s}Shoulder", (x * 0.20, 0, 1.38), (x * 0.50, 0, 1.38)),           # upper arm: 0.30
             (f"{s}ForeArm", f"{s}Arm", (x * 0.50, 0, 1.38), (x * 0.76, 0, 1.38)),            # forearm: 0.26
             (f"{s}Hand", f"{s}ForeArm", (x * 0.76, 0, 1.38), (x * 0.84, 0, 1.38)),
             (f"{s}HandIndex1", f"{s}Hand", (x * 0.84, 0, 1.38), (x * 0.90, 0, 1.38)),
             (f"{s}UpLeg", "Hips", (x * 0.10, 0, 0.95), (x * 0.10, 0, 0.52)),                 # thigh: 0.43
             (f"{s}Leg", f"{s}UpLeg", (x * 0.10, 0, 0.52), (x * 0.10, 0, 0.09)),              # shin: 0.43
             (f"{s}Foot", f"{s}Leg", (x * 0.10, 0, 0.09), (x * 0.10, -0.12, 0.03)),
             (f"{s}ToeBase", f"{s}Foot", (x * 0.10, -0.12, 0.03), (x * 0.10, -0.20, 0.0))]
for n, par, h, t in spec:
    e = arm.edit_bones.new(P + n); e.head, e.tail = Vector(h), Vector(t)
    if par: e.parent = arm.edit_bones[P + par]
bpy.ops.object.mode_set(mode='OBJECT')
meshes = []
for n, par, h, t in spec:
    a, b = Vector(h), Vector(t); mid = (a + b) / 2; ln = max((b - a).length, 0.05)
    bpy.ops.mesh.primitive_cube_add(size=1, location=mid); o = bpy.context.active_object; o.name = "m_" + n
    d = (b - a).normalized(); o.scale = (0.07 if abs(d.z) > 0.5 else ln, 0.07, ln if abs(d.z) > 0.5 else 0.07)
    if abs(d.y) > 0.5: o.scale = (0.07, ln, 0.07)
    bpy.ops.object.transform_apply(scale=True)
    vg = o.vertex_groups.new(name=P + n); vg.add(list(range(len(o.data.vertices))), 1.0, 'REPLACE'); meshes.append(o)
bpy.ops.object.select_all(action='DESELECT')
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]; bpy.ops.object.join()
body = bpy.context.active_object; body.name = "Body"
mod = body.modifiers.new("Armature", 'ARMATURE'); mod.object = rig; body.parent = rig
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_skins=True, export_yup=True)
print("EXPORTED", out)
