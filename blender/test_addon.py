"""Run: blender -b --factory-startup --python blender/test_addon.py
Exits non-zero on failure. Verifies the add-on against a real Blender."""
import os, sys, math
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dist"))
import nebula_blender
from nebula_blender import core

fails = 0
def check(cond, msg):
    global fails
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1

def make_arm(name="Rig", connected=True, loc=(0, 0, 0)):
    bpy.ops.object.armature_add(enter_editmode=True, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    arm = ob.data
    for b in list(arm.edit_bones):
        arm.edit_bones.remove(b)
    up = arm.edit_bones.new("upper"); up.head = Vector((0, 0, 1.4)); up.tail = Vector((0, 0, 1.1))
    fo = arm.edit_bones.new("fore");  fo.head = up.tail if connected else up.tail + Vector((0.05, 0, 0)); fo.tail = Vector((0, -0.2, 0.95))
    fo.parent = up; fo.use_connect = connected
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

clear_scene()
scene = bpy.context.scene
scene.render.fps = 30; scene.render.fps_base = 1.0
rig = make_arm()
target = bpy.data.objects.new("Target", None); scene.collection.objects.link(target)
target.location = (-0.15, -0.35, 1.30)

print("[bind]")
b = core.ArmBinding(rig, "upper", "fore", name="arm")
check(abs(b.L1 - 0.30) < 1e-4 and abs(b.L2 - 0.25) < 1e-3, f"bone lengths read from rig: {b.L1:.4f}, {b.L2:.4f}")
start_hand = b.hand_world().copy()
eng_hand = Vector(core.n2b(b.engine.world_positions(b.cid)[-1]))
check((eng_hand - start_hand).length < 1e-3, f"engine starts at the rig's real pose (err {(eng_hand-start_hand).length*1e3:.3f} mm)")

print("[bake]")
script = """# test
REACH 0.25 -0.30 1.15
WAIT 0.2
REACH @Target
RESET
"""
info = core.bake(b, script, start_frame=1, hold_s=0.2)
path, first, last = info["path"], info["first"], info["last"]
print(f"  baked frames {first}-{last} ({info['frames']} frames @30 fps)")
t1 = Vector((0.25, -0.30, 1.15))
d1 = min((p - t1).length for p in path)
check(d1 < 2e-3, f"hand reaches first target within {d1*1e3:.3f} mm")
d2 = min((p - target.location).length for p in path)
check(d2 < 2e-3, f"@Target resolved to the empty's location, reached within {d2*1e3:.3f} mm")
check((path[-1] - start_hand).length < 3e-3, f"RESET returns to the starting pose ({(path[-1]-start_hand).length*1e3:.3f} mm)")

# straightness of the first reach (natural bow is 4% by default)
i_arrive = next(i for i, p in enumerate(path) if (p - t1).length < 2e-3)
D = (t1 - start_hand).length; dirv = (t1 - start_hand).normalized()
dev = max(((p - start_hand) - dirv * (p - start_hand).dot(dirv)).length for p in path[:i_arrive + 1])
check(dev < 0.07 * D, f"first reach path deviation {100*dev/D:.1f}% of {D:.3f} m (bow 4% + slack)")

print("[scrub the baked animation -- not the bake loop]")
worst_hand = worst_len = worst_joint = 0.0
L1, L2 = b.L1, b.L2
for f in range(first, last + 1, 3):
    scene.frame_set(f); bpy.context.view_layer.update()
    mw = rig.matrix_world
    hand = mw @ b.pb_fo.tail
    worst_hand = max(worst_hand, (hand - path[f - first]).length)
    worst_len = max(worst_len, abs((mw @ b.pb_up.tail - mw @ b.pb_up.head).length - L1),
                    abs((mw @ b.pb_fo.tail - mw @ b.pb_fo.head).length - L2))
    worst_joint = max(worst_joint, (mw @ b.pb_up.tail - mw @ b.pb_fo.head).length)
check(worst_hand < 1.5e-3, f"keyframes reproduce the baked hand path (worst {worst_hand*1e3:.3f} mm)")
check(worst_len < 1e-4, f"bone lengths preserved through animation (worst {worst_len*1e3:.4f} mm)")
check(worst_joint < 1e-4, f"elbow stays connected (worst {worst_joint*1e3:.4f} mm)")
check(scene.frame_end >= last, "scene end frame extended to cover the bake")
b.close()

print("[errors are reported, not swallowed]")
b2 = core.ArmBinding(rig, "upper", "fore", name="arm")
for bad, frag in (("REACH 1 2", "line 1"), ("# c\nDANCE 1 2 3", "line 2"), ("REACH nan 0 0", "line 1"), ("REACH @Nope", "not found")):
    try:
        core.bake(b2, bad, start_frame=1)
        check(False, f"{bad!r} should fail")
    except ValueError as e:
        check(frag in str(e), f"{bad!r} -> {e}")
b2.close()
clear_scene(); bpy.context.scene.render.fps = 30
bad_rig = make_arm("Bad", connected=False)
try:
    core.ArmBinding(bad_rig, "upper", "fore"); check(False, "unconnected chain should be rejected")
except core.BindError as e:
    check("connected" in str(e), f"unconnected forearm rejected: {e}")
try:
    core.ArmBinding(bad_rig, "upper", "nope"); check(False, "missing bone")
except core.BindError as e:
    check("not found" in str(e), f"missing bone rejected: {e}")

print("[operator + panel registration]")
clear_scene(); scene = bpy.context.scene; scene.render.fps = 30
rig = make_arm("Rig2", loc=(1.0, 2.0, 0.0))            # armature NOT at the origin, to exercise matrix_world
nebula_blender.register()
s = scene.nebula
s.armature, s.bone_upper, s.bone_fore = rig, "upper", "fore"
bpy.ops.nebula.example_script()
check(s.script is not None and "REACH" in s.script.as_string(), "example script created and assigned")
sh = rig.matrix_world @ rig.pose.bones["upper"].head
s.script.clear(); s.script.write(f"REACH {sh.x+0.25} {sh.y-0.3} {sh.z-0.2}\n")
res = bpy.ops.nebula.bake()
check(res == {'FINISHED'}, f"nebula.bake -> {res}")
act = rig.animation_data.action if rig.animation_data else None
check(act is not None, "an action was created on the armature")
if act is not None and hasattr(act, "fcurves"):
    check(len(act.fcurves) == 8, f"8 curves keyed (2 bones x quaternion): {len(act.fcurves)}")
scene.frame_set(scene.frame_end); bpy.context.view_layer.update()
hand = rig.matrix_world @ rig.pose.bones["fore"].tail
want = Vector((sh.x + 0.25, sh.y - 0.3, sh.z - 0.2))
check((hand - want).length < 2e-3, f"offset armature: final hand {(hand-want).length*1e3:.3f} mm from target")
nebula_blender.unregister()

print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
