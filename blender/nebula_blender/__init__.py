bl_info = {
    "name": "Nebula IK",
    "author": "NANITY",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > Nebula",
    "description": "Drive a limb with the Nebula IK engine from text commands and bake keyframes",
    "category": "Animation",
}

import bpy
from bpy.props import BoolProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

EXAMPLE = """# Nebula script -- Blender world coordinates (Z up). Edit the numbers for your rig.
REACH 0.30 -0.25 1.20
WAIT 0.4
REACH 0.10 -0.40 1.55
RESET
"""


def _is_armature(self, obj):
    return obj.type == 'ARMATURE'


class NebulaSettings(PropertyGroup):
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_is_armature)
    bone_upper: StringProperty(name="Upper", description="Upper arm / thigh bone")
    bone_fore: StringProperty(name="Fore", description="Forearm / shin bone (connected child)")
    script: PointerProperty(name="Script", type=bpy.types.Text)
    idle: BoolProperty(name="Idle wander", default=False,
                       description="Add tiny non-repeating hand drift while holding")


class NEBULA_OT_example(Operator):
    bl_idname = "nebula.example_script"
    bl_label = "Create example script"

    def execute(self, context):
        txt = bpy.data.texts.get("nebula_script") or bpy.data.texts.new("nebula_script")
        txt.clear(); txt.write(EXAMPLE)
        context.scene.nebula.script = txt
        return {'FINISHED'}


class NEBULA_OT_bake(Operator):
    bl_idname = "nebula.bake"
    bl_label = "Bake to keyframes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import core
        s = context.scene.nebula
        if s.script is None:
            self.report({'ERROR'}, "Pick a script (Text datablock)")
            return {'CANCELLED'}
        binding = None
        try:
            binding = core.ArmBinding(s.armature, s.bone_upper, s.bone_fore, idle=s.idle)
            info = core.bake(binding, s.script.as_string())
        except (core.BindError, ValueError, RuntimeError, FileNotFoundError, OSError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        finally:
            if binding:
                binding.close()
        self.report({'INFO'}, f"Baked frames {info['first']}-{info['last']}")
        return {'FINISHED'}


class NEBULA_PT_panel(Panel):
    bl_label = "Nebula"
    bl_idname = "NEBULA_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Nebula"

    def draw(self, context):
        s, col = context.scene.nebula, self.layout.column(align=True)
        col.prop(s, "armature")
        if s.armature:
            col.prop_search(s, "bone_upper", s.armature.pose, "bones")
            col.prop_search(s, "bone_fore", s.armature.pose, "bones")
        col.separator()
        col.prop(s, "script")
        col.operator("nebula.example_script", icon='TEXT')
        col.prop(s, "idle")
        col.separator()
        col.operator("nebula.bake", icon='KEYFRAME')


_classes = (NebulaSettings, NEBULA_OT_example, NEBULA_OT_bake, NEBULA_PT_panel)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.nebula = PointerProperty(type=NebulaSettings)


def unregister():
    del bpy.types.Scene.nebula
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
