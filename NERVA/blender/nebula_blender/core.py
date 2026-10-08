"""Nebula <-> Blender bridge: bind a 2-bone limb, run engine commands, bake keyframes.

No UI here (so it can be driven headlessly and tested). The engine is Y-up with
bones along +X; Blender is Z-up. Conversion happens ONLY in b2n()/n2b().
Poses are applied by aiming each pose bone at the engine's joint positions
(shortest-arc, parent first), so the rig's own bone roll/axis conventions never
matter. Keyframes go in through PoseBone.keyframe_insert, which is stable across
Blender's old and new (layered) action APIs.
"""
import math

import bpy
from mathutils import Matrix, Vector

from . import nebula as nebula_mod

DEFAULT_POLE_B = Vector((0.0, 0.5, -1.0))   # Blender space: elbows bend down and back


def b2n(v):
    """Blender world (x, y, z-up) -> engine (x, y-up, z)."""
    return (v.x, v.z, -v.y)


def n2b(t):
    return Vector((t[0], -t[2], t[1]))


class BindError(ValueError):
    pass


class ArmBinding:
    """A two-bone chain (upper arm/thigh + forearm/shin) driven by Nebula."""

    def __init__(self, arm_obj, upper, fore, name="limb", idle=False, lib_path=None):
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            raise BindError("pick an armature object")
        pbs = arm_obj.pose.bones
        for b in (upper, fore):
            if b not in pbs:
                raise BindError(f"bone {b!r} not found in {arm_obj.name!r}")
        self.obj, self.name = arm_obj, name
        self.pb_up, self.pb_fo = pbs[upper], pbs[fore]
        for pb in (self.pb_up, self.pb_fo):
            pb.rotation_mode = 'QUATERNION'
        self._refresh()
        mw = arm_obj.matrix_world
        sh = mw @ self.pb_up.head
        el = mw @ self.pb_up.tail
        el2 = mw @ self.pb_fo.head
        wr = mw @ self.pb_fo.tail
        L1, L2 = (el - sh).length, (wr - el).length
        if L1 < 1e-6 or L2 < 1e-6:
            raise BindError("zero-length bone")
        if (el - el2).length > 1e-3 * (L1 + L2):
            raise BindError("forearm must be connected to the upper arm (head == parent tail)")
        self.L1, self.L2 = L1, L2

        # Elbow bend direction from the current pose, else a sensible default.
        mid = (sh + wr) * 0.5
        bend = el - mid
        pole_b = bend.normalized() if bend.length > 0.02 * (L1 + L2) else DEFAULT_POLE_B.normalized()

        self.engine = nebula_mod.Engine(lib_path)
        self.cid = self.engine.add_chain(name, b2n(sh), [L1, L2], pole=b2n(pole_b),
                                         torso_lean=0.0, max_speed=1.2)
        # The rig root is fixed to the armature, so no breathing / lean on the engine side.
        self.engine.set_param("breath_amp_m", 0.0)
        self.engine.set_param("idle_amp_m", 0.0015 if idle else 0.0)
        self.engine.teleport(self.cid, b2n(wr))     # start from the pose the rig is really in
        self.engine.set_rest(self.cid, b2n(wr))

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _refresh():
        bpy.context.view_layer.update()

    def hand_world(self):
        self._refresh()
        return self.obj.matrix_world @ self.pb_fo.tail

    def shoulder_world(self):
        self._refresh()
        return self.obj.matrix_world @ self.pb_up.head

    def _aim(self, pb, target_world):
        """Rotate pose bone `pb` (parent already posed) so its tail points at target_world."""
        inv = self.obj.matrix_world.inverted()
        m = pb.matrix
        head = m.translation.copy()
        want = ((inv @ target_world) - head)
        if want.length < 1e-9:
            return
        want.normalize()
        rot = m.to_quaternion()
        cur = (rot @ Vector((0.0, 1.0, 0.0))).normalized()
        rot = cur.rotation_difference(want) @ rot
        pb.matrix = Matrix.Translation(head) @ rot.to_matrix().to_4x4()

    def apply(self):
        """Pose the rig from the engine's current joint positions."""
        pos = [n2b(p) for p in self.engine.world_positions(self.cid)]
        self.engine.set_anchor(self.cid, b2n(self.shoulder_world()))
        self._aim(self.pb_up, pos[1])
        self._refresh()
        self._aim(self.pb_fo, pos[2])
        self._refresh()

    def key(self, frame):
        for pb in (self.pb_up, self.pb_fo):
            pb.keyframe_insert("rotation_quaternion", frame=frame)

    def close(self):
        self.engine.close()


def _target_from_tokens(tokens):
    """['0.3','-0.2','1.1'] or ['@Empty'] -> Blender-space Vector."""
    if len(tokens) == 1 and tokens[0].startswith('@'):
        ob = bpy.data.objects.get(tokens[0][1:])
        if ob is None:
            raise ValueError(f"object {tokens[0][1:]!r} not found")
        return ob.matrix_world.translation.copy()
    if len(tokens) != 3:
        raise ValueError("expected 3 numbers or @ObjectName")
    v = [float(t) for t in tokens]
    if not all(math.isfinite(x) for x in v):
        raise ValueError("non-finite coordinate")
    return Vector(v)


def bake(binding, script, start_frame=None, hold_s=0.25, max_seconds=60.0, fps=None):
    """Run a command script against the binding and bake keyframes.

    Script lines (Blender world coordinates, Z up):
        REACH x y z | REACH @ObjectName     (POINT is an alias)
        WAIT seconds
        RESET
    Blank lines and '#' comments are ignored. Returns a dict with the baked frame
    range and the per-frame hand path (Blender world space) for inspection.
    """
    scene = bpy.context.scene
    fps = fps or (scene.render.fps / scene.render.fps_base)
    dt = 1.0 / fps
    frame = scene.frame_current if start_frame is None else start_frame
    first = frame
    limit = int(max_seconds * fps)
    path, frames = [], 0
    eng, cid = binding.engine, binding.cid

    def tick():
        nonlocal frame, frames
        if frames >= limit:
            raise RuntimeError(f"bake exceeded {max_seconds:.0f}s; check the script")
        eng.update(dt)
        binding.apply()
        binding.key(frame)
        path.append(binding.hand_world().copy())
        frame += 1
        frames += 1

    # Key the starting pose.
    binding.apply(); binding.key(frame); path.append(binding.hand_world().copy()); frame += 1

    for ln, raw in enumerate(script.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        cmd, args = parts[0].upper(), parts[1:]
        try:
            if cmd in ("REACH", "POINT"):
                t = b2n(_target_from_tokens(args))
                if not eng.command(f"REACH {binding.name} {t[0]!r} {t[1]!r} {t[2]!r}"):
                    raise ValueError("engine rejected the target")
                while not eng.is_settled(cid):
                    tick()
                for _ in range(int(hold_s * fps)):
                    tick()
            elif cmd == "WAIT":
                secs = float(args[0])
                for _ in range(int(secs * fps)):
                    tick()
            elif cmd == "RESET":
                eng.command(f"RESET {binding.name}")
                while not eng.is_settled(cid):
                    tick()
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"script line {ln}: {raw.strip()!r}: {e}") from None

    last = frame - 1
    if bpy.context.scene.frame_end < last:
        bpy.context.scene.frame_end = last
    return {"first": first, "last": last, "frames": frames + 1, "path": path}
