"""ctypes binding for libnebula (the Nebula IK engine C API).

Used by both the Blender add-on and the Studio server. Coordinates in/out are
the ENGINE's (Y-up, bones along +X); hosts convert at the boundary.
"""
import ctypes as C
import json
import os
import sys

_F3 = C.c_float * 3


def _find_lib():
    env = os.environ.get("NEBULA_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    name = {"win32": "nebula.dll", "darwin": "libnebula.dylib"}.get(sys.platform, "libnebula.so")
    for d in (here, os.path.join(here, "..", "build"), os.path.join(here, "..")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found; build it (make lib) or set NEBULA_LIB")


class Engine:
    TWO_BONE, FABRIK = 0, 1

    def __init__(self, lib_path=None):
        L = self._L = C.CDLL(lib_path or _find_lib())
        P = C.c_void_p
        sig = {
            "nebula_abi_version": (C.c_int, []),
            "nebula_create": (P, []),
            "nebula_destroy": (None, [P]),
            "nebula_add_chain": (C.c_int, [P, C.c_char_p, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int,
                                           C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_float, C.c_float]),
            "nebula_find_chain": (C.c_int, [P, C.c_char_p]),
            "nebula_chain_count": (C.c_int, [P]),
            "nebula_joint_count": (C.c_int, [P, C.c_int]),
            "nebula_set_anchor": (C.c_int, [P, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float)]),
            "nebula_teleport": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_set_pole": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_set_rest": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_set_elbow_limits": (C.c_int, [P, C.c_char_p, C.c_float, C.c_float]),
            "nebula_set_max_bend": (C.c_int, [P, C.c_char_p, C.c_int, C.c_float]),
            "nebula_command": (C.c_int, [P, C.c_char_p]),
            "nebula_reach": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_set_param": (C.c_int, [P, C.c_char_p, C.c_float]),
            "nebula_update": (None, [P, C.c_float]),
            "nebula_world_positions": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_local_rotations": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
            "nebula_is_settled": (C.c_int, [P, C.c_int]),
            "nebula_hand_velocity": (C.c_int, [P, C.c_int, C.POINTER(C.c_float)]),
        }
        for n, (res, args) in sig.items():
            f = getattr(L, n)
            f.restype, f.argtypes = res, args
        if L.nebula_abi_version() != 2:
            raise RuntimeError("libnebula ABI mismatch (rebuild it: make lib)")
        self._h = L.nebula_create()
        if not self._h:
            raise MemoryError("nebula_create failed")

    def close(self):
        if getattr(self, "_h", None):
            self._L.nebula_destroy(self._h)
            self._h = None

    __del__ = close

    # -- chains ---------------------------------------------------------
    def add_chain(self, name, root, bone_lengths, pole=(0, 0, 1), solver=0, torso_lean=0.0,
                  max_speed=1.2, root_quat=None):
        n = len(bone_lengths)
        lens = (C.c_float * n)(*bone_lengths)
        q = (C.c_float * 4)(*root_quat) if root_quat else None
        cid = self._L.nebula_add_chain(self._h, name.encode(), _F3(*root), q, solver, n, lens,
                                       _F3(*pole), torso_lean, max_speed)
        if cid < 0:
            raise ValueError(f"add_chain({name!r}) rejected (duplicate name, bad bone length, or too many joints)")
        return cid

    def find_chain(self, name):
        return self._L.nebula_find_chain(self._h, name.encode())

    def chain_count(self):
        return self._L.nebula_chain_count(self._h)

    def set_anchor(self, chain, pos, quat=None):
        return bool(self._L.nebula_set_anchor(self._h, chain, _F3(*pos), (C.c_float * 4)(*quat) if quat else None))

    def teleport(self, chain, hand):
        return bool(self._L.nebula_teleport(self._h, chain, _F3(*hand)))

    def set_pole(self, chain, direction):
        return bool(self._L.nebula_set_pole(self._h, chain, _F3(*direction)))

    def set_rest(self, chain, hand):
        return bool(self._L.nebula_set_rest(self._h, chain, _F3(*hand)))

    def set_elbow_limits(self, name, min_rad, max_rad):
        return bool(self._L.nebula_set_elbow_limits(self._h, name.encode(), min_rad, max_rad))

    def set_max_bend(self, name, joint, rad):
        return bool(self._L.nebula_set_max_bend(self._h, name.encode(), joint, rad))

    def set_param(self, key, value):
        return bool(self._L.nebula_set_param(self._h, key.encode(), float(value)))

    # -- motion ---------------------------------------------------------
    def command(self, line):
        return bool(self._L.nebula_command(self._h, line.encode()))

    def reach(self, chain, target):
        return bool(self._L.nebula_reach(self._h, chain, _F3(*target)))

    def update(self, dt):
        self._L.nebula_update(self._h, dt)

    def is_settled(self, chain):
        return bool(self._L.nebula_is_settled(self._h, chain))

    def hand_velocity(self, chain):
        o = _F3()
        self._L.nebula_hand_velocity(self._h, chain, o)
        return tuple(o)

    def world_positions(self, chain):
        n = self._L.nebula_joint_count(self._h, chain)
        if n < 0:
            raise IndexError(chain)
        buf = (C.c_float * (3 * (n + 1)))()
        self._L.nebula_world_positions(self._h, chain, buf)
        return [tuple(buf[3 * i:3 * i + 3]) for i in range(n + 1)]

    def local_rotations(self, chain):
        n = self._L.nebula_joint_count(self._h, chain)
        buf = (C.c_float * (4 * n))()
        self._L.nebula_local_rotations(self._h, chain, buf)
        return [tuple(buf[4 * i:4 * i + 4]) for i in range(n)]


class Body:
    """A full humanoid body (see include/ik_body.hpp). Coordinates in world metres, Y up; at yaw 0 it faces +Z, its left is +X.

    Commands are text lines and every refusal explains itself:
        HAND <left|right> x y z | rest      FOOT <left|right> x y z [yaw_deg]     PELVIS x y z | auto
        FACE x z | auto                     LOOK x y z | auto                      TORSO x y z | auto
        GRIP <left|right> 0..1              STAND                                  BALANCE assist|none
    """

    def __init__(self, lib_path=None, scale=1.0, finger_lod=1, assist=True):
        L = self._L = C.CDLL(lib_path or _find_lib())
        P, F, I, S = C.c_void_p, C.c_float, C.c_int, C.c_char_p
        sig = {
            "nebula_abi_version": (I, []),
            "nebula_body_create": (P, [F, I, I]), "nebula_body_destroy": (None, [P]),
            "nebula_body_command": (I, [P, S, C.c_char_p, I]), "nebula_body_update": (None, [P, F]),
            "nebula_body_set_param": (I, [P, S, F]), "nebula_body_add_surface": (I, [P, F, F, F, F, F]),
            "nebula_body_clear_surfaces": (None, [P]), "nebula_body_bone_count": (I, [P]),
            "nebula_body_bone_name": (S, [P, I]), "nebula_body_bone_parent": (I, [P, I]),
            "nebula_body_bones": (I, [P, C.POINTER(C.c_float)]), "nebula_body_status_json": (I, [P, C.c_char_p, I]),
        }
        for n, (res, args) in sig.items():
            f = getattr(L, n); f.restype, f.argtypes = res, args
        if L.nebula_abi_version() != 2:
            raise RuntimeError("libnebula ABI mismatch (rebuild it: make lib)")
        self._h = L.nebula_body_create(scale, finger_lod, 1 if assist else 0)
        if not self._h:
            raise ValueError("could not create a body (scale must be 0.2..3)")
        n = L.nebula_body_bone_count(self._h)
        self.names = [L.nebula_body_bone_name(self._h, i).decode() for i in range(n)]
        self.parents = [L.nebula_body_bone_parent(self._h, i) for i in range(n)]
        self.time = 0.0

    def close(self):
        if getattr(self, "_h", None):
            self._L.nebula_body_destroy(self._h)
            self._h = None

    __del__ = close

    def command(self, line):
        """-> (accepted: bool, message: str). The message is the body's own explanation."""
        buf = C.create_string_buffer(512)
        ok = self._L.nebula_body_command(self._h, line.encode(), buf, 512)
        return bool(ok), buf.value.decode(errors="replace")

    def update(self, dt=1 / 60):
        self._L.nebula_body_update(self._h, dt)
        self.time += dt

    def run(self, seconds, dt=1 / 60, until=None):
        """Advance the simulation; stop early when until(self) is true. Returns True if `until` fired."""
        t = 0.0
        while t < seconds:
            self.update(dt); t += dt
            if until is not None and until(self):
                return True
        return False

    def set_param(self, key, value):
        return bool(self._L.nebula_body_set_param(self._h, key.encode(), float(value)))

    def add_surface(self, x0, x1, z0, z1, y):
        return bool(self._L.nebula_body_add_surface(self._h, x0, x1, z0, z1, y))

    def clear_surfaces(self):
        self._L.nebula_body_clear_surfaces(self._h)

    def bones(self):
        n = len(self.names)
        buf = (C.c_float * (10 * n))()
        self._L.nebula_body_bones(self._h, buf)
        return [{"name": self.names[i], "parent": self.parents[i], "head": tuple(buf[10 * i:10 * i + 3]),
                 "tail": tuple(buf[10 * i + 3:10 * i + 6]), "quat": tuple(buf[10 * i + 6:10 * i + 10])} for i in range(n)]

    def status(self):
        """State + the feedback events since the last call, as a dict (this is what a model gets to read)."""
        size = 4096
        while True:
            buf = C.create_string_buffer(size)
            need = self._L.nebula_body_status_json(self._h, buf, size)
            if need < size:
                return json.loads(buf.value.decode())
            size = need + 1                      # (events were consumed by the truncated call only if it fit; sizes are small in practice)
