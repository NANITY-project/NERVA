"""Nebula Studio simulation: characters, props, actions, takes. No networking here.

One thread (the sim loop, or a test calling World.step) owns the engine and all
state. Everything else talks to it through World.submit(), so there is no shared
mutable state and no engine locking. World coordinates are Y-up meters, the same
as the engine (and glTF).
"""
import math
import os
import queue
import re
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import nebula  # noqa: E402

# --- body model ------------------------------------------------------------
SHOULDER_Y, SHOULDER_W = 1.42, 0.19
L_UP, L_FORE = 0.30, 0.27
REACH_COMFORT = 0.50          # planning radius; the engine can stretch to 0.57 (+ lean)
WALK_SPEED, TURN_RATE = 1.3, 4.0
GRAB_RANGE = 0.14
SNAPSHOT_HZ = 30
COLORS = ["#c70000", "#e8e2d9", "#7a9cc6", "#c9a24b", "#8fbf9f", "#b07aa1"]


class ActionError(ValueError):
    """A command the world refuses; the message goes back to the agent."""


def slug(s):
    return re.sub(r"[^a-z0-9_]+", "_", str(s).lower()).strip("_") or "x"


# --- tiny vector helpers (tuples) ----------------------------------------
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dist(a, b): return math.dist(a, b)
def norm(a):
    l = math.hypot(*a)
    return (0.0, 0.0, 0.0) if l < 1e-9 else (a[0] / l, a[1] / l, a[2] / l)
def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class Prop:
    def __init__(self, name, pos, size=(0.2, 0.2, 0.2), tags=(), kind="box", color="#a60000"):
        self.name, self.pos, self.size = slug(name), [float(v) for v in pos], [float(v) for v in size]
        self.tags, self.kind, self.color = list(tags), kind, color
        self.held_by = None           # (char_id, side)
        self.vy = 0.0

    @property
    def graspable(self):
        return max(self.size) <= 0.35 and "fixed" not in self.tags

    def aabb(self):
        h = [s / 2 for s in self.size]
        return [self.pos[i] - h[i] for i in range(3)], [self.pos[i] + h[i] for i in range(3)]

    def top(self):
        return self.pos[1] + self.size[1] / 2

    def nearest_point(self, p):
        lo, hi = self.aabb()
        return tuple(min(max(p[i], lo[i]), hi[i]) for i in range(3))

    def contains_xz(self, x, z, margin=0.0):
        lo, hi = self.aabb()
        return lo[0] - margin <= x <= hi[0] + margin and lo[2] - margin <= z <= hi[2] + margin


class Hand:
    def __init__(self, side, cid):
        self.side, self.cid = side, cid
        self.mode = "rest"            # rest | carry | offer | target
        self.holding = None           # prop name
        self.last_cmd = None


class Character:
    def __init__(self, cid, name, persona, x, z, yaw, color):
        self.id, self.name, self.persona, self.color = cid, name, persona, color
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
        self.crouch = self.crouch_target = 0.0
        self.walk_phase = 0.0
        self.gaze = (math.sin(yaw), 0.0, math.cos(yaw))
        self.gaze_target = None       # ('prop'|'char', id)
        self.hands = {}
        self.queue = deque()
        self.current = None
        self.speech, self.speech_until = "", 0.0
        self.events = []
        self.active = True
        self.agent_on = False

    # character frame: forward = (sin yaw, 0, cos yaw); left = (cos yaw, 0, -sin yaw)
    def fwd(self): return (math.sin(self.yaw), 0.0, math.cos(self.yaw))
    def left(self): return (math.cos(self.yaw), 0.0, -math.sin(self.yaw))
    def pos(self): return (self.x, 0.0, self.z)
    def shoulder_y(self): return SHOULDER_Y - self.crouch

    def shoulder(self, side):
        s = 1.0 if side == "left" else -1.0
        return add((self.x, self.shoulder_y(), self.z), mul(self.left(), s * SHOULDER_W))

    def rest_point(self, side):
        s = 1.0 if side == "left" else -1.0
        return add(add(self.shoulder(side), mul(self.left(), s * 0.07)), add(mul(self.fwd(), 0.05), (0, -0.49, 0)))

    def carry_point(self, side):
        s = 1.0 if side == "left" else -1.0
        return add(add(self.shoulder(side), mul(self.left(), -s * 0.04)), add(mul(self.fwd(), 0.30), (0, -0.28, 0)))

    def pole(self, side):
        s = 1.0 if side == "left" else -1.0
        return add(add(mul(self.fwd(), -0.6), mul(self.left(), s * 0.25)), (0, -1.0, 0))

    def eye(self): return (self.x, 1.62 - self.crouch, self.z)

    def note(self, text):
        self.events.append(text)
        del self.events[:-12]


# --- actions ---------------------------------------------------------------
class Action:
    timeout = 20.0
    label = ""

    def start(self, w, c): pass
    def update(self, w, c, dt): return True
    def __str__(self): return self.label


class SayAct(Action):
    def __init__(self, text): self.text, self.label, self.t = text, "SAY", 0.0
    def start(self, w, c):
        c.speech, c.speech_until = self.text, w.t + 1.0 + 0.055 * len(self.text)
        self.dur = min(0.4 + 0.05 * len(self.text), 6.0)
        w.say(c, self.text)
    def update(self, w, c, dt):
        self.t += dt
        return self.t >= self.dur


class WaitAct(Action):
    def __init__(self, s): self.s, self.t, self.label = max(0.0, min(float(s), 10.0)), 0.0, "WAIT"
    def update(self, w, c, dt):
        self.t += dt
        return self.t >= self.s


class LookAct(Action):
    def __init__(self, ref): self.ref, self.label = ref, f"LOOK_AT {ref}"
    def start(self, w, c):
        kind, obj = w.resolve(self.ref)
        if kind is None:
            raise ActionError(f"unknown target '{self.ref}'")
        c.gaze_target = (kind, obj.id if kind == "char" else obj.name)


class WalkAct(Action):
    timeout = 25.0

    def __init__(self, ref=None, xz=None, stop=None, face=None):
        self.ref, self.xz, self.stop, self.face = ref, xz, stop, face
        self.label = f"WALK_TO {ref or xz}"

    def start(self, w, c):
        self.goal = None
        if self.xz is not None:
            self.goal = self.xz
        else:
            kind, obj = w.resolve(self.ref)
            if kind is None:
                raise ActionError(f"unknown target '{self.ref}'")
            self.face = self.face or (obj.pos if kind == "prop" else obj.pos())
            near = obj.nearest_point((c.x, obj.pos[1], c.z)) if kind == "prop" else obj.pos()
            stop = self.stop if self.stop is not None else (0.45 if kind == "prop" else 0.9)
            vx, vz = c.x - near[0], c.z - near[2]
            d = math.hypot(vx, vz)
            if d < 1e-6:
                vx, vz, d = -math.sin(c.yaw), -math.cos(c.yaw), 1.0
            self.goal = (near[0] + vx / d * stop, near[2] + vz / d * stop)
            self.moving_target = obj if kind == "char" else None
        c.crouch_target = 0.0
        self.turning_only = False

    def update(self, w, c, dt):
        return w.walk_step(c, self.goal, dt, self.face)


class ReachAct(Action):
    """REACH/GRAB/POINT target. Auto-approaches, crouches if needed, then reaches."""
    timeout = 30.0

    def __init__(self, side, ref=None, point=None, grab=False, point_at=False):
        self.side, self.ref, self.point, self.grab, self.point_at = side, ref, point, grab, point_at
        self.label = ("GRAB" if grab else "POINT" if point_at else "REACH") + f" {side} {ref or point}"

    def start(self, w, c):
        h = c.hands[self.side]
        self.prop = None
        if self.ref is not None:
            kind, obj = w.resolve(self.ref)
            if kind is None:
                raise ActionError(f"unknown target '{self.ref}'")
            if kind == "prop":
                self.prop = obj
        if self.grab:
            if self.prop is None:
                raise ActionError("GRAB needs a prop name")
            if not self.prop.graspable:
                raise ActionError(f"{self.prop.name} is too big or fixed to grab")
            if h.holding:
                raise ActionError(f"{self.side} hand already holds {h.holding}; DROP it first")
            if self.prop.held_by == (c.id, self.side):
                raise ActionError("already holding that")
        self.phase = "approach"
        self.t_hold = 0.0
        self.walk = None
        self._plan(w, c)

    def _target(self, w, c):
        if self.point is not None:
            return self.point
        kind, obj = w.resolve(self.ref)
        if kind == "prop":
            p = tuple(obj.pos) if (obj.graspable or self.point_at) else obj.nearest_point(c.shoulder(self.side))
            if not (obj.graspable or self.point_at):
                p = (p[0], min(p[1], obj.top()) + 0.02, p[2])
            return p
        return add(obj.pos(), (0, 1.2, 0))          # a character: chest height

    def _plan(self, w, c):
        P = self._target(w, c)
        if self.point_at:
            self.P = P
            self.phase = "reach"
            return
        want_sh = max(P[1] + 0.30, SHOULDER_Y - 0.70)
        self.crouch_needed = max(0.0, SHOULDER_Y - want_sh) if P[1] + 0.30 < SHOULDER_Y else 0.0
        dv = abs((SHOULDER_Y - self.crouch_needed) - P[1])
        hmax = math.sqrt(max(REACH_COMFORT ** 2 - dv ** 2, 0.0))
        dxz = math.hypot(P[0] - c.x, P[2] - c.z)
        self.P = P
        if dxz > hmax:
            stop = max(0.12, hmax * 0.8)
            vx, vz = c.x - P[0], c.z - P[2]
            d = math.hypot(vx, vz) or 1.0
            self.walk = WalkAct(xz=(P[0] + vx / d * stop, P[2] + vz / d * stop), face=P)
            self.walk.start(w, c)            # walk upright; crouch on arrival
            self.phase = "approach"
        else:
            self.phase = "reach"
            c.crouch_target = self.crouch_needed

    def update(self, w, c, dt):
        h = c.hands[self.side]
        if self.phase == "approach":
            if self.walk is None or self.walk.update(w, c, dt):
                self.phase = "reach"
                c.crouch_target = getattr(self, "crouch_needed", 0.0)
                w.face_toward(c, self.P, snap=False)
            else:
                return False
        if self.phase == "reach":
            P = self._target(w, c) if self.prop is not None else self.P
            if self.point_at:
                sh = c.shoulder(self.side)
                d = norm(sub(P, sh))
                P = add(sh, mul(d, 0.54))
            h.mode = "target"
            w.reach(h, P)
            self.phase = "wait"
            self.t_hold = 0.0
            self.reach_sent = P
            self.resend_until = 0.8
        if self.phase == "wait":
            self.t_hold += dt
            if self.t_hold < self.resend_until and not self.point_at:   # shoulders still moving: keep aim fresh
                P = self._target(w, c) if self.prop is not None else self.P
                if dist(P, h.last_cmd) > 0.01:
                    w.reach(h, P)
            if (w.eng.is_settled(h.cid) and self.t_hold > 0.1
                    and abs(c.crouch - c.crouch_target) < 0.03 and not hasattr(c, "face_goal")):
                if self.grab:
                    return self._finish_grab(w, c, h)
                if self.point_at:
                    self.phase, self.t_hold = "hold", 0.0
                    return False
                return True
            return False
        if self.phase == "hold":
            self.t_hold += dt
            if self.t_hold >= 1.2:
                h.mode = "rest"
                return True
        return False

    def _finish_grab(self, w, c, h):
        p = self.prop
        hand = w.hand_pos(h)
        if dist(hand, tuple(p.pos)) > GRAB_RANGE + (0.08 if p.held_by else 0.0):
            raise ActionError(f"could not reach {p.name} (hand is {dist(hand, tuple(p.pos)):.2f} m away)")
        if p.held_by:
            other = w.chars[p.held_by[0]]
            oh = other.hands[p.held_by[1]]
            oh.holding, oh.mode = None, "rest"
            other.note(f"{c.name} took {p.name} from your {oh.side} hand")
        p.held_by = (c.id, h.side)
        h.holding, h.mode = p.name, "carry"
        return True


class DropAct(Action):
    def __init__(self, side): self.side, self.label = side, f"DROP {side}"
    def start(self, w, c):
        h = c.hands[self.side]
        if not h.holding:
            raise ActionError(f"{self.side} hand is empty")
        p = w.props[h.holding]
        p.held_by, h.holding, h.mode = None, None, "rest"
        p.pos = list(w.hand_pos(h)); p.vy = 0.0


class OfferAct(Action):
    def __init__(self, side, ref): self.side, self.ref, self.label = side, ref, f"OFFER {side} {ref}"
    def start(self, w, c):
        h = c.hands[self.side]
        if not h.holding:
            raise ActionError(f"{self.side} hand is empty; GRAB something first")
        kind, other = w.resolve(self.ref)
        if kind != "char":
            raise ActionError(f"OFFER needs a character, got '{self.ref}'")
        self.other, self.t = other, 0.0
        d = norm(sub(other.pos(), c.pos()))
        w.face_toward(c, other.pos(), snap=False)
        self.P = (c.x + d[0] * 0.5, 1.15 - c.crouch, c.z + d[2] * 0.5)
        h.mode = "offer"
        w.reach(h, self.P)
    def update(self, w, c, dt):
        h = c.hands[self.side]
        if h.mode != "offer":
            return True
        self.t += dt
        return w.eng.is_settled(h.cid) and self.t > 0.3


class WaveAct(Action):
    def __init__(self, side): self.side, self.label = side, f"WAVE {side}"
    def start(self, w, c):
        self.h, self.step, self.t, self.n = c.hands[self.side], 0, 0.0, 0
        self.h.mode = "target"
        self._go(w, c, 0.0)
    def _go(self, w, c, lateral):
        s = 1.0 if self.side == "left" else -1.0
        sh = c.shoulder(self.side)
        P = add(add(sh, mul(c.left(), s * (0.22 + lateral))), add(mul(c.fwd(), 0.18), (0, 0.32, 0)))
        w.reach(self.h, P)
    def update(self, w, c, dt):
        self.t += dt
        if not w.eng.is_settled(self.h.cid):
            return False
        if self.n >= 6:
            self.h.mode = "rest"
            return True
        self.n += 1
        self._go(w, c, 0.14 if self.n % 2 else -0.14)
        return False


class ResetAct(Action):
    def __init__(self, side): self.side, self.label = side, f"RESET {side}"
    def start(self, w, c):
        h = c.hands[self.side]
        h.mode = "carry" if h.holding else "rest"


# --- parsing ---------------------------------------------------------------
HANDS = ("left", "right")


def parse_command(line):
    """Text line -> (verb, args). Raises ActionError. Names are resolved at start()."""
    s = line.strip()
    if not s:
        raise ActionError("empty command")
    verb, _, rest = s.partition(" ")
    verb, rest = verb.upper(), rest.strip()
    tok = rest.split()
    num = lambda t: _finite(t)
    if verb == "SAY":
        if not rest:
            raise ActionError("SAY needs text")
        return verb, (rest[:200].strip('"'),)
    if verb == "WAIT":
        if len(tok) != 1:
            raise ActionError("usage: WAIT <seconds>")
        return verb, (num(tok[0]),)
    if verb in ("LOOK_AT",):
        if len(tok) != 1:
            raise ActionError(f"usage: {verb} <name>")
        return verb, (slug(tok[0]),)
    if verb == "WALK_TO":
        if len(tok) == 1:
            return verb, (slug(tok[0]), None)
        if len(tok) == 2:
            return verb, (None, (num(tok[0]), num(tok[1])))
        raise ActionError("usage: WALK_TO <name> | WALK_TO <x> <z>")
    if verb in ("REACH", "GRAB", "POINT", "OFFER"):
        if not tok or tok[0].lower() not in HANDS:
            raise ActionError(f"usage: {verb} <left|right> <name>")
        side, tok = tok[0].lower(), tok[1:]
        if verb == "REACH" and len(tok) == 3:
            return verb, (side, None, (num(tok[0]), num(tok[1]), num(tok[2])))
        if len(tok) != 1:
            raise ActionError(f"usage: {verb} <left|right> <name>")
        return verb, (side, slug(tok[0]), None)
    if verb in ("DROP", "RESET", "WAVE"):
        if len(tok) != 1 or tok[0].lower() not in HANDS:
            raise ActionError(f"usage: {verb} <left|right>")
        return verb, (tok[0].lower(),)
    raise ActionError(f"unknown command '{verb}'")


def _finite(t):
    try:
        v = float(t)
    except ValueError:
        raise ActionError(f"'{t}' is not a number") from None
    if not math.isfinite(v):
        raise ActionError("non-finite number")
    return v


def build_action(verb, a):
    if verb == "SAY": return SayAct(a[0])
    if verb == "WAIT": return WaitAct(a[0])
    if verb == "LOOK_AT": return LookAct(a[0])
    if verb == "WALK_TO": return WalkAct(ref=a[0], xz=a[1])
    if verb == "REACH": return ReachAct(a[0], ref=a[1], point=a[2])
    if verb == "GRAB": return ReachAct(a[0], ref=a[1], grab=True)
    if verb == "POINT": return ReachAct(a[0], ref=a[1], point_at=True)
    if verb == "OFFER": return OfferAct(a[0], a[1])
    if verb == "DROP": return DropAct(a[0])
    if verb == "RESET": return ResetAct(a[0])
    if verb == "WAVE": return WaveAct(a[0])
    raise ActionError(f"unknown command '{verb}'")


# --- world -----------------------------------------------------------------
class World:
    def __init__(self, lib_path=None):
        self.eng = nebula.Engine(lib_path)
        self.eng.set_param("breath_amp_m", 0.0)
        self.eng.set_param("idle_amp_m", 0.0015)
        self.chars, self.props = {}, {}
        self.t, self._tick = 0.0, 0
        self.log, self._log_id = deque(maxlen=400), 0
        self.recording, self.takes = None, []
        self._ops = queue.Queue()
        self.snapshot = {}
        self.snapshot_json = "{}"
        self.meta = {}                # shown by the UI (e.g. which model is driving the characters)
        self.models = {}              # character id -> .glb bytes
        self.model_info = {}          # character id -> {"ver", "name", "skinned"}
        self.log_listeners = []       # callables(entry), invoked on the sim thread
        self._running = False
        self._thread = None
        self.build_snapshot()

    # -- thread-safe entry point ------------------------------------------
    def submit(self, fn, *args):
        """Run fn(world, *args) on the sim thread; returns a Queue yielding its result/exception."""
        out = queue.Queue(maxsize=1)
        self._ops.put((fn, args, out))
        return out

    def call(self, fn, *args, timeout=5.0):
        r = self.submit(fn, *args).get(timeout=timeout)
        if isinstance(r, Exception):
            raise r
        return r

    # -- structure -----------------------------------------------------------
    def add_character(self, name, persona="", x=0.0, z=0.0, yaw=0.0, color=None):
        base = slug(name)
        cid, n = base, 2
        while cid in self.chars or cid in self.props:
            cid, n = f"{base}_{n}", n + 1
        c = Character(cid, name, persona, x, z, yaw, color or COLORS[len(self.chars) % len(COLORS)])
        for side in HANDS:
            ch = self.eng.add_chain(f"{cid}.{side}", c.shoulder(side), [L_UP, L_FORE],
                                    pole=c.pole(side), torso_lean=0.10, max_speed=1.6)
            self.eng.set_param("breath_amp_m", 0.0)
            c.hands[side] = Hand(side, ch)
            self.eng.teleport(ch, c.rest_point(side))
            self.eng.set_rest(ch, c.rest_point(side))
            c.hands[side].last_cmd = c.rest_point(side)
        self.chars[cid] = c
        self._log("system", f"{c.name} joined the scene")
        return cid

    def remove_character(self, cid):
        c = self.chars.get(cid)
        if not c:
            raise ActionError(f"no character '{cid}'")
        for h in c.hands.values():
            if h.holding:
                self.props[h.holding].held_by = None
        c.active = False
        del self.chars[cid]
        self.models.pop(cid, None)
        self.model_info.pop(cid, None)

    def set_model(self, cid, data, name="model.glb", skinned=False):
        """Attach (data) or remove (data=None) a character's 3D model. The version number lets clients refetch."""
        if cid not in self.chars:
            raise ActionError(f"no character '{cid}'")
        if data is None:
            self.models.pop(cid, None)
            self.model_info[cid] = {"ver": self.model_info.get(cid, {}).get("ver", 0) + 1, "name": "", "skinned": False, "cleared": True}
            return 0
        ver = self.model_info.get(cid, {}).get("ver", 0) + 1
        self.models[cid] = data
        self.model_info[cid] = {"ver": ver, "name": name, "skinned": bool(skinned)}
        return ver

    def add_prop(self, name, pos, size=(0.2, 0.2, 0.2), tags=(), kind="box", color="#a60000"):
        p = Prop(name, pos, size, tags, kind, color)
        base, n = p.name, 2
        while p.name in self.props or p.name in self.chars:
            p.name, n = f"{base}_{n}", n + 1
        self.props[p.name] = p
        return p.name

    def clear_props(self, kind=None):
        for k in [k for k, p in self.props.items() if kind is None or p.kind == kind]:
            for c in self.chars.values():
                for h in c.hands.values():
                    if h.holding == k:
                        h.holding, h.mode = None, "rest"
            del self.props[k]

    def resolve(self, ref):
        ref = slug(ref)
        if ref in self.props:
            return "prop", self.props[ref]
        if ref in self.chars:
            return "char", self.chars[ref]
        for c in self.chars.values():
            if slug(c.name) == ref:
                return "char", c
        return None, None

    # -- commands ----------------------------------------------------------
    def enqueue(self, cid, line, front=False, source="user"):
        """Validate + queue one command line. Raises ActionError on rejection."""
        c = self.chars.get(cid)
        if not c:
            raise ActionError(f"no character '{cid}'")
        verb, args = parse_command(line)
        act = build_action(verb, args)
        act.source = source
        (c.queue.appendleft if front else c.queue.append)(act)
        self._log("cmd", f"{c.name}: {line.strip()}", cid)

    def stop(self, cid):
        c = self.chars[cid]
        c.queue.clear()
        c.current = None
        c.crouch_target = 0.0
        for h in c.hands.values():
            h.mode = "carry" if h.holding else "rest"

    def say(self, c, text):
        self._log("say", f"{c.name}: {text}", c.id)
        for o in self.chars.values():
            if o is not c:
                o.note(f"{c.name} said: \"{text}\"")

    def _log(self, kind, text, cid=None):
        self._log_id += 1
        e = {"id": self._log_id, "t": round(self.t, 2), "kind": kind, "text": text, "char": cid}
        self.log.append(e)
        for f in self.log_listeners:
            f(e)

    # -- helpers for actions ---------------------------------------------
    def hand_pos(self, h):
        return self.eng.world_positions(h.cid)[-1]

    def reach(self, h, p):
        self.eng.reach(h.cid, p)
        h.last_cmd = p

    def face_toward(self, c, p, snap=False):
        c.face_goal = math.atan2(p[0] - c.x, p[2] - c.z)

    def walk_step(self, c, goal, dt, face=None):
        dx, dz = goal[0] - c.x, goal[1] - c.z
        d = math.hypot(dx, dz)
        if d > 0.04:
            head = math.atan2(dx, dz)
            err = ang_diff(head, c.yaw)
            c.yaw += max(-TURN_RATE * dt, min(TURN_RATE * dt, err))
            if abs(err) < 0.9:
                sp = WALK_SPEED * min(1.0, 0.25 + d / 0.8)
                step = min(sp * dt, d)
                c.x += math.sin(c.yaw) * step
                c.z += math.cos(c.yaw) * step
                c.walk_phase += step / 0.38
            return False
        if face is not None:
            fx, fz = (face[0], face[2]) if len(face) == 3 else face
            err = ang_diff(math.atan2(fx - c.x, fz - c.z), c.yaw)
            c.yaw += max(-TURN_RATE * dt, min(TURN_RATE * dt, err))
            return abs(err) < 0.05
        return True

    # -- the tick ----------------------------------------------------------
    def step(self, dt):
        mutated = False
        while True:
            try:
                fn, args, out = self._ops.get_nowait()
            except queue.Empty:
                break
            mutated = True
            try:
                out.put(fn(self, *args))
            except Exception as e:      # reported to the caller, never kills the sim loop
                out.put(e)

        for c in list(self.chars.values()):
            self._update_char(c, dt)
        self._update_hands(dt)
        self.eng.update(dt)
        self._update_props(dt)
        self.t += dt
        self._tick += 1
        every = max(1, round(60 / SNAPSHOT_HZ))
        if mutated or self._tick % every == 0:      # an API write is visible to the very next read
            self.build_snapshot()
        if self._tick % every == 0:
            if self.recording is not None:
                self._record_frame()

    def _update_char(self, c, dt):
        c.crouch += (c.crouch_target - c.crouch) * (1 - math.exp(-6 * dt))
        if hasattr(c, "face_goal"):
            err = ang_diff(c.face_goal, c.yaw)
            c.yaw += max(-TURN_RATE * dt, min(TURN_RATE * dt, err))
            if abs(err) < 0.02:
                del c.face_goal
        if c.speech and self.t > c.speech_until:
            c.speech = ""
        if c.current is None and c.queue:
            act = c.queue.popleft()
            try:
                act.start(self, c)
                c.current, act.age = act, 0.0
                c.last_label = str(act)
            except ActionError as e:
                self._reject(c, act, e)
        if c.current is not None:
            act = c.current
            act.age += dt
            try:
                done = act.update(self, c, dt) or isinstance(act, LookAct)
                if not done and act.age > act.timeout:
                    raise ActionError("timed out")
            except ActionError as e:
                self._reject(c, act, e)
                done = True
            if done:
                c.current = None
        elif not c.queue and not any(h.mode == "target" for h in c.hands.values()):
            c.crouch_target = 0.0
        # gaze
        eye = c.eye()
        want = c.fwd()
        if c.gaze_target:
            kind, key = c.gaze_target
            tgt = None
            if kind == "prop" and key in self.props:
                tgt = tuple(self.props[key].pos)
            elif kind == "char" and key in self.chars:
                tgt = add(self.chars[key].pos(), (0, 1.6, 0))
            if tgt:
                want = norm(sub(tgt, eye))
        k = 1 - math.exp(-7 * dt)
        c.gaze = norm(tuple(c.gaze[i] + (want[i] - c.gaze[i]) * k for i in range(3)))

    def _reject(self, c, act, err):
        msg = f"{act.label or type(act).__name__} rejected: {err}"
        c.note(msg)
        self._log("error", f"{c.name}: {msg}", c.id)

    def _update_hands(self, dt):
        for c in self.chars.values():
            for side, h in c.hands.items():
                sh = c.shoulder(side)
                key = (round(sh[0], 4), round(sh[1], 4), round(sh[2], 4), round(c.yaw, 4))
                if getattr(h, "_anchor", None) != key:
                    self.eng.set_anchor(h.cid, sh)
                    self.eng.set_pole(h.cid, c.pole(side))
                    h._anchor = key
                want = None
                if h.mode == "rest" and not h.holding:
                    want = c.rest_point(side)
                elif h.mode in ("rest", "carry"):
                    want = c.carry_point(side)
                if want is not None and (h.last_cmd is None or dist(want, h.last_cmd) > 0.03):
                    self.reach(h, want)

    def _update_props(self, dt):
        for p in self.props.values():
            if p.held_by:
                c = self.chars.get(p.held_by[0])
                if c is None:
                    p.held_by = None
                    continue
                p.pos = list(self.hand_pos(c.hands[p.held_by[1]]))
                p.vy = 0.0
                continue
            floor = 0.0
            for q in self.props.values():
                if q is not p and q.contains_xz(p.pos[0], p.pos[2], 0.0) and q.top() <= p.pos[1] - p.size[1] / 2 + 0.08:
                    floor = max(floor, q.top())
            rest_y = floor + p.size[1] / 2
            if p.pos[1] > rest_y + 1e-4 and "fixed" not in p.tags:
                p.vy -= 9.8 * dt
                p.pos[1] = max(rest_y, p.pos[1] + p.vy * dt)
            elif p.pos[1] < rest_y and "fixed" not in p.tags:
                p.pos[1], p.vy = rest_y, 0.0

    # -- snapshots / takes ---------------------------------------------------
    def build_snapshot(self):
        r = lambda v: [round(x, 4) for x in v]
        chars = []
        for c in self.chars.values():
            chars.append({
                "id": c.id, "name": c.name, "color": c.color, "pos": [round(c.x, 4), 0, round(c.z, 4)],
                "yaw": round(c.yaw, 4), "crouch": round(c.crouch, 4), "walk": round(c.walk_phase, 4),
                "speech": c.speech, "busy": (getattr(c, "last_label", None) if c.current else None),
                "queued": len(c.queue), "agent": c.agent_on, "persona": c.persona,
                "arms": {s: [r(p) for p in self.eng.world_positions(h.cid)] for s, h in c.hands.items()},
                "gaze": r(c.gaze),
                "holding": {s: h.holding for s, h in c.hands.items()},
                "model": (self.model_info[c.id]["ver"] if c.id in self.models else 0),
                "model_ver": self.model_info.get(c.id, {}).get("ver", 0),
                "model_name": self.model_info.get(c.id, {}).get("name", ""),
            })
        props = [{"name": p.name, "pos": r(p.pos), "size": r(p.size), "tags": p.tags, "kind": p.kind,
                  "color": p.color, "held": bool(p.held_by)} for p in self.props.values()]
        self.snapshot = {"t": round(self.t, 3), "recording": self.recording is not None,
                         "meta": self.meta, "chars": chars, "props": props}
        import json
        self.snapshot_json = json.dumps(self.snapshot, separators=(",", ":"))

    def start_recording(self, name=None):
        if self.recording is not None:
            raise ActionError("already recording")
        self.recording = {"name": name or f"take_{len(self.takes) + 1}", "fps": SNAPSHOT_HZ,
                          "t0": self.t, "frames": []}

    def _record_frame(self):
        if len(self.recording["frames"]) < 120 * SNAPSHOT_HZ:
            s = dict(self.snapshot)
            s["t"] = round(self.t - self.recording["t0"], 3)
            self.recording["frames"].append(s)
        else:
            self.stop_recording()

    def stop_recording(self):
        if self.recording is None:
            raise ActionError("not recording")
        take, self.recording = self.recording, None
        take["id"] = len(self.takes)
        take["duration"] = take["frames"][-1]["t"] if take["frames"] else 0.0
        take["log"] = [e for e in self.log if e["t"] >= take["t0"]]
        self.takes.append(take)
        return take

    # -- sim loop ----------------------------------------------------------
    def start(self, hz=60):
        if self._thread:
            return
        self._running = True

        def loop():
            dt, nxt = 1.0 / hz, time.perf_counter()
            while self._running:
                self.step(dt)
                nxt += dt
                sl = nxt - time.perf_counter()
                if sl > 0:
                    time.sleep(sl)
                elif sl < -0.25:
                    nxt = time.perf_counter()
        self._thread = threading.Thread(target=loop, daemon=True, name="nebula-sim")
        self._thread.start()

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.eng.close()
