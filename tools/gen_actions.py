#!/usr/bin/env python3
"""
gen_actions.py - make training clips of NON-walking actions out of the NERVA/Nebula body engine (the companion of gen_dataset.py, which only walks).

  python tools/gen_actions.py --n 40 --out data/eng_actions                    # 40 clips per action, every skill below
  python tools/gen_actions.py --n 100 --actions crouch pickup jump --out /tmp/a # only these
  python tools/gen_actions.py --list                                            # the skills and the grader action each one is labelled as
  python tools/motion_grader.py data/eng_actions/clips.npz                      # the file has an `action` label per clip: every action is judged on its own

The engine has no animations: a skill here is a short SCRIPT of the engine's own text commands (PELVIS, TORSO, HAND, FOOT, LOOK, JUMP) with random sizes and timings.
The engine moves the body (balance, reach limits, planted feet, flight arc) and every frame is recorded as one 30-float ACTION vector, exactly as gen_dataset.py does.

Skills (grader action it is labelled as):
  stand (stand)         gaze wanders, tiny pelvis sway                       turn (turn)        turn on the spot in alternating foot steps, 40-170 degrees
  balance (balance)     one foot lifted and held, arms out or at the side   crouch (crouch)    pelvis lowered, trunk leaned, held, back up
  pickup (pickup)       squat/bend, one hand to the floor, grip, rise carrying    wave / point (gesture)   raised hand waving; arm held out towards a direction
  stretch (stretch)     both arms overhead, trunk leaned back or to the side   drink (drink)    hand to the mouth, head back a little, down again
  jump (jump)           crouch, JUMP to a landing point, recover

A clip is kept only if the engine accepted every command, the body did not fall and it ended standing. Then it is judged two ways (both recorded in clips.npz):
  feasible  the engine's replay verdict (the same one gen_dataset.py uses). NOT AVAILABLE for jump: the replay cannot follow a flight phase (it calls both feet off the ground
            "balance lost"), so for jump `feasible` is False and `live_ok` (the engine ran the JUMP itself without falling, and landed standing) is the only engine evidence.
  replay    how the clip was verified: ok (the replay verdict, from the default standing start), own_start (the verdict, from the widened stance the clip starts in), live (jump)
  grade     tools/motion_grader.py, against the real motion of that action (needs data/real_motions.npz; --no-grade to skip): score 0-100 and whether the grader recognises the clip as the action.
Honest limits: hands do not touch anything (no props, no floor contact: pickup reaches towards a point), there is no seat so there is no sit, no get-up and no running: the engine
cannot do those yet (docs/BODY.md). Skills are scripted, so the variety is the variety of the random sizes and timings, not new behaviour.
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_dataset as gd

SIM_DT = 1 / 60
FPS = 30
SHOULDER = {"left": (0.18, 1.43, 0.05), "right": (-0.18, 1.43, 0.05)}     # approximate, standing at the origin facing +z (left is +x); arm reach is about 0.56 m
TRUNK = 0.58                                                              # pelvis -> neck base, metres (standing)
PELVIS_Y = 0.926
NEUTRAL = dict(torso_side=0.0, torso_drop=0.0, torso_lean=0.0, look_down_deg=0.0, look_yaw_deg=0.0)


def U(rng, a, b):
    return float(rng.uniform(a, b))


def gaze(rng, yaw_deg, pitch_deg, dist=2.0, head=(0.0, 1.6, 0.05)):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return f"LOOK {head[0] + dist * math.sin(y) * math.cos(p):.3f} {head[1] + dist * math.sin(p):.3f} {head[2] + dist * math.cos(y) * math.cos(p):.3f}"


def neck_target(py, lean_deg, side_deg=0.0, pz=0.0, px=0.0):
    """TORSO command for a trunk leaned forward by lean_deg (negative = back) and sideways by side_deg, on a pelvis at (px, py, pz)."""
    l, s = math.radians(lean_deg), math.radians(side_deg)
    return f"TORSO {px + TRUNK * math.sin(s):.3f} {py + TRUNK * math.cos(l) * math.cos(s):.3f} {pz + TRUNK * math.sin(l):.3f}"


def reachable(rng, side, rmin=0.25, rmax=0.48, yaw=(-70, 70), pitch=(-50, 60), base=None):
    """A wrist target inside the arm's reach: a random direction in front of the shoulder, distance rmin..rmax (the arm reaches ~0.56 m)."""
    sx, sy, sz = base or SHOULDER[side]; y, p, r = math.radians(U(rng, *yaw)), math.radians(U(rng, *pitch)), U(rng, rmin, rmax)
    return (sx + r * math.sin(y) * math.cos(p), sy + r * math.sin(p), sz + r * math.cos(y) * math.cos(p))


def ramp(sc, t0, dur, p0, p1, bal, bias=0.0, step=0.1):
    """Append PELVIS + TORSO commands that move the body from pose p0 to p1 = (pelvis height, trunk lean deg, trunk side deg) over dur seconds (smooth start and stop),
    with the pelvis shifted at every step so the centre of mass stays over the feet. Jumping straight to the final pose makes the trunk run ahead of the hips and the body tips for a moment."""
    n = max(2, int(round(dur / step)))
    for k in range(1, n + 1):
        u = k / n; u = u * u * (3 - 2 * u); y = p0[0] + (p1[0] - p0[0]) * u; ln = p0[1] + (p1[1] - p0[1]) * u; sd = p0[2] + (p1[2] - p0[2]) * u
        z = 0.043 + bal(y, ln) + bias
        sc.append((t0 + dur * k / n - step, f"PELVIS 0 {y:.3f} {z:.3f}")); sc.append((t0 + dur * k / n - step, neck_target(y, ln, sd, pz=z)))
    return t0 + dur


STAND_POSE = (PELVIS_Y, 0.0, 0.0)


class Balance:
    """How far back (z, metres) the pelvis must sit so that the centre of mass stays over the feet, for a pelvis height and a trunk lean. Measured on the engine itself the first time it is asked
    (a few hundred short settles, ~2 s): a person leaning forward sends the hips back, and a skill that forgets it makes the engine lose its balance."""
    YS = (0.26, 0.36, 0.48, 0.60, 0.74, 0.86, 0.93); LEANS = (-25.0, 0.0, 20.0, 40.0, 60.0, 80.0); DZ = np.arange(-0.40, 0.121, 0.03)

    def __init__(self, lib):
        self.lib = lib; self.T = None

    def _margin(self, y, lean, dz):
        b = gd.new_standing(self.lib)
        try:
            for _ in range(15):
                b.update(SIM_DT)
            b.cmd(f"PELVIS 0 {y:.3f} {0.043 + dz:.3f}"); b.cmd(neck_target(y, lean, pz=0.043 + dz))
            for _ in range(200):
                b.update(SIM_DT)
            s = b.state(); return -9.0 if s[10] > 0.5 else float(s[9])
        finally:
            b.close()

    def __call__(self, y, lean):
        if self.T is None:
            self.T = np.array([[self.DZ[int(np.argmax([self._margin(yy, ll, d) for d in self.DZ]))] for ll in self.LEANS] for yy in self.YS])
        rows = [np.interp(lean, self.LEANS, self.T[i]) for i in range(len(self.YS))]
        return float(np.interp(y, self.YS, rows))


# ---------------------------------------------------------------------------------------------------------------------------------- skills
# every skill: f(rng, bal) -> (script, T, meta); bal(y, lean_deg) = pelvis z shift that keeps the balance; script = [(time_s, "COMMAND ..."), ...]; the clip ends T seconds after the start (the body is then back at rest and standing)
def sk_stand(rng, bal):
    T = U(rng, 3.5, 7.0); sc = []; t = U(rng, 0.2, 0.8)
    while t < T - 1.0:
        sc.append((t, gaze(rng, rng.normal(0, 22), rng.normal(0, 8)))); t += U(rng, 0.9, 2.6)
    t = U(rng, 0.3, 1.0)                                                         # weight shifts and sway: the pelvis target drifts a few centimetres
    while t < T - 1.2:
        sc.append((t, f"PELVIS {U(rng, -0.035, 0.035):.4f} {PELVIS_Y - U(rng, 0.0, 0.02):.4f} {0.043 + U(rng, -0.03, 0.03):.4f}")); t += U(rng, 1.0, 2.4)
    sc.append((T - 1.2, "PELVIS auto"))
    if rng.random() < 0.18:                                                      # a hand comes up (scratch, adjust, fold arms) and goes back
        side = "left" if rng.random() < 0.5 else "right"; sg = 1.0 if side == "left" else -1.0; th = U(rng, 0.6, max(0.8, T - 3.2))
        tgt = (sg * U(rng, 0.0, 0.2), U(rng, 1.05, 1.42), U(rng, 0.20, 0.34)); sc += [(th, f"HAND {side} {sg * 0.30:.3f} 1.15 0.15"), (th + 0.8, f"HAND {side} {tgt[0]:.3f} {tgt[1]:.3f} {tgt[2]:.3f}"),
                                                                                     (th + 0.8 + U(rng, 0.8, 1.8), f"HAND {side} {sg * 0.30:.3f} 1.15 0.15"), (th + 2.4 + 1.8, f"HAND {side} rest")]
        T = max(T, th + 6.0)
    if rng.random() < 0.4:
        sc.append((U(rng, 1.0, max(1.2, T - 2.5)), f"FOOT {'left' if rng.random() < 0.5 else 'right'} {U(rng, -0.04, 0.04) + 0.1:.3f} 0 {U(rng, -0.05, 0.08):.3f}"))
    return sc, T, {}


def sk_turn(rng, bal):
    ang = U(rng, 40, 170) * (1 if rng.random() < 0.5 else -1); n = int(math.ceil(abs(ang) / 50.0)) + 1; w = U(rng, 0.09, 0.12); side = int(rng.integers(0, 2))
    sc = []; t = U(rng, 0.3, 0.7); dt_step = U(rng, 0.42, 0.70)
    for i in range(1, n + 1):
        h = math.radians(ang) * i / n; sgn = 1.0 if side == 0 else -1.0
        sc.append((t, f"FOOT {'left' if side == 0 else 'right'} {sgn * w * math.cos(h):.4f} 0 {-sgn * w * math.sin(h) + 0.043:.4f} {math.degrees(h):.2f}")); side = 1 - side; t += dt_step
    h = math.radians(ang); sgn = 1.0 if side == 0 else -1.0                      # closing step: the other foot comes level
    sc.append((t, f"FOOT {'left' if side == 0 else 'right'} {sgn * w * math.cos(h):.4f} 0 {-sgn * w * math.sin(h) + 0.043:.4f} {math.degrees(h):.2f}"))
    sc.append((0.1, gaze(rng, math.degrees(math.radians(ang) * 0.5), U(rng, -28, -8)))); sc.append((t + 0.9, "LOOK auto"))
    return sc, t + 2.0, dict(angle=ang)


def sk_balance(rng, bal):
    side = "left" if rng.random() < 0.5 else "right"; sg = 1.0 if side == "left" else -1.0; T_hold = U(rng, 2.0, 5.0)
    x = sg * U(rng, 0.06, 0.16); y = U(rng, 0.10, 0.32); z = U(rng, -0.12, 0.28); t0 = U(rng, 0.4, 0.9); up = U(rng, 1.4, 2.2)
    sc = [(t0, f"FOOT {side} {x:.3f} {y:.3f} {z:.3f}")]
    if rng.random() < 0.7:                                                       # arms out, as people do on one leg
        for s2, g in (("left", 1.0), ("right", -1.0)):
            sc.append((t0 - 0.2, f"HAND {s2} {g * U(rng, 0.50, 0.65):.3f} {U(rng, 1.15, 1.40):.3f} {U(rng, -0.05, 0.15):.3f}"))
        sc += [(t0 + up + T_hold, "HAND left rest"), (t0 + up + T_hold, "HAND right rest")]
    sc.append((t0 + up + T_hold, f"FOOT {side} {sg * 0.1:.3f} 0 0.043"))
    return sc, t0 + up + T_hold + 2.4, {}


def sk_crouch(rng, bal):
    y = U(rng, 0.50, 0.74); lean = U(rng, 15.0, 52.0); bias = U(rng, -0.012, 0.012); down = U(rng, 0.7, 1.2); hold = U(rng, 1.8, 4.0); sc = []; t0 = U(rng, 0.3, 0.8); rec = 0.0; pre = []
    if rng.random() < 0.7:                                                       # a wider stance first: people crouch with their feet apart, one a little ahead
        sep = U(rng, 0.26, 0.44); st_ = U(rng, -0.12, 0.14)
        pre = [f"FOOT left {sep / 2:.3f} 0 {st_:.3f}", f"FOOT right {-sep / 2:.3f} 0 {-st_:.3f}"]
        sc += [(t0, pre[0]), (t0 + 0.2, pre[1])]; t0 += 2.8; rec = t0 - 0.3
    te = ramp(sc, t0, down, STAND_POSE, (y, lean, 0.0), bal, bias)
    if rng.random() < 0.6:
        sc.append((te, gaze(rng, rng.normal(0, 10), U(rng, -35, -5))))
    if rng.random() < 0.5:
        for s2, g in (("left", 1.0), ("right", -1.0)):
            sc.append((te + 0.1, f"HAND {s2} {g * U(rng, 0.22, 0.4):.3f} {y + U(rng, 0.05, 0.45):.3f} {U(rng, 0.2, 0.45):.3f}"))
    tu = te + hold
    sc += [(tu, "HAND left rest"), (tu, "HAND right rest"), (tu, "LOOK auto")]
    te2 = ramp(sc, tu, down, (y, lean, 0.0), STAND_POSE, bal, bias)
    sc += [(te2 + 0.1, "PELVIS auto"), (te2 + 0.1, "TORSO auto")]
    return sc, te2 + 0.9, dict(rec_from=rec, pre=pre)


def sk_pickup(rng, bal):
    side = "left" if rng.random() < 0.5 else "right"; sg = 1.0 if side == "left" else -1.0; squat = rng.random() < 0.5; bias = U(rng, -0.012, 0.012)
    y = U(rng, 0.34, 0.56) if squat else U(rng, 0.66, 0.84); lean = U(rng, 25, 50) if squat else U(rng, 55, 85); sd = sg * U(rng, 0.0, 22.0)
    zz = 0.043 + bal(y, lean) + bias; l = math.radians(lean); sh = (sg * 0.188, y + TRUNK * math.cos(l) - 0.12 * math.sin(l), zz + TRUNK * math.sin(l) + 0.12 * math.cos(l) * 0 + 0.0)
    pit = math.radians(U(rng, -75, -30)); yw = math.radians(U(rng, -25, 25)) + (math.radians(10) * sg); rr = U(rng, 0.26, 0.42)
    obj = (sh[0] + rr * math.sin(yw) * math.cos(pit), max(0.02, sh[1] + rr * math.sin(pit)), sh[2] + rr * math.cos(yw) * math.cos(pit)); down = U(rng, 0.9, 1.5); t0 = U(rng, 0.3, 0.7); sc = []
    te = ramp(sc, t0, down, STAND_POSE, (y, lean, sd), bal, bias)
    sc += [(t0 + 0.2, gaze(rng, 0, U(rng, -40, -20))), (te, f"HAND {side} {obj[0]:.3f} {obj[1]:.3f} {obj[2]:.3f}")]
    tg = te + U(rng, 0.9, 1.4); sc.append((tg, f"GRIP {side} 1"))
    tu = tg + U(rng, 0.3, 0.6); carry = (sg * U(rng, 0.10, 0.2), U(rng, 0.95, 1.2), U(rng, 0.25, 0.4))
    sc += [(tu, "LOOK auto"), (tu, f"HAND {side} {carry[0]:.3f} {carry[1]:.3f} {carry[2]:.3f}")]
    te2 = ramp(sc, tu, down, (y, lean, sd), STAND_POSE, bal, bias); sc += [(te2 + 0.1, "PELVIS auto"), (te2 + 0.1, "TORSO auto")]
    th = te2 + U(rng, 0.8, 2.0); sc += [(th, f"HAND {side} rest"), (th, f"GRIP {side} 0")]
    return sc, th + 1.6, dict(squat=squat)


def sk_wave(rng, bal):
    side = "left" if rng.random() < 0.5 else "right"; sg = 1.0 if side == "left" else -1.0; ncyc = int(rng.integers(2, 6)); half = U(rng, 0.16, 0.30); t0 = U(rng, 0.3, 0.7)
    cx = sg * U(rng, 0.28, 0.40); cy = U(rng, 1.60, 1.78); cz = U(rng, 0.10, 0.30); amp = U(rng, 0.08, 0.16)
    sc = [(t0, f"HAND {side} {cx:.3f} {cy:.3f} {cz:.3f}"), (t0, gaze(rng, rng.normal(0, 8), U(rng, -3, 6)))]
    t = t0 + U(rng, 0.8, 1.1)
    for k in range(ncyc * 2):
        sc.append((t, f"HAND {side} {cx + (amp if k % 2 == 0 else -amp):.3f} {cy + U(rng, -0.02, 0.02):.3f} {cz:.3f}")); t += half
    sc += [(t + 0.3, f"HAND {side} rest"), (t + 0.3, "LOOK auto")]
    return sc, t + 2.0, {}


def sk_point(rng, bal):
    side = "left" if rng.random() < 0.5 else "right"; yaw = U(rng, -70, 70); hold = U(rng, 1.0, 3.0); t0 = U(rng, 0.3, 0.7)
    tgt = reachable(rng, side, 0.44, 0.55, yaw=(yaw - 8, yaw + 8), pitch=(-10, 25))
    far = (tgt[0] * 6, 1.4 + (tgt[1] - SHOULDER[side][1]) * 6, tgt[2] * 6)
    sc = [(t0, f"HAND {side} {tgt[0]:.3f} {tgt[1]:.3f} {tgt[2]:.3f}"), (t0 + 0.1, f"LOOK {far[0]:.3f} {far[1]:.3f} {far[2]:.3f}"), (t0 + 0.2, f"GRIP {side} 0.8"),
          (t0 + 1.0 + hold, f"HAND {side} rest"), (t0 + 1.0 + hold, f"GRIP {side} 0"), (t0 + 1.0 + hold, "LOOK auto")]
    return sc, t0 + 1.0 + hold + 1.8, {}


def sk_stretch(rng, bal):
    side_lean = U(rng, -3, 3); back = -U(rng, 0, 3); hold = U(rng, 2.0, 4.0); t0 = U(rng, 0.3, 0.7); up = U(rng, 1.6, 2.4); bias = U(rng, -0.01, 0.01); sc = []
    te = ramp(sc, t0, 0.8, STAND_POSE, (PELVIS_Y, back, side_lean), bal, bias)
    via = U(rng, 0.34, 0.44); sc += [(te, f"HAND left {via:.3f} 1.20 0.12"), (te, f"HAND right {-via:.3f} 1.20 0.12"), (te + 0.9, f"HAND left {U(rng, 0.10, 0.22):.3f} {U(rng, 1.60, 1.74):.3f} {U(rng, 0.0, 0.14):.3f}"),
                                      (te + 0.9, f"HAND right {-U(rng, 0.10, 0.22):.3f} {U(rng, 1.60, 1.74):.3f} {U(rng, 0.0, 0.14):.3f}"), (te + 1.2, gaze(rng, 0, U(rng, 20, 45)))]
    tr = te + up + hold; sc += [(tr, f"HAND left {via:.3f} 1.20 0.12"), (tr, f"HAND right {-via:.3f} 1.20 0.12"), (tr + 0.9, "HAND left rest"), (tr + 0.9, "HAND right rest"), (tr, "LOOK auto")]
    te2 = ramp(sc, tr + 1.8, 0.8, (PELVIS_Y, back, side_lean), STAND_POSE, bal, bias); sc += [(te2 + 0.1, "PELVIS auto"), (te2 + 0.1, "TORSO auto")]
    return sc, te2 + 1.8, {}


def sk_drink(rng, bal):
    side = "left" if rng.random() < 0.5 else "right"; sg = 1.0 if side == "left" else -1.0; hold = U(rng, 1.2, 3.0); t0 = U(rng, 0.3, 0.7); up = U(rng, 1.0, 1.6)
    mouth = (sg * U(rng, 0.0, 0.05), U(rng, 1.42, 1.52), U(rng, 0.16, 0.26))
    sc = [(t0, f"HAND {side} {mouth[0]:.3f} {mouth[1]:.3f} {mouth[2]:.3f}"), (t0 + 0.2, f"GRIP {side} 0.7"), (t0 + up * 0.7, gaze(rng, 0, U(rng, 8, 25)))]
    te = t0 + up + hold; sc += [(te, "LOOK auto"), (te + 0.2, f"HAND {side} rest"), (te + 0.2, f"GRIP {side} 0")]
    return sc, te + up + 1.2, {}


def sk_jump(rng, bal):
    y = U(rng, 0.60, 0.80); lean = U(rng, 8.0, 38.0); bias = U(rng, -0.01, 0.01); dx = U(rng, -0.25, 0.25); dz = U(rng, 0.0, 0.9) if rng.random() < 0.85 else U(rng, -0.4, 0.0)
    crouch = U(rng, 0.55, 1.0); t0 = U(rng, 0.3, 0.6); apex = f" {U(rng, 0.10, 0.30):.3f}" if rng.random() < 0.5 else ""; sc = []
    te = ramp(sc, t0, crouch, STAND_POSE, (y, lean, 0.0), bal, bias)
    swing = rng.random() < 0.75
    if swing:                                                                    # arms swing back while crouching, then up and forward with the take-off
        for s2, g in (("left", 1.0), ("right", -1.0)):
            sc.append((t0 + 0.1, f"HAND {s2} {g * U(rng, 0.20, 0.32):.3f} {y + U(rng, 0.10, 0.30):.3f} {-U(rng, 0.10, 0.28):.3f}"))
    tj = te + U(rng, 0.15, 0.4); sc += [(tj - 0.05, "TORSO auto"), (tj, f"JUMP {dx:.3f} 0 {dz + 0.043:.3f}{apex}")]
    if swing:
        for s2, g in (("left", 1.0), ("right", -1.0)):
            sc.append((tj + 0.05, f"HAND {s2} {g * U(rng, 0.20, 0.32):.3f} {U(rng, 1.25, 1.65):.3f} {U(rng, 0.25, 0.45):.3f}"))
    sc += [(tj + 1.0, "HAND left rest"), (tj + 1.0, "HAND right rest"), (tj + 1.9, "PELVIS auto")]
    return sc, tj + 3.4, dict(flight=True)


SKILLS = {                                    # name -> (function, grader action label)
    "stand": (sk_stand, "stand"), "turn": (sk_turn, "turn"), "balance": (sk_balance, "balance"), "crouch": (sk_crouch, "crouch"), "pickup": (sk_pickup, "pickup"),
    "wave": (sk_wave, "gesture"), "point": (sk_point, "gesture"), "stretch": (sk_stretch, "stretch"), "drink": (sk_drink, "drink"), "jump": (sk_jump, "jump"),
}


# ---------------------------------------------------------------------------------------------------------------------------------- running a script
def run_script(lib, rng, script, T, flight=False, rec_from=0.0):
    """Play a script in the engine and record the actions. Returns (actions[F,30] or None, info). Dropped: any refused command, a fall, a body that does not end standing."""
    b = gd.new_standing(lib)
    try:
        b.param("hand_idle_m", float(rng.uniform(0.0, 0.004)))
        for _ in range(15):
            b.update(SIM_DT)
        st = dict(NEUTRAL); st["look_down_deg"] = 0.0
        cmds = sorted(script, key=lambda x: x[0]); ci = 0; frames = []; every = int(round(60 / FPS)); refused = []; air = 0
        for i in range(int(T * 60)):
            t = i * SIM_DT
            while ci < len(cmds) and cmds[ci][0] <= t:
                ok, msg = b.cmd(cmds[ci][1])
                if not ok:
                    refused.append(f"{cmds[ci][1].split()[0]}: {msg[:90]}")
                ci += 1
            if refused:
                return None, "refused " + refused[0]
            b.update(SIM_DT); s = b.state()
            if s[10] > 0.5:
                return None, "fell"
            if s[15] == 2 and s[21] == 2:
                air += 1
            if i % every == 0 and t >= rec_from:
                frames.append(gd.action_from(b, s, st))
        s = b.state()
        if s[34] != 0 or s[15] != 0 or s[21] != 0:
            return None, "did not end standing (a foot still moving or in the air)"
        if flight and air == 0:
            return None, "no flight phase"
        a = np.array(frames, np.float32)
        if not np.isfinite(a).all():
            return None, "non-finite values"
        return a, dict(dt=1.0 / FPS, airborne_frames=air // 2)
    finally:
        b.close()


def prepared_body(lib, pre):
    """A standing body that has first been given the commands in `pre` (a wider stance, say) and settled: the start the replay verdict must judge a clip from when the clip does not start from the default stance."""
    b = gd.new_standing(lib)
    for c in pre:
        b.cmd(c)
        for _ in range(90):
            b.update(SIM_DT)
    for _ in range(90):
        b.update(SIM_DT)
    return b


def trim(a, dt, pad=0.4, min_s=2.0):
    """Cut the still stretches off the head and tail of a clip (keeping `pad` seconds around the motion): real clips are cut around the action, ours start from rest and end at rest."""
    mv = np.abs(np.diff(a[:, [0, 1, 2, 4, 5, 6, 9, 10, 11, 14, 15, 16, 18, 19, 20, 26, 27, 28]], axis=0)).max(axis=1) / dt
    act = np.where(mv > 0.03)[0]
    if len(act) == 0:
        return a
    k = int(round(pad / dt)); i0 = max(0, act[0] - k); i1 = min(len(a), act[-1] + 2 + k)
    need = int(round(min_s / dt))
    if i1 - i0 < need:
        extra = need - (i1 - i0); i0 = max(0, i0 - extra // 2); i1 = min(len(a), i1 + extra - extra // 2)
    return a[i0:i1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="clips to keep PER SKILL"); ap.add_argument("--out", default="data/eng_actions"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actions", nargs="*", default=None, help="skills to run (default: all; --list shows them)"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--try-factor", type=float, default=6.0, help="give up on a skill after this many attempts per wanted clip")
    ap.add_argument("--real", default="data/real_motions.npz"); ap.add_argument("--no-grade", action="store_true", help="do not score the clips with tools/motion_grader.py")
    ap.add_argument("--trim", action="store_true", help="cut the still stretches off the start and end of each clip (off by default: it did not raise the grades)"); ap.add_argument("--lib"); ap.add_argument("--repo"); ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (f, act) in SKILLS.items():
            print(f"  {k:8s} -> {act}")
        return
    names = a.actions or list(SKILLS)
    bad = [n for n in names if n not in SKILLS]
    if bad:
        sys.exit("unknown skill(s): " + ", ".join(bad) + "  (--list)")
    lib = gd.Lib(a.lib, a.repo, build=not a.no_build); chk = gd.new_standing(lib); bal = Balance(lib); os.makedirs(a.out, exist_ok=True)
    prof = G = None
    if not a.no_grade:
        try:
            import motion_grader as G
            _, RF = G.load_real(a.real); zr = np.load(a.real, allow_pickle=True); prof = G.Profiles(RF, zr["action"])
        except Exception as ex:
            print(f"(grader not available, clips are not scored: {str(ex)[:100]})"); prof = None
    clips, labels, skill, feas, live, score, recog, flight, replay = [], [], [], [], [], [], [], [], []
    t0 = time.time(); stats = {}
    for k, nm in enumerate(names):
        f, act = SKILLS[nm]; rng = np.random.default_rng([a.seed, k]); kept = tries = 0; st = stats[nm] = dict(tries=0, kept=0, dropped={}, verdict_ok=0, graded=0, passed=0)
        while kept < a.n and tries < a.n * a.try_factor:
            tries += 1; script, T, meta = f(rng, bal); fl = bool(meta.get("flight"))
            x, info = run_script(lib, rng, script, T, flight=fl, rec_from=float(meta.get("rec_from", 0.0)))
            if x is None:
                key = info.split(":")[0][:60] if not info.startswith("refused") else info[:70]; st["dropped"][key] = st["dropped"].get(key, 0) + 1; continue
            if a.trim or fl:
                x = trim(x, info["dt"], pad=0.6 if fl else 0.4, min_s=2.5 if fl else 2.0)
            pre = meta.get("pre")
            if pre:
                cb = prepared_body(lib, pre)
                try:
                    v = cb.evaluate(x, info["dt"])
                finally:
                    cb.close()
            else:
                v = chk.evaluate(x, info["dt"])
            ok = bool(v["feasible"])
            if not ok and not fl:
                st["dropped"]["engine verdict: infeasible"] = st["dropped"].get("engine verdict: infeasible", 0) + 1; continue
            sc = np.nan; rc = False
            if prof is not None:
                try:
                    r = G.grade_clip(x, info["dt"], act, prof); sc = float(r["score"]) if r["score"] is not None else np.nan; rc = bool(r["recognised"]); st["graded"] += 1
                    st["passed"] += int(rc and sc >= G.PASS_SCORE)
                except Exception:
                    pass
            clips.append(x); labels.append(act); skill.append(nm); feas.append(ok); live.append(True); score.append(sc); recog.append(rc); flight.append(fl); replay.append("live" if fl else ("own_start" if meta.get("pre") else "ok")); kept += 1; st["verdict_ok"] += int(ok)
        st["tries"] = tries; st["kept"] = kept
        print(f"  {nm:8s} -> {act:8s} kept {kept}/{a.n} in {tries} tries" + (f", grader: {st['passed']}/{st['graded']} pass and are recognised" if st["graded"] else "") + (f"; dropped: {st['dropped']}" if st["dropped"] else ""), flush=True)
    if not clips:
        sys.exit("no clips were produced")
    dt = 1.0 / FPS; off = np.cumsum([0] + [len(c) for c in clips])
    np.savez_compressed(os.path.join(a.out, "clips.npz"), actions=np.concatenate(clips).astype(np.float32), offsets=off, action=np.array(labels), skill=np.array(skill), feasible=np.array(feas), live_ok=np.array(live),
                        flight=np.array(flight), replay=np.array(replay), grade=np.array(score, np.float32), recognised=np.array(recog), dt=np.float32(dt), action_names=np.array(gd.ACTION_NAMES))
    json.dump(dict(n_clips=len(clips), seconds=float(off[-1] * dt), skills=stats, wall_seconds=round(time.time() - t0, 1)), open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"\n{len(clips)} clips, {off[-1] * dt:.0f} s of motion in {time.time() - t0:.0f}s -> {a.out}/clips.npz")
    chk.close()


if __name__ == "__main__":
    main()
