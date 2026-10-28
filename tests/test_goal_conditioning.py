#!/usr/bin/env python3
"""
tests/test_goal_conditioning.py - the goal numbers of the prior and the window merging of the closed loop. Needs numpy + torch (train_cvqvae imports it), no engine, no data.

  python tests/test_goal_conditioning.py

  1. goal_features / arc_goal agree: an arc built with arc_goal(turn, path, seconds) and RENDERED as a pelvis path gives back its own dyaw / dx / dz / speed (also after moving
     the window to a random world pose and canonicalising it again, and with a window that is only partly valid)
  2. repeated merge_window reproduces stitch(): commit `stride` frames of each merged window, keep the rest as pending, repeat
  3. tools/closed_loop.py plan_window_goal: the arc it asks for ENDS ON the target when the turn is not clipped, and is clipped to turn_max when it is
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import train_cvqvae as T

FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def render_arc(turn, path, n=64, dt=1 / 30):
    """A pelvis path with constant curvature, canonical frame: starts at the origin facing +z, turns by `turn` rad in total, `path` metres long, n frames. -> [n,30] (pelvis channels only)."""
    u = np.linspace(0, 1, n); w = np.zeros((n, 30))
    if abs(turn) < 1e-9:
        w[:, 0] = 0.0; w[:, 2] = path * u
    else:
        w[:, 0] = path * (1 - np.cos(turn * u)) / turn; w[:, 2] = path * np.sin(turn * u) / turn
    w[:, 3] = turn * u; w[:, 1] = 0.95
    return w


def world_window(w, px, pz, yaw):
    """Move a canonical window to a world pose (the inverse of canonicalize for the pelvis channels)."""
    return T.to_world(w, px, pz, yaw)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 1
print("[goal_features / arc_goal]")
dt, n = 1 / 30, 64; sec = (n - 1) * dt
cases = [(0.0, 1.2), (0.4, 1.5), (-0.4, 1.5), (1.0, 0.9), (-1.5, 0.8), (2.5, 0.6), (0.05, 2.0), (-0.01, 0.1)]
worst = 0.0
for turn, path in cases:
    w = render_arc(turn, path, n, dt)[None]
    got = T.goal_features(w.astype(np.float32), np.array([n]), dt)[0]; want = T.arc_goal(turn, path, sec)
    err = np.abs(got - want); worst = max(worst, float(err.max()))
    check(err[0] < 1e-5 and err[1] < 2e-3 and err[2] < 2e-3 and err[3] < 2e-3 * max(1.0, want[3]),
          f"turn {turn:+.2f} rad, path {path:.2f} m: features {np.round(got, 4).tolist()} vs arc_goal {np.round(want, 4).tolist()}")
print(f"        (largest difference over all cases {worst:.1e}; a rendered polyline is slightly shorter than its arc)")

rng = np.random.default_rng(0)
for turn, path in cases[:6]:                                       # moved to a random world pose, then canonicalised again: the same numbers
    px, pz, yaw = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-8, 8)
    w = world_window(render_arc(turn, path, n, dt), px, pz, yaw)[None].astype(np.float32)
    got = T.goal_features(T.canonicalize(w), np.array([n]), dt)[0]; want = T.arc_goal(turn, path, sec)
    check(np.abs(got - want).max() < 3e-3, f"turn {turn:+.2f}: from the world pose ({px:.1f}, {pz:.1f}, yaw {yaw:.1f}) canonicalised -> {np.round(got, 3).tolist()}")

b = T.arc_goal(np.array([0.0, 0.5, -0.5]), np.array([1.0, 1.0, 1.0]), 2.0)
check(b.shape == (3, 4) and np.allclose(b[:, 3], 0.5) and b[1, 1] > 0 > b[2, 1], "arc_goal broadcasts over arrays; a left turn (+) has dx > 0, a right turn dx < 0")
wv = np.stack([render_arc(0.6, 1.4, n, dt)] * 2).astype(np.float32)
g_half = T.goal_features(wv, np.array([n, n // 2]), dt)
check(abs(g_half[1, 0] - 0.6 * (n // 2 - 1) / (n - 1)) < 1e-4, "a partly valid window is measured over its valid frames only")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 2
print("[merge_window reproduces stitch]")


def random_window(rng, W=64, turn=0.0):
    """A decoded-looking canonical window: pelvis walking, feet and hands moving, contact flags 0/1. Not a real walk, but every channel varies and yaws pass +-pi."""
    t = np.arange(W) / 30.0; w = np.zeros((W, 30))
    yaw = turn * np.linspace(0, 1, W) + rng.uniform(-0.2, 0.2) + 0.3 * np.sin(2 * t)
    w[:, 3] = yaw; w[:, 1] = 0.95 + 0.02 * np.sin(6 * t)
    sp = rng.uniform(0.6, 1.4); w[:, 0] = np.cumsum(sp / 30 * np.sin(yaw)); w[:, 2] = np.cumsum(sp / 30 * np.cos(yaw))
    for (base, side) in ((4, 0.1), (9, -0.1)):
        w[:, base] = w[:, 0] + side + 0.02 * rng.normal(size=W).cumsum() * 0.05; w[:, base + 1] = 0.03 * np.abs(np.sin(3 * t + base)); w[:, base + 2] = w[:, 2] + 0.2 * np.sin(3 * t + base)
        w[:, base + 3] = yaw + rng.uniform(-0.2, 0.2); w[:, base + 4] = (np.sin(3 * t + base) > 0).astype(float)
    for base in (14, 18):
        w[:, base:base + 3] = w[:, [0, 1, 2]] + rng.normal(size=3) * 0.2; w[:, base + 3] = (rng.random(W) > 0.5).astype(float)
    w[:, 22:25] = w[:, [0, 1, 2]]; w[:, 25] = 0.0; w[:, 26:29] = w[:, [0, 1, 2]]; w[:, 29] = (rng.random(W) > 0.7).astype(float)
    return w.astype(np.float32)


for stride, nwin, turn in ((32, 5, 0.0), (32, 6, 1.7), (16, 7, -2.2), (48, 4, 0.8), (32, 2, 3.5)):
    r = np.random.default_rng(stride * 10 + nwin)
    wins = [random_window(r, 64, turn * r.uniform(0.5, 1.5)) for _ in range(nwin)]
    ref = T.stitch(wins, stride)
    pending = None; out = []
    for i, w in enumerate(wins):
        m = T.merge_window(pending, w, stride, origin=(0.0, 0.0, 0.0))
        assert len(m) == 64
        if i < nwin - 1:
            out.append(m[:stride]); pending = m[stride:]
            assert len(pending) == 64 - stride
        else:
            out.append(m)
    got = np.concatenate(out)
    d = np.abs(got - ref).max() if got.shape == ref.shape else float("inf")
    check(got.shape == ref.shape and d < 1e-5, f"stride {stride}, {nwin} windows, turns up to {turn:.1f} rad: {got.shape[0]} frames, largest difference to stitch() {d:.1e}")
    check(np.array_equal(got[:, T.FLAGS], ref[:, T.FLAGS]), "  contact / on flags identical")

m0 = T.merge_window(None, wins[0], 32, origin=(1.0, -2.0, 0.7)); c0 = T.canonicalize(wins[0][None])[0]
check(abs(m0[0, 0] - 1.0) < 1e-5 and abs(m0[0, 2] + 2.0) < 1e-5 and abs(m0[0, 3] - 0.7) < 1e-5 and abs(float(np.hypot(m0[-1, 0] - 1.0, m0[-1, 2] + 2.0)) - float(np.hypot(c0[-1, 0], c0[-1, 2]))) < 1e-4,
      "the first window is placed at origin = (x, z, yaw) and keeps its shape (the body's pose in the closed loop)")
try:
    T.merge_window(np.zeros((5, 30), np.float32), wins[0], 32); check(False, "a pending tail of the wrong length is refused")
except AssertionError:
    check(True, "a pending tail of the wrong length is refused")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- 3
print("[closed_loop.plan_window_goal]")
import closed_loop as CL
stats = dict(mean=[0, 0, 0, 0], std=[1, 1, 1, 1], turn_max=0.7, speed_cruise=1.1, speed_max=1.6)
st = np.zeros(35, np.float32)
for (px, pz, yaw, gx, gz) in ((0, 0, 0, 0.3, 1.2), (2, -1, 0.5, 2.6, 0.4), (0, 0, 0, 0.5, 3.0), (1, 2, 0.4, 2.5, 4.0), (-1, 0.5, -1.0, -2.0, 2.0), (0, 0, 2.0, 0.3, 1.0)):
    st[0], st[2], st[3] = px, pz, yaw
    feat, info = CL.plan_window_goal(st, (gx, gz), stats, 2.1, "arc")
    dx, dz = feat[1], feat[2]; ex = px + dx * np.cos(yaw) + dz * np.sin(yaw); ez = pz - dx * np.sin(yaw) + dz * np.cos(yaw)       # the arc's end, canonical -> world
    reached = np.hypot(ex - gx, ez - gz)
    if not info["turn_clipped"]:
        full = np.hypot(gx - px, gz - pz) <= 1.1 * 2.1 * np.sinc(info["turn"] / 2 / np.pi) + 1e-6
        check((reached < 1e-3) if full else (np.hypot(ex - px, ez - pz) > 0.9 * 1.1 * 2.1 * np.sinc(info["turn"] / 2 / np.pi)),
              f"pose ({px:g}, {pz:g}, yaw {yaw:g}) -> target ({gx:g}, {gz:g}): unclipped arc ends {reached:.4f} m from the target" + ("" if full else " (target farther than one window: ends on the way)"))
    else:
        check(abs(abs(feat[0]) - stats["turn_max"]) < 1e-6 and info["remaining_turn"] is not None, f"pose ({px:g}, {pz:g}, yaw {yaw:g}) -> target ({gx:g}, {gz:g}): remaining turn {np.degrees(info['remaining_turn']):.0f} deg is clipped to turn_max")
    check(abs(feat[3] * 2.1 - info["path"]) < 1e-4 and feat[3] <= stats["speed_cruise"] + 1e-6, "  speed never above cruise; path / seconds = speed")
st[:] = 0; feat, info = CL.plan_window_goal(st, (0.0, 0.2), stats, 2.1, "arc")
check(feat[3] < 0.2, f"a target 0.2 m ahead asks for a slow window ({feat[3]:.2f} m/s), not a cruise-speed overshoot")
feat, info = CL.plan_window_goal(st, None, stats, 2.1, "rate", turn_deg=60.0)
check(abs(feat[0] - 0.7) < 1e-6 and info["turn_clipped"], "rate mode: 60 deg per window is above turn_max 0.7 rad and is clipped")
feat, info = CL.plan_window_goal(st, None, stats, 2.1, "rate", turn_deg=-20.0)
check(abs(feat[0] + np.radians(20)) < 1e-6 and not info["turn_clipped"] and feat[1] < 0, "rate mode: -20 deg per window is passed through (turns right: dx < 0)")

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED"); sys.exit(1)
print("all goal-conditioning checks passed")
