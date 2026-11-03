#!/usr/bin/env python3
"""
tests/test_goal_layout.py - the prior's conditioning layout with engine-channel targets (tools/goal_layout.py) and the places that build or consume it.

  python tests/test_goal_layout.py      numpy only (no torch, no NEON, no engine, no weights)

  1 layout        25 names, 5 groups that partition the columns, channel columns match the 30-channel action vector
  2 features      a window's groups measured from the data: known / on / target point, valid frames only, the on-threshold
  3 stats         scale-only: mean 0, std 1 for flags, rms for numbers; so a masked or unspecified entry is 0 as raw AND as normalised number (NEON / numpy prior change nothing)
  4 planners      expand_path (path-only priors unchanged), set_channel / unspecify: the three states of a group
  5 masking       p_win / p_seq, kept groups untouched, dropped groups zero including their known flag
  6 server        parse_timeline gives a 25-number goal for a full-layout prior and the same 4 numbers as before for an old one
  7 closed loop   GL.measured: the features of a decoded window in the prior's layout
Not covered here (needs torch / a trained prior): training with the masks, and sampling from such a prior.
"""
import argparse, ast, importlib.util, json, math, os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "UI"))
import goal_layout as GL  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# train_cvqvae imports torch: take the numpy functions out of the file with ast (the same trick tests/test_ui_goal_timeline.py uses)
src = open(os.path.join(ROOT, "tools", "train_cvqvae.py")).read(); tree = ast.parse(src)
want_fn, want_var = {"canonicalize", "to_world", "stitch", "goal_features", "arc_goal", "goal_stats", "norm_goal"}, {"NCH", "XZ", "YAWS", "FLAGS", "GOAL_NAMES", "NG"}
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want_fn) or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want_var for t in n.targets))]
shim = types.ModuleType("train_cvqvae_shim"); shim.np = np
exec(compile(ast.Module(body=keep, type_ignores=[]), "train_cvqvae(shim)", "exec"), shim.__dict__)


def windows():
    """4 canonical windows [4,64,30]: none on | handR on throughout | handR on 40% | handR on 60% of the 50 valid frames (junk after them)."""
    X = np.zeros((4, 64, 30), np.float32)
    X[:, :, 1] = 0.0
    X[1, :, 21] = 1.0; X[1, :, 18], X[1, :, 19], X[1, :, 20] = 0.2, 1.1, 0.6
    X[2, :26, 21] = 1.0; X[2, :26, 18:21] = (0.5, 1.5, 0.9)
    X[3, :30, 21] = 1.0; X[3, :30, 18] = np.linspace(0.0, 0.4, 30); X[3, :30, 19], X[3, :30, 20] = 1.0, 0.5
    X[3, 50:, 21] = 1.0; X[3, 50:, 18:21] = 9.0                                 # beyond the valid frames: must be ignored
    X[0, :, 14:17] = (0.3, 0.9, 0.1)                                            # handL positions with on = 0: not a target, must not leak into the goal
    return X, np.array([64, 64, 64, 50])


print("[1] layout")
check(len(GL.FULL_NAMES) == 25 and GL.FULL_NAMES[0] == "path_known" and GL.FULL_NAMES[1:5] == ["dyaw", "dx", "dz", "speed"], "25 names: path_known, dyaw, dx, dz, speed, then 4 x (known, on, x, y, z)")
cols = sorted(c for g in GL.GROUPS for c in GL.GROUP_COLS[g])
check(cols == list(range(25)) and GL.GROUP_MATRIX.sum(0).tolist() == [1.0] * 25, "the 5 groups partition the 25 columns")
check([GL.FULL_NAMES[c] for c in GL.GROUP_COLS["handR"]] == ["handR_known", "handR_on", "handR_x", "handR_y", "handR_z"], "group columns carry the right names")
check({g: c for g, c in GL.CHANNEL_COLS.items()} == {"handL": (14, 15, 16, 17), "handR": (18, 19, 20, 21), "look": (22, 23, 24, 25), "torso": (26, 27, 28, 29)}, "channel columns = the 30-channel action vector (x y z on)")
check(GL.layout_of([0] * 4) == "path" and GL.layout_of({"mean": [0] * 25}) == "full" and GL.layout_of(25) == "full", "layout_of: 4 -> path, 25 -> full")
try:
    GL.layout_of(7)
    check(False, "an unknown size is refused")
except ValueError:
    check(True, "an unknown size is refused")

print("[2] features measured from windows")
X, valid = windows()
path4 = np.arange(16, dtype=np.float32).reshape(4, 4)
g = GL.goal_features_full(path4, X, valid)
S = GL.GROUP_COLS
check(g.shape == (4, 25) and np.array_equal(g[:, 1:5], path4) and np.all(g[:, 0] == 1.0), "path numbers carried over, path known")
check(np.all(g[:, [S[k][0] for k in GL.GROUPS[1:]]] == 1.0), "every measured group is known")
check(np.all(g[0, S["handR"][1:]] == 0) and np.all(g[0, S["handL"][1:]] == 0), "channel off: on = 0 and no target numbers (handL positions with on = 0 do not leak)")
check(g[1, S["handR"][1]] == 1 and np.allclose(g[1, S["handR"][2:]], [0.2, 1.1, 0.6]), "channel on throughout: on = 1, target = the point")
check(g[2, S["handR"][1]] == 0 and np.all(g[2, S["handR"][2:]] == 0), "on for 40% of the frames (below the 50% threshold): treated as off")
check(g[3, S["handR"][1]] == 1 and np.allclose(g[3, S["handR"][2:]], [0.2, 1.0, 0.5], atol=1e-6), "valid frames only: 60% of 50 frames on, target = mean over the on frames, junk after the valid part ignored")
check(g.dtype == np.float32, "float32")

print("[3] scale-only statistics")
legacy = dict(mean=[0, 0, 0, 0], std=[1, 1, 1, 1], turn_max=0.9, speed_cruise=1.2, speed_max=2.1)
st = GL.goal_stats_full(g, legacy)
check(st["mean"] == [0.0] * 25 and len(st["std"]) == 25 and st["layout"] == "full" and st["names"] == GL.FULL_NAMES, "mean 0 everywhere, 25 stds, layout and names stored")
check(st["turn_max"] == 0.9 and st["speed_cruise"] == 1.2 and st["speed_max"] == 2.1, "turn_max / speed_cruise / speed_max carried over for the planners")
flags = [c for k in GL.GROUPS for c in (S[k][0],)] + [S[k][1] for k in GL.GROUPS[1:]]
check(all(st["std"][c] == 1.0 for c in flags), "flags keep std 1 (known 0/1, on 0/1)")
on_rows = g[:, S["handR"][1]] > 0.5
check(abs(st["std"][S["handR"][3]] - float(np.sqrt((g[on_rows, S["handR"][3]].astype(np.float64) ** 2).mean()))) < 1e-9, "a target coordinate's std = rms over the windows where the group is on")
check(abs(st["on_share"]["handR"] - 0.5) < 1e-9 and st["on_share"]["handL"] == 0.0, "on_share reports how much of the data has each channel on (here handR 2 of 4)")
norm = lambda raw: (np.asarray(raw, np.float32) - np.asarray(st["mean"], np.float32)) / np.asarray(st["std"], np.float32)
check(np.all(norm(np.zeros(25)) == 0.0), "a masked / unspecified entry (raw 0) is exactly 0 for the network: NEON and the numpy prior ((raw - mean) / std) need no change")

print("[4] planners")
a4 = np.array([0.5, 0.1, 1.2, 1.1], np.float32)
check(GL.expand_path(a4, legacy) is a4 or np.array_equal(GL.expand_path(a4, legacy), a4), "path-only prior: the 4 numbers unchanged")
e = GL.expand_path(a4, st)
check(e.shape == (25,) and e[0] == 1 and np.array_equal(e[1:5], a4) and not e[5:].any(), "full prior: path known + 4 numbers, every channel group unspecified (known 0, zeros)")
check(GL.expand_path(np.tile(a4, (3, 6, 1)), st).shape == (3, 6, 25), "batches broadcast")
t = GL.set_channel(e, st, "handR", True, (0.3, 1.2, 2.0))
check(t[S["handR"][0]] == 1 and t[S["handR"][1]] == 1 and np.allclose(t[S["handR"][2:]], [0.3, 1.2, 2.0]) and np.array_equal(t[:5], e[:5]) and not t[S["look"]].any(), "set_channel on: known, on, the point; other groups untouched")
o = GL.set_channel(t, st, "handR", False)
check(o[S["handR"][0]] == 1 and o[S["handR"][1]] == 0 and not o[S["handR"][2:]].any(), "set_channel off: known 1, on 0, no point = 'the engine's procedural layer drives this part'")
u = GL.unspecify(t, st, "handR")
check(not u[S["handR"]].any(), "unspecify: known 0 and all zeros = 'nothing is said'")
for bad in (lambda: GL.set_channel(e, st, "handR", True), lambda: GL.set_channel(e, st, "knee", False), lambda: GL.set_channel(a4, legacy, "handR", False), lambda: GL.set_channel(e, st, "handR", True, (1, 2, float("nan")))):
    try:
        bad()
        check(False, "invalid channel requests are refused")
        break
    except ValueError:
        pass
else:
    check(True, "invalid channel requests are refused (no point for an 'on', unknown group, path-only prior, non-finite point)")

print("[5] masking (partial control)")
rng = np.random.default_rng(0)
B, L = 4000, 5
base = np.ones((B, L, 25), np.float32)
check(np.array_equal(GL.apply_keep(base, GL.sample_keep(rng, B, L, 0.0, 0.0)), base), "p = 0: nothing hidden")
z = GL.apply_keep(base, GL.sample_keep(rng, B, L, 1.0, 0.0))
check(not z.any(), "p_win = 1: everything hidden, known flags included")
keep = GL.sample_keep(rng, B, L, 0.25, 0.0)
m = GL.apply_keep(base, keep)
per_group_rate = 1 - keep.mean(axis=(0, 1))
check(np.all(np.abs(per_group_rate - 0.25) < 0.02), f"p_win = 0.25: each group hidden in about 25% of windows ({np.round(per_group_rate, 3).tolist()})")
ok = True
for gi, gname in enumerate(GL.GROUPS):
    cols = GL.GROUP_COLS[gname]
    ok &= bool(np.all(m[..., cols] == keep[..., gi:gi + 1].astype(np.float32)))       # a hidden group is all 0 (known flag included), a kept one untouched
check(ok, "a dropped group is zero in all its columns including known; a kept group is bit-identical")
flat = keep.reshape(B * L, 5)
corr = np.corrcoef((~flat).astype(float).T)
check(abs(flat.all(1).mean() - 0.75 ** 5) < 0.01 and (~flat.any(1)).mean() < 0.004 and float(np.abs(corr - np.eye(5)).max()) < 0.03,
      f"groups are hidden independently (all kept {flat.all(1).mean():.3f} vs {0.75 ** 5:.3f} expected, max pairwise correlation {float(np.abs(corr - np.eye(5)).max()):.3f})")
keep = GL.sample_keep(rng, B, L, 0.0, 0.5)
same = np.all(keep == keep[:, :1, :], axis=1)
check(bool(same.all()) and abs((~keep[:, 0, :]).mean() - 0.5) < 0.02, "p_seq = 0.5: a group is hidden in ALL windows of a sequence or none, about half the sequences")
mix = GL.sample_keep(rng, B, L, 0.25, 0.15)
check(abs((~mix).mean() - (1 - 0.75 * 0.85)) < 0.015, "both together: hidden share = 1 - (1 - p_win)(1 - p_seq)")

print("[6] server.parse_timeline")
spec = importlib.util.spec_from_file_location("nerva_ui_server", os.path.join(ROOT, "UI", "server.py")); server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
rows_json = json.dumps([{"turn": 30, "speed": 1.0}, {"unspecified": True}, {"turn": -20, "distance": 1.5}])
W, dt = 64, 1 / 30
old_p = types.SimpleNamespace(goal_dim=4, goal_stats=legacy)
full_p = types.SimpleNamespace(goal_dim=25, goal_stats=st)
r_old = server.parse_timeline({"timeline": rows_json}, shim, old_p, W, dt)
r_full = server.parse_timeline({"timeline": rows_json}, shim, full_p, W, dt)
want0 = shim.arc_goal(math.radians(30), 1.0 * (W - 1) * dt, (W - 1) * dt)
check(len(r_old[0]["goal"]) == 4 and np.allclose(r_old[0]["goal"], want0, atol=1e-6), "old prior (4 numbers): exactly the numbers the server made before this change")
check(len(r_full[0]["goal"]) == 25 and np.allclose(r_full[0]["goal"][1:5], want0, atol=1e-6) and r_full[0]["goal"][0] == 1.0 and not any(r_full[0]["goal"][5:]), "full prior: 25 numbers, path known, every channel group unspecified")
check(r_old[1]["goal"] is None and r_full[1]["goal"] is None, "an unspecified window still means NO goal (the null goal) for both")
check(len(r_full[2]["goal"]) == 25 and r_full[2]["turn"] == -20, "other rows and their asked-for numbers unchanged")

print("[7] closed loop: what a decoded window measured")
Xd, vd = windows()
one = Xd[1:2]
m_full = GL.measured(shim, one, np.array([64]), dt, st)
m_path = GL.measured(shim, one, np.array([64]), dt, legacy)
check(m_path.shape == (4,) and np.allclose(m_path, shim.goal_features(one, np.array([64]), dt)[0]), "path-only prior: the usual 4 measured numbers")
check(m_full.shape == (25,) and np.allclose(m_full[1:5], m_path) and m_full[S["handR"][1]] == 1 and np.allclose(m_full[S["handR"][2:]], [0.2, 1.1, 0.6], atol=1e-6), "full prior: the same path numbers + the decoded window's channel state")

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED")
    sys.exit(1)
print("\nall goal-layout checks passed")
