#!/usr/bin/env python3
"""
tests/test_ui_goal_timeline.py - server.generate() with a goal TIMELINE that has UNSPECIFIED windows (UI/server.py parse_timeline + generate, UI/goal_plan.py rows).

  python tests/test_ui_goal_timeline.py      numpy only (no torch, no NEON binary, no libnebula, no weights)

Real: UI/server.py, UI/goal_plan.py, the Goal Translator's stage C, and arc_goal / goal_features / canonicalize of tools/train_cvqvae.py (taken out of the file with ast, it imports torch).
Stand-ins: a goal-conditioned prior that records what it is told, a NEON session that records the JSON ops it receives, no body engine.

  1 numpy prior   a row with numbers declares its goal (arc_goal over (W - 1) * dt) before that window; an unspecified row declares NONE (the null goal), never the previous one;
                  'goal' (asked vs got) events come only from windows that asked for something
  2 NEON          cvq_prior_next carries "goal": [4 numbers] for a specified row and "goal": null for an unspecified one
  3 text -> run   the planner's rows for the motivating prompt run end to end: R R L L U U
"""
import argparse, ast, importlib.util, json, math, os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "UI"))
FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


src = open(os.path.join(ROOT, "tools", "train_cvqvae.py")).read(); tree = ast.parse(src)
want_fn, want_var = {"canonicalize", "to_world", "stitch", "goal_features", "arc_goal"}, {"NCH", "XZ", "YAWS", "FLAGS", "GOAL_NAMES", "NG"}
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want_fn) or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want_var for t in n.targets))]
shim = types.ModuleType("train_cvqvae_shim"); shim.np = np
exec(compile(ast.Module(body=keep, type_ignores=[]), "train_cvqvae(shim)", "exec"), shim.__dict__)
spec = importlib.util.spec_from_file_location("nerva_ui_server", os.path.join(ROOT, "UI", "server.py")); server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
server.ARGS = argparse.Namespace(neon_dir=None, neon_bin=None, no_clone=True, numpy_prior=False, no_engine=True, threads=0, trust_pickle=False, lm="", lm_model="", lm_key_env="", goal_translator="")
server.train_cvqvae_mod = lambda: shim
server.get_engine = lambda: None
S = server.S

NAMES = ["walk", "run"]
K, NCODE, WINDOW, STRIDE, DT = 32, 16, 64, 32, 1 / 30
STATS = dict(mean=[0, 0, 0.5, 0.8], std=[0.3, 0.3, 0.5, 0.5], turn_max=0.9, speed_cruise=1.2, speed_max=2.1)


class FakePrior:
    SEP, ACT0, NULL = K, K + 1, K + 1 + len(NAMES)
    n, windows, stride, names, params = NCODE, 8, STRIDE, NAMES, 1
    n_actions, L, d, K = len(NAMES), [], 1, K
    goal_dim, goal_names, goal_stats, n_styles = 4, shim.GOAL_NAMES, STATS, 0

    def __init__(self):
        self.declared, self.cfg = [], 1.0

    def reset(self, style=None):
        self.declared = []

    def add_window_goal(self, raw):
        self.declared.append(None if raw is None else [float(x) for x in raw])

    def set_cfg_scale(self, s):
        self.cfg = s

    def sample(self, ctx, rng, temperature, top_k, top_p):
        return int(rng.integers(0, K)), 1.0, 0.5


def window(turn_rad=0.0, v=1.2):
    w = np.zeros((WINDOW, 30), np.float32); t = np.arange(WINDOW) / 30.0
    w[:, 1] = 0.95; yaw = turn_rad * t / t[-1]; w[:, 3] = yaw
    w[:, 0] = np.cumsum(v / 30 * np.sin(yaw)); w[:, 2] = np.cumsum(v / 30 * np.cos(yaw))
    for b in (4, 9):
        w[:, b + 4] = 1.0
    return w


class FakeSession:
    name = "stand-in NEON"

    def __init__(self):
        self.requests, self.k = [], 0

    def decode(self, codes, action):
        return window(0.3)

    def ask(self, req):
        self.requests.append(req)
        if req["op"] != "cvq_prior_next":
            return {}
        g = req.get("goal")
        turn = 0.0 if g is None else g[0]
        return dict(codes=list(range(NCODE)), frames=window(turn).tolist(), entropy=[1.0] * NCODE, probs=[0.5] * NCODE)

    def close(self):
        pass


def install(native):
    S.vae = dict(cfg=dict(window=WINDOW, downs=2, num_codes=K, n_actions=len(NAMES)), names=NAMES, dt=DT, gguf="x", file="vae.pt")
    S.prior, S.prior_file = FakePrior(), "prior.pt"
    S.session, S.native_prior, S.native_goal = FakeSession(), native, native


def run(rows, native):
    install(native)
    return list(server.generate(dict(timeline=json.dumps(rows), seed="1"), lambda: False)), S.prior, S.session


sec = (WINDOW - 1) * DT
rows = [dict(turn=-25.0, speed=1.2), dict(turn=25.0, speed=1.0), {"unspecified": True}, {"unspecified": True}]

print("[numpy prior]")
evs, pr, _ = run(rows, native=False)
check(len([e for e in evs if e["type"] == "window"]) == 4, "a timeline with no window count runs one window per row")
check([g is None for g in pr.declared[:4]] == [False, False, True, True], f"goals declared per window: {['goal' if g else 'none' for g in pr.declared[:4]]}")
want0 = shim.arc_goal(math.radians(-25.0), 1.2 * sec, sec)
check(np.allclose(pr.declared[0], want0, atol=1e-5), "window 1's goal is arc_goal(turn, speed * (W - 1) * dt, ...)")
check(pr.declared[0][0] < 0 and pr.declared[1][0] > 0, "right is dyaw < 0, left dyaw > 0")
ge = [e for e in evs if e["type"] == "goal"]
check([e["w"] for e in ge] == [0, 1], f"asked-vs-got events only for the windows that asked for something: windows {[e['w'] for e in ge]}")
check(next(e for e in evs if e["type"] == "start")["goals"] == 4, "start event reports 4 window goals")

print("[NEON session]")
evs, pr, sess = run(rows, native=True)
nx = [r for r in sess.requests if r["op"] == "cvq_prior_next"]
check(len(nx) == 4 and "goal" in nx[0] and len(nx[0]["goal"]) == 4, "cvq_prior_next carries 4 goal numbers for a specified row")
check(all("goal" in r and r["goal"] is None for r in nx[2:]), "an unspecified row sends \"goal\": null (explicitly no goal, not the previous window's)")
check(all(r["op"] != "cvq_prior_reset" or "goals" not in r for r in sess.requests), "no plan is attached to the reset: every window carries its own goal")

print("[text -> run]")
import goal_plan
gp = goal_plan.GoalPlanner()
res = gp.plan("Walkrightandthenleftthenwaveurhandandidkmovearound?", STATS, WINDOW, STRIDE, DT)
evs, pr, sess = run(res["rows"], native=False)
check([g is None for g in pr.declared[:6]] == [False] * 4 + [True] * 2, "the motivating prompt: 4 windows with goals, then 2 with none")
check(len([e for e in evs if e["type"] == "goal"]) == 4 and evs[-1]["type"] == "done", "asked-vs-got for the 4 specified windows, run finishes")
print()
print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
sys.exit(1 if FAILS else 0)
