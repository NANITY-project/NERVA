#!/usr/bin/env python3
"""
tests/test_goal_plan.py - UI/goal_plan.py (the Goal Translator behind the Steering box) and its fit with server.parse_timeline.

    python3 tests/test_goal_plan.py
    needs numpy; torch + gguf (pip install torch gguf) add: conventions parity with train_cvqvae, and a trained-model backend built from random weights

  1 planner     the motivating prompt becomes right, left, unspecified rows; the wave is reported as ignored; nothing is guessed; errors are errors
  2 clamping    a prior whose data holds less turn / speed than asked gets clamped values and a note
  3 timeline    server.parse_timeline takes those rows, an unspecified row has no goal and no asked-vs-got, the numbers equal arc_goal over (W - 1) * dt
  4 parity      (torch) the translator's arc_goal / sign convention is train_cvqvae's own
  5 model       (torch) a translator GGUF exported by tools/goal_translator gives plans through the same interface, source "model"
"""
import argparse, json, math, os, subprocess, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "UI")); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np
import goal_plan

FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


STATS = dict(mean=[0, 0, 0.5, 0.8], std=[0.3, 0.3, 0.5, 0.5], turn_max=0.9, speed_cruise=1.2, speed_max=2.1)
W, STRIDE, DT = 64, 48, 1.0 / 30.0
gp = goal_plan.GoalPlanner()
C = gp.C

print("[planner: stage C]")
r = gp.plan("Walkrightandthenleftthenwaveurhandandidkmovearound?", STATS, W, STRIDE, DT)
kinds = ["U" if x.get("unspecified") else ("L" if x["turn"] > 0 else "R" if x["turn"] < 0 else "S") for x in r["rows"]]
check("".join(kinds) == "RRLLUU" and r["source"] == "stage_c", f"right, left, unspecified windows, source reported as stage_c: {''.join(kinds)}")
check(r["ignored"] and "wave" in r["ignored"][0], f"the wave is reported as not expressible: {r['ignored']}")
check(all(abs(x["turn"]) <= math.degrees(STATS["turn_max"]) + 0.1 and x["speed"] <= STATS["speed_max"] for x in r["rows"] if not x.get("unspecified")), "no row asks for more than the prior's data holds")
for bad, why in (("", "empty"), ("x" * 300, "too long"), ("sit down and dance", "nothing expressible")):
    try:
        gp.plan(bad, STATS, W, STRIDE, DT); check(False, f"{why}: an error")
    except goal_plan.GoalPlanError as e:
        check(True, f"{why}: an error the page can show ({str(e)[:70]})")
try:
    gp.plan("walk left", None, W, STRIDE, DT); check(False, "a prior without goal_stats is refused")
except goal_plan.GoalPlanError:
    check(True, "a prior without goal_stats is refused")

print("[clamping to the prior]")
small = dict(STATS, turn_max=0.2, speed_max=1.0, speed_cruise=0.8)
r2 = gp.plan("turn left 90 degrees then run", small, W, STRIDE, DT)
specified = [x for x in r2["rows"] if not x.get("unspecified")]
check(all(abs(x["turn"]) <= math.degrees(0.2) + 0.1 and x["speed"] <= 1.0 for x in specified), "turn and speed are clamped to the prior's turn_max / speed_max")

print("[timeline]")
import server
rows = r["rows"]
tc = types.SimpleNamespace(arc_goal=C.arc_goal)
p = types.SimpleNamespace(goal_dim=4, goal_stats=STATS)
tl = server.parse_timeline(dict(timeline=json.dumps(rows)), tc, p, W, DT)
check(len(tl) == len(rows) and [x["goal"] is None for x in tl] == [bool(x.get("unspecified")) for x in rows], "parse_timeline accepts the planner's rows; unspecified rows have goal None")
sec = (W - 1) * DT
g0 = tl[0]["goal"]; want = C.arc_goal(math.radians(rows[0]["turn"]), rows[0]["speed"] * sec, sec)
check(np.allclose(g0, want, atol=1e-6), "the goal numbers of a row are arc_goal over (W - 1) * dt")
check(g0[0] < 0 and g0[1] < 0 and tl[2]["goal"][0] > 0 and tl[2]["goal"][1] > 0, "right turn: dyaw < 0 and dx < 0; left turn: both > 0")
try:
    server.parse_timeline(dict(timeline=json.dumps([{"unspecified": True}] * 600)), tc, p, W, DT); check(False, "over 500 windows is refused")
except RuntimeError:
    check(True, "over 500 windows is refused")

try:
    import torch, gguf
    HAVE = True
except ImportError:
    HAVE = False
    print("[torch] skipped (pip install torch gguf)")
if HAVE:
    print("[parity with train_cvqvae]")
    import train_cvqvae as T
    t_, p_ = np.array([0.5, -0.5, 0.0, 0.3]), np.array([1.0, 2.0, 0.5, 0.0])
    check(np.allclose(T.arc_goal(t_, p_, 2.0), C.arc_goal(t_, p_, 2.0), atol=1e-6), "arc_goal of the translator = train_cvqvae.arc_goal")
    check(T.GOAL_NAMES == C.GOAL_NAMES, "goal names agree")
    n = 5; Xc = np.zeros((1, 64, 30), np.float32); h = 0.3
    for i in range(64):
        Xc[0, i, 3] = h * i / 63; Xc[0, i, 0] = 0; Xc[0, i, 2] = i * 0.02
    gf = T.goal_features(Xc, np.array([64]), DT)[0]
    check(gf[0] > 0, "train_cvqvae.goal_features: yaw growing is dyaw > 0 (the sign convention the translator uses for LEFT)")
    print("[model backend]")
    with tempfile.TemporaryDirectory() as td:
        gtdir = os.path.join(ROOT, "tools", "goal_translator")
        code = ("import sys, torch; sys.path.insert(0, %r); import gt_torch as G, gt_conventions as C\n"
                "torch.manual_seed(1); cfg = G.make_config('tiny'); m = G.build_model(cfg)\n"
                "[p.data.add_(0.05 * torch.randn_like(p)) for p in m.parameters()]\n"
                "meta = dict(layout_version=C.LAYOUT_VERSION, stats=dict(turn_max=0.9, speed_cruise=1.2, speed_max=2.1), timing=dict(W=64, stride=48, dt=1/30), n_turn_bins=C.N_TURN_BINS,\n"
                "            n_speed_bins=C.N_SPEED_BINS, max_windows=C.MAX_WINDOWS, max_input_bytes=C.MAX_INPUT_BYTES, vocab_size=C.VOCAB_SIZE, turn_positive_is_left=True)\n"
                "G.save_checkpoint(%r, m, cfg, meta, 0)\n") % (gtdir, os.path.join(td, "x.pt"))
        ok1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        ok2 = subprocess.run([sys.executable, os.path.join(gtdir, "export_goal_translator.py"), os.path.join(td, "x.pt"), os.path.join(td, "x.gguf")], capture_output=True, text=True)
        check(ok1.returncode == 0 and ok2.returncode == 0, "a random-weight translator GGUF was exported" + ("" if ok2.returncode == 0 else ": " + (ok1.stderr + ok2.stderr)[-300:]))
        if ok2.returncode == 0:
            gm = goal_plan.GoalPlanner(os.path.join(td, "x.gguf"))
            check(gm.info()["source"] == "model", "info() says model")
            try:
                rm = gm.plan("walk left then right", STATS, W, STRIDE, DT)
                check(rm["source"] == "model" and all(x.get("unspecified") or ("turn" in x and "speed" in x) for x in rm["rows"]), f"the model backend returns {rm['windows']} well-formed rows, source model")
            except goal_plan.GoalPlanError as e:
                check(True, f"the model backend raised its error to the user instead of falling back: {str(e)[:80]}")
            try:
                goal_plan.GoalPlanner(os.path.join(td, "missing.gguf")); check(False, "a missing model file is an error")
            except goal_plan.GoalPlanError:
                check(True, "a missing / bad model file is an error, not a silent switch to stage C")

print()
print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
sys.exit(1 if FAILS else 0)
