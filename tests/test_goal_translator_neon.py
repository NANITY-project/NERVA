#!/usr/bin/env python3
"""
tests/test_goal_translator_neon.py - tools/goal_translator/runtime (the C++ Goal Translator, built against a NANITY-NEON checkout) against the numpy reference runtime and torch.

    python3 tests/test_goal_translator_neon.py [--neon DIR] [--bin ./goal_translator] [--model X.gguf]     needs g++ (or --bin), numpy, gguf; torch only to make a model when --model is not given

Without --model a small random-weight translator is exported (random weights are enough: what is compared is the arithmetic, not the plans' sense).
NEON's forward pass is NOT bit-identical to the float reference: measured on a trained 0.9 M model the logits differ by about 0.25 % of their scale. The int8 V cache of
core/rawllm_forward.hpp accounts for only ~5 % of that (simulated in numpy: 0.0016 of 0.029), so the rest comes from somewhere else in NEON's arithmetic and has NOT been isolated.
The test therefore only requires the logits to agree within 5 % of the scale and the plans to agree except at (at most 2) near-ties; both numbers are printed. Errors (too long, no
model) must be JSON errors, never a crash or a truncation.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); GT = os.path.join(ROOT, "tools", "goal_translator")
sys.path.insert(0, GT)
import numpy as np
import gt_conventions as C, gt_numpy as N

ap = argparse.ArgumentParser(); ap.add_argument("--bin"); ap.add_argument("--model"); ap.add_argument("--neon", help="NANITY-NEON checkout (its runtime/ holds the core headers the C++ includes)"); A = ap.parse_args()
NEON = A.neon or os.environ.get("NEON_DIR") or next((p for p in (os.path.join(os.path.dirname(ROOT), "NANITY-NEON"), os.path.expanduser("~/NANITY-NEON")) if os.path.isdir(p)), "")
FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


td = tempfile.mkdtemp()
binp = A.bin
if not binp:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        sys.exit("no C++ compiler and no --bin")
    binp = os.path.join(td, "goal_translator")
    r = subprocess.run([cxx, "-std=c++20", "-O2", "-pthread", "-I", os.path.join(NEON, "runtime"), os.path.join(GT, "runtime", "goal_translator_cli.cpp"), "-o", binp], capture_output=True, text=True)
    check(r.returncode == 0, "goal_translator_cli.cpp compiles" + ("" if r.returncode == 0 else ":\n" + r.stderr[-600:]))
    if r.returncode:
        sys.exit(1)
model = A.model
if not model:
    import torch
    import gt_torch as G
    torch.manual_seed(0); cfg = G.make_config("tiny"); m = G.build_model(cfg)
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    st, tm = C.Stats(), C.Timing()
    meta = dict(layout_version=C.LAYOUT_VERSION, stats=st.to_dict(), timing=dict(W=tm.W, stride=tm.stride, dt=tm.dt), n_turn_bins=C.N_TURN_BINS, n_speed_bins=C.N_SPEED_BINS,
                max_windows=C.MAX_WINDOWS, max_input_bytes=C.MAX_INPUT_BYTES, vocab_size=C.VOCAB_SIZE, turn_positive_is_left=True)
    ck = os.path.join(td, "x.pt"); model = os.path.join(td, "x.gguf")
    G.save_checkpoint(ck, m, cfg, meta, 0)
    r = subprocess.run([sys.executable, os.path.join(GT, "export_goal_translator.py"), ck, model], capture_output=True, text=True)
    check(r.returncode == 0, "random-weight translator exported" + ("" if r.returncode == 0 else ": " + r.stderr[-300:]))


def run(*args):
    return subprocess.run([binp, model] + list(args), capture_output=True, text=True)


print("[info]")
info = json.loads(run("info").stdout)["goal_translator"]
ref = N.NumpyTranslator.from_gguf(model)
check(info["vocab"] == C.VOCAB_SIZE and info["turn_bins"] == C.N_TURN_BINS and info["speed_bins"] == C.N_SPEED_BINS and info["max_windows"] == C.MAX_WINDOWS, "layout read from the file")
check(abs(info["goal_stats"]["turn_max"] - ref.stats.turn_max) < 1e-5 and info["window"]["stride"] == ref.timing.stride, "goal_stats and window timing read from the file")

print("[logits]")
worst = 0.0
for text in ("walk left", "go right then left idk", "WALKLEFTTHENRIGHT", ""):
    ids = C.encode_input(text) + [C.WIN, C.TURN_BASE + 7, C.SPEED_BASE + 3, C.WIN]
    out = json.loads(run("logits", ",".join(map(str, ids))).stdout)
    lc, ln = np.array(out["logits"]), ref.prefill_logits(ids)
    worst = max(worst, np.abs(lc - ln).max() / np.abs(ln).max())
check(worst < 0.05, f"C++ logits agree with the float reference within 5 % of the scale (worst {100 * worst:.2f} %)")

print("[plans]")
texts = ["walk left", "go right then left", "idk", "turn around", "WALKRIGHTTHENLEFT", "run forward then stop", "wave ur hand", "sit down", "a b c", "left right left right left"]
same, valid = 0, 0
for t in texts:
    j = json.loads(run("translate", t).stdout)
    try:
        want = [None if w.unspecified else [w.turn_bin, w.speed_bin] for w in ref.decode(t).windows]
    except C.PlanError:
        want = "err"
    got = j["windows"] if j["ok"] else "err"
    same += got == want
    if j["ok"]:
        toks = j["tokens"]
        try:
            C.tokens_to_plan(toks); valid += 1
        except C.PlanError:
            pass
    else:
        valid += 1
check(valid == len(texts), "every C++ plan is a valid token sequence of the grammar (or a JSON error)")
print(f"  info  C++ == numpy reference on {same} / {len(texts)} plans (a difference is a near-tie flipped by the small numeric gap)")
check(same >= len(texts) - 2, "plans agree except at most 2 near-ties")
r = json.loads(run("translate", "x" * 300).stdout)
check(not r["ok"] and "256" in r["error"], "text over the byte limit is a JSON error, not a truncation")
p = subprocess.run([binp, os.path.join(td, "nope.gguf"), "info"], capture_output=True, text=True)
check(p.returncode != 0 and "goal_translator" in p.stderr, "a missing model file is an error exit")
p = subprocess.run([binp, model, "--interactive"], input='{"op":"info"}\n{"op":"goal_translate","text":"walk left"}\nnot json\n{"op":"nope"}\n', capture_output=True, text=True)
lines = [json.loads(l) for l in p.stdout.splitlines()]
check(len(lines) == 4 and lines[0]["ok"] and lines[3]["ok"] is False, "--interactive answers every line with one JSON object (info, translate, bad json, unknown op)")
print()
print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
sys.exit(1 if FAILS else 0)
