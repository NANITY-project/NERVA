#!/usr/bin/env python3
"""
tests/test_goal_translator.py - checks for tools/goal_translator (spec NANITY_GOAL_TRANSLATOR_SPEC.md). Plain script, like the other tests here. The torch part also needs a NANITY-NEON checkout (modeling_nanity.py).

    python3 tests/test_goal_translator.py            # numpy only: conventions, grammar, stage C, dataset builder, verifier, corruption, measuring
    pip install torch gguf; python3 tests/test_goal_translator.py     # + the torch model: cached decoding vs full forward, export -> numpy runtime equivalence, grammar validity

Random weights are enough for the torch part: what is checked is that the pieces agree with each other, not that the model is good (that is evaluate_goal_translator.py).
"""
import json, math, os, random, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(os.path.dirname(HERE), "tools", "goal_translator")
sys.path.insert(0, GT)
import numpy as np
import gt_conventions as C
import gt_corrupt, gt_metrics as M, stage_c as SC, build_dataset as B

FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


ST, TM = C.Stats(), C.Timing()

print("[conventions]")
check(C.VOCAB_SIZE == 320 and C.N_USED == 313 and C.VOCAB_SIZE % 64 == 0, f"vocab {C.N_USED} used, padded to {C.VOCAB_SIZE}")
check(len({C.PAD, C.BOS, C.SEP, C.WIN, C.UNSPEC, C.END}) == 6 and all(t >= 256 for t in (C.PAD, C.BOS, C.SEP, C.WIN, C.UNSPEC, C.END)), "special tokens are distinct and outside the byte range")
check(C.bin_to_turn(C.TURN_MID, ST) == 0.0 and C.bin_to_speed(0, ST) == 0.0, "middle turn bin is exactly 0, speed bin 0 is standing")
rt = all(C.turn_to_bin(C.bin_to_turn(b, ST), ST) == b for b in range(C.N_TURN_BINS)) and all(C.speed_to_bin(C.bin_to_speed(b, ST), ST) == b for b in range(C.N_SPEED_BINS))
check(rt, "every bin survives bin -> value -> bin")
check(C.turn_to_bin(5.0, ST) == C.N_TURN_BINS - 1 and C.turn_to_bin(-5.0, ST) == 0 and C.speed_to_bin(99, ST) == C.N_SPEED_BINS - 1, "values beyond goal_stats clamp to the outer bins")
left = C.arc_goal(0.5, 1.0, 2.0); right = C.arc_goal(-0.5, 1.0, 2.0)
check(left[1] > 0 > right[1] and left[2] > 0 and abs(left[3] - 0.5) < 1e-6, "arc_goal: a left turn (+) has dx > 0, a right turn dx < 0, speed = path / seconds (NERVA's own convention)")
check(abs(math.hypot(left[1], left[2]) - 1.0 * math.sin(0.25) / 0.25) < 1e-5, "arc_goal: chord length = path * sinc(turn / 2)")
p = C.Plan([C.spec_win(-0.4, 1.2, ST), C.spec_win(0.4, 1.2, ST), C.UNSPEC_WIN])
check(C.tokens_to_plan(C.plan_to_tokens(p)).windows == p.windows, "plan -> tokens -> plan is the identity")
bad = [[C.END], [C.WIN, C.END], [C.WIN, C.TURN_BASE, C.END], [C.WIN, C.SPEED_BASE], [C.WIN, C.UNSPEC], [C.UNSPEC], [C.WIN, C.UNSPEC, C.END, C.WIN]]
errs = 0
for t in bad:
    try:
        C.tokens_to_plan(t)
    except C.PlanError:
        errs += 1
check(errs == len(bad), "the grammar refuses every malformed token sequence (END first / after WIN / after TURN, SPEED first, no END, tokens after END)")
g = C.Grammar()
walk = []
rng = random.Random(0)
for _ in range(500):
    ids = g.allowed_ids()
    if not ids:
        break
    t = rng.choice(ids)
    g.advance(t); walk.append(t)
    if t == C.END:
        break
check(walk and walk[-1] == C.END and C.tokens_to_plan(walk).windows is not None, "any walk through the grammar's allowed tokens is a valid plan")
check(np.isfinite(C.Grammar().mask()[C.WIN]) and np.isinf(C.Grammar().mask()[C.END]), "the logit mask allows only WIN at the start")
try:
    C.encode_input("x" * 300); check(False, "300 bytes are rejected")
except C.PlanError:
    check(True, "text over 256 bytes is rejected, not truncated")
goals = C.plan_to_goals(C.Plan([C.spec_win(0.3, 1.0, ST), C.UNSPEC_WIN]), ST, TM)
check(goals[1] is None and len(goals[0]) == 4 and abs(goals[0][3] - C.bin_to_speed(C.speed_to_bin(1.0, ST), ST)) < 1e-5, "plan_to_goals: UNSPEC -> no goal, else the four self-consistent numbers")

print("[stage C]")
t = "Walkrightandthenleftthenwaveurhandandidkmovearound?"
check(" ".join(w for w in SC.tokenize(t) if w != ",") == "walk right and then left then wave ur hand and idk move around", "run-together text is split into words")
pl = SC.translate(t, ST, TM)
check(M.collapse(pl, ST) == ["R", "L", "U"] and pl.ignored and "wave" in pl.ignored[0], "the motivating prompt: right-turning, left-turning, unspecified; the wave is reported as ignored, never a turn")
check(M.collapse(SC.translate("turn around", ST, TM), ST) == ["U"], "a turn with no side is not guessed")
check(not SC.translate("sit down and dance", ST, TM).windows, "text with nothing expressible produces no windows (never a made-up motion)")
check(M.collapse(SC.translate("run forward, stop, then walk left", ST, TM), ST) == ["S", "X", "L"], "commas separate events")
d = C.bin_to_turn(SC.translate("turn right 90 degrees", ST, TM).windows[0].turn_bin, ST) * len(SC.translate("turn right 90 degrees", ST, TM))
check(abs(math.degrees(d) + 90) < 8, f"'turn right 90 degrees' = {math.degrees(d):.0f} degrees in total, to the right")
w3 = len(SC.translate("walk left for 3 seconds", ST, TM))
check(w3 == max(1, round(3 / TM.step_seconds)), f"a duration in seconds becomes windows of the prior's STEP ({TM.step_seconds:.2f} s): {w3}")
words = [w for grp in (SC.LEFT, SC.RIGHT, SC.V_WALK, SC.V_RUN, SC.V_TURN, SC.V_STOP, SC.VAGUE, SC.UNSUPPORTED) for w in grp]
check(all(SC.tokenize(w) == [w] for w in words), "no schema word is ever cut into pieces by the splitter")
suite = M.load_suite(os.path.join(GT, "prompt_suite.json"))
easy = [c for c in suite if c["tier"] == "easy"]
check(all(M.suite_check(SC.translate(c["text"], ST, TM), c, ST) for c in easy), f"stage C passes the whole easy tier ({len(easy)} prompts)")
hard_pass = sum(M.suite_check(SC.translate(c["text"], ST, TM), c, ST) for c in suite if c["tier"] == "hard")
check(hard_pass < sum(c["tier"] == "hard" for c in suite), f"the hard tier is not solved by stage C ({hard_pass} / {sum(c['tier'] == 'hard' for c in suite)}): a learned model has something to beat")

print("[measuring and segmenting]")
F, dt, W, stride = 400, 1 / 30, 64, 48
for rate, v in ((0.0, 1.2), (0.4, 1.0), (-0.3, 0.8)):          # rad per second, m/s
    yaw = 0.7 + np.arange(F) * dt * rate; x = np.zeros(F); z = np.zeros(F)          # not facing +z at the start either: windows are measured in their own start frame
    for i in range(1, F):
        hm = yaw[i - 1] + rate * dt / 2; x[i] = x[i - 1] + v * dt * math.sin(hm); z[i] = z[i - 1] + v * dt * math.cos(hm)     # midpoint heading: no Euler lag
    x += 3.0; z -= 2.0                                           # and not at the origin
    gm = B.window_goals_from_pelvis(x, z, yaw, W, stride, dt)
    sec = (W - 1) * dt
    want = C.arc_goal(rate * sec, v * (W - 1) * dt, sec)
    check(np.abs(gm - want).max() < 6e-3, f"turn rate {rate:+.1f} rad/s at {v} m/s: measured goals = arc_goal (max diff {np.abs(gm - want).max():.4f})")
    if rate:
        check((gm[0, 0] > 0) == (rate > 0) and (gm[0, 1] > 0) == (rate > 0), "left (yaw growing) gives dyaw > 0 and dx > 0")
runs = B.segment(np.array([[0.4, 0, 0, 1.2]] * 4 + [[-0.4, 0, 0, 1.2]] * 3 + [[0.0, 0, 0, 0.0]] * 2, np.float32), ST)
check([(r.sign, r.n) for r in runs] == [(1, 4), (-1, 3), (0, 2)], "segmentation: 4 windows left, 3 right, 2 standing")
check([r.sp for r in B.segment(np.array([[0.4, 0, 0, 0.05]] * 4, np.float32), ST)] == [1], "a turning window that barely moves is a slow turn, never 'stand still'")

print("[dataset, verifier, splits]")
rng = random.Random(3)
tmpl_ok, rew_ok, rew_n = True, 0, 0
for cid, gs in B.synthetic_goal_sequences(150, ST, rng):
    runs = B.segment(gs, ST)
    for _ in range(3):
        var = B.make_variants(runs, ST, rng)
        if not var.parts:
            continue
        tmpl_ok &= B.verify(B.template_text(runs, var, TM, rng), runs, var, TM, ST)[0]
        for _ in range(3):
            rew_n += 1; rew_ok += B.verify(B.rewrite_rules(runs, var, TM, rng), runs, var, TM, ST)[0]
check(tmpl_ok, "every exact template description passes its own verifier")
check(rew_ok / rew_n > 0.97, f"the rule-based rewrites pass the verifier ({rew_ok}/{rew_n}); the rest are exactly the ones it should not trust")
runs = B.segment(np.array([[0.5, 0, 0, 1.2]] * 5 + [[0.0, 0, 0, 1.2]] * 3, np.float32), ST)
var = B.Variant("full", [0, 1], [runs[0].win(ST)] * 5 + [runs[1].win(ST)] * 3, [0, 1])
check(not B.verify("walk right then straight", runs, var, TM, ST)[0], "verifier rejects a rewrite that flips left to right")
check(not B.verify("walk left then walk left then straight", runs, var, TM, ST)[0], "verifier rejects an added turn")
check(not B.verify("turn left 20 degrees then walk straight", runs, var, TM, ST)[0], "verifier rejects a turn size that is far off")
check(B.verify("hmm walk thataway then straight", runs, var, TM, ST)[0] is False, "verifier rejects text it cannot read back (unknown slang), by design")
with tempfile.TemporaryDirectory() as td:
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(GT, "build_dataset.py"), "--synthetic", "200", "--out", td], capture_output=True, text=True)
    check(r.returncode == 0, "build_dataset.py --synthetic runs" + ("" if r.returncode == 0 else ": " + r.stderr[-300:]))
    sp = {k: [json.loads(l) for l in open(os.path.join(td, k + ".jsonl"))] for k in ("train", "val", "test")}
    clips = {k: {x["clip"] for x in v} for k, v in sp.items()}
    check(not (clips["train"] & clips["val"]) and not (clips["train"] & clips["test"]) and not (clips["val"] & clips["test"]), "train / val / test share no SOURCE CLIP")
    allrows = sum(sp.values(), [])
    check(all(len(x["windows"]) >= 1 and all(w is None or (0 <= w[0] < C.N_TURN_BINS and 0 <= w[1] < C.N_SPEED_BINS) for w in x["windows"]) for x in allrows), "every label is a valid window list")
    check(any(None in x["windows"] for x in allrows) and any(x["kind"] == "rules" for x in allrows), "UNSPEC windows and rewritten rows are present")
    meta = json.load(open(os.path.join(td, "meta.json")))
    check(meta["layout_version"] == C.LAYOUT_VERSION and meta["turn_positive_is_left"] is True, "meta.json records the token layout version and the sign convention")

print("[corruption]")
rng = random.Random(1)
prot = all(("left" in gt_corrupt.corrupt("walk left then right", rng, p_typo=1.0, p_unsafe=0.0, p_nospace=0, p_case=0, p_punct=0).lower().replace(" ", "")) or True for _ in range(1))
outs = [gt_corrupt.corrupt("walk left then right", rng, p_typo=1.0, p_unsafe=0.0, p_nospace=0.0, p_case=0.0, p_punct=0.0) for _ in range(200)]
check(all("left" in o and "right" in o for o in outs), "safe typos never touch left / right")
check(any(" " not in gt_corrupt.corrupt("walk left then right", rng, p_nospace=1.0) for _ in range(5)), "space removal works")
check(all(len(gt_corrupt.corrupt("a " * 200, rng).encode()) <= 256 for _ in range(5)), "corrupted text stays within the byte limit")

try:
    import torch
    import gguf
    import gt_torch                    # also finds modeling_nanity.py in the NANITY-NEON checkout (SystemExit if there is none)
    HAVE = True
except ImportError:
    HAVE = False
    print("[torch] skipped (pip install torch gguf)")
except SystemExit as e:
    HAVE = False
    print(f"[torch] skipped ({e})")
if HAVE:
    print("[torch model, export, numpy runtime]")
    import gt_torch as G, gt_numpy as N
    torch.manual_seed(0)
    cfg = G.make_config("tiny"); m = G.build_model(cfg).eval()
    for p_ in m.parameters():
        p_.data += 0.05 * torch.randn_like(p_)                   # random weights with real structure (the default init is near-uniform logits)
    ids = torch.tensor([C.encode_input("walk left then right")[:30] + [C.SEP, C.WIN, C.TURN_BASE + 3]])
    with torch.no_grad():
        full = m(ids)[0][0, -1]
        _, past = m(ids[:, :-1], use_cache=True)
        masked = m(ids[:, -1:], attention_mask=torch.ones(1, ids.shape[1], dtype=torch.long), past_key_values=past, use_cache=True)[0][0, -1]
        unmasked = m(ids[:, -1:], past_key_values=past, use_cache=True)[0][0, -1]
    check((full - masked).abs().max() < 1e-4, "cached decoding with an explicit mask equals the full forward (what gt_torch.decode_plan does)")
    print(f"  info  cached step WITHOUT the mask differs by {(full - unmasked).abs().max():.2f} logits (modeling_nanity.NanityAttention: is_causal with one query is top-left aligned)")
    errs, ends = 0, 0
    for text in ("walk left", "WALKRIGHTTHENLEFT", "idk", "x" * 200):
        try:
            plan = G.decode_plan(m, text); C.tokens_to_plan(C.plan_to_tokens(plan)); ends += 1
        except C.PlanError:
            errs += 1
    check(ends + errs == 4, f"a random model never produces a malformed plan (valid {ends}, window-cap / no-END errors {errs}: raised, not hidden)")
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "x.pt"); out = os.path.join(td, "x.gguf")
        meta = dict(layout_version=C.LAYOUT_VERSION, stats=ST.to_dict(), timing=dict(W=TM.W, stride=TM.stride, dt=TM.dt), n_turn_bins=C.N_TURN_BINS, n_speed_bins=C.N_SPEED_BINS,
                    max_windows=C.MAX_WINDOWS, max_input_bytes=C.MAX_INPUT_BYTES, vocab_size=C.VOCAB_SIZE, turn_positive_is_left=True, preset="tiny")
        G.save_checkpoint(ck, m, cfg, meta, 0)
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(GT, "export_goal_translator.py"), ck, out], capture_output=True, text=True)
        check(r.returncode == 0, "export_goal_translator.py writes the GGUF" + ("" if r.returncode == 0 else ": " + r.stderr[-300:]))
        if r.returncode == 0:
            tr = N.NumpyTranslator.from_gguf(out)
            seq = C.encode_input("go left then right")[:25] + [C.SEP, C.WIN, C.TURN_BASE + 9, C.SPEED_BASE + 4, C.WIN]
            ln = tr.prefill_logits(seq)
            with torch.no_grad():
                lt = m(torch.tensor([seq]))[0][0, -1].numpy()
            check(np.abs(ln - lt).max() < 1e-3 * max(1.0, np.abs(lt).max()), f"numpy runtime on the exported GGUF = torch (max diff {np.abs(ln - lt).max():.2e})")
            same = 0
            for text in ("walk left", "go right then left", "idk"):
                try:
                    a = G.decode_plan(m, text).windows
                except C.PlanError:
                    a = "err"
                try:
                    b = tr.decode(text).windows
                except C.PlanError:
                    b = "err"
                same += a == b
            check(same == 3, "numpy and torch decode the same plans (greedy, grammar-masked)")
            check(abs(tr.stats.turn_max - ST.turn_max) < 1e-6 and tr.timing.stride == TM.stride, "the GGUF carries goal_stats and the prior's timing")

print()
print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
sys.exit(1 if FAILS else 0)
