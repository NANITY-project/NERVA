#!/usr/bin/env python3
"""
tests/test_closed_loop.py - tools/closed_loop.py against the REAL engine (libnebula; built on first use) and the real model code.

  python tests/test_closed_loop.py                       # everything below; a few minutes on a CPU
  python tests/test_closed_loop.py --quick               # parts 1-3 only (no training)
  python tests/test_closed_loop.py --cc data/cc_motions.npz   # part 4 on the real ACCAD + 100STYLE set (default: data/cc_motions.npz if it exists)

  1 helpers      anchor_pending puts the unplayed tail on the body's pose, a stand clip is feasible and holds the pose, ranking puts feasible first, the token context has the trainer's layout
  2 loop         a stand-in generator that replays one engine-recorded straight walk, window by window, so the loop's own logic is tested without a trained model: the body ARRIVES at the
                 target; every window is logged; fallbacks: no feasible candidate -> retries -> stand clip -> "stuck"; a body that is "unstable" after the commit is RESTORED (clone / restore:
                 the walking never happened) ; a generator that makes no progress is detected as stuck
  3 cli          random-initialised CVQVAE + prior WITH style ids saved as checkpoints, tools/closed_loop.py run as a command: --style is honoured, closed_loop.npz is view_clip.py-format,
                 the log and summary are written
  4 end to end   tiny VAE + prior TRAINED with tools/train_cvqvae.py on windows cut from the motion set, 100STYLE ids included, then the closed loop with --style and --cfg-scale.
                 With data/cc_motions.npz (tools/get_cc_motions.py) the windows come from the real ACCAD / 100STYLE clips and the style ids are the real 100STYLE names.
                 WITHOUT it a stand-in set is made from engine walks with invented style names (StyleA / StyleB): that checks the plumbing only, it is NOT 100STYLE
The models of part 4 are far too small to move well: the test checks that the loop runs, logs and writes valid files with them, not that the motion is good.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); TOOLS = os.path.join(ROOT, "tools"); sys.path.insert(0, TOOLS)
import gen_dataset as g
import train_cvqvae as T
import closed_loop as CL

FAILS = []; DT = 1 / 30


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def quiet(*_a, **_k):
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ engine clips
def long_steps(st, n=26):
    """A straight walk of n steps (the engine's plan_steps, without the random length / turn)."""
    side = 0; cz = 0.0; steps = []
    for i in range(1, n + 1):
        cz += st["stride"] * (0.5 if i == 1 else 1.0); sgn = 1.0 if side == 0 else -1.0
        steps.append((side, sgn * st["step_width"], cz, 0.0)); side = 1 - side
    sgn = 1.0 if side == 0 else -1.0; steps.append((side, sgn * st["step_width"], cz, 0.0)); return steps


def straight_walk(lib, n=24):
    """A long straight walk recorded from the engine: footsteps are fed to the body as its queue drains (it only holds a few at a time). -> actions [T,30] at 30 fps from the standing pose."""
    st = dict(step_time=1.5, stride=0.4, step_width=0.05, turn_deg=0.0, arm_swing=0.4, twist=0.15, bob=0.98, torso_drop=0.0, torso_lean=0.0, torso_side=0.0, look_down_deg=0.0, look_yaw_deg=0.0)
    b = g.new_standing(lib)
    try:
        b.param("step_time_scale", st["step_time"]); b.param("arm_swing_gain", st["arm_swing"]); b.param("twist_gain", st["twist"]); b.param("leg_len_frac", st["bob"]); b.param("hand_idle_m", 0.0)
        todo = long_steps(st, n); frames = []; settle = 0
        for _ in range(15):
            b.update(1 / 60)
        for tick in range(60 * 90):
            while todo and b.state()[34] < 3:
                side, x, z, yaw = todo[0]; ok, msg = b.cmd(f"FOOT {'left' if side == 0 else 'right'} {x:.4f} 0 {z:.4f} {yaw:.2f}")
                if not ok:
                    raise AssertionError("step refused: " + msg)
                todo.pop(0)
            b.update(1 / 60); s = b.state()
            assert s[10] < 0.5, "the recording body fell"
            if tick % 2 == 0:
                frames.append(g.action_from(b, s, st))
            if not todo and s[34] == 0 and s[15] == 0 and s[21] == 0:
                settle += 1
                if settle > 36:
                    break
            else:
                settle = 0
        return np.array(frames, np.float32)
    finally:
        b.close()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ stand-in generator
class FakePrior:
    """The parts of CodePrior the loop reads (a real CodePrior is built too, for build_ctx)."""
    def __init__(self, real, stats):
        self.__dict__.update(dict(windows=real.windows, NULL=real.NULL, ACT0=real.ACT0, SEP=real.SEP, goal_dim=real.goal_dim, n=real.n, K=real.K, meta=dict(goal_stats=stats, style_names=None)))


class FakeVAE:
    """decode_codes returns window number codes[:,0]: frames [m*stride, m*stride + 64) of a recorded walk, in the model's raw output format (norm mean 0 / std 1, flags as logits)."""
    window = 64

    def __init__(self, clip, stride, mode="walk", bad_after=None):
        self.clip, self.stride, self.mode, self.bad_after = clip, stride, mode, bad_after

    def decode_codes(self, codes, acts):
        out = []
        for c in codes[:, 0].tolist():
            off = int(c) * self.stride; w = self.clip[off:off + 64].copy()
            if len(w) < 64:
                w = np.concatenate([w, np.repeat(w[-1:], 64 - len(w), 0)])
            if self.mode == "stand":
                w = CL.stand_clip(STAND_STATE, 64)
            if self.mode == "garbage":
                w = w + np.random.default_rng(c).normal(0, 0.5, w.shape).astype(np.float32)
            y = w.astype(np.float32).copy(); y[:, T.FLAGS] = np.where(w[:, T.FLAGS] > 0.5, 1.0, -1.0); out.append(y)
        return torch.from_numpy(np.stack(out))


STAND_STATE = None


def fake_sampler(stride, n):
    def sample_window(prior, ctx, goal, style, temperature, top_k, top_p, cfg_scale=1.0):
        m = (ctx.shape[1] - 1) // (n + 1)                              # windows already in the context = this window's index in the walk
        codes = torch.full((ctx.shape[0], n), m, dtype=torch.long); return codes, torch.cat([ctx, codes], 1)
    return sample_window


def make_loop(lib, clip, goal, vae_mode="walk", k=4, windows=14, mode="arc", **kw):
    global STAND_STATE
    body = g.new_standing(lib); STAND_STATE = body.state().copy()
    real = T.CodePrior(K=16, n_actions=2, n=16, windows=12, d=32, layers=1, heads=2, goal_dim=4)
    stats = dict(mean=[0, 0, 0, 0], std=[1, 1, 1, 1], turn_max=0.7, speed_cruise=0.9, speed_max=1.4)
    norm = dict(mean=np.zeros(30), std=np.ones(30)); names = ["walk", "stand"]
    kw.setdefault("turn_deg", 0.0)
    loop = CL.ClosedLoop(body, FakeVAE(clip, 32, vae_mode), norm, FakePrior(real, stats), names, 32, DT, "walk", None, goal, mode, k, 0.9, 40, 1.0, 1.0, None, arrive=0.3, max_windows=windows, log=quiet, **kw)
    return body, loop


# ================================================================================================================================================================== 1 helpers
def part1(lib):
    print("[1 helpers]")
    clip = straight_walk(lib)
    p = np.zeros((20, 30)); p[:, 0] = np.linspace(1, 2, 20); p[:, 2] = np.linspace(0, 1, 20); p[:, 3] = 0.5 + np.linspace(0, 0.3, 20); p[:, 4] = p[:, 0] + 0.1; p[:, 6] = p[:, 2]; p[:, 7] = p[:, 3] + 0.1
    q = CL.anchor_pending(p.astype(np.float32), -3.0, 4.0, 2.0)
    check(abs(q[0, 0] + 3) < 1e-5 and abs(q[0, 2] - 4) < 1e-5 and abs(q[0, 3] - 2.0) < 1e-5, "anchor_pending: the first frame is at the body pose (x, z, yaw)")
    d_old = np.hypot(p[-1, 0] - p[0, 0], p[-1, 2] - p[0, 2]); d_new = np.hypot(q[-1, 0] - q[0, 0], q[-1, 2] - q[0, 2])
    foot_off = np.hypot(q[:, 4] - q[:, 0], q[:, 6] - q[:, 2]); check(abs(d_old - d_new) < 1e-5 and np.allclose(foot_off, 0.1, atol=1e-5) and abs((q[-1, 3] - q[0, 3]) - 0.3) < 1e-5,
                                                                      "anchor_pending is a rigid move: path length, foot offsets from the pelvis and the heading change are kept")
    w0 = T.canonicalize(np.stack([p, p])[:, :20].astype(np.float32)); w1 = T.canonicalize(q[None].astype(np.float32))
    check(np.abs(w0[0] - w1[0]).max() < 1e-4, "  and canonicalising the moved tail gives the same canonical frames as the original")

    w = np.zeros((20, 30)); w[:, 4] = 0.1; w[:, 8] = 1; w[8:14, 8] = 0; w[8:14, 4] = np.linspace(0.1, 0.5, 6); w[14:, 4] = 0.5; w[14:, 8] = 1; w[:, 13] = 1; w[:, 9] = -0.1
    stt = np.zeros(35); stt[11], stt[12], stt[13] = 0.107, 0.001, 0.004; stt[17] = -0.1; q = CL.anchor_feet(w.astype(np.float32), stt)
    check(np.allclose(q[:8, 4], 0.107, atol=1e-6) and abs(q[0, 6] - 0.004) < 1e-6 and np.allclose(q[14:, 4], 0.5) and np.allclose(q[:, 9], -0.1) and 0.107 > q[8, 4] - 0.1 - 0.0 and q[8, 4] < 0.5,
          "anchor_feet: the planted foot is moved onto the body's sole for its whole contact run, the shift fades out over the swing, the next landing and the other foot are untouched")
    b = g.new_standing(lib); s0 = b.state().copy(); c = CL.stand_clip(s0, 32); v = b.evaluate(c, DT)
    check(v["feasible"] and v["score"] < 1e-3, f"stand clip from the standing body is feasible (score {v['score']:.1e})")
    b.evaluate(c, DT, commit=True); s1 = b.state().copy()
    check(np.abs(s1[[0, 1, 2, 3]] - s0[[0, 1, 2, 3]]).max() < 5e-3 and s1[10] < 0.5, "  committing it keeps the pelvis pose and the body does not fall")
    snap = b.clone(); b.evaluate(clip[:40], DT, commit=True); moved = float(np.hypot(*(b.state()[[0, 2]] - s1[[0, 2]])))
    b.restore(snap); check(moved > 0.05 and np.abs(b.state() - s1).max() < 1e-4, f"clone / restore: committed walk moved the body {moved:.2f} m, restore puts the state back")
    snap.close(); b.close()

    wb = g.new_standing(lib); whole = wb.evaluate(clip, DT); wb.close()
    jb = g.new_standing(lib); j = CL.judge_window(jb, clip, len(clip), DT); jb.close()
    check(whole["feasible"] and j["feasible"] == whole["feasible"] and abs(j["score"] - whole["score"]) < 1e-6 and j["bad_frames"] == whole["bad_frames"],
          f"judge_window over a whole clip gives the engine's own verdict (score {j['score']:.5f} vs {whole['score']:.5f}, {j['bad_frames']} bad frames)")
    rng = np.random.default_rng(5); trunc_bad = look_bad = n_ch = nclip = 0
    for _ in range(80):                                            # engine clips (random gait styles) that are feasible as a whole, judged chunk by chunk
        stl = g.sample_style(rng); stl["turn_deg"] = 0.0; a, _info = g.record_clip(lib, rng, stl)
        if a is None or len(a) < 160:
            continue
        cb = g.new_standing(lib); ok = cb.evaluate(a, DT)["feasible"]; cb.close()
        if not ok:
            continue
        nclip += 1; tb = g.new_standing(lib); jb = g.new_standing(lib)
        for i in range(0, len(a) - 63, 32):
            n_ch += 1; trunc_bad += int(not tb.evaluate(a[i:i + 32], DT, commit=True)["feasible"])          # cut at `stride`: no landing look-ahead
            look_bad += int(not CL.judge_window(jb, a[i:i + 64], 32, DT)["feasible"]); jb.evaluate(a[i:i + 32], DT, commit=True)
        tb.close(); jb.close()
        if nclip >= 4:
            break
    check(nclip >= 2 and look_bad == 0 and trunc_bad > 0, f"look-ahead: {n_ch} chunks of {nclip} engine-feasible walks fail {trunc_bad}x when cut at stride, {look_bad}x when judged with the rest of the window as look-ahead")

    ok = CL.rank_key(dict(feasible=True, score=0.0004), 5.0, 1e-3) < CL.rank_key(dict(feasible=False, score=0.0), 0.0, 1e-3)
    ok &= CL.rank_key(dict(feasible=True, score=0.0004), 5.0, 1e-3) < CL.rank_key(dict(feasible=True, score=0.004), 0.0, 1e-3)
    ok &= CL.rank_key(dict(feasible=True, score=0.0001), 2.0, 1e-3) > CL.rank_key(dict(feasible=True, score=0.0004), 1.0, 1e-3)
    check(ok, "rank_key: feasible first, then the engine score, scores within score_tol tie and progress decides")

    pr = T.CodePrior(K=16, n_actions=3, n=16, windows=4, d=32, layers=1, heads=2, goal_dim=4)
    ctx = CL.build_ctx(pr, 2, [np.arange(16), np.arange(16) + 1], 3, "cpu")
    check(ctx.shape == (3, 1 + 2 * 17) and int(ctx[0, 0]) == pr.ACT0 + 2 and int(ctx[0, 17]) == pr.SEP and int(ctx[0, -1]) == pr.SEP and int(ctx[1, 18]) == 1, "build_ctx: [ACTION] codes SEP codes SEP, as generate_codes builds it")
    gc = T.goal_context(torch.arange(8.0).reshape(1, 4, 2).repeat(1, 1, 1), 2, 4); check(gc.shape == (1, 4, 2) and float(gc[0, 2, 0]) == 4.0 and float(gc[0, 3, 0]) == 0.0, "goal_context: slot of the window being sampled is the last used slot (the layout the loop fills)")


# ================================================================================================================================================================== 2 loop
def part2(lib):
    print("[2 loop, stand-in generator]")
    clip = straight_walk(lib); n = 16; print(f"        recorded straight walk: {len(clip)} frames, {clip[-1, 2]:.1f} m")
    old = CL.T.sample_window; CL.T.sample_window = fake_sampler(32, n)
    try:
        body, loop = make_loop(lib, clip, (0.0, 5.0), k=3, windows=16)
        out, rows, summ = loop.run()
        check(summ["status"] == "arrived" and summ["final_distance"] < 0.31 and summ["stand_clips"] == 0 and summ["faults"] == 0, f"straight walk to (0, 5): {summ['status']} after {summ['windows']} windows, {summ['final_distance']:.2f} m from the target")
        check(out.shape == (summ["windows"] * 32, 30) and np.isfinite(out).all(), f"the played clip has {len(out)} frames = windows x stride, all finite")
        smp = [r for r in rows if r["mode"] == "sample"]
        check(len(smp) == summ["windows"] and all(len(r["scores"]) == 3 and r["chosen"] in (0, 1, 2) and r["goal_raw"] and r["dist"] is not None and "remaining_turn" in r for r in smp),
              "the log has one row per window with the k scores, the chosen candidate and the remaining goal")
        check(smp[0]["dist"] > smp[-1]["dist"] and abs(smp[0]["remaining_turn"]) < 0.1, f"distance falls ({smp[0]['dist']:.2f} -> {smp[-1]['dist']:.2f} m) and the target is straight ahead (remaining turn {smp[0]['remaining_turn']:.3f} rad)")
        check(smp[0]["turn_asked"] == 0.0 or abs(smp[0]["turn_asked"]) < 0.2, "  so the goal asks for (almost) no turn")
        # the played clip, replayed from a standing body by the engine (what view_clip.py does), is the same motion
        chk = g.new_standing(lib); v = chk.evaluate(out, DT); check(v["feasible"] and abs(out[-1, 2] - 5.0) < 0.5, f"replaying the closed-loop frames from a standing body (as view_clip.py does): feasible={v['feasible']}, ends at z = {out[-1, 2]:.2f} m")
        chk.close(); body.close()

        body, loop = make_loop(lib, clip, (0.0, 5.0), vae_mode="garbage", k=3, windows=10, retries=1, max_faults=2)
        out, rows, summ = loop.run()
        check(summ["status"] == "stuck" and summ["stand_clips"] == 2 and any("no feasible candidate" in r.get("events", []) for r in rows),
              f"no feasible candidate: resampled ({sum(r['attempt'] > 0 for r in rows)} retries), stand clips ({summ['stand_clips']}), then status {summ['status']}")
        stand_rows = [r for r in rows if r["mode"] == "stand"]; check(len(stand_rows) == 2 and all(r["feasible"] for r in stand_rows), "  the stand fallbacks are feasible")
        body.close()

        body, loop = make_loop(lib, clip, (0.0, 5.0), k=3, windows=10, retries=1, max_faults=1, margin_min=10.0)       # every commit looks unstable
        s0 = body.state().copy(); out, rows, summ = loop.run(); s1 = body.state()
        check(any(r["mode"] == "restored" for r in rows) and summ["restores"] >= 2, f"unstable after the commit: {summ['restores']} restores with fresh seeds before the stand clip")
        check(float(np.hypot(*(s1[[0, 2]] - s0[[0, 2]]))) < 0.02 and summ["status"] == "stuck", f"  the restored walks never happened: the body is {np.hypot(*(s1[[0, 2]] - s0[[0, 2]])):.3f} m from its start")
        seeds = [r["seed"] for r in rows if r["window"] == 0 and r["mode"] != "stand"]; check(len(set(seeds)) == len(seeds) and len(seeds) >= 2, f"  every retry used a fresh seed ({len(seeds)} seeds)")
        body.close()

        body, loop = make_loop(lib, clip, (0.0, 5.0), vae_mode="stand", k=3, windows=30, stuck_windows=3, stuck_progress=0.15, max_faults=1)
        out, rows, summ = loop.run()
        check(summ["status"] == "stuck" and summ["stand_clips"] >= 1 and summ["windows"] < 30, f"a generator that never moves is detected as stuck after {summ['windows']} windows ({summ['stand_clips']} stand clip(s))")
        body.close()

        body, loop = make_loop(lib, clip, None, k=2, windows=4, mode="rate", turn_deg=25.0)
        out, rows, summ = loop.run()
        check(summ["status"] == "max_windows" and summ["windows"] == 4 and abs(rows[0]["turn_asked"] - 0.4363) < 1e-3, f"rate mode needs no target: {summ['windows']} windows, turn asked {rows[0]['turn_asked']:.3f} rad")
        body.close()
    finally:
        CL.T.sample_window = old


# ================================================================================================================================================================== 3 cli
def tiny_models(out, names, stride=32, styles=("none", "StyleA", "StyleB"), mean=None, std=0.02):
    """Random-initialised (untrained) VAE + prior checkpoints in tools/train_cvqvae.py's format, prior with goal conditioning and style ids."""
    torch.manual_seed(0)
    cfg = dict(n_actions=len(names), window=64, hidden=64, emb=16, code_dim=32, num_codes=32, downs=2, n_res=1, dropout=0.0, ema_decay=0.99, restart_thresh=0.5, cond_enc=True)
    vae = T.build_model(cfg)
    b = g.Body.__new__(g.Body)
    m = np.zeros(30) if mean is None else mean; sd = np.ones(30) * std; sd[T.FLAGS] = 1.0; m = m.copy(); m[T.FLAGS] = 0.0
    norm = dict(mean=m, std=sd, dstd=np.ones(30))
    torch.save(dict(model=vae.state_dict(), cfg=cfg, norm={k: v.tolist() for k, v in norm.items()}, action_names=names, dt=DT, args={}, step=0, epoch=0, best_val=0.0), os.path.join(out, "best.pt"))
    pcfg = dict(K=32, n_actions=len(names), n=16, windows=3, d=32, layers=1, heads=2, dropout=0.0, goal_dim=4, n_styles=len(styles))
    prior = T.CodePrior(**pcfg); stats = dict(mean=[0.0, 0.0, 0.5, 0.5], std=[0.3, 0.1, 0.4, 0.3], turn_max=0.6, speed_cruise=0.9, speed_max=1.4)
    torch.save(dict(model=prior.state_dict(), cfg=pcfg, names=names, stride=stride, dt=DT, val_nll=0.0, unigram_val=0.0, args={}, goal_stats=stats, goal_names=T.GOAL_NAMES, style_names=list(styles)), os.path.join(out, "prior_best.pt"))


def run_cli(run, extra, timeout=600):
    cmd = [sys.executable, os.path.join(TOOLS, "closed_loop.py"), "--run", run, "--no-build", "--lib", os.path.join(ROOT, "build", "libnebula.so"), "--device", "cpu"] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)


def check_outputs(run, style=None, label=""):
    npz = os.path.join(run, "closed_loop.npz"); ok = os.path.exists(npz)
    check(ok, f"{label}closed_loop.npz written")
    if not ok:
        return
    z = np.load(npz, allow_pickle=False); N = len(z["offsets"]) - 1
    check({"actions", "offsets", "dt", "action", "labels", "description", "style", "style_names"} <= set(z.files) and z["actions"].shape[1] == 30 and N == 1 and z["offsets"][-1] == len(z["actions"]) and np.isfinite(z["actions"]).all(),
          f"{label}  view_clip.py format: actions {z['actions'].shape}, offsets {z['offsets'].tolist()}, dt {float(z['dt']):.4f}")
    if style is not None:
        check("style_label" in z.files and str(z["style_label"][0]) == style[0] and int(z["style"][0, 0]) == style[1], f"{label}  the style is recorded: {z['style_label'][0] if 'style_label' in z.files else None} id {int(z['style'][0, 0])}")
    rows = [json.loads(l) for l in open(os.path.join(run, "closed_loop_log.jsonl"))]; summ = json.load(open(os.path.join(run, "closed_loop_summary.json")))
    check(len(rows) >= 1 and all("window" in r and "mode" in r for r in rows) and summ["status"] in ("arrived", "max_windows", "stuck"), f"{label}  log {len(rows)} rows, summary status {summ['status']}, {summ['windows']} windows")
    return rows, summ


def part3(lib):
    print("[3 cli, random-initialised models with style ids]")
    with tempfile.TemporaryDirectory() as run:
        b = g.new_standing(lib); st = b.state().copy(); b.close(); mean = CL.stand_clip(st, 1)[0].astype(np.float64)
        tiny_models(run, ["walk", "stand"], mean=mean)
        r = run_cli(run, ["--goal", "0", "3", "--windows", "4", "--k", "3", "--style", "StyleB", "--cfg-scale", "2", "--retries", "1", "--max-faults", "1", "--seed", "3"])
        check(r.returncode == 0, f"closed_loop.py exits 0 (--style StyleB --cfg-scale 2){'' if r.returncode == 0 else ': ' + (r.stderr or r.stdout)[-400:]}")
        check("data limits" in r.stdout and "turn_max" in r.stdout, "it prints the turn_max / speed limits of the prior before it starts")
        check_outputs(run, ("StyleB", 2), "StyleB: ")
        r = run_cli(run, ["--goal", "0", "3", "--style", "Zombie"]); check(r.returncode != 0 and "StyleA" in (r.stderr + r.stdout), "an unknown style is refused and the known ones are listed")
        r = run_cli(run, ["--goal", "0", "3", "--action", "dance"]); check(r.returncode != 0 and "walk" in (r.stderr + r.stdout), "an unknown action is refused and the known ones are listed")
        r = run_cli(run, ["--goal-mode", "rate", "--turn-deg", "90", "--windows", "2", "--k", "2", "--max-faults", "1", "--retries", "0"])
        check(r.returncode == 0 and "above turn_max" in r.stdout, "rate mode with 90 deg per window warns that turn_max (the data) does not contain it")
        r = run_cli(run, []); check(r.returncode != 0 and "--goal" in (r.stderr + r.stdout), "arc mode without --goal is refused")


# ================================================================================================================================================================== 4 end to end
def windows_from_clips(clips, acts, styles, W=64, stride=32):
    X, V, A_, S = [], [], [], []
    for c, a, s in zip(clips, acts, styles):
        for st in range(0, len(c) - W + 1, stride):
            X.append(c[st:st + W]); V.append(W); A_.append(a); S.append(s)
    return np.stack(X).astype(np.float32), np.array(V, np.int16), np.array(A_, np.int16), np.array(S, np.int16)


def load_motion_set(path, lib):
    """-> (clips, action name per clip, style name per clip, dt, source note). From the CC set if given, else engine walks with invented style names."""
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False); off = z["offsets"]; acts = z["actions"]; an = [str(x) for x in z["action"]]; sl = [str(x) for x in z["style_label"]] if "style_label" in z.files else [""] * (len(off) - 1)
        keep = [i for i in range(len(off) - 1) if off[i + 1] - off[i] >= 64 and an[i] in ("walk", "style_walk", "stand", "style_idle")][:400]
        return [acts[off[i]:off[i + 1]] for i in keep], [an[i] if an[i] in ("walk", "stand") else ("walk" if "walk" in an[i] else "stand") for i in keep], [sl[i] for i in keep], float(z["dt"]), f"real {path}: {len(keep)} clips"
    rng = np.random.default_rng(1); clips, acts, sts = [], [], []; tries = 0
    while len(clips) < 36 and tries < 200:
        tries += 1; st = g.sample_style(rng); st["stride"] = float(rng.uniform(0.3, 0.5)); a, info = g.record_clip(lib, rng, st)
        if a is not None and len(a) >= 100:
            clips.append(a); acts.append("walk"); sts.append(["StyleA", "StyleB", ""][len(clips) % 3])
    return clips, acts, sts, DT, f"stand-in: {len(clips)} engine walks with INVENTED style names (StyleA / StyleB), not 100STYLE"


def part4(lib, cc):
    print("[4 end to end: tiny trained models with style ids]")
    clips, acts, sts, dt, note = load_motion_set(cc, lib); print(f"        motion set: {note}")
    check(len(clips) >= 8, f"{len(clips)} clips to cut windows from")
    if len(clips) < 8:
        return
    names = sorted(set(acts)); snames = ["none"] + sorted({s for s in sts if s}); sid = {n: i for i, n in enumerate(snames)}
    idx = np.arange(len(clips)); val = set(idx[::6].tolist())
    with tempfile.TemporaryDirectory() as tmp:
        data, run = os.path.join(tmp, "vae"), os.path.join(tmp, "run"); os.makedirs(data); os.makedirs(run)
        for nm, sel in (("train", [i for i in idx if i not in val]), ("val", sorted(val))):
            X, V, A_, S = windows_from_clips([clips[i] for i in sel], [names.index(acts[i]) for i in sel], [sid[sts[i]] if sts[i] else 0 for i in sel])
            np.savez(os.path.join(data, f"{nm}_windows.npz"), windows=X, valid=V, action=A_, action_names=np.array(names), dt=np.float32(dt), style=S, style_names=np.array(snames))
            print(f"        {nm}: {len(X)} windows")
        json.dump(dict(channels=g.ACTION_NAMES), open(os.path.join(data, "stats.json"), "w"))
        cmd = [sys.executable, os.path.join(TOOLS, "train_cvqvae.py"), "--data", data, "--out", run, "--device", "cpu", "--amp", "none", "--hidden", "64", "--emb", "16", "--code-dim", "32", "--num-codes", "64", "--n-res", "1",
               "--epochs", "3", "--batch", "32", "--prior-dim", "64", "--prior-layers", "2", "--prior-heads", "2", "--prior-epochs", "4", "--prior-batch", "32", "--prior-windows", "3", "--no-generate", "--seed", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1500)
        check(r.returncode == 0 and os.path.exists(os.path.join(run, "prior_best.pt")), f"train_cvqvae.py trains the tiny VAE + prior (styles: {', '.join(snames[1:]) or 'none'}){'' if r.returncode == 0 else ': ' + (r.stderr or r.stdout)[-600:]}")
        if r.returncode != 0:
            return
        pk = torch.load(os.path.join(run, "prior_best.pt"), map_location="cpu", weights_only=False)
        check(pk["style_names"] == snames and pk["goal_stats"]["turn_max"] > 0, f"the prior stores style_names {pk['style_names']} and goal_stats (turn_max {pk['goal_stats']['turn_max']:.2f} rad, cruise {pk['goal_stats']['speed_cruise']:.2f} m/s)")
        for sty in snames[1:3]:
            r = run_cli(run, ["--goal", "1.0", "3.0", "--windows", "6", "--k", "4", "--style", sty, "--cfg-scale", "2", "--retries", "1", "--max-faults", "2", "--seed", "1", "--action", names[0]])
            check(r.returncode == 0, f"closed_loop.py with the trained models, --style {sty}{'' if r.returncode == 0 else ': ' + (r.stderr or r.stdout)[-500:]}")
            if r.returncode == 0:
                out = check_outputs(run, (sty, sid[sty]), f"{sty}: ")
                if out:
                    print(f"        {sty}: {out[1]['status']}, {out[1]['windows']} windows, {out[1]['stand_clips']} stand clips, {out[1]['restores']} restores, {out[1]['final_distance']:.2f} m from the target")
        r = run_cli(run, ["--goal-mode", "rate", "--turn-deg", "20", "--windows", "4", "--k", "3", "--retries", "1", "--max-faults", "2"])
        check(r.returncode == 0, "rate mode with the trained models" + ("" if r.returncode == 0 else ": " + (r.stderr or r.stdout)[-400:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--cc", default=os.path.join(ROOT, "data", "cc_motions.npz")); a = ap.parse_args()
    lib = g.Lib(None, ROOT, build=True)
    part1(lib); part2(lib); part3(lib)
    if not a.quick:
        part4(lib, a.cc)
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED"); sys.exit(1)
    print("all closed-loop checks passed")


if __name__ == "__main__":
    main()
