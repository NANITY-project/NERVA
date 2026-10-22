#!/usr/bin/env python3
"""
vae_eval.py - is what the motion model produces any good? Judged against REAL unseen motion, not against itself.

  python tools/vae_eval.py --samples runs/cvqvae/recon_preview.npz --desc reconstruction --data data/vae
  python tools/vae_eval.py --samples samples.npz --data data/vae --engine           # + the engine's verdict on samples that start at rest
  python tools/vae_eval.py --samples samples.npz --data data/vae --json out.json    # machine-readable

Samples can be (a) a clip file (actions / offsets / dt / action: what train_cvqvae.py --export-recon writes, or any tools/*.npz) or (b) a window file (windows [N,T,30],
action = index into action_names, optional valid, dt). Every sample is moved to the canonical frame (frame 0: pelvis at the origin, facing +z: the same convention as
train_cvqvae.py) before any distance is measured. `--data` is the folder tools/build_vae_dataset.py wrote (train_windows.npz, val_windows.npz).

What it measures, per action, for the samples AND for the real validation windows (performers the model never saw), side by side - the real column is the yardstick,
because a 2 s window is short and even real windows do not always pass a gait grader:
  pass %         the CMU grader: the jitter / planted-foot gates pass AND the fit to the labelled action is >= 60
  recognised %   the grader's classifier puts the window in the labelled action's family
  nearest train  median distance (m, RMS over 18 position channels x 48 points, see motion_vary.clip_rep) from a sample to its NEAREST training window. The real
                 validation windows give the scale: how far unseen real motion is from the training set.
  replay %       share of samples nearer to a training window than the 10% closest real validation windows are. About 10% is normal; a lot more means the model is
                 handing back training motion (memorising), which is the thing the muscle memory / variants are meant to avoid.
  diversity      mean distance between two random samples of the action / the same for real validation windows. Far below 1 = collapsed onto a few motions.
  engine         (--engine) the engine judges a clip from a STANDING START, so only samples that start at rest can be judged (mid-stride windows are infeasible by
                 construction, I measured 0 of 35); the tool reports how many qualify and how many of those the engine accepts, with the same number for real windows.
Verdict lines at the end say which of the three failure modes (worse than real, replaying, collapsed) apply. They are rules of thumb, not proofs.
Honest limits: this judges KINEMATICS, not whether the motion is what someone asked for. The grader only knows the 24 CMU actions (plus what you promote in the memory),
so a model that makes valid motion of a different action than labelled shows up as a low recognised % but not as a quality failure. Reconstructions of validation windows
test the autoencoder; they say nothing about generation from a prior (which train_cvqvae.py does not have yet): evaluate the prior's samples the same way.
"""
import argparse, collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import motion_features as mf, motion_grader as G, motion_vary as V

XZ = [(0, 2), (4, 6), (9, 11), (14, 16), (18, 20), (22, 24), (26, 28)]; YAWS = [3, 7, 12]
PASS_FIT, COPY_PCT, MIN_PAIRS = 60.0, 10.0, 6


def canonicalize(W):
    """[N,T,30] -> same shape; frame 0 of every window: pelvis at x = z = 0 facing +z (identical to train_cvqvae.canonicalize)."""
    W = W.astype(np.float64).copy()
    for y in YAWS:
        W[:, :, y] = np.unwrap(W[:, :, y], axis=1)
    y0 = W[:, 0, 3].copy(); px, pz = W[:, 0, 0].copy(), W[:, 0, 2].copy(); c, s = np.cos(y0)[:, None], np.sin(y0)[:, None]
    for xi, zi in XZ:
        dx, dz = W[:, :, xi] - px[:, None], W[:, :, zi] - pz[:, None]; W[:, :, xi] = dx * c - dz * s; W[:, :, zi] = dx * s + dz * c
    for y in YAWS:
        off = W[:, 0, y] - y0; off_w = (off + np.pi) % (2 * np.pi) - np.pi; W[:, :, y] = W[:, :, y] - y0[:, None] - (off - off_w)[:, None]
    return W.astype(np.float32)


def canon_clip(c):
    return canonicalize(np.asarray(c, np.float32)[None])[0]


def load_samples(path, desc=None):
    z = np.load(path, allow_pickle=True); dt = float(z["dt"])
    if "windows" in z.files:
        X = z["windows"]; valid = z["valid"] if "valid" in z.files else np.full(len(X), X.shape[1]); names = [str(n) for n in z["action_names"]]
        clips = [X[i, :int(valid[i])] for i in range(len(X))]; labels = [names[int(a)] if int(a) < len(names) else "?" for a in z["action"]]; d = None
    else:
        clips = G.split_clips(z); labels = [str(x) for x in z["action"]] if "action" in z.files else ["?"] * len(clips); d = [str(x) for x in z["description"]] if "description" in z.files else None
    if desc:
        if d is None:
            sys.exit("--desc needs a `description` array in the samples file")
        keep = [i for i, x in enumerate(d) if desc in x]; clips = [clips[i] for i in keep]; labels = [labels[i] for i in keep]
    return clips, labels, dt


def load_ref(folder, split):
    p = os.path.join(folder, f"{split}_windows.npz")
    if not os.path.exists(p):
        sys.exit(f"{p} not found (--data is the folder build_vae_dataset.py wrote)")
    z = np.load(p, allow_pickle=True); names = [str(n) for n in z["action_names"]]; X, v = z["windows"], z["valid"]
    return [X[i, :int(v[i])] for i in range(len(X))], [names[int(a)] for a in z["action"]], float(z["dt"])


def reps_of(clips):
    return np.stack([V.clip_rep(canon_clip(c)) for c in clips]) if clips else np.zeros((0, V.REP_FRAMES * len(V.POS)))


def nn_dist(S, T, chunk=256):
    """RMS distance from each row of S to the nearest row of T (m per channel and point)."""
    T = T.astype(np.float64); tt = (T ** 2).sum(1); out = []
    for i in range(0, len(S), chunk):
        s = S[i:i + chunk].astype(np.float64); d2 = (s ** 2).sum(1)[:, None] + tt[None] - 2 * s @ T.T; out.append(np.sqrt(np.maximum(d2.min(1), 0) / S.shape[1]))
    return np.concatenate(out) if out else np.zeros(0)


def pair_dist(R, rng, n=300):
    if len(R) < 2:
        return np.nan
    i = rng.integers(0, len(R), n); j = rng.integers(0, len(R), n); k = i != j
    return float(np.sqrt(((R[i[k]] - R[j[k]]) ** 2).mean(1)).mean()) if k.any() else np.nan


def standing_start(c, dt):
    """Starts at rest: pelvis moves < 0.15 m/s over the first 0.2 s and both feet are flagged planted."""
    n = max(2, min(len(c), int(round(0.2 / dt)) + 1)); sp = np.hypot(c[n - 1, 0] - c[0, 0], c[n - 1, 2] - c[0, 2]) / ((n - 1) * dt)
    return bool(sp < 0.15 and c[0, 8] > 0.5 and c[0, 13] > 0.5)


def grade_all(clips, labels, dt, prof):
    rows = []
    for c, lab in zip(clips, labels):
        v = mf.extract(c, dt)
        if np.isfinite(v).sum() < 0.6 * len(v):
            rows.append(dict(ok=False, rec=False, fit=None, gate=None)); continue
        act = lab if lab in prof.p else V.closest_action(v, prof)[0]
        r = G.grade_features(v, act, prof, use_match=False); top = prof.recognise(v)[0][0]
        rows.append(dict(ok=bool(r["fit"] is not None and r["gate"] >= 0.99 and r["fit"] >= PASS_FIT), rec=bool(top in G.family(act)), fit=r["fit"], gate=r["gate"]))
    return rows


def pct(x):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:4.0f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True); ap.add_argument("--data", default="data/vae"); ap.add_argument("--desc", help="keep only samples whose description contains this (e.g. reconstruction)")
    ap.add_argument("--real", default=G.REAL_DEFAULT); ap.add_argument("--engine", action="store_true"); ap.add_argument("--max-per-action", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    zr, RF = G.load_real(a.real); prof = G.Profiles(RF, zr["action"])
    sc, sl, dt = load_samples(a.samples, a.desc); vc, vl, dtv = load_ref(a.data, "val"); tc, tl, dtt = load_ref(a.data, "train")
    assert abs(dt - dtv) < 1e-4 and abs(dtt - dtv) < 1e-6, "samples, train and val windows must share one dt"
    print(f"{len(sc)} samples, {len(vc)} real validation windows (unseen performers), {len(tc)} training windows")
    if not sc:
        sys.exit("no samples")
    if not vc or not tc:
        sys.exit("the data folder has no validation (or training) windows: rebuild with build_vae_dataset.py --hours larger / --val larger")
    TR = reps_of(tc); VR = reps_of(vc); SR = reps_of(sc)
    ref_nn = nn_dist(VR, TR); smp_nn = nn_dist(SR, TR); copy_thr = float(np.percentile(ref_nn, COPY_PCT))
    sg, vg = grade_all(sc, sl, dt, prof), grade_all(vc, vl, dt, prof)
    chk = None
    if a.engine:
        try:
            import gen_dataset as gd; chk = gd.new_standing(gd.Lib())
        except Exception as ex:
            print(f"engine not available ({str(ex)[:100]}): skipping the engine column")
    acts = sorted(set(sl)); res = {}; sel = lambda labs, A: [i for i, x in enumerate(labs) if x == A]
    print(f"\n{'action':12s} {'n':>4s} | {'pass%':>5s} {'(real)':>6s} | {'recog%':>6s} {'(real)':>6s} | {'nearest train m':>15s} {'(real)':>7s} | {'replay%':>7s} | {'diversity':>9s}" + (" | engine ok / starts-at-rest (real)" if chk else ""))
    tot = collections.Counter()
    for A in acts:
        si = sel(sl, A)[:a.max_per_action]; vi = sel(vl, A)
        if not si:
            continue
        p_s = np.mean([sg[i]["ok"] for i in si]); r_s = np.mean([sg[i]["rec"] for i in si]); nn_s = float(np.median(smp_nn[si])); cp = float(np.mean(smp_nn[si] < copy_thr))
        div_s = pair_dist(SR[si], rng) if len(si) >= MIN_PAIRS else np.nan
        if len(vi) >= 1:
            p_v = np.mean([vg[i]["ok"] for i in vi]); r_v = np.mean([vg[i]["rec"] for i in vi]); nn_v = float(np.median(ref_nn[vi]))
        else:
            p_v = r_v = nn_v = None
        div_v = pair_dist(VR[vi], rng) if len(vi) >= MIN_PAIRS else np.nan; ratio = (div_s / div_v) if (div_v and not np.isnan(div_v) and not np.isnan(div_s)) else np.nan
        eng = ""
        if chk is not None:
            qs = [i for i in si if standing_start(sc[i], dt)]; qv = [i for i in vi if standing_start(vc[i], dt)]
            fs = sum(bool(chk.evaluate(sc[i], dt)["feasible"]) for i in qs); fv = sum(bool(chk.evaluate(vc[i], dt)["feasible"]) for i in qv)
            eng = f" | {fs}/{len(qs)} ({fv}/{len(qv)})"
        print(f"{A:12s} {len(si):4d} | {pct(p_s)} {pct(p_v):>6s} | {pct(r_s):>6s} {pct(r_v):>6s} | {nn_s:15.3f} {('%.3f' % nn_v) if nn_v is not None else '  n/a':>7s} | {pct(cp):>7s} | "
              f"{('%.2f' % ratio) if not np.isnan(ratio) else '  n/a':>9s}{eng}")
        res[A] = dict(n=len(si), pass_samples=float(p_s), pass_real=None if p_v is None else float(p_v), recog_samples=float(r_s), recog_real=None if r_v is None else float(r_v), nearest_train_samples=nn_s,
                      nearest_train_real=nn_v, replay=cp, diversity_ratio=None if np.isnan(ratio) else float(ratio))
        tot["n"] += len(si); tot["pass"] += p_s * len(si); tot["pv"] += (p_v or 0) * len(si); tot["copy"] += cp * len(si)
    n = max(1, tot["n"]); allp, allv, allc = tot["pass"] / n, tot["pv"] / n, tot["copy"] / n
    print(f"\nall actions (sample-weighted): pass {100 * allp:.0f}% (real validation windows {100 * allv:.0f}%), replay {100 * allc:.0f}% (about {COPY_PCT:.0f}% is normal), "
          f"nearest-training distance median {np.median(smp_nn):.3f} m (real unseen {np.median(ref_nn):.3f} m)")
    print("verdict:")
    print("  quality   " + ("OK: as often valid as real unseen windows" if allp >= allv - 0.10 else f"WORSE than real unseen motion by {100 * (allv - allp):.0f} points: look at the failing actions' gates / fit (motion_grader.py --verbose on the samples)"))
    print("  replay    " + ("OK" if allc <= 0.25 else f"REPLAYING TRAINING DATA: {100 * allc:.0f}% of samples are closer to a training window than the closest {COPY_PCT:.0f}% of real unseen windows (reconstructions of training windows look like this; samples from a prior should not)"))
    ratios = [r["diversity_ratio"] for r in res.values() if r["diversity_ratio"] is not None]
    print("  diversity " + ("not measurable (few samples per action)" if not ratios else ("OK" if np.median(ratios) >= 0.6 else f"LOW: samples of an action are {np.median(ratios):.2f}x as different from each other as real windows are (collapsed onto few motions)")))
    if a.json:
        json.dump(dict(per_action=res, pass_all=allp, pass_real=allv, replay_all=allc, copy_threshold_m=copy_thr), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
