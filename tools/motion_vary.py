#!/usr/bin/env python3
"""
motion_vary.py - make a NEW clip out of a stored one, so a stored motion is never played back the same way twice.

  python tools/motion_vary.py data/clips.npz --clip 3 --n 8 --out data/var.npz     # 8 variants of clip 3, each graded; writes the ones that pass
  python tools/motion_vary.py data/clips.npz --clip 3 --n 8 --action walk          # grade as this action (default: the closest known action)
  python tools/motion_vary.py data/clips.npz --clip 3 --n 8 --uncommon 0.4         # 40% of the variants get a strong, rarer twist
  python tools/view_clip.py data/var.npz --open                                     # watch them

What "variation" means here (and what it does not)
  The variation is SMOOTH and body-plausible. Random high-frequency jitter is deliberately NOT used: the grader's jitter gate (and the engine) reject it, and a VAE
  trained on it learns noise instead of motion. The "life" comes from these operations, each applied with its own probability and a random strength:
    mirror      left/right swapped (a person who is left-handed does the same thing the other way round)
    time warp   a different overall speed (+-20% typical) with a slow speed-up/slow-down along the clip
    amplitude   deeper/shallower body height changes, hands reaching a bit further or less, a more or less leaning torso
    gaze        the head direction turns and pitches a little, with a slow wander
    drift       slow (< 1 Hz) wander of hands, torso and pelvis height of 0.5-2 cm: the body never holds perfectly still
    tremor      1-4 Hz sway of 1-3 mm on hands/torso: the micro-motion a real body has (below the 6 Hz the jitter gate looks at)
  The STRENGTH of a variant is drawn from a heavy-tailed distribution: most variants are subtle, some are not. With probability --uncommon (default 0.12) a variant
  gets an "uncommon" multiplier (x1.8): a rarer style that is still checked by the same grader, so it only survives if it is still a body that could move so.
  Every variant is then CHECKED: the jitter/contact gates must pass, the fit to its action must stay >= max(60, source fit - 20), and it must differ from the source
  by a minimum feature distance (so it is not a copy). Variants that fail are dropped, not repaired.
Honest limits: a variant is the SAME motion performed differently, not a new motion. 100 variants of one clip add far less information than 100 different clips, and the
VAE only learns what the source clips contain. Mirror assumes the clip is canonical (left = +x of the body's heading at the start), as gen_dataset.py and bvh_to_clip.py write
them. Variants of real CMU clips are graded by the CMU-based grader, NOT by the engine (the engine's feasibility verdict rejects every real clip, see build_vae_dataset.py).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_features as mf, motion_grader as G

# action-vector layout (30): pelvis xyz yaw | footL xyz yaw contact | footR xyz yaw contact | handL xyz on | handR xyz on | look xyz on | torso xyz on
XZ = [(0, 2), (4, 6), (9, 11), (14, 16), (18, 20), (22, 24), (26, 28)]
YAWS, FLAGS = [3, 7, 12], [8, 13, 17, 21, 25, 29]
CONT = [i for i in range(30) if i not in FLAGS]
OPS = dict(mirror=0.5, warp=0.8, amplitude=0.8, gaze=0.7, drift=0.8, tremor=0.6)
STRENGTH_MEDIAN, STRENGTH_SIGMA, STRENGTH_RANGE, UNCOMMON_X = 0.6, 0.55, (0.1, 1.6), 1.8
MIN_FIT, FIT_DROP, MIN_DIST = 60.0, 20.0, 0.012   # MIN_DIST: metres of RMS difference between two clips (see clip_rep); mirroring a walk is ~0.05, a 3% speed change ~0.01
POS = [i for i in CONT if i not in YAWS]            # the 18 position channels
REP_FRAMES = 48


def smooth_noise(rng, n, dt, sigma_s, std=1.0, shape=()):
    """Gaussian-smoothed white noise (low-pass, ~0.19/sigma_s Hz corner), unit std x std. shape: extra independent channels -> array [n, *shape]."""
    k = max(1, int(round(sigma_s / dt))); t = np.arange(-4 * k, 4 * k + 1); ker = np.exp(-0.5 * (t / k) ** 2); ker /= np.sqrt((ker ** 2).sum())    # unit-variance output
    m = int(np.prod(shape)) if shape else 1; w = rng.normal(size=(n + 8 * k, m)); out = np.stack([np.convolve(w[:, j], ker, "valid")[:n] for j in range(m)], 1)
    return (out * std).reshape((n,) + tuple(shape)) if shape else (out[:, 0] * std)


def mirror(a):
    b = a.copy(); b[:, 4:9], b[:, 9:14] = a[:, 9:14], a[:, 4:9]; b[:, 14:18], b[:, 18:22] = a[:, 18:22], a[:, 14:18]
    for x, _ in XZ:
        b[:, x] *= -1
    for y in YAWS:
        b[:, y] *= -1
    return b


def warp(a, dt, rng, s):
    """Resample in time: global speed f (lognormal around 1) with a slow modulation; flags take the nearest source frame."""
    F = len(a); f = float(np.exp(rng.normal(0, 0.12 * s ** 0.8))); f = float(np.clip(f, 0.75, 1.3))
    mod = 1.0 + 0.10 * s * smooth_noise(rng, 4 * F, dt, 0.8); idx = [0.0]
    while idx[-1] < F - 1 and len(idx) < 4 * F:
        idx.append(idx[-1] + f * mod[len(idx)])
    idx = np.array(idx[:-1]); idx = idx[idx <= F - 1]
    if len(idx) < 12:
        return a
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, F - 1); w = (idx - i0)[:, None]; src = a.astype(np.float64).copy()
    for y in YAWS:
        src[:, y] = np.unwrap(src[:, y])
    out = src[i0] * (1 - w) + src[i1] * w; near = np.rint(idx).astype(int).clip(0, F - 1)
    out[:, FLAGS] = a[near][:, FLAGS]
    return out.astype(np.float32)


def amplitude(a, rng, s):
    b = a.astype(np.float64).copy(); pel = b[:, 0:3].copy(); u = lambda lo, hi: float(rng.uniform(lo, hi))
    med = np.median(pel[:, 1]); g = 1.0 + s * u(-0.08, 0.10); b[:, 1] = med + g * (pel[:, 1] - med)                    # bigger / smaller up-down of the body
    k = 1.0 + s * u(-0.08, 0.08)
    for h in ((14, 15, 16), (18, 19, 20)):                                                                              # hands reach a bit further / less, relative to the pelvis
        kk = k * (1.0 + 0.3 * s * u(-0.1, 0.1)); b[:, h] = b[:, [0, 1, 2]] + kk * (a[:, list(h)] - pel)
    kt = 1.0 + s * u(-0.25, 0.3); b[:, [26, 28]] = b[:, [0, 2]] + kt * (a[:, [26, 28]] - pel[:, [0, 2]])               # torso leans more / less (horizontal part only: trunk length stays)
    return b.astype(np.float32)


def gaze(a, dt, rng, s):
    b = a.astype(np.float64).copy(); n = len(a); neck = b[:, 26:29]; v = b[:, 22:25] - neck; L = np.linalg.norm(v, axis=1) + 1e-9
    yaw = np.arctan2(v[:, 0], v[:, 2]); el = np.arcsin(np.clip(v[:, 1] / L, -1, 1))
    dy = np.radians(s * (rng.uniform(-6, 6) + 3.0 * smooth_noise(rng, n, dt, 0.7))); dp = np.radians(s * (rng.uniform(-4, 4) + 2.0 * smooth_noise(rng, n, dt, 0.7)))
    yaw, el = yaw + dy, np.clip(el + dp, -1.2, 1.2)
    b[:, 22:25] = neck + L[:, None] * np.stack([np.cos(el) * np.sin(yaw), np.sin(el), np.cos(el) * np.cos(yaw)], 1)
    return b.astype(np.float32)


def drift(a, dt, rng, s):
    b = a.astype(np.float64).copy(); n = len(a); amp = 0.01 * s * rng.uniform(0.5, 1.5)
    for h in ((14, 15, 16), (18, 19, 20)):
        b[:, h] += smooth_noise(rng, n, dt, 0.35, amp, (3,))
    b[:, [26, 28]] += smooth_noise(rng, n, dt, 0.5, 0.5 * amp, (2,)); b[:, 1] += smooth_noise(rng, n, dt, 0.6, 0.3 * amp)
    return b.astype(np.float32)


def tremor(a, dt, rng, s):
    b = a.astype(np.float64).copy(); n = len(a); amp = 0.002 * s * rng.uniform(0.5, 1.5)
    band = lambda shape: smooth_noise(rng, n, dt, 0.06, 1.0, shape) - smooth_noise(rng, n, dt, 0.25, 1.0, shape)              # ~1-4 Hz band: below the 6 Hz the jitter gate measures
    for h in ((14, 15, 16), (18, 19, 20)):
        b[:, h] += amp * band((3,))
    b[:, [26, 27, 28]] += 0.5 * amp * band((3,))
    return b.astype(np.float32)


def draw_strength(rng, uncommon=0.12):
    """Heavy-tailed: mostly subtle; with probability `uncommon` an extra x1.8 for a rarer style. Returns (strength, was_uncommon)."""
    s = float(np.clip(np.exp(rng.normal(np.log(STRENGTH_MEDIAN), STRENGTH_SIGMA)), *STRENGTH_RANGE)); u = bool(rng.random() < uncommon)
    return (s * UNCOMMON_X if u else s), u


def make_variant(a, dt, rng, uncommon=0.12, ops=None):
    """One variant of clip a [F,30]. Returns (clip, info). The result is NOT checked: use check_variant."""
    a = np.asarray(a, np.float32); s, unc = draw_strength(rng, uncommon); done = []; b = a
    P = dict(OPS) if ops is None else {k: (1.0 if k in ops else 0.0) for k in OPS}
    for name in ("mirror", "warp", "amplitude", "gaze", "drift", "tremor"):
        if rng.random() >= P[name]:
            continue
        b = {"mirror": lambda: mirror(b), "warp": lambda: warp(b, dt, rng, s), "amplitude": lambda: amplitude(b, rng, s), "gaze": lambda: gaze(b, dt, rng, s),
             "drift": lambda: drift(b, dt, rng, s), "tremor": lambda: tremor(b, dt, rng, s)}[name](); done.append(name)
    return b, dict(ops=done, strength=round(s, 3), uncommon=unc)


def closest_action(v, prof):
    fits = {k: G.grade_features(v, k, prof, use_match=False)["fit"] for k in prof.p}; fits = {k: f for k, f in fits.items() if f is not None}
    k = max(fits, key=fits.get) if fits else None
    return k, (fits[k] if k else None)


def clip_rep(a):
    """A clip as a fixed-size vector: the position channels resampled to REP_FRAMES points along normalised time (metres). Distances between these are metres of RMS difference."""
    a = np.asarray(a, np.float64); t = np.linspace(0, len(a) - 1, REP_FRAMES); i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, len(a) - 1); w = (t - i0)[:, None]
    return ((a[i0][:, POS] * (1 - w) + a[i1][:, POS] * w)).ravel()


def rep_dist(r, others):
    """Smallest RMS difference (m, per channel and point) between rep r and any of `others` (list of reps)."""
    if not len(others):
        return np.inf
    return float(np.sqrt(((np.asarray(others) - r) ** 2).mean(1)).min())


def duplicate_pairs(clips, thr=0.01):
    """(i, j) pairs, i < j, of clips whose rep distance is < thr metres: the same motion stored twice (the CMU set has 39 such pairs under different performer ids,
    e.g. subjects 77 and 139, 91 and 105, 93 and 103; nothing lies between 2 mm and 5 cm, so the threshold is not delicate)."""
    R = np.stack([clip_rep(c) for c in clips]).astype(np.float64); sq = (R ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None] - 2 * R @ R.T, 0) / R.shape[1]
    i, j = np.where(np.triu(d2 < thr ** 2, 1)); return list(zip(i.tolist(), j.tolist()))


def merge_groups(labels, pairs):
    """labels: one group label per clip; pairs: clip index pairs that are the same motion. Returns the labels with every group that shares a duplicate merged (union-find)."""
    par = {l: l for l in set(labels)}
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in pairs:
        a, b = find(labels[i]), find(labels[j])
        if a != b:
            par[b] = a
    return [find(l) for l in labels]


def check_variant(b, dt, action, prof, src_fit, src_rep, sib_reps=()):
    """Accept/reject a variant. Returns (ok, why, features, fit, distance). Distance = rep_dist to the source and the already accepted siblings (not a copy of any)."""
    v = mf.extract(b, dt)
    if not np.all(np.isfinite(np.asarray(b))) or np.isfinite(v).sum() < 0.6 * len(v):
        return False, "unmeasurable", v, None, None
    r = G.grade_features(v, action, prof, use_match=False)
    if r["fit"] is None:
        return False, "unmeasurable", v, None, None
    if r["gate"] < 0.99:
        return False, "gate (jitter / planted-foot mismatch)", v, r["fit"], None
    floor = max(MIN_FIT, (src_fit if src_fit is not None else 100.0) - FIT_DROP)
    if r["fit"] < floor:
        return False, f"fit {r['fit']:.0f} < {floor:.0f}", v, r["fit"], None
    d = rep_dist(clip_rep(b), [src_rep] + list(sib_reps))
    if d < MIN_DIST:
        return False, f"too close to its source/siblings ({d:.3f} m)", v, r["fit"], d
    return True, "", v, r["fit"], d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz"); ap.add_argument("--clip", type=int, required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default="data/var.npz")
    ap.add_argument("--action"); ap.add_argument("--uncommon", type=float, default=0.12); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--real", default=G.REAL_DEFAULT)
    ap.add_argument("--max-tries", type=int, default=0)
    a = ap.parse_args(); import motion_memory as MM
    z = np.load(a.npz, allow_pickle=True); dt = float(z["dt"]); clip = G.split_clips(z)[a.clip]; zr, RF = G.load_real(a.real); prof = G.Profiles(RF, zr["action"])
    v0 = mf.extract(clip, dt); act, fit0 = closest_action(v0, prof)
    if a.action:
        act = a.action; fit0 = G.grade_features(v0, act, prof, use_match=False)["fit"]
    print(f"source clip {a.clip}: {len(clip) * dt:.1f} s, graded as '{act}' (fit {fit0:.0f})")
    rng = np.random.default_rng(a.seed); kept, infos, sib, why = [], [], [], {}; tries = 0; cap = a.max_tries or a.n * 8; r0 = clip_rep(clip)
    while len(kept) < a.n and tries < cap:
        tries += 1; b, info = make_variant(clip, dt, rng, a.uncommon); ok, w, v, fit, d = check_variant(b, dt, act, prof, fit0, r0, sib)
        if ok:
            kept.append(b); sib.append(clip_rep(b)); info.update(fit=round(fit), dist=round(d, 3)); infos.append(info)
        else:
            k = w.split(" (")[0].split(" <")[0]; why[k] = why.get(k, 0) + 1
    for i, inf in enumerate(infos):
        print(f"  variant {i}: ops {','.join(inf['ops']) or '-':38s} strength {inf['strength']:.2f}{' UNCOMMON' if inf['uncommon'] else ''}   fit {inf['fit']}   distance {inf['dist']} m from the nearest of source/siblings")
    print(f"{len(kept)} kept of {tries} tries" + (f"; rejected because: {why}" if why else ""))
    if kept:
        allc = [clip] + kept
        np.savez_compressed(a.out, actions=np.concatenate(allc).astype(np.float32), offsets=np.cumsum([0] + [len(c) for c in allc]), dt=np.float32(dt),
                            action=np.array([act] * len(allc)), labels=np.array(["source"] + [f"variant {i}" for i in range(len(kept))]),
                            description=np.array(["the source clip"] + [",".join(i["ops"]) for i in infos]), style=np.zeros((len(allc), 1), np.float32), style_names=np.array(["none"]))
        print(f"wrote {a.out}: the source first, then the variants  (python tools/view_clip.py {a.out} --open)")


if __name__ == "__main__":
    main()
