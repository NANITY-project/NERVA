#!/usr/bin/env python3
"""
get_real_motions.py - download a LARGE, varied set of real motion-capture clips (CMU Graphics Lab database) and cut them into labelled examples:
sit, stand, walk, run, jump, crouch, get up, kick, throw, dance ... (see tools/motion_actions.py for the whole list and how a trial becomes an example).

  python tools/get_real_motions.py                       # every action  -> data/real_motions.npz   (first run downloads ~150-300 MB, cached in data/cmu_cache/)
  python tools/get_real_motions.py --actions sit stand   # only these
  python tools/get_real_motions.py --quick               # a few clips per action: for a fast smoke test
  python tools/get_real_motions.py --list                # show what would be downloaded, download nothing
  python tools/get_real_motions.py --define "wave=^wave$,^waving$"      # your own action, matched on the CMU trial description

How the examples are made
  * a trial is matched to an action by its CMU description (anchored patterns: "walking, squats, kicking" is a mix and belongs to none)
  * trials are picked round-robin over performers, so 10 examples come from 10 people, not one person 10 times
  * a long trial gives several non-overlapping windows (up to --per-trial); each window is placed by what the action is about: the busiest part for walking,
    around the flight for jumps, around the lowest the pelvis gets for sit / crouch / pick up / get up, a still upright stretch for stand
  * "stand" has almost no trials of its own in CMU, so still upright stretches are also found inside every other downloaded trial
Output (--out): actions [F,30] float32 in the engine's action layout, offsets [N+1], dt, angles [F,5], labels (trial id), action (per clip), performer,
description, features [N,K] + feature_names (the numbers the grader uses, computed once here), action_names.
Data: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), BVH conversion by B. Hahne (cgspeed), mirrored at github.com/sfd158/cmu-mocap. Credit them.
"""
import argparse, concurrent.futures as cf, math, os, sys, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_to_clip, motion_actions as MA, motion_features as mf

MAX_BVH_MB = 25
LOCOMOTION_MIN_SPEED = {"walk": 0.35, "walk_styled": 0.25, "run": 1.2, "sneak": 0.15, "limp": 0.2, "carry": 0.35}


def download(tid, cache, tries=3):
    path = os.path.join(cache, tid + ".bvh")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    url = MA.BVH_URL.format(d=tid.split("_")[0].zfill(3), f=tid); err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                if int(r.headers.get("Content-Length") or 0) > MAX_BVH_MB * 1e6:
                    return None
                data = r.read()
            if len(data) < 1000:
                raise IOError("empty response")
            open(path + ".part", "wb").write(data); os.replace(path + ".part", path); return path
        except Exception as e:
            err = e; time.sleep(1.5 * (k + 1))
    print(f"  could not download {tid}: {err}", flush=True)
    return None


def windows(A, dt, act, per_trial):
    """[(i0, i1)] windows of this trial that are examples of the action."""
    F = len(A); W = int(round(act.secs / dt)); minlen = int(round(1.0 / dt))
    if F < minlen:
        return []
    if F <= int(W * 1.25):
        return [(0, F)]
    S = lambda x: mf._ma(x, int(round(0.2 / dt)))
    pel = S(A[:, 0:3]); sp = np.r_[0, np.linalg.norm(np.diff(pel[:, [0, 2]], axis=0), axis=1)] / dt
    def clamp(c):
        i0 = int(np.clip(c - W // 2, 0, F - W)); return (i0, i0 + W)
    if act.window == "low":                                    # around the lowest pelvis: the sit / crouch / pick-up itself, with the lead-in and the way back
        return [clamp(int(np.argmin(pel[:, 1])))]
    if act.window == "air":                                    # around the longest flight
        fl = (A[:, 8] < 0.5) & (A[:, 13] < 0.5); runs = mf._runs(fl)
        if runs:
            r = max(runs, key=lambda r: r[1] - r[0]); return [clamp((r[0] + r[1]) // 2)]
    if act.window == "quiet":
        return [(0, min(F, W))]
    out = []
    for i0 in range(0, F - W + 1, W):                          # busiest non-overlapping windows
        seg = slice(i0, i0 + W); e = float(sp[seg].mean()) + 0.5 * float(np.linalg.norm(np.diff(A[seg][:, [14, 15, 16, 18, 19, 20]], axis=0), axis=1).mean() / dt)
        if act.name in LOCOMOTION_MIN_SPEED and sp[seg].mean() < LOCOMOTION_MIN_SPEED[act.name]:
            continue
        out.append((e, i0))
    out.sort(reverse=True)
    return [(i0, i0 + W) for _, i0 in out[:per_trial]]


def canonical(A):
    """Move a clip so it starts at the origin facing +Z (the same convention as gen_dataset.py), whatever part of the trial it was cut from."""
    A = A.copy(); y0 = float(A[0, 3]); c, s_ = np.cos(-y0), np.sin(-y0); o = A[0, [0, 2]].copy()
    for k in (0, 4, 9, 14, 18, 22, 26):
        x = A[:, [k, k + 2]] - o; A[:, k] = x[:, 0] * c + x[:, 1] * s_; A[:, k + 2] = -x[:, 0] * s_ + x[:, 1] * c
    for k in (3, 7, 12):
        A[:, k] -= y0
    return A


def is_standing(v):
    g = lambda k: v[mf.INDEX[k]]
    return g("pelvis_h_med") > 0.85 and g("leg_ext_med") > 0.93 and g("speed_mean") < 0.2 and g("step_rate") < 0.8 and g("contact_none") < 0.02 and g("vert_speed_p95") < 0.25 and g("hand_speed_mean") < 0.30 and g("quiet_frac") > 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/real_motions.npz"); ap.add_argument("--cache", default="data/cmu_cache"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--actions", nargs="+", help="only these actions (default: all)"); ap.add_argument("--define", action="append", default=[], help='NAME=regex,regex')
    ap.add_argument("--per-trial", type=int, default=3, help="max windows cut from one long trial"); ap.add_argument("--max-clips", type=int, help="override every action's clip limit")
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--quick", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    trials, _ = MA.fetch_index(a.cache); acts = [MA.define(d) for d in a.define] + MA.ACTIONS
    if a.actions:
        acts = [x for x in acts if x.name in a.actions]
        if not acts:
            sys.exit("no such action; known: " + ", ".join(x.name for x in MA.ACTIONS))
    for x in acts:
        if a.quick:
            x.n_max = min(x.n_max, 6); x.min_n = 3
        if a.max_clips:
            x.n_max = a.max_clips
    per_trial = 1 if a.quick else a.per_trial
    got = MA.assign(trials, acts); chosen = {}
    for x in acts:
        n_trials = max(x.min_n, int(math.ceil(x.n_max / per_trial * 2.5))) if x.window == "energy" else int(x.n_max * 1.3) + 2
        chosen[x.name] = MA.spread_over_subjects(got[x.name], n_trials)
    need = sorted({t for v in chosen.values() for t, _ in v})
    if a.list:
        for x in acts:
            print(f"{x.name:12s}{len(got[x.name]):5d} trials match, {len(chosen[x.name]):4d} downloaded, {x.n_max} clips kept")
        print(f"{len(need)} BVH files in total"); return
    # the standing action also mines other trials, so those must be present too: take the sit / stretch / dance / gesture trials already chosen (no extra downloads)
    print(f"downloading {len(need)} BVH files with {a.jobs} threads (cached ones are skipped) ...", flush=True); t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        paths = dict(zip(need, ex.map(lambda t: download(t, a.cache), need)))
    print(f"  done in {time.time() - t0:.0f}s, {sum(1 for p in paths.values() if p)} available", flush=True)
    desc = dict(trials); conv = {}
    def convert(tid):
        if tid not in conv:
            try:
                conv[tid] = bvh_to_clip.convert(paths[tid], a.fps)
            except Exception as e:
                print(f"  could not convert {tid}: {type(e).__name__}: {e}", flush=True); conv[tid] = None
        return conv[tid]
    clips, meta = [], []                                       # (A, ang) ; (action, trial, performer, description)
    for x in acts:
        n0 = len(clips); mined = []
        for tid, d in chosen[x.name]:
            if len(clips) - n0 >= x.n_max or not paths.get(tid):
                continue
            r = convert(tid)
            if r is None:
                continue
            A, ang, dt = r
            for i0, i1 in windows(A, dt, x, per_trial)[: max(1, x.n_max - (len(clips) - n0))]:
                clips.append((canonical(A[i0:i1]), ang[i0:i1], dt)); meta.append((x.name, tid, MA.subject_of(tid), d))
        if x.derived == "quiet_stand":                         # still upright stretches inside the other downloaded trials
            tr_act = {t: nm for nm, v in got.items() for t, _ in v}            # only trials whose body is not busy with the arms or a stunt: their quiet stretches are plain standing
            ok_src = {"walk", "walk_styled", "sit", "pickup", "get_up", "balance", "turn", "run", "carry", "limp", "sneak", "crouch"}
            pool = [t for t, p in paths.items() if p and tr_act.get(t) in ok_src and t not in {m[1] for m in meta if m[0] == x.name}]
            for tid in MA.spread_over_subjects([(t, desc[t]) for t in pool], len(pool)):
                if len(clips) - n0 >= x.n_max:
                    break
                r = convert(tid[0])
                if r is None:
                    continue
                A, ang, dt = r; W = int(round(x.secs / dt))
                for i0 in range(0, len(A) - W + 1, W // 2):
                    v = mf.extract(A[i0:i0 + W], dt)
                    if is_standing(v):
                        clips.append((canonical(A[i0:i0 + W]), ang[i0:i0 + W], dt)); meta.append((x.name, tid[0], MA.subject_of(tid[0]), "still upright stretch of: " + desc[tid[0]])); break
        k = len(clips) - n0; perf = len({m[2] for m in meta[n0:]})
        print(f"  {x.name:12s}{k:4d} clips from {perf:2d} performers" + ("   (too few: dropped)" if k < x.min_n else ""), flush=True)
        if k < x.min_n:
            del clips[n0:], meta[n0:]
    if not clips:
        sys.exit("no clips: check your internet connection, or put the .bvh files in " + a.cache)
    dt = float(np.mean([c[2] for c in clips])); off = np.cumsum([0] + [len(c[0]) for c in clips])
    feats = np.array([mf.extract(c[0], dt) for c in clips], np.float32)
    np.savez_compressed(a.out, actions=np.concatenate([c[0] for c in clips]).astype(np.float32), offsets=off, angles=np.concatenate([c[1] for c in clips]), dt=np.float32(dt),
                        labels=np.array([m[1] for m in meta]), action=np.array([m[0] for m in meta]), performer=np.array([m[2] for m in meta]), description=np.array([m[3] for m in meta]),
                        features=feats, feature_names=np.array(mf.NAMES), action_names=np.array(sorted({m[0] for m in meta})), kind=np.array(["real"] * len(clips)),
                        style=np.zeros((len(clips), 1), np.float32), style_names=np.array(["real"]))
    print(f"\nwrote {a.out}: {len(clips)} clips, {len({m[2] for m in meta})} performers, {len({m[0] for m in meta})} actions, {off[-1] * dt / 60:.1f} minutes of real motion")


if __name__ == "__main__":
    main()
