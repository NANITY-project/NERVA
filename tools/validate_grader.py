#!/usr/bin/env python3
"""
validate_grader.py - is the grader any good? Tests it on REAL captures it has never seen, and on corrupted ones.

  python tools/validate_grader.py                       # all actions, about 30 s once data/real_motions.npz exists (it is downloaded if missing)
  python tools/validate_grader.py --actions sit stand walk
  python tools/validate_grader.py --engine data/clips.npz --engine-action walk       # also grade your engine's clips as that action

1. HELD-OUT REAL clips. Every real clip is graded against profiles built WITHOUT its performer (leave-one-performer-out; with fewer than 4 performers: without
   that one trial), so the examples cannot grade themselves. A good grader gives real clips a high score and recognises their action: you get the share of real
   clips that pass, and the "confusion" (what each action is mistaken for).
2. WRONG-LABEL test. The same real clips are graded as every OTHER action. A good grader fails nearly all of them: this is the test that a "sit" dataset cannot
   pass by being some motion that happens to look human.
3. CORRUPTED real clips (noise, blur, frozen frames, time-reversed contacts, scaled-up limbs, blended with another clip): they must score clearly below the clean ones.
4. Optionally your engine's own clips, judged as the action you name.
Exit status 1 if the grader fails its own bar (so it can sit in a test pipeline): held-out real clips must pass >= 75% (mean score >= 80) with the right action
recognised >= 60% overall, wrong-label clips must fail >= 90%, and corrupted clips must score clearly lower (noise and blends >= 15 points, limbs 40% too far out >= 8).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_features as mf, motion_grader as G

CORRUPT = ["noise 8 mm", "blur (9-frame average)", "frozen every 4th frame", "contacts flipped", "limbs 40% too far out", "blend with another clip", "reversed in time"]


def corrupt(rng, a, other, kind):
    c = a.copy(); cont = [i for i in (0, 1, 2, 4, 5, 6, 9, 10, 11, 14, 15, 16, 18, 19, 20, 26, 27, 28)]
    if kind == "noise 8 mm":
        c[:, cont] += rng.normal(0, 0.008, (len(a), len(cont)))
    elif kind == "blur (9-frame average)":
        k = 9; ker = np.ones(k) / k
        for i in cont:
            c[:, i] = np.convolve(np.pad(a[:, i], (k // 2, k // 2), mode="edge"), ker, "valid")
    elif kind == "frozen every 4th frame":
        for i in range(len(a)):
            c[i] = a[i - i % 4]
    elif kind == "contacts flipped":
        c[:, [8, 13]] = 1 - c[:, [8, 13]]
    elif kind == "limbs 40% too far out":
        for cols in ([4, 5, 6], [9, 10, 11], [14, 15, 16], [18, 19, 20]):
            c[:, cols] = a[:, 0:3] + 1.4 * (a[:, cols] - a[:, 0:3])
    elif kind == "blend with another clip":
        n = min(len(a), len(other)); c = 0.5 * a[:n] + 0.5 * other[:n]
    elif kind == "reversed in time":
        c = a[::-1].copy(); c[:, 3] = c[:, 3]
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=G.REAL_DEFAULT); ap.add_argument("--actions", nargs="+"); ap.add_argument("--engine"); ap.add_argument("--engine-action", default="walk")
    ap.add_argument("--n-corrupt", type=int, default=60, help="clean clips to corrupt"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); z, F = G.load_real(a.real); act = z["action"]; perf = z["performer"]; lab = z["labels"]; A, off, dt = z["actions"], z["offsets"], float(z["dt"])
    names = sorted(set(act)); sel = [n for n in names if (not a.actions or n in a.actions)]; N = len(F)
    # ---- 1. held-out real clips
    held = {}                                                                # action -> list of (score, recognised, rank)
    cache = {}
    for i in range(N):
        if act[i] not in sel:
            continue
        n_perf = len(set(perf[act == act[i]]))
        key = (act[i], perf[i]) if n_perf >= 4 else (act[i], lab[i])
        if key not in cache:
            ex = (act == act[i]) & ((perf == perf[i]) if n_perf >= 4 else (lab == lab[i])); cache[key] = G.Profiles(F, act, exclude=ex)
        P = cache[key]
        if act[i] not in P.p:
            continue
        r = G.grade_features(F[i], act[i], P); rec = P.recognise(F[i])
        if r["score"] is not None:
            held.setdefault(act[i], []).append((r["score"], rec[0][0], next(k for k, (x, _) in enumerate(rec) if x == act[i]) + 1, rec[0][0]))
    print("=" * 104); print("1. HELD-OUT REAL clips: do real people score high, and is their action recognised? (leave-one-performer-out)")
    print(f"   {'action':12s}{'clips':>6s}{'mean':>7s}{'pass>=70':>10s}{'recognised':>12s}{'top-3':>7s}   mistaken for")
    allsc, allrec = [], []
    for n in sel:
        h = held.get(n)
        if not h:
            continue
        s = np.array([x[0] for x in h]); rc = np.array([x[1] in G.family(n) for x in h]); t3 = np.array([x[2] <= 3 for x in h]); mis = {}
        for x in h:
            if x[1] not in G.family(n):
                mis[x[1]] = mis.get(x[1], 0) + 1
        allsc += list(s); allrec += list(rc)
        print(f"   {n:12s}{len(h):6d}{s.mean():7.0f}{100 * np.mean(s >= G.PASS_SCORE):9.0f}%{100 * rc.mean():11.0f}%{100 * t3.mean():6.0f}%   " + ", ".join(f"{k} x{v}" for k, v in sorted(mis.items(), key=lambda kv: -kv[1])[:3]))
    allsc = np.array(allsc); real_pass = float(np.mean(allsc >= G.PASS_SCORE)); real_rec = float(np.mean(allrec))
    print(f"   ALL: mean {allsc.mean():.0f}, {100 * real_pass:.0f}% pass, {100 * real_rec:.0f}% recognised")
    # ---- 2. wrong label
    print("\n2. WRONG-LABEL test: real clips graded as a different action (should FAIL; this is what stops 'any human-looking motion' from passing as 'sit')")
    full = G.Profiles(F, act); wrong, close = [], {}
    rng = np.random.default_rng(0)
    for i in range(N):
        if act[i] not in sel:
            continue
        for b in rng.permutation([x for x in full.names if x not in G.family(act[i])])[:4]:
            r = G.grade_features(F[i], b, full)
            if r["score"] is not None:
                wrong.append(r["score"]); close.setdefault((act[i], b), []).append(r["score"])
    wrong = np.array(wrong); wrong_fail = float(np.mean(wrong < G.PASS_SCORE))
    print(f"   {len(wrong)} wrong-label gradings: mean score {wrong.mean():.0f}, {100 * wrong_fail:.0f}% fail (a clip passes at {G.PASS_SCORE:.0f})")
    worst = sorted(((np.mean(v), k) for k, v in close.items() if len(v) >= 3), reverse=True)[:6]
    for x, y in (("sit", "stand"), ("stand", "sit"), ("walk", "run"), ("run", "walk"), ("walk", "jump"), ("sit", "walk")):
        if (x, y) in close:
            v = np.array(close[(x, y)]); print(f"   real '{x}' clips graded as '{y}': mean {v.mean():.0f}, {100 * np.mean(v < G.PASS_SCORE):.0f}% fail")
    print("   easiest to confuse (graded as the second action, mean score): " + ", ".join(f"{k[0]} as {k[1]} {m:.0f}" for m, k in worst))
    # ---- 3. corruptions
    print("\n3. CORRUPTED real clips graded as their own action (clean clips shown first); a good grader drops them well below the clean score")
    ids = [i for i in range(N) if act[i] in sel]; ids = list(rng.permutation(ids)[: a.n_corrupt]); res = {"(clean)": []}
    for i in ids:
        c = A[off[i]:off[i + 1]]; j = ids[(ids.index(i) + 1) % len(ids)]; o = A[off[j]:off[j + 1]]
        P = cache.get((act[i], perf[i])) or cache.get((act[i], lab[i])) or full
        P = P if act[i] in P.p else full
        res["(clean)"].append(G.grade_features(mf.extract(c, dt), act[i], P)["score"])
        for k in CORRUPT:
            r = G.grade_features(mf.extract(corrupt(rng, c, o, k).astype(np.float32), dt), act[i], P)["score"]
            res.setdefault(k, []).append(r if r is not None else np.nan)
    clean = float(np.nanmean(res["(clean)"])); print(f"   {'(clean clips)':30s}{clean:6.0f}")
    drops = {}
    for k in CORRUPT:
        m = float(np.nanmean(res[k])); drops[k] = clean - m; print(f"   {k:30s}{m:6.0f}   ({clean - m:+.0f} points)")
    # ---- 4. engine
    if a.engine:
        z2 = np.load(a.engine, allow_pickle=True); cl = G.split_clips(z2); print(f"\n4. YOUR ENGINE: {a.engine} judged as '{a.engine_action}'"); G.report(G.grade_dataset(cl, float(z2["dt"]), a.engine_action, full), verbose=True)
    # ---- bar
    print("=" * 104)
    bar = [("held-out real clips pass >= 75%", real_pass >= 0.75), ("held-out real mean score >= 80", allsc.mean() >= 80), ("real clips recognised as own action >= 60%", real_rec >= 0.60),
           ("wrong-label clips fail >= 90%", wrong_fail >= 0.90), ("corruptions must cost points: noise >= 15, blend >= 15, limbs >= 8", drops["noise 8 mm"] >= 15 and drops["blend with another clip"] >= 15 and drops["limbs 40% too far out"] >= 8)]
    for t, ok in bar:
        print(f"   [{'ok' if ok else 'FAIL'}] {t}")
    print("   other corruptions, points lost: " + ", ".join(f"{k} {drops[k]:+.0f}" for k in CORRUPT if k not in ("noise 8 mm", "limbs 40% too far out", "blend with another clip")))
    print("   (blur and time reversal are physically plausible motion: the grader is NOT expected to catch them; use the engine's own verdict for feasibility)")
    sys.exit(0 if all(ok for _, ok in bar) else 1)


if __name__ == "__main__":
    main()
