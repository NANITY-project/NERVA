#!/usr/bin/env python3
"""
motion_grader.py - judge a DATASET of motion clips against REAL examples of the action it claims to be.

  python tools/motion_grader.py data/sit_clips.npz --action sit          # is this a good "sit" dataset?
  python tools/motion_grader.py data/clips.npz                           # the file has an `action` label per clip: every action in it is judged separately
  python tools/motion_grader.py data/clips.npz --action walk --clips 0 1 2 --verbose      # per-clip report cards
  python tools/motion_grader.py --list                                   # the actions the grader knows and how many real examples each has

Input: an npz as written by gen_dataset.py / bvh_to_clip.py: actions [F,30], offsets [N+1], dt  (+ optional `action` [N] strings, one label per clip).
Reference: data/real_motions.npz from tools/get_real_motions.py (it is downloaded automatically the first time).

What it judges (the old grader judged "is this a human WALK" on thousands of random clips; this one asks "is this a good dataset of <action>")
  1. every clip is scored 0-100 on ~40 body measures (tools/motion_features.py) against the range real performers of THAT action fall in
  2. does it LOOK LIKE the requested action and not another one? each clip is matched to all actions: a "sit" clip that matches "stand" best fails
  3. is the dataset varied? 400 near-identical clips are worth one: spread compared with the spread of the real examples, plus a duplicate count
  4. which measures are off, for the whole dataset: "pelvis_h_min 0.78 m, real 0.45-0.60 m: body never gets as low as real examples"
The verdict (PASS / MARGINAL / FAIL) needs all of: mean score, share of clips that pass, share recognised as the action, and variety.
Honest limits: "human-like" means "inside the range of the real CMU performers (74 people, mostly young adults, in a lab)". Hand-drawn, stylised or exaggerated
motions can score low on purpose of their style: pass --allow GROUP (height trunk legs feet rhythm speed turn hands) to stop a group costing points; the jitter
gate (motion a body cannot make) is never waived. Small actions have few real examples (see --list): the profile of "balance" rests on 8 clips, so trust it less.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_features as mf

REAL_DEFAULT = "data/real_motions.npz"
PASS_SCORE = 70.0           # a clip passes at this score
GATE_FLOOR = 0.35           # a failed gate (jitter) leaves at most this share of the points
# physical-consistency measures are FIXED, not learned from the examples: a body has no business above ~0.07 m/s of >6 Hz velocity (real, after the hand filter:
# 0.01-0.06), and a foot flagged planted must be on the floor. A failed gate caps the score. (typical lo, hi, tolerable lo, hi)
FIXED = {"jitter_hf": (0.0, 0.07, 0.0, 0.16), "contact_mismatch": (0.0, 0.08, 0.0, 0.45)}
WIDEN, TOL = 0.10, 0.75         # typical range = real 10-90% widened by 10% of its width; the score reaches 0 three quarters of a width beyond that
MATCH_FLOOR = 0.40              # a clip the classifier does not take for the requested action (or a close relative) keeps at most this share of its range score
# actions that kinematics from pelvis, feet, hands and neck cannot tell apart reliably: confusing them within a family does not count as "not recognised"
FAMILIES = [{"walk", "walk_styled", "limp", "sneak", "carry", "turn", "crouch"}, {"jump", "hop"}, {"drink", "gesture", "chores", "dribble", "dance", "stretch", "balance"},
            {"punch", "throw", "kick"}, {"crouch", "pickup", "sneak"}, {"sit", "get_up"}]
LDA_SHRINK = 0.25
ONE_SIDED = ("planted_slide",)
VERDICT = dict(score=75.0, passing=0.70, recognised=0.70, variety=0.35)


def trapezoid(v, tlo, thi, lo, hi):
    if v is None or not np.isfinite(v):
        return None
    if tlo <= v <= thi:
        return 1.0
    if v < tlo:
        return float(np.clip((v - lo) / (tlo - lo), 0, 1)) if tlo > lo else 0.0
    return float(np.clip((hi - v) / (hi - thi), 0, 1)) if hi > thi else 0.0


# ------------------------------------------------------------------------------------------------------------------ profiles
def _width(q10, q90, floor):
    return max(q90 - q10, floor)


def build_profile(F, others, min_n=4):
    """Per-feature ranges and weights of one action. F: [n,K] features of its real examples; others: [m,K] features of every other action (for the weights)."""
    K = F.shape[1]; prof = dict(n=len(F), med=np.full(K, np.nan), typ=np.full((K, 2), np.nan), tol=np.full((K, 2), np.nan), w=np.zeros(K), scale=np.full(K, np.nan))
    for j in range(K):
        x = F[:, j]; x = x[np.isfinite(x)]
        if len(x) < max(min_n, 0.5 * len(F)):
            continue
        q10, q50, q90 = np.percentile(x, [10, 50, 90]); w = _width(q10, q90, mf.FLOOR[j]); f = mf.FLOOR[j]
        prof["med"][j] = q50; prof["typ"][j] = (q10 - WIDEN * w, q90 + WIDEN * w); prof["scale"][j] = w + f
        prof["tol"][j] = (q10 - WIDEN * w - TOL * w - f, q90 + WIDEN * w + TOL * w + f)
        o = others[:, j]; o = o[np.isfinite(o)]
        if len(o) >= 8:                                           # how much this measure tells the action apart from the others: distance of medians / pooled spread
            iq = lambda v: np.subtract(*np.percentile(v, [75, 25]))
            d = abs(q50 - np.median(o)) / (0.5 * (iq(x) + iq(o)) + f); prof["w"][j] = float(np.clip(d, 0.1, 3.0))
        else:
            prof["w"][j] = 1.0
    for n in ONE_SIDED:                                           # less is never worse (a perfectly planted foot is not "off the real range"): no lower limit
        j = mf.INDEX[n]
        if np.isfinite(prof["typ"][j][0]):
            prof["typ"][j][0] = 0.0; prof["tol"][j][0] = -1.0
    for n, (tl, th, ll, lh) in FIXED.items():
        if n == "jitter_hf":                                      # running and jumping are broadband (impacts): the limit is the real examples' own 95th percentile, never below the default
            x = F[:, mf.INDEX[n]]; x = x[np.isfinite(x)]
            if len(x) >= 8:
                th = max(th, 1.1 * float(np.percentile(x, 95))); lh = th + max(lh - FIXED[n][1], th)
        j = mf.INDEX[n]; prof["typ"][j] = (tl, th); prof["tol"][j] = (ll, lh); prof["w"][j] = 2.0; prof["med"][j] = 0.5 * (tl + th); prof["scale"][j] = th - tl
    return prof


def family(a):
    """The action and everything it is allowed to be mistaken for (a family overlaps others: crouch is both a walk style and a low body posture)."""
    out = {a}
    for f in FAMILIES:
        if a in f:
            out |= f
    return out


class LDA:
    """Regularised linear discriminant analysis over the real examples: P(action | clip). Standardised features, shrunk pooled covariance, equal priors."""
    def __init__(self, X, y):
        self.cls = sorted(set(y)); y = np.asarray(y); self.cols = [j for j, n in enumerate(mf.NAMES) if n not in FIXED]
        X = X[:, self.cols]; self.med = np.nanmedian(X, axis=0); self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        X = np.where(np.isfinite(X), X, self.med); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; Z = (X - self.mu) / self.sd
        M = np.array([Z[y == c].mean(0) for c in self.cls]); R = np.concatenate([Z[y == c] - M[k] for k, c in enumerate(self.cls)])
        Sw = R.T @ R / max(1, len(R) - len(self.cls)); S = (1 - LDA_SHRINK) * Sw + LDA_SHRINK * np.eye(len(Sw)) * np.trace(Sw) / len(Sw)
        self.W = np.linalg.solve(S, M.T); self.b = -0.5 * (M.T * self.W).sum(0)

    def proba(self, v):
        x = np.asarray(v, np.float64)[self.cols]; x = (np.where(np.isfinite(x), x, self.med) - self.mu) / self.sd
        d = x @ self.W + self.b; d = d - d.max(); e = np.exp(d); return dict(zip(self.cls, e / e.sum()))


class Profiles:
    """All actions' profiles, built from the real examples. `exclude` (a boolean mask over the real clips) leaves clips out: that is how the grader is validated honestly."""
    def __init__(self, F, action, performer=None, trial=None, exclude=None):
        self.names = sorted(set(action)); self.F = F; self.action = np.asarray(action); keep = np.ones(len(F), bool) if exclude is None else ~exclude
        self.p = {}
        for a in self.names:
            m = (self.action == a) & keep
            if m.sum() >= 3:
                self.p[a] = build_profile(F[m], F[(self.action != a) & keep])
        self.names = list(self.p)
        self.lda = LDA(F[keep & np.isin(self.action, self.names)], self.action[keep & np.isin(self.action, self.names)])

    def distance(self, v, a):
        """How far a feature vector is from action a's centre, in units of that action's own spread (robust mean, each measure capped at 5)."""
        p = self.p[a]; ok = np.isfinite(v) & np.isfinite(p["med"]); w = p["w"][ok]
        if not ok.any():
            return np.inf
        z = np.minimum(np.abs(v[ok] - p["med"][ok]) / p["scale"][ok], 5.0); return float((w * z).sum() / w.sum())

    def recognise(self, v):
        """[(action, probability)] from best to worst match (the classifier's posterior)."""
        pr = self.lda.proba(v); return sorted(pr.items(), key=lambda kv: -kv[1])

    def match(self, v, a):
        """Probability that the clip is `a` or a close relative of it (same family)."""
        pr = self.lda.proba(v); return float(sum(pr.get(b, 0.0) for b in family(a)))


def load_real(path=REAL_DEFAULT, cache_dir=None):
    if not os.path.exists(path):
        print(f"{path} not found: downloading the real motion set first (once, about a minute) ...", flush=True)
        import get_real_motions
        argv = sys.argv; sys.argv = ["get_real_motions.py", "--out", path] + (["--cache", cache_dir] if cache_dir else []); get_real_motions.main(); sys.argv = argv
    z = np.load(path, allow_pickle=True)
    if "features" in z.files and list(z["feature_names"]) == mf.NAMES:
        F = z["features"].astype(np.float64)
    else:                                                          # features of an older file, or the feature list changed: recompute
        A, off, dt = z["actions"], z["offsets"], float(z["dt"]); F = np.array([mf.extract(A[off[i]:off[i + 1]], dt) for i in range(len(off) - 1)])
    return z, F


def get_profiles(path=REAL_DEFAULT):
    z, F = load_real(path); return Profiles(F, z["action"]), z, F


# ------------------------------------------------------------------------------------------------------------------ grading
def grade_features(v, action, prof, allow=(), use_match=True):
    """Score one clip's features against action's profile. allow = feature names or group names whose deviation costs nothing (the jitter gate never is waivable)."""
    p = prof.p[action]; allowed = set()
    for x in allow:
        allowed |= {n for n in mf.NAMES if mf.GROUP[n] == x} if x in mf.GROUP_NAMES else ({x} if x in mf.INDEX else set())
    allowed -= set(mf.GATES); items = []; num = den = 0.0; gate = 1.0
    for j, n in enumerate(mf.NAMES):
        if not np.isfinite(p["med"][j]) or p["w"][j] <= 0:
            continue
        x = v[j]; sc = trapezoid(x, *p["typ"][j], *p["tol"][j]) if np.isfinite(x) else None
        if sc is None:
            continue
        lo_t, hi_t = p["typ"][j]; low, high = mf.TEXT[n]; note = "" if sc >= 0.999 else (low if x < lo_t else high)
        items.append(dict(name=n, value=float(x), score=sc, weight=float(p["w"][j]), unit=mf.UNIT[n], group=mf.GROUP[n], typical=(float(lo_t), float(hi_t)), allowed=n in allowed, note=note))
        num += p["w"][j] * (1.0 if n in allowed else sc); den += p["w"][j]
        if n in mf.GATES:
            gate = min(gate, sc)
    if den == 0:
        return dict(score=None, fit=None, match=0.0, items=[], findings=["no measurable features (clip too short?)"], gate=1.0)
    gf = GATE_FLOOR + (1 - GATE_FLOOR) * gate; raw = 100 * sum(i["weight"] * i["score"] for i in items) / sum(i["weight"] for i in items)
    pm = prof.match(v, action) if use_match else 1.0; mf_ = MATCH_FLOOR + (1 - MATCH_FLOOR) * min(1.0, pm / 0.5)
    bad = sorted([i for i in items if i["score"] < 0.75], key=lambda i: -(i["weight"] * (1 - i["score"])))
    lines = [f"{i['name']} = {i['value']:.3g} {i['unit']} (real {i['typical'][0]:.3g} to {i['typical'][1]:.3g}): {i['note']}" + (" (allowed by the request)" if i["allowed"] else "") for i in bad]
    if gate < 0.999:
        lines.insert(0, f"GATE: motion is not smooth enough to be a body's (jitter score {gate:.2f}): the score is capped at {100 * gf:.0f}% of what the other measures would give")
    fit = float(gf * 100 * num / den)
    if mf_ < 0.999:
        lines.insert(0, f"does not look like '{action}': the classifier gives {100 * pm:.0f}% to it or its relatives; the range score {fit:.0f} is scaled by {mf_:.2f}")
    return dict(score=fit * mf_, fit=fit, match=pm, unwaived_score=float(gf * raw), gate=float(gate), items=items, findings=lines)


def grade_clip(actions, dt, action, prof, allow=()):
    """Report card of one clip as an example of `action`: score, findings, and which action it most resembles."""
    v = mf.extract(actions, dt); r = grade_features(v, action, prof, allow); rec = prof.recognise(v)
    r["recognised_as"] = rec[0][0]; r["recognised"] = rec[0][0] in family(action); r["rank"] = next(i for i, (a, _) in enumerate(rec) if a == action) + 1 if action in prof.p else None; r["ranking"] = rec[:3]; r["features"] = v
    return r


def split_clips(z):
    A, off = z["actions"], z["offsets"]; return [A[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def grade_dataset(clips, dt, action, prof, allow=(), real_F=None, real_action=None):
    """Judge a whole dataset (list of [F,30] arrays) as examples of `action`. Returns a dict with the verdict and everything behind it."""
    if action not in prof.p:
        raise KeyError(f"unknown action '{action}'. Known: {', '.join(prof.p)}")
    cards = [grade_clip(c, dt, action, prof, allow) for c in clips]; ok = [c for c in cards if c["score"] is not None]
    if not ok:
        return dict(action=action, n=len(clips), verdict="FAIL", reason="no clip could be measured (too short?)", cards=cards)
    sc = np.array([c["score"] for c in ok]); FM = np.array([c["features"] for c in ok]); p = prof.p[action]
    recognised = float(np.mean([c["recognised"] for c in ok])); top3 = float(np.mean([c["rank"] <= 3 for c in ok]))
    confused = {}
    for c in ok:
        if not c["recognised"]:
            confused[c["recognised_as"]] = confused.get(c["recognised_as"], 0) + 1
    # variety: spread of the dataset against the spread of the real examples, over the measures that matter for this action; and near-duplicates
    realF = prof.F[prof.action == action]; ratios = []
    for j in range(len(mf.NAMES)):
        if p["w"][j] >= 1.0 and np.isfinite(p["med"][j]) and mf.NAMES[j] not in mf.GATES:
            a_, b_ = FM[:, j][np.isfinite(FM[:, j])], realF[:, j][np.isfinite(realF[:, j])]
            if len(a_) > 2 and len(b_) > 2:
                iq = lambda v: np.subtract(*np.percentile(v, [90, 10])); ratios.append(iq(a_) / (iq(b_) + mf.FLOOR[j]))
    variety = float(np.clip(np.median(ratios), 0, 1.5)) if ratios else np.nan
    Z = (np.nan_to_num((FM - p["med"]) / p["scale"]))[:, p["w"] >= 1.0]; dup = 0
    if len(Z) > 1:
        idx = np.random.default_rng(0).choice(len(Z), min(len(Z), 400), replace=False); Zs = Z[idx]; D = np.linalg.norm(Zs[:, None] - Zs[None], axis=2) / np.sqrt(max(1, Zs.shape[1])); np.fill_diagonal(D, np.inf)
        dup = float(np.mean(D.min(1) < 0.03))
    # which measures are off for the dataset as a whole
    off = []; waived = set()
    for g in allow:
        waived |= {n for n in mf.NAMES if mf.GROUP[n] == g} if g in mf.GROUP_NAMES else {g}
    waived -= set(mf.GATES)
    for j, n in enumerate(mf.NAMES):
        col = FM[:, j][np.isfinite(FM[:, j])]
        if len(col) < 3 or not np.isfinite(p["med"][j]) or p["w"][j] <= 0 or n in waived:
            continue
        s = trapezoid(float(np.median(col)), *p["typ"][j], *p["tol"][j])
        if s < 0.75:
            m = float(np.median(col)); lo, hi = p["typ"][j]; off.append((float(p["w"][j] * (1 - s)), n, m, lo, hi, mf.UNIT[n], mf.TEXT[n][0] if m < lo else mf.TEXT[n][1]))
    off.sort(reverse=True)
    res = dict(action=action, n=len(clips), measured=len(ok), mean=float(sc.mean()), median=float(np.median(sc)), p10=float(np.percentile(sc, 10)), passing=float(np.mean(sc >= PASS_SCORE)),
               recognised=recognised, top3=top3, confused=dict(sorted(confused.items(), key=lambda kv: -kv[1])), variety=variety, duplicates=dup, off=[o[1:] for o in off[:8]], cards=cards,
               real_n=int(p["n"]), real_performers=None)
    fails = []
    if res["mean"] < VERDICT["score"]: fails.append(f"mean score {res['mean']:.0f} < {VERDICT['score']:.0f}")
    if res["passing"] < VERDICT["passing"]: fails.append(f"only {100 * res['passing']:.0f}% of clips score >= {PASS_SCORE:.0f}")
    if recognised < VERDICT["recognised"]: fails.append(f"only {100 * recognised:.0f}% of clips look like '{action}' (rest look like {', '.join(list(res['confused'])[:3]) or 'nothing'})")
    if np.isfinite(variety) and variety < VERDICT["variety"]: fails.append(f"clips are too alike (spread {100 * variety:.0f}% of real)")
    res["problems"] = fails
    res["verdict"] = "PASS" if not fails else ("MARGINAL" if len(fails) == 1 and res["mean"] >= 60 else "FAIL")
    return res


def report(r, verbose=False, top=5):
    a = r["action"]
    if "mean" not in r:
        print(f"\n== {a}: {r['verdict']}  {r.get('reason', '')}"); return
    print(f"\n== {a}: {r['verdict']}   ({r['n']} clips vs {r['real_n']} real examples)")
    print(f"   score        mean {r['mean']:.0f}   median {r['median']:.0f}   10th percentile {r['p10']:.0f}   clips scoring >= {PASS_SCORE:.0f}: {100 * r['passing']:.0f}%")
    print(f"   recognised   {100 * r['recognised']:.0f}% of clips match '{a}' best (top-3: {100 * r['top3']:.0f}%)" + (f"; the rest look like: " + ", ".join(f"{k} x{v}" for k, v in list(r['confused'].items())[:4]) if r["confused"] else ""))
    print(f"   variety      spread {100 * r['variety']:.0f}% of the real examples' spread, {100 * r['duplicates']:.0f}% near-duplicate clips" if np.isfinite(r["variety"]) else "   variety      n/a")
    for pr in r["problems"]:
        print(f"   PROBLEM: {pr}")
    if r["off"]:
        print("   what is off across the dataset (median of your clips vs the range real examples fall in):")
        for n, m, lo, hi, u, why in r["off"][:top]:
            print(f"     {n:20s} {m:8.3g} {u:6s} real {lo:.3g}..{hi:.3g}   {why}")
    if verbose:
        order = sorted(range(len(r["cards"])), key=lambda i: (r["cards"][i]["score"] if r["cards"][i]["score"] is not None else -1))
        print("   weakest clips:")
        for i in order[:top]:
            c = r["cards"][i]; print(f"     clip {i}: {c['score'] if c['score'] is None else round(c['score'])}/100, looks like {c['recognised_as']}" + ("".join("\n        - " + l for l in c["findings"][:3])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?"); ap.add_argument("--action", help="judge every clip as this action (default: the file's own per-clip `action` labels)")
    ap.add_argument("--real", default=REAL_DEFAULT); ap.add_argument("--allow", nargs="*", default=[], help="groups/measures that may deviate: " + " ".join(mf.GROUP_NAMES))
    ap.add_argument("--clips", type=int, nargs="+", help="only these clip indices"); ap.add_argument("--verbose", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args(); prof, z0, F0 = get_profiles(a.real)
    if a.list or not a.npz:
        perf = {}
        for act, pf in zip(z0["action"], z0["performer"]):
            perf.setdefault(act, set()).add(pf)
        print(f"{'action':12s}{'real clips':>11s}{'performers':>12s}   most telling measures")
        for n in prof.names:
            p = prof.p[n]; top = np.argsort(-p["w"])[:4]; print(f"{n:12s}{p['n']:11d}{len(perf[n]):12d}   " + ", ".join(mf.NAMES[j] for j in top))
        return
    z = np.load(a.npz, allow_pickle=True); clips = split_clips(z); dt = float(z["dt"]); labels = [str(x) for x in z["action"]] if "action" in z.files else None
    if a.clips:
        clips = [clips[i] for i in a.clips]; labels = [labels[i] for i in a.clips] if labels else None
    if a.action:
        groups = {a.action: clips}
    elif labels:
        groups = {}
        for c, l in zip(clips, labels):
            groups.setdefault(l, []).append(c)
    else:
        sys.exit("the file has no per-clip `action` labels: pass --action NAME")
    out = {}
    for act, cl in groups.items():
        if act not in prof.p:
            print(f"\n== {act}: not graded (no real examples of '{act}'; known: {', '.join(prof.names)})"); continue
        r = grade_dataset(cl, dt, act, prof, a.allow); report(r, a.verbose); out[act] = {k: v for k, v in r.items() if k != "cards"}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)
    bad = [k for k, v in out.items() if v["verdict"] == "FAIL"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
