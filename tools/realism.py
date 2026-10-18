#!/usr/bin/env python3
"""
realism.py - how HUMAN does a walking clip move? (the engine's verdict only answers "is it physically possible?")

  python tools/realism.py data/clips.npz --clips 0 1 2              # grade engine/generator clips
  python tools/realism.py real.npz --labels 07_01 20_08             # grade real mocap converted with bvh_to_clip.py
  python tools/realism.py data/clips.npz --allow zombie             # do not punish the deviations a request asked for

From Python:
  import realism
  r = realism.grade_clip(actions, dt, angles=None, allow=())       # r["score"] 0-100, r["items"], r["findings"]

How it judges: it measures ~25 things about the walk (timing, step geometry, pelvis bounce and sway, arm swing and its coordination with the
legs, knee and elbow bend, smoothness) and compares each with the range healthy adults fall in (from the gait-analysis literature; the table
RULES below says which range and why). Each measure scores 1 inside the typical range and falls to 0 at the limit of the tolerable range.
The overall score is the weighted mean x 100. Nothing here is learned, so it is only as good as those ranges: validate it on real captures
(--validate) before trusting it, and recalibrate the ranges from real data when you have more of it.

Intent: `allow` stops the listed measures from costing points (they earn full credit) without hiding them. A zombie request should not be punished
for a stiff, asymmetric, leaning, slow gait, so allow=GROUPS["zombie"] still reports those numbers but only what must stay human (smooth, no jitter,
no slamming feet) can lower the score. The more you allow, the less the score says: with the zombie preset it is close to a smoothness check.
Limits you should know: it judges steady walking (it needs a few steps), it cannot see faces or hands, and "human-like" here means
"inside the normal range of walking", so a graceful, quirky or very tired walk can score lower than it deserves; that is what `allow` is for.
"""
import argparse, os, sys
import numpy as np

# name: (typical lo, typical hi, tolerable lo, tolerable hi, weight, unit, group, meaning if too low, meaning if too high)
RULES = {
    "speed_mps":        (0.55, 1.90, 0.25, 2.5, 1.0, "m/s", "tempo", "very slow", "running speed, not a walk"),
    "walk_ratio":       (0.0055, 0.0075, 0.0035, 0.0100, 2.0, "m per step/min", "tempo", "short quick steps (shuffling)", "long slow steps (loping)"),
    "step_time_cv":     (0.0, 0.06, 0.0, 0.22, 1.5, "", "rhythm", "", "uneven rhythm"),
    "step_time_asym":   (0.0, 0.06, 0.0, 0.25, 1.0, "", "asymmetry", "", "left and right steps take different times (limp)"),
    "step_len_asym":    (0.0, 0.10, 0.0, 0.35, 1.0, "", "asymmetry", "", "left and right steps have different lengths"),
    "stance_pct":       (48.0, 66.0, 38.0, 76.0, 1.0, "% of cycle", "asymmetry", "feet leave the ground too early (bouncy)", "feet stay down too long (shuffle)"),
    "double_support":   (0.0, 0.30, 0.0, 0.50, 0.7, "of time", "tempo", "", "both feet down too often (very cautious)"),
    "step_width_m":     (0.03, 0.13, 0.0, 0.24, 1.0, "m", "asymmetry", "feet land on one line (tightrope)", "wide waddling stance"),
    "foot_clearance_m": (0.08, 0.22, 0.03, 0.34, 1.0, "m", "knees", "feet barely leave the floor (shuffle)", "feet lifted very high (marching)"),
    "pelvis_bob_cm":    (2.0, 6.5, 0.4, 12.0, 2.0, "cm", "bounce", "the body glides instead of bobbing (no weight shift in height)", "bouncing up and down too much"),
    "bounces_per_step": (0.8, 1.3, 0.4, 1.8, 1.0, "", "bounce", "pelvis height does not follow the steps", "pelvis height bounces more than once per step"),
    "pelvis_sway_cm":   (2.5, 8.0, 0.3, 15.0, 1.5, "cm", "bounce", "no sideways weight shift over the stance foot", "swaying side to side too much"),
    "arm_swing_cm":     (10.0, 60.0, 2.0, 100.0, 1.5, "cm", "arms", "arms hang still", "arms flail"),
    "arm_leg_coord":    (0.5, 1.0, -0.3, 1.0, 2.0, "corr", "arms", "arms are not swinging opposite to the legs", ""),
    "neck_bob_ratio":   (0.4, 1.1, 0.0, 1.7, 0.8, "", "posture", "", "the head bounces more than the pelvis (nothing absorbs the bob)"),
    "trunk_lean_deg":   (-2.0, 9.0, -8.0, 22.0, 1.0, "deg", "posture", "leaning back", "leaning forward"),
    "heading_err_deg":  (0.0, 12.0, 0.0, 50.0, 1.0, "deg", "tempo", "", "body does not face where it is going"),
    "smoothness_hf":    (0.0, 0.02, 0.0, 0.15, 2.0, "", "smooth", "", "jittery: high-frequency motion a body does not make"),
    "touchdown_speed":  (0.0, 0.8, 0.0, 2.5, 1.0, "m/s", "smooth", "", "feet slam down"),
    "knee_swing_deg":   (45.0, 75.0, 25.0, 100.0, 2.0, "deg", "knees", "legs swing through almost straight (stiff)", "knees fold too far"),
    "knee_stance_deg":  (8.0, 30.0, 0.0, 52.0, 2.0, "deg", "knees", "knees stay locked straight in stance (robotic)", "knees buckle in stance"),
    "elbow_flex_deg":   (5.0, 50.0, 0.0, 100.0, 0.8, "deg", "arms", "arms straight as sticks", "arms held bent"),
}
# what a stylised request may legitimately change (pass to allow=): a zombie is slow, stiff, asymmetric, leaning, arms out of step...
GROUPS = {
    "zombie": ["tempo", "asymmetry", "arms", "posture", "knees", "rhythm", "bounce"],
    "tired": ["tempo", "arms", "posture", "bounce"],
    "limp": ["asymmetry", "rhythm", "tempo"],
    "drunk": ["asymmetry", "rhythm", "tempo", "posture", "bounce", "arms"],
    "robot": ["knees", "arms", "bounce", "posture"],
    "none": [],
}


# Gates: some things are not "a bit worse", they make a clip unusable whatever else is right. Jitter (motion a body cannot make) caps the score:
# final = weighted mean x (0.35 + 0.65 x worst gate score), so a fully failed gate leaves at most 35% of the points. Presets never waive a gate.
GATES = ("smoothness_hf",)
GATE_FLOOR = 0.35


def trapezoid(v, tlo, thi, lo, hi):
    if v is None or not np.isfinite(v):
        return None
    if tlo <= v <= thi:
        return 1.0
    if v < tlo:
        return float(np.clip((v - lo) / (tlo - lo), 0, 1)) if tlo > lo else 0.0
    return float(np.clip((hi - v) / (hi - thi), 0, 1)) if hi > thi else 0.0


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _ma(x, w):
    """Centred moving average along the first axis (edges held)."""
    w = max(1, int(w) | 1); k = np.ones(w) / w; pad = [(w // 2, w // 2)] + [(0, 0)] * (x.ndim - 1); p = np.pad(x, pad, mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, "valid"), 0, p)


def features(a, dt, angles=None):
    """Measure the walk. Returns {metric: value or None}, plus '_info' with what was found."""
    a = np.asarray(a, np.float64); F = len(a); out = {k: None for k in RULES}; info = dict(frames=F)
    pel = a[:, [0, 1, 2]]; sp = np.r_[0, np.linalg.norm(np.diff(pel[:, [0, 2]], axis=0), axis=1)] / dt; sps = _ma(sp, int(round(0.4 / dt)))
    runs = [r for r in _runs(sps > 0.3) if r[1] - r[0] > 0.8 / dt]
    if not runs:
        info["note"] = "no walking segment (the body barely moves)"; out["_info"] = info; return out
    i0, i1 = max(runs, key=lambda r: r[1] - r[0]); n = i1 - i0
    seg = slice(i0, i1); info.update(segment=(int(i0), int(i1)), seconds=n * dt)
    cL, cR = a[:, 8] > 0.5, a[:, 13] > 0.5; sole = (a[:, [4, 5, 6]], a[:, [9, 10, 11]]); con = (cL, cR)
    # direction of travel: from the path averaged over about a stride, NOT from the raw pelvis, which sways sideways by several cm and would tilt the axis by 10-20 degrees
    ps = _ma(pel[:, [0, 2]], int(round(1.0 / dt))); h = max(1, int(round(0.4 / dt))); ahead = np.roll(ps, -h, 0) - np.roll(ps, h, 0); ahead[:h] = ahead[h]; ahead[-h:] = ahead[-h - 1]
    d = ahead / (np.linalg.norm(ahead, axis=1, keepdims=True) + 1e-9); perp = np.stack([d[:, 1], -d[:, 0]], 1)
    # gait events inside the segment: heel strikes (contact onsets) per foot
    hs = {s: [i for i in np.where(np.diff(con[s].astype(int)) == 1)[0] + 1 if i0 <= i < i1] for s in (0, 1)}
    events = sorted([(i, s) for s in (0, 1) for i in hs[s]]); info["steps"] = len(events)
    speed = float(sp[seg].mean()); out["speed_mps"] = speed
    out["heading_err_deg"] = float(np.degrees(np.abs((a[seg, 3] - np.arctan2(d[seg, 0], d[seg, 1]) + np.pi) % (2 * np.pi) - np.pi)).mean())
    if len(events) >= 3:
        iv = np.diff([e[0] for e in events]) * dt; step_t = float(iv.mean()); cad = 60.0 / step_t
        out["step_time_cv"] = float(iv.std() / iv.mean())
        alt = [iv[k] for k in range(len(iv)) if events[k][1] == 0]; alt2 = [iv[k] for k in range(len(iv)) if events[k][1] == 1]
        if alt and alt2:
            out["step_time_asym"] = float(abs(np.mean(alt) - np.mean(alt2)) / iv.mean())
        L, W, side_len = [], [], {0: [], 1: []}
        for i, s in events:
            o = 1 - s; v = sole[s][i][[0, 2]] - sole[o][i][[0, 2]]; ln = float(v @ d[i]); wd = abs(float(v @ perp[i]))
            if ln > 0:
                L.append(ln); side_len[s].append(ln)
            W.append(wd)
        if L:
            out["walk_ratio"] = float(np.mean(L) / cad); info["step_length_m"] = float(np.mean(L)); info["cadence_spm"] = cad
        if side_len[0] and side_len[1]:
            out["step_len_asym"] = float(abs(np.mean(side_len[0]) - np.mean(side_len[1])) / np.mean(L))
        if W:
            out["step_width_m"] = float(np.median(W))
        st = []
        for s in (0, 1):
            h_ = hs[s]
            for k in range(len(h_) - 1):
                off = np.where(~con[s][h_[k]:h_[k + 1]])[0]
                if len(off):
                    st.append(off[0] / (h_[k + 1] - h_[k]))
        if st:
            out["stance_pct"] = float(100 * np.mean(st))
        out["double_support"] = float((cL[seg] & cR[seg]).mean())
        # bounce: pelvis height and sideways sway, detrended with a moving average one stride long
        w = int(round(2 * step_t / dt)); trim = max(1, w // 2)
        xz = _ma(pel[seg][:, [0, 2]], 3); dev = xz - _ma(xz, w); lat = (dev * perp[seg]).sum(1)                 # sideways deviation from the stride-averaged path
        yb0 = _ma(pel[seg, 1], 3); ver = yb0 - _ma(yb0, w)
        ok_ = slice(trim, -trim) if len(lat) > 2 * trim + 3 else slice(None)
        out["pelvis_bob_cm"] = float(np.ptp(ver[ok_]) * 100); out["pelvis_sway_cm"] = float(np.ptp(lat[ok_]) * 100)
        yb = _ma(pel[seg, 1], 3); yd = (yb - _ma(yb, w))[trim:-trim] if len(yb) > 2 * trim + 3 else yb - yb.mean()
        if len(yd) > 5 and np.ptp(yd) > 1e-3:
            pk = [j for j in range(1, len(yd) - 1) if yd[j] >= yd[j - 1] and yd[j] > yd[j + 1] and yd[j] > yd.min() + 0.5 * np.ptp(yd)]
            keep = []
            for j in pk:
                if not keep or (j - keep[-1]) * dt > 0.6 * step_t:
                    keep.append(j)
                elif yd[j] > yd[keep[-1]]:
                    keep[-1] = j
            out["bounces_per_step"] = float(len(keep) / (len(yd) * dt / step_t))
        neck = a[:, [26, 27, 28]]; nbx = _ma(neck[seg, 1], 3); nd = (nbx - _ma(nbx, w))[trim:-trim] if len(nbx) > 2 * trim + 3 else nbx - nbx.mean()
        if np.ptp(yd) > 1e-3:
            out["neck_bob_ratio"] = float(np.ptp(nd) / np.ptp(yd))
        # arms: fore-aft excursion of the wrists relative to the pelvis, and their coordination with the legs
        fa = lambda p: ((p[:, [0, 2]] - pel[:, [0, 2]]) * d).sum(1)
        wl, wr = fa(a[:, [14, 15, 16]])[seg], fa(a[:, [18, 19, 20]])[seg]
        amp = float(0.5 * (np.ptp(_ma(wl, 3)) + np.ptp(_ma(wr, 3))) * 100); out["arm_swing_cm"] = amp
        fl, fr = fa(sole[0])[seg], fa(sole[1])[seg]
        if amp > 3.0 and fl.std() > 1e-3 and wl.std() > 1e-4 and wr.std() > 1e-4:
            out["arm_leg_coord"] = float(0.5 * (np.corrcoef(wl, fr)[0, 1] + np.corrcoef(wr, fl)[0, 1]))
        # touchdown speed: how fast a foot is still falling just before it lands
        td = [max(0.0, (sole[s][i - 4, 1] - sole[s][i - 1, 1]) / (3 * dt)) for i, s in events if i >= 4]
        if td:
            out["touchdown_speed"] = float(np.median(td))
        sw = []
        for s in (0, 1):
            for r0, r1 in _runs(~con[s]):
                if i0 <= r0 and r1 <= i1 and r1 - r0 > 3:
                    sw.append(float(sole[s][r0:r1, 1].max()))
        if sw:
            out["foot_clearance_m"] = float(np.median(sw))
    # smoothness: share of the velocity power above 6 Hz (a moving body has almost none)
    chans = [pel[seg], a[seg][:, [26, 27, 28]], a[seg][:, [14, 15, 16]], a[seg][:, [18, 19, 20]]]
    fr_ = []
    for c in chans:
        for k in range(3):
            v = np.gradient(c[:, k], dt); v = v - v.mean()
            if len(v) < 8 or v.std() < 1e-6:
                continue
            P = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2; f = np.fft.rfftfreq(len(v), dt); fr_.append(P[f > 6.0].sum() / (P.sum() + 1e-12))
    if fr_:
        out["smoothness_hf"] = float(np.median(fr_))
    neck = a[seg][:, [26, 27, 28]] - pel[seg]; lean = np.degrees(np.arctan2((neck[:, [0, 2]] * d[seg]).sum(1), neck[:, 1])); out["trunk_lean_deg"] = float(np.median(lean))
    if angles is not None:
        g = np.asarray(angles, np.float64); kn = [g[:, 0], g[:, 1]]; swing, stance = [], []
        for s in (0, 1):
            for r0, r1 in _runs(~con[s]):
                if i0 <= r0 and r1 <= i1 and r1 - r0 > 3:
                    swing.append(kn[s][r0:r1].max())
            for r0, r1 in _runs(con[s]):
                if i0 <= r0 and r1 <= i1 and r1 - r0 > 3:
                    stance.append(kn[s][r0:r0 + max(2, int(0.4 * (r1 - r0)))].max())          # loading response; the toe-off bend belongs to the swing
        if swing:
            out["knee_swing_deg"] = float(np.median(swing))
        if stance:
            out["knee_stance_deg"] = float(np.median(stance))
        out["elbow_flex_deg"] = float(np.mean(g[seg][:, [2, 3]]))
    out["_info"] = info
    return out


def grade(feat, allow=()):
    """Score the measured features. allow = metric names or group names whose deviation the request asked for."""
    allowed = set()
    for x in allow:
        allowed |= {k for k, r in RULES.items() if r[6] == x} if x in {r[6] for r in RULES.values()} else ({x} if x in RULES else set())
    items, num, den = [], 0.0, 0.0
    for k, (tlo, thi, lo, hi, wt, unit, grp, mlo, mhi) in RULES.items():
        v = feat.get(k); sc = trapezoid(v, tlo, thi, lo, hi)
        if v is None:
            items.append(dict(name=k, value=None, score=None, weight=wt, unit=unit, group=grp, typical=(tlo, thi), allowed=k in allowed, note="not measurable on this clip")); continue
        note = "" if sc >= 0.999 else (mlo if v < tlo else mhi)
        items.append(dict(name=k, value=float(v), score=sc, weight=wt, unit=unit, group=grp, typical=(tlo, thi), allowed=k in allowed, note=note))
        num += wt * (1.0 if k in allowed else sc); den += wt          # an allowed measure is not punished: it earns full credit (never lowers the score)
    items_ok = [i for i in items if i["score"] is not None]
    gate = min([i["score"] for i in items_ok if i["name"] in GATES and i["name"] not in allowed] or [1.0]); gf = GATE_FLOOR + (1 - GATE_FLOOR) * gate
    findings = sorted([i for i in items_ok if i["score"] < 0.75], key=lambda i: -(i["weight"] * (1 - i["score"])))
    lines = []
    for i in findings:
        lo, hi = i["typical"]; tag = " (allowed by the request)" if i["allowed"] else ""
        lines.append(f"{i['name']} = {i['value']:.3g} {i['unit']} (human {lo:g}-{hi:g}): {i['note']}{tag}")
    raw = 100 * sum(i["weight"] * i["score"] for i in items_ok) / max(1e-9, sum(i["weight"] for i in items_ok))
    if gate < 0.999:
        lines.insert(0, f"GATE: motion is not smooth enough to be a body's (worst gate score {gate:.2f}); the score is capped at {100 * gf:.0f}% of what the other measures would give")
    return dict(score=float(gf * 100 * num / den) if den else None, human_likeness=float(gf * raw), gate=float(gate), items=items, findings=lines, measured=len(items_ok), of=len(items))


def grade_clip(actions, dt, angles=None, allow=()):
    f = features(actions, dt, angles); r = grade(f, allow); r["info"] = f.get("_info", {}); return r


def angles_from_engine(lib, actions, dt):
    """Knee / elbow flexion and trunk lean of the BODY THE ENGINE ACTUALLY DRAWS for this clip (replayed through its tracking)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import view_clip, gen_dataset as g
    rp = view_clip.replay(lib, actions, dt); b = g.new_standing(lib); nm = {n: i for i, n in enumerate(b.names)}; b.close()
    B = rp["bones"].astype(np.float64)
    seg = lambda n: B[:, nm[n], 3:6] - B[:, nm[n], 0:3]
    def fl(u, v):
        cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9); return np.degrees(np.arccos(np.clip(cs, -1, 1)))
    return np.stack([fl(seg("thigh_L"), seg("shin_L")), fl(seg("thigh_R"), seg("shin_R")), fl(seg("upper_arm_L"), seg("forearm_L")), fl(seg("upper_arm_R"), seg("forearm_R")), np.zeros(len(B))], 1)


def card(name, r):
    s = r["score"]; print(f"\n{name}: realism {s:.0f}/100" + (f"  (all measures: {r['human_likeness']:.0f})" if abs(r['human_likeness'] - s) > 0.5 else "") + f"  [{r['measured']}/{r['of']} measures, {r['info'].get('steps', 0)} steps, {r['info'].get('seconds', 0):.1f} s walking]")
    for l in r["findings"][:6]:
        print("   - " + l)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz"); ap.add_argument("--clips", type=int, nargs="+"); ap.add_argument("--labels", nargs="+"); ap.add_argument("--allow", nargs="*", default=[], help="groups/metrics, or a preset: " + ", ".join(GROUPS))
    ap.add_argument("--lib"); ap.add_argument("--repo"); ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(); z = np.load(a.npz, allow_pickle=True); acts, off, dt = z["actions"], z["offsets"], float(z["dt"]); N = len(off) - 1
    allow = []
    for x in a.allow:
        allow += GROUPS.get(x, [x])
    ids = a.clips if a.clips else ([i for i, l in enumerate(z["labels"]) if l in a.labels] if a.labels and "labels" in z else list(range(min(N, 8))))
    lib = None
    for i in ids:
        clip = acts[off[i]:off[i + 1]]; ang = None
        if "angles" in z:
            ang = z["angles"][off[i]:off[i + 1]]
        else:
            if lib is None:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gen_dataset as g; lib = g.Lib(a.lib, a.repo, build=not a.no_build)
            ang = angles_from_engine(lib, clip, dt)
        card(f"clip {z['labels'][i] if 'labels' in z else i}", grade_clip(clip, dt, ang, allow))


if __name__ == "__main__":
    main()
