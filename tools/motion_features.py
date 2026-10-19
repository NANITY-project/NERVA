#!/usr/bin/env python3
"""
motion_features.py - ~40 numbers that describe HOW a body moves, for any action (not just walking).

  import motion_features as mf
  v = mf.extract(actions, dt)          # actions [F,30] (the engine's action vector), dt seconds per frame -> float array in mf.NAMES order (nan = undefined)
  mf.table(v)                          # the same as a readable list

The old grader measured gait events (heel strikes, step width, arm swing) which only exist while walking. These features are defined for a sitting,
standing, jumping or dancing body too, and they come ONLY from the 30-number action vector, so a real motion-capture clip and an engine/generator clip
are measured by exactly the same code (no side channel such as joint angles that one of them lacks):
  height   pelvis / neck height: how low does it get, how much does it move up and down
  trunk    lean forward / sideways, trunk length
  legs     how straight the legs are (pelvis-to-ankle distance over leg length: 1 = straight, ~0.6 = deep squat), left/right difference
  feet     planted / airborne shares, spread, lift, sliding of a planted foot, steps per second
  speed    travel speed, acceleration, vertical speed, turning, forward/backward/sideways travel
  hands    hand speed and reach relative to the pelvis, hands to face, one hand vs two
  rhythm   how repetitive the motion is, how still the body is
  smooth   jitter: velocity power above 6 Hz in m/s (a body cannot produce it, a noisy generator does)
All lengths are metres on the engine-sized skeleton (leg 0.87 m), all rates per second, so clips of different length are comparable.
"""
import numpy as np

ANKLE_H = 0.08          # ankle joint height above the sole (engine skeleton)
LEG = 0.87              # pelvis to ankle when standing straight

# name, unit, group, floor (the smallest difference worth caring about: stops a perfectly constant real feature from getting a zero-width range),
# text when the clip is LOWER than real examples, text when it is HIGHER
FEATURES = [
    ("pelvis_h_med", "m", "height", 0.03, "pelvis stays lower than in real examples", "pelvis stays higher than in real examples"),
    ("pelvis_h_min", "m", "height", 0.04, "body goes lower than real examples do", "body never gets as low as real examples (does it reach the seat / the ground?)"),
    ("pelvis_h_max", "m", "height", 0.04, "body never rises as high as real examples", "body rises higher than real examples"),
    ("pelvis_h_range", "m", "height", 0.04, "pelvis height barely changes where real examples go up/down", "pelvis goes up and down much more than real examples"),
    ("neck_h_med", "m", "height", 0.04, "neck/head carried lower than real", "neck/head carried higher than real"),
    ("neck_h_min", "m", "height", 0.05, "neck gets lower than real examples", "neck never gets as low as real examples"),
    ("trunk_len_med", "m", "trunk", 0.03, "trunk is shorter than a real one (bent over / collapsed)", "trunk is longer than a real one (stretched)"),
    ("trunk_lean_med", "deg", "trunk", 3.0, "leans back more than real examples", "leans forward more than real examples"),
    ("trunk_lean_absmax", "deg", "trunk", 4.0, "trunk stays more upright than real examples", "trunk bends/leans further than real examples"),
    ("trunk_side_absmax", "deg", "trunk", 3.0, "trunk stays more upright sideways", "trunk leans sideways more than real examples"),
    ("leg_ext_med", "0-1", "legs", 0.03, "legs bent more than real examples", "legs straighter than real examples"),
    ("leg_ext_min", "0-1", "legs", 0.05, "legs fold deeper than real examples", "legs never bend as much as real examples (knees too stiff)"),
    ("leg_ext_asym", "0-1", "legs", 0.03, "left and right leg behave more alike than real", "one leg bent/lifted much more than the other"),
    ("contact_both", "share", "feet", 0.05, "both feet planted less often than real", "both feet planted more often than real"),
    ("contact_none", "share", "feet", 0.05, "feet in the air less often than real", "feet in the air more often than real"),
    ("foot_sep_med", "m", "feet", 0.03, "feet closer together than real", "feet further apart than real"),
    ("foot_sep_p95", "m", "feet", 0.04, "feet never spread as far as real", "feet spread further than real"),
    ("foot_fwd_p95", "m", "feet", 0.04, "no fore-aft foot separation (steps too short)", "feet more staggered than real"),
    ("foot_lift_p95", "m", "feet", 0.03, "feet barely leave the floor", "feet lifted higher than real"),
    ("planted_slide", "m/s", "feet", 0.03, "", "planted feet slide (skating)"),
    ("step_rate", "1/s", "rhythm", 0.15, "fewer foot touchdowns per second than real", "more foot touchdowns per second than real"),
    ("speed_mean", "m/s", "speed", 0.06, "travels slower than real examples", "travels faster than real examples"),
    ("speed_p95", "m/s", "speed", 0.10, "peak speed lower than real", "peak speed higher than real"),
    ("vert_speed_p95", "m/s", "speed", 0.06, "pelvis moves up/down more slowly than real", "pelvis moves up/down faster than real"),
    ("accel_p95", "m/s2", "speed", 0.40, "gentler accelerations than real", "harsher accelerations than real"),
    ("yaw_rate_mean", "deg/s", "turn", 6.0, "turns less than real examples", "turns more than real examples"),
    ("yaw_net_rate", "deg/s", "turn", 5.0, "net turning less than real", "net turning more than real"),
    ("fwd_frac", "-1..1", "speed", 0.10, "travels backwards/sideways relative to where it faces", "travels more straight ahead than real"),
    ("quiet_frac", "share", "rhythm", 0.05, "body is still less often than real", "body is still more often than real"),
    ("motion_energy", "m/s", "rhythm", 0.04, "whole body moves less than real", "whole body moves more than real"),
    ("periodicity", "0-1", "rhythm", 0.08, "motion is less repetitive than real", "motion is more repetitive than real"),
    ("hand_speed_mean", "m/s", "hands", 0.05, "hands move less than real", "hands move more than real"),
    ("hand_speed_p95", "m/s", "hands", 0.10, "hands never move as fast as real", "hands move faster than real"),
    ("hand_asym", "0-1", "hands", 0.06, "both hands move alike (real: one hand leads)", "one hand moves, the other does not (real: both)"),
    ("hand_h_max", "m", "hands", 0.05, "hands never rise as high (relative to pelvis) as real", "hands rise higher than real"),
    ("hand_h_min", "m", "hands", 0.05, "hands drop lower than real", "hands never drop as low as real (no reaching down)"),
    ("hand_fwd_p95", "m", "hands", 0.05, "hands never reach as far forward as real", "hands reach further forward than real"),
    ("hand_sep_med", "m", "hands", 0.04, "hands closer together than real", "hands further apart than real"),
    ("hand_neck_min", "m", "hands", 0.04, "hands come closer to the face than real", "hands never come as close to the face as real"),
    ("gaze_pitch_med", "deg", "trunk", 4.0, "looks further down than real", "looks further up than real"),
    ("jitter_hf", "m/s", "smooth", 0.01, "", "jittery: high-frequency motion a body does not make"),
    ("contact_mismatch", "share", "feet", 0.02, "", "contact flags disagree with the feet (flagged planted while in the air, or flat and still but flagged airborne)"),
]
NAMES = [f[0] for f in FEATURES]
INDEX = {n: i for i, n in enumerate(NAMES)}
UNIT = {f[0]: f[1] for f in FEATURES}
GROUP = {f[0]: f[2] for f in FEATURES}
FLOOR = np.array([f[3] for f in FEATURES], np.float64)
TEXT = {f[0]: (f[4], f[5]) for f in FEATURES}
GROUP_NAMES = ["height", "trunk", "legs", "feet", "rhythm", "speed", "turn", "hands", "smooth"]
GATES = ("jitter_hf", "contact_mismatch")           # a body cannot produce these: failing one caps the score whatever else is right
MIN_FRAMES = 12


def _ma(x, w):
    """Centred moving average along axis 0 (edges held)."""
    w = max(1, int(w) | 1)
    if w == 1 or len(x) < 2:
        return x.copy()
    k = np.ones(w) / w; pad = [(w // 2, w // 2)] + [(0, 0)] * (x.ndim - 1); p = np.pad(x, pad, mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, "valid"), 0, p)


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _pct(x, q):
    x = np.asarray(x, np.float64)
    return float(np.percentile(x, q)) if x.size else np.nan


def extract(actions, dt):
    """Feature vector (len(NAMES)) of one clip. nan where a feature cannot be measured (clip too short, no planted foot ...)."""
    a = np.asarray(actions, np.float64); F = len(a); out = np.full(len(NAMES), np.nan)
    if F < MIN_FRAMES or not np.all(np.isfinite(a)):
        return out
    put = lambda k, v: out.__setitem__(INDEX[k], float(v))
    S = lambda x, w=0.1: _ma(x, int(round(w / dt)))
    pel = a[:, 0:3]; yaw = np.unwrap(a[:, 3]); sL, sR = a[:, 4:7], a[:, 9:12]; cL, cR = a[:, 8] > 0.5, a[:, 13] > 0.5
    wL, wR = a[:, 14:17], a[:, 18:21]; look = a[:, 22:25]; neck = a[:, 26:29]; dur = F * dt
    hd = np.stack([np.sin(yaw), np.cos(yaw)], 1); lf = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
    # --- height
    put("pelvis_h_med", np.median(pel[:, 1])); put("pelvis_h_min", _pct(pel[:, 1], 2)); put("pelvis_h_max", _pct(pel[:, 1], 98))
    put("pelvis_h_range", _pct(pel[:, 1], 98) - _pct(pel[:, 1], 2)); put("neck_h_med", np.median(neck[:, 1])); put("neck_h_min", _pct(neck[:, 1], 2))
    # --- trunk
    tv = neck - pel; put("trunk_len_med", np.median(np.linalg.norm(tv, axis=1)))
    ty = np.where(np.abs(tv[:, 1]) < 1e-3, 1e-3, tv[:, 1])
    lean = np.degrees(np.arctan2((tv[:, [0, 2]] * hd).sum(1), ty)); side = np.degrees(np.arctan2((tv[:, [0, 2]] * lf).sum(1), ty))
    put("trunk_lean_med", np.median(lean)); put("trunk_lean_absmax", _pct(np.abs(lean), 95)); put("trunk_side_absmax", _pct(np.abs(side), 95))
    gv = look - neck; put("gaze_pitch_med", np.median(np.degrees(np.arctan2(gv[:, 1], np.hypot(gv[:, 0], gv[:, 2]) + 1e-6))))
    # --- legs
    up = np.array([0, ANKLE_H, 0]); eL = np.clip(np.linalg.norm(pel - (sL + up), axis=1) / LEG, 0, 1.2); eR = np.clip(np.linalg.norm(pel - (sR + up), axis=1) / LEG, 0, 1.2)
    em = 0.5 * (eL + eR); put("leg_ext_med", np.median(em)); put("leg_ext_min", _pct(em, 5)); put("leg_ext_asym", np.median(np.abs(eL - eR)))
    # --- feet
    put("contact_both", (cL & cR).mean()); put("contact_none", (~cL & ~cR).mean())
    sep = np.linalg.norm(sL[:, [0, 2]] - sR[:, [0, 2]], axis=1); put("foot_sep_med", np.median(sep)); put("foot_sep_p95", _pct(sep, 95))
    put("foot_fwd_p95", _pct(np.abs(((sL - sR)[:, [0, 2]] * hd).sum(1)), 95)); put("foot_lift_p95", _pct(np.maximum(sL[:, 1], sR[:, 1]), 95))
    sl = []
    for s, c in ((sL, cL), (sR, cR)):
        if c.sum() > 2:
            sv = np.gradient(S(s[:, [0, 2]], 0.07), dt, axis=0); sl.append(np.linalg.norm(sv, axis=1)[c])
    if sl:
        put("planted_slide", np.concatenate(sl).mean())
    bad = []
    for s_, c_ in ((sL, cL), (sR, cR)):
        fs = np.linalg.norm(np.gradient(S(s_[:, [0, 2]], 0.07), dt, axis=0), axis=1)
        bad.append((c_ & (s_[:, 1] > 0.04)) | (~c_ & (s_[:, 1] < 0.006) & (fs < 0.12)))
    put("contact_mismatch", np.mean(bad))
    put("step_rate", sum(int((np.diff(c.astype(int)) == 1).sum()) for c in (cL, cR)) / dur)
    # --- speed (smoothed over 0.1 s: mocap noise would otherwise dominate)
    ps = S(pel); v = np.gradient(ps, dt, axis=0); sp = np.hypot(v[:, 0], v[:, 2]); vy = v[:, 1]
    put("speed_mean", sp.mean()); put("speed_p95", _pct(sp, 95)); put("vert_speed_p95", _pct(np.abs(vy), 95))
    ps2 = S(pel, 0.15); acc = np.gradient(np.gradient(ps2, dt, axis=0), dt, axis=0); put("accel_p95", _pct(np.linalg.norm(acc, axis=1), 95))
    ys = S(yaw[:, None], 0.2)[:, 0]; put("yaw_rate_mean", np.degrees(np.abs(np.gradient(ys, dt))).mean()); put("yaw_net_rate", np.degrees(abs(yaw[-1] - yaw[0])) / dur)
    mv = sp > 0.2
    put("fwd_frac", float(np.mean((v[mv][:, [0, 2]] * hd[mv]).sum(1) / sp[mv])) if mv.sum() * dt > 0.3 else 0.0)
    # --- relative motion of hands / feet / neck against the pelvis
    rel = lambda q: S(q - pel)
    pts = [rel(wL), rel(wR), rel(sL), rel(sR), rel(neck)]; spd = [np.linalg.norm(np.gradient(p, dt, axis=0), axis=1) for p in pts]
    hsL, hsR = spd[0], spd[1]; allsp = np.mean(spd + [sp], axis=0)
    put("motion_energy", allsp.mean()); put("hand_speed_mean", 0.5 * (hsL.mean() + hsR.mean())); put("hand_speed_p95", _pct(np.maximum(hsL, hsR), 95))
    put("hand_asym", abs(hsL.mean() - hsR.mean()) / (hsL.mean() + hsR.mean() + 1e-6))
    put("quiet_frac", np.mean((sp < 0.15) & (np.abs(vy) < 0.15) & (np.maximum(hsL, hsR) < 0.4)))
    hl, hr = wL[:, 1] - pel[:, 1], wR[:, 1] - pel[:, 1]; put("hand_h_max", _pct(np.maximum(hl, hr), 95)); put("hand_h_min", _pct(np.minimum(hl, hr), 5))
    fw = lambda w: ((w - pel)[:, [0, 2]] * hd).sum(1); put("hand_fwd_p95", _pct(np.maximum(fw(wL), fw(wR)), 95))
    put("hand_sep_med", np.median(np.linalg.norm(wL - wR, axis=1))); put("hand_neck_min", _pct(np.minimum(np.linalg.norm(wL - neck, axis=1), np.linalg.norm(wR - neck, axis=1)), 5))
    # --- rhythm: strongest repeat of the whole-body speed signal at a lag of 0.3 s .. 2.5 s
    if dur >= 1.6:
        e = allsp - allsp.mean()
        if e.std() > 1e-4:
            n = len(e); f = np.fft.rfft(e, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac = ac / (ac[0] + 1e-12)
            lo, hi = int(round(0.3 / dt)), min(int(round(2.5 / dt)), n // 2)
            put("periodicity", float(np.clip(ac[lo:hi].max(), 0, 1)) if hi > lo + 1 else 0.0)
        else:
            put("periodicity", 0.0)
    # --- smoothness: RMS of the velocity above 6 Hz (m/s) over pelvis, neck and both wrists
    if dur >= 0.5:
        rms = []
        for ch in (pel, neck, wL, wR):
            for k in range(3):
                vv = np.gradient(ch[:, k], dt); vv = vv - np.polyval(np.polyfit(np.arange(F), vv, 1), np.arange(F))
                sp_ = np.fft.rfft(vv); fr = np.fft.rfftfreq(F, dt); sp_[fr <= 6.0] = 0; rms.append(float(np.sqrt(np.mean(np.fft.irfft(sp_, F) ** 2))))
        put("jitter_hf", np.mean(rms))
    return out


def table(v):
    return [(n, float(x), UNIT[n]) for n, x in zip(NAMES, v)]


if __name__ == "__main__":
    import sys
    z = np.load(sys.argv[1], allow_pickle=True); A, off, dt = z["actions"], z["offsets"], float(z["dt"])
    for i in range(min(3, len(off) - 1)):
        print(f"clip {i}"); [print(f"  {n:20s}{x:10.3f} {u}") for n, x, u in table(extract(A[off[i]:off[i + 1]], dt))]
