#!/usr/bin/env python3
"""
fit_torso_prior.py - derive the torso's JOINT LIMITS and POSE PRIOR from real motion capture, and write include/ik_torso_prior.hpp.

  python tools/fit_torso_prior.py                         # ~4 trials per action from the CMU database -> include/ik_torso_prior.hpp (downloads ~10 MB, cached)
  python tools/fit_torso_prior.py --per-action 8 --out /tmp/prior.hpp
  python tools/fit_torso_prior.py --report-only           # fit, print the report, write nothing

What it fits (all in the joint coordinates of tools/torso_math.py: flex = hinge, lat = side bend, twist, per joint, radians):
  * LIMITS  per joint, per coordinate: the range real people actually use (0.2 / 99.8 percentile, widened by 12 % + 2 degrees), never wider than an anatomical cap.
            Left/right (lat, twist) are symmetric. Lying on the floor (get_up) is left out: the engine's pelvis is upright, and a body that is lying down is not a torso pose.
  * PRIOR   a Gaussian over the joint coordinates. The MEAN is the natural standing / walking posture (a gentle S-curve of the spine: a few degrees per joint),
            adjusted so the trunk has zero net lean and the head is level, which are the engine's conventions. The COVARIANCE comes from all the motion in the corpus:
            it says how a person shares a bend among the joints and how the joints move together, which is what picks one pose out of the many that reach the same
            point. Spine: one 9x9 block (coupled chain); neck, head, clavicle: 3x3 each. Mirrored data are added, so the prior is left/right symmetric.
The corpus is split by PERFORMER: the fit uses some people, the report checks the limits and the prior on people the fit never saw.
Data: CMU Graphics Lab Motion Capture Database (BVH conversion by B. Hahne, mirrored at github.com/sfd158/cmu-mocap). Credit them.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_to_clip, get_real_motions, motion_actions as MA, torso_math as T

NEUTRAL_ACTIONS = ("stand", "walk", "turn", "balance", "gesture", "drink")        # ordinary upright behaviour: where the natural posture is read off
ENGINE_SPINE = (0.13, 0.15, 0.14)                                                 # the engine's spine bone lengths (ik_body.hpp, scale 1), for the zero-net-lean convention
# hard anatomical sanity caps (radians). The data decide the limits; these only stop a capture glitch from widening a limit past what a body can do.
CAP = np.radians(np.array([        # flex_min flex_max lat_max twist_max   (lat and twist are symmetric)
    [-45, 85, 45, 45], [-45, 85, 45, 55], [-45, 85, 45, 55], [-70, 80, 55, 80], [-60, 70, 50, 85], [-60, 60, 60, 60], [-60, 60, 60, 60]], np.float64))


def pick_trials(per_action, held_mod):
    trials, _ = MA.fetch_index(os.path.join("data", "cmu_cache")); got = MA.assign(trials); fit, held = [], []
    for name, items in got.items():
        for tid, _d in MA.spread_over_subjects(items, per_action):
            (held if int(MA.subject_of(tid)) % held_mod == 0 else fit).append((tid, name))
    return fit, held


def load_set(tids, cache, fps=30, max_secs=40):
    out = []
    for tid, act in tids:
        p = get_real_motions.download(tid, cache)
        if p is None:
            continue
        try:
            A, ang, tq, dt = bvh_to_clip.convert_full(p, fps)
        except Exception as e:
            print(f"  skip {tid}: {e}"); continue
        if tq is None:
            continue
        n = int(max_secs / dt); out.append((tid, act, T.local_coords(tq[:n].astype(np.float64))))
    return out


def mirror(C):
    """Reflect left/right: lat and twist change sign, flex does not, the two clavicles swap (and mirror too)."""
    M = C.copy(); M[..., 1:] *= -1; M[:, [5, 6]] = M[:, [6, 5]]; return M


def pctl(x, q):
    return np.percentile(x, q, axis=0)


def fit_limits(C):
    """C [F,7,3]. Returns L [7,6] = flex_min, flex_max, lat_min, lat_max, twist_min, twist_max (radians)."""
    Cm = np.concatenate([C, mirror(C)]); L = np.zeros((T.N, 6))
    for k in range(T.N):
        f = Cm[:, k, 0]; l = Cm[:, k, 1]; w = Cm[:, k, 2]
        lo, hi = pctl(f, 0.2), pctl(f, 99.8); pad = 0.12 * (hi - lo) + np.radians(2)
        L[k, 0] = max(lo - pad, CAP[k, 0]); L[k, 1] = min(hi + pad, CAP[k, 1])
        la = max(abs(pctl(l, 0.2)), abs(pctl(l, 99.8))); la = min(la + 0.12 * la + np.radians(2), CAP[k, 2]); L[k, 2], L[k, 3] = -la, la
        wa = max(abs(pctl(w, 0.2)), abs(pctl(w, 99.8))); wa = min(wa + 0.12 * wa + np.radians(2), CAP[k, 3]); L[k, 4], L[k, 5] = -wa, wa
    L[6] = L[5].copy(); L[6, 2:] = [-L[5, 3], -L[5, 2], -L[5, 5], -L[5, 4]]                         # the right clavicle is the mirror of the left one
    return L


def fit_prior(C, C_neutral):
    """Mean from the neutral frames, covariance from everything (both mirrored). Returns mean [7,3], spine precision [9,9], precision [4,3,3] (neck, head, clav L, clav R)."""
    Cm = np.concatenate([C, mirror(C)]); Nm = np.concatenate([C_neutral, mirror(C_neutral)])
    mean = np.median(Nm, axis=0); mean[:5, 1:] = 0.0                                               # natural posture is left/right symmetric (lat, twist 0) for the midline joints
    mean[5, 1:] = 0.5 * (mean[5, 1:] - mean[6, 1:] * np.array([1, 1]))                              # clavicles: mirror-average
    mean[5, 0] = mean[6, 0] = 0.5 * (mean[5, 0] + mean[6, 0]); mean[6, 1:] = -mean[5, 1:]
    # convention: zero net lean of the spine (the neck base is over the pelvis); the head is already calibrated per file (bvh_to_clip.torso_orientations)
    ph = np.cumsum(mean[:3, 0]); Ls = np.array(ENGINE_SPINE); d = 0.0
    for _ in range(60):
        g = (Ls * np.sin(ph + d)).sum(); h = (Ls * np.cos(ph + d)).sum(); d -= g / h
    mean[0, 0] += d
    floor = np.radians(1.0) ** 2                                                                    # variance floor: no joint is stiffer than 1 degree sd
    def prec(X):
        S = np.cov(X.T) + np.eye(X.shape[1]) * floor; return np.linalg.inv(S)
    # mirror symmetry decouples the sagittal coordinates (flex) from the frontal ones (lat, twist)
    X = Cm[:, :3, :].reshape(len(Cm), 9); S = np.cov(X.T) + np.eye(9) * floor
    sag = [0, 3, 6]; fro = [1, 2, 4, 5, 7, 8]; S2 = np.zeros_like(S)
    S2[np.ix_(sag, sag)] = S[np.ix_(sag, sag)]; S2[np.ix_(fro, fro)] = S[np.ix_(fro, fro)]
    P9 = np.linalg.inv(S2)
    P3 = np.zeros((4, 3, 3))
    for i, k in enumerate((3, 4, 5)):
        Xk = Cm[:, k, :] if k != 5 else np.concatenate([Cm[:, 5, :], Cm[:, 6, :] * np.array([1, -1, -1])])    # left clavicle pooled with the mirrored right one
        S = np.cov(Xk.T) + np.eye(3) * floor; S[0, 1:] = 0.0; S[1:, 0] = 0.0; P3[i] = np.linalg.inv(S)
    P3[3] = P3[2] * np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])                                 # right clavicle: the mirrored precision
    return mean, P9, P3


def coverage(C, L):
    """Percent of frames with every joint inside its limits, and per joint."""
    ok = np.ones(len(C), bool); per = []
    for k in range(T.N):
        f, l, w = C[:, k, 0], C[:, k, 1], C[:, k, 2]
        box = (f >= L[k, 0]) & (f <= L[k, 1]) & (l >= L[k, 2]) & (l <= L[k, 3]) & (w >= L[k, 4]) & (w <= L[k, 5]); ok &= box; per.append(100.0 * box.mean())
    return 100.0 * ok.mean(), per


def write_header(path, L, mean, P9, P3, info):
    def lit(v):                                                          # a valid C++ float literal: 0 -> "0.0f", 1e-05 -> "1e-05f"
        t = f"{float(v):.6g}"; t = "0" if t == "-0" else t
        return t + ("f" if any(c in t for c in ".en") else ".0f")
    f32 = lambda a: ", ".join(lit(v) for v in np.ravel(a))
    with open(path, "w") as h:
        h.write("// ik_torso_prior.hpp -- GENERATED by tools/fit_torso_prior.py. Do not edit by hand: re-run the script.\n")
        h.write(f"// Fitted on {info}\n// Joint order: spine1 spine2 spine3 neck head clavicle_L clavicle_R. Angles in radians (flex, lat, twist: see tools/torso_math.py).\n")
        h.write("#pragma once\nnamespace ik { namespace torso_prior {\n")
        h.write("// limits per joint: flex_min flex_max lat_min lat_max twist_min twist_max\n")
        h.write(f"constexpr float kLimits[7][6] = {{\n" + ",\n".join("    {" + f32(L[k]) + "}" for k in range(7)) + "};\n")
        h.write("// natural posture (the prior mean): flex lat twist per joint\n")
        h.write(f"constexpr float kMean[7][3] = {{\n" + ",\n".join("    {" + f32(mean[k]) + "}" for k in range(7)) + "};\n")
        h.write("// prior of the three spine joints together: index = 3 * joint + coordinate (joint 0..2 = spine1..3, coordinate 0 flex, 1 lat, 2 twist)\n")
        h.write("// covariance (used to place a bend where real people bend) and its inverse, the precision (used to score a pose)\n")
        h.write(f"constexpr float kSpineCovariance[9][9] = {{\n" + ",\n".join("    {" + f32(np.linalg.inv(P9)[i]) + "}" for i in range(9)) + "};\n")
        h.write(f"constexpr float kSpinePrecision[9][9] = {{\n" + ",\n".join("    {" + f32(P9[i]) + "}" for i in range(9)) + "};\n")
        h.write("// prior precision of neck, head, clavicle_L, clavicle_R (3x3 each, order flex, lat, twist)\n")
        h.write(f"constexpr float kJointPrecision[4][3][3] = {{\n" + ",\n".join("    {" + ", ".join("{" + f32(P3[j][r]) + "}" for r in range(3)) + "}" for j in range(4)) + "};\n")
        h.write("}} // namespace ik::torso_prior\n")


_PRIOR = {}


def load_prior(path=None):
    """The constants of include/ik_torso_prior.hpp as numpy arrays (what the C++ uses): limits [7,6], mean [7,3], spine_precision [9,9], joint_precision [4,3,3]."""
    import re
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "include", "ik_torso_prior.hpp")
    if path in _PRIOR:
        return _PRIOR[path]
    txt = open(path).read(); out = {}
    for name, key, shape in (("kLimits", "limits", (7, 6)), ("kMean", "mean", (7, 3)), ("kSpinePrecision", "spine_precision", (9, 9)), ("kJointPrecision", "joint_precision", (4, 3, 3))):
        body = re.search(name + r"\[[^=]*=\s*\{(.*?)\};", txt, re.S).group(1)
        out[key] = np.array([float(x.rstrip("f")) for x in re.findall(r"-?\d[\d.]*(?:e[-+]?\d+)?f", body)], np.float64).reshape(shape)
    _PRIOR[path] = out; return out


def torso_prior_limits():
    return load_prior()["limits"]


def pose_energy(C):
    """How typical a torso pose is under the prior: Mahalanobis distance squared per coordinate of the spine (9), neck (3) and head (3); about 1 for a real pose.
    C: [7,3] (flex, lat, twist) as from torso_math.local_coords. The clavicles are left out (the engine's is measured from its own rest)."""
    P = load_prior(); d = (C[:3] - P["mean"][:3]).reshape(9); e = d @ P["spine_precision"] @ d
    for ci, k in ((0, 3), (1, 4)):
        v = C[k] - P["mean"][k]; e += v @ P["joint_precision"][ci] @ v
    return e / 15.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-action", type=int, default=4); ap.add_argument("--held-mod", type=int, default=4, help="performers whose number is a multiple of this are held out")
    ap.add_argument("--cache", default=os.path.join("data", "cmu_cache")); ap.add_argument("--out", default=os.path.join("include", "ik_torso_prior.hpp"))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True)
    fit, held = pick_trials(a.per_action, a.held_mod); print(f"{len(fit)} fit trials, {len(held)} held-out trials (other performers)")
    F = load_set(fit, a.cache); H = load_set(held, a.cache)
    Cl = np.concatenate([c for _, act, c in F if act != "get_up"])
    Cf = np.concatenate([c for _, _, c in F]); Ch = np.concatenate([c for _, _, c in H]); Cn = np.concatenate([c for _, act, c in F if act in NEUTRAL_ACTIONS])
    print(f"fit frames {len(Cf)} ({len(F)} trials, {len({t.split('_')[0] for t, _, _ in F})} performers) | held-out frames {len(Ch)} ({len(H)} trials, "
          f"{len({t.split('_')[0] for t, _, _ in H})} performers) | neutral frames {len(Cn)}")
    L = fit_limits(Cl); mean, P9, P3 = fit_prior(Cf, Cn); d = np.degrees
    print("\nLIMITS (degrees)   flex[min,max]    lat[+-]    twist[+-]    frames inside: fit / held-out")
    cf, pf = coverage(Cl, L); Chl = np.concatenate([c for _, act, c in H if act != "get_up"]); ch, ph = coverage(Chl, L)
    for k in range(T.N):
        print(f"  {T.NAMES[k]:11s} [{d(L[k,0]):6.1f},{d(L[k,1]):6.1f}]  {d(L[k,3]):6.1f}    {d(L[k,5]):6.1f}        {pf[k]:5.1f}% / {ph[k]:5.1f}%")
    print(f"  every joint inside its limits:  fit {cf:.2f}%   held-out {ch:.2f}%")
    print("\nPRIOR MEAN, the natural posture (degrees)    flex     lat   twist")
    for k in range(T.N):
        print(f"  {T.NAMES[k]:11s}                      {d(mean[k,0]):7.2f} {d(mean[k,1]):7.2f} {d(mean[k,2]):7.2f}")
    ph_ = np.cumsum(mean[:5, 0]); print("  cumulative pitch of spine1, spine2, spine3, neck, head (deg):", np.round(d(ph_), 2))
    print("  prior sd (deg), spine joints, order flex x3, lat x3, twist x3:", np.round(d(np.sqrt(np.diag(np.linalg.inv(P9)))), 1))
    if not a.report_only:
        write_header(a.out, L, mean, P9, P3, f"{len(F)} CMU trials from {len({t.split('_')[0] for t, _, _ in F})} performers (held out: {len(H)} trials, {len({t.split('_')[0] for t, _, _ in H})} other performers)")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
