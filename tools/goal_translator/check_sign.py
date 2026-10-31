#!/usr/bin/env python3
"""
check_sign.py - the one-time check of the left/right sign convention (spec section 6), so the whole dataset is not mirrored by mistake.

  python tools/goal_translator/check_sign.py --clips runs/data/clips.npz                  # checks the data against the convention in gt_conventions.py
  python tools/goal_translator/check_sign.py --clips clips.npz --clip 12 --expect left    # a clip YOU know turns left (from its source / by eye): must be measured as left
  python tools/goal_translator/check_sign.py --selftest                                   # no data: the checker on a synthetic left-turning walker and its mirror image

gt_conventions.TURN_POSITIVE_IS_LEFT = True says: turn > 0 is the character's own LEFT. Two independent things have to hold for that to be right in YOUR data:
  1 body frame   in the pelvis's own frame (rotated by the pelvis yaw) the channel called footL is on the +x side and footR on the -x side. That is what "x = the body's left"
                 (gen_dataset.py) means, measured instead of assumed.
  2 turning      windows whose heading grows (dyaw > 0) move toward +x in their own start frame (dx > 0), i.e. positive turn and +x agree.
Both are read from the clips, no model. If either fails the data is mirrored (or the channels are swapped): do NOT train; fix the data or the constant, once, in gt_conventions.py.
With --expect, one clip whose direction you know independently (--expect left|right) must also measure that way (total yaw change and net sideways displacement).
Exit code 0 = consistent, 1 = not.
"""
import argparse, math, sys

import numpy as np

import gt_conventions as C
import build_dataset as B

FOOT_L, FOOT_R = 4, 9          # channels: pelvis xyz yaw | footL xyz yaw contact | footR xyz yaw contact | ... (tools/gen_dataset.py)


def body_frame_lateral(a):
    """a: [F,30] -> per frame, the lateral (x in the pelvis's own frame) position of footL and footR relative to the pelvis."""
    yaw = np.unwrap(a[:, 3].astype(np.float64)); c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for f in (FOOT_L, FOOT_R):
        dx, dz = a[:, f] - a[:, 0], a[:, f + 2] - a[:, 2]
        out.append(dx * c - dz * s)                      # same rotation as train_cvqvae.canonicalize: x' = the old left axis
    return out


def check_clips(actions, offsets, W=64, stride=16, dt=1 / 30.0, verbose=True):
    lat_ok = lat_n = 0
    sign_ok = sign_n = 0
    margin = []
    for i in range(len(offsets) - 1):
        a = actions[offsets[i]:offsets[i + 1]]
        if len(a) < W:
            continue
        L, R = body_frame_lateral(a)
        d = L - R                                        # > 0 when footL is to the +x side of footR
        lat_ok += int((d > 0).sum()); lat_n += len(d); margin.append(float(np.median(d)))
        g = B.window_goals_from_pelvis(a[:, 0], a[:, 2], a[:, 3], W, stride, dt)
        m = (np.abs(g[:, 0]) > 0.1) & (g[:, 3] > 0.2)
        sign_ok += int((np.sign(g[m, 0]) == np.sign(g[m, 1])).sum()); sign_n += int(m.sum())
    res = dict(foot_frac=lat_ok / max(lat_n, 1), foot_n=lat_n, foot_margin=float(np.median(margin)) if margin else float("nan"), turn_frac=sign_ok / max(sign_n, 1), turn_n=sign_n)
    return res


def judge(res):
    problems = []
    if res["foot_n"] == 0:
        problems.append("no clip long enough to check")
    elif res["foot_frac"] < 0.9:
        problems.append(f"footL is on the +x side of footR in only {100 * res['foot_frac']:.0f} % of frames: x is NOT the character's left in this data (mirrored, or footL / footR swapped)")
    if res["turn_n"] < 10:
        problems.append(f"only {res['turn_n']} turning windows: not enough turning in the clips to check the turn sign (add clips with turns)")
    elif res["turn_frac"] < 0.9:
        problems.append(f"yaw growing moves toward +x in only {100 * res['turn_frac']:.0f} % of turning windows: dyaw > 0 is not 'toward the body's +x' here")
    return problems


def clip_direction(a, dt=1 / 30.0):
    yaw = np.unwrap(a[:, 3].astype(np.float64)); c, s = math.cos(yaw[0]), math.sin(yaw[0])
    dx, dz = a[-1, 0] - a[0, 0], a[-1, 2] - a[0, 2]
    return float(yaw[-1] - yaw[0]), float(dx * c - dz * s)       # total heading change, net sideways displacement in the start frame (+ = left under the convention)


def synthetic(left=True, F=600, dt=1 / 30.0, rate=0.35, v=1.0):
    """A walker turning left (or right): pelvis path, and feet 0.1 m to each side of the pelvis in its own frame. The feet do not depend on the turn direction: footL is the body's left."""
    sgn = 1.0 if left else -1.0
    a = np.zeros((F, 30), np.float32); yaw = 0.0; x = z = 0.0
    for i in range(F):
        a[i, 0], a[i, 2], a[i, 3] = x, z, yaw
        for foot, side in ((FOOT_L, +1.0), (FOOT_R, -1.0)):             # the body's left (+ along its own x) is where footL is, by definition of the data layout
            lx = side * 0.1; lz = 0.0
            a[i, foot] = x + lx * math.cos(yaw) + lz * math.sin(yaw); a[i, foot + 2] = z - lx * math.sin(yaw) + lz * math.cos(yaw)
        hm = yaw + sgn * rate * dt / 2
        x += v * dt * math.sin(hm); z += v * dt * math.cos(hm); yaw += sgn * rate * dt
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips"); ap.add_argument("--clip", type=int); ap.add_argument("--expect", choices=["left", "right"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--timing", default="", help="W,stride,dt (default 64,16,1/30: any window length works for this check)")
    a = ap.parse_args()
    if a.selftest:
        ok = True
        for left in (True, False):
            act = synthetic(left); res = check_clips(act, np.array([0, len(act)]))
            bad = judge(res)
            print(f"synthetic {'left' if left else 'right'}-turning walker: feet {100 * res['foot_frac']:.0f} %, turns {100 * res['turn_frac']:.0f} %  ->  {'consistent' if not bad else 'INCONSISTENT: ' + '; '.join(bad)}")
            ok &= not bad
        mirrored = synthetic(True).copy(); mirrored[:, [FOOT_L, FOOT_R]] = mirrored[:, [FOOT_R, FOOT_L]]; mirrored[:, [FOOT_L + 2, FOOT_R + 2]] = mirrored[:, [FOOT_R + 2, FOOT_L + 2]]
        bad = judge(check_clips(mirrored, np.array([0, len(mirrored)])))
        print(f"deliberately swapped footL / footR: {'caught: ' + bad[0][:90] if bad else 'NOT caught'}")
        sys.exit(0 if ok and bad else 1)
    if not a.clips:
        ap.error("--clips FILE (or --selftest)")
    z = np.load(a.clips, allow_pickle=False); A, off = z["actions"], z["offsets"]
    dt = float(z["dt"]) if "dt" in z.files else 1 / 30.0
    W, stride = 64, 16
    if a.timing:
        W, stride, dt = int(a.timing.split(",")[0]), int(a.timing.split(",")[1]), float(a.timing.split(",")[2])
    res = check_clips(A, off, W, stride, dt)
    print(f"{len(off) - 1} clips: footL on the +x side of footR in {100 * res['foot_frac']:.1f} % of {res['foot_n']} frames (median gap {res['foot_margin']:+.3f} m); "
          f"dyaw > 0 moves toward +x in {100 * res['turn_frac']:.1f} % of {res['turn_n']} turning windows")
    problems = judge(res)
    if a.clip is not None:
        dyaw, dside = clip_direction(A[off[a.clip]:off[a.clip + 1]])
        measured = "left" if dyaw > 0 else "right"
        print(f"clip {a.clip}: heading change {math.degrees(dyaw):+.0f} deg, net sideways {dside:+.2f} m  ->  turns {measured} under TURN_POSITIVE_IS_LEFT = {C.TURN_POSITIVE_IS_LEFT}")
        if a.expect:
            m = measured if C.TURN_POSITIVE_IS_LEFT else ("right" if measured == "left" else "left")
            if m != a.expect:
                problems.append(f"clip {a.clip} is known to turn {a.expect} but is measured as {m}")
            if (dside > 0) != (dyaw > 0) and abs(dyaw) > 0.3:
                problems.append(f"clip {a.clip}: heading change and sideways displacement disagree in sign")
    elif a.expect:
        ap.error("--expect needs --clip N")
    if problems:
        print("INCONSISTENT:\n  - " + "\n  - ".join(problems)); sys.exit(1)
    print(f"consistent: TURN_POSITIVE_IS_LEFT = {C.TURN_POSITIVE_IS_LEFT} holds for this data")


if __name__ == "__main__":
    main()
