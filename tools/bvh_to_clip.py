#!/usr/bin/env python3
"""
bvh_to_clip.py - turn real motion capture (BVH: CMU, 100STYLE, ACCAD/OSU ...) into the same 30-number action clips the engine uses.

  python tools/bvh_to_clip.py 07_01.bvh 08_01.bvh --out real.npz
  python tools/bvh_to_clip.py *.bvh --out real.npz --fps 30

Why: a realism grader is only worth something if real people score well on it. This lets you grade real walks with exactly the same code
as engine / generator clips (tools/realism.py), and, if you want, replay them in the viewer.

What it does: forward kinematics of the BVH skeleton, uniformly scaled so the leg (hip to ankle) is the engine's length (so the file's unit - cm, inch, m - does not matter),
rotated/translated so the clip starts at the origin facing +Z, then reads off pelvis, foot soles, foot yaw, contact flags (foot low AND slow),
wrists, neck base and a gaze point. It also saves the joint angles the engine cannot be asked for (knee, elbow, trunk lean) as `angles`.
Skeletons: joints are found by NAME, whatever the capture system called them (CMU LeftUpLeg/LeftLeg/LeftFoot/LeftToeBase, 100STYLE LeftHip/LeftKnee/LeftAnkle/LeftToe with\nLeftShoulder = upper arm, Mixamo, Bip01 L Thigh ...; see PARTS below). A file whose up axis is not +Y (some exports are Z-up) is turned upright first. If a joint cannot be\nfound the error lists the joint names of the file: `python tools/bvh_to_clip.py --probe file.bvh` shows how each one was read.\nOutput npz keys match gen_dataset.py (actions, offsets, dt, ...) plus: angles [F,5] (kneeL kneeR elbowL elbowR trunk_lean, degrees), labels.
Contact detection is a heuristic; mocap feet roll and slip a little, so real clips will not pass the engine's strict "planted feet never slide"
verdict. That is expected: judge them with realism.py, not with the physics verdict.
"""
import argparse, os, re, sys, warnings
import numpy as np

ENGINE_LEG = 0.867      # hip joint to ankle joint, metres (thigh 0.437 + shin 0.430 in the engine)
ANKLE_H = 0.08          # ankle joint height above the sole in the engine
TOE_H = 0.03            # toe-base joint height above the floor


def load_bvh(path):
    txt = open(path, encoding="utf-8", errors="ignore").read().replace("\r", "")
    head, _, mot = txt.partition("MOTION")
    toks = head.replace("{", " { ").replace("}", " } ").split()
    joints, stack, i = [], [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            joints.append(dict(name=toks[i + 1], parent=stack[-1] if stack else -1, off=None, ch=[])); stack.append(len(joints) - 1); i += 2; continue
        if t == "End":
            stack.append(-2); i += 2; continue                      # End Site: skip its block
        if t == "OFFSET" and stack and stack[-1] >= 0:
            joints[stack[-1]]["off"] = np.array([float(x) for x in toks[i + 1:i + 4]]); i += 4; continue
        if t == "CHANNELS" and stack and stack[-1] >= 0:
            n = int(toks[i + 1]); joints[stack[-1]]["ch"] = toks[i + 2:i + 2 + n]; i += 2 + n; continue
        if t == "}":
            stack.pop()
        i += 1
    fr = int(re.search(r"Frames:\s*(\d+)", mot).group(1)); ft = float(re.search(r"Frame Time:\s*([\d.eE+-]+)", mot).group(1))
    data = np.array(mot.split("Time:", 1)[1].split()[1:], dtype=np.float64).reshape(fr, -1)
    return joints, data, ft


# ---------------------------------------------------------------------------------------------------------------- skeleton adapters
# The converter below is written in CMU joint names. Any other skeleton is renamed to them first, by reading each joint name as (side, body part).
# PARTS: CMU name of the role -> body-part words that mean the same joint in other skeletons, best first. The joint's own position is what counts:
# "knee" is where the thigh ends, "ankle" where the shin ends, "wrist" is the hand joint, "elbow" the forearm joint (the CMU / 100STYLE offsets mean the same).
PARTS = {"UpLeg": ["upleg", "upperleg", "thigh", "hip", "legupper", "femur"], "Leg": ["leg", "lowerleg", "calf", "shin", "knee", "leglower", "tibia"],
         "Foot": ["foot", "ankle"], "ToeBase": ["toebase", "toe", "toes", "toe0", "ball"],
         "Arm": ["arm", "upperarm", "uparm", "armupper", "humerus"], "ForeArm": ["forearm", "lowerarm", "elbow", "armlower", "lowarm"], "Hand": ["hand", "wrist"]}
CENTRE = {"Neck": ["neck", "neck1", "neck0", "necklower"], "Head": ["head"]}
_SIDES = {"left": "Left", "l": "Left", "lft": "Left", "right": "Right", "r": "Right", "rgt": "Right"}


def _side_part(name):
    """'LeftUpLeg' -> ('Left', 'upleg'); 'Bip01 L Thigh' -> ('Left', 'thigh'); 'arm.R' -> ('Right', 'arm'); 'Neck' -> (None, 'neck')."""
    raw = name.split(":")[-1]; glued = re.match(r"^([LR])(?=[A-Z])", raw)             # LThigh, RThumb, LHipJoint
    toks = re.sub(r"[^A-Za-z0-9]+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)).lower().split()
    toks = [t for t in toks if t not in ("bip01", "bip", "mixamorig")]
    if glued and toks:
        side = "Left" if glued.group(1) == "L" else "Right"; toks[0] = toks[0][1:] or None; toks = [t for t in toks if t]; return side, "".join(toks)
    if toks and toks[0] in _SIDES and len(toks) > 1:
        return _SIDES[toks[0]], "".join(toks[1:])
    if toks and toks[-1] in _SIDES and len(toks) > 1:
        return _SIDES[toks[-1]], "".join(toks[:-1])
    return None, "".join(toks)


def resolve_roles(joints):
    """{CMU name: index of the joint that plays that role}. Raises ValueError listing the file's joints when a role is missing."""
    sp = [_side_part(j["name"]) for j in joints]; found = {}
    for side in ("Left", "Right"):
        for role, words in PARTS.items():
            cand = list(words) + (["shoulder"] if role == "Arm" else [])          # "shoulder" is the upper arm in 100STYLE (its clavicle is "Collar"); in CMU it is the clavicle, but CMU has "Arm" and wins first
            for w in cand:
                hit = [i for i, (s, p) in enumerate(sp) if s == side and p == w]
                if hit:
                    found[side + role] = hit[0]; break
    for role, words in CENTRE.items():
        for w in words:
            hit = [i for i, (s, p) in enumerate(sp) if s is None and p == w]
            if hit:
                found[role] = hit[0]; break
    need = [s + r for s in ("Left", "Right") for r in PARTS] + list(CENTRE)
    miss = [n for n in need if n not in found]
    if miss:
        raise ValueError("cannot find joint(s) " + ", ".join(miss) + " in this skeleton; its joints are: " + ", ".join(j["name"] for j in joints))
    return {n: found[n] for n in need}


def adapt_skeleton(joints):
    """Rename the joints that play the converter's roles to the CMU names it uses (in place). Returns the {role: original name} that was read."""
    roles = resolve_roles(joints); orig = {n: joints[i]["name"] for n, i in roles.items()}; targets = set(roles.values())
    for k, j in enumerate(joints):
        if k not in targets and j["name"] in roles:                                    # an unrelated joint that happens to carry a CMU name: step aside
            j["name"] = "_" + j["name"]
    for n, i in roles.items():
        joints[i]["name"] = n
    return orig


def up_rotation(rest):
    """3x3 rotation that turns the skeleton's own 'up' (feet to head in its rest pose) to +Y. Identity for the usual Y-up BVH; Z-up exports are laid upright."""
    u = rest["Head"][0][0] - 0.5 * (rest["LeftFoot"][0][0] + rest["RightFoot"][0][0]); n = float(np.linalg.norm(u))
    if n < 1e-9 or np.abs(u).max() < 0.7 * n:
        warnings.warn("cannot tell which axis is up from the rest pose; assuming +Y"); return np.eye(3)
    k = int(np.argmax(np.abs(u))); up = np.zeros(3); up[k] = np.sign(u[k])
    if k == 1 and up[1] > 0:
        return np.eye(3)
    y = np.array([0.0, 1.0, 0.0]); c = float(up @ y)
    if c < -0.999:
        return np.diag([-1.0, -1.0, 1.0])                                          # upside down: half a turn about Z
    v = np.cross(up, y); V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V / (1.0 + c)


def rotate_world(sk, Q):
    return {n: (p @ Q.T, np.einsum("ij,fjk->fik", Q, R)) for n, (p, R) in sk.items()}


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a); n = len(a); R = np.zeros((n, 3, 3)); R[:, :, :] = np.eye(3)
    if axis == "X":
        R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    elif axis == "Y":
        R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c
    else:
        R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c
    return R


def forward_kinematics(joints, data):
    F = len(data); col = 0; pos, R = {}, {}
    for j, jt in enumerate(joints):
        ch = jt["ch"]; vals = data[:, col:col + len(ch)]; col += len(ch)
        local = np.tile(np.eye(3), (F, 1, 1)); t = np.zeros((F, 3))
        for k, c in enumerate(ch):
            if c.endswith("position"):
                t[:, "XYZ".index(c[0])] = vals[:, k]
            else:
                local = local @ rot(c[0], vals[:, k])
        if jt["parent"] < 0:
            pos[j], R[j] = t, local
        else:
            p = jt["parent"]; pos[j] = pos[p] + np.einsum("fij,j->fi", R[p], jt["off"]); R[j] = R[p] @ local
    return {joints[j]["name"]: (pos[j], R[j]) for j in pos}


def flexion(a, b, c):
    """Angle in degrees between segment a->b and b->c (0 = straight limb)."""
    u, v = b - a, c - b; cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def clean_runs(c, k):
    """Remove runs of 1s, then runs of 0s, shorter than k frames (contact blips and gaps)."""
    for val in (1.0, 0.0):
        d = np.diff(np.r_[0.0, (c == val).astype(float), 0.0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(st, en):
            if b - a < k:
                c[a:b] = 1.0 - val
    return c


def lowpass(x, fs, fc=6.0, order=4):
    """Zero-phase low-pass along axis 0 (FFT, Butterworth-shaped gain, linear trend removed and reflect-padded so the ends do not ring)."""
    F = len(x)
    if F < 8 or fc >= fs / 2:
        return x
    x = np.asarray(x, np.float64); ramp = np.linspace(0, 1, F).reshape((-1,) + (1,) * (x.ndim - 1)) * (x[-1] - x[0]) + x[0]; r = x - ramp
    pad = min(F - 1, int(fs)); rp = np.pad(r, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    f = np.fft.rfftfreq(len(rp), 1.0 / fs); h = (1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.fft.irfft(np.fft.rfft(rp, axis=0) * h, len(rp), axis=0)[pad:pad + F] + ramp


def convert(path, fps=30, hand_cutoff_hz=6.0, rows=None):
    """BVH file -> (actions [F,30], angles [F,5], dt). hand_cutoff_hz: the CMU hand joints jitter by about 2 cm (marker noise: 0.4-1 m/s of >6 Hz velocity even
    while standing still), so hands, neck and gaze are low-passed at this cutoff at the file's native rate (120 Hz) before the clip is decimated to `fps`.
    Human voluntary motion lives well below 6 Hz; without this the 'real' reference would teach a grader that jitter is normal. None/0 turns it off.
    rows: (first, last) frame of the FILE to convert (its native rate), e.g. the cut points a dataset ships; None = the whole file."""
    joints, data, ft = load_bvh(path); adapt_skeleton(joints)
    if rows is not None:
        data = data[max(0, int(rows[0])):int(rows[1])]
        if len(data) < 2:
            raise ValueError(f"frame range {tuple(rows)} leaves no frames")
    # corrupt rows (all zeros) appear at the start/end of a few CMU files. Only those are trimmed: the old "hips below half the median height" rule
    # also deleted the lying / crouching / sitting frames of non-walking clips, which are exactly the frames those actions are about.
    ok = np.abs(data).sum(axis=1) > 1e-9
    if not ok.any():
        raise ValueError("every row of the file is zero")
    idx = np.where(ok)[0]; data = data[idx[0]:idx[-1] + 1]
    sk = forward_kinematics(joints, data); rest = forward_kinematics(joints, np.zeros((1, data.shape[1]))); Q = up_rotation(rest)      # all-zero channels = the skeleton's rest pose
    # which way the head's own axes look at rest: the body faces left x up, and at rest (all rotations zero) a joint's axes ARE the file's axes. CMU: +Z. Snapped to the
    # nearest file axis (a rest pose is axis-aligned by convention) so tiny asymmetries in the offsets do not rotate the gaze.
    f0 = np.cross(rest["LeftUpLeg"][0][0] - rest["RightUpLeg"][0][0], Q.T @ np.array([0.0, 1.0, 0.0])); f0 = f0 / (np.linalg.norm(f0) + 1e-12)
    ax = np.zeros(3); k0 = int(np.argmax(np.abs(f0))); ax[k0] = np.sign(f0[k0]); head_fwd = ax if float(f0 @ ax) > np.cos(np.radians(8.0)) else f0
    if not np.allclose(Q, np.eye(3)):
        sk = rotate_world(sk, Q); rest = rotate_world(rest, Q)
    P = lambda n: sk[n][0]
    # engine-sized: leg length from the skeleton's own offsets
    off = {j["name"]: np.linalg.norm(j["off"]) for j in joints if j["off"] is not None}
    scale = ENGINE_LEG / (off["LeftLeg"] + off["LeftFoot"])
    hip = 0.5 * (P("LeftUpLeg") + P("RightUpLeg")) * scale
    left = (P("LeftUpLeg") - P("RightUpLeg")); fwd = np.cross(left, [0, 1, 0]); fwd[:, 1] = 0
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    # canonical frame: start at the origin facing +Z
    c0, y0 = hip[0].copy(), yaw[0]; c0[1] = 0
    cy, sy = np.cos(-y0), np.sin(-y0)
    def canon(x):
        x = x * scale - c0
        return np.stack([x[:, 0] * cy + x[:, 2] * sy, x[:, 1], -x[:, 0] * sy + x[:, 2] * cy], 1)
    hip_c = canon(P("LeftUpLeg") * 0.5 + P("RightUpLeg") * 0.5); yaw_c = yaw - y0
    lank, rank = canon(P("LeftFoot")), canon(P("RightFoot")); ltoe, rtoe = canon(P("LeftToeBase")), canon(P("RightToeBase"))
    dt0 = ft; step = max(1, int(round(1 / (fps * dt0)))); dt = dt0 * step
    sp = lambda q: np.r_[0.0, np.linalg.norm(q[2:, [0, 2]] - q[:-2, [0, 2]], axis=1) / (2 * dt0), 0.0]
    def contact(ank, toe):
        """A planted foot is a STILL foot (a still heel or a still ball), whatever its measured height: heights drift by centimetres in mocap."""
        # the reference is a low PERCENTILE, not the minimum: a few CMU files have a corrupt frame (ankle 0.8 m below the floor) that would otherwise
        # push every frame of the clip out of reach and leave a seated or standing person with no planted foot at all
        c = ((np.minimum(sp(ank), sp(toe)) < 0.5) & ((ank[:, 1] - np.percentile(ank[:, 1], 2)) < 0.3)).astype(float)
        return clean_runs(c, int(round(0.1 / dt0)))
    lc, rc = contact(lank, ltoe), contact(rank, rtoe)
    planted = np.r_[lank[lc > 0.5, 1], rank[rc > 0.5, 1]]
    # the floor, from the ankles of planted feet (a clip with no planted foot at all, e.g. one long flight, falls back to its lowest ankles)
    floor = (np.median(planted) if len(planted) else np.percentile(np.r_[lank[:, 1], rank[:, 1]], 5)) - ANKLE_H
    def foot(ank, toe, c):
        d = toe - ank; fy = np.unwrap(np.arctan2(d[:, 0], d[:, 2]))
        t = np.arange(len(ank)); idx = np.where(np.diff(np.r_[0, c, 0]) != 0)[0]; seg = list(zip(idx[::2], idx[1::2]))
        if seg:                                                  # this foot's own planted ankle height, interpolated through the swings
            base = np.interp(t, [(a + b) / 2 for a, b in seg], [np.median(ank[a:b, 1]) for a, b in seg])
        else:
            base = np.full(len(ank), floor + ANKLE_H)
        sole = ank.copy(); sole[:, 1] = np.maximum(ank[:, 1] - base, 0.0); sole[c > 0.5, 1] = 0.0
        return sole, fy
    hip_c[:, 1] -= floor
    ls, ly = foot(lank, ltoe, lc); rs, ry = foot(rank, rtoe, rc)
    for a_ in (lank, rank, ltoe, rtoe):
        a_[:, 1] -= floor
    rv = (rest["Neck"][0][0] - 0.5 * (rest["LeftUpLeg"][0][0] + rest["RightUpLeg"][0][0])) * scale      # neck above the hip centre in the skeleton's own upright rest pose
    rest_xz = np.array([rv[0], rv[2]])                                                              # its horizontal part is the skeleton's built-in lean/offset
    neck = canon(P("Neck")); head = canon(P("Head")); lw, rw = canon(P("LeftHand")), canon(P("RightHand"))
    for a_ in (neck, head, lw, rw):
        a_[:, 1] -= floor
    neck[:, 0] -= rest_xz[0] * np.cos(yaw_c) + rest_xz[1] * np.sin(yaw_c); neck[:, 2] -= -rest_xz[0] * np.sin(yaw_c) + rest_xz[1] * np.cos(yaw_c)   # an upright trunk = zero lean
    hf = np.einsum("fij,j->fi", sk["Head"][1], head_fwd); hf = np.stack([hf[:, 0] * cy + hf[:, 2] * sy, hf[:, 1], -hf[:, 0] * sy + hf[:, 2] * cy], 1)
    look = head + 2.0 * hf
    if hand_cutoff_hz:
        lw, rw, neck, look = (lowpass(q, 1.0 / dt0, hand_cutoff_hz) for q in (lw, rw, neck, look))
    A = np.zeros((len(hip_c), 30), np.float32)
    A[:, 0:3] = hip_c; A[:, 3] = yaw_c
    A[:, 4:7] = ls; A[:, 7] = ly - 0.0; A[:, 8] = lc; A[:, 9:12] = rs; A[:, 12] = ry; A[:, 13] = rc
    A[:, 14:17] = lw; A[:, 17] = 1; A[:, 18:21] = rw; A[:, 21] = 1; A[:, 22:25] = look; A[:, 25] = 1; A[:, 26:29] = neck; A[:, 29] = 1
    # joint angles (degrees): knees, elbows, trunk lean (+ = forward of the heading)
    kn = lambda s: flexion(P(s + "UpLeg"), P(s + "Leg"), P(s + "Foot"))
    el = lambda s: flexion(P(s + "Arm"), P(s + "ForeArm"), P(s + "Hand"))
    tr = neck - hip_c; hdir = np.stack([np.sin(yaw_c), np.zeros_like(yaw_c), np.cos(yaw_c)], 1)
    lean = np.degrees(np.arctan2((tr * hdir).sum(1), tr[:, 1]))
    ang = np.stack([kn("Left"), kn("Right"), el("Left"), el("Right"), lean], 1).astype(np.float32)
    return A[::step], ang[::step], dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bvh", nargs="+"); ap.add_argument("--out", default="real.npz"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--probe", action="store_true", help="only show how the skeleton of each file is read (joint roles, up axis, frames, rate), convert nothing")
    a = ap.parse_args(); acts, angs, labels, dts = [], [], [], []
    if a.probe:
        for p in a.bvh:
            joints, data, ft = load_bvh(p); print(f"{p}: {len(joints)} joints, {len(data)} frames at {1 / ft:.1f} Hz")
            try:
                orig = adapt_skeleton(joints); rest = forward_kinematics(joints, np.zeros((1, data.shape[1]))); Q = up_rotation(rest)
                print("  roles read as:  " + ", ".join(f"{k} <- {v}" for k, v in orig.items()) + f"\n  up axis: {'+Y' if np.allclose(Q, np.eye(3)) else 'not +Y, turned upright'}")
            except ValueError as e:
                print("  " + str(e))
        return
    for p in a.bvh:
        A, ang, dt = convert(p, a.fps); acts.append(A); angs.append(ang); labels.append(os.path.splitext(os.path.basename(p))[0]); dts.append(dt)
        print(f"  {labels[-1]}: {len(A)} frames ({len(A) * dt:.1f} s)")
    off = np.cumsum([0] + [len(x) for x in acts])
    np.savez_compressed(a.out, actions=np.concatenate(acts), offsets=off, angles=np.concatenate(angs), labels=np.array(labels), dt=np.float32(np.mean(dts)),
                        style=np.zeros((len(acts), 1), np.float32), style_names=np.array(["real"]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
