#!/usr/bin/env python3
"""
torso_math.py - the torso joint conventions, shared by bvh_to_clip.py, fit_torso_prior.py and the validators, and mirrored exactly by
include/ik_torso.hpp. (numpy, quaternions are (w, x, y, z), Hamilton convention, like the engine.)

The engine's seven torso bones, in this order:
    0 spine1 (lumbar)   1 spine2   2 spine3 (chest)   3 neck   4 head   5 clavicle_L   6 clavicle_R
Every bone has a FRAME: Y along the bone, Z the anatomical front, X = Y x Z (so X is the body's left when the bone points up).
A bone's parent: spine1 hangs off the pelvis, spine2 off spine1, spine3 off spine2, neck / clavicles off spine3, head off neck.

A joint's LOCAL rotation is parent_frame^-1 * child_frame. It is described by three angles (radians), the SWING-TWIST decomposition:
    twist  rotation about the child's own bone axis (Y); + = turn to the body's LEFT
    flex   the hinge: bend about X;                      + = forward  (the bone tilts toward +Z, the front)
    lat    the side bend: bend about Z;                  + = toward the body's RIGHT (the bone tilts toward -X)
local = Exp(flex * X + lat * Z) * RotY(twist). (Exp = the rotation vector, exact for any swing below 180 degrees.)
A clavicle's bone points SIDEWAYS (Y = +X for the left one, -X for the right one) while the chest's Y points up, so its local rotation has a constant
REST part: RotZ(-90 deg) for the left, RotZ(+90 deg) for the right. The three angles describe the deviation from that rest, local = REST * Exp(..) * RotY(twist),
so for a clavicle flex is protraction (+) / retraction (-), lat is elevation (left: +, right: -) / depression, twist is its axial roll.
"""
import numpy as np

NAMES = ["spine1", "spine2", "spine3", "neck", "head", "clavicle_L", "clavicle_R"]
PARENT = [-1, 0, 1, 2, 3, 2, 2]               # index into NAMES; -1 = the pelvis (upright, heading frame)
N = len(NAMES)
REST = np.array([[1.0, 0.0, 0.0, 0.0]] * 5 + [[np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])   # rest local rotation of each joint


def normalize(v, eps=1e-12):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def frame_matrix(dirs, hints):
    """[N,3] bone directions + [N,3] front hints -> [N,3,3] frames with columns (X, Y, Z): Y = dir, Z = hint made perpendicular to Y, X = Y x Z.
    The same construction as body_detail::frameQ in ik_body.hpp."""
    y = normalize(dirs); z = np.asarray(hints, np.float64); z = z - y * (z * y).sum(-1, keepdims=True)
    bad = (z ** 2).sum(-1) < 1e-10                                         # hint parallel to the bone: any perpendicular will do
    if bad.any():
        h = np.where((np.abs(y[..., 1]) < 0.9)[..., None], np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        z = np.where(bad[..., None], h - y * (h * y).sum(-1, keepdims=True), z)
    z = normalize(z); x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def mat_to_quat(M):
    """[...,3,3] -> [...,4] (w,x,y,z), w >= 0."""
    M = np.asarray(M, np.float64); sh = M.shape[:-2]; M = M.reshape(-1, 3, 3)
    m00, m01, m02 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]; m10, m11, m12 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]; m20, m21, m22 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    tr = m00 + m11 + m22; q = np.zeros((len(M), 4))
    a = tr > 0; s = np.sqrt(np.maximum(tr + 1, 1e-12)) * 2
    q[a] = np.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], 1)[a]
    b = ~a & (m00 > m11) & (m00 > m22); s = np.sqrt(np.maximum(1 + m00 - m11 - m22, 1e-12)) * 2
    q[b] = np.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], 1)[b]
    c = ~a & ~b & (m11 > m22); s = np.sqrt(np.maximum(1 + m11 - m00 - m22, 1e-12)) * 2
    q[c] = np.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], 1)[c]
    d = ~a & ~b & ~c; s = np.sqrt(np.maximum(1 + m22 - m00 - m11, 1e-12)) * 2
    q[d] = np.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], 1)[d]
    q = q / np.linalg.norm(q, axis=1, keepdims=True); q[q[:, 0] < 0] *= -1
    return q.reshape(sh + (4,))


def quat_to_mat(q):
    q = np.asarray(q, np.float64); w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.empty(q.shape[:-1] + (3, 3))
    M[..., 0, 0] = 1 - 2 * (y * y + z * z); M[..., 0, 1] = 2 * (x * y - w * z); M[..., 0, 2] = 2 * (x * z + w * y)
    M[..., 1, 0] = 2 * (x * y + w * z); M[..., 1, 1] = 1 - 2 * (x * x + z * z); M[..., 1, 2] = 2 * (y * z - w * x)
    M[..., 2, 0] = 2 * (x * z - w * y); M[..., 2, 1] = 2 * (y * z + w * x); M[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return M


def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0); bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    q = np.asarray(q, np.float64); return q * np.array([1.0, -1.0, -1.0, -1.0])


def unwrap_quats(q):
    """Flip signs along axis 0 so consecutive quaternions are on the same hemisphere (q and -q are one rotation; interpolation needs continuity)."""
    q = np.array(q, np.float64)
    for i in range(1, len(q)):
        flip = (q[i] * q[i - 1]).sum(-1) < 0; q[i][flip] *= -1
    return q


def angle_between(a, b):
    """Rotation angle in DEGREES between two orientations (q and -q are the same orientation)."""
    d = np.abs((np.asarray(a, np.float64) * np.asarray(b, np.float64)).sum(-1)).clip(0, 1); return np.degrees(2 * np.arccos(d))


def to_coords(q):
    """local rotation [...,4] -> [...,3] (flex, lat, twist), radians. See the module docstring for the meaning of each."""
    q = np.asarray(q, np.float64); q = np.where(q[..., :1] < 0, -q, q)       # q and -q are one rotation: take w >= 0, so the twist below is in [-180, 180] degrees
    w, y = q[..., 0], q[..., 2]
    psi = 2 * np.arctan2(y, w)                                              # twist about Y
    n = np.sqrt(w * w + y * y); n_safe = np.where(n < 1e-9, 1.0, n)
    t = np.stack([np.where(n < 1e-9, 1.0, w / n_safe), np.zeros_like(w), np.where(n < 1e-9, 0.0, y / n_safe), np.zeros_like(w)], -1)
    s = qmul(q, qconj(t)); s = np.where((s[..., :1] < 0), -s, s)             # swing, w >= 0 (shortest way)
    v = s[..., 1:]; vn = np.linalg.norm(v, axis=-1); ang = 2 * np.arctan2(vn, s[..., 0]); k = np.where(vn < 1e-12, 0.0, ang / np.where(vn < 1e-12, 1.0, vn))
    return np.stack([v[..., 0] * k, v[..., 2] * k, psi], -1)


def from_coords(c):
    """[...,3] (flex, lat, twist) -> local rotation [...,4]."""
    c = np.asarray(c, np.float64); fx, lz, ps = c[..., 0], c[..., 1], c[..., 2]
    th = np.sqrt(fx * fx + lz * lz); h = th / 2; k = np.where(th < 1e-12, 0.5, np.sin(h) / np.where(th < 1e-12, 1.0, th))
    sw = np.stack([np.cos(h), fx * k, np.zeros_like(fx), lz * k], -1)
    tw = np.stack([np.cos(ps / 2), np.zeros_like(ps), np.sin(ps / 2), np.zeros_like(ps)], -1)
    return qmul(sw, tw)


def local_quats(Q):
    """Q [F,7,4] frames in the pelvis heading frame -> local rotations [F,7,4] (each relative to its parent bone; spine1 relative to the upright pelvis)."""
    L = np.empty_like(Q)
    for k in range(N):
        L[:, k] = Q[:, k] if PARENT[k] < 0 else qmul(qconj(Q[:, PARENT[k]]), Q[:, k])
    return L


def local_coords(Q):
    """[F,7,4] frames -> [F,7,3] (flex, lat, twist), measured from each joint's rest (REST^-1 * local)."""
    return to_coords(qmul(qconj(REST)[None], local_quats(Q)))


def frames_from_coords(C):
    """[F,7,3] (flex, lat, twist) -> [F,7,4] frames in the heading frame: the inverse of local_coords (forward kinematics down the torso)."""
    C = np.asarray(C, np.float64); Q = np.empty(C.shape[:-1] + (4,)); loc = qmul(REST[None], from_coords(C))
    for k in range(N):
        Q[:, k] = loc[:, k] if PARENT[k] < 0 else qmul(Q[:, PARENT[k]], loc[:, k])
    return Q
