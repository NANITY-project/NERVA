// ik_torso.hpp -- the torso as a chain of anatomical joints: orientation targets, joint limits and a learned pose prior.
//
// Before this the spine was a 3-bone FABRIK chain that bent equally everywhere (a cone limit of 22 degrees per bone), the neck and head were two
// yaw/pitch rotations, the clavicles a fixed direction, and nothing could be told the trunk's actual ORIENTATION. This header adds three things:
//   1. ORIENTATION TARGETS for spine1, spine2, spine3, neck, head and both clavicles (what a motion-capture file has: tools/bvh_to_clip.py -> `torso`).
//   2. JOINT LIMITS per joint: flex (the hinge), lat (side bend) and twist ranges, fitted to real people (tools/fit_torso_prior.py -> ik_torso_prior.hpp).
//   3. A POSE PRIOR: a Gaussian over the nine spine coordinates. Asked to reach a neck-base target, the spine does not bend equally: it takes the pose
//      that real people would (the least surprising one under the prior), starting from their natural posture, a gentle S-curve.
// The conventions are the ones in tools/torso_math.py, which this file mirrors line for line (quaternions (w,x,y,z), Y along a bone, Z = anatomical front):
//   local rotation of a joint = REST * Exp(flex * X + lat * Z) * RotY(twist);   flex + = forward, lat + = toward the body's right, twist + = turn left.
#pragma once
#include <cmath>
#include <algorithm>
#include "ik_math.hpp"
#include "ik_torso_prior.hpp"

namespace ik { namespace torso {

constexpr int kJoints = 7;                       // spine1 spine2 spine3 neck head clavicle_L clavicle_R
enum Joint { Spine1 = 0, Spine2, Spine3, Neck, Head, ClavL, ClavR };
constexpr int kParent[kJoints] = {-1, 0, 1, 2, 3, 2, 2};

struct Coord { float flex = 0.f, lat = 0.f, twist = 0.f; };

inline float wrap_pi(float a) { while (a > PI) a -= 2.f * PI; while (a < -PI) a += 2.f * PI; return a; }

// the rest local rotation of each joint (identity except the clavicles, whose bone lies sideways: RotZ(-90) left, RotZ(+90) right)
inline Quat rest(int k) {
    if (k == ClavL) return Quat(std::cos(-PI / 4.f), 0.f, 0.f, std::sin(-PI / 4.f));
    if (k == ClavR) return Quat(std::cos(PI / 4.f), 0.f, 0.f, std::sin(PI / 4.f));
    return Quat::identity();
}

// rotation -> (flex, lat, twist): swing-twist about the bone's own Y axis
inline Coord to_coord(Quat q) {
    if (q.w < 0.f) q = Quat(-q.w, -q.x, -q.y, -q.z);
    Coord c; float psi = 2.f * std::atan2(q.y, q.w), n = std::sqrt(q.w * q.w + q.y * q.y);
    Quat t = n < 1e-9f ? Quat::identity() : Quat(q.w / n, 0.f, q.y / n, 0.f);
    Quat s = q * t.conjugate(); if (s.w < 0.f) s = Quat(-s.w, -s.x, -s.y, -s.z);
    float vn = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    float k = vn < 1e-9f ? 0.f : 2.f * std::atan2(vn, s.w) / vn;
    c.flex = s.x * k; c.lat = s.z * k; c.twist = psi;
    return c;
}
inline Quat from_coord(Coord c) {
    float th = std::sqrt(c.flex * c.flex + c.lat * c.lat), h = th * 0.5f, k = th < 1e-9f ? 0.5f : std::sin(h) / th;
    Quat sw(std::cos(h), c.flex * k, 0.f, c.lat * k), tw(std::cos(c.twist * 0.5f), 0.f, std::sin(c.twist * 0.5f), 0.f);
    return sw * tw;
}

// ---- limits -------------------------------------------------------------------------------------------------------------------------------
// Clamp a joint's coordinates to the fitted ranges. `scale` widens or narrows them (1 = the data). Returns the largest angle moved (radians).
inline float clamp_limits(int k, Coord& c, float scale = 1.f) {
    const float* L = torso_prior::kLimits[k]; float moved = 0.f;
    auto cl = [&](float& v, float lo, float hi) { float o = v; v = clampf(v, lo * scale, hi * scale); moved = std::max(moved, std::fabs(v - o)); };
    cl(c.flex, L[0], L[1]); cl(c.lat, L[2], L[3]); cl(c.twist, L[4], L[5]);
    return moved;
}

// ---- orientation targets ------------------------------------------------------------------------------------------------------------------
// target[k]: the wanted orientation of torso bone k in the pelvis HEADING frame (upright, facing +Z). Each joint's local rotation is read off the
// targets, clamped to its limits, and the chain is rebuilt from the pelvis up, so a limited joint pushes its error up the chain like a real one.
// `clav_neutral[s]`: the ENGINE's clavicle frame at its rest posture, in chest coordinates (the engine's skeleton is not the capture skeleton: its
// clavicle starts below the shoulder, the capture's above it); a clavicle's coordinates are measured from the natural posture (the prior mean),
// so a capture that holds its clavicle at its own natural posture gives the engine's natural posture.
// out[k]: the rebuilt frames, heading frame. Returns the largest limit violation (radians); coords (optional) the clamped joint coordinates.
inline float solve_orientation(const Quat target[kJoints], const Quat clav_neutral[2], float limit_scale, Quat out[kJoints], Coord* coords = nullptr) {
    float worst = 0.f;
    for (int k = 0; k < kJoints; ++k) {
        int p = kParent[k];
        Quat local = p < 0 ? target[k] : target[p].conjugate() * target[k];                 // the capture's local rotation
        Coord c = to_coord(rest(k).conjugate() * local);
        if (limit_scale > 0.f) worst = std::max(worst, clamp_limits(k, c, limit_scale));
        if (coords) coords[k] = c;
        Quat le;
        if (k >= ClavL) { Coord d{c.flex - torso_prior::kMean[k][0], c.lat - torso_prior::kMean[k][1], c.twist - torso_prior::kMean[k][2]}; le = clav_neutral[k - ClavL] * from_coord(d); }
        else le = from_coord(c);
        out[k] = (p < 0 ? le : out[p] * le).normalized();
    }
    return worst;
}

// ---- the pose prior -----------------------------------------------------------------------------------------------------------------------
// x[3 * joint + coordinate], joint 0..2 = spine1..3 (flex, lat, twist).
struct SpinePose { float x[9]; };

inline SpinePose spine_mean() {
    SpinePose p; for (int j = 0; j < 3; ++j) for (int c = 0; c < 3; ++c) p.x[3 * j + c] = torso_prior::kMean[j][c];
    return p;
}
// The three spine frames (heading frame) of a pose, and the neck-base position (relative to the lumbar joint) for bone lengths `len`.
inline Vec3 spine_fk(const SpinePose& p, const float len[3], Quat q[3]) {
    Quat cur = Quat::identity(); Vec3 pos;
    for (int j = 0; j < 3; ++j) {
        cur = (cur * from_coord(Coord{p.x[3 * j], p.x[3 * j + 1], p.x[3 * j + 2]})).normalized();
        q[j] = cur; pos += cur.rotate(Vec3(0.f, 1.f, 0.f)) * len[j];
    }
    return pos;
}
// How unusual a spine pose is: the Mahalanobis distance squared to the prior, divided by the 9 coordinates (about 1 for a typical real pose).
inline float spine_energy(const SpinePose& p) {
    float d[9], e = 0.f; for (int i = 0; i < 9; ++i) d[i] = p.x[i] - spine_mean().x[i];
    for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) e += d[i] * torso_prior::kSpinePrecision[i][j] * d[j];
    return e / 9.f;
}
// Mahalanobis distance squared / 3 of the neck, head or a clavicle (ci: 0 neck, 1 head, 2 left clavicle, 3 right clavicle) from its natural posture.
inline float joint_energy(int ci, Coord c) {
    int k = Neck + ci; float d[3] = {c.flex - torso_prior::kMean[k][0], c.lat - torso_prior::kMean[k][1], c.twist - torso_prior::kMean[k][2]}, e = 0.f;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) e += d[i] * torso_prior::kJointPrecision[ci][i][j] * d[j];
    return e / 3.f;
}

namespace detail {
// solve A y = r for a small dense symmetric positive matrix (n <= 4), in place; false if singular
inline bool solve_small(float A[4][4], float r[4], int n) {
    for (int i = 0; i < n; ++i) {
        int piv = i; for (int k = i + 1; k < n; ++k) if (std::fabs(A[k][i]) > std::fabs(A[piv][i])) piv = k;
        if (std::fabs(A[piv][i]) < 1e-12f) return false;
        if (piv != i) { for (int j = 0; j < n; ++j) std::swap(A[i][j], A[piv][j]); std::swap(r[i], r[piv]); }
        for (int k = i + 1; k < n; ++k) { float f = A[k][i] / A[i][i]; for (int j = i; j < n; ++j) A[k][j] -= f * A[i][j]; r[k] -= f * r[i]; }
    }
    for (int i = n - 1; i >= 0; --i) { float s = r[i]; for (int j = i + 1; j < n; ++j) s -= A[i][j] * r[j]; r[i] = s / A[i][i]; }
    return true;
}
} // namespace detail

// Find the spine pose a real person would take to put the neck base at `target` (relative to the lumbar joint, heading frame) while the three joints'
// twists add up to `twist_sum`: the pose closest to the natural posture in the prior's metric, i.e. minimise (x-mean)' P (x-mean) subject to the
// constraints. Newton steps on the constraints with the covariance as the metric (x += S J' (J S J')^-1 r); a joint that reaches a limit is held there
// and the others take over. `stiff` scales the covariance (1 = people as measured, <1 = a stiffer body, >1 a looser one).
// Returns the pose; `residual` (optional) = how far the neck base is from the target when the limits make it unreachable (metres).
inline SpinePose solve_spine(const Vec3& target, float twist_sum, const float len[3], float limit_scale, float stiff, const SpinePose* start, float* residual = nullptr) {
    SpinePose p = start ? *start : spine_mean(); bool fixed[9] = {false};
    const float (*S0)[9] = torso_prior::kSpineCovariance; Quat q[3];
    float res = 0.f;
    for (int pass = 0; pass < 3; ++pass) {
        float S[9][9];
        for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) S[i][j] = (fixed[i] || fixed[j]) ? 0.f : S0[i][j] * stiff;
        for (int it = 0; it < 4; ++it) {
            Vec3 pos = spine_fk(p, len, q); float r[4] = {target.x - pos.x, target.y - pos.y, target.z - pos.z, twist_sum - (p.x[2] + p.x[5] + p.x[8])};
            res = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            if (res < 1e-4f && std::fabs(r[3]) < 1e-3f) break;
            float J[4][9] = {{0}}; const float h = 1e-3f;
            for (int i = 0; i < 9; ++i) {
                SpinePose pp = p; pp.x[i] += h; Quat qq[3]; Vec3 pj = spine_fk(pp, len, qq);
                J[0][i] = (pj.x - pos.x) / h; J[1][i] = (pj.y - pos.y) / h; J[2][i] = (pj.z - pos.z) / h; J[3][i] = (i % 3 == 2) ? 1.f : 0.f;
            }
            float SJt[9][4]; for (int i = 0; i < 9; ++i) for (int c = 0; c < 4; ++c) { float s = 0.f; for (int j = 0; j < 9; ++j) s += S[i][j] * J[c][j]; SJt[i][c] = s; }
            float A[4][4], y[4] = {r[0], r[1], r[2], r[3]};
            for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) { float s = 0.f; for (int i = 0; i < 9; ++i) s += J[a][i] * SJt[i][b]; A[a][b] = s + (a == b ? 1e-7f : 0.f); }
            if (!detail::solve_small(A, y, 4)) break;
            for (int i = 0; i < 9; ++i) { float d = 0.f; for (int c = 0; c < 4; ++c) d += SJt[i][c] * y[c]; p.x[i] += clampf(d, -0.35f, 0.35f); }
        }
        bool hit = false;
        if (limit_scale > 0.f) for (int j = 0; j < 3; ++j) {
            Coord c{p.x[3 * j], p.x[3 * j + 1], p.x[3 * j + 2]}; float m = clamp_limits(j, c, limit_scale);
            if (m > 1e-5f) { for (int k = 0; k < 3; ++k) { if (std::fabs((&c.flex)[k] - p.x[3 * j + k]) > 1e-5f) fixed[3 * j + k] = true; p.x[3 * j + k] = (&c.flex)[k]; } hit = true; }
        }
        if (!hit) break;
    }
    if (residual) { Vec3 pos = spine_fk(p, len, q); *residual = (target - pos).length(); (void)res; }
    return p;
}

}} // namespace ik::torso
