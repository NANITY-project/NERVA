// ik_solver_fabrik.hpp -- FABRIK (Aristidou & Lazarus 2011) for chains with
// more than 2 bones: fingers, spine, tails, tentacles.
//
// Changes from the original:
//  * fixed arrays, no heap allocation;
//  * warm-started from the chain's current pose (the engine now solves every
//    frame, so this typically converges in 1-3 iterations);
//  * per-joint CONE limits (Joint::set_max_bend) enforced inside the forward
//    pass, so the whole chain re-solves around a limit instead of the limit
//    being clipped afterwards (which breaks the end-effector position);
//  * stagnation exit: constrained/unreachable poses stop as soon as an
//    iteration stops helping instead of burning all max_iterations.
#pragma once
#include "ik_math.hpp"
#include "ik_skeleton.hpp"

namespace ik {

// Rotate unit vector `d` toward unit `axis` until it is within the cone.
inline Vec3 clamp_to_cone(const Vec3& d, const Vec3& axis, float cos_max, float sin_max) {
    float c = dot(d, axis);
    if (c >= cos_max) return d;
    Vec3 perp = d - axis * c;
    if (perp.length_sq() < EPS * EPS) {              // d anti-parallel to axis
        Vec3 h = (std::abs(axis.y) < 0.99f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        perp = h - axis * dot(h, axis);
    }
    return axis * cos_max + perp.normalized() * sin_max;
}

inline bool solve_fabrik(Chain& chain, const Vec3& target,
                          float tolerance = 1e-3f, int max_iterations = 10) {
    const int n = chain.count;
    if (n < 1) return false;

    float len[kMaxJoints];
    Vec3 pos[kMaxJoints + 1];
    float total = 0.f;
    bool any_cone = false;
    for (int i = 0; i < n; ++i) {
        len[i] = chain.joints[i].bone_length;
        if (!(len[i] > EPS)) return false;           // zero/negative/NaN bone
        total += len[i];
        any_cone |= chain.joints[i].limit_bend;
    }
    chain.world_positions(pos);
    const Vec3 root = pos[0];
    const Vec3 root_dir = chain.root_orientation.x_axis();

    const float dist_sq = (target - root).length_sq();
    if (dist_sq > total * total) {
        Vec3 dir = (target - root).normalized();
        Vec3 p = root;
        for (int i = 0; i < n; ++i) { p += dir * len[i]; pos[i + 1] = p; }
    } else {
        const float tol_sq = tolerance * tolerance;
        float prev_err = 1e30f;
        for (int iter = 0; iter < max_iterations; ++iter) {
            float err = (pos[n] - target).length_sq();
            if (err < tol_sq) break;
            if (prev_err - err < 1e-10f) break;     // stagnated (limit-bound)
            prev_err = err;

            // Backward: pin the tip to the target, walk back to the root.
            pos[n] = target;
            for (int i = n - 1; i >= 0; --i) {
                Vec3 dir = (pos[i] - pos[i + 1]).normalized();
                pos[i] = pos[i + 1] + dir * len[i];
            }
            // Forward: pin the root, walk out, enforcing cone limits.
            pos[0] = root;
            Vec3 parent_dir = root_dir;
            for (int i = 1; i <= n; ++i) {
                Vec3 dir = (pos[i] - pos[i - 1]).normalized();
                const Joint& j = chain.joints[i - 1];
                if (any_cone && j.limit_bend)
                    dir = clamp_to_cone(dir, parent_dir, j.cos_max_bend, j.sin_max_bend);
                pos[i] = pos[i - 1] + dir * len[i - 1];
                parent_dir = dir;
            }
        }
    }

    // Positions -> per-joint local rotations.
    Quat accumulated = chain.root_orientation;
    for (int i = 0; i < n; ++i) {
        Vec3 world_dir = (pos[i + 1] - pos[i]) * (1.f / len[i]);   // bone length is known: no sqrt
        chain.joints[i].local_rotation = local_rotation_for_unit_dir(accumulated, world_dir);
        apply_hinge_constraint(chain.joints[i]);
        accumulated = accumulated * chain.joints[i].local_rotation;
    }
    return true;
}

} // namespace ik
