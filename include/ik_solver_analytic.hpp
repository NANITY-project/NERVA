// ik_solver_analytic.hpp -- exact 2-bone IK via the law of cosines.
//
// Closed-form: no iteration, no tolerance, exact every frame. Right tool for
// upper-arm + forearm, thigh + shin. For longer chains use FABRIK.
#pragma once
#include "ik_math.hpp"
#include "ik_skeleton.hpp"

namespace ik {

// Reach-distance window implied by elbow/knee angle limits (interior angle
// between the two bones; PI = straight). Computed once per chain, not per
// solve. Limiting reach via the law of cosines is exact and frame-invariant
// -- there is no rotation axis to get wrong.
inline void two_bone_reach_limits(float L1, float L2, float min_angle_rad, float max_angle_rad,
                                   float& reach_min, float& reach_max) {
    reach_min = std::abs(L1 - L2) + EPS;
    reach_max = L1 + L2 - EPS;
    if (min_angle_rad > 0.f)
        reach_min = std::max(reach_min, std::sqrt(std::max(0.f,
            L1 * L1 + L2 * L2 - 2.f * L1 * L2 * std::cos(min_angle_rad))));
    if (max_angle_rad < PI)
        reach_max = std::min(reach_max, std::sqrt(std::max(0.f,
            L1 * L1 + L2 * L2 - 2.f * L1 * L2 * std::cos(max_angle_rad))));
    if (reach_min > reach_max) reach_min = reach_max;
}

// Fast path: reach window precomputed. `pole_dir` picks which way the joint
// bends (doesn't need to be unit or exactly perpendicular; just not parallel
// to the reach direction). If elbow_dir_out is given it receives the unit
// world direction of the upper bone, for pole-continuity across frames.
//
// Versus the original: no acos/cos/sin (sin derived from cos), no
// normalize() for the forearm (its vector is known analytically), no
// allocation, and the bone->rotation conversion is the specialised one.
inline bool solve_two_bone_reach(Chain& chain, const Vec3& target, const Vec3& pole_dir,
                                  float reach_min, float reach_max,
                                  Vec3* elbow_dir_out = nullptr) {
    if (chain.count != 2) return false;
    const float L1 = chain.joints[0].bone_length;
    const float L2 = chain.joints[1].bone_length;
    if (L1 <= EPS || L2 <= EPS) return false;

    Vec3 to_target = target - chain.root_position;
    float d2 = to_target.length_sq();
    Vec3 dir;
    float d, inv_d;
    if (d2 > EPS * EPS) {
        inv_d = inv_sqrt(d2);
        d = d2 * inv_d;
        dir = to_target * inv_d;
    } else {
        inv_d = 0.f; d = 0.f; dir = Vec3(1.f, 0.f, 0.f);
    }
    float dc = clampf(d, reach_min, reach_max);
    if (dc != d) inv_d = 1.f / dc;                 // clamped: need 1/dc instead

    // Law of cosines, one multiply by the reciprocal instead of a divide;
    // sin from cos instead of acos + sin + cos.
    float cos_root = clampf((L1 * L1 - L2 * L2 + dc * dc) * (0.5f / L1) * inv_d, -1.f, 1.f);
    float sin_root = fast_sqrt(1.f - cos_root * cos_root);

    Vec3 pole = pole_dir - dir * dot(pole_dir, dir);
    if (pole.length_sq() < EPS) {
        Vec3 fb = (std::abs(dir.y) < 0.99f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        pole = fb - dir * dot(fb, dir);
    }
    pole = pole.normalized();

    Vec3 upper_dir = dir * cos_root + pole * sin_root;              // unit
    Vec3 lower_dir = (dir * dc - upper_dir * L1) * (1.f / L2);      // unit, length known analytically

    chain.joints[0].local_rotation = local_rotation_for_unit_dir(chain.root_orientation, upper_dir);
    Quat after0 = chain.root_orientation * chain.joints[0].local_rotation;
    chain.joints[1].local_rotation = local_rotation_for_unit_dir(after0, lower_dir);

    if (elbow_dir_out) *elbow_dir_out = upper_dir;
    return true;
}

// Original signature, kept for compatibility. Prefer solve_two_bone_reach
// with cached limits in hot code.
inline bool solve_two_bone_ik(Chain& chain, const Vec3& target, const Vec3& pole_dir,
                               float min_elbow_angle_rad = 0.f, float max_elbow_angle_rad = PI) {
    if (chain.count != 2) return false;
    float rmin, rmax;
    two_bone_reach_limits(chain.joints[0].bone_length, chain.joints[1].bone_length,
                          min_elbow_angle_rad, max_elbow_angle_rad, rmin, rmax);
    return solve_two_bone_reach(chain, target, pole_dir, rmin, rmax);
}

} // namespace ik
