// ik_skeleton.hpp -- joint chain representation + forward kinematics.
//
// Chains are fixed-capacity (IK_MAX_JOINTS) with no heap allocation, so a
// Chain is trivially copyable and FK/solving never touches the allocator.
// Names live in the engine (cold data), not in the hot joint array.
#pragma once
#include <array>
#include <vector>
#include "ik_math.hpp"

#ifndef IK_MAX_JOINTS
#define IK_MAX_JOINTS 12
#endif

namespace ik {

constexpr int kMaxJoints = IK_MAX_JOINTS;

// One joint in a chain. `bone_length` is the length of the bone running
// FROM this joint TO its child. `local_rotation` is relative to the parent.
struct Joint {
    float bone_length = 0.f;
    Quat local_rotation = Quat::identity();

    // Optional single-axis hinge clamp -- see apply_hinge_constraint().
    bool constrain_hinge = false;
    Vec3 hinge_axis{0.f, 0.f, 1.f};
    float min_angle = -PI;
    float max_angle = PI;

    // Cone limit used by FABRIK: this bone may not deviate more than
    // max_bend from its PARENT bone's direction. Needs no axis calibration,
    // unlike the hinge mechanism, and is applied inside the solver so the
    // rest of the chain adapts instead of being clipped afterwards.
    bool limit_bend = false;
    float cos_max_bend = -1.f, sin_max_bend = 0.f;

    void set_max_bend(float rad) {
        if (rad >= PI - 1e-3f) { limit_bend = false; cos_max_bend = -1.f; sin_max_bend = 0.f; return; }
        rad = std::max(rad, 0.f);
        limit_bend = true;
        cos_max_bend = std::cos(rad);
        sin_max_bend = std::sin(rad);
    }
};

struct Chain {
    Vec3 root_position;
    Quat root_orientation = Quat::identity();
    int count = 0;
    std::array<Joint, kMaxJoints> joints{};

    float total_length() const {
        float sum = 0.f;
        for (int i = 0; i < count; ++i) sum += joints[i].bone_length;
        return sum;
    }

    // Forward kinematics into a caller-provided buffer of count+1 entries
    // (root, each joint, end effector). Bones extend along local +X.
    void world_positions(Vec3* out) const {
        Vec3 pos = root_position;
        Quat rot = root_orientation;
        out[0] = pos;
        for (int i = 0; i < count; ++i) {
            rot = rot * joints[i].local_rotation;
            pos += rot.x_axis() * joints[i].bone_length;
            out[i + 1] = pos;
        }
    }

    // Convenience (allocates) -- fine for tests and tools, not for hot loops.
    std::vector<Vec3> world_positions() const {
        std::vector<Vec3> v(count + 1);
        world_positions(v.data());
        return v;
    }

    Vec3 end_effector_position() const {
        Vec3 pos = root_position;
        Quat rot = root_orientation;
        for (int i = 0; i < count; ++i) {
            rot = rot * joints[i].local_rotation;
            pos += rot.x_axis() * joints[i].bone_length;
        }
        return pos;
    }
};

// Given the accumulated world rotation up to this joint's parent and the
// world-space direction this joint's bone must point, return the joint's
// LOCAL rotation. Bones extend along local +X at rest.
//
// Specialised shortest-arc rotation from +X: for unit t, the quaternion is
// (1+tx, 0, -tz, ty) / sqrt(2+2tx) -- no cross/dot/normalize round trips.
// Fast path: caller guarantees `unit_world_dir` is (nearly) unit length --
// true wherever the bone length is known (solvers divide by it), which saves
// a normalisation per joint per solve.
inline Quat local_rotation_for_unit_dir(const Quat& parent_world_rot,
                                         const Vec3& unit_world_dir) {
    Vec3 t = parent_world_rot.rotate_inverse(unit_world_dir);
    if (t.x < -1.f + EPS) return Quat(0.f, 0.f, 0.f, -1.f);   // 180 deg about -Z
    float inv = inv_sqrt(2.f + 2.f * t.x);
    return Quat((1.f + t.x) * inv, 0.f, -t.z * inv, t.y * inv);
}

// General version: normalises its input (and tolerates zero vectors).
inline Quat local_rotation_for_world_dir(const Quat& parent_world_rot,
                                          const Vec3& target_world_dir) {
    float l2 = target_world_dir.length_sq();
    if (l2 < EPS * EPS) return Quat::identity();
    return local_rotation_for_unit_dir(parent_world_rot, target_world_dir * inv_sqrt(l2));
}

// Clamp a joint's local_rotation to a single-axis hinge swing.
//
// CAVEAT (found the hard way): with a fixed world-space pole vector a
// joint's local rotation axis is only approximately constant across target
// directions. hinge_axis must be calibrated empirically (solve once, read
// the joint's axis back) -- a guessed axis silently produces a garbage pose.
// Fine for small-range joints (fingers). For 2-bone limbs use the elbow
// angle limits instead; for FABRIK chains prefer set_max_bend (cone limit).
inline void apply_hinge_constraint(Joint& j) {
    if (!j.constrain_hinge) return;
    float theta = 2.f * std::acos(clampf(j.local_rotation.w, -1.f, 1.f));
    if (theta < EPS) return;
    Vec3 axis(j.local_rotation.x, j.local_rotation.y, j.local_rotation.z);
    axis = axis.normalized();
    Vec3 hinge = j.hinge_axis.normalized();
    float signed_theta = (dot(axis, hinge) >= 0.f) ? theta : -theta;
    signed_theta = clampf(signed_theta, j.min_angle, j.max_angle);
    j.local_rotation = Quat::from_axis_angle(hinge, signed_theta);
}

} // namespace ik
