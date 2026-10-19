// ik_engine.hpp -- the piece the model actually talks to.
//
// Owns named chains (e.g. "right_arm") addressed by name OR by a cheap
// integer ChainId, and animates them in TASK SPACE:
//
//   * the hand setpoint follows a minimum-jerk quintic (Flash & Hogan 1985)
//     in Cartesian space -> straight-ish hand paths with a bell-shaped
//     velocity profile, the way human reaches actually look;
//   * IK is solved every frame against that setpoint (2-bone analytic is a
//     handful of flops; FABRIK is warm-started), so constraints, torso lean
//     and pole continuity all stay consistent for the whole motion instead of
//     only at the endpoints;
//   * retargeting mid-motion preserves position, velocity AND acceleration
//     (quintic with arbitrary initial conditions) -- no velocity snap;
//   * movement time follows Fitts' law, capped by the chain's max speed;
//   * the reaction delay applies only when starting from rest.
//
// Text protocol (one line per command, deliberately not JSON):
//   REACH right_arm 0.35 1.20 -0.10
//   POINT right_arm 0.80 1.40  0.20
//   ORIENT right_arm fx fy fz [ux uy uz]
//   RESET right_arm
#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "ik_math.hpp"
#include "ik_skeleton.hpp"
#include "ik_solver_analytic.hpp"
#include "ik_solver_fabrik.hpp"

namespace ik {

enum class SolverType { TwoBoneAnalytic, Fabrik };

using ChainId = int;
constexpr ChainId kNoChain = -1;

// 10t^3 - 15t^4 + 6t^5: zero velocity and acceleration at both ends.
inline float minimum_jerk_s(float t) {
    t = clampf(t, 0.f, 1.f);
    return t * t * t * (10.f + t * (-15.f + 6.f * t));
}

// Tunables, all public on IKEngine::params.
struct MotionParams {
    float reaction_delay = 0.10f;   // s, only when starting from rest
    float fitts_a = 0.10f;          // movement time T = a + b*log2(1 + D/w)
    float fitts_b = 0.09f;
    float fitts_width = 0.03f;      // effective target width, m
    float min_duration = 0.12f;
    float max_duration = 1.5f;
    float path_curvature = 0.04f;   // peak lateral bow of a reach, as a fraction of its length (0 = ruler-straight)
    float pole_relax_rate = 2.0f;   // 1/s: elbow drifts back to its natural pole when idle
    float lean_omega = 12.f;        // rad/s, torso-lean spring
    float idle_amp_m = 0.0015f;     // idle hand wander amplitude (0 disables)
    float breath_amp_m = 0.002f;    // shoulder rise/fall (0 disables)
    float breath_hz = 0.25f;
    // Error-bounded solve skipping: if neither the hand setpoint nor the root
    // has moved more than this since the last solve, the pose is left as is.
    // 0.1 mm is far below a pixel at any sane scale; it lets settled chains
    // (the common case) mostly skip IK while idle-wandering. 0 = solve on any
    // change. A solve is always forced on the frame a reach completes, so
    // endpoints stay exact.
    float solve_epsilon_m = 1e-4f;
};

// Quintic x(t) from (p0, v0, a0) at t=0 to (pf, 0, 0) at t=T, plus an
// optional lateral "bow" whose value, velocity and acceleration are all zero
// at both ends (64*(tau(1-tau))^3), so it cannot break the boundary
// conditions.
struct Trajectory {
    Vec3 c0, c1, c2, c3, c4, c5;
    Vec3 target;
    Vec3 bow_dir;
    float bow_amp = 0.f;
    float T = 1.f, t = 0.f;

    void plan(Vec3 p0, Vec3 v0, Vec3 a0, Vec3 pf, float duration, Vec3 bdir, float bamp) {
        T = duration; t = 0.f; target = pf; bow_dir = bdir; bow_amp = bamp;
        Vec3 d = pf - p0;
        float T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
        c0 = p0; c1 = v0; c2 = a0 * 0.5f;
        c3 = (d * 20.f - v0 * (12.f * T) - a0 * (3.f * T2)) * (1.f / (2.f * T3));
        c4 = (d * -30.f + v0 * (16.f * T) + a0 * (3.f * T2)) * (1.f / (2.f * T4));
        c5 = (d * 12.f - v0 * (6.f * T) - a0 * T2) * (1.f / (2.f * T5));
    }

    // Position only -- all the per-frame path needs.
    Vec3 position(float tt) const {
        Vec3 p = c0 + (c1 + (c2 + (c3 + (c4 + c5 * tt) * tt) * tt) * tt) * tt;
        if (bow_amp != 0.f) {
            float tau = tt / T, u = tau * (1.f - tau);
            p += bow_dir * (bow_amp * 64.f * u * u * u);
        }
        return p;
    }

    void eval(float tt, Vec3& p, Vec3& v, Vec3& a) const {
        p = c0 + (c1 + (c2 + (c3 + (c4 + c5 * tt) * tt) * tt) * tt) * tt;
        v = c1 + (c2 * 2.f + (c3 * 3.f + (c4 * 4.f + c5 * (5.f * tt)) * tt) * tt) * tt;
        a = c2 * 2.f + (c3 * 6.f + (c4 * 12.f + c5 * (20.f * tt)) * tt) * tt;
        if (bow_amp != 0.f) {
            float tau = tt / T, u = tau * (1.f - tau), w = 1.f - 2.f * tau;
            float b  = 64.f * u * u * u;
            float db = 192.f * u * u * w / T;
            float dd = (384.f * u * w * w - 384.f * u * u) / (T * T);
            p += bow_dir * (bow_amp * b);
            v += bow_dir * (bow_amp * db);
            a += bow_dir * (bow_amp * dd);
        }
    }
};

class IKEngine {
public:
    MotionParams params;

    // joint_specs: ordered (joint_name, bone_length) from the joint after the
    // root out to the end effector. torso_lean_m: how far the root may shift
    // toward a far target (0 = rigid). max_speed_m_per_s caps PEAK hand speed.
    // Returns the chain's id (kNoChain if the spec is invalid/too long).
    ChainId add_chain(const std::string& name, Vec3 root_position, Quat root_orientation,
                      const std::vector<std::pair<std::string, float>>& joint_specs,
                      SolverType solver, Vec3 default_pole_dir,
                      float torso_lean_m = 0.f, float max_speed_m_per_s = 1.2f) {
        if (joint_specs.empty() || (int)joint_specs.size() > kMaxJoints) return kNoChain;
        if (by_name_.count(name)) return kNoChain;
        if (solver == SolverType::TwoBoneAnalytic && joint_specs.size() != 2) return kNoChain;
        for (auto& js : joint_specs) if (!(js.second > EPS)) return kNoChain;

        ChainRT c;
        ChainCold cold;
        cold.name = name;
        c.chain.root_position = root_position;
        c.chain.root_orientation = root_orientation;
        c.chain.count = (int)joint_specs.size();
        for (int i = 0; i < c.chain.count; ++i) {
            cold.joint_names.push_back(joint_specs[i].first);
            c.chain.joints[i].bone_length = joint_specs[i].second;
        }
        c.solver = solver;
        c.anchor = root_position;
        c.pole_default = default_pole_dir.normalized();
        c.torso_lean = torso_lean_m;
        c.max_speed = std::max(max_speed_m_per_s, 0.01f);
        c.reach = c.chain.total_length();
        refresh_limits(c);
        c.base_pos = c.chain.end_effector_position();
        c.rest_target = c.base_pos;
        c.idle_seed = 17.31f * (float)chains_.size();
        ChainId id = (ChainId)chains_.size();
        by_name_[name] = id;
        chains_.push_back(std::move(c));
        cold_.push_back(std::move(cold));
        return id;
    }

    // Move a chain's root (character walks, armature moved in a host app).
    // Orientation is the root frame: bones extend along its local +X.
    bool set_anchor(ChainId id, Vec3 position, Quat orientation = Quat::identity()) {
        if (!valid(id) || !position.is_finite()) return false;
        ChainRT& c = chains_[id];
        c.anchor = position;
        c.chain.root_orientation = orientation.normalized();
        c.solved = false;
        return true;
    }

    // Snap the hand to `p` with no motion and solve immediately. Use after
    // add_chain to bind the engine to a pose that already exists in a host app,
    // so the first reach starts from where the limb really is.
    bool teleport_to(ChainId id, Vec3 p) {
        if (!valid(id) || !p.is_finite()) return false;
        ChainRT& c = chains_[id];
        c.base_pos = p;
        c.moving = false; c.settled = true; c.delay_left = 0.f;
        c.lean = c.lean_vel = Vec3();
        c.solved = false;
        step(c, 1e-4f, 0.f);
        return true;
    }

    // Natural elbow/knee bend direction (world space). Update when a character turns.
    bool set_pole(ChainId id, Vec3 dir) {
        if (!valid(id) || !dir.is_finite() || dir.length_sq() < EPS) return false;
        chains_[id].pole_default = dir.normalized();
        return true;
    }

    bool set_rest_target(ChainId id, Vec3 p) {
        if (!valid(id) || !p.is_finite()) return false;
        chains_[id].rest_target = p;
        return true;
    }

    int chain_count() const { return (int)chains_.size(); }
    const std::string& chain_name(ChainId id) const { return cold_.at(id).name; }

    ChainId find_chain(const std::string& name) const {
        auto it = by_name_.find(name);
        return it == by_name_.end() ? kNoChain : it->second;
    }
    bool has_chain(const std::string& name) const { return find_chain(name) != kNoChain; }

    void set_reaction_delay(float seconds) { params.reaction_delay = seconds; }

    // Where RESET brings the hand (default: the chain's initial extension).
    bool set_rest_target(const std::string& name, Vec3 p) {
        ChainId id = find_chain(name);
        if (id == kNoChain || !p.is_finite()) return false;
        chains_[id].rest_target = p;
        return true;
    }

    bool set_hinge_constraint(const std::string& name, int joint_index,
                               Vec3 hinge_axis, float min_angle_rad, float max_angle_rad) {
        Joint* j = joint_ptr(name, joint_index);
        if (!j) return false;
        j->constrain_hinge = true;
        j->hinge_axis = hinge_axis;
        j->min_angle = min_angle_rad;
        j->max_angle = max_angle_rad;
        return true;
    }

    // FABRIK cone limit: joint `joint_index`'s bone may deviate at most
    // max_bend_rad from its parent bone. No calibration needed.
    bool set_max_bend(const std::string& name, int joint_index, float max_bend_rad) {
        Joint* j = joint_ptr(name, joint_index);
        if (!j) return false;
        j->set_max_bend(max_bend_rad);
        return true;
    }

    // Interior elbow/knee angle limits (PI = straight). Two-bone chains only.
    bool set_elbow_limits(const std::string& name, float min_angle_rad, float max_angle_rad) {
        ChainId id = find_chain(name);
        if (id == kNoChain) return false;
        chains_[id].min_elbow = min_angle_rad;
        chains_[id].max_elbow = max_angle_rad;
        refresh_limits(chains_[id]);
        chains_[id].solved = false;
        return true;
    }
    bool set_elbow_limit(const std::string& name, float min_angle_rad) {
        return set_elbow_limits(name, min_angle_rad, PI);
    }

    Quat joint_local_rotation(const std::string& name, int joint_index) const {
        const ChainRT& c = chains_.at(require(name));
        return c.chain.joints.at(joint_index).local_rotation;
    }

    // ---- Start / retarget a reach --------------------------------------
    bool reach_to(const std::string& name, Vec3 target) { return reach_to(find_chain(name), target); }

    bool reach_to(ChainId id, Vec3 target) {
        if (!valid(id) || !target.is_finite()) return false;
        ChainRT& c = chains_[id];

        // Current position/velocity/acceleration (derivatives are evaluated
        // here, on demand, not every frame).
        Vec3 p0 = c.base_pos, v0, a0;
        if (c.moving && c.delay_left <= 0.f) c.traj.eval(c.traj.t, p0, v0, a0);
        Vec3 delta = target - p0;
        float D = delta.length();

        // Fitts' law, then never exceed the chain's peak speed
        // (min-jerk peak speed = 1.875 * D / T).
        float T = params.fitts_a + params.fitts_b * std::log2(1.f + D / params.fitts_width);
        T = clampf(T, params.min_duration, params.max_duration);
        T = std::max(T, 1.875f * D / c.max_speed);

        Vec3 bow_dir{}; float bow_amp = 0.f;
        if (D > 1e-3f && params.path_curvature > 0.f) {
            Vec3 dir = delta * (1.f / D);
            Vec3 b = cross(cross(dir, Vec3(0, 1, 0)), dir);          // "up", made perpendicular to the path
            if (b.length_sq() < 1e-4f) b = c.pole_default - dir * dot(c.pole_default, dir);
            bow_dir = b.normalized();
            bow_amp = params.path_curvature * D;
        }

        // A retarget mid-motion keeps its momentum and does NOT pay the
        // reaction delay again; only a start from rest does.
        float delay = c.moving ? c.delay_left : params.reaction_delay;
        c.traj.plan(p0, v0, a0, target, T, bow_dir, bow_amp);
        c.delay_left = delay;
        c.moving = true;
        c.settled = false;
        return true;
    }

    // Independently target an end-effector orientation. duration_s <= 0 picks
    // a duration from the rotation angle (bigger turns take longer).
    bool set_end_effector_orientation(const std::string& name, Vec3 forward, Vec3 up,
                                       float duration_s = -1.f) {
        ChainId id = find_chain(name);
        if (id == kNoChain || !forward.is_finite() || !up.is_finite()) return false;
        start_orientation(chains_[id], Quat::look_rotation(forward, up), duration_s);
        return true;
    }

    Quat end_effector_orientation(const std::string& name) const {
        return chains_.at(require(name)).o_cur;
    }

    bool reset_chain(const std::string& name) {
        ChainId id = find_chain(name);
        if (id == kNoChain) return false;
        reach_to(id, chains_[id].rest_target);
        start_orientation(chains_[id], Quat::identity(), -1.f);
        return true;
    }

    // ---- Per-frame step -------------------------------------------------
    void update(float dt) {
        if (!(dt > 0.f)) return;
        breath_t_ += dt;
        float breath = params.breath_amp_m > 0.f
            ? std::sin(2.f * PI * params.breath_hz * breath_t_) * params.breath_amp_m : 0.f;
        for (ChainRT& c : chains_) step(c, dt, breath);
    }

    // ---- Queries --------------------------------------------------------
    Vec3 end_effector_position(const std::string& name) const {
        return chains_.at(require(name)).chain.end_effector_position();
    }
    Vec3 end_effector_position(ChainId id) const { return chains_.at(id).chain.end_effector_position(); }
    Vec3 end_effector_setpoint(ChainId id) const { return chains_.at(id).base_pos; }
    Vec3 end_effector_velocity(ChainId id) const {
        const ChainRT& c = chains_.at(id);
        if (!c.moving || c.delay_left > 0.f) return Vec3();
        Vec3 p, v, a; c.traj.eval(c.traj.t, p, v, a);
        return v;
    }
    Vec3 root_position(ChainId id) const { return chains_.at(id).chain.root_position; }
    float trajectory_duration(ChainId id) const { return chains_.at(id).traj.T; }
    bool is_settled(ChainId id) const { return chains_.at(id).settled; }
    const Chain& chain(ChainId id) const { return chains_.at(id).chain; }

    // ---- Text protocol --------------------------------------------------
    // Returns false for anything malformed (unknown command/chain, missing or
    // non-numeric or non-finite arguments, trailing junk) so the caller can
    // send the model a corrective message instead of crashing or, worse,
    // feeding NaN into the solver.
    bool execute_command(const std::string& line) {
        const char* p = line.c_str();
        std::string cmd, chain_name;
        if (!word(p, cmd)) return false;
        if (cmd == "REACH" || cmd == "POINT") {
            float v[3];
            if (!word(p, chain_name) || !numbers(p, v, 3, 3)) return false;
            ChainId id = find_chain(chain_name);
            return id != kNoChain && reach_to(id, Vec3(v[0], v[1], v[2]));
        }
        if (cmd == "RESET") {
            if (!word(p, chain_name) || !end_of_line(p)) return false;
            return reset_chain(chain_name);
        }
        if (cmd == "ORIENT") {
            float v[6];
            if (!word(p, chain_name)) return false;
            int n = numbers(p, v, 3, 6);
            if (n != 3 && n != 6) return false;
            Vec3 up = (n == 6) ? Vec3(v[3], v[4], v[5]) : Vec3(0, 1, 0);
            return set_end_effector_orientation(chain_name, Vec3(v[0], v[1], v[2]), up);
        }
        return false;
    }

private:
    struct ChainRT {
        SolverType solver = SolverType::TwoBoneAnalytic;
        Vec3 anchor, pole_default{0, 0, 1}, rest_target;
        float torso_lean = 0.f, max_speed = 1.2f, reach = 0.f;
        float min_elbow = 0.f, max_elbow = PI, reach_min = 0.f, reach_max = 0.f;

        Trajectory traj;
        bool moving = false, settled = true;
        float delay_left = 0.f;
        Vec3 base_pos;                    // current hand setpoint (derivatives: see Trajectory::eval)

        Vec3 lean, lean_vel;
        float idle_t = 0.f, idle_gain = 0.f, idle_seed = 0.f;
        ValueNoise1 noise_x, noise_y, noise_z;

        bool has_elbow = false;
        Vec3 last_elbow_dir{0, 0, 1};

        bool solved = false;              // skip the solve when nothing changed
        Vec3 last_setpoint, last_root;

        bool orient_active = false;
        Quat o_start, o_target, o_cur;
        float o_t = 0.f, o_T = 0.25f;

        Chain chain;                      // last: root + joints[0..1] follow the hot scalars

    };

    // Names are only touched by the text protocol / tools, never per frame,
    // so they live in a parallel array and don't bloat the hot stride.
    struct ChainCold {
        std::string name;
        std::vector<std::string> joint_names;
    };

    std::vector<ChainRT> chains_;
    std::vector<ChainCold> cold_;
    std::unordered_map<std::string, ChainId> by_name_;
    float breath_t_ = 0.f;

    bool valid(ChainId id) const { return id >= 0 && id < (ChainId)chains_.size(); }
    ChainId require(const std::string& name) const {
        ChainId id = find_chain(name);
        if (id == kNoChain) throw std::out_of_range("unknown chain: " + name);
        return id;
    }
    Joint* joint_ptr(const std::string& name, int idx) {
        ChainId id = find_chain(name);
        if (id == kNoChain || idx < 0 || idx >= chains_[id].chain.count) return nullptr;
        return &chains_[id].chain.joints[idx];
    }
    static void refresh_limits(ChainRT& c) {
        if (c.chain.count == 2)
            two_bone_reach_limits(c.chain.joints[0].bone_length, c.chain.joints[1].bone_length,
                                  c.min_elbow, c.max_elbow, c.reach_min, c.reach_max);
    }
    static void start_orientation(ChainRT& c, const Quat& target, float duration_s) {
        c.o_start = c.o_cur;
        c.o_target = target;
        c.o_t = 0.f;
        float ang = c.o_cur.angle_to(target);
        c.o_T = duration_s > 0.f ? std::max(duration_s, 0.01f)
                                 : clampf(0.15f + 0.30f * ang / PI, 0.15f, 0.6f);
        c.orient_active = true;
    }

    void step(ChainRT& c, float dt, float breath) {
        // 1. Advance the hand trajectory (after any reaction delay).
        bool force_solve = false;
        if (c.moving) {
            float adv = dt;
            if (c.delay_left > 0.f) {
                if (dt <= c.delay_left) { c.delay_left -= dt; adv = 0.f; }
                else { adv = dt - c.delay_left; c.delay_left = 0.f; }
            }
            if (adv > 0.f) {
                c.traj.t += adv;
                if (c.traj.t >= c.traj.T) {
                    c.base_pos = c.traj.target;
                    c.moving = false;
                    c.settled = true;
                    force_solve = true;
                } else {
                    c.base_pos = c.traj.position(c.traj.t);
                }
            }
        }

        // 2. Idle wander: smooth non-repeating noise, faded in once settled
        //    and faded out (continuously) when a new motion begins.
        Vec3 setpoint = c.base_pos;
        if (params.idle_amp_m > 0.f) {
            float g = 4.f * dt;
            c.idle_gain += clampf((c.settled ? 1.f : 0.f) - c.idle_gain, -g, g);
            if (c.idle_gain > 0.f) {
                c.idle_t += dt;
                float s = c.idle_seed, t = c.idle_t;
                setpoint += Vec3(c.noise_x(t * 0.55f + s),
                                 c.noise_y(t * 0.70f + s + 17.3f),
                                 c.noise_z(t * 0.45f + s + 41.7f)) * (params.idle_amp_m * c.idle_gain);
            }
        }

        // 3. Root: breathing plus a SMOOTHED torso lean toward far targets
        //    (the original assigned the lean instantly, teleporting the hand).
        Vec3 root = c.anchor;
        if (c.torso_lean > EPS) {
            Vec3 to = setpoint - c.anchor;
            float d2 = to.length_sq();
            Vec3 tgt;
            if (d2 > EPS * EPS && c.reach > EPS) {
                float inv = inv_sqrt(d2);
                float ext = d2 * inv / c.reach;
                float amt = clampf((ext - 0.7f) / 0.3f, 0.f, 1.f) * c.torso_lean;
                tgt = to * (amt * inv);
            }
            smooth_damp(c.lean, tgt, c.lean_vel, params.lean_omega, dt);
            if ((c.lean - tgt).length_sq() < 1e-10f && c.lean_vel.length_sq() < 1e-8f) {
                c.lean = tgt; c.lean_vel = Vec3();
            }
            root += c.lean;
        }
        root.y += breath;

        // 4. Solve IK against the setpoint -- unless it is within solve_epsilon
        //    of where the last solve left it.
        const float eps2 = params.solve_epsilon_m * params.solve_epsilon_m;
        if (!c.solved || force_solve ||
            (setpoint - c.last_setpoint).length_sq() > eps2 ||
            (root - c.last_root).length_sq() > eps2 ||
            (eps2 == 0.f && (setpoint != c.last_setpoint || root != c.last_root))) {
            c.chain.root_position = root;
            if (c.solver == SolverType::TwoBoneAnalytic) {
                Vec3 pole = c.pole_default;
                if (c.has_elbow)   // continuity, relaxing toward the natural pole
                    pole = lerp(c.last_elbow_dir, c.pole_default,
                                clampf(params.pole_relax_rate * dt, 0.f, 1.f));   // solver normalises
                Vec3 elbow;
                if (solve_two_bone_reach(c.chain, setpoint, pole, c.reach_min, c.reach_max, &elbow)) {
                    c.last_elbow_dir = elbow;
                    c.has_elbow = true;
                }
            } else {
                solve_fabrik(c.chain, setpoint);
            }
            c.last_setpoint = setpoint;
            c.last_root = root;
            c.solved = true;
        }

        // 5. Wrist orientation on its own timeline.
        if (c.orient_active) {
            c.o_t += dt;
            float s = minimum_jerk_s(c.o_t / c.o_T);
            c.o_cur = Quat::slerp(c.o_start, c.o_target, s);
            if (c.o_t >= c.o_T) { c.o_cur = c.o_target; c.orient_active = false; }
        }
    }

    // ---- allocation-free-ish command parsing helpers -------------------
    static bool is_ws(char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; }
    static bool word(const char*& p, std::string& out) {
        while (is_ws(*p)) ++p;
        const char* s = p;
        while (*p && !is_ws(*p)) ++p;
        out.assign(s, p);
        return !out.empty();
    }
    static bool end_of_line(const char* p) { while (is_ws(*p)) ++p; return *p == '\0'; }
    // Parses between min_n and max_n finite floats, then requires end of line.
    static int numbers(const char*& p, float* out, int min_n, int max_n) {
        int n = 0;
        while (n < max_n) {
            while (is_ws(*p)) ++p;
            if (!*p) break;
            char* e = nullptr;
            float v = std::strtof(p, &e);
            if (e == p || !std::isfinite(v) || (*e && !is_ws(*e))) return 0;
            out[n++] = v;
            p = e;
        }
        if (n < min_n || !end_of_line(p)) return 0;
        return n;
    }
};

} // namespace ik
