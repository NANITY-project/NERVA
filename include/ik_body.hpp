// ik_body.hpp -- a full humanoid body with ground contact and balance.
//
// The model (or a harness) drives it with COORDINATES; the body supplies the physics:
//   * ~22 bones (6 core + 4/arm + 4/leg), up to ~50 with full fingers
//   * feet that plant on surfaces and do not slide, never sink into them, and cannot be placed out of reach
//   * a centre-of-mass / support-polygon balance model (extrapolated CoM, Hof 2008), falling, and stepping
//   * measured feedback with reasons ("FOOT rejected: step too long (1.24 m, max 0.90 m)")
// There are no walk/jump/wave animations in here. A walk is a SEQUENCE of FOOT/PELVIS coordinates; reusable
// skills are built from those by whoever drives the body.
//
// World: Y up, metres. At yaw 0 the body faces +Z and its LEFT is +X.
// Text protocol (one command per line; every reply says why when it refuses):
//   HAND  <left|right> x y z | rest      wrist target (the clavicle and arm do the rest)
//   FOOT  <left|right> x y z [yaw_deg]   sole target (point on the sole under the ankle); y at a surface = a step
//   PELVIS x y z | auto                  hip target;  FACE x z | auto   pelvis facing
//   LOOK x y z | auto                    head look target;  TORSO x y z | auto   neck-base target
//   GRIP <left|right> 0..1               finger curl;  STAND   recover from a fall;  BALANCE assist|none
//   JUMP  x y z [apex_m]                 push off with both feet and land at (x, y=surface, z); the engine solves the arc.
//                                        Take-off speed is limited by how far the legs can push: crouch first with PELVIS.
//                                        While airborne FOOT x y z sets where a foot will land instead of queueing a step.
//   GAIT  dynamic|legacy                 dynamic (default): the centre of mass is simulated as an inverted pendulum and steered through the
//                                        queued footfalls with capture-point control (no stop at every step); legacy: the old quasi-static shuffle.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>
#include <memory>
#include "ik_engine.hpp"

namespace ik {

struct BoneOut { const char* name; int parent; Vec3 head, tail; Quat q; };   // q: Y axis along the bone, Z = anatomical front

namespace body_detail {
struct V2 { float x = 0.f, z = 0.f; };
inline float cross2(V2 o, V2 a, V2 b) { return (a.x - o.x) * (b.z - o.z) - (a.z - o.z) * (b.x - o.x); }
inline std::vector<V2> hull(std::vector<V2> p) {                      // convex hull, counter-clockwise
    std::sort(p.begin(), p.end(), [](V2 a, V2 b) { return a.x < b.x || (a.x == b.x && a.z < b.z); });
    p.erase(std::unique(p.begin(), p.end(), [](V2 a, V2 b) { return a.x == b.x && a.z == b.z; }), p.end());
    if (p.size() < 3) return p;
    std::vector<V2> h(2 * p.size()); size_t k = 0;
    for (size_t i = 0; i < p.size(); ++i) { while (k >= 2 && cross2(h[k - 2], h[k - 1], p[i]) <= 0) --k; h[k++] = p[i]; }
    for (size_t i = p.size() - 1, t = k + 1; i > 0; --i) { while (k >= t && cross2(h[k - 2], h[k - 1], p[i - 1]) <= 0) --k; h[k++] = p[i - 1]; }
    h.resize(k - 1);
    return h;
}
inline float hull_margin(const std::vector<V2>& h, V2 p) {                 // signed distance to the boundary, inside positive
    if (h.empty()) return -1e9f;
    if (h.size() == 1) return -std::hypot(p.x - h[0].x, p.z - h[0].z);
    if (h.size() == 2) {
        float ex = h[1].x - h[0].x, ez = h[1].z - h[0].z, l2 = ex * ex + ez * ez;
        float t = l2 > 0 ? clampf(((p.x - h[0].x) * ex + (p.z - h[0].z) * ez) / l2, 0.f, 1.f) : 0.f;
        return -std::hypot(p.x - (h[0].x + ex * t), p.z - (h[0].z + ez * t));
    }
    float m = 1e9f;
    for (size_t i = 0; i < h.size(); ++i) {
        V2 a = h[i], b = h[(i + 1) % h.size()]; float ex = b.x - a.x, ez = b.z - a.z, l = std::hypot(ex, ez);
        if (l < 1e-9f) continue;
        m = std::min(m, (p.x - a.x) * (-ez / l) + (p.z - a.z) * (ex / l));
    }
    return m;
}
inline V2 hull_project(const std::vector<V2>& h, V2 p, float safe) {       // nudge p until it is at least `safe` inside
    if (h.empty()) return p;
    if (h.size() < 3) { V2 c; for (auto& q : h) { c.x += q.x; c.z += q.z; } c.x /= h.size(); c.z /= h.size(); return c; }
    for (int it = 0; it < 8; ++it) {
        bool moved = false;
        for (size_t i = 0; i < h.size(); ++i) {
            V2 a = h[i], b = h[(i + 1) % h.size()]; float ex = b.x - a.x, ez = b.z - a.z, l = std::hypot(ex, ez);
            if (l < 1e-9f) continue;
            float nx = -ez / l, nz = ex / l, d = (p.x - a.x) * nx + (p.z - a.z) * nz;
            if (d < safe) { p.x += nx * (safe - d); p.z += nz * (safe - d); moved = true; }
        }
        if (!moved) break;
    }
    return p;
}
inline float wrap(float a) { return std::fmod(std::fmod(a + PI, 2 * PI) + 2 * PI, 2 * PI) - PI; }
inline Vec3 fwd(float yaw) { return Vec3(std::sin(yaw), 0.f, std::cos(yaw)); }
inline Vec3 lft(float yaw) { return Vec3(std::cos(yaw), 0.f, -std::sin(yaw)); }
inline Quat frameQ(Vec3 y, Vec3 z) {                                   // Y along the bone, Z as close to `z` as possible
    y = y.normalized(); if (y.length_sq() < 0.5f) y = Vec3(0, 1, 0);
    z = z - y * dot(z, y);
    if (z.length_sq() < 1e-8f) { Vec3 h = std::fabs(y.y) < 0.9f ? Vec3(0, 1, 0) : Vec3(0, 0, 1); z = h - y * dot(h, y); }
    return Quat::look_rotation(z.normalized(), y);
}
inline float swing_pitch(float tau) {                                  // cosmetic foot pitch during a swing, toe-up positive
    if (tau < 0.3f) { float k = 1.f - tau / 0.3f; return -0.5f * k * k; }
    if (tau > 0.7f) return 0.25f * std::sin(PI * (tau - 0.7f) / 0.3f);
    return 0.f;
}
} // namespace body_detail

class Body {
public:
    enum class Balance { None = 0, Assist = 1 };
    struct Params {
        float scale = 1.f;                  // 1.0 ~ 1.75 m tall; scales every length
        int finger_lod = 1;                 // 0 none, 1 one bone per finger, 2 full phalanges
        Balance balance = Balance::Assist;  // Assist: weight shifts before a step, CoM kept inside the support polygon
        float safe_margin = 0.02f;          // m the extrapolated CoM is kept inside the support polygon
        float step_time_scale = 1.f;
        float max_stride = 0.90f;           // m, support foot -> target (scaled)
        float max_step_up = 0.45f, max_step_down = 0.60f;
        float fall_margin = -0.03f, fall_time = 0.30f;
        float swing_reach_slack = 0.03f;      // m: how far past 0.99 x leg reach the pelvis may let a LANDING leg stretch (the old smoothing let it reach ~5 cm; 0 = hard cap, but then the pelvis dips 2 cm+ more); tracking absorbs this with heel lift
        float shift_settle_speed = 0.04f;   // m/s: how still the pelvis must be over the support foot before the other foot lifts
        float jump_accel = 10.f;            // m/s^2 net push-off acceleration of the legs (take-off speed = sqrt(2 * accel * push stroke))
        float ankle_push = 0.06f;           // m of extra stroke from the ankles/toes (so even a standing hop is possible)
        float max_launch_speed = 3.4f;      // m/s, total take-off speed cap
        float hand_idle_m = 0.0015f;        // tiny hand drift when idle (0 = none, for deterministic tests)
        // ---- dynamic gait (reduced-order physics: linear inverted pendulum + capture point, see docs/GAIT.md)
        bool dynamic_gait = true;
        float step_ds_time = 0.10f;         // s, minimum double support (weight transfer) before the next foot lifts
        float swing_base = 0.30f, swing_per_m = 0.25f;   // swing time = base + per_m * stride (s)
        float leg_len_frac = 0.985f;        // stance leg length as a fraction of the straight leg: the pelvis vaults over it (the bob comes from this)
        float arm_swing_gain = 0.5f;        // resting hands swing m per m of the opposite foot's fore-aft offset (0 = off)
        float twist_gain = 0.17f;           // pelvis yaw (rad) per m of foot fore-aft offset; the chest counter-rotates (0 = off)
        float head_turn_omega = 14.f;       // rad/s, head/gaze dynamics (critically damped; 0 = instant like a camera)
        float recover_time = 1.5f;          // s after a fall before the body stands itself back up (0 = stay fallen until STAND)
    };
    struct Result { bool ok; std::string msg; };
    struct Surface { float x0, x1, z0, z1, y; };
    enum class FootState { Planted, Swing, Air };

    Params params;

    Body() : Body(Params{}) {}                       // (two constructors: a nested struct's default initializers can't appear in a default argument)
    explicit Body(const Params& p) : params(p) { build(); }
    // Copyable: a copy is a full snapshot (pose, gait/tracking state, queued steps, events). bones_ point at the bone names, which live in one
    // immutable list shared by all copies (never reallocated after build()), so a copy's pointers stay valid.
    Body(const Body&) = default;
    Body& operator=(const Body&) = default;

    void set_hand_idle(float m) { params.hand_idle_m = m; arms_.params.idle_amp_m = m; }

    // ------------------------------------------------------------------ world
    void add_surface(float x0, float x1, float z0, float z1, float y) { surfaces_.push_back({std::min(x0, x1), std::max(x0, x1), std::min(z0, z1), std::max(z0, z1), y}); }
    void clear_surfaces() { surfaces_.clear(); }
    // Highest surface under (x,z) that a foot standing at height `ref_y` could be on: tops above ref_y + max_step_up are NOT floors
    // (a window top far above you is something to walk under/in front of, not ground that snaps your foot up to it).
    float surface_at(float x, float z, float ref_y) const {
        float y = 0.f, lim = ref_y + params.max_step_up * params.scale + 1e-3f;
        for (const auto& s : surfaces_) if (s.y <= lim && x >= s.x0 && x <= s.x1 && z >= s.z0 && z <= s.z1) y = std::max(y, s.y);
        return y;
    }
    float surface_at(float x, float z) const {
        float y = 0.f;
        for (const auto& s : surfaces_) if (x >= s.x0 && x <= s.x1 && z >= s.z0 && z <= s.z1) y = std::max(y, s.y);
        return y;
    }

    // ------------------------------------------------------------------ commands
    Result command(const std::string& line) {
        std::vector<std::string> t = tokens(line);
        if (t.empty()) return fail("empty command");
        std::string v = upper(t[0]);
        auto num = [&](size_t i, float& out) { return i < t.size() && parse(t[i], out); };
        auto side = [&](size_t i) { return i < t.size() ? (upper(t[i]) == "LEFT" ? 0 : upper(t[i]) == "RIGHT" ? 1 : -1) : -1; };
        float a, b, c, d;
        if (v == "STAND") return t.size() == 1 ? stand() : fail("usage: STAND");
        if (v == "BALANCE") {
            if (t.size() != 2 || (upper(t[1]) != "ASSIST" && upper(t[1]) != "NONE")) return fail("usage: BALANCE assist|none");
            params.balance = upper(t[1]) == "ASSIST" ? Balance::Assist : Balance::None; return ok("balance " + lower(t[1]));
        }
        if (v == "HAND") {
            int s = side(1);
            if (s < 0) return fail("usage: HAND <left|right> x y z | rest");
            if (t.size() == 3 && upper(t[2]) == "REST") { hands_[s].target_mode = false; hands_[s].has_cmd = false; return ok(side_name(s) + " hand relaxed"); }
            if (t.size() == 5 && num(2, a) && num(3, b) && num(4, c)) return cmd_hand(s, Vec3(a, b, c));
            return fail("usage: HAND <left|right> x y z | rest  (numbers must be finite)");
        }
        if (v == "FOOT") {
            int s = side(1);
            if (s < 0 || (t.size() != 5 && t.size() != 6) || !num(2, a) || !num(3, b) || !num(4, c)) return fail("usage: FOOT <left|right> x y z [yaw_deg]");
            bool hy = t.size() == 6; d = 0; if (hy && !num(5, d)) return fail("yaw must be a finite number of degrees");
            return cmd_foot(s, Vec3(a, b, c), hy, d * PI / 180.f);
        }
        if (v == "PELVIS") {
            if (t.size() == 2 && upper(t[1]) == "AUTO") { pelvis_user_ = false; return ok("pelvis follows the feet"); }
            if (t.size() == 4 && num(1, a) && num(2, b) && num(3, c)) { pelvis_user_ = true; pelvis_tgt_ = Vec3(a, b, c); return ok("pelvis target set"); }
            return fail("usage: PELVIS x y z | auto");
        }
        if (v == "FACE") {
            if (t.size() == 2 && upper(t[1]) == "AUTO") { face_user_ = false; return ok("pelvis faces along the feet"); }
            if (t.size() == 3 && num(1, a) && num(2, b)) { face_user_ = true; face_pt_ = Vec3(a, 0, b); turn_warned_ = false; return ok("facing target set"); }
            return fail("usage: FACE x z | auto");
        }
        if (v == "LOOK") {
            if (t.size() == 2 && upper(t[1]) == "AUTO") { look_user_ = false; return ok("head looks ahead"); }
            if (t.size() == 4 && num(1, a) && num(2, b) && num(3, c)) { look_user_ = true; look_tgt_ = Vec3(a, b, c); look_warned_ = false; return ok("look target set"); }
            return fail("usage: LOOK x y z | auto");
        }
        if (v == "TORSO") {
            if (t.size() == 2 && upper(t[1]) == "AUTO") { torso_user_ = false; return ok("torso upright"); }
            if (t.size() == 4 && num(1, a) && num(2, b) && num(3, c)) { torso_user_ = true; torso_tgt_ = Vec3(a, b, c); return ok("neck-base target set"); }
            return fail("usage: TORSO x y z | auto");
        }
        if (v == "JUMP") {
            if ((t.size() != 4 && t.size() != 5) || !num(1, a) || !num(2, b) || !num(3, c)) return fail("usage: JUMP x y z [apex_m]  (land at x, surface height y, z)");
            bool ha = t.size() == 5; d = 0; if (ha && !(num(4, d) && d >= 0.f)) return fail("apex_m must be a finite number >= 0 (clearance above the higher of take-off and landing height)");
            return cmd_jump(Vec3(a, b, c), ha, d);
        }
        if (v == "GRIP") {
            int s = side(1);
            if (s < 0 || t.size() != 3 || !num(2, a)) return fail("usage: GRIP <left|right> 0..1");
            hands_[s].curl = clampf(a, 0.f, 1.f); return ok(side_name(s) + " grip " + fmt(hands_[s].curl));
        }
        if (v == "GAIT") {
            if (t.size() != 2 || (upper(t[1]) != "DYNAMIC" && upper(t[1]) != "LEGACY")) return fail("usage: GAIT dynamic|legacy");
            params.dynamic_gait = upper(t[1]) == "DYNAMIC"; dyn_init_ = false; return ok("gait " + lower(t[1]));
        }
        if (v == "SURFACE") {
            float e;
            if (t.size() == 6 && num(1, a) && num(2, b) && num(3, c) && num(4, d) && num(5, e)) { add_surface(a, b, c, d, e); return ok("surface added"); }
            return fail("usage: SURFACE x0 x1 z0 z1 y");
        }
        return fail("unknown command '" + t[0] + "'");
    }

    // ------------------------------------------------------------------ simulation
    void update(float dt) {
        if (!(dt > 0.f) || tracking_) return;                    // while tracking, track() drives the body
        dt = std::min(dt, 0.05f);
        time_ += dt;
        if (!fallen_) {
            fall_t_ = 0.f;
            if (flight_ != Flight::None) update_flight(dt);
            else { step_machine(dt); update_feet(dt); update_pelvis(dt); }
        } else if (params.recover_time > 0.f && (fall_t_ += dt) > params.recover_time) { fall_t_ = 0.f; stand(); event("recovered from the fall (the pose was reset upright)"); }
        pose_upper(dt, false);
        pose_legs();
        compute_bones();
        compute_balance(dt);
    }

    // ------------------------------------------------------------------ learning interface
    // A fixed-size state vector for models, and a per-frame TRACKING mode: a generator (VAE / diffusion / retargeted video) gives the
    // pose it wants every frame and the body enforces what physics forbids (no sliding planted feet, no sinking below the ground, legs and
    // arms only as long as they are) and reports how much it had to change. The gait controller is not used while tracking.
    static constexpr int STATE_DIM = 35, ACTION_DIM = 30, REPORT_DIM = 12;
    static const char* const* state_names() {
        static const char* n[STATE_DIM] = {"pelvis_x", "pelvis_y", "pelvis_z", "yaw", "com_x", "com_y", "com_z", "com_vx", "com_vz", "margin_m", "fallen",
            "footL_x", "footL_y", "footL_z", "footL_yaw", "footL_state", "footL_heel", "footR_x", "footR_y", "footR_z", "footR_yaw", "footR_state", "footR_heel",
            "wristL_x", "wristL_y", "wristL_z", "wristR_x", "wristR_y", "wristR_z", "handL_targeted", "handR_targeted", "head_x", "head_y", "head_z", "steps_queued"};
        return n;
    }
    void get_state(float* o) const {                       // world metres / radians; foot_state 0 planted, 1 swing, 2 air
        int i = 0; auto v3 = [&](Vec3 v) { o[i++] = v.x; o[i++] = v.y; o[i++] = v.z; };
        v3(pelvis_); o[i++] = yaw_; v3(com_); o[i++] = vcom_.x; o[i++] = vcom_.z; o[i++] = margin_; o[i++] = fallen_ ? 1.f : 0.f;
        for (int s = 0; s < 2; ++s) { v3(feet_[s].sole); o[i++] = feet_[s].yaw; o[i++] = (float)(int)feet_[s].state; o[i++] = feet_[s].heel; }
        v3(wrist_[0]); v3(wrist_[1]); o[i++] = hands_[0].target_mode ? 1.f : 0.f; o[i++] = hands_[1].target_mode ? 1.f : 0.f; v3(head_center_); o[i++] = (float)steps_queued();
    }
    // action layout (ACTION_DIM floats): pelvis xyz, yaw | footL xyz yaw contact | footR xyz yaw contact | handL xyz on | handR xyz on | look xyz on | torso xyz on
    // report layout (REPORT_DIM floats, metres), by index: 0,1 slide removed L,R | 2,3 sink removed L,R | 4 pelvis moved by leg reach | 5,6 leg unreachable L,R |
    //                                                       7 reserved (0) | 8,9 hand unreachable L,R | 10 balance margin (negative = outside the support area) | 11 any-violation (0/1)
    void track(const float* a, float dt, float* r) {
        using namespace body_detail;
        if (!(dt > 0.f)) dt = 1.f / 60.f; dt = std::min(dt, 0.05f);
        if (!tracking_) { tracking_ = true; queue_.clear(); act_.on = false; flight_ = Flight::None; fallen_ = false; pelvis_user_ = false; fl_has_tgt_[0] = fl_has_tgt_[1] = false;
            com_init_ = false; pvel_ = Vec3(); }      // start with no velocity history: the first frame snaps the pose (lean, twist), and that jump is not a CoM velocity
        fallen_ = false; time_ += dt; for (int i = 0; i < REPORT_DIM; ++i) r[i] = 0.f;
        Vec3 pel(a[0], a[1], a[2]); float yaw = a[3];
        const float* fa[2] = {a + 4, a + 9};
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; Vec3 want(fa[s][0], fa[s][1], fa[s][2]); float wyaw = fa[s][3]; bool contact = fa[s][4] > 0.5f;
            track_fv_[s] = (want - f.sole) * (1.f / dt);
            float ground = surface_at(want.x, want.z, f.surface);
            if (want.y < ground) { r[2 + s] = ground - want.y; want.y = ground; }
            if (contact && f.state == FootState::Planted && !track_free_[s]) {          // a planted foot does not slide: keep where it is
                r[0 + s] = std::hypot(want.x - f.sole.x, want.z - f.sole.z);
                f.surface = ground; f.sole.y = ground;
            } else { f.sole = want; f.yaw = wyaw; f.surface = ground; }
            if (contact && std::fabs(want.y - ground) < 0.03f) { f.state = FootState::Planted; f.sole.y = ground; track_free_[s] = false; }
            else { f.state = FootState::Swing; track_free_[s] = true; }
            f.heel = 0.f; f.pitch = 0.f; f.toe = 0.f;
        }
        yaw_ = yaw; lean_ = head_dev_ = V2{0, 0};
        {   // anatomy the generator does not have to spell out: the hips twist with the legs, the arms swing, a trailing heel rises when the leg would overstretch
            Vec3 fw = fwd(yaw_); float zl = (feet_[0].sole.x - pel.x) * fw.x + (feet_[0].sole.z - pel.z) * fw.z, zr = (feet_[1].sole.x - pel.x) * fw.x + (feet_[1].sole.z - pel.z) * fw.z;
            float tw = clampf(-params.twist_gain * clampf(zl - zr, -0.7f, 0.7f), -0.14f, 0.14f); damp1(twist_, twist_v_, tw, 12.f, dt); chest_twist_ = -1.1f * twist_;
            for (int s = 0; s < 2; ++s) { const Foot& o = feet_[1 - s]; float fo = (o.sole.x - pel.x) * fw.x + (o.sole.z - pel.z) * fw.z; damp1(arm_swing_[s], arm_swing_v_[s], clampf(params.arm_swing_gain * fo, -0.16f * params.scale, 0.16f * params.scale), 9.f, dt); }
            for (int s = 0; s < 2; ++s) {
                Foot& f = feet_[s]; f.heel = 0.f; if (f.state != FootState::Planted && f.sole.y > f.surface + 0.15f) continue;
                for (float hh = 0.f; hh <= 0.6f; hh += 0.1f) { f.heel = hh; if ((hip_pos(s, pel, hyaw()) - ankle_of(f)).length() <= 0.995f * rmax_leg_) break; }
                if (f.state == FootState::Planted) f.pitch = -f.heel;
            }
        }
        Vec3 np_ = pel; clamp_reach(np_, hyaw()); r[4] = (np_ - pel).length();
        pvel_ = Vec3((np_.x - pelvis_.x) / dt, (np_.y - pelvis_.y) / dt, (np_.z - pelvis_.z) / dt); pelvis_ = np_;
        const float* ha[2] = {a + 14, a + 18};
        for (int s = 0; s < 2; ++s) {
            bool on = ha[s][3] > 0.5f; hands_[s].target_mode = on; hands_[s].has_cmd = on; if (on) hands_[s].cmd = Vec3(ha[s][0], ha[s][1], ha[s][2]);
        }
        look_user_ = a[25] > 0.5f; if (look_user_) look_tgt_ = Vec3(a[22], a[23], a[24]);
        torso_user_ = a[29] > 0.5f; if (torso_user_) torso_tgt_ = Vec3(a[26], a[27], a[28]);
        float keep = params.head_turn_omega; params.head_turn_omega = 0.f;           // no head lag while tracking: the generator already made it smooth
        pose_upper(dt, false);
        for (int s = 0; s < 2; ++s) if (hands_[s].target_mode) { arms_.teleport_to(arm_id_[s], hands_[s].cmd); }
        arms_.update(1e-4f);
        for (int s = 0; s < 2; ++s) { Vec3 p[3]; arms_.chain(arm_id_[s]).world_positions(p); shoulder_[s] = p[0]; elbow_[s] = p[1]; wrist_[s] = p[2]; if (hands_[s].target_mode) r[8 + s] = (p[2] - hands_[s].cmd).length(); }
        params.head_turn_omega = keep;
        pose_legs();
        for (int s = 0; s < 2; ++s) r[5 + s] = (feet_[s].ankle - ankle_of(feet_[s])).length();
        compute_bones(); compute_balance(dt);
        const bool unb = margin_ < params.fall_margin; fallen_ = false; unstable_t_ = 0.f;           // tracking reports a loss of balance, it never latches a fall
        r[10] = unb ? -std::fabs(margin_) : margin_; float v = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[8] + r[9]; r[11] = (v > 0.005f || unb) ? 1.f : 0.f;
                report_unbalanced_ = unb;
        track_land_has_[0] = track_land_has_[1] = false;                                  // a hint is for one frame only
    }
    // Optional look-ahead for ONE track() call: where foot s will touch down next. A generator output is a whole clip, so the caller can
    // see this; the balance check then counts the landing footprint (as the walking engine does) instead of guessing from the foot's velocity.
    void track_land(int s, Vec3 p, float yaw) { if (s < 0 || s > 1) return; track_land_has_[s] = true; track_land_[s] = p; track_land_yaw_[s] = yaw; }
    // Track a whole clip (n frames of ACTION_DIM floats): looks ahead for each lifted foot's next touchdown. reports: n*REPORT_DIM, states: n*STATE_DIM
    // (either may be null). Returns the number of frames that reported a loss of balance.
    int track_clip(const float* a, int n, float dt, float* reports, float* states) { return track_clip_each(a, n, dt, reports, states, [](int) {}); }
    // The same, calling each(i) after frame i has been tracked (e.g. to read the bones of that frame).
    template <class Fn> int track_clip_each(const float* a, int n, float dt, float* reports, float* states, Fn&& each) {
        float tmp[REPORT_DIM]; int bad = 0;
        for (int i = 0; i < n; ++i) {
            const float* ai = a + (size_t)i * ACTION_DIM;
            for (int s = 0; s < 2; ++s) {
                auto contact = [&](int k, int f) { return a[(size_t)k * ACTION_DIM + 4 + 5 * f + 4] > 0.5f; };
                int from = i;
                if (contact(i, s)) {            // loaded foot that lifts within ~0.4 s, while the other foot is down: the step is already being prepared (weight shift)
                    if (!contact(i, 1 - s)) continue;
                    int last = std::min(n - 1, i + (int)(0.4f / dt)), k = i + 1; while (k <= last && contact(k, s)) ++k;
                    if (k > last) continue; from = k;
                }
                for (int j = from + 1; j < n; ++j) if (contact(j, s)) { const float* aj = a + (size_t)j * ACTION_DIM + 4 + 5 * s; track_land(s, Vec3(aj[0], aj[1], aj[2]), aj[3]); break; }
            }
            track(ai, dt, reports ? reports + (size_t)i * REPORT_DIM : tmp);
            if (track_unbalanced()) ++bad;
            if (states) get_state(states + (size_t)i * STATE_DIM);
            each(i);
        }
        return bad;
    }

    // ------------------------------------------------------------------ clip evaluation: one number and plain-English reasons
    // What a movement controller / trainer needs from the engine: "is this candidate motion feasible, how badly does it fail, and why".
    struct ClipVerdict {
        bool feasible = true;                 // nothing needed correcting (> 5 mm) and balance held on every frame
        int frames = 0, bad_frames = 0;       // bad: any correction above 5 mm, or a loss of balance
        float score = 0.f;                    // mean per-frame cost in metres (corrections + how far the CoM left the support area): 0 = feasible, lower is better
        int worst_frame = -1; float worst_cost = 0.f;
        std::vector<std::string> issues;      // one plain-English line per run of frames, most severe first (empty when feasible)
    };
    static float frame_cost(const float* r, float fall_margin) {
        float c = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[8] + r[9];
        if (r[10] < fall_margin) c -= r[10];
        return c;
    }
    // Turn n tracking reports into readable issues (frame ranges, times, how much, what the engine did about it).
    std::vector<std::string> explain_reports(const float* rep, int n, float dt) const {
        if (!(dt > 0.f)) dt = 1.f / 60.f; dt = std::min(dt, 0.05f);
        static const char* side[2] = {"left", "right"};
        static const int idx[10] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10};                  // category -> report slot
        auto val = [&](int f, int c) -> float { const float* r = rep + (size_t)f * REPORT_DIM; return c == 9 ? (r[10] < params.fall_margin ? -r[10] : 0.f) : r[idx[c]]; };
        struct Span { float peak; std::string text; };
        std::vector<Span> spans; char b[320];
        for (int c = 0; c < 10; ++c) {
            const float thr = c == 9 ? 0.f : 0.005f;
            for (int f = 0; f < n; ) {
                if (!(val(f, c) > thr)) { ++f; continue; }
                int a = f, last = f; float peak = val(f, c);
                for (int g = f + 1; g < n && g <= last + 4; ++g) if (val(g, c) > thr) { last = g; peak = std::max(peak, val(g, c)); }   // runs split by a gap of up to 4 frames stay one issue
                f = last + 1;
                char when[64]; std::snprintf(when, sizeof when, "frames %d-%d (%.2f-%.2f s)", a, last, a * dt, (last + 1) * dt);
                const float cm = peak * 100.f; const char* sd = c < 2 ? side[c] : c < 4 ? side[c - 2] : c < 7 ? side[c - 5] : c < 9 ? side[c - 7] : "";
                switch (c) {
                    case 0: case 1: std::snprintf(b, sizeof b, "%s: the %s foot is marked in contact but the clip moves it up to %.1f cm from where it is planted; a planted foot cannot slide, so it was held in place", when, sd, cm); break;
                    case 2: case 3: std::snprintf(b, sizeof b, "%s: the %s foot goes up to %.1f cm below the ground; it was lifted onto the surface", when, sd, cm); break;
                    case 4: std::snprintf(b, sizeof b, "%s: the pelvis is up to %.1f cm farther from the planted feet than the legs can reach; it was pulled back", when, cm); break;
                    case 5: case 6: std::snprintf(b, sizeof b, "%s: the %s leg cannot reach its foot (short by up to %.1f cm); the foot is too far from the hip for the leg length", when, sd, cm); break;
                    case 7: case 8: std::snprintf(b, sizeof b, "%s: the %s hand target is out of the arm's reach by up to %.1f cm", when, sd, cm); break;
                    default: std::snprintf(b, sizeof b, "%s: balance is lost; the centre of mass is up to %.1f cm outside the support area, so the body would fall", when, cm); break;
                }
                spans.push_back({peak, b});
            }
        }
        std::stable_sort(spans.begin(), spans.end(), [](const Span& x, const Span& y) { return x.peak > y.peak; });
        std::vector<std::string> out; const size_t cap = 8;
        for (size_t i = 0; i < spans.size() && i < cap; ++i) out.push_back(spans[i].text);
        if (spans.size() > cap) out.push_back("and " + std::to_string(spans.size() - cap) + " more smaller issues");
        return out;
    }
    // Track a clip FROM THE CURRENT STATE and judge it. This body ADVANCES to the clip's end; to try a candidate without committing, use preview_clip.
    // reports (n*REPORT_DIM) and states (n*STATE_DIM) are optional.
    ClipVerdict evaluate_clip(const float* a, int n, float dt, float* reports = nullptr, float* states = nullptr) {
        ClipVerdict v; v.frames = n < 0 ? 0 : n; if (n <= 0 || !a) return v;
        for (size_t i = 0; i < (size_t)n * ACTION_DIM; ++i) if (!std::isfinite(a[i])) {
            v.feasible = false; v.bad_frames = n; v.score = 1e3f; v.worst_frame = (int)(i / ACTION_DIM);
            v.issues.push_back("frame " + std::to_string(i / ACTION_DIM) + " has a non-finite number (NaN or infinity) in action value " + std::to_string(i % ACTION_DIM) + "; the clip was not tracked");
            return v;
        }
        std::vector<float> own; float* r = reports; if (!r) { own.resize((size_t)n * REPORT_DIM); r = own.data(); }
        track_clip(a, n, dt, r, states);
        float sum = 0.f;
        for (int i = 0; i < n; ++i) {
            const float* ri = r + (size_t)i * REPORT_DIM; float c = frame_cost(ri, params.fall_margin);
            sum += c; if (ri[11] > 0.5f) ++v.bad_frames; if (c > v.worst_cost || v.worst_frame < 0) { v.worst_cost = c; v.worst_frame = i; }
        }
        v.score = sum / (float)n; v.feasible = v.bad_frames == 0; if (!v.feasible) v.issues = explain_reports(r, n, dt);
        return v;
    }
    // Try a candidate clip from the current state on a COPY: this body is left exactly as it was.
    ClipVerdict preview_clip(const float* a, int n, float dt, float* reports = nullptr, float* states = nullptr) const { Body t(*this); return t.evaluate_clip(a, n, dt, reports, states); }

    // Leave tracking: the commanded-control state (feet planted where they are, hands relaxed, gait controller back on) resumes from the current pose.
    void track_end() {
        if (!tracking_) return; tracking_ = false;
        for (int s = 0; s < 2; ++s) { Foot& f = feet_[s]; f.state = FootState::Planted; f.sole.y = f.surface = surface_at(f.sole.x, f.sole.z, f.surface); track_free_[s] = false; hands_[s].target_mode = false; hands_[s].has_cmd = false; }
        look_user_ = false; torso_user_ = false; dyn_init_ = false; com_init_ = false; track_land_has_[0] = track_land_has_[1] = false;
        pvel_ = Vec3(); vcom_ = {};      // resume from rest: the last tracked frame's pelvis velocity can be a huge artefact of a correction, and the gait controller starts its capture-point state from it
    }
    bool tracking() const { return tracking_; }
    bool track_unbalanced() const { return report_unbalanced_; }

    // ------------------------------------------------------------------ queries
    const std::vector<BoneOut>& bones() const { return bones_; }
    int bone_index(const std::string& n) const { for (size_t i = 0; i < names_->size(); ++i) if ((*names_)[i] == n) return (int)i; return -1; }
    Vec3 pelvis() const { return pelvis_; }
    float yaw() const { return yaw_; }
    Vec3 com() const { return com_; }
    float margin() const { return margin_; }
    bool fallen() const { return fallen_; }
    Vec3 foot_sole(int s) const { return feet_[s].sole; }
    float foot_yaw(int s) const { return feet_[s].yaw; }
    FootState foot_state(int s) const { return feet_[s].state; }
    Vec3 wrist(int s) const { return wrist_[s]; }
    Vec3 hand_target(int s) const { return hands_[s].cmd; }
    bool hand_targeted(int s) const { return hands_[s].target_mode; }
    bool hand_settled(int s) const { return arms_.is_settled(arm_id_[s]); }
    bool airborne() const { return flight_ == Flight::Air; }          // feet off the ground (the push-off before it still has both feet planted)
    bool speed_ok_for_test() const { return std::hypot(pvel_.x, pvel_.z) < 0.05f; }
    bool stepping() const { return act_.on || !queue_.empty(); }
    int steps_queued() const { return (int)queue_.size() + (act_.on ? 1 : 0); }
    float time() const { return time_; }
    float leg_reach_max() const { return rmax_leg_; }
    int bone_count() const { return (int)bones_.size(); }
    std::vector<std::string> take_events() { auto e = std::move(events_); events_.clear(); return e; }
    // plant/unplant check used by tests: contact points of a planted foot on the ground plane
    std::vector<Vec3> foot_contact_points(int s) const {
        const Foot& f = feet_[s]; Vec3 fw = body_detail::fwd(f.yaw), lf = body_detail::lft(f.yaw);
        std::vector<Vec3> p;
        for (float lat : {D.heel_hw, -D.heel_hw}) p.push_back(f.sole - fw * D.heel + lf * lat);
        for (float lat : {D.ball_hw, -D.ball_hw}) p.push_back(f.sole + fw * D.ball + lf * lat);
        return p;
    }

    std::string status_json() {
        std::string j; char b[256];
        auto v3 = [&](Vec3 v) { std::snprintf(b, sizeof b, "[%.3f,%.3f,%.3f]", v.x, v.y, v.z); return std::string(b); };
        std::snprintf(b, sizeof b, "{\"t\":%.2f,\"state\":\"%s\",\"balance\":\"%s\",\"margin_cm\":%.1f,", time_, fallen_ ? "fallen" : flight_ == Flight::Air ? "airborne" : "standing", params.balance == Balance::Assist ? "assist" : "none", std::max(margin_, -999.f) * 100.f);
        j += b; j += "\"pelvis\":" + v3(pelvis_) + ",\"com\":" + v3(com_) + ",";
        std::snprintf(b, sizeof b, "\"yaw_deg\":%.1f,\"steps_queued\":%d,", yaw_ * 180.f / PI, steps_queued()); j += b;
        j += "\"feet\":{";
        for (int s = 0; s < 2; ++s) {
            const char* st = feet_[s].state == FootState::Planted ? "planted" : feet_[s].state == FootState::Swing ? "swinging" : "air";
            std::snprintf(b, sizeof b, "\"%s\":{\"state\":\"%s\",\"yaw_deg\":%.1f,\"pos\":", side_name(s).c_str(), st, feet_[s].yaw * 180.f / PI); j += b; j += v3(feet_[s].sole) + "}"; if (s == 0) j += ",";
        }
        j += "},\"hands\":{";
        for (int s = 0; s < 2; ++s) {
            j += "\"" + side_name(s) + "\":{\"pos\":" + v3(wrist_[s]);
            if (hands_[s].target_mode) { j += ",\"target\":" + v3(hands_[s].cmd); std::snprintf(b, sizeof b, ",\"error_cm\":%.1f", (wrist_[s] - hands_[s].cmd).length() * 100.f); j += b; }
            j += "}"; if (s == 0) j += ",";
        }
        j += "},\"events\":[";
        auto ev = take_events();
        for (size_t i = 0; i < ev.size(); ++i) { std::string e = ev[i]; for (char& ch : e) if (ch == '"' || ch == '\\') ch = '\''; j += (i ? ",\"" : "\"") + e + "\""; }
        j += "]}";
        return j;
    }

private:
    // ------------------------------------------------------------------ dimensions (metres at scale 1, ~1.75 m tall)
    struct Dim {
        float thigh, shin, ankle_h, heel, ball, toe_len, ball_hw, heel_hw, hip_w, pelvis_bone, neck, head, clav, clav_attach, upper, fore, hand;
        float spine[3]; float fingers[5];
        explicit Dim(float s = 1.f) : thigh(.43f * s), shin(.43f * s), ankle_h(.08f * s), heel(.06f * s), ball(.15f * s), toe_len(.06f * s), ball_hw(.045f * s), heel_hw(.03f * s),
            hip_w(.10f * s), pelvis_bone(.06f * s), neck(.10f * s), head(.22f * s), clav(.17f * s), clav_attach(.02f * s), upper(.30f * s), fore(.27f * s), hand(.09f * s),
            spine{.13f * s, .15f * s, .14f * s}, fingers{.075f * s, .092f * s, .100f * s, .092f * s, .075f * s} {}
    } D;

    struct Foot {
        FootState state = FootState::Planted;
        Vec3 sole; float yaw = 0.f, pitch = 0.f, toe = 0.f, surface = 0.f, heel = 0.f;   // heel: rad the heel is raised while the ball stays on the ground (toe-off)
        Trajectory traj; float yaw0 = 0.f, yaw1 = 0.f; Vec3 end_sole; bool end_planted = true; float end_surface = 0.f;
        Chain leg; Vec3 hip, knee, ankle;
    };
    struct Hand { bool target_mode = false, has_cmd = false; Vec3 cmd, last; bool has_last = false; float curl = 0.f; };
    struct StepCmd { int side; Vec3 target; bool has_yaw; float yaw; };
    struct Active { bool on = false; int side = 0; StepCmd cmd{0, Vec3(), false, 0.f}; enum { Shift, Swing } phase = Shift; float t = 0.f; };

    Foot feet_[2]; Hand hands_[2];
    Vec3 pelvis_, pelvis_prev_, pvel_; float yaw_ = 0.f;
    bool pelvis_user_ = false, face_user_ = false, look_user_ = false, torso_user_ = false, turn_warned_ = false, look_warned_ = false;
    Vec3 pelvis_tgt_, face_pt_, look_tgt_, torso_tgt_;
    Chain spine_; IKEngine arms_; ChainId arm_id_[2] = {kNoChain, kNoChain};
    Vec3 shoulder_[2], elbow_[2], wrist_[2], head_center_;
    float rmin_leg_ = 0.f, rmax_leg_ = 0.f;
    std::vector<Surface> surfaces_;
    std::deque<StepCmd> queue_; Active act_;
    bool fallen_ = false; float unstable_t_ = 0.f, time_ = 0.f, fall_t_ = 0.f;
    bool tracking_ = false, track_free_[2] = {false, false}, report_unbalanced_ = false; Vec3 track_fv_[2];
    bool track_land_has_[2] = {false, false}; Vec3 track_land_[2]; float track_land_yaw_[2] = {0.f, 0.f};     // where a lifted foot will land (hint for the next track() call)
    enum class Flight { None, Push, Air } flight_ = Flight::None;
    Vec3 fl_v_, fl_prev_free_; float fl_y0_ = 0.f, fl_push_t_ = 0.f, fl_push_T_ = 0.f, fl_air_t_ = 0.f;
    bool fl_has_tgt_[2] = {false, false}; body_detail::V2 fl_tgt_[2]; bool fl_has_yaw_[2] = {false, false}; float fl_yaw_[2] = {0.f, 0.f};
    Vec3 com_; body_detail::V2 com_off_, vcom_, xcom_; float margin_ = 0.f; bool com_init_ = false; Vec3 com_prev_;
    std::vector<body_detail::V2> support_;
    std::vector<BoneOut> bones_; std::shared_ptr<std::vector<std::string>> names_ = std::make_shared<std::vector<std::string>>(); std::vector<float> mass_; std::vector<std::string> events_;
    struct Idx { int clav[2], upper[2], fore[2], hand[2], finger0[2], thigh[2], shin[2], foot[2], toe[2], nfingers = 0; } I{};
    Quat chest_q_;
    // dynamic gait state
    body_detail::V2 cm_, cmv_, cop_f_, lean_, head_dev_; bool dyn_init_ = false;
    float twist_ = 0.f, twist_v_ = 0.f, chest_twist_ = 0.f, arm_swing_[2] = {0.f, 0.f}, arm_swing_v_[2] = {0.f, 0.f};
    float hl_yaw_ = 0.f, hl_pitch_ = 0.f, hl_chest_ = 0.f, hl_vyaw_ = 0.f, hl_vpitch_ = 0.f, hl_vchest_ = 0.f;
    float hyaw() const { return yaw_ + twist_; }                      // hip-line heading: the body heading plus the walking twist
    bool dyn_on() const { return params.dynamic_gait && params.balance == Balance::Assist && !pelvis_user_; }

    // ------------------------------------------------------------------ helpers
    static std::string upper(std::string s) { for (char& c : s) c = (char)std::toupper((unsigned char)c); return s; }
    static std::string lower(std::string s) { for (char& c : s) c = (char)std::tolower((unsigned char)c); return s; }
    static std::string side_name(int s) { return s == 0 ? "left" : "right"; }
    static std::string fmt(float v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return b; }
    static float sg(int s) { return s == 0 ? 1.f : -1.f; }
    static Result ok(std::string m) { return {true, std::move(m)}; }
    static Result fail(std::string m) { return {false, std::move(m)}; }
    static std::vector<std::string> tokens(const std::string& line) {
        std::vector<std::string> t; std::string cur;
        for (char ch : line) { if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') { if (!cur.empty()) t.push_back(cur), cur.clear(); } else cur += ch; }
        if (!cur.empty()) t.push_back(cur);
        return t;
    }
    static bool parse(const std::string& s, float& out) {
        char* e = nullptr; float v = std::strtof(s.c_str(), &e);
        if (e == s.c_str() || *e || !std::isfinite(v)) return false;
        out = v; return true;
    }
    void event(const std::string& s) { if (events_.size() < 64) events_.push_back(s); }
    static std::string xz(Vec3 p) { char b[64]; std::snprintf(b, sizeof b, "(%.2f, %.2f)", p.x, p.z); return b; }

    using V2 = body_detail::V2;
    static void damp1(float& x, float& v, float target, float omega, float dt) {        // critically damped scalar (same formula as smooth_damp)
        float q = omega * dt, e = 1.f / (1.f + q + 0.48f * q * q + 0.235f * q * q * q), ch = x - target, tmp = (v + ch * omega) * dt;
        v = (v - tmp * omega) * e; x = target + (ch + tmp) * e;
    }

    // ------------------------------------------------------------------ dynamic gait
    // The centre of mass is a linear inverted pendulum pivoting on the centre of pressure (CoP), which can only sit inside the support foot:
    //     cm'' = w^2 (cm - cop),   w = sqrt(g / h),   capture point (DCM)  xi = cm + cm' / w,   xi' = w (xi - cop).
    // The queued footfalls fix where the CoP will be (each stance foot). Integrating the DCM backwards from "come to rest between the last
    // two feet" gives the DCM needed at every touchdown; the controller picks the CoP each frame so the DCM arrives there (dead-beat on the
    // remaining swing time). Nothing stops between steps: the body falls onto the next foot and the next foot catches it. The pelvis is
    // the CoM minus a measured offset; its height follows from a constant-length stance leg (so it bobs), not from a target.
    // ankle joint position: above the sole point when the foot is flat; when the heel is raised the foot pivots about the ball (toe-off)
    Vec3 ankle_of(const Foot& f) const {
        if (f.heel < 1e-4f) return f.sole + Vec3(0, D.ankle_h, 0);
        Vec3 fw = body_detail::fwd(f.yaw); float c = std::cos(f.heel), sn = std::sin(f.heel);
        float a = -D.ball * c + D.ankle_h * sn, u = D.ankle_h * c + D.ball * sn;               // ankle relative to the ball, in (forward, up)
        return f.sole + fw * D.ball + fw * a + Vec3(0, u, 0);
    }
    float dyn_w() const { float h = std::max(0.3f, com_.y - 0.5f * (feet_[0].surface + feet_[1].surface)); return std::sqrt(9.81f / h); }
    V2 cop_ref_of(Vec3 sole, float yaw) const { Vec3 fw = body_detail::fwd(yaw); float o = 0.5f * (D.ball - D.heel); return {sole.x + fw.x * o, sole.z + fw.z * o}; }
    struct GaitPlan { bool steps = false; int stance = 0; V2 c0, xi_s0, xi_td0, mid, mid_land; float T0 = 0.4f; };
    GaitPlan make_plan(float w, float T_first) const {
        struct St { V2 c, land; float T; };
        GaitPlan P; V2 pos[2]; bool air[2] = {false, false};
        for (int s = 0; s < 2; ++s) { pos[s] = cop_ref_of(feet_[s].sole, feet_[s].yaw); air[s] = feet_[s].state == FootState::Air; }
        std::vector<St> L; L.reserve(10);
        auto add = [&](const StepCmd& c, float Tov) {
            float yw = c.has_yaw ? c.yaw : feet_[c.side].yaw; V2 land = cop_ref_of(c.target, yw);
            float h = std::hypot(land.x - pos[c.side].x, land.z - pos[c.side].z);
            L.push_back({pos[1 - c.side], land, Tov > 0.f ? Tov : swing_time(h, 0.f)}); pos[c.side] = land;
            air[c.side] = c.target.y > surface_at(c.target.x, c.target.z, feet_[1 - c.side].surface) + 0.02f;      // a raised foot is held in the air: it is not a support
            if (L.size() == 1) P.stance = 1 - c.side;
        };
        if (act_.on) add(act_.cmd, act_.phase == Active::Swing ? T_first : 0.f);
        for (const StepCmd& c : queue_) if (L.size() < 9) add(c, 0.f);
        { V2 m{0, 0}; int n = 0; for (int s = 0; s < 2; ++s) if (!air[s]) { m.x += pos[s].x; m.z += pos[s].z; ++n; }
          P.mid_land = n ? V2{m.x / n, m.z / n} : V2{0.5f * (pos[0].x + pos[1].x), 0.5f * (pos[0].z + pos[1].z)}; }
        if (L.empty()) {
            V2 m{0, 0}; int n = 0;
            for (int s = 0; s < 2; ++s) if (feet_[s].state == FootState::Planted) { m.x += pos[s].x; m.z += pos[s].z; ++n; }
            P.mid = n ? V2{m.x / n, m.z / n} : P.mid_land; return P;
        }
        P.steps = true; P.mid = P.mid_land; P.c0 = L[0].c; P.T0 = L[0].T;
        V2 xi = P.mid_land; std::vector<V2> td(L.size());
        td[L.size() - 1] = xi;
        for (int i = (int)L.size() - 2; i >= 0; --i) {
            V2 c = L[i + 1].c; float E = std::exp(-w * (params.step_ds_time + L[i + 1].T));
            td[i] = {c.x + (td[i + 1].x - c.x) * E, c.z + (td[i + 1].z - c.z) * E};
        }
        P.xi_td0 = td[0]; float E0 = std::exp(-w * L[0].T);
        P.xi_s0 = {P.c0.x + (td[0].x - P.c0.x) * E0, P.c0.z + (td[0].z - P.c0.z) * E0};
        return P;
    }

    // Where the DCM is steered before the swing foot lifts. From (nearly) standing it must stay over the support foot; the capture
    // happens during the swing. When already moving (continuous walking) the plan value is used as is.
    V2 shift_ref(const GaitPlan& P) const {
        V2 r = P.xi_s0;
        const Foot& rear = feet_[1 - P.stance];                         // the foot about to swing: the CoM cannot be taken further from it than the leg reaches
        if (rear.state == FootState::Planted && std::hypot(cmv_.x, cmv_.z) <= 0.3f) {
            V2 q = cop_ref_of(rear.sole, rear.yaw); float dx = r.x - q.x, dz = r.z - q.z, d = std::hypot(dx, dz), lim = 0.36f * params.scale;
            if (d > lim) { r.x = q.x + dx * lim / d; r.z = q.z + dz * lim / d; }
        }
        if (std::hypot(cmv_.x, cmv_.z) > 0.3f || feet_[P.stance].state != FootState::Planted) return r;
        return body_detail::hull_project(foot_polygon(P.stance), r, 0.02f);
    }

    void update_pelvis_dynamic(float dt, const int* planted, int np) {
        using namespace body_detail;
        float surf = 0.f; for (int i = 0; i < np; ++i) surf += feet_[planted[i]].surface / np;
        float h = std::max(0.3f, com_.y - surf), w = std::sqrt(9.81f / h);
        if (!dyn_init_) { cm_ = {com_.x, com_.z}; cmv_ = {pvel_.x, pvel_.z}; cop_f_ = cm_; dyn_init_ = true; }
        V2 xi{cm_.x + cmv_.x / w, cm_.z + cmv_.z / w};
        const bool ss = act_.on && act_.phase == Active::Swing && feet_[1 - act_.side].state == FootState::Planted;
        float Trem = ss ? std::max(feet_[act_.side].traj.T - feet_[act_.side].traj.t, 0.12f) : 0.f;
        GaitPlan P = make_plan(w, Trem); V2 c;
        if (ss && P.steps) {                                         // single support: dead-beat the DCM onto the planned touchdown value
            float E = std::exp(w * Trem); c = {(P.xi_td0.x - xi.x * E) / (1.f - E), (P.xi_td0.z - xi.z * E) / (1.f - E)};
            c = hull_project(foot_polygon(1 - act_.side), c, 0.012f);
        } else {                                                     // double support / standing: track the DCM reference inside the support polygon
            V2 ref = P.steps ? shift_ref(P) : P.mid; c = {ref.x + 2.5f * (xi.x - ref.x), ref.z + 2.5f * (xi.z - ref.z)};   // classic DCM hold: converge on the reference
            c = hull_project(support_poly(), c, 0.01f);
        }
        float a = 1.f - std::exp(-dt / 0.045f); cop_f_.x += (c.x - cop_f_.x) * a; cop_f_.z += (c.z - cop_f_.z) * a;   // force transfer is not instantaneous
        float ch = std::cosh(w * dt), sh = std::sinh(w * dt);          // exact LIPM step with the CoP held over dt
        V2 e{cm_.x - cop_f_.x, cm_.z - cop_f_.z}, en{e.x * ch + cmv_.x / w * sh, e.z * ch + cmv_.z / w * sh}, vn{e.x * w * sh + cmv_.x * ch, e.z * w * sh + cmv_.z * ch};
        cm_ = {cop_f_.x + en.x, cop_f_.z + en.z}; cmv_ = vn;
        pelvis_prev_ = pelvis_;
        Vec3 np_(cm_.x - com_off_.x, pelvis_.y, cm_.z - com_off_.z);
        // pelvis twist: the hip on the leading-leg side swings forward; the chest counter-rotates
        Vec3 fw = fwd(yaw_);
        float zl = (feet_[0].sole.x - pelvis_.x) * fw.x + (feet_[0].sole.z - pelvis_.z) * fw.z, zr = (feet_[1].sole.x - pelvis_.x) * fw.x + (feet_[1].sole.z - pelvis_.z) * fw.z;
        float tw = -params.twist_gain * clampf(zl - zr, -0.7f, 0.7f); tw = clampf(tw, -0.14f, 0.14f);
        damp1(twist_, twist_v_, tw, 12.f, dt); chest_twist_ = -1.1f * twist_;
        // pelvis height: constant-length stance leg(s). Each planted leg bounds the height; a swinging leg may not be over-extended either.
        float Lcap = params.leg_len_frac * (D.thigh + D.shin), yt = 1e9f;
        for (int i = 0; i < np; ++i) {
            const Foot& f = feet_[planted[i]]; Vec3 H = hip_pos(planted[i], np_, hyaw()), A = ankle_of(f); float dh = std::hypot(H.x - A.x, H.z - A.z);
            yt = std::min(yt, A.y + std::sqrt(std::max(Lcap * Lcap - dh * dh, 0.09f * Lcap * Lcap)));
        }
        float y_swing = 1e9f;                                          // a swinging leg may not be over-extended: y_swing is a hard cap (with swing_reach_slack) because the smoothing below lags behind a landing foot
        for (int s = 0; s < 2; ++s) if (feet_[s].state != FootState::Planted) {
            const Foot& f = feet_[s]; Vec3 H = hip_pos(s, np_, hyaw()); float dh = std::hypot(H.x - f.sole.x, H.z - f.sole.z), r = 0.99f * rmax_leg_;
            float ys = f.sole.y + D.ankle_h + std::sqrt(std::max(r * r - dh * dh, 0.f)); yt = std::min(yt, ys);
            float rs = r + params.swing_reach_slack * params.scale; y_swing = std::min(y_swing, f.sole.y + D.ankle_h + std::sqrt(std::max(rs * rs - dh * dh, 0.f)));
        }
        { Vec3 cy(0, pelvis_.y, 0), ty(0, yt, 0), vy(0, pvel_.y, 0); smooth_damp(cy, ty, vy, 18.f, dt); float dy = clampf(cy.y - pelvis_.y, -1.5f * params.scale * dt, 1.5f * params.scale * dt); np_.y = std::min(pelvis_.y + dy, y_swing); }
        clamp_reach(np_, hyaw());
        pvel_ = Vec3((np_.x - pelvis_prev_.x) / dt, (np_.y - pelvis_prev_.y) / dt, (np_.z - pelvis_prev_.z) / dt);
        pelvis_ = np_; cm_ = {pelvis_.x + com_off_.x, pelvis_.z + com_off_.z};
        for (int s = 0; s < 2; ++s) {                                  // toe-off: the trailing foot rolls onto its ball while the other foot is down or landing
            Foot& f = feet_[s]; float want = 0.f;
            if (f.state == FootState::Planted) {
                V2 r = cop_ref_of(f.sole, f.yaw); float rel = (r.x - pelvis_.x) * fw.x + (r.z - pelvis_.z) * fw.z;
                want = clampf((-rel - 0.10f) / 0.22f, 0.f, 1.f) * 0.55f;
            }
            f.heel += (want - f.heel) * (1.f - std::exp(-dt * 18.f)); if (f.state == FootState::Planted) f.pitch = -f.heel;
        }
        // secondary motion (cosmetic physics): trunk leans along the CoM-CoP line, the head cancels part of the lateral sway, hands swing opposite the legs
        V2 lt{(cm_.x - cop_f_.x) / h * 0.25f + cmv_.x * 0.05f, (cm_.z - cop_f_.z) / h * 0.25f + cmv_.z * 0.05f};
        float lm = std::hypot(lt.x, lt.z); if (lm > 0.25f) { lt.x *= 0.25f / lm; lt.z *= 0.25f / lm; }
        float k = 1.f - std::exp(-dt * 10.f); lean_.x += (lt.x - lean_.x) * k; lean_.z += (lt.z - lean_.z) * k;
        V2 r0 = cop_ref_of(feet_[0].state == FootState::Swing && act_.on ? feet_[0].end_sole : feet_[0].sole, feet_[0].yaw), r1 = cop_ref_of(feet_[1].state == FootState::Swing && act_.on ? feet_[1].end_sole : feet_[1].sole, feet_[1].yaw);
        Vec3 lf = lft(yaw_); float lat = (pelvis_.x - 0.5f * (r0.x + r1.x)) * lf.x + (pelvis_.z - 0.5f * (r0.z + r1.z)) * lf.z;
        head_dev_ = {lf.x * -0.5f * lat, lf.z * -0.5f * lat};
        for (int s = 0; s < 2; ++s) {
            const Foot& o = feet_[1 - s]; float fo = (o.sole.x - pelvis_.x) * fw.x + (o.sole.z - pelvis_.z) * fw.z;
            float tg = clampf(params.arm_swing_gain * fo, -0.16f * params.scale, 0.16f * params.scale); damp1(arm_swing_[s], arm_swing_v_[s], tg, 9.f, dt);
        }
    }

    // ------------------------------------------------------------------ construction
    void build() {
        params.finger_lod = std::max(0, std::min(2, params.finger_lod));
        params.scale = std::max(0.2f, std::min(3.f, params.scale));
        D = Dim(params.scale);
        two_bone_reach_limits(D.thigh, D.shin, 30.f * PI / 180.f, 175.f * PI / 180.f, rmin_leg_, rmax_leg_);
        // bone table
        names_->reserve(64); mass_.reserve(64); std::vector<int> parents; std::vector<float> masses;
        auto add = [&](const std::string& n, int par, float m) { names_->push_back(n); parents.push_back(par); masses.push_back(m); return (int)names_->size() - 1; };
        int pel = add("pelvis", -1, .145f), sp1 = add("spine1", pel, .125f), sp2 = add("spine2", sp1, .125f), sp3 = add("spine3", sp2, .102f);
        int nk = add("neck", sp3, .012f); add("head", nk, .069f);
        const char* S[2] = {"L", "R"};
        for (int s = 0; s < 2; ++s) {
            std::string u = std::string("_") + S[s];
            I.clav[s] = add("clavicle" + u, sp3, .005f); I.upper[s] = add("upper_arm" + u, I.clav[s], .028f);
            I.fore[s] = add("forearm" + u, I.upper[s], .016f); I.hand[s] = add("hand" + u, I.fore[s], .006f);
            I.finger0[s] = (int)names_->size();
            if (params.finger_lod == 1) { for (const char* f : {"thumb", "index", "middle", "ring", "pinky"}) add(std::string(f) + u, I.hand[s], 0.f); }
            else if (params.finger_lod == 2) {
                for (int k = 1; k <= 2; ++k) add("thumb_" + std::to_string(k) + u, k == 1 ? I.hand[s] : (int)names_->size() - 1, 0.f);
                for (const char* f : {"index", "middle", "ring", "pinky"}) { int prev = I.hand[s]; for (int k = 1; k <= 3; ++k) prev = add(std::string(f) + "_" + std::to_string(k) + u, prev, 0.f); }
            }
        }
        I.nfingers = params.finger_lod == 0 ? 0 : (params.finger_lod == 1 ? 5 : 14);
        for (int s = 0; s < 2; ++s) {
            std::string u = std::string("_") + S[s];
            I.thigh[s] = add("thigh" + u, pel, .100f); I.shin[s] = add("shin" + u, I.thigh[s], .0465f);
            I.foot[s] = add("foot" + u, I.shin[s], .0145f); I.toe[s] = add("toe" + u, I.foot[s], .002f);
        }
        float M = 0; for (float m : masses) M += m;
        for (float& m : masses) m /= M;
        mass_ = masses;
        bones_.resize(names_->size());
        for (size_t i = 0; i < names_->size(); ++i) { bones_[i].name = (*names_)[i].c_str(); bones_[i].parent = parents[i]; }

        // pose: standing, feet under the hips, legs slightly bent
        yaw_ = 0.f;
        float standing_y = D.ankle_h + 0.985f * (D.thigh + D.shin);
        pelvis_ = Vec3(0, standing_y, 0); pelvis_prev_ = pelvis_;
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; f.sole = Vec3(sg(s) * D.hip_w, 0, 0); f.yaw = 0; f.state = FootState::Planted; f.surface = 0;
            f.leg.count = 2; f.leg.joints[0].bone_length = D.thigh; f.leg.joints[1].bone_length = D.shin;
        }
        spine_.count = 3; for (int i = 0; i < 3; ++i) { spine_.joints[i].bone_length = D.spine[i]; spine_.joints[i].set_max_bend(22.f * PI / 180.f); }
        arms_.params.reaction_delay = 0.f; arms_.params.breath_amp_m = 0.f; arms_.params.idle_amp_m = params.hand_idle_m;
        for (int s = 0; s < 2; ++s) {
            arm_id_[s] = arms_.add_chain(side_name(s), Vec3(), Quat::identity(), {{"upper", D.upper}, {"fore", D.fore}}, SolverType::TwoBoneAnalytic, Vec3(0, -1, -0.5f), 0.f, 1.8f);
            arms_.set_elbow_limits(side_name(s), 25.f * PI / 180.f, 175.f * PI / 180.f);
        }
        pose_upper(0.f, true);
        pose_legs(); compute_bones(); compute_balance(1e-3f);
        { V2 m{0.5f * (cop_ref_of(feet_[0].sole, 0.f).x + cop_ref_of(feet_[1].sole, 0.f).x), cop_ref_of(feet_[0].sole, 0.f).z};   // start with the CoM over the feet
          pelvis_.x += m.x - com_.x; pelvis_.z += m.z - com_.z; pelvis_prev_ = pelvis_; pose_upper(0.f, true); pose_legs(); compute_bones(); com_init_ = false; compute_balance(1e-3f); }
    }

    // ------------------------------------------------------------------ commands
    Result cmd_hand(int s, Vec3 target) {
        if (fallen_) return fail("HAND rejected: the body has fallen; STAND first");
        float dist = (target - shoulder_[s]).length(), reach = (D.upper + D.fore) * 0.985f;
        hands_[s].target_mode = true; hands_[s].has_cmd = true; hands_[s].cmd = target;
        arms_.reach_to(arm_id_[s], target);
        char b[160];
        if (dist > reach) { std::snprintf(b, sizeof b, "%s hand moving; target is %.2f m from the shoulder but the arm reaches %.2f m: it will stretch to its limit", side_name(s).c_str(), dist, reach); return ok(b); }
        std::snprintf(b, sizeof b, "%s hand moving, %.2f m from the shoulder", side_name(s).c_str(), dist); return ok(b);
    }

    Result cmd_foot(int s, Vec3 target, bool has_yaw, float yaw) {
        if (fallen_) return fail("FOOT rejected: the body has fallen; STAND first");
        if (flight_ != Flight::None) {
            fl_has_tgt_[s] = true; fl_tgt_[s] = {target.x, target.z}; fl_has_yaw_[s] = has_yaw; fl_yaw_[s] = yaw;
            char b[140]; std::snprintf(b, sizeof b, "%s foot will land near (%.2f, %.2f) if the leg can reach it", side_name(s).c_str(), target.x, target.z); return ok(b);
        }
        if (queue_.size() >= 8) return fail("FOOT rejected: 8 steps are already queued");
        if (std::hypot(target.x - pelvis_.x, target.z - pelvis_.z) > 3.f) return fail("FOOT rejected: target is more than 3 m from the body; step toward it in stages");
        queue_.push_back({s, target, has_yaw, yaw});
        char b[120]; std::snprintf(b, sizeof b, "%s foot step queued (%d ahead)", side_name(s).c_str(), (int)queue_.size() + (act_.on ? 1 : 0) - 1); return ok(b);
    }

    Result stand() {
        fallen_ = false; unstable_t_ = 0.f; queue_.clear(); act_.on = false; pelvis_user_ = false; flight_ = Flight::None; fl_has_tgt_[0] = fl_has_tgt_[1] = false;
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; Vec3 l = body_detail::lft(yaw_);
            f.sole = Vec3(pelvis_.x, 0, pelvis_.z) + l * (sg(s) * D.hip_w); f.yaw = yaw_; f.pitch = 0; f.toe = 0;
            f.surface = surface_at(f.sole.x, f.sole.z, std::max(feet_[0].surface, feet_[1].surface)); f.sole.y = f.surface; f.state = FootState::Planted;
        }
        pelvis_.y = std::max(feet_[0].surface, feet_[1].surface) + D.ankle_h + 0.985f * (D.thigh + D.shin); pvel_ = Vec3();
        com_init_ = false; dyn_init_ = false;
        event("stood up (no fall or get-up animation: the pose was reset)");
        return ok("standing");
    }

    // ------------------------------------------------------------------ jumping (coordinates in, physics out)
    // The model says WHERE to land; the engine solves the ballistic arc and refuses what the legs cannot do. There is no jump clip:
    // the legs push through a stroke (the crouch the caller set with PELVIS + an ankle push), the body then flies on a parabola, and the
    // feet meet whatever surface is under them. A landing the balance model cannot hold ends in a fall (BALANCE none shows this plainly).
    float surface_below(float x, float z, float ref_y) const {                 // highest surface whose top is at or just under ref_y
        float y = 0.f;
        const float tol = 0.06f * params.scale;               // a foot that arrives within 6 cm of a top lands on it (edge scrape)
        for (const auto& s : surfaces_) if (s.y <= ref_y + tol && x >= s.x0 && x <= s.x1 && z >= s.z0 && z <= s.z1) y = std::max(y, s.y);
        return y;
    }
    struct JumpPlan { bool ok; std::string why; float stroke, vy, T, vmax_y; Vec3 v; float y_launch, y_land; };
    JumpPlan plan_jump(Vec3 tgt, bool has_apex, float clear) const {
        JumpPlan P{}; char b[320];
        float L = D.thigh + D.shin, sc = params.scale;
        float surf0 = 0.5f * (feet_[0].surface + feet_[1].surface);
        float y_ext = surf0 + D.ankle_h + 0.985f * L, y0 = pelvis_.y;
        P.y_launch = y_ext + params.ankle_push * sc;
        P.stroke = std::max(0.02f, P.y_launch - y0);
        P.vmax_y = std::sqrt(2.f * params.jump_accel * P.stroke);
        float land_s = surface_below(tgt.x, tgt.z, tgt.y);
        P.y_land = land_s + D.ankle_h + 0.985f * L;
        const float g = 9.81f;
        float dxh = tgt.x - pelvis_.x, dzh = tgt.z - pelvis_.z, dh = std::hypot(dxh, dzh);
        float base = std::max(P.y_launch, P.y_land), clr = clear;
        if (!has_apex) {                                   // default arc: about 45 degrees (the cheapest in take-off speed), as high as the legs can give
            float avail = P.y_launch + 0.5f * P.vmax_y * P.vmax_y / g - base;     // clearance the stroke can pay for
            clr = std::min(std::max(0.10f * sc, dh / 4.f), avail);
            if (clr < 0.005f) {
                std::snprintf(b, sizeof b, "JUMP rejected: the legs cannot lift the body to that landing height: a %.0f cm push stroke reaches %.2f m for the pelvis, the landing needs %.2f m (crouch deeper with PELVIS)", P.stroke * 100.f, P.y_launch + 0.5f * P.vmax_y * P.vmax_y / g, base); P.why = b; return P;
            }
        }
        float apex = base + clr;
        P.vy = std::sqrt(2.f * g * std::max(0.f, apex - P.y_launch));
        float tu = P.vy / g, td = std::sqrt(2.f * std::max(0.f, apex - P.y_land) / g); P.T = tu + td;
        float dx = tgt.x - pelvis_.x, dz = tgt.z - pelvis_.z;
        P.v = Vec3(P.T > 1e-4f ? dx / P.T : 0.f, P.vy, P.T > 1e-4f ? dz / P.T : 0.f);
        float sp = std::sqrt(P.v.x * P.v.x + P.v.y * P.v.y + P.v.z * P.v.z), vh = std::hypot(P.v.x, P.v.z);
        P.ok = false;
        if (P.vy > P.vmax_y + 1e-4f) {
            std::snprintf(b, sizeof b, "JUMP rejected: that landing needs %.2f m/s upward take-off but a %.0f cm push stroke gives at most %.2f m/s (crouch deeper with PELVIS to push through a longer stroke, or lower the apex)", P.vy, P.stroke * 100.f, P.vmax_y); P.why = b; return P;
        }
        if (sp > params.max_launch_speed + 1e-4f) {
            float lo = 0.f, hi = 1.f;                                         // how far along the same direction can it go?
            for (int i = 0; i < 20; ++i) { float k = 0.5f * (lo + hi); float vx = P.v.x * k, vz = P.v.z * k; if (std::sqrt(vx * vx + P.v.y * P.v.y + vz * vz) <= params.max_launch_speed) lo = k; else hi = k; }
            std::snprintf(b, sizeof b, "JUMP rejected: %.2f m horizontally in %.2f s needs a %.2f m/s take-off, the legs give at most %.2f m/s; about %.2f m is reachable in that direction", std::hypot(dx, dz), P.T, sp, params.max_launch_speed, lo * std::hypot(dx, dz)); P.why = b; return P;
        }
        (void)vh; P.ok = true; return P;
    }

    Result cmd_jump(Vec3 tgt, bool has_apex, float clear) {
        if (fallen_) return fail("JUMP rejected: the body has fallen; STAND first");
        if (flight_ != Flight::None) return fail("JUMP rejected: already in the air");
        if (stepping()) return fail("JUMP rejected: finish the queued steps first (steps are still queued or moving)");
        if (feet_[0].state != FootState::Planted || feet_[1].state != FootState::Planted) return fail("JUMP rejected: both feet must be on the ground to push off");
        if (std::hypot(tgt.x - pelvis_.x, tgt.z - pelvis_.z) > 4.f) return fail("JUMP rejected: target is more than 4 m away");
        JumpPlan P = plan_jump(tgt, has_apex, clear);
        if (!P.ok) return fail(P.why);
        dyn_init_ = false; flight_ = Flight::Push; fl_v_ = P.v; fl_y0_ = pelvis_.y; fl_push_t_ = 0.f; fl_push_T_ = std::max(0.05f, 2.f * P.stroke / std::max(P.vy, 0.5f));
        fl_has_tgt_[0] = fl_has_tgt_[1] = false; queue_.clear(); act_.on = false;
        char b[220]; std::snprintf(b, sizeof b, "jumping: push-off %.0f cm over %.2f s, take-off %.2f m/s up and %.2f m/s forward, in the air about %.2f s",
                                   P.stroke * 100.f, fl_push_T_, P.vy, std::hypot(P.v.x, P.v.z), P.T);
        return ok(b);
    }

    void update_flight(float dt) {
        using namespace body_detail;
        const float g = 9.81f, L = D.thigh + D.shin;
        pelvis_prev_ = pelvis_;
        float want_yaw = face_user_ ? std::atan2(face_pt_.x - pelvis_.x, face_pt_.z - pelvis_.z) : yaw_;
        yaw_ += clampf(wrap(want_yaw - yaw_), -3.f * dt, 3.f * dt);
        if (flight_ == Flight::Push) {
            fl_push_t_ += dt; float tau = std::min(fl_push_t_, fl_push_T_);
            float a = fl_v_.y / fl_push_T_;
            pelvis_.y = fl_y0_ + 0.5f * a * tau * tau; pvel_ = Vec3(0, a * tau, 0);
            clamp_push_reach();
            if (fl_push_t_ >= fl_push_T_) {
                flight_ = Flight::Air; fl_air_t_ = 0.f; pvel_ = fl_v_;
                for (int s = 0; s < 2; ++s) feet_[s].state = FootState::Air;
                fl_prev_free_ = Vec3(0, pelvis_.y - D.ankle_h - 0.985f * L, 0);
                event("airborne");
            }
            return;
        }
        fl_air_t_ += dt;
        fl_v_.y -= g * dt; pelvis_ += fl_v_ * dt; pvel_ = fl_v_;
        float free_y = pelvis_.y - D.ankle_h - 0.985f * L;
        collide_walls(std::max(free_y, fl_prev_free_.y));
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s];
            V2 fx = air_foot_xz(s); f.sole.x = fx.x; f.sole.z = fx.z;
            if (fl_has_yaw_[s]) f.yaw += clampf(wrap(fl_yaw_[s] - f.yaw), -6.f * dt, 6.f * dt); else f.yaw = yaw_;
            float below = surface_below(f.sole.x, f.sole.z, std::max(free_y, fl_prev_free_.y)), gap = std::max(0.f, free_y - below);
            f.sole.y = free_y + std::min(0.10f * params.scale, 0.5f * gap); f.pitch = 0.f; f.toe = 0.f;
        }
        bool down = fl_v_.y < 0.f;
        float touch = -1e9f; float surfs[2];
        for (int s = 0; s < 2; ++s) surfs[s] = surface_below(feet_[s].sole.x, feet_[s].sole.z, std::max(free_y, fl_prev_free_.y));
        float hi = std::max(surfs[0], surfs[1]);
        if (down && free_y <= hi) touch = hi;
        fl_prev_free_.y = free_y;
        if (touch > -1e8f) land(surfs, touch);
    }

    // Where a foot is in the air: under its hip, pulled toward its landing target (a foot only swings so far from the hip).
    body_detail::V2 air_foot_xz(int s) const {
        using namespace body_detail;
        const float L = D.thigh + D.shin; Vec3 l = lft(yaw_);
        V2 hip{pelvis_.x + l.x * sg(s) * D.hip_w, pelvis_.z + l.z * sg(s) * D.hip_w}, want = fl_has_tgt_[s] ? fl_tgt_[s] : hip;
        float dx = want.x - hip.x, dz = want.z - hip.z, dl = std::hypot(dx, dz), cap = 0.45f * L;
        if (dl > cap) { dx *= cap / dl; dz *= cap / dl; }
        return {hip.x + dx, hip.z + dz};
    }

    // A surface whose top is well above the feet is a wall in flight: the body is pushed out of it sideways (and loses that speed).
    void collide_walls(float ref_y) {
        const float tol = 0.06f * params.scale, eps = 0.02f;
        for (int it = 0; it < 4; ++it) for (int s = 0; s < 2; ++s) for (const auto& sf : surfaces_) {
            if (sf.y <= ref_y + tol) continue;
            body_detail::V2 p = air_foot_xz(s);
            if (p.x < sf.x0 || p.x > sf.x1 || p.z < sf.z0 || p.z > sf.z1) continue;
            float d[4] = {p.x - sf.x0, sf.x1 - p.x, p.z - sf.z0, sf.z1 - p.z}; int k = 0; for (int i = 1; i < 4; ++i) if (d[i] < d[k]) k = i;
            if (k == 0) { pelvis_.x -= d[0] + eps; if (fl_v_.x > 0) fl_v_.x = 0; } else if (k == 1) { pelvis_.x += d[1] + eps; if (fl_v_.x < 0) fl_v_.x = 0; }
            else if (k == 2) { pelvis_.z -= d[2] + eps; if (fl_v_.z > 0) fl_v_.z = 0; } else { pelvis_.z += d[3] + eps; if (fl_v_.z < 0) fl_v_.z = 0; }
            event("hit the side of a surface in the air");
        }
    }

    void clamp_push_reach() {                  // while the feet are still planted the legs cannot be longer than they are
        for (int s = 0; s < 2; ++s) {
            Vec3 A = ankle_of(feet_[s]), H = hip_pos(s, pelvis_, hyaw()), d = H - A; float l = d.length(), hi = 0.995f * rmax_leg_ + params.ankle_push * params.scale;
            if (l > hi) pelvis_ -= d * ((l - hi) / l);
        }
    }

    void land(const float surfs[2], float highest) {
        const float L = D.thigh + D.shin;
        float vy = fl_v_.y;
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; f.sole.y = surfs[s]; f.surface = surfs[s]; f.state = FootState::Planted; f.pitch = 0.f; f.toe = 0.f; f.heel = 0.f;
            fl_has_tgt_[s] = false; fl_has_yaw_[s] = false;
        }
        if (std::fabs(surfs[0] - surfs[1]) > 0.02f) event("landed with one foot higher than the other: the lower leg may not reach its surface");
        float y_ext = highest + D.ankle_h + 0.985f * L;
        float dip = std::min(vy * vy / (2.f * 25.f), 0.22f * params.scale);       // the knees absorb the impact over ~25 m/s^2
        pelvis_.y = y_ext - dip; pvel_ = Vec3(fl_v_.x, 0.f, fl_v_.z);
        flight_ = Flight::None; unstable_t_ = 0.f; reach_warned_ = false; dyn_init_ = false;
        char b[160]; std::snprintf(b, sizeof b, "landed at (%.2f, %.2f), coming down at %.2f m/s, still moving %.2f m/s horizontally", pelvis_.x, pelvis_.z, -vy, std::hypot(fl_v_.x, fl_v_.z));
        event(b);
    }

    // ------------------------------------------------------------------ stepping
    bool validate_step(const StepCmd& c, Vec3& end, bool& planted, float& surf, std::string& why) {
        const Foot& f = feet_[c.side]; const Foot& sup = feet_[1 - c.side]; char b[200];
        if (sup.state != FootState::Planted) { why = side_name(c.side) + " foot step dropped: the other foot is not on the ground (use JUMP to leave the ground with both feet)"; return false; }
        float stride = std::hypot(c.target.x - sup.sole.x, c.target.z - sup.sole.z), maxs = params.max_stride * params.scale;
        if (stride > maxs) { std::snprintf(b, sizeof b, "%s foot step rejected: step too long (%.2f m from the support foot, max %.2f m)", side_name(c.side).c_str(), stride, maxs); why = b; return false; }
        if (stride < 0.07f * params.scale) { std::snprintf(b, sizeof b, "%s foot step rejected: target is %.0f cm from the other foot (feet would collide)", side_name(c.side).c_str(), stride * 100.f); why = b; return false; }
        float s = surface_at(c.target.x, c.target.z, sup.surface); end = c.target; planted = true; surf = s;
        if (end.y < s - 1e-3f) { event(side_name(c.side) + " foot target was below the surface: raised to it"); end.y = s; }
        if (end.y <= s + 0.02f) {
            end.y = s;
            float dh = s - sup.surface;
            if (dh > params.max_step_up * params.scale) { std::snprintf(b, sizeof b, "%s foot step rejected: step up of %.2f m is higher than the legs can manage (max %.2f m)", side_name(c.side).c_str(), dh, params.max_step_up * params.scale); why = b; return false; }
            if (-dh > params.max_step_down * params.scale) { std::snprintf(b, sizeof b, "%s foot step rejected: drop of %.2f m is too far (max %.2f m)", side_name(c.side).c_str(), -dh, params.max_step_down * params.scale); why = b; return false; }
        } else {
            planted = false;
            if (end.y - sup.surface > 0.6f * params.scale) { std::snprintf(b, sizeof b, "%s foot step rejected: lifting the foot %.2f m is higher than the leg can go (max %.2f m)", side_name(c.side).c_str(), end.y - sup.surface, 0.6f * params.scale); why = b; return false; }
        }
        (void)f; return true;
    }

    void begin_step(const StepCmd& c) {
        act_.on = true; act_.side = c.side; act_.cmd = c; act_.t = 0.f;
        if (params.balance == Balance::Assist && feet_[c.side].state == FootState::Planted) act_.phase = Active::Shift;
        else start_swing();
    }

    float swing_time(float h, float dy) const {
        float stride = std::sqrt(h * h + dy * dy);
        if (!params.dynamic_gait) return clampf(0.30f + 0.40f * stride, 0.30f, 1.1f) * params.step_time_scale;
        return clampf(params.swing_base + params.swing_per_m * stride, 0.28f, 1.1f) * params.step_time_scale;
    }

    // A step the pendulum cannot catch in the nominal time is taken more slowly (a person slows down when the next foot is far):
    // find the shortest swing time, up to 1.9x nominal, for which the dead-beat CoP stays inside the support foot.
    bool swing_ok_ = true;
    float feasible_swing_time(float T_nom) {
        const float w = dyn_w(); V2 xi{cm_.x + cmv_.x / w, cm_.z + cmv_.z / w}; if (!dyn_init_) xi = {com_.x, com_.z};
        for (float k = 0.8f; k <= 1.91f; k += 0.1f) {
            float T = T_nom * k; GaitPlan P = make_plan(w, T); float E = std::exp(w * T);
            V2 c{(P.xi_td0.x - xi.x * E) / (1.f - E), (P.xi_td0.z - xi.z * E) / (1.f - E)};
            if (body_detail::hull_margin(foot_polygon(1 - act_.side), c) >= 0.004f) { swing_ok_ = true; return T; }
        }
        swing_ok_ = false; return T_nom * 1.9f;
    }
    // Assist: if the pendulum cannot catch the commanded landing even at the slowest swing, land nearer to the support foot (and say so).
    void adjust_for_capture() {
        Foot& f = feet_[act_.side]; const Foot& sup = feet_[1 - act_.side];
        float h0 = std::hypot(act_.cmd.target.x - f.sole.x, act_.cmd.target.z - f.sole.z);
        feasible_swing_time(swing_time(h0, 0.f)); if (swing_ok_) return;
        Vec3 orig = act_.cmd.target; float dx = orig.x - sup.sole.x, dz = orig.z - sup.sole.z, d0 = std::hypot(dx, dz);
        for (float k : {0.8f, 0.65f, 0.5f, 0.35f, 0.2f, 0.1f, 0.05f}) {
            act_.cmd.target.x = sup.sole.x + dx * k; act_.cmd.target.z = sup.sole.z + dz * k;
            float h = std::hypot(act_.cmd.target.x - f.sole.x, act_.cmd.target.z - f.sole.z); feasible_swing_time(swing_time(h, 0.f));
            if (swing_ok_) { char b[200]; std::snprintf(b, sizeof b, "%s foot target pulled in from %.2f m to %.2f m from the support foot: the body could not catch itself at the requested distance (assist)", side_name(act_.side).c_str(), d0, d0 * k); event(b); return; }
        }
        act_.cmd.target = orig;                                              // nothing helps: take the step as asked, the balance model decides
    }

    void start_swing() {
        Foot& f = feet_[act_.side]; Vec3 end; bool planted; float surf; std::string why;
        if (dyn_on() && f.state == FootState::Planted && act_.cmd.target.y <= surface_at(act_.cmd.target.x, act_.cmd.target.z, feet_[1 - act_.side].surface) + 0.02f) adjust_for_capture();
        if (!validate_step(act_.cmd, end, planted, surf, why)) { event(why); act_.on = false; return; }
        float h = std::hypot(end.x - f.sole.x, end.z - f.sole.z), dy = end.y - f.sole.y, stride = std::sqrt(h * h + dy * dy);
        float T = swing_time(h, dy);
        if (dyn_on() && f.state == FootState::Planted) { float Tn = T; T = feasible_swing_time(Tn); (void)Tn; }
        float lift = clampf(0.06f + 0.12f * h, 0.05f, 0.16f) * params.scale + std::max(0.f, dy);
        Vec3 p0 = f.sole, v0, a0;
        if (f.state == FootState::Swing) f.traj.eval(f.traj.t, p0, v0, a0);
        f.traj.plan(p0, v0, a0, end, T, Vec3(0, 1, 0), lift);
        f.yaw0 = f.yaw; f.yaw1 = act_.cmd.has_yaw ? f.yaw + body_detail::wrap(act_.cmd.yaw - f.yaw) : f.yaw;
        f.end_sole = end; f.end_planted = planted; f.end_surface = surf; f.state = FootState::Swing; act_.phase = Active::Swing;
    }

    void step_machine(float dt) {
        if (!act_.on && !queue_.empty()) {
            StepCmd c = queue_.front(); queue_.pop_front();
            Vec3 end; bool pl; float sf; std::string why;
            if (!validate_step(c, end, pl, sf, why)) event(why); else begin_step(c);
        }
        if (act_.on && act_.phase == Active::Shift) {
            act_.t += dt;
            if (dyn_on()) {                                           // ready when the pendulum can reach the plan with the CoP inside the support foot
                const float w = dyn_w(); GaitPlan P = make_plan(w, 0.f); V2 xi{cm_.x + cmv_.x / w, cm_.z + cmv_.z / w};
                if (!dyn_init_) { xi = {com_.x, com_.z}; }
                float E = std::exp(w * std::max(P.T0, 0.12f)); V2 c{(P.xi_td0.x - xi.x * E) / (1.f - E), (P.xi_td0.z - xi.z * E) / (1.f - E)};
                float m = body_detail::hull_margin(foot_polygon(1 - act_.side), c);
                float unload = body_detail::hull_margin(foot_polygon(1 - act_.side), V2{cop_f_.x, cop_f_.z});   // the swing foot lifts once the CoP is under the support foot (it is unloaded)
                                bool close = std::hypot(xi.x - shift_ref(P).x, xi.z - shift_ref(P).z) < 0.015f; bool still = std::hypot(cmv_.x, cmv_.z) < 0.15f, inside = body_detail::hull_margin(foot_polygon(1 - act_.side), xi) >= 0.012f;   // quasi-static: the DCM is inside the support foot
                if (((m >= 0.006f || close || (still && inside)) && unload >= -0.004f && act_.t >= params.step_ds_time) || act_.t > 1.0f) {
                    if (m < 0.f) event("weight shift timed out before the CoM could be caught by the support foot; stepping anyway");
                    start_swing();
                }
                return;
            }
            const Foot& sup = feet_[1 - act_.side];
            float m = single_margin(1 - act_.side);
            float pv = std::hypot(pvel_.x, pvel_.z);
            if ((m >= 0.5f * params.safe_margin && pv < params.shift_settle_speed) || act_.t > 1.5f) {
                if (m < 0.f) event("weight shift timed out before the CoM was over the support foot; stepping anyway");
                (void)sup; start_swing();
            }
        }
    }

    std::vector<body_detail::V2> foot_polygon(int s) const {
        std::vector<body_detail::V2> p;
        for (const Vec3& c : foot_contact_points(s)) p.push_back({c.x, c.z});
        return body_detail::hull(p);
    }
    float single_margin(int s) const { return body_detail::hull_margin(foot_polygon(s), xcom_); }

    void update_feet(float dt) {
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s];
            if (f.state != FootState::Swing) continue;
            f.traj.t += dt;
            float tau = clampf(f.traj.t / f.traj.T, 0.f, 1.f);
            if (f.traj.t >= f.traj.T) {
                f.sole = f.end_sole; f.yaw = f.yaw1; f.pitch = 0.f; f.toe = 0.f; f.heel = 0.f;
                if (f.end_planted) {
                    f.state = FootState::Planted; f.surface = f.end_surface;
                    event(side_name(s) + " foot planted at " + xz(f.sole));
                } else { f.state = FootState::Air; event(side_name(s) + " foot held in the air"); }
                if (act_.on && act_.side == s) act_.on = false;
                continue;
            }
            f.sole = f.traj.position(f.traj.t);
            f.sole.y = std::max(f.sole.y, surface_at(f.sole.x, f.sole.z, feet_[1 - s].surface));          // never through a surface
            f.yaw = f.yaw0 + (f.yaw1 - f.yaw0) * minimum_jerk_s(tau);
            f.heel *= std::exp(-dt * 14.f); f.pitch = body_detail::swing_pitch(tau); f.toe = std::max(0.f, -f.pitch) * 0.8f;
        }
    }

    // ------------------------------------------------------------------ pelvis: targets, smoothing, constraints, balance
    Vec3 hip_pos(int s, Vec3 pelvis, float yaw) const { return pelvis + body_detail::lft(yaw) * (sg(s) * D.hip_w); }

    void clamp_reach(Vec3& p, float yaw) {
        for (int it = 0; it < 4; ++it)
            for (int s = 0; s < 2; ++s) {
                if (feet_[s].state != FootState::Planted) continue;
                Vec3 A = ankle_of(feet_[s]), H = hip_pos(s, p, yaw), d = H - A; float l = d.length();
                float hi = 0.995f * rmax_leg_, lo = rmin_leg_;
                if (l > hi) { p -= d * ((l - hi) / l); if (!reach_warned_) { event("pelvis limited by leg length (the planted feet do not slide)"); reach_warned_ = true; } }
                else if (l < lo && l > 1e-6f) p += d * ((lo - l) / l);
            }
    }
    bool reach_warned_ = false;

    void update_pelvis(float dt) {
        using namespace body_detail;
        int planted[2], np = 0; for (int s = 0; s < 2; ++s) if (feet_[s].state == FootState::Planted) planted[np++] = s;
        pelvis_prev_ = pelvis_;
        // yaw
        float ref = yaw_, twist_ref = yaw_;
        if (np) { float sx = 0, cx = 0; for (int i = 0; i < np; ++i) { sx += std::sin(feet_[planted[i]].yaw); cx += std::cos(feet_[planted[i]].yaw); } ref = twist_ref = std::atan2(sx, cx); }
        float want_yaw = face_user_ ? std::atan2(face_pt_.x - pelvis_.x, face_pt_.z - pelvis_.z) : ref;
        yaw_ += clampf(wrap(want_yaw - yaw_), -3.f * dt, 3.f * dt);
        if (np) {
            float lim = 60.f * PI / 180.f, diff = wrap(yaw_ - twist_ref);
            if (std::fabs(diff) > lim) { yaw_ = twist_ref + (diff > 0 ? lim : -lim); if (face_user_ && !turn_warned_) { event("FACE limited: planted feet allow 60 degrees of twist; step the feet around (FOOT with a yaw) to turn further"); turn_warned_ = true; } }
        }
        if (np > 0 && dyn_on()) { update_pelvis_dynamic(dt, planted, np); return; }
        dyn_init_ = false;                                            // legacy path: the walking-only extras fade out
        { float k = std::exp(-dt * 8.f); twist_ *= k; chest_twist_ *= k; lean_.x *= k; lean_.z *= k; head_dev_.x *= k; head_dev_.z *= k; for (int s = 0; s < 2; ++s) arm_swing_[s] *= k; }
        // target
        const bool override_on = act_.on && params.balance == Balance::Assist && feet_[1 - act_.side].state == FootState::Planted;
        V2 txz{pelvis_.x, pelvis_.z}; float surf = 0.f;
        if (np) {
            V2 mid{0, 0}; surf = 0;
            for (int i = 0; i < np; ++i) { mid.x += feet_[planted[i]].sole.x / np; mid.z += feet_[planted[i]].sole.z / np; surf += feet_[planted[i]].surface / np; }
            if (params.balance == Balance::Assist) txz = mid;          // assist: weight stays over the feet; none: the pelvis only moves when told to
        }
        if (override_on) {                                           // put the CoM (not the pelvis) over the middle of the support foot
            int so = 1 - act_.side; const Foot& sup = feet_[so]; V2 c{0, 0}; auto pts = foot_contact_points(so);
            for (const Vec3& q : pts) { c.x += q.x / pts.size(); c.z += q.z / pts.size(); }
            txz = {c.x - com_off_.x, c.z - com_off_.z}; surf = sup.surface;
        }
        else if (pelvis_user_) txz = {pelvis_tgt_.x, pelvis_tgt_.z};
        float ty = pelvis_user_ ? pelvis_tgt_.y : surf + D.ankle_h + 0.985f * (D.thigh + D.shin);
        Vec3 cur = pelvis_, vel = pvel_; smooth_damp(cur, Vec3(txz.x, ty, txz.z), vel, 10.f, dt);
        Vec3 dlt = cur - pelvis_; float hmax = 1.4f * params.scale * dt, vmax = 1.0f * params.scale * dt, hl = std::hypot(dlt.x, dlt.z);
        if (hl > hmax) { dlt.x *= hmax / hl; dlt.z *= hmax / hl; }
        dlt.y = clampf(dlt.y, -vmax, vmax);
        Vec3 np_ = pelvis_ + dlt;
        clamp_reach(np_, hyaw());
        if (params.balance == Balance::Assist && !support_.empty()) {                // keep the extrapolated CoM inside the support polygon
            std::vector<V2> poly = np ? support_poly() : std::vector<V2>();
            if (!poly.empty()) {
                float h = std::max(0.3f, com_.y - surf), w0 = std::sqrt(9.81f / h);
                V2 v{(np_.x - pelvis_prev_.x) / dt, (np_.z - pelvis_prev_.z) / dt};
                V2 x{np_.x + com_off_.x + v.x / w0, np_.z + com_off_.z + v.z / w0};
                if (hull_margin(poly, x) < params.safe_margin) {
                    // x = pos + off + (pos - prev)/(dt*w0): moving the pelvis by d moves x by k*d, with k = 1 + 1/(dt*w0).
                    // Dividing by k is exactly the capture-point speed limit (it also avoids a frame-to-frame oscillation).
                    V2 pr = hull_project(poly, x, params.safe_margin); float k = 1.f + 1.f / (dt * w0);
                    np_.x += (pr.x - x.x) / k; np_.z += (pr.z - x.z) / k; clamp_reach(np_, hyaw());
                }
            }
        }
        pvel_ = Vec3((np_.x - pelvis_prev_.x) / dt, (np_.y - pelvis_prev_.y) / dt, (np_.z - pelvis_prev_.z) / dt);
        pelvis_ = np_;
    }

    std::vector<body_detail::V2> support_poly() const {
        std::vector<body_detail::V2> pts;
        for (int s = 0; s < 2; ++s) if (feet_[s].state == FootState::Planted) for (const Vec3& c : foot_contact_points(s)) pts.push_back({c.x, c.z});
        return body_detail::hull(pts);
    }

    // ------------------------------------------------------------------ pose: spine, head, clavicles, arms
    void pose_upper(float dt, bool init) {
        using namespace body_detail;
        Vec3 up(0, 1, 0);
        Vec3 root = pelvis_ + up * D.pelvis_bone;
        float total = D.spine[0] + D.spine[1] + D.spine[2];
        spine_.root_position = root; spine_.root_orientation = Quat::rotation_between(Vec3(1, 0, 0), up);
        Vec3 t = (torso_user_ ? torso_tgt_ - root : up * total + Vec3(lean_.x + head_dev_.x, 0.f, lean_.z + head_dev_.z) );
        float l = t.length(); if (l < 1e-4f) t = up * total, l = total;
        Vec3 dir = t * (1.f / l);
        if (dir.y < std::cos(55.f * PI / 180.f)) { Vec3 hz(dir.x, 0, dir.z); hz = hz.normalized(); dir = up * std::cos(55.f * PI / 180.f) + hz * std::sin(55.f * PI / 180.f); }
        solve_fabrik(spine_, root + dir * std::min(l, 0.995f * total), 1e-3f, 8);
        Vec3 sp[4]; spine_.world_positions(sp);
        Vec3 cup = (sp[3] - sp[2]).normalized(); if (cup.length_sq() < 0.5f) cup = up;
        // head look (relative to the chest frame)
        Vec3 headc = sp[3] + cup * (D.neck + D.head * 0.5f);
        float lyaw = 0, lpitch = 0, chest_yaw = yaw_;
        if (look_user_) {
            Vec3 d = (look_tgt_ - headc).normalized();
            float hz = std::hypot(d.x, d.z), hw = clampf((hz - 0.05f) / 0.35f, 0.f, 1.f);       // gaze almost straight up/down: its heading is meaningless, keep the current one
            float gy = std::atan2(d.x, d.z); chest_yaw = yaw_ + hw * clampf(0.35f * wrap(gy - yaw_), -0.44f, 0.44f);
            Quat q0 = frameQ(cup, fwd(chest_yaw)); Vec3 Z = q0.rotate(Vec3(0, 0, 1)), X = q0.rotate(Vec3(1, 0, 0));
            float dx = dot(d, X), dz = dot(d, Z), dyy = dot(d, cup);
            lyaw = std::atan2(dx, dz) * hw; lpitch = std::atan2(dyy, std::hypot(dx, dz));
            float ym = 75.f * PI / 180.f, pu = 50.f * PI / 180.f, pd = -55.f * PI / 180.f;
            if ((std::fabs(lyaw) > ym || lpitch > pu || lpitch < pd) && !look_warned_) { event("LOOK limited by neck range (75 deg turn, +50/-55 deg tilt); the body must turn to look further"); look_warned_ = true; }
            lyaw = clampf(lyaw, -ym, ym); lpitch = clampf(lpitch, pd, pu);
        }
        {   // head and gaze have inertia: a LOOK target is chased by a critically damped spring instead of snapping like a camera
            float co = chest_yaw - yaw_;
            if (init || dt <= 0.f || params.head_turn_omega <= 0.f) { hl_chest_ = co; hl_yaw_ = lyaw; hl_pitch_ = lpitch; hl_vchest_ = hl_vyaw_ = hl_vpitch_ = 0.f; }
            else { damp1(hl_chest_, hl_vchest_, co, 0.7f * params.head_turn_omega, dt); damp1(hl_yaw_, hl_vyaw_, lyaw, params.head_turn_omega, dt); damp1(hl_pitch_, hl_vpitch_, lpitch, params.head_turn_omega, dt); }
            chest_yaw = yaw_ + hl_chest_ + chest_twist_; lyaw = hl_yaw_; lpitch = hl_pitch_;
        }
        Quat qc = frameQ(cup, fwd(chest_yaw)); chest_q_ = qc;
        auto qyaw = [](float a) { return Quat::from_axis_angle(Vec3(0, 1, 0), a); };
        auto qpit = [](float a) { return Quat::from_axis_angle(Vec3(1, 0, 0), -a); };
        Quat qn = qc * qyaw(lyaw * 0.4f) * qpit(lpitch * 0.4f), qh = qc * qyaw(lyaw) * qpit(lpitch);
        neck_q_ = qn; head_q_ = qh;
        neck_tip_ = sp[3] + qn.rotate(Vec3(0, 1, 0)) * D.neck; head_tip_ = neck_tip_ + qh.rotate(Vec3(0, 1, 0)) * D.head;
        head_center_ = (neck_tip_ + head_tip_) * 0.5f;
        for (int i = 0; i < 4; ++i) sp_[i] = sp[i];
        // clavicles + arms
        Vec3 X = qc.rotate(Vec3(1, 0, 0)), Y = qc.rotate(Vec3(0, 1, 0)), Z = qc.rotate(Vec3(0, 0, 1));
        for (int s = 0; s < 2; ++s) {
            float g = sg(s);
            Vec3 attach = sp[3] + X * (g * D.clav_attach);
            Vec3 rest = (X * (g * 0.97f) + Y * -0.12f + Z * -0.08f).normalized(), cd = rest;
            if (hands_[s].target_mode && hands_[s].has_cmd) {
                Vec3 toh = (hands_[s].cmd - attach).normalized();
                if (toh.length_sq() > 0.5f) cd = clamp_to_cone((rest * 0.7f + toh * 0.3f).normalized(), rest, std::cos(30.f * PI / 180.f), std::sin(30.f * PI / 180.f));
            }
            clav_attach_[s] = attach; shoulder_[s] = attach + cd * D.clav;
            arms_.set_anchor(arm_id_[s], shoulder_[s]);
            arms_.set_pole(arm_id_[s], (Z * -0.6f + X * (g * 0.25f) + Vec3(0, -1, 0)));
            Vec3 restpt = shoulder_[s] + X * (g * 0.07f * params.scale) + Z * (0.05f * params.scale) + Vec3(0, -(D.upper + D.fore) * 0.86f, 0);
            restpt += fwd(yaw_) * arm_swing_[s];
            const float thr = (std::fabs(arm_swing_[s]) > 0.004f || std::fabs(arm_swing_v_[s]) > 0.02f) ? 0.003f : 0.03f;
            if (!hands_[s].target_mode && (!hands_[s].has_last || (restpt - hands_[s].last).length() > thr * params.scale)) { arms_.reach_to(arm_id_[s], restpt); hands_[s].last = restpt; hands_[s].has_last = true; }
            if (init) arms_.teleport_to(arm_id_[s], restpt);
        }
        arms_.update(init ? 1e-4f : dt);
        for (int s = 0; s < 2; ++s) { Vec3 p[3]; arms_.chain(arm_id_[s]).world_positions(p); shoulder_[s] = p[0]; elbow_[s] = p[1]; wrist_[s] = p[2]; }
    }
    Quat neck_q_, head_q_; Vec3 neck_tip_, head_tip_, sp_[4], clav_attach_[2];

    // ------------------------------------------------------------------ pose: legs
    void pose_legs() {
        using namespace body_detail;
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; f.leg.root_position = hip_pos(s, pelvis_, hyaw());
            Vec3 pole = (fwd(f.yaw) * 0.6f + fwd(yaw_) * 0.4f).normalized();
            solve_two_bone_reach(f.leg, ankle_of(f), pole, rmin_leg_, 0.999f * rmax_leg_);
            Vec3 p[3]; f.leg.world_positions(p); f.hip = p[0]; f.knee = p[1]; f.ankle = p[2];
        }
    }

    // ------------------------------------------------------------------ bones (positions + orientations) and fingers
    void put(int i, Vec3 h, Vec3 t, Vec3 zhint) { bones_[i].head = h; bones_[i].tail = t; bones_[i].q = body_detail::frameQ(t - h, zhint); }

    void compute_bones() {
        using namespace body_detail;
        Vec3 up(0, 1, 0), pf = fwd(hyaw());
        put(0, pelvis_, pelvis_ + up * D.pelvis_bone, pf);
        Vec3 zc = chest_q_.rotate(Vec3(0, 0, 1)), Xc = chest_q_.rotate(Vec3(1, 0, 0));
        for (int i = 0; i < 3; ++i) { float k = (i + 1) / 3.f; put(1 + i, sp_[i], sp_[i + 1], pf * (1.f - k) + zc * k); }
        put(4, sp_[3], neck_tip_, neck_q_.rotate(Vec3(0, 0, 1)));
        put(5, neck_tip_, head_tip_, head_q_.rotate(Vec3(0, 0, 1)));
        for (int s = 0; s < 2; ++s) {
            float g = sg(s);
            put(I.clav[s], clav_attach_[s], shoulder_[s], zc);
            Vec3 pole = (zc * -0.6f + Xc * (g * 0.25f) + Vec3(0, -1, 0));
            put(I.upper[s], shoulder_[s], elbow_[s], -pole); put(I.fore[s], elbow_[s], wrist_[s], -pole);
            Vec3 f = (wrist_[s] - elbow_[s]).normalized(), n = -Xc * g;                 // palm faces the body at rest
            Vec3 knuckle = wrist_[s] + f * D.hand;
            put(I.hand[s], wrist_[s], knuckle, n);
            if (params.finger_lod) fingers(s, f, n, knuckle);
        }
        for (int s = 0; s < 2; ++s) {
            const Foot& ft = feet_[s]; Vec3 fw = fwd(ft.yaw), lf = lft(ft.yaw);
            put(I.thigh[s], ft.hip, ft.knee, fw); put(I.shin[s], ft.knee, ft.ankle, fw);
            Quat qp = Quat::from_axis_angle(lf, -ft.pitch);                              // toe-up positive
            Vec3 ball = ft.ankle + qp.rotate(fw * D.ball + Vec3(0, -D.ankle_h, 0));
            Vec3 tdir = Quat::from_axis_angle(lf, ft.toe).rotate(qp.rotate(fw));
            put(I.foot[s], ft.ankle, ball, up); put(I.toe[s], ball, ball + tdir * D.toe_len, up);
        }
    }

    void fingers(int s, Vec3 f, Vec3 n, Vec3 knuckle) {
        float g = sg(s), curl = hands_[s].curl;
        n = (n - f * dot(n, f)).normalized(); if (n.length_sq() < 0.5f) n = Vec3(0, 0, 1);
        Vec3 lt = cross(n, f) * g;                                                       // across the palm toward the thumb
        const float off[5] = {0.045f, 0.030f, 0.010f, -0.010f, -0.028f}, spread[5] = {0.70f, 0.10f, 0.0f, -0.08f, -0.18f};
        int idx = I.finger0[s];
        for (int k = 0; k < 5; ++k) {
            Vec3 base = knuckle + lt * (off[k] * params.scale) + (k == 0 ? n * (-0.01f * params.scale) : Vec3());
            Vec3 dir = f * std::cos(spread[k]) + lt * std::sin(spread[k]);
            float L = D.fingers[k];
            if (params.finger_lod == 1) {
                float th = curl * (k == 0 ? 1.1f : 1.4f);
                Vec3 d = dir * std::cos(th) + n * std::sin(th);
                put(idx++, base, base + d * L, n);
            } else {
                int segs = k == 0 ? 2 : 3; const float fr3[3] = {0.45f, 0.30f, 0.25f}, fr2[2] = {0.55f, 0.45f}, fl[3] = {1.2f, 1.5f, 1.0f};
                float th = 0; Vec3 p = base;
                for (int q = 0; q < segs; ++q) {
                    th += curl * fl[q] * (k == 0 ? 0.8f : 1.0f);
                    Vec3 d = dir * std::cos(th) + n * std::sin(th), e = p + d * (L * (segs == 3 ? fr3[q] : fr2[q]));
                    put(idx++, p, e, n); p = e;
                }
            }
        }
    }

    // ------------------------------------------------------------------ balance
    void compute_balance(float dt) {
        using namespace body_detail;
        Vec3 c; for (size_t i = 0; i < bones_.size(); ++i) c += (bones_[i].head + bones_[i].tail) * (0.5f * mass_[i]);
        com_ = c;
        if (!com_init_) { com_prev_ = com_; vcom_ = {}; com_init_ = true; }
        V2 raw{(com_.x - com_prev_.x) / dt, (com_.z - com_prev_.z) / dt};
        vcom_.x = vcom_.x * 0.6f + raw.x * 0.4f; vcom_.z = vcom_.z * 0.6f + raw.z * 0.4f;
        com_prev_ = com_;
        support_ = support_poly();
        if (params.dynamic_gait && params.balance == Balance::Assist && act_.on && flight_ == Flight::None) {
            const Foot& f = feet_[act_.side]; bool swing = act_.phase == Active::Swing;      // dynamic walking: the DCM may leave the stance foot as long as the landing foot can catch it
            Vec3 land = swing ? f.end_sole : act_.cmd.target; float ly = swing ? f.yaw1 : (act_.cmd.has_yaw ? act_.cmd.yaw : f.yaw);
            if (!swing || f.end_planted) {
                std::vector<V2> pts = support_; Vec3 fw = fwd(ly), lf = lft(ly);
                for (float lat : {D.heel_hw, -D.heel_hw}) { Vec3 q = land - fw * D.heel + lf * lat; pts.push_back({q.x, q.z}); }
                for (float lat : {D.ball_hw, -D.ball_hw}) { Vec3 q = land + fw * D.ball + lf * lat; pts.push_back({q.x, q.z}); }
                support_ = hull(pts);
            }
        }
        if (tracking_) {        // a tracked walk is dynamic: the DCM may leave the stance foot as long as the other foot is (about to be) down; count its footprint where it is
            std::vector<V2> pts = support_;
            for (int s = 0; s < 2; ++s) {
                if (track_land_has_[s]) {      // the clip says where this foot lands next: count that footprint, exactly like the walking engine does (also while it is still being loaded)
                    Vec3 fw = fwd(track_land_yaw_[s]), lf = lft(track_land_yaw_[s]), L = track_land_[s];
                    for (float lat : {D.heel_hw, -D.heel_hw}) { Vec3 q = L - fw * D.heel + lf * lat; pts.push_back({q.x, q.z}); }
                    for (float lat : {D.ball_hw, -D.ball_hw}) { Vec3 q = L + fw * D.ball + lf * lat; pts.push_back({q.x, q.z}); }
                    continue;
                }
                if (feet_[s].state == FootState::Planted) continue;
                // no look-ahead (single frames): where the swing foot is, and where it will be in ~0.25 s at its current velocity
                for (const Vec3& q : foot_contact_points(s)) pts.push_back({q.x, q.z});
                Vec3 d = track_fv_[s] * 0.25f; d.y = 0.f; if (d.length() > 0.6f) d = d.normalized() * 0.6f;
                for (const Vec3& q : foot_contact_points(s)) pts.push_back({q.x + d.x, q.z + d.z});
            }
            if (pts.size() > support_.size()) support_ = hull(pts);
        }
        int np = 0; float surf = 0; for (int s = 0; s < 2; ++s) if (feet_[s].state == FootState::Planted) { ++np; surf += feet_[s].surface; }
        surf = np ? surf / np : 0.f;
        float h = std::max(0.3f, com_.y - surf), w0 = std::sqrt(9.81f / h);
        xcom_ = {com_.x + vcom_.x / w0, com_.z + vcom_.z / w0};
        margin_ = np ? hull_margin(support_, xcom_) : (flight_ != Flight::None ? 0.f : -1.f);
        com_off_ = {com_.x - pelvis_.x, com_.z - pelvis_.z};
        bool unstable = flight_ == Flight::None && margin_ < params.fall_margin;
        unstable_t_ = unstable ? unstable_t_ + dt : 0.f;
        if (unstable_t_ > params.fall_time && !fallen_) {
            fallen_ = true; queue_.clear(); act_.on = false;
            char b[200];
            if (np == 0) std::snprintf(b, sizeof b, "FELL: no foot on the ground");
            else std::snprintf(b, sizeof b, "FELL: the centre of mass was %.0f cm outside the support polygon (balance %s)", -margin_ * 100.f, params.balance == Balance::Assist ? "assist" : "none");
            event(b);
        }
    }
};

} // namespace ik
