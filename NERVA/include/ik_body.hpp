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
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>
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
        float shift_settle_speed = 0.04f;   // m/s: how still the pelvis must be over the support foot before the other foot lifts
        float hand_idle_m = 0.0015f;        // tiny hand drift when idle (0 = none, for deterministic tests)
    };
    struct Result { bool ok; std::string msg; };
    struct Surface { float x0, x1, z0, z1, y; };
    enum class FootState { Planted, Swing, Air };

    Params params;

    Body() : Body(Params{}) {}                       // (two constructors: a nested struct's default initializers can't appear in a default argument)
    explicit Body(const Params& p) : params(p) { build(); }
    Body(const Body&) = delete;                       // bones_ hold pointers into names_
    Body& operator=(const Body&) = delete;

    void set_hand_idle(float m) { params.hand_idle_m = m; arms_.params.idle_amp_m = m; }

    // ------------------------------------------------------------------ world
    void add_surface(float x0, float x1, float z0, float z1, float y) { surfaces_.push_back({std::min(x0, x1), std::max(x0, x1), std::min(z0, z1), std::max(z0, z1), y}); }
    void clear_surfaces() { surfaces_.clear(); }
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
        if (v == "GRIP") {
            int s = side(1);
            if (s < 0 || t.size() != 3 || !num(2, a)) return fail("usage: GRIP <left|right> 0..1");
            hands_[s].curl = clampf(a, 0.f, 1.f); return ok(side_name(s) + " grip " + fmt(hands_[s].curl));
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
        if (!(dt > 0.f)) return;
        dt = std::min(dt, 0.05f);
        time_ += dt;
        if (!fallen_) { step_machine(dt); update_feet(dt); update_pelvis(dt); }
        pose_upper(dt, false);
        pose_legs();
        compute_bones();
        compute_balance(dt);
    }

    // ------------------------------------------------------------------ queries
    const std::vector<BoneOut>& bones() const { return bones_; }
    int bone_index(const std::string& n) const { for (size_t i = 0; i < names_.size(); ++i) if (names_[i] == n) return (int)i; return -1; }
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
        std::snprintf(b, sizeof b, "{\"t\":%.2f,\"state\":\"%s\",\"balance\":\"%s\",\"margin_cm\":%.1f,", time_, fallen_ ? "fallen" : "standing", params.balance == Balance::Assist ? "assist" : "none", std::max(margin_, -999.f) * 100.f);
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
        Vec3 sole; float yaw = 0.f, pitch = 0.f, toe = 0.f, surface = 0.f;
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
    bool fallen_ = false; float unstable_t_ = 0.f, time_ = 0.f;
    Vec3 com_; body_detail::V2 com_off_, vcom_, xcom_; float margin_ = 0.f; bool com_init_ = false; Vec3 com_prev_;
    std::vector<body_detail::V2> support_;
    std::vector<BoneOut> bones_; std::vector<std::string> names_; std::vector<float> mass_; std::vector<std::string> events_;
    struct Idx { int clav[2], upper[2], fore[2], hand[2], finger0[2], thigh[2], shin[2], foot[2], toe[2], nfingers = 0; } I{};
    Quat chest_q_;

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

    // ------------------------------------------------------------------ construction
    void build() {
        params.finger_lod = std::max(0, std::min(2, params.finger_lod));
        params.scale = std::max(0.2f, std::min(3.f, params.scale));
        D = Dim(params.scale);
        two_bone_reach_limits(D.thigh, D.shin, 30.f * PI / 180.f, 175.f * PI / 180.f, rmin_leg_, rmax_leg_);
        // bone table
        names_.reserve(64); mass_.reserve(64); std::vector<int> parents; std::vector<float> masses;
        auto add = [&](const std::string& n, int par, float m) { names_.push_back(n); parents.push_back(par); masses.push_back(m); return (int)names_.size() - 1; };
        int pel = add("pelvis", -1, .145f), sp1 = add("spine1", pel, .125f), sp2 = add("spine2", sp1, .125f), sp3 = add("spine3", sp2, .102f);
        int nk = add("neck", sp3, .012f); add("head", nk, .069f);
        const char* S[2] = {"L", "R"};
        for (int s = 0; s < 2; ++s) {
            std::string u = std::string("_") + S[s];
            I.clav[s] = add("clavicle" + u, sp3, .005f); I.upper[s] = add("upper_arm" + u, I.clav[s], .028f);
            I.fore[s] = add("forearm" + u, I.upper[s], .016f); I.hand[s] = add("hand" + u, I.fore[s], .006f);
            I.finger0[s] = (int)names_.size();
            if (params.finger_lod == 1) { for (const char* f : {"thumb", "index", "middle", "ring", "pinky"}) add(std::string(f) + u, I.hand[s], 0.f); }
            else if (params.finger_lod == 2) {
                for (int k = 1; k <= 2; ++k) add("thumb_" + std::to_string(k) + u, k == 1 ? I.hand[s] : (int)names_.size() - 1, 0.f);
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
        bones_.resize(names_.size());
        for (size_t i = 0; i < names_.size(); ++i) { bones_[i].name = names_[i].c_str(); bones_[i].parent = parents[i]; }

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
        if (queue_.size() >= 8) return fail("FOOT rejected: 8 steps are already queued");
        if (std::hypot(target.x - pelvis_.x, target.z - pelvis_.z) > 3.f) return fail("FOOT rejected: target is more than 3 m from the body; step toward it in stages");
        queue_.push_back({s, target, has_yaw, yaw});
        char b[120]; std::snprintf(b, sizeof b, "%s foot step queued (%d ahead)", side_name(s).c_str(), (int)queue_.size() + (act_.on ? 1 : 0) - 1); return ok(b);
    }

    Result stand() {
        fallen_ = false; unstable_t_ = 0.f; queue_.clear(); act_.on = false; pelvis_user_ = false;
        for (int s = 0; s < 2; ++s) {
            Foot& f = feet_[s]; Vec3 l = body_detail::lft(yaw_);
            f.sole = Vec3(pelvis_.x, 0, pelvis_.z) + l * (sg(s) * D.hip_w); f.yaw = yaw_; f.pitch = 0; f.toe = 0;
            f.surface = surface_at(f.sole.x, f.sole.z); f.sole.y = f.surface; f.state = FootState::Planted;
        }
        pelvis_.y = std::max(feet_[0].surface, feet_[1].surface) + D.ankle_h + 0.985f * (D.thigh + D.shin); pvel_ = Vec3();
        com_init_ = false;
        event("stood up (no fall or get-up animation: the pose was reset)");
        return ok("standing");
    }

    // ------------------------------------------------------------------ stepping
    bool validate_step(const StepCmd& c, Vec3& end, bool& planted, float& surf, std::string& why) {
        const Foot& f = feet_[c.side]; const Foot& sup = feet_[1 - c.side]; char b[200];
        if (sup.state != FootState::Planted) { why = side_name(c.side) + " foot step dropped: the other foot is not on the ground (jumping is not implemented)"; return false; }
        float stride = std::hypot(c.target.x - sup.sole.x, c.target.z - sup.sole.z), maxs = params.max_stride * params.scale;
        if (stride > maxs) { std::snprintf(b, sizeof b, "%s foot step rejected: step too long (%.2f m from the support foot, max %.2f m)", side_name(c.side).c_str(), stride, maxs); why = b; return false; }
        if (stride < 0.07f * params.scale) { std::snprintf(b, sizeof b, "%s foot step rejected: target is %.0f cm from the other foot (feet would collide)", side_name(c.side).c_str(), stride * 100.f); why = b; return false; }
        float s = surface_at(c.target.x, c.target.z); end = c.target; planted = true; surf = s;
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

    void start_swing() {
        Foot& f = feet_[act_.side]; Vec3 end; bool planted; float surf; std::string why;
        if (!validate_step(act_.cmd, end, planted, surf, why)) { event(why); act_.on = false; return; }
        float h = std::hypot(end.x - f.sole.x, end.z - f.sole.z), dy = end.y - f.sole.y, stride = std::sqrt(h * h + dy * dy);
        float T = clampf(0.30f + 0.40f * stride, 0.30f, 1.1f) * params.step_time_scale;
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
                f.sole = f.end_sole; f.yaw = f.yaw1; f.pitch = 0.f; f.toe = 0.f;
                if (f.end_planted) {
                    f.state = FootState::Planted; f.surface = f.end_surface;
                    event(side_name(s) + " foot planted at " + xz(f.sole));
                } else { f.state = FootState::Air; event(side_name(s) + " foot held in the air"); }
                if (act_.on && act_.side == s) act_.on = false;
                continue;
            }
            f.sole = f.traj.position(f.traj.t);
            f.sole.y = std::max(f.sole.y, surface_at(f.sole.x, f.sole.z));          // never through a surface
            f.yaw = f.yaw0 + (f.yaw1 - f.yaw0) * minimum_jerk_s(tau);
            f.pitch = body_detail::swing_pitch(tau); f.toe = std::max(0.f, -f.pitch) * 0.8f;
        }
    }

    // ------------------------------------------------------------------ pelvis: targets, smoothing, constraints, balance
    Vec3 hip_pos(int s, Vec3 pelvis, float yaw) const { return pelvis + body_detail::lft(yaw) * (sg(s) * D.hip_w); }

    void clamp_reach(Vec3& p, float yaw) {
        for (int it = 0; it < 4; ++it)
            for (int s = 0; s < 2; ++s) {
                if (feet_[s].state != FootState::Planted) continue;
                Vec3 A = feet_[s].sole + Vec3(0, D.ankle_h, 0), H = hip_pos(s, p, yaw), d = H - A; float l = d.length();
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
        clamp_reach(np_, yaw_);
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
                    np_.x += (pr.x - x.x) / k; np_.z += (pr.z - x.z) / k; clamp_reach(np_, yaw_);
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
        Vec3 t = (torso_user_ ? torso_tgt_ : root + up * total) - root;
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
            float gy = std::atan2(d.x, d.z); chest_yaw = yaw_ + clampf(0.35f * wrap(gy - yaw_), -0.44f, 0.44f);
            Quat q0 = frameQ(cup, fwd(chest_yaw)); Vec3 Z = q0.rotate(Vec3(0, 0, 1)), X = q0.rotate(Vec3(1, 0, 0));
            float dx = dot(d, X), dz = dot(d, Z), dyy = dot(d, cup);
            lyaw = std::atan2(dx, dz); lpitch = std::atan2(dyy, std::hypot(dx, dz));
            float ym = 75.f * PI / 180.f, pu = 50.f * PI / 180.f, pd = -55.f * PI / 180.f;
            if ((std::fabs(lyaw) > ym || lpitch > pu || lpitch < pd) && !look_warned_) { event("LOOK limited by neck range (75 deg turn, +50/-55 deg tilt); the body must turn to look further"); look_warned_ = true; }
            lyaw = clampf(lyaw, -ym, ym); lpitch = clampf(lpitch, pd, pu);
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
            if (!hands_[s].target_mode && (!hands_[s].has_last || (restpt - hands_[s].last).length() > 0.03f * params.scale)) { arms_.reach_to(arm_id_[s], restpt); hands_[s].last = restpt; hands_[s].has_last = true; }
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
            Foot& f = feet_[s]; f.leg.root_position = hip_pos(s, pelvis_, yaw_);
            Vec3 pole = (fwd(f.yaw) * 0.6f + fwd(yaw_) * 0.4f).normalized();
            solve_two_bone_reach(f.leg, f.sole + Vec3(0, D.ankle_h, 0), pole, rmin_leg_, 0.999f * rmax_leg_);
            Vec3 p[3]; f.leg.world_positions(p); f.hip = p[0]; f.knee = p[1]; f.ankle = p[2];
        }
    }

    // ------------------------------------------------------------------ bones (positions + orientations) and fingers
    void put(int i, Vec3 h, Vec3 t, Vec3 zhint) { bones_[i].head = h; bones_[i].tail = t; bones_[i].q = body_detail::frameQ(t - h, zhint); }

    void compute_bones() {
        using namespace body_detail;
        Vec3 up(0, 1, 0), pf = fwd(yaw_);
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
        int np = 0; float surf = 0; for (int s = 0; s < 2; ++s) if (feet_[s].state == FootState::Planted) { ++np; surf += feet_[s].surface; }
        surf = np ? surf / np : 0.f;
        float h = std::max(0.3f, com_.y - surf), w0 = std::sqrt(9.81f / h);
        xcom_ = {com_.x + vcom_.x / w0, com_.z + vcom_.z / w0};
        margin_ = np ? hull_margin(support_, xcom_) : -1.f;
        com_off_ = {com_.x - pelvis_.x, com_.z - pelvis_.z};
        bool unstable = margin_ < params.fall_margin;
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
