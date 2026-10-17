#include "nebula_capi.h"
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include "ik_engine.hpp"
#include "ik_body.hpp"

using namespace ik;
struct NebulaEngine { IKEngine e; };
struct NebulaBody { ik::Body b; explicit NebulaBody(const ik::Body::Params& p) : b(p) {} };

static int copy_out(const std::string& s, char* buf, int cap) {      // truncating, always NUL-terminated
    if (buf && cap > 0) { int n = std::min<int>((int)s.size(), cap - 1); std::memcpy(buf, s.data(), n); buf[n] = 0; }
    return (int)s.size();
}

static Vec3 v3(const float* p) { return Vec3(p[0], p[1], p[2]); }
static void put3(float* o, const Vec3& v) { o[0] = v.x; o[1] = v.y; o[2] = v.z; }

extern "C" {

int nebula_abi_version(void) { return 3; }

NebulaEngine* nebula_create(void) { return new (std::nothrow) NebulaEngine(); }
void nebula_destroy(NebulaEngine* h) { delete h; }

int nebula_add_chain(NebulaEngine* h, const char* name, const float root[3], const float* q,
                     int solver, int n, const float* lens, const float pole[3],
                     float lean, float vmax) {
    if (!h || !name || !root || !lens || !pole || n < 1) return -1;
    std::vector<std::pair<std::string, float>> specs;
    for (int i = 0; i < n; ++i) specs.emplace_back("j" + std::to_string(i), lens[i]);
    Quat rq = q ? Quat(q[0], q[1], q[2], q[3]).normalized() : Quat::identity();
    return h->e.add_chain(name, v3(root), rq, specs,
                          solver == NEBULA_SOLVER_FABRIK ? SolverType::Fabrik : SolverType::TwoBoneAnalytic,
                          v3(pole), lean, vmax);
}
int nebula_find_chain(NebulaEngine* h, const char* name) { return (h && name) ? h->e.find_chain(name) : -1; }
int nebula_chain_count(NebulaEngine* h) { return h ? h->e.chain_count() : 0; }
int nebula_joint_count(NebulaEngine* h, int c) {
    if (!h || c < 0 || c >= h->e.chain_count()) return -1;
    return h->e.chain(c).count;
}
int nebula_set_anchor(NebulaEngine* h, int c, const float p[3], const float* q) {
    if (!h || !p) return 0;
    return h->e.set_anchor(c, v3(p), q ? Quat(q[0], q[1], q[2], q[3]) : Quat::identity());
}
int nebula_teleport(NebulaEngine* h, int c, const float p[3]) { return h && p && h->e.teleport_to(c, v3(p)); }
int nebula_set_pole(NebulaEngine* h, int c, const float p[3]) { return h && p && h->e.set_pole(c, v3(p)); }
int nebula_set_rest(NebulaEngine* h, int c, const float p[3]) { return h && p && h->e.set_rest_target(c, v3(p)); }
int nebula_set_elbow_limits(NebulaEngine* h, const char* n, float lo, float hi) { return h && n && h->e.set_elbow_limits(n, lo, hi); }
int nebula_set_max_bend(NebulaEngine* h, const char* n, int j, float r) { return h && n && h->e.set_max_bend(n, j, r); }
int nebula_command(NebulaEngine* h, const char* line) { return h && line && h->e.execute_command(line); }
int nebula_reach(NebulaEngine* h, int c, const float t[3]) { return h && t && h->e.reach_to(c, v3(t)); }

int nebula_set_param(NebulaEngine* h, const char* k, float v) {
    if (!h || !k || !std::isfinite(v)) return 0;
    MotionParams& p = h->e.params;
    struct { const char* n; float* f; } t[] = {
        {"reaction_delay", &p.reaction_delay}, {"fitts_a", &p.fitts_a}, {"fitts_b", &p.fitts_b},
        {"fitts_width", &p.fitts_width}, {"min_duration", &p.min_duration}, {"max_duration", &p.max_duration},
        {"path_curvature", &p.path_curvature}, {"pole_relax_rate", &p.pole_relax_rate},
        {"lean_omega", &p.lean_omega}, {"idle_amp_m", &p.idle_amp_m}, {"breath_amp_m", &p.breath_amp_m},
        {"breath_hz", &p.breath_hz}, {"solve_epsilon_m", &p.solve_epsilon_m}};
    for (auto& e : t) if (!std::strcmp(e.n, k)) { *e.f = v; return 1; }
    return 0;
}

void nebula_update(NebulaEngine* h, float dt) { if (h) h->e.update(dt); }

int nebula_world_positions(NebulaEngine* h, int c, float* out) {
    if (!h || !out || c < 0 || c >= h->e.chain_count()) return -1;
    Vec3 pos[kMaxJoints + 1];
    const Chain& ch = h->e.chain(c);
    ch.world_positions(pos);
    for (int i = 0; i <= ch.count; ++i) put3(out + 3 * i, pos[i]);
    return ch.count + 1;
}
int nebula_local_rotations(NebulaEngine* h, int c, float* out) {
    if (!h || !out || c < 0 || c >= h->e.chain_count()) return -1;
    const Chain& ch = h->e.chain(c);
    for (int i = 0; i < ch.count; ++i) {
        const Quat& q = ch.joints[i].local_rotation;
        out[4 * i] = q.w; out[4 * i + 1] = q.x; out[4 * i + 2] = q.y; out[4 * i + 3] = q.z;
    }
    return ch.count;
}
int nebula_is_settled(NebulaEngine* h, int c) { return (h && c >= 0 && c < h->e.chain_count()) ? (int)h->e.is_settled(c) : -1; }
int nebula_hand_velocity(NebulaEngine* h, int c, float out[3]) {
    if (!h || !out || c < 0 || c >= h->e.chain_count()) return 0;
    put3(out, h->e.end_effector_velocity(c));
    return 1;
}

// ---------------------------------------------------------------- body
NebulaBody* nebula_body_create(float scale, int lod, int assist) {
    ik::Body::Params p;
    if (!std::isfinite(scale) || scale < 0.2f || scale > 3.f) return nullptr;
    p.scale = scale; p.finger_lod = lod; p.balance = assist ? ik::Body::Balance::Assist : ik::Body::Balance::None;
    try { return new NebulaBody(p); } catch (...) { return nullptr; }
}
void nebula_body_destroy(NebulaBody* h) { delete h; }

int nebula_body_command(NebulaBody* h, const char* line, char* msg, int cap) {
    if (!h || !line) { copy_out("null argument", msg, cap); return 0; }
    ik::Body::Result r = h->b.command(line);
    copy_out(r.msg, msg, cap);
    return r.ok ? 1 : 0;
}
void nebula_body_update(NebulaBody* h, float dt) { if (h) h->b.update(dt); }

int nebula_body_set_param(NebulaBody* h, const char* k, float v) {
    if (!h || !k || !std::isfinite(v)) return 0;
    ik::Body::Params& p = h->b.params;
    if (!std::strcmp(k, "balance")) { p.balance = v >= 0.5f ? ik::Body::Balance::Assist : ik::Body::Balance::None; return 1; }
    if (!std::strcmp(k, "dynamic_gait")) { p.dynamic_gait = v >= 0.5f; return 1; }
    if (!std::strcmp(k, "hand_idle_m")) { h->b.set_hand_idle(std::max(0.f, v)); return 1; }
    struct { const char* n; float* f; float lo, hi; } t[] = {
        {"safe_margin", &p.safe_margin, 0.f, 0.1f}, {"step_time_scale", &p.step_time_scale, 0.2f, 5.f}, {"max_stride", &p.max_stride, 0.1f, 1.5f},
        {"max_step_up", &p.max_step_up, 0.f, 1.f}, {"max_step_down", &p.max_step_down, 0.f, 2.f}, {"fall_margin", &p.fall_margin, -1.f, 0.f},
        {"fall_time", &p.fall_time, 0.f, 5.f}, {"shift_settle_speed", &p.shift_settle_speed, 0.f, 1e9f},
        {"recover_time", &p.recover_time, 0.f, 60.f}, {"arm_swing_gain", &p.arm_swing_gain, 0.f, 2.f}, {"twist_gain", &p.twist_gain, 0.f, 1.f}, {"head_turn_omega", &p.head_turn_omega, 0.f, 100.f},
        {"leg_len_frac", &p.leg_len_frac, 0.8f, 0.999f}, {"step_ds_time", &p.step_ds_time, 0.02f, 0.5f}, {"swing_reach_slack", &p.swing_reach_slack, 0.f, 0.15f}};
    for (auto& e : t) if (!std::strcmp(e.n, k)) { *e.f = std::min(std::max(v, e.lo), e.hi); return 1; }
    return 0;
}
int nebula_body_add_surface(NebulaBody* h, float x0, float x1, float z0, float z1, float y) {
    if (!h || !std::isfinite(x0 + x1 + z0 + z1 + y)) return 0;
    h->b.add_surface(x0, x1, z0, z1, y); return 1;
}
void nebula_body_clear_surfaces(NebulaBody* h) { if (h) h->b.clear_surfaces(); }
int nebula_body_bone_count(NebulaBody* h) { return h ? h->b.bone_count() : 0; }
const char* nebula_body_bone_name(NebulaBody* h, int i) { return (h && i >= 0 && i < h->b.bone_count()) ? h->b.bones()[i].name : nullptr; }
int nebula_body_bone_parent(NebulaBody* h, int i) { return (h && i >= 0 && i < h->b.bone_count()) ? h->b.bones()[i].parent : -2; }
int nebula_body_bones(NebulaBody* h, float* out) {
    if (!h || !out) return 0;
    const auto& bs = h->b.bones();
    for (size_t i = 0; i < bs.size(); ++i) {
        float* o = out + 10 * i;
        o[0] = bs[i].head.x; o[1] = bs[i].head.y; o[2] = bs[i].head.z; o[3] = bs[i].tail.x; o[4] = bs[i].tail.y; o[5] = bs[i].tail.z;
        o[6] = bs[i].q.w; o[7] = bs[i].q.x; o[8] = bs[i].q.y; o[9] = bs[i].q.z;
    }
    return (int)bs.size();
}
int nebula_body_status_json(NebulaBody* h, char* buf, int cap) { return h ? copy_out(h->b.status_json(), buf, cap) : copy_out("{}", buf, cap); }

int nebula_body_state_dim(void) { return ik::Body::STATE_DIM; }
int nebula_body_action_dim(void) { return ik::Body::ACTION_DIM; }
int nebula_body_report_dim(void) { return ik::Body::REPORT_DIM; }
const char* nebula_body_state_name(int i) { return (i >= 0 && i < ik::Body::STATE_DIM) ? ik::Body::state_names()[i] : nullptr; }
int nebula_body_get_state(NebulaBody* h, float* out) { if (!h || !out) return 0; h->b.get_state(out); return ik::Body::STATE_DIM; }
int nebula_body_track(NebulaBody* h, const float* a, float dt, float* rep) {
    if (!h || !a) return 0; float tmp[ik::Body::REPORT_DIM]; h->b.track(a, dt, rep ? rep : tmp); return h->b.track_unbalanced() ? 1 : 0;
}
int nebula_body_track_clip(NebulaBody* h, const float* a, int n, float dt, float* reports, float* states) {
    if (!h || !a || n <= 0) return 0;      // look-ahead (where each lifted foot lands) lives in Body::track_clip
    return h->b.track_clip(a, n, dt, reports, states);
}
// --- snapshots and clip verdicts -------------------------------------------------------------------------------------------------
NebulaBody* nebula_body_clone(NebulaBody* h) { return h ? new (std::nothrow) NebulaBody(*h) : nullptr; }       // full snapshot (pose, gait/tracking state, queued steps, params, surfaces)
int nebula_body_restore(NebulaBody* dst, const NebulaBody* src) { if (!dst || !src) return 0; dst->b = src->b; return 1; }

static void json_str(std::string& o, const std::string& s) {
    o += '"'; for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if ((unsigned char)c < 0x20) o += ' '; else o += c; } o += '"';
}
int nebula_body_evaluate_clip(NebulaBody* h, const float* a, int n, float dt, int commit, float* reports, float* states, char* json, int json_cap) {
    if (!h || !a || n <= 0) { copy_out("{\"feasible\":false,\"issues\":[\"no clip given\"]}", json, json_cap); return -1; }
    ik::Body::ClipVerdict v;
    if (commit) v = h->b.evaluate_clip(a, n, dt, reports, states); else v = h->b.preview_clip(a, n, dt, reports, states);
    std::string j; char b[160];
    std::snprintf(b, sizeof b, "{\"feasible\":%s,\"frames\":%d,\"bad_frames\":%d,\"score\":%.6g,\"worst_frame\":%d,\"worst_cost\":%.6g,\"issues\":[", v.feasible ? "true" : "false", v.frames, v.bad_frames, v.score, v.worst_frame, v.worst_cost);
    j += b; for (size_t i = 0; i < v.issues.size(); ++i) { if (i) j += ','; json_str(j, v.issues[i]); } j += "]}";
    copy_out(j, json, json_cap);
    return v.bad_frames;
}
/* Track a clip and also return every frame's bones (n * bone_count * 10 floats, same layout as nebula_body_bones) - for viewers. Same tracking as track_clip. */
int nebula_body_replay_clip(NebulaBody* h, const float* a, int n, float dt, float* reports, float* states, float* bones) {
    if (!h || !a || n <= 0) return -1;
    const size_t nb = h->b.bones().size();
    return h->b.track_clip_each(a, n, dt, reports, states, [&](int i) {
        if (!bones) return;
        const auto& bs = h->b.bones(); float* base = bones + (size_t)i * nb * 10;
        for (size_t k = 0; k < nb; ++k) {
            float* o = base + 10 * k;
            o[0] = bs[k].head.x; o[1] = bs[k].head.y; o[2] = bs[k].head.z; o[3] = bs[k].tail.x; o[4] = bs[k].tail.y; o[5] = bs[k].tail.z;
            o[6] = bs[k].q.w; o[7] = bs[k].q.x; o[8] = bs[k].q.y; o[9] = bs[k].q.z;
        }
    });
}
void nebula_body_track_end(NebulaBody* h) { if (h) h->b.track_end(); }
void nebula_body_reset(NebulaBody* h) { if (h) { ik::Body::Params p = h->b.params; h->b.~Body(); new (&h->b) ik::Body(p); } }

} // extern "C"
