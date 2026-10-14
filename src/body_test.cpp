// Body tests: physical invariants, not just "does it run". Exits non-zero on failure.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include "ik_body.hpp"
using namespace ik;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) ++g_pass; else { ++g_fail; std::printf("  FAIL (line %d) %s -- ", __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static const float DT = 1.f / 60.f;

static Body mk(bool assist = true, int lod = 1, float scale = 1.f) {
    Body::Params p; p.balance = assist ? Body::Balance::Assist : Body::Balance::None; p.hand_idle_m = 0.f; p.finger_lod = lod; p.scale = scale; return Body(p);
}
static void run(Body& b, float s) { for (float t = 0; t < s; t += DT) b.update(DT); }
static bool run_until(Body& b, float limit, bool (*done)(Body&)) { for (float t = 0; t < limit; t += DT) { b.update(DT); if (done(b)) return true; } return false; }
static bool idle(Body& b) { return !b.stepping(); }
static const BoneOut& bone(const Body& b, const char* n) { int i = b.bone_index(n); static BoneOut none{}; return i < 0 ? none : b.bones()[i]; }
static bool has(const std::vector<std::string>& ev, const char* frag) { for (auto& e : ev) if (e.find(frag) != std::string::npos) return true; return false; }
static std::string joined(const std::vector<std::string>& ev) { std::string s; for (auto& e : ev) s += e + " | "; return s; }
static float ang(Vec3 a, Vec3 b) { return std::acos(clampf(dot(a.normalized(), b.normalized()), -1.f, 1.f)) * 180.f / PI; }
static bool finite_all(const Body& b) {
    for (const auto& x : b.bones()) if (!x.head.is_finite() || !x.tail.is_finite() || !std::isfinite(x.q.w + x.q.x + x.q.y + x.q.z)) return false;
    return b.pelvis().is_finite() && b.com().is_finite() && std::isfinite(b.margin());
}
static float expect_len(const std::string& n, float s) {
    auto is = [&](const char* p) { return n.rfind(p, 0) == 0; };
    if (is("pelvis")) return .06f * s; if (is("spine1")) return .13f * s; if (is("spine2")) return .15f * s; if (is("spine3")) return .14f * s;
    if (is("neck")) return .10f * s; if (is("head")) return .22f * s; if (is("clavicle")) return .17f * s; if (is("upper_arm")) return .30f * s;
    if (is("forearm")) return .27f * s; if (is("hand")) return .09f * s; if (is("thigh") || is("shin")) return .43f * s;
    if (is("foot")) return std::sqrt(.15f * .15f + .08f * .08f) * s; if (is("toe")) return .06f * s;
    return -1.f;                                                                            // fingers: checked separately
}
static float worst_length_error(const Body& b, float s) {
    float w = 0; for (const auto& x : b.bones()) { float e = expect_len(x.name, s); if (e > 0) w = std::max(w, std::fabs((x.tail - x.head).length() - e)); } return w;
}

static void test_skeleton() {
    std::printf("[skeleton]\n");
    for (auto [lod, want] : {std::pair{0, 22}, {1, 32}, {2, 50}}) { Body b = mk(true, lod); CHECK(b.bone_count() == want, "LOD %d has %d bones, want %d", lod, b.bone_count(), want); }
    Body b = mk(); std::set<std::string> names; bool order = true;
    for (int i = 0; i < b.bone_count(); ++i) { names.insert(b.bones()[i].name); if (b.bones()[i].parent >= i) order = false; }
    CHECK(names.size() == (size_t)b.bone_count(), "bone names must be unique"); CHECK(order, "parents must precede children"); CHECK(b.bones()[0].parent == -1, "pelvis is the root");
    run(b, 1.f);
    float head_top = bone(b, "head").tail.y; std::printf("  standing height %.3f m\n", head_top);
    CHECK(std::fabs(head_top - 1.72f) < 0.05f, "~1.72 m tall at scale 1, got %.3f", head_top);
    Body half = mk(true, 1, 0.5f); run(half, 1.f); CHECK(std::fabs(bone(half, "head").tail.y - head_top * 0.5f) < 0.03f, "scale 0.5 halves the height (%.3f)", bone(half, "head").tail.y);
    CHECK(worst_length_error(b, 1.f) < 1e-3f, "bone lengths at rest (worst %.2e)", worst_length_error(b, 1.f));
    CHECK(worst_length_error(half, 0.5f) < 1e-3f, "scaled bone lengths (worst %.2e)", worst_length_error(half, 0.5f));
    float worst_axis = 0, worst_ortho = 0;
    for (const auto& x : b.bones()) {
        worst_axis = std::max(worst_axis, ang(x.q.rotate(Vec3(0, 1, 0)), x.tail - x.head));
        worst_ortho = std::max(worst_ortho, std::fabs(dot(x.q.rotate(Vec3(0, 1, 0)), x.q.rotate(Vec3(0, 0, 1)))));
    }
    CHECK(worst_axis < 0.1f, "every bone's Y axis lies along the bone (worst %.3f deg)", worst_axis); CHECK(worst_ortho < 1e-3f, "bone frames are orthonormal");
    // chains are connected end to end
    float gap = 0; const char* chain[][2] = {{"pelvis", "spine1"}, {"spine1", "spine2"}, {"spine2", "spine3"}, {"spine3", "neck"}, {"neck", "head"},
        {"upper_arm_L", "forearm_L"}, {"forearm_L", "hand_L"}, {"upper_arm_R", "forearm_R"}, {"forearm_R", "hand_R"}, {"thigh_L", "shin_L"}, {"shin_L", "foot_L"}, {"foot_L", "toe_L"}, {"thigh_R", "shin_R"}, {"shin_R", "foot_R"}, {"foot_R", "toe_R"}};
    for (auto& c : chain) gap = std::max(gap, (bone(b, c[0]).tail - bone(b, c[1]).head).length());
    CHECK(gap < 1e-3f, "limb chains are connected (worst gap %.2e)", gap);
}

static void test_standing() {
    std::printf("[standing]\n");
    Body b = mk(); run(b, 5.f);
    for (int s = 0; s < 2; ++s) {
        CHECK(b.foot_state(s) == Body::FootState::Planted, "foot %d planted", s); CHECK(std::fabs(b.foot_sole(s).y) < 1e-6f, "sole on the ground");
        for (const Vec3& c : b.foot_contact_points(s)) CHECK(c.y > -1e-5f, "contact point below the floor (%.6f)", c.y);
    }
    std::printf("  balance margin %.1f cm, pelvis y %.3f\n", b.margin() * 100, b.pelvis().y);
    CHECK(b.margin() > 0.03f, "standing is stable with margin > 3 cm, got %.3f", b.margin()); CHECK(!b.fallen(), "does not fall by itself");
    CHECK(std::fabs(b.pelvis().y - (0.08f + 0.985f * 0.86f)) < 2e-3f, "standing pelvis height %.3f", b.pelvis().y);
    Vec3 p0 = b.pelvis(); run(b, 5.f); CHECK((b.pelvis() - p0).length() < 1e-4f, "stands still (drift %.2e)", (b.pelvis() - p0).length());
    CHECK(b.take_events().empty(), "no events while idle");
}

static void test_hands() {
    std::printf("[hands, clavicle]\n");
    Body b = mk(); run(b, 1.f);
    float worst = 0; int n = 0;
    for (int s = 0; s < 2; ++s) for (float az : {0.f, 0.8f, 1.6f}) for (float h : {-0.35f, 0.f, 0.3f, 0.5f}) {
        float g = s == 0 ? 1.f : -1.f; Vec3 sh = bone(b, s ? "upper_arm_R" : "upper_arm_L").head;
        Vec3 t = sh + Vec3(std::sin(az) * 0.40f * g * 0.5f + g * 0.1f, h, std::cos(az) * 0.40f);
        if ((t - sh).length() > 0.52f) continue;
        auto r = b.command(std::string("HAND ") + (s ? "right " : "left ") + std::to_string(t.x) + " " + std::to_string(t.y) + " " + std::to_string(t.z)); CHECK(r.ok, "%s", r.msg.c_str());
        run(b, 1.6f); worst = std::max(worst, (b.wrist(s) - t).length()); ++n;
    }
    std::printf("  %d reaches, worst wrist error %.2f mm\n", n, worst * 1000);
    CHECK(n >= 12 && worst < 0.004f, "wrist reaches reachable targets (worst %.4f m over %d)", worst, n);
    CHECK(worst_length_error(b, 1.f) < 1e-3f, "arm bone lengths preserved (worst %.2e)", worst_length_error(b, 1.f));
    Body c = mk(); run(c, 1.f); float rest_y = bone(c, "upper_arm_L").head.y;
    Vec3 sh = bone(c, "upper_arm_L").head; c.command("HAND left " + std::to_string(sh.x + 0.1f) + " " + std::to_string(sh.y + 0.45f) + " " + std::to_string(sh.z + 0.2f)); run(c, 2.f);
    float raised = bone(c, "upper_arm_L").head.y; std::printf("  shoulder rises %.1f cm when the hand reaches high (clavicle)\n", (raised - rest_y) * 100);
    CHECK(raised - rest_y > 0.015f, "clavicle lifts the shoulder for a high reach (%.3f)", raised - rest_y);
    Body d = mk(); run(d, 1.f); auto r = d.command("HAND right -3 1 3"); run(d, 2.f);
    CHECK(r.ok && r.msg.find("stretch") != std::string::npos, "unreachable target is accepted with a warning: %s", r.msg.c_str()); CHECK(finite_all(d), "no NaN when stretching");
    float arm = (bone(d, "upper_arm_R").head - d.wrist(1)).length(); CHECK(arm < 0.575f && arm > 0.5f, "arm stretched straight toward it (%.3f m)", arm);
    d.command("HAND right rest"); run(d, 2.f); CHECK(!d.hand_targeted(1), "HAND rest releases the target");
}

static void test_pelvis_and_contact() {
    std::printf("[pelvis constraints, planted feet never slide]\n");
    Body b = mk(); run(b, 1.f); Vec3 s0 = b.foot_sole(0), s1 = b.foot_sole(1); std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1.f, 1.f);
    float slide = 0, maxlen = 0, minknee = 180; bool pen = false;
    for (int i = 0; i < 40; ++i) {
        b.command("PELVIS " + std::to_string(U(rng) * 0.6f) + " " + std::to_string(0.3f + (U(rng) + 1) * 0.5f) + " " + std::to_string(U(rng) * 0.6f));
        for (int f = 0; f < 45; ++f) {
            b.update(DT); slide = std::max(slide, std::max((b.foot_sole(0) - s0).length(), (b.foot_sole(1) - s1).length()));
            for (int s = 0; s < 2; ++s) { for (const Vec3& c : b.foot_contact_points(s)) if (c.y < -1e-4f) pen = true;
                const char* th = s ? "thigh_R" : "thigh_L"; const char* sh = s ? "shin_R" : "shin_L"; const char* ft = s ? "foot_R" : "foot_L";
                maxlen = std::max(maxlen, (bone(b, th).head - bone(b, ft).head).length());
                Vec3 a = (bone(b, th).head - bone(b, sh).head).normalized(), c2 = (bone(b, ft).head - bone(b, sh).head).normalized(); minknee = std::min(minknee, ang(a, c2)); }
        }
        if (b.fallen()) break;
    }
    std::printf("  40 random pelvis targets: foot slide %.2e m, max hip-ankle %.4f (limit %.4f), tightest knee %.1f deg\n", slide, maxlen, b.leg_reach_max(), minknee);
    CHECK(slide < 1e-5f, "planted feet never move while the pelvis does (%.2e)", slide); CHECK(!pen, "feet never sink into the floor");
    CHECK(maxlen <= b.leg_reach_max() + 1e-3f, "legs never exceed their length (%.4f)", maxlen); CHECK(minknee > 28.f, "knees respect their flexion limit (%.1f deg)", minknee);
    CHECK(!b.fallen(), "assist balance never lets pelvis commands knock the body over"); CHECK(worst_length_error(b, 1.f) < 1e-3f, "bone lengths intact (%.2e)", worst_length_error(b, 1.f));
    Body h = mk(); run(h, 1.f); h.command("PELVIS 0 3 0"); run(h, 3.f);
    CHECK((bone(h, "thigh_L").head - bone(h, "foot_L").head).length() <= h.leg_reach_max() + 1e-3f, "PELVIS far too high is clamped by leg length"); CHECK(has(h.take_events(), "leg length"), "...and the body says so");
    Body l = mk(); run(l, 1.f); l.command("PELVIS 0 0.1 0"); run(l, 3.f);
    CHECK(l.pelvis().y > 0.28f && l.pelvis().y < 0.5f, "PELVIS far too low stops at the deepest squat (y=%.3f)", l.pelvis().y); CHECK(!l.fallen(), "a deep squat is stable");
    Body w = mk(); run(w, 1.f); w.command("PELVIS 2 0.93 0"); run(w, 4.f);
    std::printf("  asked for the pelvis 2 m sideways: it stopped at x=%.3f, margin %.1f cm\n", w.pelvis().x, w.margin() * 100);
    CHECK(w.pelvis().x < 0.25f && w.margin() > 0.01f && !w.fallen(), "assist holds the centre of mass over the feet (x=%.3f, margin %.3f)", w.pelvis().x, w.margin());
}

static void test_balance_modes() {
    std::printf("[balance: assist vs none]\n");
    Body n = mk(false); run(n, 1.f); n.command("PELVIS 0.4 0.93 0");
    bool fell = run_until(n, 3.f, [](Body& b) { return b.fallen(); }); auto ev = n.take_events();
    std::printf("  no assist, pelvis 0.4 m sideways: fell=%d (%s)\n", (int)fell, ev.empty() ? "" : ev.back().c_str());
    CHECK(fell, "without assist, leaning far outside the feet makes the body fall"); CHECK(has(ev, "FELL"), "the fall is reported with its cause: %s", joined(ev).c_str());
    CHECK(!n.command("HAND left 0.3 1 0.3").ok, "commands are refused while fallen"); CHECK(!n.command("FOOT left 0.1 0 0.3").ok, "steps are refused while fallen");
    CHECK(n.command("STAND").ok && !n.fallen(), "STAND recovers"); run(n, 2.f);
    CHECK(!n.fallen() && n.margin() > 0.03f && n.foot_state(0) == Body::FootState::Planted, "stable and planted after STAND (margin %.3f)", n.margin());
    Body a = mk(true); run(a, 1.f); a.command("PELVIS 0.4 0.93 0"); float minm = 9; for (float t = 0; t < 10.f; t += DT) { a.update(DT); minm = std::min(minm, a.margin()); }
    CHECK(!a.fallen() && minm > -0.005f, "assist never falls from the same command (min margin %.3f)", minm);
    // lifting a foot with the weight centred: the body must really fall without the weight-shift controller
    Body f = mk(false); run(f, 1.f); f.command("FOOT right -0.10 0 0.30");
    bool fell2 = run_until(f, 4.f, [](Body& b) { return b.fallen(); });
    std::printf("  no assist, step with the weight still centred: fell=%d\n", (int)fell2);
    CHECK(fell2, "stepping without shifting weight first is a fall (balance is physics, not scripted away)");
    Body g = mk(true); run(g, 1.f); g.command("FOOT right -0.10 0 0.30"); run(g, 3.f); CHECK(!g.fallen(), "the same step with assist does not fall");
}

static void test_stepping() {
    std::printf("[stepping]\n");
    Body b = mk(); run(b, 1.f); Vec3 left0 = b.foot_sole(0); b.take_events();
    auto r = b.command("FOOT right -0.10 0 0.35"); CHECK(r.ok, "%s", r.msg.c_str());
    float t_swing = -1, minm = 9, maxlift = 0, left_slide = 0, miny = 9, pelvis_x_at_lift = 9; bool prev_swing = false; float t = 0;
    for (; t < 4.f; t += DT) {
        b.update(DT); minm = std::min(minm, b.margin());
        bool sw = b.foot_state(1) == Body::FootState::Swing;
        if (sw && !prev_swing) { t_swing = t; pelvis_x_at_lift = b.pelvis().x - left0.x; }
        prev_swing = sw; maxlift = std::max(maxlift, b.foot_sole(1).y); left_slide = std::max(left_slide, (b.foot_sole(0) - left0).length());
        for (const Vec3& c : b.foot_contact_points(1)) miny = std::min(miny, c.y);
        if (!b.stepping() && t > 0.2f) break;
    }
    auto ev = b.take_events();
    std::printf("  step 0.35 m: weight shifted over the support foot %.2f s in (pelvis %.1f cm from it), done in %.2f s, lift %.1f cm, min margin %.1f cm\n", t_swing, pelvis_x_at_lift * 100, t, maxlift * 100, minm * 100);
    CHECK(b.foot_state(1) == Body::FootState::Planted, "the foot lands and plants"); CHECK((b.foot_sole(1) - Vec3(-0.10f, 0, 0.35f)).length() < 0.005f, "lands within 5 mm of the target (%.4f)", (b.foot_sole(1) - Vec3(-0.10f, 0, 0.35f)).length());
    CHECK(t_swing > 0.1f, "it shifts weight before lifting (%.2f s)", t_swing); CHECK(std::fabs(pelvis_x_at_lift) < 0.05f, "pelvis is over the support foot when the other lifts (%.3f)", pelvis_x_at_lift);
    CHECK(maxlift > 0.04f && maxlift < 0.25f, "the swing foot clears the ground (%.3f)", maxlift); CHECK(miny > -1e-4f, "no contact point goes below the floor (%.5f)", miny);
    CHECK(left_slide < 1e-6f, "the support foot does not slide (%.2e)", left_slide); CHECK(minm > -0.01f, "stays balanced throughout (min margin %.3f)", minm);
    CHECK(has(ev, "planted at"), "landing is reported: %s", joined(ev).c_str()); CHECK(t < 2.5f, "a step takes a plausible time (%.2f s)", t);

    // validation, each with a reason
    Body v = mk(); run(v, 1.f); v.add_surface(-3, 3, 0.5f, 3, 0.2f); v.add_surface(-3, 3, 3.5f, 6, 0.6f); v.add_surface(-3, 3, -4, -1, -0.0f);
    struct { const char* cmd; const char* frag; } bad[] = {{"FOOT right -0.1 0 1.5", "too long"}, {"FOOT right 0.05 0 0.0", "collide"}};
    for (auto& c : bad) { v.command(c.cmd); run(v, 0.3f); auto e = v.take_events(); CHECK(has(e, c.frag), "'%s' must say '%s': %s", c.cmd, c.frag, joined(e).c_str()); CHECK(v.foot_state(1) == Body::FootState::Planted && v.foot_sole(1).z == 0.f, "a refused step moves nothing"); }
    v.command("FOOT right -0.1 0.2 0.70"); run(v, 3.f); CHECK(v.take_events().size() >= 1, "step events");
    CHECK(std::fabs(v.foot_sole(1).y - 0.2f) < 1e-4f && v.foot_state(1) == Body::FootState::Planted, "stepping onto a 20 cm platform plants at its height (y=%.3f)", v.foot_sole(1).y);
    v.command("FOOT left 0.1 0.2 0.8"); run(v, 3.f); run(v, 1.f);
    std::printf("  stepped up 0.2 m: pelvis y %.3f (flat: 0.927)\n", v.pelvis().y);
    CHECK(v.pelvis().y > 0.927f + 0.15f && !v.fallen(), "the pelvis rises with the platform and the body stays up (y=%.3f)", v.pelvis().y);
    v.command("FOOT right -0.1 0.6 1.0"); run(v, 1.f); (void)v.take_events();       // 40 cm up from a 20 cm platform at z in [0.5,3] -> no surface there: foot target at 0.6 is air (leg lift 0.4 <= 0.6)
    Body w = mk(); w.add_surface(-3, 3, 0.3f, 3, 0.7f); run(w, 1.f); w.command("FOOT right -0.1 0.7 0.5"); run(w, 0.5f); auto ew = w.take_events();
    CHECK(has(ew, "step up of"), "a 70 cm step up is refused with its height: %s", joined(ew).c_str());
    Body d = mk(); d.add_surface(-3, 3, -5, -0.5f, -1.0f); run(d, 1.f); d.command("FOOT right -0.1 -1.0 0.0"); run(d, 0.5f); (void)d.take_events();
    Body air = mk(); run(air, 1.f); air.command("FOOT right -0.1 0.2 0.3"); run(air, 3.f);
    CHECK(air.foot_state(1) == Body::FootState::Air && !air.fallen(), "a foot can be held in the air while the other supports"); air.take_events();
    air.command("FOOT left 0.1 0.2 0.3"); run(air, 1.f); auto ea = air.take_events();
    CHECK(has(ea, "not on the ground"), "lifting the second foot is refused (jumping isn't implemented): %s", joined(ea).c_str());
    air.command("FOOT right -0.1 0 0.3"); run(air, 3.f); CHECK(air.foot_state(1) == Body::FootState::Planted, "the held foot can be put back down");
}

static void test_walking_is_coordinates() {
    std::printf("[walking = a sequence of FOOT coordinates]\n");
    Body b = mk(); run(b, 1.f); b.take_events();
    Vec3 planted_at[2]; bool have[2] = {false, false}; float slide = 0, minm = 9, t_total = 0; int steps = 0;
    for (int i = 0; i < 8; ++i) {
        int s = (i % 2 == 0) ? 1 : 0, o = 1 - s; Vec3 tgt = b.foot_sole(o) + Vec3(s == 0 ? 0.10f - b.foot_sole(o).x * 0 : 0, 0, 0.45f); tgt.x = (s == 0 ? 0.10f : -0.10f); tgt.y = 0;
        auto r = b.command(std::string("FOOT ") + (s ? "right " : "left ") + std::to_string(tgt.x) + " 0 " + std::to_string(tgt.z)); CHECK(r.ok, "%s", r.msg.c_str());
        for (float t = 0; t < 4.f && !b.fallen(); t += DT) {
            b.update(DT); t_total += DT; minm = std::min(minm, b.margin());
            for (int f = 0; f < 2; ++f) { if (b.foot_state(f) == Body::FootState::Planted) { if (!have[f]) { planted_at[f] = b.foot_sole(f); have[f] = true; } else slide = std::max(slide, (b.foot_sole(f) - planted_at[f]).length()); } else have[f] = false; }
            if (!b.stepping() && t > 0.1f) break;
        }
        ++steps; if (b.fallen()) break;
    }
    run(b, 1.5f); float dist = b.pelvis().z;
    std::printf("  %d steps: pelvis moved %.2f m in %.1f s (%.2f m/s), min margin %.1f cm, foot slide %.1e m, fell=%d\n", steps, dist, t_total, dist / t_total, minm * 100, slide, (int)b.fallen());
    CHECK(!b.fallen(), "walks 8 steps without falling"); CHECK(dist > 2.5f, "the body actually travelled (%.2f m)", dist); CHECK(slide < 1e-5f, "no foot ever slides while planted (%.2e)", slide);
    CHECK(minm > -0.015f, "balance held through every step (min margin %.3f)", minm); CHECK(dist / t_total > 0.3f, "a plausible slow walk (%.2f m/s)", dist / t_total);
    CHECK(b.foot_state(0) == Body::FootState::Planted && b.foot_state(1) == Body::FootState::Planted, "ends standing"); CHECK(worst_length_error(b, 1.f) < 1e-3f, "bones intact after walking");
    // queued steps run in order without further commands
    Body q = mk(); run(q, 1.f); q.command("FOOT right -0.1 0 0.4"); q.command("FOOT left 0.1 0 0.8"); q.command("FOOT right -0.1 0 1.2"); CHECK(q.steps_queued() == 3, "three steps queued (%d)", q.steps_queued());
    run_until(q, 12.f, idle); run(q, 1.5f); CHECK(!q.fallen() && std::fabs(q.foot_sole(1).z - 1.2f) < 0.005f && std::fabs(q.foot_sole(0).z - 0.8f) < 0.005f, "queued steps all land in order (%.3f, %.3f)", q.foot_sole(0).z, q.foot_sole(1).z);
}

static void test_head_and_torso() {
    std::printf("[look, torso, fingers]\n");
    Body b = mk(); run(b, 1.f); Vec3 hc = (bone(b, "head").head + bone(b, "head").tail) * 0.5f;
    float ay = 40.f * PI / 180.f, ap = 10.f * PI / 180.f; Vec3 dir(std::sin(ay) * std::cos(ap), std::sin(ap), std::cos(ay) * std::cos(ap)), tgt = hc + dir * 2.f;
    b.command("LOOK " + std::to_string(tgt.x) + " " + std::to_string(tgt.y) + " " + std::to_string(tgt.z)); run(b, 1.f);
    hc = (bone(b, "head").head + bone(b, "head").tail) * 0.5f; Vec3 front = bone(b, "head").q.rotate(Vec3(0, 0, 1)); float e = ang(front, tgt - hc);
    std::printf("  look 40 deg left / 10 up: head faces within %.2f deg\n", e); CHECK(e < 2.f, "head faces the target (%.2f deg off)", e);
    float an = ang(bone(b, "neck").q.rotate(Vec3(0, 0, 1)), Vec3(0, 0, 1)), ah = ang(front, Vec3(0, 0, 1));
    CHECK(an > 0.2f * ah && an < ah, "the neck shares the turn with the head (%.1f of %.1f deg)", an, ah);
    Body c = mk(); run(c, 1.f); hc = (bone(c, "head").head + bone(c, "head").tail) * 0.5f; c.command("LOOK " + std::to_string(hc.x - 1) + " " + std::to_string(hc.y) + " " + std::to_string(hc.z - 1)); run(c, 1.5f);
    float behind = ang(bone(c, "head").q.rotate(Vec3(0, 0, 1)), bone(c, "spine3").q.rotate(Vec3(0, 0, 1))); auto ev = c.take_events();
    std::printf("  look behind-left (225 deg): head turns %.1f deg from the chest (limit ~75)\n", behind);
    CHECK(behind < 85.f && has(ev, "LOOK limited"), "an out-of-range look is clamped and reported (%.1f deg): %s", behind, joined(ev).c_str()); CHECK(finite_all(c), "finite");
    Body t = mk(); run(t, 1.f); Vec3 n0 = bone(t, "neck").head; t.command("TORSO " + std::to_string(n0.x) + " " + std::to_string(n0.y - 0.12f) + " " + std::to_string(n0.z + 0.2f)); run(t, 1.f);
    Vec3 n1 = bone(t, "neck").head; std::printf("  TORSO forward lean: neck base moved %.1f cm forward, %.1f cm down\n", (n1.z - n0.z) * 100, (n0.y - n1.y) * 100);
    CHECK(n1.z - n0.z > 0.08f && n0.y - n1.y > 0.02f, "the spine bends to lean the torso"); CHECK(worst_length_error(t, 1.f) < 1e-3f, "spine bones keep their length (%.2e)", worst_length_error(t, 1.f));
    float bend = ang(bone(t, "spine1").tail - bone(t, "spine1").head, bone(t, "spine3").tail - bone(t, "spine3").head); CHECK(bend < 3 * 22.f + 1.f, "spine bend within its limits (%.1f deg)", bend);
    // fingers
    Body f = mk(); run(f, 1.f); Vec3 base0 = bone(f, "index_L").head, tip0 = bone(f, "index_L").tail, wr = bone(f, "hand_L").head; f.command("GRIP left 1"); run(f, 0.1f);
    Vec3 tip1 = bone(f, "index_L").tail; float closer = (tip0 - wr).length() - (tip1 - wr).length();
    std::printf("  GRIP 1: index fingertip moves %.1f cm toward the palm\n", closer * 100); CHECK(closer > 0.02f, "closing the grip curls the fingers (%.3f)", closer);
    CHECK(!f.command("GRIP left 2").ok == false && f.command("GRIP left 2").msg.find("1.00") != std::string::npos, "grip is clamped to 0..1"); CHECK(!f.command("GRIP middle 1").ok, "bad side refused");
    Vec3 thumb = bone(f, "thumb_L").tail, pinky = bone(f, "pinky_L").tail; (void)base0; CHECK((bone(f, "index_L").head - bone(f, "pinky_L").head).length() > 0.04f, "fingers are spread across the palm");
    Body f0 = mk(true, 0); run(f0, 1.f); CHECK(f0.bone_index("index_L") < 0 && f0.command("GRIP left 1").ok, "LOD 0 has no finger bones and ignores GRIP harmlessly");
    Body f2 = mk(true, 2); run(f2, 1.f); f2.command("GRIP right 1"); run(f2, 0.1f); float gap = 0;
    for (const char* fg : {"index", "middle", "ring", "pinky"}) { gap = std::max(gap, (bone(f2, (std::string(fg) + "_1_R").c_str()).tail - bone(f2, (std::string(fg) + "_2_R").c_str()).head).length()); gap = std::max(gap, (bone(f2, (std::string(fg) + "_2_R").c_str()).tail - bone(f2, (std::string(fg) + "_3_R").c_str()).head).length()); }
    CHECK(gap < 1e-5f, "full-detail phalanges are connected end to end (%.2e)", gap); (void)thumb; (void)pinky;
}

static void test_protocol_and_fuzz() {
    std::printf("[protocol, fuzz]\n");
    Body b = mk(); run(b, 0.5f);
    for (const char* bad : {"", "DANCE", "HAND", "HAND middle 1 1 1", "HAND left 1 1", "HAND left nan 1 1", "HAND left 1 inf 1", "FOOT right 1 2", "FOOT right a b c", "FOOT right 0 0 0 x", "PELVIS 1 2", "FACE 1", "LOOK 1 2", "TORSO", "GRIP left", "BALANCE maybe", "STAND now", "SURFACE 1 2"}) {
        auto r = b.command(bad); CHECK(!r.ok && !r.msg.empty(), "'%s' must be refused with a message", bad);
    }
    CHECK(b.command("balance NONE").ok && b.command("Balance assist").ok, "verbs and arguments are case-insensitive"); CHECK(finite_all(b), "state survives garbage");
    // fuzz: random commands + random dt; invariants must always hold
    std::mt19937 rng(12345); std::uniform_real_distribution<float> U(0.f, 1.f);
    auto num = [&]() { float r = U(rng); if (r < 0.05f) return std::string("nan"); if (r < 0.08f) return std::string("inf"); if (r < 0.15f) return std::to_string((U(rng) - 0.5f) * 2e6f); return std::to_string((U(rng) - 0.5f) * 3.f); };
    const char* sides[] = {"left", "right", "middle"};
    Body f = mk(U(rng) < 0.5f); f.add_surface(-2, 2, 0.5f, 3, 0.15f); f.add_surface(-2, 2, -3, -0.5f, 0.3f);
    int bad_frames = 0; float worstlen = 0; std::string why;
    for (int i = 0; i < 40000; ++i) {
        int k = (int)(U(rng) * 12); std::string cmd;
        switch (k) {
            case 0: cmd = std::string("HAND ") + sides[(int)(U(rng) * 3)] + " " + num() + " " + num() + " " + num(); break;
            case 1: case 2: cmd = std::string("FOOT ") + sides[(int)(U(rng) * 3)] + " " + num() + " " + num() + " " + num(); break;
            case 3: cmd = "PELVIS " + num() + " " + num() + " " + num(); break;
            case 4: cmd = "FACE " + num() + " " + num(); break;
            case 5: cmd = "LOOK " + num() + " " + num() + " " + num(); break;
            case 6: cmd = "TORSO " + num() + " " + num() + " " + num(); break;
            case 7: cmd = std::string("GRIP ") + sides[(int)(U(rng) * 3)] + " " + num(); break;
            case 8: cmd = U(rng) < 0.5f ? "STAND" : "PELVIS auto"; break;
            case 9: cmd = std::string("BALANCE ") + (U(rng) < 0.7f ? "assist" : "none"); break;
            default: cmd = std::string("FOOT ") + sides[(int)(U(rng) * 2)] + " " + std::to_string((U(rng) - 0.5f) * 0.6f) + " " + std::to_string(U(rng) * 0.4f) + " " + std::to_string((U(rng) - 0.3f) * 1.2f);
        }
        f.command(cmd);
        int n = 1 + (int)(U(rng) * 4); for (int j = 0; j < n; ++j) f.update(1.f / (30.f + U(rng) * 120.f));
        if (i % 250 == 0) {
            f.take_events(); bool ok = finite_all(f);
            if (ok) { worstlen = std::max(worstlen, worst_length_error(f, 1.f)); for (int s = 0; s < 2; ++s) if (f.foot_state(s) == Body::FootState::Planted && f.foot_sole(s).y < f.surface_at(f.foot_sole(s).x, f.foot_sole(s).z) - 1e-3f) ok = false;
                if (std::fabs(f.pelvis().x) > 20 || std::fabs(f.pelvis().z) > 20) ok = false; }
            if (!ok) { ++bad_frames; if (why.empty()) why = "after '" + cmd + "'"; }
        }
    }
    std::printf("  40000 random commands: bad states %d, worst bone-length error %.2e m\n", bad_frames, worstlen);
    CHECK(bad_frames == 0, "invariants held through the fuzz run (first failure %s)", why.c_str()); CHECK(worstlen < 2e-3f, "bone lengths stayed exact (%.2e)", worstlen);
    Body p = mk(); run(p, 0.5f); auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 20000; ++i) p.update(DT);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20000; std::printf("  update cost: %.1f us per body-frame (32 bones)\n", us);
    CHECK(us < 150.0, "per-frame cost is small (%.1f us)", us);
}

int main() {
    test_skeleton(); test_standing(); test_hands(); test_pelvis_and_contact(); test_balance_modes(); test_stepping(); test_walking_is_coordinates(); test_head_and_torso(); test_protocol_and_fuzz();
    std::printf("\n%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
