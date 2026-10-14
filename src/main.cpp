// main.cpp -- correctness tests + usage examples. Exits non-zero on failure,
// so `make test` works in CI. Every check prints what it measured.
#include <cstdio>
#include <vector>
#include <algorithm>
#include "ik_engine.hpp"

using namespace ik;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; } else { ++g_fail; \
    std::printf("  FAIL (%s:%d) %s  -- ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const float DT = 1.f / 60.f;

static ChainId add_arm(IKEngine& e, const char* name = "arm", float torso_lean = 0.10f) {
    return e.add_chain(name, Vec3(0.f, 1.4f, 0.f), Quat::identity(), {{"elbow", 0.30f}, {"wrist", 0.25f}},
                       SolverType::TwoBoneAnalytic, Vec3(0.3f, -1.f, 0.6f),
                       torso_lean, /*max_speed_m_per_s=*/1.2f);
}
// Deterministic physics-free test setup: no idle wander, no breathing.
static void quiet(IKEngine& e) { e.params.idle_amp_m = 0.f; e.params.breath_amp_m = 0.f; }
static void run(IKEngine& e, float seconds, float dt = DT) { for (float t = 0; t < seconds; t += dt) e.update(dt); }
static float elbow_angle(const Chain& c) {
    auto p = c.world_positions();
    return std::acos(clampf(dot((p[0] - p[1]).normalized(), (p[2] - p[1]).normalized()), -1.f, 1.f));
}

static void test_trajectory_math() {
    std::printf("[quintic with arbitrary initial conditions]\n");
    Trajectory tr;
    Vec3 p0(0.1f, 0.2f, 0.3f), v0(0.5f, -0.3f, 0.2f), a0(1.f, 0.5f, -2.f), pf(0.6f, 0.9f, -0.2f);
    tr.plan(p0, v0, a0, pf, 0.5f, Vec3(), 0.f);
    Vec3 p, v, a;
    tr.eval(0.f, p, v, a);
    CHECK((p - p0).length() < 1e-5f && (v - v0).length() < 1e-5f && (a - a0).length() < 1e-4f, "start state wrong");
    tr.eval(0.5f, p, v, a);
    std::printf("  end: pos err %.2e m, vel %.2e m/s, acc %.2e m/s^2\n", (p - pf).length(), v.length(), a.length());
    CHECK((p - pf).length() < 1e-4f && v.length() < 1e-3f && a.length() < 5e-3f, "end state not (pf, 0, 0)");
    // bow must not disturb boundary conditions
    tr.plan(p0, Vec3(), Vec3(), pf, 0.5f, Vec3(0, 1, 0), 0.05f);
    tr.eval(0.f, p, v, a);  CHECK(v.length() < 1e-5f && a.length() < 1e-4f, "bow broke start conditions");
    tr.eval(0.5f, p, v, a); CHECK((p - pf).length() < 1e-4f && v.length() < 1e-3f && a.length() < 5e-3f, "bow broke end conditions");
}

static void test_reach_and_profile() {
    std::printf("[reach: convergence, straight path, bell-shaped velocity, reaction delay]\n");
    IKEngine e; quiet(e); e.params.path_curvature = 0.f;
    ChainId id = add_arm(e);
    run(e, 0.5f);
    Vec3 p0 = e.end_effector_position(id), target(0.30f, 1.20f, 0.20f);

    const float dt = 1.f / 240.f;
    e.reach_to(id, target);
    float T = e.trajectory_duration(id), D = (target - p0).length();
    Vec3 dir = (target - p0) * (1.f / D);
    float maxdev = 0, peak = 0, early_move = 0, first_moving_t = -1;
    Vec3 prev = p0;
    std::vector<float> speeds;
    for (int i = 0; i < (int)((T + 0.3f) / dt); ++i) {
        e.update(dt);
        Vec3 cur = e.end_effector_position(id);
        float sp = (cur - prev).length() / dt; speeds.push_back(sp); peak = std::max(peak, sp);
        Vec3 rel = cur - p0; maxdev = std::max(maxdev, (rel - dir * dot(rel, dir)).length());
        if ((i + 1) * dt <= 0.09f) early_move = std::max(early_move, (cur - p0).length());
        if (first_moving_t < 0 && (cur - p0).length() > 1e-4f) first_moving_t = (i + 1) * dt;
        prev = cur;
    }
    float err = (e.end_effector_position(id) - target).length();
    std::printf("  D=%.3f m, T=%.3f s (Fitts), final error %.2e m, path deviation %.2e m\n", D, T, err, maxdev);
    std::printf("  peak speed %.3f m/s vs theoretical 1.875*D/T = %.3f m/s\n", peak, 1.875f * D / T);
    std::printf("  first motion at %.3f s (reaction delay 0.100 s)\n", first_moving_t);
    CHECK(err < 1e-4f, "did not converge: %.2e", err);
    CHECK(maxdev < 2e-4f, "path not straight: %.2e", maxdev);
    CHECK(std::fabs(peak - 1.875f * D / T) < 0.03f * 1.875f * D / T, "peak speed off min-jerk theory");
    CHECK(early_move < 1e-5f, "moved during the reaction delay (%.2e)", early_move);
    CHECK(first_moving_t > 0.099f && first_moving_t < 0.13f, "motion started at %.3f", first_moving_t);
    // bell shape: rises to a single peak near the middle, near-zero at both ends
    size_t ipk = std::max_element(speeds.begin(), speeds.end()) - speeds.begin();
    float t_pk = (ipk + 1) * dt - 0.1f;
    CHECK(std::fabs(t_pk / T - 0.5f) < 0.06f, "peak at %.2f of the movement, expected 0.50", t_pk / T);
    std::printf("  velocity peaks at %.0f%% of the movement\n", 100.f * t_pk / T);

    // with the default natural bow the path leaves the line by ~path_curvature*D
    IKEngine e2; quiet(e2); ChainId id2 = add_arm(e2); run(e2, 0.5f);
    Vec3 q0 = e2.end_effector_position(id2); e2.reach_to(id2, target);
    float dev2 = 0; for (int i = 0; i < 120; ++i) { e2.update(DT); Vec3 rel = e2.end_effector_position(id2) - q0; dev2 = std::max(dev2, (rel - dir * dot(rel, dir)).length()); }
    std::printf("  default natural bow: %.1f%% of distance (configured %.0f%%)\n", 100.f * dev2 / D, 100.f * e2.params.path_curvature);
    CHECK(dev2 > 0.02f * D && dev2 < 0.06f * D, "bow %.3f of D", dev2 / D);
}

static void test_retarget_continuity() {
    std::printf("[retarget mid-motion: velocity continuity, no second reaction delay]\n");
    IKEngine e; quiet(e);
    ChainId id = add_arm(e); run(e, 0.5f);
    e.reach_to(id, Vec3(0.30f, 1.20f, 0.20f)); run(e, 1.f);
    e.reach_to(id, Vec3(-0.30f, 1.45f, 0.20f));
    Vec3 prev = e.end_effector_position(id); std::vector<Vec3> vel; int rf = 0; float peak = 0;
    for (int f = 0; f < 60; ++f) {
        if (f == 22) { e.reach_to(id, Vec3(0.10f, 1.05f, 0.40f)); rf = (int)vel.size(); }
        e.update(DT);
        Vec3 cur = e.end_effector_position(id);
        vel.push_back((cur - prev) * (1.f / DT)); peak = std::max(peak, vel.back().length()); prev = cur;
    }
    float jump = 0;
    for (int k = rf; k < rf + 3; ++k) jump = std::max(jump, (vel[k] - vel[k - 1]).length());
    std::printf("  velocity jump across retarget: %.3f m/s = %.1f%% of peak %.2f m/s (original engine: 132%%)\n", jump, 100.f * jump / peak, peak);
    CHECK(jump < 0.15f * peak, "velocity snapped at retarget");
    CHECK(vel[rf].length() > 0.5f * vel[rf - 1].length(), "motion stalled at retarget (re-paid the reaction delay?)");
}

static void test_torso_lean() {
    std::printf("[torso lean: smooth, and actually extends reach]\n");
    IKEngine e; quiet(e);
    ChainId id = add_arm(e); run(e, 0.5f);
    Vec3 far_target(0.55f, 1.40f, 0.15f);       // 0.57 m from the shoulder > 0.55 m arm
    Vec3 hand_before = e.end_effector_position(id);
    e.reach_to(id, far_target);
    CHECK((e.end_effector_position(id) - hand_before).length() < 1e-6f, "hand teleported inside reach_to()");
    Vec3 prev_root = e.root_position(id); float max_step = 0;
    for (int f = 0; f < 120; ++f) { e.update(DT); Vec3 r = e.root_position(id); max_step = std::max(max_step, (r - prev_root).length()); prev_root = r; }
    float err = (e.end_effector_position(id) - far_target).length();
    float lean = (e.root_position(id) - Vec3(0, 1.4f, 0)).length();
    std::printf("  shoulder moved %.1f mm total, max %.1f mm/frame (original: %.0f mm in ONE frame); hand error %.2e m\n", lean * 1e3f, max_step * 1e3f, 100.0, err);
    CHECK(max_step < 0.02f, "shoulder popped: %.1f mm/frame", max_step * 1e3f);
    CHECK(err < 1e-3f, "lean did not bring the far target into reach: %.2e", err);
}

static void test_limits_and_unreachable() {
    std::printf("[unreachable targets, elbow limits]\n");
    IKEngine e; quiet(e);
    ChainId id = add_arm(e, "arm", /*torso_lean=*/0.f);      // lean would move the shoulder and muddy the angle checks
    // A 3 m reach is capped by the chain's 1.2 m/s peak speed: movement time ~4.7 s.
    e.reach_to(id, Vec3(3.f, 1.4f, 0.f)); run(e, 6.f);
    Vec3 h = e.end_effector_position(id), r = e.root_position(id);
    std::printf("  unreachable: hand at (%.3f, %.3f, %.3f), shoulder-hand %.4f m (arm length 0.55)\n", h.x, h.y, h.z, (h - r).length());
    CHECK(h.is_finite(), "NaN/inf pose");
    CHECK(std::fabs((h - r).length() - 0.55f) < 2e-3f, "arm not fully extended toward target");

    e.set_elbow_limit("arm", 25.f * PI / 180.f);
    e.reach_to(id, Vec3(0.02f, 1.39f, 0.02f)); run(e, 6.f);
    float ang = elbow_angle(e.chain(id)) * 180.f / PI;
    std::printf("  target at the shoulder: elbow interior angle %.2f deg (min 25)\n", ang);
    CHECK(ang > 24.9f && ang < 25.5f, "elbow should sit on its 25 deg limit: %.2f", ang);

    e.set_elbow_limits("arm", 25.f * PI / 180.f, 160.f * PI / 180.f);
    e.reach_to(id, Vec3(0.54f, 1.4f, 0.f)); run(e, 6.f);
    ang = elbow_angle(e.chain(id)) * 180.f / PI;
    std::printf("  near-full reach: elbow interior angle %.2f deg (max 160)\n", ang);
    CHECK(ang < 160.1f, "elbow hyper-extended: %.2f", ang);
    CHECK(ang > 155.f, "elbow limit clipped far too much: %.2f", ang);
}

static void test_fabrik() {
    std::printf("[FABRIK: convergence, cone limits]\n");
    IKEngine e; quiet(e);
    ChainId f = e.add_chain("finger", Vec3(0, 1, 0), Quat::identity(),
                            {{"a", 0.04f}, {"b", 0.03f}, {"c", 0.02f}, {"d", 0.02f}},
                            SolverType::Fabrik, Vec3(0, 1, 0), 0.f, 1.f);
    Vec3 t(0.06f, 1.05f, 0.03f);
    e.reach_to(f, t); run(e, 1.5f);
    float err = (e.end_effector_position(f) - t).length();
    std::printf("  unconstrained: error %.2e m\n", err);
    CHECK(err < 1.5e-3f, "FABRIK did not converge (%.2e)", err);

    IKEngine e2; quiet(e2);
    ChainId g = e2.add_chain("finger", Vec3(0, 1, 0), Quat::identity(),
                             {{"a", 0.04f}, {"b", 0.03f}, {"c", 0.02f}, {"d", 0.02f}},
                             SolverType::Fabrik, Vec3(0, 1, 0), 0.f, 1.f);
    const float lim = 30.f * PI / 180.f;
    for (int j = 0; j < 4; ++j) e2.set_max_bend("finger", j, lim);
    e2.reach_to(g, Vec3(-0.05f, 1.0f, 0.04f)); run(e2, 1.5f);   // behind the base: forces the limits to bind
    auto p = e2.chain(g).world_positions();
    float worst = 0; Vec3 prev_dir = Vec3(1, 0, 0);
    for (int i = 0; i < 4; ++i) {
        Vec3 d = (p[i + 1] - p[i]).normalized();
        worst = std::max(worst, std::acos(clampf(dot(d, prev_dir), -1.f, 1.f)));
        prev_dir = d;
    }
    std::printf("  30 deg cone limits: worst joint bend %.2f deg, pose finite=%d\n", worst * 180.f / PI, (int)p[4].is_finite());
    CHECK(worst < lim + 0.5f * PI / 180.f, "a joint exceeded its cone limit: %.2f deg", worst * 180.f / PI);
    CHECK(p[4].is_finite(), "NaN in constrained pose");
}

static void test_idle_and_skip() {
    std::printf("[idle wander + error-bounded solve skipping]\n");
    IKEngine e;
    ChainId id = add_arm(e); run(e, 1.f);
    e.reach_to(id, Vec3(0.30f, 1.20f, 0.20f)); run(e, 2.f);
    Vec3 c = e.end_effector_position(id); float lo = 1e9f, hi = 0.f;
    for (int i = 0; i < 600; ++i) { e.update(DT); float d = (e.end_effector_position(id) - c).length(); lo = std::min(lo, d); hi = std::max(hi, d); }
    std::printf("  idle drift over 10 s: max %.2f mm from settled pose\n", hi * 1e3f);
    CHECK(hi > 2e-4f && hi < 6e-3f, "idle wander out of range: %.2f mm", hi * 1e3f);

    // skipping must be visually lossless: compare against solving every frame
    IKEngine a, b; b.params.solve_epsilon_m = 0.f;
    ChainId ia = add_arm(a), ib = add_arm(b);
    a.reach_to(ia, Vec3(0.3f, 1.2f, 0.2f)); b.reach_to(ib, Vec3(0.3f, 1.2f, 0.2f));
    float worst = 0;
    for (int i = 0; i < 900; ++i) { a.update(DT); b.update(DT); worst = std::max(worst, (a.end_effector_position(ia) - b.end_effector_position(ib)).length()); }
    std::printf("  worst hand difference, skipping vs solving every frame: %.3f mm\n", worst * 1e3f);
    CHECK(worst < 3e-4f, "solve skipping visibly changed the pose: %.3f mm", worst * 1e3f);
}

static void test_orientation_and_reset() {
    std::printf("[wrist orientation, RESET, protocol]\n");
    IKEngine e; quiet(e);
    ChainId id = add_arm(e); Vec3 rest = e.end_effector_position(id);
    CHECK(e.execute_command("ORIENT arm 0 -1 0 0 0 1"), "ORIENT rejected");
    run(e, 1.f);
    Vec3 palm = e.end_effector_orientation("arm").rotate(Vec3(0, 0, 1));
    std::printf("  palm-forward after ORIENT: (%.3f, %.3f, %.3f), want (0,-1,0)\n", palm.x, palm.y, palm.z);
    CHECK((palm - Vec3(0, -1, 0)).length() < 1e-3f, "wrong palm direction");

    CHECK(e.execute_command("REACH arm 0.10 1.10 0.30"), "valid REACH rejected");
    run(e, 2.f);
    CHECK((e.end_effector_position(id) - Vec3(0.10f, 1.10f, 0.30f)).length() < 1e-3f, "REACH missed");
    CHECK(e.execute_command("RESET arm"), "RESET rejected"); run(e, 2.f);
    std::printf("  RESET returns hand to within %.2e m of its rest position\n", (e.end_effector_position(id) - rest).length());
    CHECK((e.end_effector_position(id) - rest).length() < 1e-3f, "RESET did not return to rest");

    const char* bad[] = {"", "DANCE arm", "REACH nope 0 0 0", "REACH arm 1 2", "REACH arm 1 2 3 4", "REACH arm 1 2 3abc",
                         "REACH arm nan 0 0", "REACH arm inf 0 0", "REACH arm x y z", "RESET", "RESET arm extra", "ORIENT arm 0 0"};
    int rejected = 0;
    for (const char* b : bad) rejected += !e.execute_command(b);
    std::printf("  malformed commands rejected: %d / %d\n", rejected, (int)(sizeof(bad) / sizeof(bad[0])));
    CHECK(rejected == (int)(sizeof(bad) / sizeof(bad[0])), "some malformed command was accepted");
    run(e, 0.5f);
    CHECK(e.end_effector_position(id).is_finite(), "NaN leaked into the pose");

    IKEngine e2;
    CHECK(add_arm(e2, "a") == 0 && add_arm(e2, "a") == kNoChain, "duplicate chain name accepted");
    CHECK(e2.add_chain("z", Vec3(), Quat::identity(), {{"a", 0.f}, {"b", 0.2f}}, SolverType::TwoBoneAnalytic, Vec3(0, 1, 0)) == kNoChain, "zero-length bone accepted");
    CHECK(e2.add_chain("y", Vec3(), Quat::identity(), {{"a", 0.3f}}, SolverType::TwoBoneAnalytic, Vec3(0, 1, 0)) == kNoChain, "1-bone two-bone chain accepted");
}

int main() {
    test_trajectory_math();
    test_reach_and_profile();
    test_retarget_continuity();
    test_torso_lean();
    test_limits_and_unreachable();
    test_fabrik();
    test_idle_and_skip();
    test_orientation_and_reset();
    std::printf("\n%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
