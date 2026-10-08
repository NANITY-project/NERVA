// Realism metrics measured through the public API only (old and new engine).
#include <cstdio>
#include <vector>
#include <algorithm>
#include "ik_engine.hpp"
using namespace ik;

static IKEngine make() {
    IKEngine e;
    e.add_chain("arm", Vec3(0.f, 1.4f, 0.f), Quat::identity(), {{"elbow", 0.30f}, {"wrist", 0.25f}},
                SolverType::TwoBoneAnalytic, Vec3(0.3f, -1.f, 0.6f), 0.10f, 1.2f);
    e.set_elbow_limit("arm", 25.f * PI / 180.f);
    return e;
}
int main() {
    const float dt = 1.f / 60.f;
    // Settle to a start pose first.
    auto settle = [&](IKEngine& e, Vec3 t) { e.reach_to("arm", t); for (int i = 0; i < 180; ++i) e.update(dt); };

    // ---- M1: path straightness (max deviation from start->target line / distance)
    {
        IKEngine e = make();
        Vec3 A(0.30f, 1.20f, 0.20f), B(-0.25f, 1.55f, 0.25f);
        settle(e, A);
        Vec3 p0 = e.end_effector_position("arm");
        e.reach_to("arm", B);
        Vec3 dir = (B - p0); float D = dir.length(); dir = dir * (1.f / D);
        float maxdev = 0;
        for (int i = 0; i < 120; ++i) {
            e.update(dt);
            Vec3 p = e.end_effector_position("arm") - p0;
            Vec3 perp = p - dir * dot(p, dir);
            maxdev = std::max(maxdev, perp.length());
        }
        std::printf("M1 path deviation from straight line : %.1f%% of reach distance (D=%.3f m)\n", 100.f * maxdev / D, D);
    }
    // ---- M2: hand teleport across the reach_to() call on a far target (torso lean pop)
    {
        IKEngine e = make();
        settle(e, Vec3(0.10f, 1.10f, 0.05f));
        Vec3 before = e.end_effector_position("arm");
        e.reach_to("arm", Vec3(0.55f, 1.40f, 0.15f));
        Vec3 after = e.end_effector_position("arm");
        std::printf("M2 hand jump inside reach_to() call    : %.1f mm\n", (after - before).length() * 1000.f);
    }
    // ---- M3: velocity discontinuity when retargeting mid-motion
    {
        IKEngine e = make();
        settle(e, Vec3(0.30f, 1.20f, 0.20f));
        e.reach_to("arm", Vec3(-0.30f, 1.45f, 0.20f));
        Vec3 prev = e.end_effector_position("arm"), pv;
        std::vector<Vec3> vel;
        int retarget_frame = 0;
        float peak = 0;
        for (int f = 0; f < 60; ++f) {
            if (f == 22) { e.reach_to("arm", Vec3(0.10f, 1.05f, 0.40f)); retarget_frame = (int)vel.size(); }
            e.update(dt);
            Vec3 cur = e.end_effector_position("arm");
            vel.push_back((cur - prev) * (1.f / dt));
            peak = std::max(peak, vel.back().length());
            prev = cur;
        }
        float jump = 0;
        for (int k = retarget_frame; k < retarget_frame + 3 && k < (int)vel.size(); ++k)
            jump = std::max(jump, (vel[k] - vel[k - 1]).length());
        std::printf("M3 velocity jump at mid-motion retarget: %.3f m/s  (%.0f%% of peak %.2f m/s)\n", jump, 100.f * jump / peak, peak);
    }
    return 0;
}
