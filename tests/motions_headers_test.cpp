// motion_headers_test.cpp - do the generated motion headers (tools/export_headers.py) compile and decode into sane motion?
//
//   g++ -std=c++17 -O1 -Wall -Wextra -Iinclude/motion src/motion_headers_test.cpp -o motion_headers_test && ./motion_headers_test
//   (or: make motion-test, which also exports the headers and grades the data first)
//
// Checks per action set: clip and frame counts add up, every decoded value is finite and inside the human range, each clip starts at the origin facing +Z,
// contact flags are 0/1, the pelvis stays at a plausible height. Exit status 1 on the first violation, so it can sit in a build pipeline.
#include "nerva_motion.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace nerva::motion;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

int main() {
    std::printf("%-12s %-11s %6s %8s %6s %10s %12s\n", "action", "group", "clips", "frames", "fps", "seconds", "pelvis y");
    long total_frames_all = 0;
    for (const ActionSet& s : kActions) {
        const int total = total_frames(s);
        CHECK(s.clips > 0, "%s has no clips", s.name);
        CHECK(s.fps >= 10 && s.fps <= 60, "%s fps %d", s.name, s.fps);
        double ymin = 1e9, ymax = -1e9; std::vector<float> c;
        for (int i = 0; i < s.clips; ++i) {
            CHECK(s.frames[i] >= 10, "%s clip %d has %d frames", s.name, i, (int)s.frames[i]);
            CHECK(s.source[i] != nullptr && s.source[i][0] != 0, "%s clip %d has no source", s.name, i);
            decode_clip(s, i, c);
            CHECK((int)c.size() == s.frames[i] * kActionDim, "%s clip %d decoded size", s.name, i);
            CHECK(std::fabs(c[0]) < 0.01f && std::fabs(c[2]) < 0.01f && std::fabs(c[3]) < 0.01f, "%s clip %d does not start at the origin facing +Z (%.3f %.3f yaw %.3f)", s.name, i, c[0], c[2], c[3]);
            for (int f = 0; f < s.frames[i]; ++f) {
                const float* a = &c[(size_t)f * kActionDim];
                for (int k = 0; k < kActionDim; ++k) {
                    CHECK(std::isfinite(a[k]), "%s clip %d frame %d value %d not finite", s.name, i, f, k);
                    if (k == 3 || k == 7 || k == 12) continue;                          // yaw is unwrapped: may exceed a few turns
                    if (k == 8 || k == 13 || k == 17 || k == 21 || k == 25 || k == 29) { CHECK(a[k] == 0.0f || a[k] == 1.0f, "%s flag %d = %f", s.name, k, a[k]); continue; }
                    CHECK(std::fabs(a[k]) < 12.0f, "%s clip %d frame %d value %d = %f out of range", s.name, i, f, k, a[k]);
                }
                ymin = std::fmin(ymin, a[1]); ymax = std::fmax(ymax, a[1]);
                CHECK(a[1] > -0.15f && a[1] < 1.8f, "%s clip %d frame %d pelvis y %.3f", s.name, i, f, a[1]);   // lying clips sit a few cm under the floor estimate, hops rise to ~1.4 m
            }
        }
        std::printf("%-12s %-11s %6d %8d %6d %10.1f %5.2f..%-5.2f\n", s.name, s.group, s.clips, total, s.fps, (double)total / s.fps, ymin, ymax);
        total_frames_all += total;
        CHECK(find(s.name) == &s, "find(\"%s\") does not return the set", s.name);
    }
    CHECK(find("no_such_action") == nullptr, "find() of an unknown name must be null");
    CHECK(kProfileCount >= kActionCount, "profiles %d, sets %d", kProfileCount, kActionCount);      // every action has a profile; only the actions the CC BY sets cover have clips
    for (const ActionSet& s : kActions) CHECK(find_profile(s.name) != nullptr, "%s has clips but no profile", s.name);
    for (int p = 0; p < kProfileCount; ++p) {
        const Profile& pr = kProfiles[p]; int known = 0;
        for (int j = 0; j < kFeatureCount; ++j) {
            if (std::isfinite(pr.med[j])) { ++known; CHECK(pr.typ_lo[j] <= pr.typ_hi[j] && pr.tol_lo[j] <= pr.typ_lo[j] && pr.typ_hi[j] <= pr.tol_hi[j], "%s profile %s ranges not nested", pr.action, kFeatureNames[j]); }
        }
        CHECK(known >= kFeatureCount / 2, "%s profile has only %d measures", pr.action, known);
    }
    std::printf("\n%d action sets, %ld frames in total, %d profiles: %s\n", kActionCount, total_frames_all, kProfileCount, g_fail ? "FAILED" : "all checks passed");
    return g_fail ? 1 : 0;
}
