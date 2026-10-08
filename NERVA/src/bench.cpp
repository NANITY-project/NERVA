// Works against both the old and new engine API (same calls).
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "ik_engine.hpp"
using namespace ik;
using clk = std::chrono::steady_clock;

static uint32_t rng_state = 12345u;
static float rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
                     return (rng_state & 0xFFFFFF) / float(0x1000000); }

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    const int frames = 600;           // 10 s @ 60 Hz
    const float dt = 1.f / 60.f;
    std::vector<std::string> names;
    IKEngine eng;
    for (int i = 0; i < N; ++i) {
        names.push_back("c" + std::to_string(i));
        Vec3 root(rnd(), 1.4f, rnd());
        if (i % 2 == 0)
            eng.add_chain(names.back(), root, Quat::identity(), {{"e", 0.30f}, {"w", 0.25f}},
                          SolverType::TwoBoneAnalytic, Vec3(0.3f, -1.f, 0.6f), 0.1f, 1.2f);
        else
            eng.add_chain(names.back(), root, Quat::identity(),
                          {{"a", 0.04f}, {"b", 0.03f}, {"c", 0.02f}, {"d", 0.02f}},
                          SolverType::Fabrik, Vec3(0, 1, 0), 0.f, 1.0f);
    }
    bool idle_mode = argc > 2 && std::string(argv[2]) == "idle";
    if (idle_mode) {   // one reach each, let everything settle, then time pure idle frames
        for (int i = 0; i < N; ++i) eng.reach_to(names[i], Vec3(0.3f + rnd() * .1f, 1.3f, 0.2f));
        for (int f = 0; f < 300; ++f) eng.update(dt);
    }
    double t_reach = 0, t_update = 0; long reaches = 0;
    for (int f = 0; f < frames; ++f) {
        auto a = clk::now();
        if (!idle_mode) for (int i = 0; i < N; ++i) if ((f + i) % 30 == 0) {   // each chain retargets every 0.5 s
            const Chain* dummy = nullptr; (void)dummy;
            float r = (i % 2 == 0) ? 0.45f : 0.07f;
            Vec3 root_guess(0, 1.4f, 0);
            (void)root_guess;
            eng.reach_to(names[i], Vec3(rnd() * 1.f + (rnd() - .5f) * 2 * r, 1.4f + (rnd() - .5f) * 2 * r, (rnd() - .5f) * 2 * r + rnd()));
            ++reaches;
        }
        auto b = clk::now();
        eng.update(dt);
        auto c = clk::now();
        t_reach  += std::chrono::duration<double>(b - a).count();
        t_update += std::chrono::duration<double>(c - b).count();
    }
    double total = t_reach + t_update;
    std::printf("N=%d chains, %d frames\n", N, frames);
    if (reaches) std::printf("  reach_to : %8.1f ns/call   (%ld calls)\n", t_reach / reaches * 1e9, reaches);
    std::printf("  update   : %8.1f ns/chain-frame\n", t_update / (double(N) * frames) * 1e9);
    std::printf("  total    : %8.1f ns/chain-frame  => %.2f ms per 60Hz frame for %d chains\n",
                total / (double(N) * frames) * 1e9, total / frames * 1e3, N);
    return 0;
}
