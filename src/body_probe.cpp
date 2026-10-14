#include <cstdio>
#include "ik_body.hpp"
using namespace ik;
int main() {
    Body b;
    for (int i = 0; i < 120; ++i) b.update(1.f / 60.f);
    std::printf("bones=%d pelvis=(%.3f %.3f %.3f) margin=%.1f cm fallen=%d\n", b.bone_count(), b.pelvis().x, b.pelvis().y, b.pelvis().z, b.margin() * 100, (int)b.fallen());
    std::puts(b.status_json().c_str());
}
