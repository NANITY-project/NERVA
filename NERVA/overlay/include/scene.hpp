// scene.hpp -- screen pixels <-> body world, and window tops -> body surfaces.
//
// Side view. World: Y up, metres; the body faces +Z at yaw 0 and its LEFT is +X.
//   screen x = W/2 + z*ppm            (screen-right is +Z: a body at yaw 0 faces right)
//   screen y = H - floor_off - y*ppm  (world y = 0 is the floor line at the bottom of the output)
// The viewer therefore looks along +X: the body's RIGHT side is nearest, its LEFT side farthest.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "hypr.hpp"

namespace nerva {

struct View {
    int W = 1920, H = 1080;
    float ppm = 114.f;                                  // pixels per metre (default: 1.75 m body = ~200 px)
    float floor_off = 0.f;                              // px between the output bottom and the floor line
    float top_off = 0.f;                                // px added to each window top (compositor borders)
    float sx(float z) const { return W * 0.5f + z * ppm; }
    float sy(float y) const { return H - floor_off - y * ppm; }
    float z_of(float px) const { return (px - W * 0.5f) / ppm; }
    float y_of(float py) const { return (H - floor_off - py) / ppm; }
};

struct SurfaceDef {
    float z0, z1, y;                                    // world metres (depth range is fixed, see kDepth)
    std::string cls;                                    // window class, for the model's perception
    std::string address;
    int src = -1;                                       // index into the window list it came from
    bool operator==(const SurfaceDef& o) const { return z0 == o.z0 && z1 == o.z1 && y == o.y && address == o.address; }
};
constexpr float kDepth = 1.0f;                          // surfaces are slabs x in [-kDepth, +kDepth]

// Window tops that a character can stand on. Output-relative window rects in, surfaces out.
inline std::vector<SurfaceDef> surfaces_from_windows(const View& v, const std::vector<Window>& wins, float max_height_m = 1e9f) {
    std::vector<SurfaceDef> out;
    for (size_t i = 0; i < wins.size(); ++i) {
        const Window& w = wins[i];
        float top = (float)w.y + v.top_off;
        float y = v.y_of(top);
        float x0 = std::max(0.f, (float)w.x), x1 = std::min((float)v.W, (float)(w.x + w.w));
        if (x1 - x0 < 24.f) continue;                   // too narrow to stand on
        if (!(y > 0.02f) || y > max_height_m) continue; // at/below the floor, or off the top of the screen
        out.push_back({v.z_of(x0), v.z_of(x1), y, w.cls, w.address, (int)i});
    }
    return out;
}

inline std::string json_escape(const std::string& s) {
    std::string o; char b[8];
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t";
        else if (c < 0x20) { std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

}  // namespace nerva
