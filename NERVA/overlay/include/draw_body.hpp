// draw_body.hpp -- draws an ik::Body (side view) onto a Canvas. "Blacked-out entity": near-black fill, accent outline.
#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "ik_body.hpp"
#include "raster.hpp"
#include "scene.hpp"
#include "hypr.hpp"

namespace nerva {

struct Style {
    Color fill{0.035f, 0.035f, 0.045f, 0.96f};
    Color accent{0.55f, 0.07f, 0.09f, 1.f};            // dark red (NANITY palette)
    Color debug{0.95f, 0.75f, 0.2f, 0.9f};
    float ring_px = 1.6f;
    bool debug_overlay = false;
    float opacity = 1.f;                                // 0..1, multiplies every colour (used to ghost the body near the pointer)
};

// true when point (px,py) is within `margin` px of the rectangle
inline bool near_rect(const Rect& r, float px, float py, float margin) {
    return !r.empty() && px >= r.x0 - margin && px <= r.x1 + margin && py >= r.y0 - margin && py <= r.y1 + margin;
}

// radius in metres at body scale 1
inline float bone_radius_m(const std::string& n) {
    auto has = [&](const char* p) { return n.rfind(p, 0) == 0; };
    if (has("pelvis")) return 0.090f;
    if (has("spine3")) return 0.085f;
    if (has("spine")) return 0.078f;
    if (has("neck")) return 0.034f;
    if (has("head")) return 0.105f;
    if (has("clavicle")) return 0.034f;
    if (has("upper_arm")) return 0.040f;
    if (has("forearm")) return 0.033f;
    if (has("hand")) return 0.030f;
    if (has("thigh")) return 0.068f;
    if (has("shin")) return 0.050f;
    if (has("foot")) return 0.036f;
    if (has("toe")) return 0.024f;
    return 0.009f;                                      // fingers
}
inline int bone_group(const std::string& n) {           // 0 = far side (left), 1 = core, 2 = near side (right); the viewer looks along +X
    if (n.size() > 2 && n[n.size() - 2] == '_' && n.back() == 'L') return 0;
    if (n.size() > 2 && n[n.size() - 2] == '_' && n.back() == 'R') return 2;
    return 1;
}

struct DrawnBone { float ax, ay, bx, by, r; int group; bool circle; };

// Projects every bone to screen space. Foot and toe capsules are lifted by their radius so the sole touches the floor line.
inline std::vector<DrawnBone> project_bones(const ik::Body& body, const View& v) {
    std::vector<DrawnBone> out;
    out.reserve(body.bones().size());
    for (const ik::BoneOut& b : body.bones()) {
        std::string n = b.name;
        DrawnBone d{};
        d.r = std::max(1.5f, bone_radius_m(n) * body.params.scale * v.ppm);
        d.ax = v.sx(b.head.z); d.ay = v.sy(b.head.y); d.bx = v.sx(b.tail.z); d.by = v.sy(b.tail.y);
        d.group = bone_group(n); d.circle = n.rfind("head", 0) == 0;
        if (d.circle) { d.ax = d.bx = 0.5f * (d.ax + d.bx); d.ay = d.by = 0.5f * (d.ay + d.by); }
        if (n.rfind("foot", 0) == 0 || n.rfind("toe", 0) == 0) { d.ay -= d.r; d.by -= d.r; }
        out.push_back(d);
    }
    return out;
}

inline Rect draw_body(Canvas& cv, const ik::Body& body, const View& v, const Style& style, const std::vector<SurfaceDef>* surfaces = nullptr) {
    Rect bb;
    const Style& st = style;
    std::vector<DrawnBone> bones = project_bones(body, v);
    for (int g = 0; g < 3; ++g) {
        for (const DrawnBone& d : bones) if (d.group == g) bb = unite(bb, cv.capsule(d.ax, d.ay, d.bx, d.by, d.r + st.ring_px, st.accent));
        for (const DrawnBone& d : bones) if (d.group == g) bb = unite(bb, cv.capsule(d.ax, d.ay, d.bx, d.by, d.r, st.fill));
    }
    if (st.debug_overlay) {
        float fy = v.sy(0.f);
        bb = unite(bb, cv.capsule(0.f, fy, (float)v.W, fy, 0.8f, st.debug));
        if (surfaces) for (const SurfaceDef& s : *surfaces) { float y = v.sy(s.y); bb = unite(bb, cv.capsule(v.sx(s.z0), y, v.sx(s.z1), y, 1.2f, st.debug)); }
        ik::Vec3 c = body.com(); bb = unite(bb, cv.capsule(v.sx(c.z), v.sy(c.y), v.sx(c.z), v.sy(c.y), 3.f, st.debug));
        for (int s = 0; s < 2; ++s) for (const ik::Vec3& p : body.foot_contact_points(s)) bb = unite(bb, cv.capsule(v.sx(p.z), v.sy(p.y), v.sx(p.z), v.sy(p.y), 1.6f, st.debug));
    }
    if (st.opacity < 0.999f) {                          // scale the finished picture, so overlapping limbs do not add up to opaque
        float k = st.opacity < 0.f ? 0.f : st.opacity;
        Rect r = clip(bb, cv.w, cv.h);
        for (int y = r.y0; y < r.y1; ++y)
            for (int x = r.x0; x < r.x1; ++x) {
                uint32_t& p = cv.px[(size_t)y * cv.stride + x];
                if (!p) continue;
                auto sc = [&](uint32_t c) { return (uint32_t)(c * k + 0.5f); };
                p = (sc(p >> 24) << 24) | (sc((p >> 16) & 255) << 16) | (sc((p >> 8) & 255) << 8) | sc(p & 255);
            }
    }
    return bb;
}

// A thin outline around the window the human has granted to the AI pointer (dimmer while that window is not focused).
inline Rect draw_grant_frame(Canvas& cv, const Window& w, bool focused, const Style& st) {
    Color c = st.accent; c.r = std::min(1.f, c.r * 1.5f + 0.12f); c.g = std::min(1.f, c.g * 1.5f + 0.05f); c.b = std::min(1.f, c.b * 1.5f + 0.05f); c.a = focused ? 0.95f : 0.4f;
    float x0 = w.x + 1.5f, y0 = w.y + 1.5f, x1 = w.x + w.w - 1.5f, y1 = w.y + w.h - 1.5f; Rect bb;
    bb = unite(bb, cv.capsule(x0, y0, x1, y0, 1.5f, c)); bb = unite(bb, cv.capsule(x0, y1, x1, y1, 1.5f, c));
    bb = unite(bb, cv.capsule(x0, y0, x0, y1, 1.5f, c)); bb = unite(bb, cv.capsule(x1, y0, x1, y1, 1.5f, c));
    return bb;
}

}  // namespace nerva
