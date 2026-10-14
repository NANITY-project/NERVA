// raster.hpp -- tiny CPU rasteriser for the overlay: anti-aliased capsules into a premultiplied ARGB8888 buffer.
// No dependencies; testable without a display.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace nerva {

struct Rect {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;                          // [x0,x1) x [y0,y1)
    bool empty() const { return x1 <= x0 || y1 <= y0; }
};
inline Rect unite(const Rect& a, const Rect& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return {std::min(a.x0, b.x0), std::min(a.y0, b.y0), std::max(a.x1, b.x1), std::max(a.y1, b.y1)};
}
inline Rect clip(const Rect& a, int w, int h) {
    Rect r{std::max(a.x0, 0), std::max(a.y0, 0), std::min(a.x1, w), std::min(a.y1, h)};
    if (r.empty()) return Rect{};
    return r;
}

struct Color { float r, g, b, a; };                               // straight (non-premultiplied) 0..1

struct Canvas {
    uint32_t* px = nullptr;                                       // ARGB8888, premultiplied, little-endian word
    int w = 0, h = 0, stride = 0;                                 // stride in pixels

    void clear(Rect r) {
        r = clip(r, w, h);
        for (int y = r.y0; y < r.y1; ++y) std::memset(px + (size_t)y * stride + r.x0, 0, sizeof(uint32_t) * (size_t)(r.x1 - r.x0));
    }
    // composite `c` with coverage `cov` over pixel (x,y)
    inline void blend(int x, int y, const Color& c, float cov) {
        float a = c.a * cov;
        if (a <= 0.f) return;
        uint32_t d = px[(size_t)y * stride + x];
        float inv = 1.f - a;
        float da = ((d >> 24) & 255) * (1.f / 255.f), dr = ((d >> 16) & 255) * (1.f / 255.f), dg = ((d >> 8) & 255) * (1.f / 255.f), db = (d & 255) * (1.f / 255.f);
        float oa = a + da * inv, orr = c.r * a + dr * inv, og = c.g * a + dg * inv, ob = c.b * a + db * inv;
        auto q = [](float v) { return (uint32_t)(std::min(std::max(v, 0.f), 1.f) * 255.f + 0.5f); };
        px[(size_t)y * stride + x] = (q(oa) << 24) | (q(orr) << 16) | (q(og) << 8) | q(ob);
    }
    // Anti-aliased capsule (segment a-b, radius r). Returns the pixel rect that was touched (clipped to the canvas).
    Rect capsule(float ax, float ay, float bx, float by, float r, const Color& c) {
        if (!(std::isfinite(ax) && std::isfinite(ay) && std::isfinite(bx) && std::isfinite(by) && std::isfinite(r)) || r <= 0.f) return Rect{};
        float lim = 1e6f;                                          // keep int conversion safe
        if (std::fabs(ax) > lim || std::fabs(ay) > lim || std::fabs(bx) > lim || std::fabs(by) > lim || r > lim) return Rect{};
        Rect bb = clip({(int)std::floor(std::min(ax, bx) - r - 1.f), (int)std::floor(std::min(ay, by) - r - 1.f),
                        (int)std::ceil(std::max(ax, bx) + r + 1.f), (int)std::ceil(std::max(ay, by) + r + 1.f)}, w, h);
        if (bb.empty()) return bb;
        float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, il2 = l2 > 1e-12f ? 1.f / l2 : 0.f;
        for (int y = bb.y0; y < bb.y1; ++y) {
            float py = y + 0.5f;
            for (int x = bb.x0; x < bb.x1; ++x) {
                float pxx = x + 0.5f;
                float t = ((pxx - ax) * dx + (py - ay) * dy) * il2; t = t < 0.f ? 0.f : t > 1.f ? 1.f : t;
                float ex = pxx - (ax + dx * t), ey = py - (ay + dy * t);
                float d = std::sqrt(ex * ex + ey * ey) - r;
                float cov = 0.5f - d;
                if (cov <= 0.f) continue;
                blend(x, y, c, cov > 1.f ? 1.f : cov);
            }
        }
        return bb;
    }
    // Debug/test output: composite over a solid background and write a binary PPM.
    bool write_ppm(const char* path, uint8_t bgr = 200, uint8_t bgg = 200, uint8_t bgb = 200) const {
        FILE* f = std::fopen(path, "wb");
        if (!f) return false;
        std::fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint32_t p = px[(size_t)y * stride + x];
                unsigned a = (p >> 24) & 255, r = (p >> 16) & 255, g = (p >> 8) & 255, b = p & 255;
                unsigned o[3] = {r + bgr * (255 - a) / 255, g + bgg * (255 - a) / 255, b + bgb * (255 - a) / 255};
                for (unsigned v : o) std::fputc((int)std::min(v, 255u), f);
            }
        std::fclose(f);
        return true;
    }
};

}  // namespace nerva
