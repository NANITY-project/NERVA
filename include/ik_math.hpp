// ik_math.hpp -- minimal, dependency-free 3D math for the IK engine.
// No glm, no Eigen -- just what IK actually needs: Vec3 and Quat.
#pragma once
#include <cmath>
#include <algorithm>
#if (defined(__SSE__) || defined(_M_X64) || defined(_M_AMD64)) && !defined(IK_NO_FAST_RSQRT)
#include <xmmintrin.h>
#endif

namespace ik {

constexpr float PI  = 3.14159265358979323846f;
constexpr float EPS = 1e-6f;

// 1/sqrt(x). sqrtss + divss are ~25 cycles of serial latency; the SSE
// reciprocal estimate plus one Newton-Raphson step is ~4x faster and accurate
// to ~2e-7 relative (float epsilon territory). Define IK_NO_FAST_RSQRT for
// the exact version (other architectures fall back automatically).
#if (defined(__SSE__) || defined(_M_X64) || defined(_M_AMD64)) && !defined(IK_NO_FAST_RSQRT)
inline float inv_sqrt(float x) {
    float y = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
    return y * (1.5f - 0.5f * x * y * y);
}
#else
inline float inv_sqrt(float x) { return 1.f / std::sqrt(x); }
#endif

// sqrt via x * rsqrt(x); returns 0 for x <= 0 instead of NaN.
inline float fast_sqrt(float x) { return x * inv_sqrt(std::max(x, 1e-30f)); }

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    constexpr bool operator==(const Vec3& o) const { return x == o.x && y == o.y && z == o.z; }
    constexpr bool operator!=(const Vec3& o) const { return !(*this == o); }

    constexpr float length_sq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_sq()); }
    bool is_finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

    // One sqrt + one divide (multiply by reciprocal) instead of three divides.
    Vec3 normalized() const {
        float l2 = length_sq();
        if (l2 < EPS * EPS) return {0.f, 0.f, 0.f};
        float inv = inv_sqrt(l2);
        return {x * inv, y * inv, z * inv};
    }
};

constexpr Vec3 operator*(float s, const Vec3& v) { return v * s; }

constexpr float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

constexpr Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }

constexpr float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }


// Unit quaternion, (w, x, y, z), Hamilton convention.
struct Quat {
    float w = 1.f, x = 0.f, y = 0.f, z = 0.f;

    constexpr Quat() = default;
    constexpr Quat(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}

    static constexpr Quat identity() { return Quat(1.f, 0.f, 0.f, 0.f); }

    static Quat from_axis_angle(Vec3 axis, float angle_rad) {
        axis = axis.normalized();
        float half = angle_rad * 0.5f;
        float s = std::sin(half);
        return Quat(std::cos(half), axis.x * s, axis.y * s, axis.z * s);
    }

    // Shortest-arc rotation that takes unit vector `from` to unit vector `to`.
    static Quat rotation_between(Vec3 from, Vec3 to) {
        from = from.normalized();
        to = to.normalized();
        float d = clampf(dot(from, to), -1.f, 1.f);
        if (d > 1.f - EPS) return Quat::identity();
        if (d < -1.f + EPS) {
            Vec3 axis = cross(Vec3(1, 0, 0), from);
            if (axis.length_sq() < EPS) axis = cross(Vec3(0, 1, 0), from);
            return from_axis_angle(axis.normalized(), PI);
        }
        Vec3 axis = cross(from, to);
        float s = std::sqrt((1.f + d) * 2.f);
        float inv_s = 1.f / s;
        return Quat(s * 0.5f, axis.x * inv_s, axis.y * inv_s, axis.z * inv_s);
    }

    constexpr Quat operator*(const Quat& o) const {
        return Quat(
            w * o.w - x * o.x - y * o.y - z * o.z,
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w);
    }

    Quat normalized() const {
        float l2 = w * w + x * x + y * y + z * z;
        if (l2 < EPS * EPS) return Quat::identity();
        float inv = inv_sqrt(l2);
        return Quat(w * inv, x * inv, y * inv, z * inv);
    }

    constexpr Quat conjugate() const { return Quat(w, -x, -y, -z); }

    constexpr Vec3 rotate(const Vec3& v) const {
        Vec3 qv(x, y, z);
        Vec3 t = cross(qv, v) * 2.f;
        return v + t * w + cross(qv, t);
    }

    // Same as conjugate().rotate(v) without building the conjugate.
    constexpr Vec3 rotate_inverse(const Vec3& v) const {
        Vec3 qv(-x, -y, -z);
        Vec3 t = cross(qv, v) * 2.f;
        return v + t * w + cross(qv, t);
    }

    // rotate(Vec3(1,0,0)) -- the bone direction -- from the first matrix
    // column, no cross products. This is the hot call in forward kinematics.
    constexpr Vec3 x_axis() const {
        return {1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y)};
    }

    constexpr float dot4(const Quat& o) const { return w * o.w + x * o.x + y * o.y + z * o.z; }

    // Rotation angle (radians) between two orientations, shortest way round.
    float angle_to(const Quat& o) const {
        return 2.f * std::acos(clampf(std::fabs(dot4(o)), 0.f, 1.f));
    }

    // Normalized lerp: 2-3x cheaper than slerp, and for the small per-frame
    // angles this engine deals in it is indistinguishable.
    static Quat nlerp(const Quat& a, Quat b, float t) {
        if (a.dot4(b) < 0.f) b = Quat(-b.w, -b.x, -b.y, -b.z);
        return Quat(a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t,
                    a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t).normalized();
    }

    // Spherical interpolation. Falls back to nlerp for near-parallel inputs,
    // and uses sqrt/atan2 instead of acos + sin(acos()).
    static Quat slerp(const Quat& a, Quat b, float t) {
        float d = a.dot4(b);
        if (d < 0.f) { b = Quat(-b.w, -b.x, -b.y, -b.z); d = -d; }
        if (d > 0.9995f) return nlerp(a, b, t);
        float sin0 = std::sqrt(std::max(0.f, 1.f - d * d));
        float theta0 = std::atan2(sin0, d);
        float inv = 1.f / sin0;
        float s0 = std::sin((1.f - t) * theta0) * inv;
        float s1 = std::sin(t * theta0) * inv;
        return Quat(a.w * s0 + b.w * s1, a.x * s0 + b.x * s1,
                    a.y * s0 + b.y * s1, a.z * s0 + b.z * s1);
    }

    // Full orientation from a forward direction + an up hint (pins both the
    // forward axis and the roll around it). Columns are (right, up, forward).
    static Quat look_rotation(Vec3 forward, Vec3 up) {
        forward = forward.normalized();
        if (forward.length_sq() < EPS) forward = Vec3(1.f, 0.f, 0.f);
        Vec3 right = cross(up, forward);
        if (right.length_sq() < EPS) {
            Vec3 fallback_up = (std::abs(forward.y) < 0.99f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            right = cross(fallback_up, forward);
        }
        right = right.normalized();
        Vec3 true_up = cross(forward, right);

        float m00 = right.x, m01 = true_up.x, m02 = forward.x;
        float m10 = right.y, m11 = true_up.y, m12 = forward.y;
        float m20 = right.z, m21 = true_up.z, m22 = forward.z;

        float trace = m00 + m11 + m22;
        Quat q;
        if (trace > 0.f) {
            float s = 0.5f / std::sqrt(trace + 1.f);
            q.w = 0.25f / s;
            q.x = (m21 - m12) * s;
            q.y = (m02 - m20) * s;
            q.z = (m10 - m01) * s;
        } else if (m00 > m11 && m00 > m22) {
            float s = 2.f * std::sqrt(1.f + m00 - m11 - m22);
            q.w = (m21 - m12) / s;
            q.x = 0.25f * s;
            q.y = (m01 + m10) / s;
            q.z = (m02 + m20) / s;
        } else if (m11 > m22) {
            float s = 2.f * std::sqrt(1.f + m11 - m00 - m22);
            q.w = (m02 - m20) / s;
            q.x = (m01 + m10) / s;
            q.y = 0.25f * s;
            q.z = (m12 + m21) / s;
        } else {
            float s = 2.f * std::sqrt(1.f + m22 - m00 - m11);
            q.w = (m10 - m01) / s;
            q.x = (m02 + m20) / s;
            q.y = (m12 + m21) / s;
            q.z = 0.25f * s;
        }
        return q.normalized();
    }
};

// Critically damped spring step (exact, unconditionally stable for any dt --
// unlike semi-implicit Euler springs which blow up when dt*omega grows).
// Moves `cur` toward `target`; `vel` is carried between calls.
inline void smooth_damp(Vec3& cur, const Vec3& target, Vec3& vel, float omega, float dt) {
    float x = omega * dt;
    float e = 1.f / (1.f + x + 0.48f * x * x + 0.235f * x * x * x);
    Vec3 change = cur - target;
    Vec3 temp = (vel + change * omega) * dt;
    vel = (vel - temp * omega) * e;
    cur = target + (change + temp) * e;
}

// Smooth, non-repeating 1-D value noise in [-1, 1]: hashed lattice values,
// quintic fade between them. Used for idle motion so there is no visible
// loop period. The lattice values are CACHED: input advances slowly, so the
// hashes are recomputed only when the integer cell changes (every ~1-2 s for
// idle motion) instead of every call. x must be >= 0.
struct ValueNoise1 {
    int cell = -1;
    float a = 0.f, b = 0.f;

    static float hash(int n) {
        unsigned h = (unsigned)n * 0x9E3779B1u;
        h ^= h >> 15; h *= 0x85EBCA77u; h ^= h >> 13;
        return (float)(h & 0xFFFFu) * (2.f / 65535.f) - 1.f;
    }
    float operator()(float x) {
        int i = (int)x;                       // x >= 0, so truncation == floor
        if (i != cell) { a = hash(i); b = hash(i + 1); cell = i; }
        float f = x - (float)i;
        float u = f * f * f * (f * (f * 6.f - 15.f) + 10.f);
        return a + (b - a) * u;
    }
};

} // namespace ik
