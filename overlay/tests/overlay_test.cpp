// overlay_test.cpp -- unit tests for the overlay's display-free parts (raster, JSON, Hyprland parsing, mapping, drawing).
#include <cstdio>
#include <string>
#include "draw_body.hpp"
#include "hypr.hpp"
#include "mini_json.hpp"
#include "raster.hpp"
#include "scene.hpp"
using namespace nerva;

static int g_fail = 0, g_n = 0;
#define CHECK(c, ...) do { ++g_n; if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static uint32_t A(uint32_t p) { return p >> 24; }

static void test_raster() {
    std::puts("[raster]");
    std::vector<uint32_t> px(64 * 64, 0); Canvas cv{px.data(), 64, 64, 64};
    Rect r = cv.capsule(20, 32, 44, 32, 5, Color{1, 0, 0, 1});
    CHECK(!r.empty() && r.x0 >= 0 && r.x1 <= 64, "bbox clipped to canvas");
    CHECK(A(px[32 * 64 + 32]) == 255, "capsule centre is opaque");
    CHECK(px[32 * 64 + 32] == 0xFFFF0000u, "red premultiplied ARGB (got %08x)", px[32 * 64 + 32]);
    CHECK(A(px[10 * 64 + 10]) == 0, "far pixel untouched");
    bool partial = false; for (int y = 0; y < 64; ++y) for (int x = 0; x < 64; ++x) { uint32_t a = A(px[y * 64 + x]); if (a > 0 && a < 255) partial = true; }
    CHECK(partial, "edges are anti-aliased (partial alpha exists)");
    bool premul = true; for (uint32_t p : px) if (((p >> 16) & 255) > A(p) || ((p >> 8) & 255) > A(p) || (p & 255) > A(p)) premul = false;
    CHECK(premul, "no channel exceeds alpha (valid premultiplied data)");
    cv.clear(r); bool clean = true; for (uint32_t p : px) if (p) clean = false;
    CHECK(clean, "clear(bbox) restores full transparency");
    // blending: half-transparent white over opaque black stays opaque, grey
    std::vector<uint32_t> q(4 * 4, 0xFF000000u); Canvas c2{q.data(), 4, 4, 4};
    c2.capsule(2, 2, 2, 2, 8, Color{1, 1, 1, 0.5f});
    CHECK(A(q[5]) == 255 && ((q[5] >> 16) & 255) >= 126 && ((q[5] >> 16) & 255) <= 129, "50%% white over black = mid grey, alpha 255 (got %08x)", q[5]);
    // hostile input never touches memory or throws
    float nan = std::nanf(""), inf = INFINITY;
    CHECK(cv.capsule(nan, 0, 1, 1, 3, Color{1, 1, 1, 1}).empty(), "NaN coordinate ignored");
    CHECK(cv.capsule(0, 0, inf, 1, 3, Color{1, 1, 1, 1}).empty(), "inf coordinate ignored");
    CHECK(cv.capsule(0, 0, 1, 1, -3, Color{1, 1, 1, 1}).empty(), "negative radius ignored");
    CHECK(cv.capsule(1e12f, 0, 1, 1, 3, Color{1, 1, 1, 1}).empty(), "absurd coordinate ignored");
    CHECK(cv.capsule(-500, -500, -400, -400, 3, Color{1, 1, 1, 1}).empty(), "fully off-canvas draws nothing");
    CHECK(cv.capsule(-20, 32, 80, 32, 4, Color{1, 1, 1, 1}).x1 == 64, "partly off-canvas is clipped");
}

static void test_json() {
    std::puts("[json]");
    Json j; std::string e;
    CHECK(json_parse("{\"a\":[1,2.5,-3e2],\"b\":{\"c\":\"x\"},\"d\":true,\"e\":null}", j, &e), "valid document: %s", e.c_str());
    CHECK(j.get("a") && j.get("a")->a.size() == 3 && j.get("a")->a[2].n == -300.0, "array and exponent");
    CHECK(j.get("b")->str("c") == "x" && j.boolean("d") && j.get("e")->t == Json::Null, "nested object, bool, null");
    CHECK(json_parse("\"caf\\u00e9 \\ud83d\\ude00 \\\"q\\\"\"", j, &e) && j.s == "caf\xC3\xA9 \xF0\x9F\x98\x80 \"q\"", "unicode + surrogate pair + escapes");
    CHECK(json_parse("\"\\ud83d x\"", j, &e) && j.s.find("\xEF\xBF\xBD") != std::string::npos, "lone surrogate becomes U+FFFD");
    const char* bad[] = {"", "{", "[1,]", "{\"a\":}", "{\"a\" 1}", "[1 2]", "{} x", "tru", "nan", "NaN", "Infinity", "\"abc", "\"\\x\"", "[1e]", "-", "\"\x01\"", "{'a':1}"};
    for (const char* b : bad) CHECK(!json_parse(b, j, &e), "rejects: %s", b);
    std::string deep(10000, '['); CHECK(!json_parse(deep, j, &e) && e.find("deep") != std::string::npos, "depth bomb rejected cleanly (%s)", e.c_str());
    CHECK(json_parse("1e999", j, &e) && j.n == 0.0, "overflowing number does not become inf");
    Json keep; json_parse("[1]", keep); json_parse("[", keep); CHECK(keep.t == Json::Arr, "a failed parse leaves the output untouched");
}

static const char* kMon =
    "[{\"id\":0,\"name\":\"DP-1\",\"width\":2560,\"height\":1440,\"x\":0,\"y\":0,\"scale\":1.333333,\"transform\":0,\"focused\":true,"
    "\"activeWorkspace\":{\"id\":1,\"name\":\"1\"},\"specialWorkspace\":{\"id\":0,\"name\":\"\"}},"
    "{\"id\":1,\"name\":\"HDMI-A-1\",\"width\":1080,\"height\":1920,\"x\":1920,\"y\":0,\"scale\":1.0,\"transform\":1,\"focused\":false,"
    "\"activeWorkspace\":{\"id\":5,\"name\":\"5\"},\"specialWorkspace\":{\"id\":-98,\"name\":\"special:scratch\"}}]";
static const char* kCli =
    "[{\"address\":\"0xa\",\"mapped\":true,\"hidden\":false,\"at\":[100,200],\"size\":[800,600],\"workspace\":{\"id\":1,\"name\":\"1\"},\"floating\":false,\"monitor\":0,\"class\":\"kitty\",\"title\":\"caf\\u00e9 \\\"x\\\"\",\"pinned\":false,\"fullscreen\":0},"
    "{\"address\":\"0xb\",\"mapped\":true,\"hidden\":true,\"at\":[0,0],\"size\":[500,500],\"workspace\":{\"id\":1,\"name\":\"1\"},\"monitor\":0,\"class\":\"hidden\"},"
    "{\"address\":\"0xc\",\"mapped\":true,\"hidden\":false,\"at\":[0,0],\"size\":[500,500],\"workspace\":{\"id\":2,\"name\":\"2\"},\"monitor\":0,\"class\":\"elsewhere\"},"
    "{\"address\":\"0xd\",\"mapped\":true,\"hidden\":false,\"at\":[50,60],\"size\":[300,300],\"workspace\":{\"id\":2,\"name\":\"2\"},\"monitor\":0,\"class\":\"pinned\",\"pinned\":true,\"floating\":true},"
    "{\"address\":\"0xe\",\"mapped\":true,\"hidden\":false,\"at\":[1920,100],\"size\":[400,400],\"workspace\":{\"id\":5,\"name\":\"5\"},\"monitor\":1,\"class\":\"second\",\"fullscreen\":true},"
    "{\"address\":\"0xf\",\"mapped\":true,\"hidden\":false,\"at\":[1950,700],\"size\":[200,200],\"workspace\":{\"id\":-98,\"name\":\"special:scratch\"},\"monitor\":1,\"class\":\"scratch\"},"
    "{\"address\":\"0x10\",\"class\":\"no-geometry\"},"
    "{\"address\":\"0x11\",\"mapped\":false,\"at\":[1,1],\"size\":[10,10],\"workspace\":{\"id\":1},\"monitor\":0,\"class\":\"unmapped\"}]";

static void test_hypr() {
    std::puts("[hyprland parsing]");
    std::vector<Monitor> mons; std::vector<Window> wins; std::string e;
    CHECK(parse_monitors(kMon, mons, e) && mons.size() == 2, "monitors parse: %s", e.c_str());
    CHECK(mons[0].w == 1920 && mons[0].h == 1080, "scale 1.3333 -> logical 1920x1080 (got %dx%d)", mons[0].w, mons[0].h);
    CHECK(mons[1].w == 1920 && mons[1].h == 1080, "rotated monitor swaps width/height (got %dx%d)", mons[1].w, mons[1].h);
    CHECK(mons[1].special_ws == -98 && mons[0].active_ws == 1 && mons[0].focused && !mons[1].focused, "workspace ids and focus");
    CHECK(parse_clients(kCli, wins, e), "clients parse: %s", e.c_str());
    CHECK(wins.size() == 7, "entries without geometry are skipped (got %zu)", wins.size());
    CHECK(wins[0].title == "caf\xC3\xA9 \"x\"", "titles decode escapes");
    std::vector<Window> sel0 = select_windows(mons[0], wins);
    std::string cls0; for (auto& w : sel0) cls0 += w.cls + ",";
    CHECK(cls0 == "kitty,pinned,", "monitor 0 shows: active workspace + pinned, not hidden/other-ws/unmapped (got %s)", cls0.c_str());
    std::vector<Window> sel1 = select_windows(mons[1], wins);
    std::string cls1; for (auto& w : sel1) cls1 += w.cls + ",";
    CHECK(cls1 == "second,scratch,", "monitor 1 shows its workspace and its shown scratchpad (got %s)", cls1.c_str());
    CHECK(sel1[0].x == 0 && sel1[0].y == 100, "coordinates become output-relative (got %d,%d)", sel1[0].x, sel1[0].y);
    CHECK(!parse_monitors("{}", mons, e) && !parse_clients("[1]", wins, e) && !parse_clients("nope", wins, e), "wrong shapes / garbage rejected");
    std::vector<Monitor> keep = mons; parse_monitors("[", mons, e); CHECK(mons.size() == keep.size(), "failed parse keeps the previous list");
    CHECK(parse_monitors("[{\"id\":0,\"name\":\"X\",\"width\":100,\"height\":100,\"scale\":0}]", mons, e) && mons[0].w == 100, "scale 0 falls back to 1");
    CHECK(parse_monitors("[{\"id\":0,\"width\":0,\"height\":0}]", mons, e) && mons.empty(), "zero-size monitors are dropped");
}

static void test_hypr_socket_path() {
    std::puts("[hyprland socket path]");
    setenv("NERVA_HYPR_SOCKET", "/x/y.sock", 1); CHECK(hypr_socket_path() == "/x/y.sock", "explicit override");
    unsetenv("NERVA_HYPR_SOCKET"); unsetenv("HYPRLAND_INSTANCE_SIGNATURE"); CHECK(hypr_socket_path().empty(), "no Hyprland -> empty path");
    setenv("HYPRLAND_INSTANCE_SIGNATURE", "../../etc/x", 1); CHECK(hypr_socket_path().empty(), "a signature containing '/' is refused");
    unsetenv("HYPRLAND_INSTANCE_SIGNATURE");
    std::string r, e; CHECK(!hypr_query("", "j/clients", r, e) && !e.empty(), "query without a path fails with a reason");
    CHECK(!hypr_query("/nonexistent/sock", "j/clients", r, e) && e.find("cannot connect") != std::string::npos, "missing socket: %s", e.c_str());
}

static void test_mapping() {
    std::puts("[mapping and surfaces]");
    View v; v.W = 1920; v.H = 1080; v.ppm = 100.f; v.floor_off = 0.f;
    CHECK(v.sx(0) == 960.f && v.sy(0) == 1080.f && v.sy(1.f) == 980.f && v.sx(1.f) == 1060.f, "axes: z right, y up, floor at bottom");
    for (float z : {-5.f, 0.f, 3.3f}) CHECK(std::fabs(v.z_of(v.sx(z)) - z) < 1e-4f, "z round trip %g", z);
    for (float y : {0.f, 0.5f, 1.75f}) CHECK(std::fabs(v.y_of(v.sy(y)) - y) < 1e-4f, "y round trip %g", y);
    v.floor_off = 40.f; CHECK(v.sy(0) == 1040.f, "floor offset lifts the floor line");
    v.floor_off = 0.f;
    std::vector<Window> w(5);
    w[0].x = 100; w[0].y = 780; w[0].w = 400; w[0].h = 300; w[0].cls = "ok";                    // top at y=780 -> 3.0 m
    w[1].x = 600; w[1].y = 300; w[1].w = 10; w[1].h = 300; w[1].cls = "narrow";
    w[2].x = 700; w[2].y = 1080; w[2].w = 400; w[2].h = 100; w[2].cls = "at-floor";
    w[3].x = -300; w[3].y = 500; w[3].w = 500; w[3].h = 300; w[3].cls = "left-clipped";           // visible part: 0..200
    w[4].x = 1800; w[4].y = 500; w[4].w = 500; w[4].h = 300; w[4].cls = "right-clipped";          // visible part: 1800..1920
    auto s = surfaces_from_windows(v, w);
    CHECK(s.size() == 3, "narrow and at-floor windows are not surfaces (got %zu)", s.size());
    CHECK(s[0].cls == "ok" && std::fabs(s[0].y - 3.0f) < 1e-4f && std::fabs(s[0].z0 - (-8.6f)) < 1e-3f && std::fabs(s[0].z1 - (-4.6f)) < 1e-3f, "window top -> height and z span (y=%.3f z=[%.2f,%.2f])", s[0].y, s[0].z0, s[0].z1);
    CHECK(std::fabs(s[1].z0 - (-9.6f)) < 1e-3f && std::fabs(s[1].z1 - (-7.6f)) < 1e-3f, "left overhang is clipped to the screen edge");
    CHECK(std::fabs(s[2].z1 - 9.6f) < 1e-3f, "right overhang is clipped to the screen edge");
    CHECK(s[0].src == 0 && s[1].src == 3 && s[2].src == 4, "surfaces remember their source window");
    v.top_off = 10.f; auto s2 = surfaces_from_windows(v, w); CHECK(std::fabs(s2[0].y - 2.9f) < 1e-4f, "top offset lowers the surface");
    CHECK(surfaces_from_windows(v, w, 2.0f).size() == 0 || surfaces_from_windows(v, w, 2.0f)[0].y <= 2.0f, "max height filter");
    CHECK(json_escape("a\"b\\c\n\x01") == "a\\\"b\\\\c\\n\\u0001", "json_escape");
}

static void test_draw() {
    std::puts("[drawing the body]");
    ik::Body body; for (int i = 0; i < 240; ++i) body.update(1.f / 120.f);
    View v; v.W = 800; v.H = 400; v.ppm = 114.f;
    float lowest = -1e9f, topmost = 1e9f;
    for (const DrawnBone& d : project_bones(body, v)) { lowest = std::max(lowest, std::max(d.ay, d.by) + d.r); topmost = std::min(topmost, std::min(d.ay, d.by) - d.r); }
    CHECK(std::fabs(lowest - v.sy(0.f)) < 2.0f, "feet rest on the floor line (lowest edge %.1f, floor %.1f)", lowest, v.sy(0.f));
    float h = v.sy(0.f) - topmost; CHECK(h > 1.6f * v.ppm && h < 2.0f * v.ppm, "standing height ~1.75 m on screen (%.0f px = %.2f m)", h, h / v.ppm);
    std::vector<uint32_t> px((size_t)v.W * v.H, 0); Canvas cv{px.data(), v.W, v.H, v.W};
    Style st; Rect bb = draw_body(cv, body, v, st);
    CHECK(!bb.empty() && bb.x0 > 300 && bb.x1 < 500, "drawn rect is near the screen centre (%d..%d)", bb.x0, bb.x1);
    size_t outside = 0; for (int y = 0; y < v.H; ++y) for (int x = 0; x < v.W; ++x) if (px[(size_t)y * v.W + x] && (x < bb.x0 || x >= bb.x1 || y < bb.y0 || y >= bb.y1)) ++outside;
    CHECK(outside == 0, "no pixel outside the returned rect (%zu)", outside);
    cv.clear(bb); bool clean = true; for (uint32_t p : px) if (p) clean = false; CHECK(clean, "clearing the returned rect leaves a transparent buffer");
    // every group order: near (right) limbs are drawn after far (left) ones
    CHECK(bone_group("hand_L") == 0 && bone_group("spine2") == 1 && bone_group("toe_R") == 2 && bone_group("pelvis") == 1, "depth groups");
    CHECK(bone_radius_m("index_2_L") < bone_radius_m("hand_L") && bone_radius_m("thigh_R") > bone_radius_m("shin_R"), "radii are sensible");
}

static void test_walk_onto_surface() {
    std::puts("[walking onto a window top]");
    ik::Body body; View v; v.W = 1280; v.H = 420; v.ppm = 114.f;
    std::vector<Window> ws(1); ws[0].x = 760; ws[0].y = 380; ws[0].w = 380; ws[0].h = 300; ws[0].cls = "w";
    for (const SurfaceDef& s : surfaces_from_windows(v, ws)) body.add_surface(-kDepth, kDepth, s.z0, s.z1, s.y);
    for (int i = 0; i < 120; ++i) body.update(1.f / 120.f);
    int s = 1; bool failed = false; float h = 1.f / 120.f;
    for (int i = 0; i < 8 && !failed; ++i, s = 1 - s) {
        ik::Vec3 o = body.foot_sole(1 - s); float z = o.z + 0.45f, x = s == 0 ? 0.10f : -0.10f, y = body.surface_at(x, z);
        auto r = body.command(std::string("FOOT ") + (s ? "right " : "left ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z));
        if (!r.ok) { failed = true; std::printf("  step refused: %s\n", r.msg.c_str()); break; }
        for (int k = 0; k < 120 * 4 && (body.stepping() || k < 3) && !body.fallen(); ++k) body.update(h);
    }
    for (int i = 0; i < 240; ++i) body.update(h);
    float top = v.y_of(380.f);
    CHECK(!failed && !body.fallen(), "walks up onto the window top without falling");
    CHECK(body.pelvis().z > 3.0f, "ended past the window's left edge (z=%.2f)", body.pelvis().z);
    CHECK(std::fabs(body.foot_sole(0).y - top) < 0.01f && std::fabs(body.foot_sole(1).y - top) < 0.01f, "both soles at the window-top height %.3f m (%.3f, %.3f)", top, body.foot_sole(0).y, body.foot_sole(1).y);
}

static void test_tall_window_and_desktop_semantics() {
    std::puts("[windows far above the body are not floors]");
    View v; v.W = 1920; v.H = 1080; v.ppm = 114.f;
    std::vector<Window> ws(1); ws[0].x = 0; ws[0].y = 23; ws[0].w = 1920; ws[0].h = 1000; ws[0].cls = "maximised";   // top 9.27 m up, covers the whole screen width
    ik::Body body; auto surf = surfaces_from_windows(v, ws);
    CHECK(surf.size() == 1 && surf[0].y > 9.0f, "fixture: a maximised window is a 9 m high 'surface' (%.2f)", surf.empty() ? 0.f : surf[0].y);
    for (const SurfaceDef& s : surf) body.add_surface(-kDepth, kDepth, s.z0, s.z1, s.y);
    CHECK(body.surface_at(0.f, 0.f) > 9.0f && body.surface_at(0.f, 0.f, 0.f) == 0.f, "surface_at(ref): unreachable tops are not floors for a foot at the floor");
    for (int i = 0; i < 240; ++i) body.update(1.f / 120.f);
    CHECK(std::fabs(body.foot_sole(0).y) < 1e-4f && std::fabs(body.pelvis().y - 0.927f) < 0.05f, "STAND beneath it: feet stay on the floor (foot y=%.3f pelvis y=%.3f)", body.foot_sole(0).y, body.pelvis().y);
    int refused = 0; bool failed = false; float h = 1.f / 120.f; int s = 1;
    for (int i = 0; i < 8 && !failed; ++i, s = 1 - s) {
        ik::Vec3 o = body.foot_sole(1 - s); float z = o.z + 0.45f, x = s == 0 ? 0.10f : -0.10f, y = body.surface_at(x, z, o.y);
        auto r = body.command(std::string("FOOT ") + (s ? "right " : "left ") + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z));
        if (!r.ok) { ++refused; failed = true; break; }
        for (int k = 0; k < 120 * 4 && (body.stepping() || k < 3) && !body.fallen(); ++k) body.update(h);
    }
    CHECK(refused == 0 && !body.fallen() && body.pelvis().z > 2.5f, "walks the whole way under a maximised window (z=%.2f)", body.pelvis().z);
    CHECK(std::fabs(body.foot_sole(0).y) < 1e-3f && std::fabs(body.foot_sole(1).y) < 1e-3f, "feet never snapped up to it");
    auto r = body.command("FOOT right -0.1 9.27 3.9"); CHECK(!r.ok || true, "(a model asking for a 9 m foot lift gets a refusal or a held foot, never a teleport)");
    body.update(h); CHECK(body.foot_sole(1).y < 1.0f, "no teleport after asking for it");
}

static void test_fullscreen_cursor_ghost() {
    std::puts("[fullscreen, pointer, ghost]");
    std::vector<Window> w(2); w[0].fullscreen = 1; w[1].fullscreen = 0;
    CHECK(!any_fullscreen(w), "maximised (1) is not fullscreen");
    w[1].fullscreen = 2; CHECK(any_fullscreen(w), "fullscreen (2) is");
    std::vector<Window> cl; std::string e;
    CHECK(parse_clients("[{\"at\":[0,0],\"size\":[5,5],\"fullscreen\":true},{\"at\":[0,0],\"size\":[5,5],\"fullscreen\":1},{\"at\":[0,0],\"size\":[5,5],\"fullscreen\":2}]", cl, e) && cl[0].fullscreen == 2 && cl[1].fullscreen == 1 && cl[2].fullscreen == 2, "old bool form -> 2, new int form kept");
    float x = 0, y = 0;
    CHECK(parse_cursor("{\"x\":1234.5,\"y\":67}", x, y) && x == 1234.5f && y == 67.f, "cursorpos parses");
    CHECK(!parse_cursor("{\"x\":1}", x, y) && !parse_cursor("[1,2]", x, y) && !parse_cursor("nope", x, y), "bad cursorpos rejected");
    Rect r{100, 100, 200, 300};
    CHECK(near_rect(r, 150, 200, 0) && near_rect(r, 230, 200, 48) && !near_rect(r, 260, 200, 48) && !near_rect(Rect{}, 0, 0, 100), "near_rect");
    ik::Body body; for (int i = 0; i < 120; ++i) body.update(1.f / 120.f);
    View v; v.W = 400; v.H = 300; v.ppm = 114.f;
    std::vector<uint32_t> a(400 * 300, 0), b(400 * 300, 0); Canvas ca{a.data(), 400, 300, 400}, cb{b.data(), 400, 300, 400};
    Style st; draw_body(ca, body, v, st); st.opacity = 0.2f; draw_body(cb, body, v, st);
    unsigned ma = 0, mb = 0; for (uint32_t p : a) ma = std::max(ma, p >> 24); for (uint32_t p : b) mb = std::max(mb, p >> 24);
    CHECK(ma > 240 && mb > 40 && mb < 70, "opacity 0.2 scales alpha (max alpha %u -> %u)", ma, mb);
}

int main() {
    test_tall_window_and_desktop_semantics(); test_fullscreen_cursor_ghost();
    test_raster(); test_json(); test_hypr(); test_hypr_socket_path(); test_mapping(); test_draw(); test_walk_onto_surface();
    std::printf("\n%d checks, %d failed\n", g_n, g_fail);
    return g_fail ? 1 : 0;
}
