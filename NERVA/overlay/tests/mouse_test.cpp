// mouse_test.cpp -- the AI pointer's safety gate and the verified pointer, against a simulated compositor.
#include <cstdio>
#include <string>
#include <vector>
#include "mouse_gate.hpp"
#include "pointer.hpp"
using namespace nerva;

static int g_fail = 0, g_n = 0;
#define CHECK(c, ...) do { ++g_n; if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static Window mkwin(const char* addr, const char* cls, int x, int y, int w, int h) {
    Window v; v.address = addr; v.cls = cls; v.x = x; v.y = y; v.w = w; v.h = h; v.mapped = true; return v;
}

static void test_gate() {
    std::puts("[gate: grant rules]");
    MouseGate g; Window kitty = mkwin("0x1", "kitty", 100, 100, 800, 600);
    CHECK(!g.check_move(300, 300, 0).ok && g.check_move(300, 300, 0).why.find("no app is granted") != std::string::npos, "nothing works before a grant");
    CHECK(!g.check_click("left", 300, 300, 0).ok && !g.check_scroll(1, 300, 300).ok, "click and scroll are refused too");
    for (const char* bad : {"org.keepassxc.KeePassXC", "Bitwarden", "polkit-gnome-authentication-agent-1", "pinentry-qt", "gcr-prompter", "ssh-askpass", "org.kde.kwalletd5", "1Password", "hyprpolkitagent", "org.gnome.seahorse.Application"}) {
        Verdict v = g.grant(mkwin("0x9", bad, 0, 0, 500, 500), 0);
        CHECK(!v.ok && v.why.find("never-grant") != std::string::npos, "never granted: %s", bad);
    }
    CHECK(!g.active(), "refused grants leave nothing active");
    Window fs = kitty; fs.fullscreen = 2; CHECK(!g.grant(fs, 0).ok, "fullscreen windows cannot be granted");
    Window hid = kitty; hid.hidden = true; CHECK(!g.grant(hid, 0).ok, "hidden windows cannot be granted");
    CHECK(!g.grant(mkwin("0x2", "tiny", 0, 0, 30, 30), 0).ok, "tiny windows cannot be granted");
    CHECK(g.grant(kitty, 0).ok && g.active() && g.cls() == "kitty", "a normal window can be granted");

    std::puts("[gate: where and when]");
    CHECK(g.check_move(300, 300, 1).ok, "inside is fine");
    CHECK(!g.check_move(50, 300, 1).ok && g.check_move(50, 300, 1).why.find("outside the granted window") != std::string::npos, "left of the window refused: %s", g.check_move(50, 300, 1).why.c_str());
    CHECK(!g.check_move(300, 99, 1).ok && !g.check_move(300, 701, 1).ok && !g.check_move(901, 300, 1).ok, "above/below/right refused");
    CHECK(!g.check_move(104, 300, 1).ok && !g.check_move(300, 104, 1).ok && g.check_move(110, 300, 1).ok, "the 8 px edge margin (title bar / resize border) is off limits");
    CHECK(!g.check_move(NAN, 300, 1).ok && !g.check_move(300, INFINITY, 1).ok, "NaN / inf refused");
    g.note_move(300, 300, 1.0); CHECK(!g.check_move(800, 650, 1.01).ok && g.check_move(800, 650, 1.01).why.find("speed") != std::string::npos, "teleport-speed motion refused");
    CHECK(g.check_move(310, 300, 1.2).ok, "normal motion fine");
    { MouseGate gs; gs.grant(mkwin("0x7", "tall", 100, 100, 800, 2000), 0); gs.set_screen(1280, 720); Verdict v = gs.check_move(300, 900, 0); CHECK(!v.ok && v.why.find("outside the screen") != std::string::npos, "inside the window but below the screen is refused: %s", v.why.c_str()); CHECK(gs.check_move(300, 700, 0).ok && !gs.check_move(-1, 300, 0).ok && !gs.check_move(1280, 300, 0).ok, "screen edges"); }
    CHECK(g.check_click("left", 300, 300, 2.0).ok, "left click allowed");
    g.note_click(2.0); CHECK(!g.check_click("left", 300, 300, 2.1).ok && g.check_click("left", 300, 300, 2.3).ok, "click rate limited (4/s)");
    CHECK(!g.check_click("right", 300, 300, 5.0).ok && g.check_click("right", 300, 300, 5.0).why.find("disabled") != std::string::npos, "right click off by default");
    CHECK(!g.check_click("middle", 300, 300, 5.0).ok, "middle click refused");
    CHECK(g.check_scroll(-3, 300, 300).ok && !g.check_scroll(0, 300, 300).ok && !g.check_scroll(50, 300, 300).ok, "scroll limited to a few notches");
    CHECK(g.clicks() == 1, "clicks counted");

    std::puts("[gate: the grant ends by itself]");
    std::vector<Window> vis{kitty};
    CHECK(g.update(vis, "0x1", false, 10).empty() && g.active(), "visible and focused: stays");
    CHECK(g.check_move(300, 300, 11).ok, "...and works");
    CHECK(g.update(vis, "0x77", false, 12).empty() && g.active() && !g.check_move(300, 300, 12).ok && g.check_move(300, 300, 12).why.find("not focused") != std::string::npos, "focus moved elsewhere: pointer refused, grant kept");
    CHECK(g.update(vis, "0x1", false, 13).empty() && g.check_move(300, 300, 13).ok, "focus back: works again");
    Window moved = kitty; moved.x = 400; moved.y = 200; std::vector<Window> v2{moved}; g.update(v2, "0x1", false, 14);
    CHECK(!g.check_move(300, 300, 14).ok && g.check_move(500, 300, 14).ok, "follows the window when it moves");
    CHECK(!g.update({}, "0x1", false, 15).empty() && !g.active(), "window gone: grant revoked");
    CHECK(g.last_revoke_reason().find("closed") != std::string::npos && !g.check_move(500, 300, 15).ok, "...and everything is refused again");
    g.grant(kitty, 100); CHECK(g.update(vis, "0x1", true, 101).find("fullscreen") != std::string::npos && !g.active(), "overlay hidden by a fullscreen window: revoked");
    g.grant(kitty, 200); Window kf = kitty; kf.fullscreen = 2; CHECK(!g.update({kf}, "0x1", false, 201).empty(), "the granted window itself going fullscreen: revoked");
    g.grant(kitty, 300); CHECK(g.update(vis, "0x1", false, 300 + 299).empty() && g.active(), "just before the TTL: still granted");
    CHECK(g.update(vis, "0x1", false, 300 + 301).find("expired") != std::string::npos && !g.active(), "after 300 s: expired, the human must grant again");
    g.grant(kitty, 400); g.revoke("user"); CHECK(!g.active() && !g.check_move(300, 300, 400).ok, "revoke() stops everything at once");
    GateConfig c; c.allow_right_click = true; c.margin_px = 0; MouseGate g2(c); g2.grant(kitty, 0);
    { MouseGate g3; g3.grant(kitty, 0); g3.set_display("(hidden)"); Verdict v = g3.check_move(5, 5, 0); CHECK(v.why.find("(hidden)") != std::string::npos && v.why.find("kitty") == std::string::npos, "a hidden app is not named in refusals: %s", v.why.c_str()); }
    CHECK(g2.check_click("right", 300, 300, 1).ok && g2.check_move(100, 100, 1).ok, "config: right click and margin can be changed");
    Window ws = kitty; ws.address = "0x1"; g.grant(kitty, 500); g.grant(mkwin("0x5", "other", 0, 0, 300, 300), 500);
    CHECK(g.address() == "0x5", "a new grant replaces the old one (one window at a time)");
}

// ---- a simulated compositor --------------------------------------------------------------------------------------------
struct Sim {
    double ox = 0, oy = 0, sw = 1920, sh = 1080;     // abs 0..65535 maps onto this rectangle of the layout
    double exponent = 1.0; bool ignore = false, cursor_ok = true;
    float cx = 400, cy = 300; double fix_dx = 0, fix_dy = 0;
    std::vector<std::string> log;
};
struct SimBackend : PointerBackend {
    Sim* s; explicit SimBackend(Sim* sim) : s(sim) {}
    bool abs_move(int ax, int ay, std::string&) override {
        if (s->ignore) return true;
        s->cx = (float)(s->ox + s->sw * std::pow(ax / 65535.0, s->exponent) + s->fix_dx); s->cy = (float)(s->oy + s->sh * std::pow(ay / 65535.0, s->exponent) + s->fix_dy);
        s->log.push_back("move"); return true;
    }
    bool button(int code, bool down, std::string& err) override { if (code != BTN_LEFT && code != BTN_RIGHT) { err = "no"; return false; } s->log.push_back(std::string(code == BTN_LEFT ? "L" : "R") + (down ? "down" : "up")); return true; }
    bool wheel(int n, std::string&) override { s->log.push_back("wheel" + std::to_string(n)); return true; }
};
static PointerController make(Sim& s) {
    return PointerController(std::unique_ptr<PointerBackend>(new SimBackend(&s)), [&s](float& x, float& y) { if (!s.cursor_ok) return false; x = s.cx; y = s.cy; return true; }, [](int) {});
}

static void test_pointer() {
    std::puts("[pointer: calibration against different compositors]");
    struct Case { const char* name; double ox, oy, sw, sh; } cases[] = {
        {"one 1920x1080 monitor", 0, 0, 1920, 1080}, {"two monitors side by side (whole layout)", 0, 0, 3840, 1080},
        {"second monitor only (offset layout)", 1920, 0, 1920, 1080}, {"monitor above another", 0, -1080, 1920, 2160}, {"1.5x scaled 2560x1440 -> 1707x960", 0, 0, 1707, 960}};
    for (const Case& c : cases) {
        Sim s; s.ox = c.ox; s.oy = c.oy; s.sw = c.sw; s.sh = c.sh; s.cx = (float)(c.ox + 123); s.cy = (float)(c.oy + 77);
        PointerController p = make(s); std::string e;
        CHECK(p.calibrate(e), "%s: calibrates (%s)", c.name, e.c_str());
        CHECK(std::fabs(s.cx - (c.ox + 123)) <= 2.5 && std::fabs(s.cy - (c.oy + 77)) <= 2.5, "%s: the user's cursor is put back (%.0f,%.0f)", c.name, s.cx, s.cy);
        float tx = (float)(c.ox + c.sw * 0.31), ty = (float)(c.oy + c.sh * 0.62);
        CHECK(p.move_to(tx, ty, 1.0, e) && std::fabs(s.cx - tx) <= 2.5 && std::fabs(s.cy - ty) <= 2.5, "%s: lands on target (%.1f,%.1f) vs (%.1f,%.1f) %s", c.name, s.cx, s.cy, tx, ty, e.c_str());
        tx = (float)(c.ox + c.sw * 0.95); ty = (float)(c.oy + c.sh * 0.05);
        CHECK(p.move_to(tx, ty, 2.0, e) && std::hypot(s.cx - tx, s.cy - ty) <= 2.5, "%s: and near the far corner", c.name);
    }
    std::puts("[pointer: failures are detected, never guessed around]");
    { Sim s; s.ignore = true; PointerController p = make(s); std::string e; CHECK(!p.calibrate(e) && e.find("did not move") != std::string::npos, "ignored device: %s", e.c_str()); CHECK(!p.move_to(5, 5, 0, e) && !p.click("left", 0, e), "uncalibrated: no move, no click"); }
    { Sim s; s.exponent = 1.6; PointerController p = make(s); std::string e; CHECK(!p.calibrate(e) && e.find("not linear") != std::string::npos, "non-linear mapping: %s", e.c_str()); }
    { Sim s; s.cursor_ok = false; PointerController p = make(s); std::string e; CHECK(!p.calibrate(e) && e.find("cannot read") != std::string::npos, "unreadable cursor: %s", e.c_str()); }
    { Sim s; PointerController p = make(s); std::string e; p.calibrate(e); s.fix_dx = 90; s.fix_dy = -40;             // the layout changes under us
      CHECK(p.move_to(900, 500, 1.0, e) && std::hypot(s.cx - 900, s.cy - 500) <= 2.5, "a constant offset is corrected on the fly (%s)", e.c_str());
      s.fix_dx = 0; s.fix_dy = 0; s.exponent = 1.5; CHECK(!p.move_to(1500, 900, 2.0, e) && e.find("landed") != std::string::npos && !p.ready(), "a mapping that stops making sense: refused, and calibration is dropped (%s)", e.c_str()); }
    CHECK(!([] { AxisFit f; return fit_axis(1, 5, 1, 9, f); })() && !([] { AxisFit f; return fit_axis(0, 100, 60000, 50, f); })() && !([] { AxisFit f; return fit_axis(0, 0, 60000, 10, f); })(), "fit_axis: degenerate, decreasing and absurd spans rejected");

    std::puts("[pointer: clicks and the user taking over]");
    { Sim s; PointerController p = make(s); std::string e; p.calibrate(e);
      CHECK(!p.click("left", 1.0, e) && e.find("move the pointer first") != std::string::npos, "click before any move: refused");
      p.move_to(700, 400, 1.0, e); s.log.clear();
      CHECK(p.click("left", 2.0, e) && s.log.size() == 2 && s.log[0] == "Ldown" && s.log[1] == "Lup", "click = press then release at the verified spot");
      CHECK(p.scroll(-2, 3.0, e) && s.log.back() == "wheel-2", "scroll");
      s.cx += 40;                                                                                    // the human grabs the mouse
      CHECK(!p.click("left", 4.0, e) && e.find("user moved the mouse") != std::string::npos, "user moved the pointer: click refused (%s)", e.c_str());
      s.log.clear(); CHECK(!p.move_to(710, 400, 5.0, e) && e.find("paused") != std::string::npos && s.log.empty(), "...and the AI is paused for a few seconds, nothing was sent (%s)", e.c_str());
      CHECK(p.move_to(710, 400, 8.0, e), "after the pause the AI can move again (%s)", e.c_str());
      s.cx += 3; CHECK(p.click("left", 9.0, e), "a few pixels of drift is tolerated"); }
}

int main() { test_gate(); test_pointer(); std::printf("\n%d checks, %d failed\n", g_n, g_fail); return g_fail ? 1 : 0; }
