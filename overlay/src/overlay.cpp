// overlay.cpp -- NERVA desktop overlay for Wayland (Hyprland).
//
//   * a transparent, click-through, always-on-top wlr-layer-shell surface (no input region: every click passes through)
//   * the full-body humanoid (include/ik_body.hpp) rendered on the CPU into a wl_shm buffer, side view
//   * perception: Hyprland's window geometry -> body surfaces (window tops are floors the body can step onto)
//   * a line protocol on stdin/stdout for whoever drives the body (a model, a skill layer, a human)
//   * NO cursor control: the /dev/uinput bridge is deliberately not part of this program.
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wayland-client.h>
extern "C" {
#include "wlr-layer-shell-unstable-v1-client-protocol.h"
}
#include "draw_body.hpp"
#include "hypr.hpp"
#include "mouse_gate.hpp"
#include "perception.hpp"
#include "pointer.hpp"
#include "scene.hpp"
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

using namespace nerva;
using Clock = std::chrono::steady_clock;

static double mono_now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------------------------------------------- options
struct Options {
    float height_px = 200.f, ppm = 0.f, floor_off = 0.f, top_off = 0.f, speed = 1.f;
    int fps = 60, fingers = 1;
    std::string output, demo, render_test;
    bool mouse = false, allow_right = false, a11y = false; std::string a11y_helper; int a11y_dx = 0, a11y_dy = 0; std::string pointer = "uinput", ctl; double grant_ttl = 300;
    bool debug = false, titles = false, no_hypr = false, perceive_once = false, help = false, over_fullscreen = false, no_ghost = false;
    Style style;
};

static bool parse_hex(const char* s, Color& c) {
    if (*s == '#') ++s;
    if (std::strlen(s) != 6) return false;
    char* end; unsigned long v = std::strtoul(s, &end, 16);
    if (*end) return false;
    c.r = ((v >> 16) & 255) / 255.f; c.g = ((v >> 8) & 255) / 255.f; c.b = (v & 255) / 255.f; return true;
}

static void usage() {
    std::puts(
        "nerva-overlay -- transparent click-through desktop overlay (Wayland / Hyprland)\n"
        "  --height PX        on-screen height of the body in logical px (default 200)\n"
        "  --ppm N            pixels per metre (overrides --height)\n"
        "  --floor-off PX     lift the floor line this many px above the output bottom (e.g. to stand on a bar)\n"
        "  --top-off PX       add to every window top (compositor borders), default 0\n"
        "  --output NAME      monitor to draw on (e.g. DP-1); default: the compositor's choice (focused)\n"
        "  --speed K          simulate K times faster than real time (default 1; walking is slow, 0.31 m/s)\n"
        "  --fps N            frame pacing cap, 5..240 (default 60)\n"
        "  --fingers 0|1|2    finger level of detail (default 1)\n"
        "  --accent RRGGBB    outline colour (default 8c1217)   --fill RRGGBB   body colour (default 09090b)\n"
        "  --demo walk        harness demo: take FOOT steps to the right, over window tops, until the screen edge\n"
        "  --debug            draw the floor line, window tops, centre of mass and foot contact points\n"
        "  --titles           include window titles in PERCEIVE (default: class names only)\n"
        "  --over-fullscreen  keep drawing over fullscreen windows (default: hide and pause while a game/video is fullscreen)\n"
        "  --no-ghost         do not fade the body out when the mouse pointer is near it\n"
        "  --mouse            ENABLE the AI pointer (off by default). Nothing happens until YOU grant one window:\n"
        "                     nerva-overlay --ctl grant   (grants the focused window; bind it to a key)   --ctl revoke | status\n"
        "  --grant-ttl S      a grant lasts S seconds (default 300)      --allow-right-click   (default: left click only)\n"
        "  --pointer MODE     uinput (default) or dry-run (simulated pointer, for tests)\n"
        "  --ctl CMD          talk to a running overlay: grant | revoke | status, then exit\n"
        "  --a11y             let the model READ the screen through accessibility (AT-SPI): A11Y TREE|TEXT|FIND|FOCUSED|APPS|STATUS.\n"
        "                     It can read every window except the ones YOU hide: nerva-overlay --ctl \"hide <app>\" / \"unhide <app>\" / hidden\n"
        "                     (needs python3-gobject + at-spi2-core; passwords are never read)\n"
        "  --a11y-offset X Y  correct accessibility rectangles by X,Y px (if they sit a few pixels off the real widgets)\n"
        "  --a11y-helper PATH path of tools/a11y_helper.py (default: found next to the binary)\n"
        "  --no-hypr          do not talk to Hyprland (floor only; no fullscreen or pointer awareness)\n"
        "  --perceive-once    print one PERCEIVE json (needs Hyprland) and exit; no Wayland connection\n"
        "  --render-test F    render a test frame to F (PPM) and exit; no Wayland connection\n"
        "stdin (one command per line; replies on stdout as OK/ERR lines, body feedback as EVENT lines):\n"
        "  body commands: HAND FOOT PELVIS FACE LOOK TORSO GRIP STAND BALANCE SURFACE   (see docs/BODY.md)\n"
        "  PERCEIVE | STATUS | SPEED k | W2PX x y z | PX2W px py | QUIT\n"
        "World: metres, Y up. Screen x = W/2 + z*ppm, screen y = H - floor_off - y*ppm. The body faces +z (screen right).");
}

static bool parse_args(int argc, char** argv, Options& o, std::string& err) {
    auto val = [&](int& i, const char* name) -> const char* { if (i + 1 >= argc) { err = std::string(name) + " needs a value"; return nullptr; } return argv[++i]; };
    auto num = [&](int& i, const char* name, float lo, float hi, float& out) -> bool {
        const char* s = val(i, name); if (!s) return false;
        char* e; float v = std::strtof(s, &e);
        if (*e || !std::isfinite(v) || v < lo || v > hi) { err = std::string(name) + " must be a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"; return false; }
        out = v; return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; float t;
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--version") { std::puts("nerva-overlay 0.4 (body, surfaces, ghosting, fullscreen-hide, gated pointer --mouse, accessibility --a11y)"); std::exit(0); }
        else if (a == "--height") { if (!num(i, "--height", 40, 4000, o.height_px)) return false; }
        else if (a == "--ppm") { if (!num(i, "--ppm", 10, 5000, o.ppm)) return false; }
        else if (a == "--floor-off") { if (!num(i, "--floor-off", 0, 4000, o.floor_off)) return false; }
        else if (a == "--top-off") { if (!num(i, "--top-off", -200, 200, o.top_off)) return false; }
        else if (a == "--speed") { if (!num(i, "--speed", 0.05f, 20, o.speed)) return false; }
        else if (a == "--fps") { if (!num(i, "--fps", 5, 240, t)) return false; o.fps = (int)t; }
        else if (a == "--fingers") { if (!num(i, "--fingers", 0, 2, t)) return false; o.fingers = (int)t; }
        else if (a == "--output") { const char* s = val(i, "--output"); if (!s) return false; o.output = s; }
        else if (a == "--accent") { const char* s = val(i, "--accent"); if (!s) return false; if (!parse_hex(s, o.style.accent)) { err = "--accent wants RRGGBB"; return false; } }
        else if (a == "--fill") { const char* s = val(i, "--fill"); if (!s) return false; if (!parse_hex(s, o.style.fill)) { err = "--fill wants RRGGBB"; return false; } }
        else if (a == "--demo") { const char* s = val(i, "--demo"); if (!s) return false; o.demo = s; if (o.demo != "walk") { err = "--demo only knows 'walk'"; return false; } }
        else if (a == "--render-test") { const char* s = val(i, "--render-test"); if (!s) return false; o.render_test = s; }
        else if (a == "--debug") o.debug = true;
        else if (a == "--mouse") o.mouse = true;
        else if (a == "--a11y") o.a11y = true;
        else if (a == "--a11y-helper") { const char* v = val(i, "--a11y-helper"); if (!v) return false; o.a11y_helper = v; }
        else if (a == "--a11y-offset") { float x, y; if (!num(i, "--a11y-offset", -200, 200, x) || !num(i, "--a11y-offset", -200, 200, y)) return false; o.a11y_dx = (int)x; o.a11y_dy = (int)y; }
        else if (a == "--allow-right-click") o.allow_right = true;
        else if (a == "--grant-ttl") { float t; if (!num(i, "--grant-ttl", 5, 3600, t)) return false; o.grant_ttl = t; }
        else if (a == "--pointer") { const char* v = val(i, "--pointer"); if (!v) return false; o.pointer = v; if (o.pointer != "uinput" && o.pointer != "dry-run") { err = "--pointer is uinput or dry-run"; return false; } }
        else if (a == "--ctl") { const char* v = val(i, "--ctl"); if (!v) return false; o.ctl = v; std::string c0 = o.ctl.substr(0, o.ctl.find(' ')); if (c0 != "grant" && c0 != "revoke" && c0 != "status" && c0 != "hide" && c0 != "unhide" && c0 != "hidden") { err = "--ctl is grant | revoke | status | hide <app> | unhide <app> | hidden"; return false; } }
        else if (a == "--over-fullscreen") o.over_fullscreen = true;
        else if (a == "--no-ghost") o.no_ghost = true;
        else if (a == "--titles") o.titles = true;
        else if (a == "--no-hypr") o.no_hypr = true;
        else if (a == "--perceive-once") o.perceive_once = true;
        else { err = "unknown option '" + a + "' (try --help)"; return false; }
    }
    o.style.debug_overlay = o.debug;
    return true;
}

// ---------------------------------------------------------------------------------------------------- the world
static bool to_f(const std::string& s, float& v) { char* e; v = std::strtof(s.c_str(), &e); return !s.empty() && !*e && std::isfinite(v); }

// A simulated pointer for tests (--pointer dry-run): the 0..65535 range maps onto the overlay's monitor, nothing real is touched.
struct DryScreen { double ox = 0, oy = 0, w = 1920, h = 1080; float cx = 0, cy = 0; };
struct DryBackend : PointerBackend {
    DryScreen* s; explicit DryBackend(DryScreen* d) : s(d) {}
    bool abs_move(int ax, int ay, std::string&) override { s->cx = (float)(s->ox + s->w * ax / 65535.0); s->cy = (float)(s->oy + s->h * ay / 65535.0); return true; }
    bool button(int code, bool down, std::string&) override; bool wheel(int n, std::string&) override;
};

static void reply(const std::string& s);
static void reply_event(const std::string& s) { reply("EVENT " + s); }

struct World {
    ik::Body body;
    View view;
    Options opt;
    MouseGate gate;
    HideList hide;
    A11yClient a11y;
    long a11y_next_id = 1;
    std::unique_ptr<PointerController> ptr;
    DryScreen dry;
    std::vector<Window> windows;                       // output-relative, selected for the current monitor
    std::vector<SurfaceDef> surfaces;
    std::string monitor_name; float monitor_scale = 1.f;
    std::string perception_error; bool hypr_ok = false;
    int mon_x = 0, mon_y = 0;                          // monitor origin in the global layout (to make the pointer output-relative)
    bool hidden = false;                               // a fullscreen window is active: nothing drawn, body paused
    bool cursor_ok = false; float cursor_x = 0, cursor_y = 0; double cursor_retry = 0;

    static ik::Body::Params body_params(const Options& o) { ik::Body::Params p; p.finger_lod = o.fingers; return p; }
    static GateConfig gate_cfg(const Options& o) { GateConfig c; c.ttl_s = o.grant_ttl; c.allow_right_click = o.allow_right; return c; }
    explicit World(const Options& o) : body(body_params(o)), opt(o), gate(gate_cfg(o)) {
        hide.builtin = gate.cfg.deny_classes; a11y.helper_path = A11yClient::find_helper(o.a11y_helper);
        view.ppm = o.ppm > 0.f ? o.ppm : o.height_px / (1.75f * body.params.scale);
        view.floor_off = o.floor_off; view.top_off = o.top_off;
    }

    void set_size(int W, int H) { view.W = W; view.H = H; refresh_surfaces(true); }

    // Rebuild the body's surfaces from the current windows. Only touches the body when something actually changed.
    void refresh_surfaces(bool force = false) {
        std::vector<SurfaceDef> now = surfaces_from_windows(view, windows);
        if (!force && now == surfaces) return;
        surfaces = std::move(now);
        body.clear_surfaces();
        for (const SurfaceDef& s : surfaces) body.add_surface(-kDepth, kDepth, s.z0, s.z1, s.y);
    }

    // One perception pass. `wl_output_name` = the output the overlay surface is on, if the compositor told us.
    void perceive(const std::string& wl_output_name) {
        if (opt.no_hypr) return;
        HyprSnapshot snap; std::string err;
        if (!hypr_snapshot(snap, err)) { hypr_ok = false; perception_error = err; windows.clear(); refresh_surfaces(); if (gate.active()) end_grant("lost contact with Hyprland"); return; }
        const Monitor* m = nullptr;
        for (const Monitor& c : snap.monitors) if (!opt.output.empty() ? c.name == opt.output : (!wl_output_name.empty() && c.name == wl_output_name)) m = &c;
        if (!m) for (const Monitor& c : snap.monitors) if (c.focused) m = &c;
        if (!m && !snap.monitors.empty()) m = &snap.monitors[0];
        if (!m) { hypr_ok = false; perception_error = "Hyprland reported no monitors"; return; }
        hypr_ok = true; perception_error.clear(); monitor_name = m->name; monitor_scale = m->scale; mon_x = m->x; mon_y = m->y;
        windows = select_windows(*m, snap.windows);
        refresh_surfaces();
        bool hide_overlay = !opt.over_fullscreen && any_fullscreen(windows);
        if (hide_overlay != hidden) { hidden = hide_overlay; reply_event(hide_overlay ? "overlay hidden and body paused: a fullscreen window is active" : "overlay shown again"); }
        if (gate.active()) gate.set_display(hide.matches(gate.cls()) ? "(hidden)" : gate.cls());
        if (gate.active()) {                                                    // the grant follows the window, and ends with it
            std::string ended = gate.update(windows, active_address(), hidden || any_fullscreen(windows), mono_now());
            if (!ended.empty()) { ptr.reset(); reply_event("mouse revoked: " + ended); }
        }
    }

    // ------------------------------------------------------------------------------------------------ the AI pointer
    std::string active_address() {                                              // read-only `j/activewindow`
        std::string r, e; if (!hypr_query(hypr_socket_path(), "j/activewindow", r, e, 100)) return "";
        return parse_active_address(r);
    }
    void end_grant(const std::string& why) { gate.revoke(why); ptr.reset(); reply_event("mouse revoked: " + why); }

    bool cursor_global(float& x, float& y) {
        if (opt.pointer == "dry-run") { x = dry.cx; y = dry.cy; return true; }
        std::string r, e; return hypr_query(hypr_socket_path(), "j/cursorpos", r, e, 60) && parse_cursor(r, x, y);
    }

    // Called ONLY from the control socket (the human's channel), never from the model's stdin.
    bool grant_focused(std::string& err) {
        if (!opt.mouse) { err = "the AI pointer is disabled (start the overlay with --mouse)"; return false; }
        if (opt.no_hypr) { err = "needs Hyprland (window geometry and focus)"; return false; }
        if (hidden) { err = "a fullscreen window is active"; return false; }
        perceive(monitor_name);
        if (!hypr_ok) { err = "cannot talk to Hyprland: " + perception_error; return false; }
        std::string addr = active_address();
        if (addr.empty()) { err = "no window is focused"; return false; }
        const Window* w = nullptr; for (const Window& c : windows) if (c.address == addr) w = &c;
        if (!w) { err = "the focused window is not on the monitor this overlay draws on"; return false; }
        Verdict v = gate.grant(*w, mono_now()); if (!v.ok) { err = v.why; return false; }
        std::unique_ptr<PointerBackend> be;
        if (opt.pointer == "dry-run") { dry.ox = mon_x; dry.oy = mon_y; dry.w = view.W; dry.h = view.H; dry.cx = (float)(mon_x + 40); dry.cy = (float)(mon_y + 40); be.reset(new DryBackend(&dry)); }
        else { be = UinputPointer::open(err); if (!be) { gate.revoke("no pointer device"); return false; } }
        World* self = this;
        ptr.reset(new PointerController(std::move(be), [self](float& x, float& y) { return self->cursor_global(x, y); }, [](int ms) { usleep((useconds_t)ms * 1000); }));
        if (!ptr->calibrate(err)) { ptr.reset(); gate.revoke("calibration failed"); return false; }
        gate.set_display(hide.matches(gate.cls()) ? "(hidden)" : gate.cls());
        reply_event("mouse granted to '" + gate.display() + "' for " + std::to_string((int)opt.grant_ttl) + " s (the pointer can only act inside that window)");
        return true;
    }
    std::string ctl_command(const std::string& cmd) {                           // the human's control channel
        if (cmd == "grant") { std::string e; if (grant_focused(e)) return "OK granted " + gate.cls(); return "ERR " + e; }
        if (cmd.rfind("hide ", 0) == 0 || cmd.rfind("unhide ", 0) == 0) {
            bool h = cmd[0] == 'h'; std::string pat = cmd.substr(cmd.find(' ') + 1), e;
            if (h) { if (!hide.add(pat, e)) return "ERR " + e; } else if (!hide.remove(pat)) return "ERR '" + pat + "' is not on your hide-list (built-in protections cannot be removed)";
            if (!hide.save(e)) return "ERR hidden for now, but " + e;
            return std::string("OK ") + (h ? "hiding '" : "no longer hiding '") + HideList::lower(pat) + "' from the model";
        }
        if (cmd == "hidden") {
            std::string r = "OK your hide-list:"; for (const std::string& u : hide.user) r += " " + std::string("'") + u + "'";
            if (hide.user.empty()) r += " (empty)";
            return r + "; always hidden: password managers, authentication prompts, keyrings";
        }
        if (cmd == "revoke") { if (gate.active()) { end_grant("revoked by the user"); return "OK revoked"; } return "OK nothing was granted"; }
        if (cmd == "status") {
            if (!gate.active()) return std::string("OK not granted") + (opt.mouse ? "" : " (mouse disabled)");
            char b[160]; std::snprintf(b, sizeof b, "OK granted %s, %.0f s left, %d clicks, %s", gate.cls().c_str(), gate.seconds_left(mono_now()), gate.clicks(), gate.focused() ? "focused" : "not focused");
            return b;
        }
        return "ERR unknown control command (grant | revoke | status)";
    }
    // A11Y ...  (the model's channel: READ-only; every reply is delivered later as an `A11Y <id> {json}` line)
    std::string a11y_command(const std::vector<std::string>& t) {
        if (!opt.a11y) return "ERR accessibility reading is disabled (the user must start the overlay with --a11y)";
        A11yArgs a; std::string err;
        if (!parse_a11y_args(t, a, err)) return "ERR " + err;
        if (a.op == "status") {
            char b[300]; std::snprintf(b, sizeof b, "OK a11y enabled, helper=%s, running=%d, %zu app patterns hidden", a11y.helper_path.empty() ? "(not found)" : a11y.helper_path.c_str(), a11y.running() ? 1 : 0, hide.all().size());
            return b;
        }
        if (opt.no_hypr || !hypr_ok) return "ERR needs Hyprland perception (the window list)";
        std::string focused = (a.target.empty() && (a.op == "tree" || a.op == "text" || a.op == "focused")) || a.target == "focused" ? active_address() : "";
        int skipped = 0;
        std::vector<const Window*> targets = resolve_targets(windows, hide, a.target, focused, gate.active() ? gate.address() : "", a.op == "find" || a.op == "apps", err, skipped);
        if (targets.empty()) return "ERR " + err;
        long id = a11y_next_id++;
        if (!a11y.submit(id, build_a11y_request(id, a, targets, hide, opt.a11y_dx, opt.a11y_dy), mono_now(), err)) return "ERR " + err;
        return "OK a11y " + std::to_string(id) + " queued (" + std::to_string(targets.size()) + " window" + (targets.size() == 1 ? "" : "s") + "); the answer arrives as: A11Y " + std::to_string(id) + " {json}";
    }
    ik::Vec3 hand_tip(int side) const {
        const char* want = side == 0 ? "hand_L" : "hand_R";
        for (const ik::BoneOut& b : body.bones()) if (std::string(b.name) == want) return b.tail;
        return body.wrist(side);
    }
    // MOUSE ...  (the model's channel: pointer actions only, and only inside a window the human granted)
    std::string mouse_command(const std::vector<std::string>& t) {
        if (!opt.mouse) return "ERR the AI pointer is disabled (the user must start the overlay with --mouse)";
        std::string sub = t.size() > 1 ? t[1] : ""; for (char& c : sub) c = (char)std::tolower((unsigned char)c);
        double now = mono_now(); float a = 0, b = 0;
        gate.set_screen(view.W, view.H);
        if (sub == "status") {
            if (!gate.active()) return "OK not granted: ask the user to grant a window";
            char o[200]; std::snprintf(o, sizeof o, "OK granted '%s' window x=%d y=%d w=%d h=%d, %.0f s left, %s", gate.display().c_str(), gate.window().x, gate.window().y, gate.window().w, gate.window().h, gate.seconds_left(now), gate.focused() ? "focused" : "NOT focused");
            return o;
        }
        if (sub == "grant" || sub == "revoke") return "ERR only the user can grant or revoke (nerva-overlay --ctl grant)";
        if (!gate.active() || !ptr) return "ERR " + gate.check_move(0, 0, now).why;
        float ox = (float)mon_x, oy = (float)mon_y; std::string err;
        auto move = [&](float x, float y) -> std::string {
            Verdict v = gate.check_move(x, y, now); if (!v.ok) return "ERR " + v.why;
            if (!ptr->move_to(x + ox, y + oy, now, err)) { if (!ptr->ready()) end_grant("pointer calibration lost"); return "ERR " + err; }
            gate.note_move(ptr->x() - ox, ptr->y() - oy, now);
            char o[80]; std::snprintf(o, sizeof o, "OK pointer at %.0f %.0f", ptr->x() - ox, ptr->y() - oy); return o;
        };
        if (sub == "move") { if (t.size() == 4 && to_f(t[2], a) && to_f(t[3], b)) return move(a, b); return "ERR usage: MOUSE MOVE px py   (screen pixels inside the granted window)"; }
        if (sub == "tohand") {
            std::string sd = t.size() == 3 ? t[2] : ""; for (char& c : sd) c = (char)std::tolower((unsigned char)c);
            if (sd != "left" && sd != "right") return "ERR usage: MOUSE TOHAND left|right   (moves the pointer to where that hand's fingertip is on screen)";
            ik::Vec3 tip = hand_tip(sd == "left" ? 0 : 1); return move(view.sx(tip.z), view.sy(tip.y));
        }
        if (sub == "click") {
            std::string btn = t.size() >= 3 ? t[2] : "left"; for (char& c : btn) c = (char)std::tolower((unsigned char)c);
            if (!ptr->has_position()) return "ERR move the pointer first (MOUSE MOVE px py)";
            float px = ptr->x() - ox, py = ptr->y() - oy;
            Verdict v = gate.check_click(btn, px, py, now); if (!v.ok) return "ERR " + v.why;
            if (!ptr->click(btn, now, err)) return "ERR " + err;
            gate.note_click(now); return "OK clicked " + btn;
        }
        if (sub == "scroll") {
            int n = 0; if (!(t.size() == 3 && to_f(t[2], a) && a == std::floor(a) && std::fabs(a) < 1000)) return "ERR usage: MOUSE SCROLL n   (positive = up, negative = down)";
            n = (int)a; if (!ptr->has_position()) return "ERR move the pointer first (MOUSE MOVE px py)";
            Verdict v = gate.check_scroll(n, ptr->x() - ox, ptr->y() - oy); if (!v.ok) return "ERR " + v.why;
            if (!ptr->scroll(n, now, err)) return "ERR " + err;
            return "OK scrolled " + std::to_string(n);
        }
        return "ERR usage: MOUSE STATUS | MOVE px py | TOHAND left|right | CLICK [left] | SCROLL n";
    }

    // Pointer position from Hyprland (read-only `j/cursorpos`). On failure we back off for a second instead of stalling the loop.
    void poll_cursor(double now) {
        if (opt.no_hypr || opt.no_ghost || !hypr_ok || now < cursor_retry) { if (opt.no_hypr || !hypr_ok) cursor_ok = false; return; }
        std::string r, e; float x, y;
        if (!hypr_query(hypr_socket_path(), "j/cursorpos", r, e, 40) || !parse_cursor(r, x, y)) { cursor_ok = false; cursor_retry = now + 1.0; return; }
        cursor_ok = true; cursor_x = x - (float)mon_x; cursor_y = y - (float)mon_y;
    }

    std::string perceive_json() {
        std::string j; char b[512];
        std::snprintf(b, sizeof b, "{\"output\":{\"name\":\"%s\",\"w\":%d,\"h\":%d,\"scale\":%.2f},", json_escape(monitor_name).c_str(), view.W, view.H, monitor_scale); j += b;
        std::snprintf(b, sizeof b, "\"view\":{\"ppm\":%.2f,\"floor_px\":%.1f,\"mapping\":\"screen_x=W/2+z*ppm; screen_y=H-floor_px_off-y*ppm; body faces +z (screen right); left side is +x (away from the viewer); surfaces span x in [-%.1f,%.1f]\"},", view.ppm, view.sy(0.f), kDepth, kDepth); j += b;
        j += std::string("\"hidden\":") + (hidden ? "true" : "false") + ",";
        j += std::string("\"perception\":") + (opt.no_hypr ? "\"disabled\"" : hypr_ok ? "\"ok\"" : "\"unavailable: " + json_escape(perception_error) + "\"") + ",\"windows\":[";
        for (size_t i = 0; i < windows.size(); ++i) {
            const Window& w = windows[i]; const SurfaceDef* sd = nullptr;
            for (const SurfaceDef& s : surfaces) if (s.src == (int)i) sd = &s;
            bool hid = hide.matches(w.cls);                                   // hidden by the user: geometry only, no name, no title
            std::snprintf(b, sizeof b, "%s{\"class\":\"%s\",%s", i ? "," : "", hid ? "(hidden)" : json_escape(w.cls).c_str(), hid ? "\"hidden\":true," : ""); j += b;
            if (opt.titles && !hid) j += "\"title\":\"" + json_escape(w.title) + "\",";
            std::snprintf(b, sizeof b, "\"px\":[%d,%d,%d,%d],\"floating\":%s,\"standable\":%s", w.x, w.y, w.w, w.h, w.floating ? "true" : "false", sd ? "true" : "false"); j += b;
            if (sd) { std::snprintf(b, sizeof b, ",\"top\":{\"z\":[%.3f,%.3f],\"y\":%.3f}", sd->z0, sd->z1, sd->y); j += b; }
            j += "}";
        }
        std::string st = body.status_json();
        j += "],\"mouse\":";
        if (!opt.mouse) j += "{\"enabled\":false}";
        else if (!gate.active()) j += "{\"enabled\":true,\"granted\":false}";
        else { std::snprintf(b, sizeof b, "{\"enabled\":true,\"granted\":true,\"app\":\"%s\",\"rect\":[%d,%d,%d,%d],\"focused\":%s,\"seconds_left\":%.0f}", json_escape(gate.display()).c_str(), gate.window().x, gate.window().y, gate.window().w, gate.window().h, gate.focused() ? "true" : "false", gate.seconds_left(mono_now())); j += b; }
        j += ",\"body\":" + st + "}";
        return j;
    }
};

// ---------------------------------------------------------------------------------------------------- stdin protocol
static void reply(const std::string& s) { std::fputs(s.c_str(), stdout); std::fputc('\n', stdout); std::fflush(stdout); }

bool DryBackend::button(int code, bool down, std::string&) { reply(std::string("EVENT pointer ") + (code == BTN_LEFT ? "left" : "right") + (down ? " down" : " up") + " at " + std::to_string((int)s->cx) + " " + std::to_string((int)s->cy)); return true; }
bool DryBackend::wheel(int n, std::string&) { reply("EVENT pointer wheel " + std::to_string(n)); return true; }

static bool split_num(const std::string& line, std::vector<std::string>& tok) {
    tok.clear(); size_t i = 0;
    while (i < line.size()) { while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i; size_t j = i; while (j < line.size() && line[j] != ' ' && line[j] != '\t') ++j; if (j > i) tok.push_back(line.substr(i, j - i)); i = j; }
    return !tok.empty();
}

// returns false on QUIT
static bool handle_line(World& w, const std::string& raw, const std::string& wl_output_name) {
    std::string line = raw; while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    std::vector<std::string> t;
    if (!split_num(line, t)) return true;
    std::string v = t[0]; for (char& c : v) c = (char)std::toupper((unsigned char)c);
    float a, b, c;
    if (v == "QUIT") { reply("OK bye"); return false; }
    if (v == "MOUSE") { reply(w.mouse_command(t)); return true; }
    if (v == "A11Y") { reply(w.a11y_command(t)); return true; }
    if (v == "PERCEIVE") { w.perceive(wl_output_name); reply(w.perceive_json()); return true; }
    if (v == "STATUS") { reply(w.body.status_json()); return true; }
    if (v == "SPEED") {
        if (t.size() == 2 && to_f(t[1], a) && a >= 0.05f && a <= 20.f) { w.opt.speed = a; reply("OK speed " + t[1]); }
        else reply("ERR usage: SPEED k   (0.05 .. 20)");
        return true;
    }
    if (v == "W2PX") {
        if (t.size() == 4 && to_f(t[1], a) && to_f(t[2], b) && to_f(t[3], c)) { char o[96]; std::snprintf(o, sizeof o, "OK %.1f %.1f", w.view.sx(c), w.view.sy(b)); reply(o); }
        else reply("ERR usage: W2PX x y z   (x is depth and does not change the screen position)");
        return true;
    }
    if (v == "PX2W") {
        if (t.size() == 3 && to_f(t[1], a) && to_f(t[2], b)) { char o[128]; std::snprintf(o, sizeof o, "OK z=%.3f y=%.3f (depth x is not defined by a screen point)", w.view.z_of(a), w.view.y_of(b)); reply(o); }
        else reply("ERR usage: PX2W px py");
        return true;
    }
    ik::Body::Result r = w.body.command(line);
    reply(std::string(r.ok ? "OK " : "ERR ") + r.msg);
    return true;
}

// ---------------------------------------------------------------------------------------------------- the demo harness
struct Demo {
    bool on = false; int next = 1;
    void tick(World& w) {
        if (!on || w.body.fallen() || w.body.stepping() || w.body.time() < 1.0f) return;
        int s = next, o = 1 - s; ik::Vec3 so = w.body.foot_sole(o);
        float z = so.z + 0.45f, zmax = w.view.z_of((float)w.view.W) - 0.8f;
        if (z > zmax) { on = false; reply("EVENT demo finished: reached the right edge"); return; }
        float x = s == 0 ? 0.10f : -0.10f, y = w.body.surface_at(x, z, so.y);
        char cmd[128]; std::snprintf(cmd, sizeof cmd, "FOOT %s %.3f %.3f %.3f", s == 0 ? "left" : "right", x, y, z);
        ik::Body::Result r = w.body.command(cmd);
        if (!r.ok) { on = false; reply("EVENT demo stopped: " + r.msg); return; }
        next = o;
    }
};

static void flush_events(World& w) { for (const std::string& e : w.body.take_events()) reply("EVENT " + e); }

// ---------------------------------------------------------------------------------------------------- Wayland
struct Out { wl_output* o = nullptr; uint32_t gname = 0; std::string name; };
struct Buf { wl_buffer* b = nullptr; uint32_t* data = nullptr; size_t size = 0; bool busy = false; Rect last; };

struct App {
    wl_display* dpy = nullptr; wl_registry* reg = nullptr; wl_compositor* comp = nullptr; wl_shm* shm = nullptr; zwlr_layer_shell_v1* shell = nullptr;
    std::vector<Out*> outs;
    wl_surface* surf = nullptr; zwlr_layer_surface_v1* ls = nullptr; wl_callback* frame_cb = nullptr;
    bool configured = false, closed = false, frame_ready = true; int W = 0, H = 0;
    std::string on_output;                              // name of the output the surface is on, once entered
    Buf buf[2]; Rect prev, body_rect; bool dirty_all = true;
    uint64_t last_hash = 0; bool have_hash = false;
    Clock::time_point frame_requested;
};

static void out_geometry(void*, wl_output*, int32_t, int32_t, int32_t, int32_t, int32_t, const char*, const char*, int32_t) {}
static void out_mode(void*, wl_output*, uint32_t, int32_t, int32_t, int32_t) {}
static void out_done(void*, wl_output*) {}
static void out_scale(void*, wl_output*, int32_t) {}
static void out_name(void* d, wl_output*, const char* n) { static_cast<Out*>(d)->name = n ? n : ""; }
static void out_desc(void*, wl_output*, const char*) {}

static void reg_global(void* d, wl_registry* r, uint32_t name, const char* iface, uint32_t ver) {
    App* a = static_cast<App*>(d); std::string i = iface;
    if (i == wl_compositor_interface.name) { if (ver >= 4) a->comp = (wl_compositor*)wl_registry_bind(r, name, &wl_compositor_interface, std::min(ver, 4u)); }   // v4 = damage_buffer
    else if (i == wl_shm_interface.name) a->shm = (wl_shm*)wl_registry_bind(r, name, &wl_shm_interface, 1);
    else if (i == zwlr_layer_shell_v1_interface.name) a->shell = (zwlr_layer_shell_v1*)wl_registry_bind(r, name, &zwlr_layer_shell_v1_interface, std::min(ver, 4u));
    else if (i == wl_output_interface.name) {
        Out* o = new Out; o->gname = name; o->o = (wl_output*)wl_registry_bind(r, name, &wl_output_interface, std::min(ver, 4u));
        static wl_output_listener ol; ol = {}; ol.geometry = out_geometry; ol.mode = out_mode; ol.done = out_done; ol.scale = out_scale; ol.name = out_name; ol.description = out_desc;
        wl_output_add_listener(o->o, &ol, o); a->outs.push_back(o);
    }
}
static void reg_remove(void*, wl_registry*, uint32_t) {}

static void buf_release(void* d, wl_buffer*) { static_cast<Buf*>(d)->busy = false; }

static void free_buffers(App& a) {
    for (Buf& b : a.buf) { if (b.b) wl_buffer_destroy(b.b); if (b.data) munmap(b.data, b.size); b = Buf{}; }
}
static bool make_buffers(App& a) {
    free_buffers(a);
    size_t stride = (size_t)a.W * 4, size = stride * (size_t)a.H;
    for (Buf& b : a.buf) {
        int fd = memfd_create("nerva-overlay", MFD_CLOEXEC);
        if (fd < 0 || ftruncate(fd, (off_t)size) != 0) { if (fd >= 0) close(fd); return false; }
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { close(fd); return false; }
        wl_shm_pool* pool = wl_shm_create_pool(a.shm, fd, (int32_t)size);
        b.b = wl_shm_pool_create_buffer(pool, 0, a.W, a.H, (int32_t)stride, WL_SHM_FORMAT_ARGB8888);
        wl_shm_pool_destroy(pool); close(fd);
        b.data = static_cast<uint32_t*>(p); b.size = size; b.busy = false; b.last = Rect{};   // memfd is zero-filled = transparent
        static wl_buffer_listener bl; bl.release = buf_release; wl_buffer_add_listener(b.b, &bl, &b);
    }
    a.prev = Rect{}; a.dirty_all = true; a.have_hash = false;
    return true;
}

static void ls_configure(void* d, zwlr_layer_surface_v1* s, uint32_t serial, uint32_t w, uint32_t h) {
    App* a = static_cast<App*>(d);
    zwlr_layer_surface_v1_ack_configure(s, serial);
    if (w == 0 || h == 0 || w > 16384 || h > 16384) return;
    if (!a->configured || (int)w != a->W || (int)h != a->H) { a->W = (int)w; a->H = (int)h; if (!make_buffers(*a)) { std::fprintf(stderr, "nerva-overlay: cannot allocate shm buffers\n"); a->closed = true; return; } }
    a->configured = true;
}
static void ls_closed(void* d, zwlr_layer_surface_v1*) { static_cast<App*>(d)->closed = true; }
static void surf_enter(void* d, wl_surface*, wl_output* o) { App* a = static_cast<App*>(d); for (Out* x : a->outs) if (x->o == o) a->on_output = x->name; }
static void surf_leave(void*, wl_surface*, wl_output*) {}
static void frame_done(void* d, wl_callback* cb, uint32_t) { App* a = static_cast<App*>(d); wl_callback_destroy(cb); a->frame_cb = nullptr; a->frame_ready = true; }

static uint64_t scene_hash(const World& w, const std::vector<DrawnBone>& bones, float opacity) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](int32_t v) { h ^= (uint32_t)v; h *= 1099511628211ull; };
    mix(w.hidden ? 1 : 0); mix((int32_t)std::lround(opacity * 40.f));
    mix(w.gate.active() ? (w.gate.focused() ? 2 : 1) : 0); if (w.gate.active()) { mix(w.gate.window().x); mix(w.gate.window().y); mix(w.gate.window().w); mix(w.gate.window().h); }
    if (w.hidden) return h;
    for (const DrawnBone& d : bones) { mix((int32_t)std::lround(d.ax * 4)); mix((int32_t)std::lround(d.ay * 4)); mix((int32_t)std::lround(d.bx * 4)); mix((int32_t)std::lround(d.by * 4)); }
    if (w.opt.debug) { for (const SurfaceDef& s : w.surfaces) { mix((int32_t)std::lround(s.z0 * 100)); mix((int32_t)std::lround(s.z1 * 100)); mix((int32_t)std::lround(s.y * 100)); } mix((int32_t)w.surfaces.size()); }
    return h;
}

static bool render_frame(App& a, World& w, float opacity) {
    Buf* b = nullptr; for (Buf& x : a.buf) if (!x.busy) { b = &x; break; }
    if (!b) return false;
    uint64_t hash = scene_hash(w, project_bones(w.body, w.view), opacity);
    if (a.have_hash && hash == a.last_hash && !a.dirty_all) return false;                 // nothing visibly changed: no draw, no commit
    Canvas cv{b->data, a.W, a.H, a.W};
    cv.clear(b->last);
    Style st = w.opt.style; st.opacity = opacity;
    Rect bbody = w.hidden ? Rect{} : draw_body(cv, w.body, w.view, st, &w.surfaces), bb = bbody;
    if (w.gate.active() && !w.hidden) bb = unite(bb, draw_grant_frame(cv, w.gate.window(), w.gate.focused(), st));   // the user always sees which window the AI may use
    a.body_rect = bbody;
    Rect dmg = a.dirty_all ? Rect{0, 0, a.W, a.H} : clip(unite(a.prev, bb), a.W, a.H);
    wl_surface_attach(a.surf, b->b, 0, 0);
    if (!dmg.empty()) wl_surface_damage_buffer(a.surf, dmg.x0, dmg.y0, dmg.x1 - dmg.x0, dmg.y1 - dmg.y0);
    a.frame_cb = wl_surface_frame(a.surf);
    static wl_callback_listener cl; cl.done = frame_done; wl_callback_add_listener(a.frame_cb, &cl, &a);
    a.frame_ready = false; a.frame_requested = Clock::now();
    wl_surface_commit(a.surf);
    b->busy = true; b->last = bb; a.prev = bb; a.dirty_all = false; a.last_hash = hash; a.have_hash = true;
    return true;
}

// ---------------------------------------------------------------------------------------------------- headless modes
static int run_render_test(const Options& o) {
    World w(o);
    w.set_size(1280, 420);
    // a fake window the body can walk onto, so the debug overlay and the surface mapping show up in the picture
    Window win; win.cls = "test"; win.x = 760; win.y = 420 - 40; win.w = 380; win.h = 300; win.floating = true;
    w.windows.push_back(win); w.refresh_surfaces();
    Demo d; d.on = true;
    for (int i = 0; i < 120 * 14 && !w.body.fallen(); ++i) { d.tick(w); w.body.update(1.f / 120.f); }
    flush_events(w);
    std::vector<uint32_t> px((size_t)w.view.W * w.view.H, 0);
    Canvas cv{px.data(), w.view.W, w.view.H, w.view.W};
    draw_body(cv, w.body, w.view, w.opt.style, &w.surfaces);
    if (!cv.write_ppm(o.render_test.c_str(), 214, 214, 218)) { std::fprintf(stderr, "cannot write %s\n", o.render_test.c_str()); return 1; }
    std::printf("rendered %s: pelvis z=%.2f y=%.2f fallen=%d\n", o.render_test.c_str(), w.body.pelvis().z, w.body.pelvis().y, (int)w.body.fallen());
    return 0;
}

static int run_perceive_once(const Options& o) {
    World w(o);
    HyprSnapshot snap; std::string err;
    if (!hypr_snapshot(snap, err)) { std::fprintf(stderr, "nerva-overlay: %s\n", err.c_str()); return 1; }
    const Monitor* m = nullptr;
    for (const Monitor& c : snap.monitors) if (!o.output.empty() ? c.name == o.output : c.focused) m = &c;
    if (!m && !snap.monitors.empty()) m = &snap.monitors[0];
    if (!m) { std::fprintf(stderr, "nerva-overlay: Hyprland reported no monitors\n"); return 1; }
    w.view.W = m->w; w.view.H = m->h;
    w.hypr_ok = true; w.monitor_name = m->name; w.monitor_scale = m->scale;
    w.windows = select_windows(*m, snap.windows); w.refresh_surfaces(true);
    for (int i = 0; i < 120; ++i) w.body.update(1.f / 120.f);
    std::puts(w.perceive_json().c_str());
    return 0;
}

// ---------------------------------------------------------------------------------------------------- the human's control socket
// $XDG_RUNTIME_DIR/nerva-overlay.sock, mode 0600, same-user peers only. This is the ONLY way to grant the pointer; the model's
// stdin channel cannot reach it. (A different process of your own user could, like it could use uinput directly: that is outside
// what this program can defend against.)
static std::string ctl_path() { const char* rt = std::getenv("XDG_RUNTIME_DIR"); return (rt && *rt) ? std::string(rt) + "/nerva-overlay.sock" : ""; }
static bool ctl_addr(const std::string& path, sockaddr_un& sa) { std::memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX; if (path.empty() || path.size() >= sizeof sa.sun_path) return false; std::memcpy(sa.sun_path, path.c_str(), path.size()); return true; }

static int ctl_client(const std::string& cmd) {
    sockaddr_un sa; std::string path = ctl_path();
    if (!ctl_addr(path, sa)) { std::fprintf(stderr, "nerva-overlay: XDG_RUNTIME_DIR is not set\n"); return 1; }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0 || connect(fd, (sockaddr*)&sa, sizeof sa) != 0) { std::fprintf(stderr, "nerva-overlay: no running overlay with --mouse (%s)\n", path.c_str()); if (fd >= 0) close(fd); return 1; }
    std::string line = cmd + "\n"; if (send(fd, line.c_str(), line.size(), MSG_NOSIGNAL) < 0) { close(fd); return 1; }
    pollfd p{fd, POLLIN, 0}; char buf[512]; std::string out;
    if (poll(&p, 1, 20000) > 0) { ssize_t n = recv(fd, buf, sizeof buf - 1, 0); if (n > 0) out.assign(buf, (size_t)n); }
    close(fd);
    std::fputs(out.empty() ? "ERR no reply\n" : out.c_str(), out.empty() ? stderr : stdout);
    return out.rfind("OK", 0) == 0 ? 0 : 1;
}
static int ctl_listen(std::string& err) {
    sockaddr_un sa; std::string path = ctl_path();
    if (!ctl_addr(path, sa)) { err = "XDG_RUNTIME_DIR is not set (or the path is too long)"; return -1; }
    int probe = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (probe >= 0 && connect(probe, (sockaddr*)&sa, sizeof sa) == 0) { close(probe); err = "another nerva-overlay is already running (" + path + ")"; return -1; }
    if (probe >= 0) close(probe);
    unlink(path.c_str());
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    mode_t old = umask(0177);
    bool ok = fd >= 0 && bind(fd, (sockaddr*)&sa, sizeof sa) == 0 && chmod(path.c_str(), 0600) == 0 && listen(fd, 4) == 0;
    umask(old);
    if (!ok) { err = std::string("cannot create ") + path + ": " + std::strerror(errno); if (fd >= 0) close(fd); return -1; }
    return fd;
}
static void ctl_serve(int lfd, World& w) {
    int c = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
    if (c < 0) return;
    ucred cr{}; socklen_t cl = sizeof cr;
    if (getsockopt(c, SOL_SOCKET, SO_PEERCRED, &cr, &cl) != 0 || cr.uid != geteuid()) { close(c); return; }
    pollfd p{c, POLLIN, 0}; char buf[256]; std::string cmd;
    if (poll(&p, 1, 300) > 0) { ssize_t n = recv(c, buf, sizeof buf - 1, 0); if (n > 0) cmd.assign(buf, (size_t)n); }
    while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) cmd.pop_back();
    for (char& ch : cmd) ch = (char)std::tolower((unsigned char)ch);
    std::string r = (cmd.empty() ? std::string("ERR empty request") : w.ctl_command(cmd)) + "\n";
    send(c, r.c_str(), r.size(), MSG_NOSIGNAL); close(c);
}

// ---------------------------------------------------------------------------------------------------- main
int main(int argc, char** argv) {
    Options opt; std::string err;
    if (!parse_args(argc, argv, opt, err)) { std::fprintf(stderr, "nerva-overlay: %s\n", err.c_str()); return 2; }
    if (opt.help) { usage(); return 0; }
    if (!opt.ctl.empty()) return ctl_client(opt.ctl);
    if (!opt.render_test.empty()) return run_render_test(opt);
    if (opt.perceive_once) return run_perceive_once(opt);

    std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal); std::signal(SIGPIPE, SIG_IGN);
    App app; World world(opt); Demo demo; demo.on = !opt.demo.empty();

    app.dpy = wl_display_connect(nullptr);
    if (!app.dpy) { std::fprintf(stderr, "nerva-overlay: cannot connect to a Wayland compositor (WAYLAND_DISPLAY=%s)\n", std::getenv("WAYLAND_DISPLAY") ? std::getenv("WAYLAND_DISPLAY") : "<unset>"); return 1; }
    app.reg = wl_display_get_registry(app.dpy);
    static wl_registry_listener rl; rl.global = reg_global; rl.global_remove = reg_remove; wl_registry_add_listener(app.reg, &rl, &app);
    wl_display_roundtrip(app.dpy); wl_display_roundtrip(app.dpy);                 // globals, then output names
    if (!app.comp || !app.shm) { std::fprintf(stderr, "nerva-overlay: the compositor lacks wl_compositor (v4+) or wl_shm\n"); return 1; }
    if (!app.shell) { std::fprintf(stderr, "nerva-overlay: the compositor does not offer zwlr_layer_shell_v1 (GNOME does not; Hyprland, Sway, KDE do)\n"); return 1; }

    wl_output* target = nullptr;
    if (!opt.output.empty()) {
        for (Out* o : app.outs) if (o->name == opt.output) target = o->o;
        if (!target) { std::fprintf(stderr, "nerva-overlay: no output named '%s' (have:", opt.output.c_str()); for (Out* o : app.outs) std::fprintf(stderr, " %s", o->name.c_str()); std::fprintf(stderr, ")\n"); return 1; }
    }
    app.surf = wl_compositor_create_surface(app.comp);
    static wl_surface_listener sl; sl = {}; sl.enter = surf_enter; sl.leave = surf_leave; wl_surface_add_listener(app.surf, &sl, &app);
    wl_region* empty = wl_compositor_create_region(app.comp);                       // empty input region: every click passes through
    wl_surface_set_input_region(app.surf, empty); wl_region_destroy(empty);
    app.ls = zwlr_layer_shell_v1_get_layer_surface(app.shell, app.surf, target, ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, "nerva-overlay");
    static zwlr_layer_surface_v1_listener lsl; lsl.configure = ls_configure; lsl.closed = ls_closed; zwlr_layer_surface_v1_add_listener(app.ls, &lsl, &app);
    zwlr_layer_surface_v1_set_anchor(app.ls, ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP | ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM | ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT | ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
    zwlr_layer_surface_v1_set_exclusive_zone(app.ls, -1);                           // ignore bars and other panels' reserved space
    zwlr_layer_surface_v1_set_keyboard_interactivity(app.ls, ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
    zwlr_layer_surface_v1_set_size(app.ls, 0, 0);
    wl_surface_commit(app.surf);

    { std::string e; if (!world.hide.load(HideList::default_path(), e)) std::fprintf(stderr, "nerva-overlay: %s (hide-list not loaded)\n", e.c_str()); }
    if (opt.a11y && world.a11y.helper_path.empty()) std::fprintf(stderr, "nerva-overlay: --a11y: tools/a11y_helper.py not found (use --a11y-helper PATH); A11Y commands will fail\n");
    int lfd = -1;
    if (opt.mouse || opt.a11y) {
        if (opt.mouse && opt.pointer == "uinput" && opt.no_hypr) { std::fprintf(stderr, "nerva-overlay: --mouse needs Hyprland perception (drop --no-hypr)\n"); return 2; }
        lfd = ctl_listen(err);
        if (lfd < 0) { std::fprintf(stderr, "nerva-overlay: %s\n", err.c_str()); return 1; }
        if (opt.mouse) std::fprintf(stderr, "nerva-overlay: AI pointer enabled but NOT granted. Focus the app you want to allow and run: nerva-overlay --ctl grant\n");
        if (opt.a11y) std::fprintf(stderr, "nerva-overlay: the model can READ every window except the ones you hide: nerva-overlay --ctl \"hide <app>\"\n");
    }
    bool stdin_open = true; std::string inbuf;
    auto now_s = []() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); };
    double last_tick = now_s(), next_tick = last_tick, next_perceive = 0, acc = 0;
    const double period = 1.0 / opt.fps; const float h = 1.f / 120.f;
    bool announced = false, quit = false;
    float opacity = 1.f; double next_cursor = 0;

    while (!g_stop && !app.closed && !quit) {
        while (wl_display_prepare_read(app.dpy) != 0) wl_display_dispatch_pending(app.dpy);
        if (wl_display_flush(app.dpy) < 0 && errno != EAGAIN) { wl_display_cancel_read(app.dpy); break; }
        pollfd fds[4]; int nf = 0, ci = -1, si = -1, ai = -1;
        fds[nf++] = {wl_display_get_fd(app.dpy), POLLIN, 0};
        if (lfd >= 0) { ci = nf; fds[nf++] = {lfd, POLLIN, 0}; }
        double wait = next_tick - now_s(); int ms = wait <= 0 ? 0 : (int)std::ceil(wait * 1000.0);
        bool read_stdin = stdin_open && announced;                                 // commands wait (in the pipe) until the real output size is known
        if (world.a11y.fd() >= 0) { ai = nf; fds[nf++] = {world.a11y.fd(), POLLIN, 0}; }
        if (read_stdin) { si = nf; fds[nf++] = {0, POLLIN, 0}; }
        int pr = poll(fds, (nfds_t)nf, ms);
        if (pr < 0 && errno != EINTR) { wl_display_cancel_read(app.dpy); break; }
        if (pr > 0 && (fds[0].revents & (POLLIN | POLLERR | POLLHUP))) { if (wl_display_read_events(app.dpy) < 0) break; } else wl_display_cancel_read(app.dpy);
        if (wl_display_dispatch_pending(app.dpy) < 0) break;

        if (ci >= 0 && pr > 0 && (fds[ci].revents & POLLIN)) ctl_serve(lfd, world);
        {
            std::vector<A11yClient::Reply> rs;
            if (ai >= 0 && pr > 0 && (fds[ai].revents & (POLLIN | POLLHUP | POLLERR))) world.a11y.on_readable(mono_now(), rs);
            world.a11y.on_tick(mono_now(), rs);
            for (const A11yClient::Reply& r : rs) reply("A11Y " + std::to_string(r.id) + " " + r.json);
        }
        if (si >= 0 && pr > 0 && (fds[si].revents & (POLLIN | POLLHUP))) {
            char tmp[4096]; ssize_t n = read(0, tmp, sizeof tmp);
            if (n <= 0) stdin_open = false;                                         // EOF: keep running (launched from a keybind or service)
            else {
                inbuf.append(tmp, (size_t)n);
                size_t nl;
                while ((nl = inbuf.find('\n')) != std::string::npos) {
                    std::string line = inbuf.substr(0, nl); inbuf.erase(0, nl + 1);
                    if (line.size() > 2048) { reply("ERR line too long"); continue; }
                    if (!handle_line(world, line, app.on_output)) { quit = true; break; }
                }
                if (inbuf.size() > 8192) { inbuf.clear(); reply("ERR line too long"); }
            }
        }

        double now = now_s();
        if (now < next_tick) continue;
        next_tick = std::max(next_tick + period, now - period);                     // pace without a death spiral after a stall
        if (!app.configured) continue;
        if (!announced) { world.set_size(app.W, app.H); announced = true; reply("EVENT overlay ready " + std::to_string(app.W) + "x" + std::to_string(app.H) + " ppm=" + std::to_string((int)world.view.ppm)); }
        else if (world.view.W != app.W || world.view.H != app.H) world.set_size(app.W, app.H);

        double el = std::min(now - last_tick, 0.1); last_tick = now;
        acc += el * world.opt.speed;
        if (world.hidden) acc = 0;                                                  // paused behind a fullscreen window
        for (int n = 0; acc >= h && n < 2400; ++n, acc -= h) { demo.tick(world); world.body.update(h); }
        if (acc > 1.0) acc = 0;
        flush_events(world);

        if (now >= next_perceive) {
            std::string before = world.perception_error;
            world.perceive(app.on_output); next_perceive = now + 0.25;
            if (opt.no_hypr || !world.hypr_ok) world.monitor_name = app.on_output;
            if (world.perception_error != before && !world.perception_error.empty()) reply("EVENT perception: " + world.perception_error);
        }
        if (now >= next_cursor) { world.poll_cursor(now); next_cursor = now + 0.08; }
        {   // ghost: fade out while the pointer is near the body so it never hides what the user is pointing at
            bool near = world.cursor_ok && !world.opt.no_ghost && near_rect(app.body_rect, world.cursor_x, world.cursor_y, 48.f);
            float target = near ? 0.18f : 1.f;
            opacity += (target - opacity) * (float)std::min(1.0, el * 10.0);
            if (std::fabs(opacity - target) < 0.02f) opacity = target;
        }
        if (!app.frame_ready && Clock::now() - app.frame_requested > std::chrono::milliseconds(500)) app.frame_ready = true;   // no callback (output off?) -- do not deadlock
        if (app.frame_ready) render_frame(app, world, opacity);
    }

    if (world.gate.active()) world.end_grant("overlay is closing");
    if (lfd >= 0) { close(lfd); unlink(ctl_path().c_str()); }
    free_buffers(app);
    if (app.frame_cb) wl_callback_destroy(app.frame_cb);
    if (app.ls) zwlr_layer_surface_v1_destroy(app.ls);
    if (app.surf) wl_surface_destroy(app.surf);
    wl_display_roundtrip(app.dpy);
    wl_display_disconnect(app.dpy);
    return 0;
}
