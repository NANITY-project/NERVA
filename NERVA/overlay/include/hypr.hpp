// hypr.hpp -- Hyprland perception: monitors and windows over Hyprland's request socket (`j/monitors`, `j/clients`).
// Read-only: it only ever sends those two query strings. Wayland hides window geometry from clients, so this is the
// compositor-specific part (Sway/KDE would need their own reader behind the same two structs).
#pragma once
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "mini_json.hpp"

namespace nerva {

struct Monitor {
    int id = 0; std::string name;
    int x = 0, y = 0;                  // position in the global layout (logical px)
    int w = 0, h = 0;                  // LOGICAL size (physical / scale, swapped when rotated)
    float scale = 1.f;
    int active_ws = 0, special_ws = 0; // special_ws != 0 while a scratchpad is shown
    bool focused = false;
};

struct Window {
    std::string address, cls, title;
    int x = 0, y = 0, w = 0, h = 0;    // global layout px (as reported); select_windows() makes them output-relative
    int ws = 0, monitor = -1, fullscreen = 0, pid = 0;
    bool mapped = true, hidden = false, floating = false, pinned = false;
};

inline bool parse_monitors(const std::string& text, std::vector<Monitor>& out, std::string& err) {
    Json j; if (!json_parse(text, j, &err)) return false;
    if (j.t != Json::Arr) { err = "monitors: expected an array"; return false; }
    std::vector<Monitor> v;
    for (const Json& m : j.a) {
        if (m.t != Json::Obj) { err = "monitors: expected objects"; return false; }
        Monitor o; o.id = (int)m.num("id"); o.name = m.str("name");
        o.x = (int)m.num("x"); o.y = (int)m.num("y");
        double pw = m.num("width"), ph = m.num("height"), sc = m.num("scale", 1.0);
        if (!(sc > 0.05 && sc < 20.0)) sc = 1.0;
        int tr = (int)m.num("transform"); if (tr & 1) std::swap(pw, ph);
        o.scale = (float)sc; o.w = (int)std::lround(pw / sc); o.h = (int)std::lround(ph / sc);
        if (const Json* a = m.get("activeWorkspace")) o.active_ws = (int)a->num("id");
        if (const Json* s = m.get("specialWorkspace")) o.special_ws = (int)s->num("id");
        o.focused = m.boolean("focused");
        if (o.w > 0 && o.h > 0) v.push_back(o);
    }
    out = std::move(v); return true;
}

inline bool parse_clients(const std::string& text, std::vector<Window>& out, std::string& err) {
    Json j; if (!json_parse(text, j, &err)) return false;
    if (j.t != Json::Arr) { err = "clients: expected an array"; return false; }
    std::vector<Window> v;
    for (const Json& c : j.a) {
        if (c.t != Json::Obj) { err = "clients: expected objects"; return false; }
        const Json* at = c.get("at"); const Json* sz = c.get("size");
        if (!at || !sz || at->t != Json::Arr || sz->t != Json::Arr || at->a.size() < 2 || sz->a.size() < 2) continue;   // not a window we can place
        Window w; w.address = c.str("address"); w.cls = c.str("class"); w.title = c.str("title");
        w.x = (int)at->a[0].n; w.y = (int)at->a[1].n; w.w = (int)sz->a[0].n; w.h = (int)sz->a[1].n;
        if (const Json* ws = c.get("workspace")) w.ws = (int)ws->num("id");
        w.pid = (int)c.num("pid"); w.monitor = (int)c.num("monitor", -1); { const Json* fs = c.get("fullscreen"); w.fullscreen = !fs ? 0 : fs->t == Json::Bool ? (fs->b ? 2 : 0) : (int)fs->n; }   // 0 none, 1 maximised, 2 fullscreen (old versions: bool)
        w.mapped = c.boolean("mapped", true); w.hidden = c.boolean("hidden"); w.floating = c.boolean("floating"); w.pinned = c.boolean("pinned");
        v.push_back(std::move(w));
    }
    out = std::move(v); return true;
}

// Windows a character could stand on, for one monitor, with output-relative coordinates.
inline std::vector<Window> select_windows(const Monitor& mon, const std::vector<Window>& all) {
    std::vector<Window> r;
    for (const Window& w : all) {
        if (!w.mapped || w.hidden || w.monitor != mon.id || w.w <= 0 || w.h <= 0) continue;
        bool on_ws = w.ws == mon.active_ws || (mon.special_ws != 0 && w.ws == mon.special_ws) || w.pinned;
        if (!on_ws) continue;
        Window o = w; o.x -= mon.x; o.y -= mon.y; r.push_back(std::move(o));
    }
    return r;
}

// A window in real fullscreen (games, video) on the monitor's visible workspace. Maximised (1) does not count.
inline bool any_fullscreen(const std::vector<Window>& selected) { for (const Window& w : selected) if (w.fullscreen >= 2) return true; return false; }

inline bool parse_cursor(const std::string& text, float& x, float& y) {
    Json j; if (!json_parse(text, j) || j.t != Json::Obj || !j.get("x") || !j.get("y")) return false;
    x = (float)j.num("x"); y = (float)j.num("y"); return std::isfinite(x) && std::isfinite(y);
}

// `j/activewindow` -> the focused window's address ("" when none is focused)
inline std::string parse_active_address(const std::string& text) {
    Json j; if (!json_parse(text, j) || j.t != Json::Obj) return "";
    return j.str("address");
}

// ---- IPC ----------------------------------------------------------------------------------------------------------------
inline std::string hypr_socket_path() {
    if (const char* o = std::getenv("NERVA_HYPR_SOCKET")) return o;       // tests / unusual setups
    const char* sig = std::getenv("HYPRLAND_INSTANCE_SIGNATURE");
    if (!sig || !*sig) return "";
    std::string s = sig;
    if (s.find('/') != std::string::npos) return "";                      // never build a path from something that is not a plain name
    const char* rt = std::getenv("XDG_RUNTIME_DIR");
    if (rt && *rt) { std::string p = std::string(rt) + "/hypr/" + s + "/.socket.sock"; if (access(p.c_str(), F_OK) == 0) return p; }
    return "/tmp/hypr/" + s + "/.socket.sock";
}

// One request, one reply (the server closes after replying). Only the two query strings below are ever sent.
inline bool hypr_query(const std::string& path, const char* request, std::string& reply, std::string& err, int timeout_ms = 300) {
    reply.clear();
    if (path.empty()) { err = "Hyprland not detected (HYPRLAND_INSTANCE_SIGNATURE is unset)"; return false; }
    sockaddr_un sa{}; sa.sun_family = AF_UNIX;
    if (path.size() >= sizeof sa.sun_path) { err = "socket path too long"; return false; }
    std::snprintf(sa.sun_path, sizeof sa.sun_path, "%s", path.c_str());
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { err = "socket() failed"; return false; }
    bool ok = false;
    do {
        if (::connect(fd, (sockaddr*)&sa, sizeof sa) != 0) { err = "cannot connect to " + path; break; }
        size_t n = std::char_traits<char>::length(request);
        if (::send(fd, request, n, MSG_NOSIGNAL) != (ssize_t)n) { err = "send failed"; break; }
        char buf[16384]; const size_t cap = 8u << 20; bool eof = false;
        while (!eof) {
            pollfd p{fd, POLLIN, 0}; int pr = ::poll(&p, 1, timeout_ms);
            if (pr == 0) { err = "timed out waiting for Hyprland"; break; }
            if (pr < 0) { if (errno == EINTR) continue; err = "poll failed"; break; }
            ssize_t r = ::recv(fd, buf, sizeof buf, 0);
            if (r < 0) { if (errno == EINTR) continue; err = "recv failed"; break; }
            if (r == 0) eof = true; else { reply.append(buf, (size_t)r); if (reply.size() > cap) { err = "reply too large"; break; } }
        }
        ok = eof;
    } while (false);
    ::close(fd);
    return ok;
}

struct HyprSnapshot { std::vector<Monitor> monitors; std::vector<Window> windows; };

inline bool hypr_snapshot(HyprSnapshot& s, std::string& err) {
    std::string path = hypr_socket_path(), a, b;
    if (!hypr_query(path, "j/monitors", a, err)) return false;
    if (!hypr_query(path, "j/clients", b, err)) return false;
    HyprSnapshot t;
    if (!parse_monitors(a, t.monitors, err)) { err = "monitors: " + err; return false; }
    if (!parse_clients(b, t.windows, err)) { err = "clients: " + err; return false; }
    s = std::move(t); return true;
}

}  // namespace nerva
