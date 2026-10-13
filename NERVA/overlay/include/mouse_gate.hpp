// mouse_gate.hpp -- the safety policy for the AI's pointer. Pure logic (no I/O), so every rule is unit-tested.
//
// Rules (the user's guideline):
//   * NOTHING works until a human grants ONE specific window. The grant comes from a separate control channel,
//     never from the model's command channel.
//   * The pointer may only act inside the granted window, only while that window is focused and visible, and only for a
//     limited time. The grant ends by itself when the window closes/hides/changes workspace, when a fullscreen window
//     takes over, or when it expires.
//   * Pointer only: move, left click, scroll. No keyboard, no typing, no drag. Right click is off unless enabled.
//   * Never granted: password managers, polkit/auth prompts, keyrings, ssh/gpg prompts (class deny-list).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "hypr.hpp"

namespace nerva {

struct GateConfig {
    double ttl_s = 300.0;                     // a grant lasts this long, then must be renewed by the human
    float margin_px = 8.f;                    // keep this far inside the window edges (never the title bar / resize border)
    double min_click_interval_s = 0.25;       // at most ~4 clicks per second
    float max_speed_px_s = 6000.f;            // faster pointer motion is refused
    bool allow_right_click = false;
    int max_scroll = 5;                       // wheel notches per command
    std::vector<std::string> deny_classes = {"polkit", "pinentry", "keepass", "bitwarden", "1password", "keyring", "kwallet", "seahorse",
                                             "gcr-prompter", "ssh-askpass", "lxqt-openssh", "gnome-keyring", "authenticator", "hyprpolkitagent"};
};

struct Verdict { bool ok; std::string why; };
inline Verdict allow() { return {true, ""}; }
inline Verdict deny(const std::string& w) { return {false, w}; }

class MouseGate {
public:
    GateConfig cfg;
    explicit MouseGate(GateConfig c = GateConfig()) : cfg(std::move(c)) {}

    bool active() const { return active_; }
    bool focused() const { return focused_; }
    void set_screen(int w, int h) { sw_ = w; sh_ = h; }                       // the output size: nothing off-screen is ever a valid target
    const std::string& address() const { return addr_; }
    const std::string& cls() const { return cls_; }
    // The name the MODEL is shown for the granted window: the real class, or "(hidden)" when the user hid that app from reading.
    void set_display(const std::string& n) { display_ = n; }
    const std::string& display() const { return display_; }
    const Window& window() const { return win_; }                         // output-relative, refreshed by update()
    double seconds_left(double now) const { return active_ ? std::max(0.0, expires_ - now) : 0.0; }
    int clicks() const { return clicks_; }

    // --- human side ---------------------------------------------------------------------------------------------------
    Verdict grant(const Window& w, double now) {
        std::string lc = lower(w.cls);
        for (const std::string& d : cfg.deny_classes) if (!d.empty() && lc.find(d) != std::string::npos) return deny("'" + w.cls + "' is on the never-grant list (password / authentication prompts)");
        if (!w.mapped || w.hidden) return deny("that window is not visible");
        if (w.fullscreen >= 2) return deny("fullscreen windows cannot be granted");
        if (w.w < 60 || w.h < 60) return deny("that window is too small");
        active_ = true; addr_ = w.address; cls_ = w.cls; display_ = w.cls; win_ = w; expires_ = now + cfg.ttl_s; focused_ = true;
        clicks_ = 0; have_last_ = false; last_click_ = -1e9; reason_.clear();
        return allow();
    }
    void revoke(const std::string& why) { if (active_) { active_ = false; reason_ = why; } }
    const std::string& last_revoke_reason() const { return reason_; }

    // --- perception pass (about every 250 ms). Returns the reason if the grant just ended, else "". ---------------------
    std::string update(const std::vector<Window>& visible, const std::string& focused_address, bool overlay_hidden, double now) {
        if (!active_) return "";
        const Window* w = nullptr;
        for (const Window& c : visible) if (c.address == addr_) { w = &c; break; }
        std::string why;
        if (!w) why = "the granted window closed, hid, or left this workspace";
        else if (overlay_hidden || w->fullscreen >= 2) why = "a fullscreen window is active";
        else if (now >= expires_) why = "the grant expired";
        if (!why.empty()) { revoke(why); return why; }
        win_ = *w; focused_ = (focused_address == addr_);
        return "";
    }

    // --- model side ---------------------------------------------------------------------------------------------------
    Verdict check_move(float x, float y, double now) {
        Verdict v = common(x, y); if (!v.ok) return v;
        if (have_last_) {
            double dt = std::max(now - last_move_t_, 0.016);
            double speed = std::hypot(x - last_x_, y - last_y_) / dt;
            if (speed > cfg.max_speed_px_s) { char b[120]; std::snprintf(b, sizeof b, "pointer speed %.0f px/s is above the limit %.0f px/s", speed, cfg.max_speed_px_s); return deny(b); }
        }
        return allow();
    }
    void note_move(float x, float y, double now) { last_x_ = x; last_y_ = y; last_move_t_ = now; have_last_ = true; }

    Verdict check_click(const std::string& button, float x, float y, double now) {
        if (button != "left" && !(button == "right" && cfg.allow_right_click)) return deny(button == "right" ? "right click is disabled (start with --allow-right-click)" : "only left" + std::string(cfg.allow_right_click ? "/right" : "") + " click is supported");
        Verdict v = common(x, y); if (!v.ok) return v;
        if (now - last_click_ < cfg.min_click_interval_s) { char b[100]; std::snprintf(b, sizeof b, "clicking too fast (limit one per %.2f s)", cfg.min_click_interval_s); return deny(b); }
        return allow();
    }
    void note_click(double now) { last_click_ = now; ++clicks_; }

    Verdict check_scroll(int n, float x, float y) {
        Verdict v = common(x, y); if (!v.ok) return v;
        if (n == 0 || std::abs(n) > cfg.max_scroll) { char b[80]; std::snprintf(b, sizeof b, "scroll must be 1..%d notches (positive = up, negative = down)", cfg.max_scroll); return deny(b); }
        return allow();
    }

    // inside the granted window, minus the safety margin
    bool inside(float x, float y) const {
        float m = cfg.margin_px;
        return x >= win_.x + m && x <= win_.x + win_.w - m && y >= win_.y + m && y <= win_.y + win_.h - m;
    }

private:
    bool active_ = false, focused_ = false, have_last_ = false; int sw_ = 0, sh_ = 0;
    std::string addr_, cls_, display_, reason_; Window win_;
    double expires_ = 0, last_click_ = -1e9, last_move_t_ = 0; float last_x_ = 0, last_y_ = 0; int clicks_ = 0;

    static std::string lower(std::string s) { for (char& c : s) c = (char)std::tolower((unsigned char)c); return s; }
    Verdict common(float x, float y) const {
        if (!active_) return deny("no app is granted: the user has not allowed the pointer to act in any window");
        if (!std::isfinite(x) || !std::isfinite(y)) return deny("coordinates are not finite");
        if (sw_ > 0 && (x < 0 || y < 0 || x >= (float)sw_ || y >= (float)sh_)) { char b[120]; std::snprintf(b, sizeof b, "(%.0f, %.0f) is outside the screen (%d x %d)", x, y, sw_, sh_); return deny(b); }
        if (!focused_) return deny("the granted window is not focused");
        if (!inside(x, y)) {
            char b[200]; std::snprintf(b, sizeof b, "(%.0f, %.0f) is outside the granted window '%s' (x %d..%d, y %d..%d, %.0f px safety margin)", x, y, display_.c_str(),
                                       win_.x, win_.x + win_.w, win_.y, win_.y + win_.h, cfg.margin_px);
            return deny(b);
        }
        return allow();
    }
};

}  // namespace nerva
