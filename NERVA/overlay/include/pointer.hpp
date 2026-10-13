// pointer.hpp -- the pointer itself: a uinput device that can ONLY move an absolute pointer, press two buttons and turn a wheel
// (the device is created without any keyboard key bits, so it cannot type), plus calibration and verified motion.
//
// The compositor decides how a virtual absolute pointer maps onto the screens, and we cannot know it in advance, so instead of
// assuming we MEASURE: probe three positions, read the real cursor back (Hyprland `j/cursorpos`), fit px = a*abs + b per axis.
// Every move is then verified against the real cursor, every click re-checks that nobody (the user!) moved the pointer since.
#pragma once
#include <cerrno>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <fcntl.h>
#include <linux/input-event-codes.h>
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace nerva {

struct PointerBackend {
    virtual ~PointerBackend() {}
    virtual bool abs_move(int ax, int ay, std::string& err) = 0;          // 0..65535 on both axes
    virtual bool button(int code, bool down, std::string& err) = 0;       // BTN_LEFT / BTN_RIGHT only
    virtual bool wheel(int notches, std::string& err) = 0;
    virtual int startup_delay_ms() const { return 0; }                    // time the compositor needs to notice a new device
};

class UinputPointer : public PointerBackend {
public:
    static std::unique_ptr<UinputPointer> open(std::string& err) {
        int fd = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) { err = std::string("cannot open /dev/uinput: ") + std::strerror(errno) + " (needs a udev rule giving your user access; see overlay/README.md)"; return nullptr; }
        auto set = [&](unsigned long req, int v) { return ::ioctl(fd, req, v) == 0; };
        bool ok = set(UI_SET_EVBIT, EV_KEY) && set(UI_SET_KEYBIT, BTN_LEFT) && set(UI_SET_KEYBIT, BTN_RIGHT)      // two buttons, no keyboard keys at all
               && set(UI_SET_EVBIT, EV_ABS) && set(UI_SET_ABSBIT, ABS_X) && set(UI_SET_ABSBIT, ABS_Y)
               && set(UI_SET_EVBIT, EV_REL) && set(UI_SET_RELBIT, REL_WHEEL) && set(UI_SET_EVBIT, EV_SYN);
        for (int axis : {ABS_X, ABS_Y}) { uinput_abs_setup a; std::memset(&a, 0, sizeof a); a.code = (unsigned short)axis; a.absinfo.minimum = 0; a.absinfo.maximum = 65535; ok = ok && ::ioctl(fd, UI_ABS_SETUP, &a) == 0; }
        uinput_setup us; std::memset(&us, 0, sizeof us); us.id.bustype = BUS_VIRTUAL; us.id.vendor = 0x4e56; us.id.product = 0x0001; us.id.version = 1;
        std::snprintf(us.name, sizeof us.name, "nerva-pointer");
        ok = ok && ::ioctl(fd, UI_DEV_SETUP, &us) == 0 && ::ioctl(fd, UI_DEV_CREATE) == 0;
        if (!ok) { err = std::string("uinput setup failed: ") + std::strerror(errno); ::close(fd); return nullptr; }
        return std::unique_ptr<UinputPointer>(new UinputPointer(fd));
    }
    ~UinputPointer() override { if (fd_ >= 0) { ::ioctl(fd_, UI_DEV_DESTROY); ::close(fd_); } }
    int startup_delay_ms() const override { return 500; }
    bool abs_move(int ax, int ay, std::string& err) override { return emit(EV_ABS, ABS_X, ax, err) && emit(EV_ABS, ABS_Y, ay, err) && sync(err); }
    bool button(int code, bool down, std::string& err) override {
        if (code != BTN_LEFT && code != BTN_RIGHT) { err = "button not allowed"; return false; }
        return emit(EV_KEY, (unsigned short)code, down ? 1 : 0, err) && sync(err);
    }
    bool wheel(int n, std::string& err) override { return emit(EV_REL, REL_WHEEL, n, err) && sync(err); }
private:
    explicit UinputPointer(int fd) : fd_(fd) {}
    int fd_ = -1;
    bool emit(unsigned short type, unsigned short code, int value, std::string& err) {
        input_event ev; std::memset(&ev, 0, sizeof ev); ev.type = type; ev.code = code; ev.value = value;
        if (::write(fd_, &ev, sizeof ev) != (ssize_t)sizeof ev) { err = std::string("uinput write failed: ") + std::strerror(errno); return false; }
        return true;
    }
    bool sync(std::string& err) { return emit(EV_SYN, SYN_REPORT, 0, err); }
};

struct AxisFit {
    double a = 1, b = 0; bool ok = false;                                   // global_px = a * abs + b
    int to_abs(double px) const { double v = (px - b) / a; return (int)std::lround(v < 0 ? 0 : v > 65535 ? 65535 : v); }
};
inline bool fit_axis(double abs1, double px1, double abs2, double px2, AxisFit& f) {
    if (std::fabs(abs2 - abs1) < 1000 || !std::isfinite(px1) || !std::isfinite(px2)) return false;
    double a = (px2 - px1) / (abs2 - abs1);
    if (!(a > 0.0) || a * 65535.0 < 200.0 || a * 65535.0 > 100000.0) return false;        // implausible: a screen is 200..100000 px wide
    f.a = a; f.b = px1 - a * abs1; f.ok = true; return true;
}

class PointerController {
public:
    using CursorFn = std::function<bool(float&, float&)>;                  // real cursor, global layout px
    using SleepFn = std::function<void(int)>;
    PointerController(std::unique_ptr<PointerBackend> be, CursorFn cur, SleepFn slp) : be_(std::move(be)), cursor_(std::move(cur)), sleep_(std::move(slp)) {}

    bool ready() const { return calibrated_; }
    bool has_position() const { return have_pos_; }
    float x() const { return px_; }                                         // last verified pointer position (global px)
    float y() const { return py_; }
    double tolerance_px = 2.5;                                              // a verified move must land this close
    double takeover_pause_s = 3.0;

    bool calibrate(std::string& err) {
        if (be_->startup_delay_ms() > 0) sleep_(be_->startup_delay_ms());
        float ox, oy; if (!cursor_(ox, oy)) { err = "cannot read the real cursor position (Hyprland j/cursorpos)"; return false; }
        float px[3], py[3]; const int probe[3] = {16384, 49152, 32768};
        float lx = ox, ly = oy;
        for (int i = 0; i < 3; ++i) {
            if (!be_->abs_move(probe[i], probe[i], err)) return false;
            if (!wait_changed(lx, ly, px[i], py[i])) { err = "the pointer did not move: the compositor ignores the new device (is it enabled in your input config?)"; return false; }
            lx = px[i]; ly = py[i];
        }
        AxisFit fx, fy;
        if (!fit_axis(probe[0], px[0], probe[1], px[1], fx) || !fit_axis(probe[0], py[0], probe[1], py[1], fy)) { err = "the pointer mapping is implausible (not increasing / absurd span)"; return false; }
        double ex = std::fabs(fx.a * probe[2] + fx.b - px[2]), ey = std::fabs(fy.a * probe[2] + fy.b - py[2]);
        if (ex > 3.0 || ey > 3.0) { err = "the pointer mapping is not linear (third probe off by " + std::to_string((int)std::max(ex, ey)) + " px)"; return false; }
        fx_ = fx; fy_ = fy; calibrated_ = true; have_pos_ = false;
        std::string e2; float rx, ry;                                        // put the user's cursor back where it was
        int ax = fx_.to_abs(ox), ay = fy_.to_abs(oy);
        if (be_->abs_move(ax, ay, e2)) { sleep_(20); (void)wait_near(ox, oy, rx, ry); }
        return true;
    }

    // Move so the real cursor ends within tolerance of (gx,gy). Verified against the real cursor.
    bool move_to(float gx, float gy, double now, std::string& err) {
        if (!calibrated_) { err = "pointer not calibrated"; return false; }
        if (!untouched(now, err)) return false;
        int ax = fx_.to_abs(gx), ay = fy_.to_abs(gy);
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!be_->abs_move(ax, ay, err)) return false;
            float cx, cy;
            if (wait_near(gx, gy, cx, cy)) { px_ = cx; py_ = cy; have_pos_ = true; return true; }
            if (!cursor_(cx, cy)) { err = "cannot read the cursor"; return false; }
            if (attempt == 1) { err = "pointer landed " + std::to_string((int)std::hypot(gx - cx, gy - cy)) + " px from the target (calibration is off; grant again to recalibrate)"; calibrated_ = false; return false; }
            ax = clamp_abs(ax + (int)std::lround((gx - cx) / fx_.a)); ay = clamp_abs(ay + (int)std::lround((gy - cy) / fy_.a));   // one correction from the observed error
        }
        return false;
    }
    bool click(const std::string& button, double now, std::string& err) {
        if (!calibrated_ || !have_pos_) { err = "move the pointer first"; return false; }
        if (!untouched(now, err)) return false;
        int code = button == "right" ? BTN_RIGHT : BTN_LEFT;
        if (!be_->button(code, true, err)) return false;
        sleep_(25);
        std::string e2; bool up = be_->button(code, false, e2);              // always release, whatever happened
        if (!up) { err = "button release failed: " + e2; return false; }
        return true;
    }
    bool scroll(int notches, double now, std::string& err) {
        if (!calibrated_ || !have_pos_) { err = "move the pointer first"; return false; }
        if (!untouched(now, err)) return false;
        return be_->wheel(notches, err);
    }
    static int clamp_abs(int v) { return v < 0 ? 0 : v > 65535 ? 65535 : v; }
    // current cursor from the compositor (global px), for status
    bool read_cursor(float& x, float& y) { return cursor_(x, y); }

private:
    std::unique_ptr<PointerBackend> be_; CursorFn cursor_; SleepFn sleep_;
    AxisFit fx_, fy_; bool calibrated_ = false, have_pos_ = false; float px_ = 0, py_ = 0; double paused_until_ = 0;

    // The user touched the mouse since we last placed it: pause the AI's pointer for a few seconds.
    bool untouched(double now, std::string& err) {
        if (now < paused_until_) { err = "paused: the user moved the mouse (resumes in " + std::to_string((int)std::ceil(paused_until_ - now)) + " s)"; return false; }
        if (!have_pos_) return true;
        float cx, cy; if (!cursor_(cx, cy)) { err = "cannot read the cursor"; return false; }
        if (std::hypot(cx - px_, cy - py_) > 6.0) { paused_until_ = now + takeover_pause_s; have_pos_ = false; err = "the user moved the mouse: the AI's pointer is paused for " + std::to_string((int)takeover_pause_s) + " s"; return false; }
        return true;
    }
    bool wait_changed(float fromx, float fromy, float& cx, float& cy) {
        for (int i = 0; i < 40; ++i) { sleep_(8); if (cursor_(cx, cy) && std::hypot(cx - fromx, cy - fromy) > 0.5) { sleep_(8); cursor_(cx, cy); return true; } }
        return false;
    }
    bool wait_near(float gx, float gy, float& cx, float& cy) {
        for (int i = 0; i < 40; ++i) { sleep_(i == 0 ? 2 : 8); if (cursor_(cx, cy) && std::hypot(cx - gx, cy - gy) <= tolerance_px) return true; }
        return false;
    }
};

}  // namespace nerva
