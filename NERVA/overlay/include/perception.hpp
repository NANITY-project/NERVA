// perception.hpp -- what the model may SEE on screen: the user's hide-list, request building for the accessibility helper, and the
// helper process client. Reading is separate from acting: the model may read any window the user has not hidden; it may only
// POINT in the one window the user granted (mouse_gate.hpp).
#pragma once
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include "hypr.hpp"
#include "mini_json.hpp"
#include "scene.hpp"

extern char** environ;

namespace nerva {

// ------------------------------------------------------------------------------------------------------------ the hide-list
// Windows whose class contains one of these (case-insensitive) are invisible to the model: never read, never named. Their
// rectangle still exists as a floor for the body. Only the user changes it (control socket / the config file), never the model.
class HideList {
public:
    std::vector<std::string> user;
    std::vector<std::string> builtin;                                         // password managers, auth prompts, keyrings: always hidden

    static std::string lower(std::string s) { for (char& c : s) c = (char)std::tolower((unsigned char)c); return s; }
    static bool valid(const std::string& s) {
        if (s.empty() || s.size() > 64) return false;
        for (unsigned char c : s) if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-' || c == '+' || c == ' ')) return false;
        return true;
    }
    bool matches(const std::string& cls) const {
        std::string c = lower(cls);
        for (const std::string& p : builtin) if (!p.empty() && c.find(p) != std::string::npos) return true;
        for (const std::string& p : user) if (!p.empty() && c.find(p) != std::string::npos) return true;
        return false;
    }
    std::vector<std::string> all() const { std::vector<std::string> a = builtin; a.insert(a.end(), user.begin(), user.end()); return a; }
    bool add(const std::string& s, std::string& err) {
        std::string p = lower(s);
        if (!valid(p)) { err = "a hide pattern is 1-64 characters: letters, digits, . _ - + and space"; return false; }
        for (const std::string& u : user) if (u == p) return true;
        user.push_back(p); return true;
    }
    bool remove(const std::string& s) {
        std::string p = lower(s);
        for (size_t i = 0; i < user.size(); ++i) if (user[i] == p) { user.erase(user.begin() + (long)i); return true; }
        return false;
    }

    static std::string default_path() {
        if (const char* o = std::getenv("NERVA_CONFIG")) return o;
        const char* x = std::getenv("XDG_CONFIG_HOME"); const char* h = std::getenv("HOME");
        if (x && *x) return std::string(x) + "/nerva/perception.json";
        if (h && *h) return std::string(h) + "/.config/nerva/perception.json";
        return "";
    }
    bool load(const std::string& path, std::string& err) {
        path_ = path; user.clear();
        if (path.empty()) return true;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return true;                                                  // no file yet is fine
        std::string text; char b[4096]; size_t n;
        while ((n = std::fread(b, 1, sizeof b, f)) > 0 && text.size() < (1u << 20)) text.append(b, n);
        std::fclose(f);
        Json j; std::string e;
        if (!json_parse(text, j, &e) || j.t != Json::Obj) { err = "cannot read " + path + ": " + (e.empty() ? "not an object" : e); return false; }
        if (const Json* h = j.get("hide")) if (h->t == Json::Arr) for (const Json& v : h->a) if (v.t == Json::Str) { std::string p = lower(v.s); if (valid(p)) user.push_back(p); }
        return true;
    }
    bool save(std::string& err) const {
        if (path_.empty()) return true;
        size_t sl = path_.rfind('/');
        if (sl != std::string::npos) { std::string dir = path_.substr(0, sl); mkdir(dir.c_str(), 0700); }
        std::string tmp = path_ + ".tmp", body = "{\"hide\":[";
        for (size_t i = 0; i < user.size(); ++i) body += (i ? "," : "") + std::string("\"") + json_escape(user[i]) + "\"";
        body += "]}\n";
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) { err = std::string("cannot write ") + tmp + ": " + std::strerror(errno); return false; }
        bool ok = ::write(fd, body.c_str(), body.size()) == (ssize_t)body.size(); ::close(fd);
        if (!ok || std::rename(tmp.c_str(), path_.c_str()) != 0) { err = "cannot save " + path_; return false; }
        return true;
    }
private:
    std::string path_;
};

// ------------------------------------------------------------------------------------------------------------ A11Y command
struct A11yArgs { std::string op, target, query; int depth = 0; };

// A11Y TREE [target] [depth] | TEXT [target] | FIND <words> [IN target] | FOCUSED [target] | APPS | STATUS
inline bool parse_a11y_args(const std::vector<std::string>& t, A11yArgs& a, std::string& err) {
    a = A11yArgs();
    if (t.size() < 2) { err = "usage: A11Y TREE|TEXT|FIND|FOCUSED|APPS|STATUS ..."; return false; }
    a.op = HideList::lower(t[1]);
    if (a.op == "tree") {
        if (t.size() > 4) { err = "usage: A11Y TREE [target] [depth]"; return false; }
        if (t.size() >= 3) { char* e; long d = std::strtol(t[2].c_str(), &e, 10); if (!*e && !t[2].empty()) a.depth = (int)d; else a.target = t[2]; }
        if (t.size() == 4) { char* e; long d = std::strtol(t[3].c_str(), &e, 10); if (*e || t[3].empty()) { err = "depth must be a number 1..30"; return false; } a.depth = (int)d; }
        if (a.depth != 0 && (a.depth < 1 || a.depth > 30)) { err = "depth must be 1..30"; return false; }
    } else if (a.op == "text" || a.op == "focused") {
        if (t.size() > 3) { err = "usage: A11Y " + a.op + " [target]"; return false; }
        if (t.size() == 3) a.target = t[2];
    } else if (a.op == "find") {
        size_t n = t.size();
        if (n >= 5 && HideList::lower(t[n - 2]) == "in") { a.target = t[n - 1]; n -= 2; }
        for (size_t i = 2; i < n; ++i) a.query += (i > 2 ? " " : "") + t[i];
        if (a.query.empty() || a.query.size() > 120) { err = "usage: A11Y FIND <text> [IN target]   (1-120 characters)"; return false; }
    } else if (a.op == "apps" || a.op == "status") {
        if (t.size() != 2) { err = "usage: A11Y " + a.op; return false; }
    } else { err = "unknown A11Y command (TREE, TEXT, FIND, FOCUSED, APPS, STATUS)"; return false; }
    a.target = HideList::lower(a.target);
    return true;
}

// target: "" / "focused" | "granted" | "all" | a window address (0x...) | a class substring. Hidden windows are never returned.
inline std::vector<const Window*> resolve_targets(const std::vector<Window>& wins, const HideList& hide, const std::string& spec, const std::string& focused_addr,
                                                  const std::string& granted_addr, bool default_all, std::string& err, int& hidden_skipped, size_t cap = 16) {
    std::vector<const Window*> out; hidden_skipped = 0;
    std::string s = spec.empty() ? (default_all ? "all" : "focused") : spec;
    auto take = [&](const Window& w) { if (hide.matches(w.cls)) ++hidden_skipped; else if (out.size() < cap) out.push_back(&w); };
    if (s == "focused" || s == "granted") {
        const std::string& want = s == "focused" ? focused_addr : granted_addr;
        if (want.empty()) { err = s == "focused" ? "no window is focused" : "no window is granted"; return out; }
        for (const Window& w : wins) if (w.address == want) take(w);
        if (out.empty() && hidden_skipped == 0) err = std::string("the ") + s + " window is not on this monitor/workspace";
    } else if (s == "all") {
        for (const Window& w : wins) if (w.pid > 0) take(w);
        if (out.empty() && hidden_skipped == 0) err = "no windows";
    } else {
        for (const Window& w : wins) if (w.address == s || HideList::lower(w.cls).find(s) != std::string::npos) take(w);
        if (out.empty() && hidden_skipped == 0) err = "no window matches '" + s + "'";
    }
    if (out.empty() && hidden_skipped > 0) err = s == "all" ? "no readable windows" : "that window is hidden by the user";
    return out;
}

inline std::string build_a11y_request(long id, const A11yArgs& a, const std::vector<const Window*>& targets, const HideList& hide, int dx, int dy) {
    std::string j = "{\"id\":" + std::to_string(id) + ",\"op\":\"" + a.op + "\",\"targets\":[";
    for (size_t i = 0; i < targets.size(); ++i) {
        const Window& w = *targets[i]; char b[96];
        std::snprintf(b, sizeof b, ",\"pid\":%d,\"ox\":%d,\"oy\":%d,\"dx\":%d,\"dy\":%d}", w.pid, w.x, w.y, dx, dy);
        j += std::string(i ? "," : "") + "{\"address\":\"" + json_escape(w.address) + "\",\"cls\":\"" + json_escape(w.cls) + "\",\"title\":\"" + json_escape(w.title) + "\"" + b;
    }
    j += "],\"hide\":[";
    std::vector<std::string> h = hide.all();
    for (size_t i = 0; i < h.size(); ++i) j += (i ? "," : "") + std::string("\"") + json_escape(h[i]) + "\"";
    j += "]";
    if (a.depth > 0) j += ",\"max_depth\":" + std::to_string(a.depth);
    if (!a.query.empty()) j += ",\"query\":\"" + json_escape(a.query) + "\"";
    return j + "}";
}

// ------------------------------------------------------------------------------------------------------------ helper process
// One persistent python helper, one request in flight, a short queue. Never blocks the main loop; a hung or dead helper is killed
// and replaced, and every waiting request gets an error reply.
class A11yClient {
public:
    struct Reply { long id; std::string json; };
    std::string helper_path; double timeout_s = 8.0; size_t max_queue = 4;

    static std::string find_helper(const std::string& override_path) {
        std::vector<std::string> c;
        if (!override_path.empty()) c.push_back(override_path);
        if (const char* e = std::getenv("NERVA_A11Y_HELPER")) c.push_back(e);
        char exe[4096]; ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) { exe[n] = 0; std::string d = exe; d = d.substr(0, d.rfind('/')); c.push_back(d + "/../tools/a11y_helper.py"); c.push_back(d + "/tools/a11y_helper.py"); }
        c.push_back("/usr/share/nerva/a11y_helper.py");
        for (const std::string& p : c) if (::access(p.c_str(), R_OK) == 0) return p;
        return "";
    }
    ~A11yClient() { stop(); }
    int fd() const { return out_; }
    bool running() const { return pid_ > 0; }
    size_t pending() const { return queue_.size() + (inflight_ ? 1 : 0); }

    // Queue a request. False (with err) if the queue is full or the helper cannot be started.
    bool submit(long id, const std::string& json, double now, std::string& err) {
        if (pending() >= max_queue + 1) { err = "accessibility reader is busy (too many requests waiting)"; return false; }
        if (!running() && !start(now, err)) return false;
        queue_.push_back({id, json});
        pump(now);
        return true;
    }
    // Call when fd() is readable. Appends finished replies.
    void on_readable(double now, std::vector<Reply>& out) {
        char b[65536]; ssize_t n;
        while ((n = ::read(out_, b, sizeof b)) > 0) { buf_.append(b, (size_t)n); if (buf_.size() > (4u << 20)) { fail_all("the accessibility reader produced too much data", out); stop(); return; } }
        if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR)) { fail_all("the accessibility reader stopped (see overlay/README.md: python3-gobject / at-spi2-core)", out); stop(); return; }
        size_t nl;
        while ((nl = buf_.find('\n')) != std::string::npos) {
            std::string line = buf_.substr(0, nl); buf_.erase(0, nl + 1);
            if (!inflight_) continue;                                          // unsolicited line: ignore
            out.push_back({cur_id_, line}); inflight_ = false;
        }
        pump(now);
    }
    // Call every tick: a request that takes too long kills the helper.
    void on_tick(double now, std::vector<Reply>& out) {
        if (inflight_ && now - sent_at_ > timeout_s) { fail_all("the accessibility read timed out (an application did not answer)", out); stop(); }
    }
    void stop() {
        if (pid_ > 0) { ::kill(pid_, SIGKILL); int st; ::waitpid(pid_, &st, 0); pid_ = -1; }
        if (in_ >= 0) { ::close(in_); in_ = -1; }
        if (out_ >= 0) { ::close(out_); out_ = -1; }
        buf_.clear(); inflight_ = false;
    }

private:
    struct Req { long id; std::string json; };
    pid_t pid_ = -1; int in_ = -1, out_ = -1; std::string buf_; std::deque<Req> queue_; bool inflight_ = false; long cur_id_ = 0; double sent_at_ = 0, last_spawn_ = -1e9;

    bool start(double now, std::string& err) {
        if (helper_path.empty()) { err = "the accessibility helper (tools/a11y_helper.py) was not found; set --a11y-helper PATH"; return false; }
        if (now - last_spawn_ < 2.0) { err = "the accessibility reader just failed; retrying in a moment"; return false; }
        last_spawn_ = now;
        int to[2], from[2];
        if (::pipe2(to, O_CLOEXEC) != 0) { err = "pipe failed"; return false; }
        if (::pipe2(from, O_CLOEXEC) != 0) { ::close(to[0]); ::close(to[1]); err = "pipe failed"; return false; }
        posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, to[0], 0); posix_spawn_file_actions_adddup2(&fa, from[1], 1);
        if (!std::getenv("NERVA_A11Y_DEBUG")) posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        char* argv[] = {(char*)"python3", (char*)helper_path.c_str(), nullptr};
        pid_t p; int rc = posix_spawnp(&p, "python3", &fa, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&fa);
        ::close(to[0]); ::close(from[1]);
        if (rc != 0) { ::close(to[1]); ::close(from[0]); err = std::string("cannot start python3: ") + std::strerror(rc); return false; }
        pid_ = p; in_ = to[1]; out_ = from[0];
        ::fcntl(out_, F_SETFL, ::fcntl(out_, F_GETFL) | O_NONBLOCK);
        return true;
    }
    void pump(double now) {
        if (inflight_ || queue_.empty() || in_ < 0) return;
        Req r = queue_.front(); queue_.pop_front();
        std::string line = r.json + "\n";
        size_t off = 0;
        while (off < line.size()) { ssize_t n = ::write(in_, line.data() + off, line.size() - off); if (n < 0) { if (errno == EINTR) continue; if (errno == EAGAIN) { pollfd p{in_, POLLOUT, 0}; ::poll(&p, 1, 200); continue; } break; } off += (size_t)n; }
        inflight_ = true; cur_id_ = r.id; sent_at_ = now;
    }
    void fail_all(const std::string& why, std::vector<Reply>& out) {
        std::string j = "{\"ok\":false,\"error\":\"" + json_escape(why) + "\"}";
        if (inflight_) { out.push_back({cur_id_, j}); inflight_ = false; }
        for (const Req& r : queue_) out.push_back({r.id, j});
        queue_.clear();
    }
};

}  // namespace nerva
