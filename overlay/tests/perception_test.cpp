// perception_test.cpp -- hide-list, A11Y command parsing, target resolution, request building, and the helper process client.
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>
#include "perception.hpp"
using namespace nerva;

static int g_fail = 0, g_n = 0;
#define CHECK(c, ...) do { ++g_n; if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static Window W(const char* addr, const char* cls, int pid, int x = 0, int y = 0) { Window w; w.address = addr; w.cls = cls; w.pid = pid; w.x = x; w.y = y; w.w = 400; w.h = 300; w.title = std::string("title of ") + cls; return w; }
static std::vector<std::string> toks(const std::string& s) { std::vector<std::string> t; size_t i = 0; while (i < s.size()) { while (i < s.size() && s[i] == ' ') ++i; size_t j = i; while (j < s.size() && s[j] != ' ') ++j; if (j > i) t.push_back(s.substr(i, j - i)); i = j; } return t; }

static void test_hidelist() {
    std::puts("[hide-list]");
    HideList h; h.builtin = {"keepass", "polkit"};
    CHECK(h.matches("org.keepassxc.KeePassXC") && h.matches("Polkit-gnome") && !h.matches("firefox"), "built-in protections match case-insensitively");
    std::string e;
    CHECK(h.add("Firefox", e) && h.matches("firefox") && h.matches("org.mozilla.Firefox"), "user pattern (any case) hides matching classes");
    CHECK(h.add("firefox", e) && h.user.size() == 1, "adding twice does not duplicate");
    for (const char* bad : {"", "a\"b", "x\\y", "a/b", "<script>", "\n", "a;b", "$(rm)", "dröp"}) CHECK(!h.add(bad, e), "pattern '%s' is rejected", bad);
    CHECK(!h.add(std::string(65, 'a'), e), "too long");
    CHECK(h.remove("FIREFOX") && !h.matches("firefox") && !h.remove("firefox"), "remove");
    CHECK(!h.remove("keepass") && h.matches("keepassxc"), "built-in protections cannot be removed");
    std::string path = "/tmp/nerva_hide_test_" + std::to_string(getpid()) + "/perception.json";
    h.add("discord", e); h.add("mail", e);
    HideList g; g.builtin = h.builtin; CHECK(g.load(path, e), "missing file is fine"); g.add("discord", e); g.add("mail", e);
    CHECK(g.save(e), "save creates the directory (%s)", e.c_str());
    struct stat st; CHECK(stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "config file is mode 0600");
    HideList r; r.builtin = h.builtin; CHECK(r.load(path, e) && r.user.size() == 2 && r.matches("Discord"), "round trip");
    FILE* f = std::fopen(path.c_str(), "wb"); std::fputs("{\"hide\":[\"ok\", 5, \"bad\\\"q\", \"\", \"fine one\"]}", f); std::fclose(f);
    HideList q; CHECK(q.load(path, e) && q.user.size() == 2 && q.user[0] == "ok" && q.user[1] == "fine one", "load skips non-strings and invalid patterns");
    f = std::fopen(path.c_str(), "wb"); std::fputs("{{{ not json", f); std::fclose(f);
    HideList z; CHECK(!z.load(path, e) && !e.empty() && z.user.empty(), "corrupt config -> error, nothing hidden by it (built-ins still apply): %s", e.c_str());
    unlink(path.c_str()); rmdir(path.substr(0, path.rfind('/')).c_str());
}

static void test_args() {
    std::puts("[A11Y argument parsing]");
    A11yArgs a; std::string e;
    CHECK(parse_a11y_args(toks("A11Y TREE"), a, e) && a.op == "tree" && a.target.empty() && a.depth == 0, "TREE");
    CHECK(parse_a11y_args(toks("A11Y tree kitty"), a, e) && a.target == "kitty", "TREE target");
    CHECK(parse_a11y_args(toks("A11Y TREE 5"), a, e) && a.depth == 5 && a.target.empty(), "TREE depth only");
    CHECK(parse_a11y_args(toks("A11Y TREE Kitty 7"), a, e) && a.target == "kitty" && a.depth == 7, "TREE target depth");
    CHECK(!parse_a11y_args(toks("A11Y TREE kitty 99"), a, e) && !parse_a11y_args(toks("A11Y TREE kitty x"), a, e) && !parse_a11y_args(toks("A11Y TREE a 1 2"), a, e), "bad depths / extra args");
    CHECK(parse_a11y_args(toks("A11Y FIND save changes"), a, e) && a.query == "save changes" && a.target.empty(), "FIND query words");
    CHECK(parse_a11y_args(toks("A11Y find save changes in firefox"), a, e) && a.query == "save changes" && a.target == "firefox", "FIND ... IN target");
    CHECK(parse_a11y_args(toks("A11Y FIND log in"), a, e) && a.query == "log in" && a.target.empty(), "'in' as part of the query is not a target");
    CHECK(!parse_a11y_args(toks("A11Y FIND"), a, e) && !parse_a11y_args(toks("A11Y FIND " + std::string(130, 'x')), a, e), "FIND needs 1-120 chars");
    CHECK(parse_a11y_args(toks("A11Y TEXT granted"), a, e) && a.target == "granted" && parse_a11y_args(toks("A11Y FOCUSED"), a, e) && parse_a11y_args(toks("A11Y APPS"), a, e) && parse_a11y_args(toks("A11Y STATUS"), a, e), "TEXT / FOCUSED / APPS / STATUS");
    CHECK(!parse_a11y_args(toks("A11Y"), a, e) && !parse_a11y_args(toks("A11Y CLICK ok"), a, e) && !parse_a11y_args(toks("A11Y APPS now"), a, e), "unknown / malformed");
}

static void test_targets() {
    std::puts("[target resolution: hidden windows are never returned]");
    std::vector<Window> ws{W("0x1", "kitty", 11), W("0x2", "firefox", 12), W("0x3", "org.keepassxc.KeePassXC", 13), W("0x4", "discord", 14), W("0x5", "nopid", 0)};
    HideList h; h.builtin = {"keepass"}; std::string e; h.add("discord", e); int sk = 0;
    auto r = resolve_targets(ws, h, "", "0x2", "", true, e, sk);
    std::string got; for (auto* w : r) got += w->cls + ","; CHECK(got == "kitty,firefox,", "all: only visible windows with a pid (%s)", got.c_str());
    r = resolve_targets(ws, h, "", "0x2", "", false, e, sk); CHECK(r.size() == 1 && r[0]->cls == "firefox", "default target = the focused window");
    r = resolve_targets(ws, h, "", "0x3", "", false, e, sk); CHECK(r.empty() && e.find("hidden") != std::string::npos, "focused window hidden by the user -> nothing, with a reason (%s)", e.c_str());
    r = resolve_targets(ws, h, "", "", "", false, e, sk); CHECK(r.empty() && e == "no window is focused", "no focus");
    r = resolve_targets(ws, h, "granted", "0x2", "0x1", false, e, sk); CHECK(r.size() == 1 && r[0]->cls == "kitty", "granted target");
    r = resolve_targets(ws, h, "granted", "0x2", "", false, e, sk); CHECK(r.empty() && e == "no window is granted", "no grant");
    r = resolve_targets(ws, h, "kit", "", "", false, e, sk); CHECK(r.size() == 1 && r[0]->address == "0x1", "class substring");
    r = resolve_targets(ws, h, "0x2", "", "", false, e, sk); CHECK(r.size() == 1 && r[0]->cls == "firefox", "address");
    r = resolve_targets(ws, h, "keepass", "", "", false, e, sk); CHECK(r.empty() && sk == 1, "asking for a hidden class by name yields nothing");
    r = resolve_targets(ws, h, "disc", "", "", false, e, sk); CHECK(r.empty() && e.find("hidden by the user") != std::string::npos, "user-hidden window by name");
    r = resolve_targets(ws, h, "zzz", "", "", false, e, sk); CHECK(r.empty() && e.find("no window matches") != std::string::npos, "no match");
    std::vector<Window> many; for (int i = 0; i < 40; ++i) many.push_back(W(("0x" + std::to_string(i)).c_str(), "app", 100 + i));
    r = resolve_targets(many, h, "", "", "", true, e, sk); CHECK(r.size() == 16, "at most 16 windows per request");
}

static void test_request() {
    std::puts("[request building]");
    std::vector<Window> ws{W("0x1", "kit\"ty", 11, 100, 200)}; ws[0].title = "a \"quoted\" \\ title\nline2";
    HideList h; h.builtin = {"keepass"}; std::string e; h.add("discord", e);
    A11yArgs a; parse_a11y_args(toks("A11Y FIND save \"it\""), a, e);
    std::vector<const Window*> t{&ws[0]};
    std::string j = build_a11y_request(42, a, t, h, -3, 5);
    Json p; CHECK(json_parse(j, p, &e), "valid JSON even with quotes/newlines in titles and the query: %s", e.c_str());
    CHECK(p.num("id") == 42 && p.str("op") == "find" && p.str("query") == "save \"it\"", "id, op, query survive");
    const Json* tg = p.get("targets"); CHECK(tg && tg->a.size() == 1 && tg->a[0].num("pid") == 11 && tg->a[0].num("ox") == 100 && tg->a[0].num("oy") == 200 && tg->a[0].num("dx") == -3 && tg->a[0].num("dy") == 5, "target geometry and offset");
    CHECK(tg->a[0].str("title") == ws[0].title && tg->a[0].str("cls") == "kit\"ty", "strings round-trip exactly");
    const Json* hh = p.get("hide"); CHECK(hh && hh->a.size() == 2 && hh->a[0].s == "keepass" && hh->a[1].s == "discord", "the hide-list is passed along as a second line of defence");
    A11yArgs d; parse_a11y_args(toks("A11Y TREE kitty 6"), d, e); CHECK(json_parse(build_a11y_request(1, d, t, h, 0, 0), p, &e) && p.num("max_depth") == 6, "depth passed");
    std::vector<Window> cl; CHECK(parse_clients("[{\"at\":[1,2],\"size\":[3,4],\"pid\":4242,\"class\":\"x\"}]", cl, e) && cl[0].pid == 4242, "Hyprland pid is parsed");
}

static void drain(A11yClient& c, std::vector<A11yClient::Reply>& out, double secs) {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (c.fd() >= 0) { pollfd p{c.fd(), POLLIN, 0}; if (poll(&p, 1, 20) > 0) c.on_readable(now, out); } else usleep(20000);
        c.on_tick(now, out);
    }
}
static double nowd() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void test_client() {
    std::puts("[helper process client]");
    std::string fake = std::string(getenv("NERVA_TEST_DIR_OVERLAY") ? getenv("NERVA_TEST_DIR_OVERLAY") : ".") + "/tests/fake_helper.py";
    A11yClient c; c.helper_path = fake; c.timeout_s = 1.0; std::string e; std::vector<A11yClient::Reply> out;
    CHECK(c.submit(1, "{\"id\":1,\"op\":\"tree\"}", nowd(), e) && c.running(), "starts the helper on first request (%s)", e.c_str());
    drain(c, out, 1.0); CHECK(out.size() == 1 && out[0].id == 1 && out[0].json.find("\"echo\": \"tree\"") != std::string::npos, "reply arrives asynchronously (%zu)", out.size());
    out.clear();
    for (int i = 2; i <= 4; ++i) c.submit(i, "{\"id\":" + std::to_string(i) + ",\"op\":\"x\"}", nowd(), e);
    drain(c, out, 1.0); CHECK(out.size() == 3 && out[0].id == 2 && out[1].id == 3 && out[2].id == 4, "queued requests are answered in order");
    out.clear();
    bool refused = false; for (int i = 0; i < 8; ++i) if (!c.submit(10 + i, "{\"id\":" + std::to_string(10 + i) + ",\"op\":\"slow\"}", nowd(), e)) { refused = true; break; }
    CHECK(refused && e.find("busy") != std::string::npos, "a full queue refuses with a reason (%s)", e.c_str());
    drain(c, out, 2.0); bool timed = false; for (auto& r : out) if (r.json.find("timed out") != std::string::npos) timed = true;
    CHECK(timed && !c.running(), "a helper that hangs is killed after the timeout and every waiting request gets an error (%zu replies)", out.size());
    out.clear(); usleep(2200000);
    CHECK(c.submit(30, "{\"id\":30,\"op\":\"tree\"}", nowd(), e), "a new request after the failure restarts the helper (%s)", e.c_str());
    drain(c, out, 1.0); CHECK(out.size() == 1 && out[0].json.find("\"ok\": true") != std::string::npos, "...and works again");
    out.clear(); c.submit(31, "{\"id\":31,\"op\":\"die\"}", nowd(), e); drain(c, out, 1.0);
    CHECK(out.size() == 1 && out[0].id == 31 && out[0].json.find("stopped") != std::string::npos && !c.running(), "a helper that crashes -> an error reply, not a hang");
    out.clear(); usleep(2200000);
    A11yClient bad; bad.helper_path = ""; CHECK(!bad.submit(1, "{}", nowd(), e) && e.find("not found") != std::string::npos, "no helper file -> clear error");
    A11yClient j; j.helper_path = fake; j.timeout_s = 1.0; j.submit(5, "{\"id\":5,\"op\":\"junk\"}", nowd(), e); drain(j, out, 1.0);
    CHECK(out.size() >= 1 && out[0].id == 5, "a garbage line from the helper is passed along as the reply and does not wedge the client (%zu)", out.size());
}

int main() { test_hidelist(); test_args(); test_targets(); test_request(); test_client(); std::printf("\n%d checks, %d failed\n", g_n, g_fail); return g_fail ? 1 : 0; }
