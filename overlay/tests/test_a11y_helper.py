#!/usr/bin/env python3
"""Tests for tools/a11y_helper.py.
 1. unit tests on fake accessibility trees (always run)
 2. a REAL GTK3 window read through the real AT-SPI bus under a headless Sway (skipped when sway / gi / dbus are missing)
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import a11y_helper as H

passed = failed = skipped = 0
def check(c, m):
    global passed, failed
    if c: passed += 1
    else: failed += 1; print("  FAIL:", m)

class Fake(H.Node):
    def __init__(self, role, name="", ext=None, states=("showing", "enabled"), text=None, actions=(), kids=()):
        self._r, self._n, self._e, self._s, self._t, self._a, self._k = role, name, ext, set(states), text, list(actions), list(kids)
    def role(self): return self._r
    def name(self): return self._n
    def states(self): return set(self._s)
    def extents(self): return self._e
    def child_count(self): return len(self._k)
    def child(self, i): return self._k[i]
    def text(self, limit): return None if self._t is None else (self._t[:limit], len(self._t))
    def actions(self): return self._a

LONG = "lorem ipsum " * 500
def sample_frame(fx=0, fy=0):
    def r(x, y, w, h): return (fx + x, fy + y, w, h)
    return Fake("frame", "My App", (fx, fy, 400, 300), kids=[
        Fake("filler", "", r(0, 0, 400, 300), kids=[
            Fake("label", "Hello", r(10, 10, 100, 20)),
            Fake("push button", "OK", r(10, 40, 80, 30), actions=["click"], states=("showing", "enabled", "focusable")),
            Fake("push button", "Disabled one", r(100, 40, 80, 30), states=("showing",)),
            Fake("text", "Search", r(10, 80, 200, 30), text="abc def", states=("showing", "enabled", "editable", "focused")),
            Fake("text", "Account password", r(10, 120, 200, 30), text="hunter2-SECRET"),
            Fake("password text", "x", r(10, 160, 200, 30), text="hunter3-SECRET"),
            Fake("push button", "Not visible", r(10, 200, 80, 30), states=("enabled",)),
            Fake("push button", "Scrolled away", r(10, 900, 80, 30)),
            Fake("document web", "Page", r(220, 10, 170, 280), text=LONG),
            Fake("panel", "", r(0, 250, 50, 20), kids=[Fake("label", "Nested label", r(2, 252, 40, 10))]),
        ])])

def req(op, targets, **kw): return dict({"id": 1, "op": op, "targets": targets}, **kw)
T = lambda **kw: dict({"pid": 7, "title": "My App", "address": "0x1", "cls": "myapp", "ox": 100, "oy": 300}, **kw)
def run(frame, r, name="myapp", pid=7): return H.handle(r, lambda: [(pid, name, Fake("application", name, kids=[frame]))])

def unit():
    print("[unit: fake trees]")
    r = run(sample_frame(), req("tree", [T()])); w = r["windows"][0]; els = w["elements"]
    by = {e["name"]: e for e in els}
    check(r["ok"] and w["size"] == [400, 300], "tree ok")
    check(by["Hello"]["rect"] == [110, 310, 100, 20] and by["Hello"]["c"] == [160, 320], "rectangles are in OUTPUT pixels: window origin + offset (%s)" % by["Hello"]["rect"])
    check(by["OK"]["act"] == ["click"], "actions are listed (names only)")
    check("disabled" in by["Disabled one"]["st"] and "focused" in by["Search"]["st"] and "editable" in by["Search"]["st"], "states: disabled / focused / editable")
    check(by["Search"]["text"] == "abc def", "editable text is readable")
    check("Not visible" not in by and "Scrolled away" not in by, "elements that are not showing, or outside the window, are skipped")
    check([e["name"] for e in els if e["role"] == "filler"] == [] and "Nested label" in by, "empty containers are dropped, their children kept")
    check(by["Nested label"]["p"] == -1, "...and re-parented to the nearest kept ancestor")
    check(by["Account password"].get("secret") is True and "text" not in by["Account password"], "a field NAMED like a password is treated as secret")
    pw = [e for e in els if e["role"] == "password text"][0]; check(pw.get("secret") and "text" not in pw, "a password-text role is secret")
    blob = json.dumps(run(sample_frame(), req("tree", [T()]))) + json.dumps(run(sample_frame(), req("text", [T()]))) + json.dumps(run(sample_frame(), req("find", [T()], query="hunter"))) + json.dumps(run(sample_frame(), req("focused", [T()])))
    check("hunter2" not in blob and "hunter3" not in blob and "SECRET" not in blob, "secrets never appear in ANY output (tree, text, find, focused)")
    doc = by["Page"]; check(len(doc["text"]) <= 161 and doc["chars"] == len(LONG), "long text is cut to the limit and reports its real size")
    r2 = run(sample_frame(), req("tree", [T()], text_limit=40)); check(len([e for e in r2["windows"][0]["elements"] if e["name"] == "Page"][0]["text"]) <= 41, "text_limit is honoured")
    # frame origin not at 0 (client-side shadows) and a manual correction
    r3 = run(sample_frame(-20, -20), req("tree", [T()])); b3 = {e["name"]: e for e in r3["windows"][0]["elements"]}
    check(b3["Hello"]["rect"] == [110, 310, 100, 20], "frame origin offset (CSD shadow) is subtracted")
    r4 = run(sample_frame(), req("tree", [T(dx=-6, dy=-4)])); check({e["name"]: e for e in r4["windows"][0]["elements"]}["Hello"]["rect"] == [104, 306, 100, 20], "dx/dy correction applied")
    r5 = run(sample_frame(), req("tree", [T()], max_nodes=3)); check(r5["windows"][0]["truncated"] and len(r5["windows"][0]["elements"]) <= 3, "node cap -> truncated")
    deep = Fake("frame", "My App", (0, 0, 400, 300)); cur = deep
    for i in range(40):
        nxt = Fake("panel", "level%d" % i, (0, 0, 400, 300)); cur._k = [nxt]; cur = nxt
    r6 = run(deep, req("tree", [T()], max_depth=5)); check(r6["windows"][0]["truncated"] and len(r6["windows"][0]["elements"]) <= 6, "depth cap -> truncated")

    print("[unit: operations]")
    f = run(sample_frame(), req("find", [T()], query="ok")); check(f["ok"] and f["matches"][0]["name"] == "OK" and f["matches"][0]["c"] == [150, 355], "find 'ok' -> centre of the OK button for the pointer (%s)" % f["matches"])
    check(run(sample_frame(), req("find", [T()], query="abc"))["matches"][0]["name"] == "Search", "find matches text content too")
    check(run(sample_frame(), req("find", [T()], query="hello", role="label"))["matches"][0]["name"] == "Hello" and run(sample_frame(), req("find", [T()], query="hello", role="push button"))["matches"] == [], "find with a role filter")
    check(run(sample_frame(), req("find", [T()], query="zzz"))["matches"] == [], "no match -> empty list")
    fo = run(sample_frame(), req("focused", [T()]))["focused"]; check(fo and fo["name"] == "Search" and fo["text"] == "abc def", "focused element and its text")
    tx = run(sample_frame(), req("text", [T()]))["windows"][0]["items"]; check([i["text"] for i in tx][:3] == ["Hello", "OK", "Disabled one"], "text op lists readable text in order (%s)" % [i["text"] for i in tx][:3])
    tx2 = run(sample_frame(), req("text", [T()], total_text_limit=30))["windows"][0]["items"]; check(any("note" in i for i in tx2) and sum(len(i.get("text", "")) for i in tx2) <= 30, "text op is size-capped")
    rh = run(sample_frame(), req("tree", [T(cls="org.keepassxc.KeePassXC")], hide=["keepass"])); check(rh["windows"][0]["error"] == "hidden by the user" and "elements" not in rh["windows"][0], "an app the user hid is not read, even if asked")
    rn = run(sample_frame(), req("tree", [T(pid=99, cls="other")])); check("no accessibility tree" in rn["windows"][0]["error"], "unknown pid -> explains (is accessibility enabled?)")
    rb = run(sample_frame(), req("tree", [T(pid=99, cls="MyApp")])); check("elements" in rb["windows"][0], "falls back to matching the application name with the window class")
    two = lambda: [(7, "myapp", H.Node.__new__(Fake))]
    a = Fake("application", "myapp", kids=[Fake("frame", "Doc A", (0, 0, 300, 200)), Fake("frame", "Doc B", (0, 0, 300, 200), kids=[Fake("push button", "InB", (5, 5, 50, 20))])])
    rt = H.handle(req("tree", [T(title="Doc B")]), lambda: [(7, "myapp", a)]); check(rt["windows"][0]["elements"][0]["name"] == "InB", "several windows in one process: the title picks the right one")
    ra = H.handle(req("tree", [T(title="Something else")]), lambda: [(7, "myapp", a)]); check("cannot tell" in ra["windows"][0]["error"], "ambiguous -> says so instead of guessing")
    check(run(sample_frame(), {"id": 3, "op": "nope", "targets": []})["ok"] is False and run(sample_frame(), {"id": 3, "op": "tree", "targets": "x"})["ok"] is False and run(sample_frame(), {"id": 3, "op": "tree", "targets": [{}] * 40})["ok"] is False, "bad ops / bad targets are refused")
    boom = H.handle(req("tree", [T()]), lambda: (_ for _ in ()).throw(RuntimeError("bus gone"))); check(boom["ok"] is False and "bus gone" in boom["error"], "an exception becomes an error reply, never a crash")

    print("[unit: process protocol and read-only audit]")
    p = subprocess.Popen([sys.executable, os.path.join(TOOLS, "a11y_helper.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write("garbage\n[1,2]\n" + json.dumps({"id": 9, "op": "nope", "targets": []}) + "\n"); p.stdin.flush()
    l1, l2, l3 = p.stdout.readline(), p.stdout.readline(), p.stdout.readline(); p.stdin.close(); p.wait(5)
    check(json.loads(l1)["ok"] is False and json.loads(l2)["ok"] is False and json.loads(l3)["id"] == 9, "garbage lines get error replies and the process keeps serving")
    src = open(os.path.join(TOOLS, "a11y_helper.py")).read()
    for bad in ("do_action", "set_current_value", "set_text_contents", "insert_text", "delete_text", "grab_focus", "generate_mouse_event", "generate_keyboard_event", "set_extents", "set_position", "set_caret", "subprocess", "os.system", "socket", "open("):
        check(bad not in re.sub(r'""".*?"""', "", src, flags=re.S), "the helper never uses %s" % bad)

# ------------------------------------------------------------------------------------------------------------------ real AT-SPI
def real_inner():
    global passed, failed
    d = os.environ["NERVA_TEST_DIR"]
    subprocess.Popen(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
    app = subprocess.Popen([sys.executable, os.path.join(HERE, "gtk_testapp.py"), os.path.join(d, "alloc.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(os.path.join(d, "alloc.json")): break
        time.sleep(0.1)
    time.sleep(0.3); al = json.load(open(os.path.join(d, "alloc.json")))
    h = subprocess.Popen([sys.executable, os.path.join(TOOLS, "a11y_helper.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    def ask(**kw):
        h.stdin.write(json.dumps(dict({"id": 1}, **kw)) + "\n"); h.stdin.flush(); return json.loads(h.stdout.readline())
    tgt = {"pid": al["pid"], "title": "Nerva Test Window", "address": "0x1", "cls": "gtk_testapp.py", "ox": 200, "oy": 300}
    r = ask(op="apps", targets=[tgt, dict(tgt, pid=999999, cls="nothing", address="0x2"), dict(tgt, cls="keepassxc", address="0x3")], hide=["keepass"])
    check(r["ok"] and r["apps"][0]["readable"] and not r["apps"][1]["readable"] and r["apps"][2].get("error") == "hidden by the user" and "readable" not in r["apps"][2], "[real] apps: readable / not readable / hidden-by-user, and nothing about other applications")
    r = ask(op="tree", targets=[tgt]); els = r["windows"][0]["elements"]; by = {e["name"]: e for e in els}
    check(r["ok"] and {"Hello accessibility world", "Save changes", "Enable feature", "Search box", "Account password", "Notes"} <= set(by), "[real] all widgets found (%s)" % sorted(by))
    for nm, key in (("Hello accessibility world", "label"), ("Save changes", "button"), ("Enable feature", "check"), ("Search box", "entry"), ("Account password", "password"), ("Notes", "notes")):
        x, y, w, hh = al[key]; check(by[nm]["rect"] == [200 + x, 300 + y, w, hh], "[real] %s rect %s == window origin + its real allocation %s" % (nm, by[nm]["rect"], [200 + x, 300 + y, w, hh]))
    check(by["Save changes"]["role"] == "push button" and "click" in by["Save changes"].get("act", []), "[real] the button is a push button with a click action")
    check("checked" in by["Enable feature"].get("st", []), "[real] checkbox state: checked")
    check(by["Search box"]["text"] == "typed text here", "[real] entry text readable")
    check(by["Notes"]["text"].startswith("First line of notes"), "[real] text view content readable")
    check(by["Account password"].get("secret") is True and "text" not in by["Account password"], "[real] the password field is marked secret, content absent")
    blob = json.dumps(ask(op="tree", targets=[tgt])) + json.dumps(ask(op="text", targets=[tgt])) + json.dumps(ask(op="find", targets=[tgt], query="hunter")) + json.dumps(ask(op="find", targets=[tgt], query="SECRET"))
    check("hunter2" not in blob and "SECRET" not in blob, "[real] the real password text never leaves the app")
    f = ask(op="find", targets=[tgt], query="save"); x, y, w, hh = al["button"]
    check(f["matches"] and f["matches"][0]["c"] == [200 + x + w // 2, 300 + y + hh // 2], "[real] find 'save' -> the centre of the real button on screen")
    t = ask(op="text", targets=[tgt]); texts = " | ".join(i["text"] for i in t["windows"][0]["items"]); check("Hello accessibility world" in texts and "Second line of notes" in texts and "typed text here" in texts, "[real] text op reads the screen's text")
    r = ask(op="tree", targets=[dict(tgt, pid=999999, cls="nothing")]); check("no accessibility tree" in r["windows"][0]["error"], "[real] a process without an a11y tree is explained")
    h.stdin.close(); h.wait(5); app.terminate()

def real_outer():
    global skipped
    need = [shutil.which(x) for x in ("sway", "dbus-run-session")]
    try:
        import gi; gi.require_version("Atspi", "2.0"); gi.require_version("Gtk", "3.0"); from gi.repository import Atspi, Gtk  # noqa
    except Exception: need.append(None)
    if not all(need) or not os.path.exists("/usr/libexec/at-spi-bus-launcher"):
        print("[real AT-SPI] skipped: needs sway, dbus-run-session, python3-gi with Atspi + Gtk3, at-spi2-core"); skipped += 1; return
    print("[real AT-SPI: GTK3 window under headless sway]")
    d = tempfile.mkdtemp(); os.chmod(d, 0o700)
    env = dict(os.environ, XDG_RUNTIME_DIR=d, WLR_BACKENDS="headless", WLR_RENDERER="pixman", WLR_LIBINPUT_NO_DEVICES="1", NERVA_TEST_DIR=d)
    open(os.path.join(d, "c"), "w").write('for_window [title="Nerva Test Window"] floating enable\n')
    sw = subprocess.Popen(["sway", "-c", os.path.join(d, "c")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if os.path.exists(os.path.join(d, "wayland-1")): break
            time.sleep(0.1)
        env.update(WAYLAND_DISPLAY="wayland-1", GDK_BACKEND="wayland")
        r = subprocess.run(["dbus-run-session", "--", sys.executable, os.path.abspath(__file__), "--inner"], env=env, capture_output=True, text=True, timeout=90)
        sys.stdout.write("\n".join(l for l in r.stdout.splitlines() if l.startswith("  FAIL") or l.startswith("INNER")) + "\n")
        m = re.search(r"INNER (\d+) (\d+)", r.stdout)
        if m: globals()["passed"] += int(m.group(1)); globals()["failed"] += int(m.group(2))
        else: globals()["failed"] += 1; print("  FAIL: inner run did not finish:", r.stderr[-300:])
    finally: sw.terminate(); sw.wait(10)

if __name__ == "__main__":
    if "--inner" in sys.argv:
        real_inner(); print("INNER %d %d" % (passed, failed)); sys.exit(0)
    unit(); real_outer()
    print("\n%d checks passed, %d failed, %d skipped" % (passed, failed, skipped)); sys.exit(1 if failed else 0)
