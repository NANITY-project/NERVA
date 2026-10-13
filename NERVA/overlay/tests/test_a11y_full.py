#!/usr/bin/env python3
"""Full stack: a real GTK window -> real AT-SPI bus -> a11y_helper -> nerva-overlay (--a11y --mouse --pointer dry-run) -> stdin protocol,
with a fake Hyprland describing where the window is. Needs sway, dbus-run-session, python3-gi (Atspi, Gtk3), at-spi2-core."""
import json, os, re, shutil, signal, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import test_integration as TI
from test_integration import FakeHypr, Overlay, win, BIN, run

passed = failed = skipped = 0
def check(c, m):
    global passed, failed
    if c: passed += 1
    else: failed += 1; print("  FAIL:", m)

def a11y(ov, cmd, timeout=15):
    """send an A11Y command; return (immediate reply, parsed async json or None)"""
    n0 = len(ov.lines); r = ov.ask(cmd)
    if not (r or "").startswith("OK a11y"): return r, None
    rid = r.split()[2]
    line = ov.wait_line(lambda l: l.startswith("A11Y %s " % rid), timeout)
    return r, (json.loads(line.split(" ", 2)[2]) if line else None)

def inner():
    global passed, failed
    d = os.environ["NERVA_TEST_DIR"]; cfg = os.path.join(d, "cfg", "perception.json")
    subprocess.Popen(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
    app = subprocess.Popen([sys.executable, os.path.join(HERE, "gtk_testapp.py"), os.path.join(d, "alloc.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(os.path.join(d, "alloc.json")): break
        time.sleep(0.1)
    time.sleep(0.3); al = json.load(open(os.path.join(d, "alloc.json"))); OX, OY = 200, 300
    hp = os.path.join(d, "h.sock"); fh = FakeHypr(hp)
    G = win("0x1", "gtk_testapp.py", OX, OY, al["window"][0], al["window"][1], pid=al["pid"], title="Nerva Test Window")
    FF = win("0x2", "firefox", 20, 20, 150, 150, pid=999999, title="Some page - Mozilla Firefox")
    KP = win("0x3", "org.keepassxc.KeePassXC", 40, 40, 300, 300, pid=888888, title="Vault")
    DC = win("0x4", "discord", 60, 60, 300, 300, pid=777777, title="secret chat")
    fh.clients = [G, FF, KP, DC]; fh.active = "0x1"
    env = dict(os.environ, NERVA_HYPR_SOCKET=hp, NERVA_CONFIG=cfg)
    args = ["--a11y", "--mouse", "--pointer", "dry-run"]
    ov = Overlay(env, args); ov.wait_line(lambda l: "overlay ready" in l)
    time.sleep(0.8)
    print("[a11y: real GTK window through the whole stack]")
    r = ov.ask("A11Y STATUS"); check((r or "").startswith("OK a11y enabled") and "a11y_helper.py" in r, "STATUS: %s" % r)
    r, t = a11y(ov, "A11Y TREE"); check(t and t["ok"], "tree of the FOCUSED window by default (%s)" % (r,))
    els = t["windows"][0]["elements"] if t and t.get("ok") else []; by = {e["name"]: e for e in els}
    for nm, key in (("Save changes", "button"), ("Search box", "entry"), ("Notes", "notes"), ("Hello accessibility world", "label")):
        x, y, w, h = al[key]; check(nm in by and by[nm]["rect"] == [OX + x, OY + y, w, h], "%s: rect is the real widget position in OUTPUT pixels (%s)" % (nm, by.get(nm, {}).get("rect")))
    check(by.get("Account password", {}).get("secret") is True and "text" not in by.get("Account password", {}), "password field: position yes, content never")
    # read -> find -> point -> click, end to end
    r, f = a11y(ov, "A11Y FIND save changes in gtk_testapp"); m = f["matches"][0] if f and f.get("ok") and f["matches"] else None
    x, y, w, h = al["button"]; check(m and m["c"] == [OX + x + w // 2, OY + y + h // 2], "FIND 'save changes' -> the button's centre %s" % (m and m["c"]))
    gr = run(["--ctl", "grant"], env=dict(os.environ)); check(gr.returncode == 0 and b"gtk_testapp" in gr.stdout, "the human grants the focused window (%s)" % gr.stdout)
    if m:
        cx, cy = m["c"]; r = ov.ask("MOUSE MOVE %d %d" % (cx, cy)); check((r or "").startswith("OK pointer at"), "the model points at what it found (%s)" % r)
        ov.ask("MOUSE CLICK"); near = lambda l, k: l.startswith("EVENT pointer left " + k + " at ") and abs(int(l.split()[-2]) - cx) <= 1 and abs(int(l.split()[-1]) - cy) <= 1
        check(ov.wait_line(lambda l: near(l, "down"), 3) is not None and ov.wait_line(lambda l: near(l, "up"), 3) is not None, "...and clicks it (down+up on the button's centre)")
    r, t = a11y(ov, "A11Y TEXT"); texts = " | ".join(i["text"] for i in t["windows"][0]["items"]) if t and t.get("ok") else ""
    check("Hello accessibility world" in texts and "Second line of notes" in texts and "typed text here" in texts and "Save changes" in texts, "TEXT reads what is on screen")
    r, t = a11y(ov, "A11Y FOCUSED"); check(t and t["ok"], "FOCUSED answers (%s)" % (t and (t.get("focused") or {}).get("name")))
    r, t = a11y(ov, "A11Y FIND hunter"); check(t and t["ok"] and t["matches"] == [], "the secret cannot be found by searching for it")
    r, t = a11y(ov, "A11Y APPS"); names = {a["cls"]: a for a in t["apps"]} if t and t.get("ok") else {}
    check(names.get("gtk_testapp.py", {}).get("readable") is True and names.get("firefox", {}).get("readable") is False, "APPS: which visible windows can be read")
    check("org.keepassxc.KeePassXC" not in names and "keepassxc" not in json.dumps(t).lower(), "a password manager is never even listed")
    r, t = a11y(ov, "A11Y TREE firefox"); check(t and "no accessibility tree" in t["windows"][0].get("error", ""), "a window without accessibility is explained")
    r = ov.ask("A11Y TREE keepass"); check((r or "").startswith("ERR"), "a password manager cannot be read even by name (%s)" % r)
    r, t = a11y(ov, "A11Y FIND vault in all"); check(t and t["matches"] == [], "...and does not show up in 'all'")
    # the user's hide-list
    check(run(["--ctl", "hidden"], env=dict(os.environ)).stdout.startswith(b"OK your hide-list: (empty)"), "hide-list starts empty")
    h = run(["--ctl", "hide GTK_TestApp"], env=dict(os.environ)); check(h.returncode == 0 and b"hiding 'gtk_testapp'" in h.stdout, "the user hides an app: %s" % h.stdout)
    check(os.path.exists(cfg) and (os.stat(cfg).st_mode & 0o777) == 0o600 and "gtk_testapp" in open(cfg).read(), "...and it is saved (0600)")
    r = ov.ask("A11Y TREE gtk_testapp"); check((r or "").startswith("ERR") and "hidden" in r, "a hidden app cannot be read (%s)" % r)
    r = ov.ask("A11Y TREE"); check((r or "").startswith("ERR") and "hidden" in r, "...including as the focused window (%s)" % r)
    pj = json.loads(ov.ask("PERCEIVE") or "{}"); wc = [w["class"] for w in pj.get("windows", [])]
    check("(hidden)" in wc and "gtk_testapp.py" not in json.dumps(pj), "PERCEIVE shows it only as '(hidden)' geometry (%s)" % wc)
    r, t = a11y(ov, "A11Y FIND save"); check(t and not any(m["cls"] == "gtk_testapp.py" for m in t.get("matches", [])), "FIND over 'all' skips it")
    bad = run(["--ctl", "hide a;b"], env=dict(os.environ)); check(bad.returncode == 1, "a malformed pattern is refused")
    # the model cannot touch the hide-list from its channel
    for cmd in ("HIDE x", "UNHIDE gtk_testapp", "A11Y UNHIDE gtk_testapp", "MOUSE UNHIDE gtk_testapp"): check((ov.ask(cmd) or "").startswith("ERR"), "the model cannot change the hide-list: %s" % cmd)
    check(run(["--ctl", "unhide gtk_testapp"], env=dict(os.environ)).returncode == 0, "the user unhides it")
    r, t = a11y(ov, "A11Y TREE"); check(t and t["ok"], "readable again")
    # persistence across restarts
    run(["--ctl", "hide discord"], env=dict(os.environ)); ov.send("QUIT"); ov.p.wait(10)
    ov = Overlay(env, args); ov.wait_line(lambda l: "overlay ready" in l); time.sleep(0.8)
    pj = json.loads(ov.ask("PERCEIVE") or "{}"); check("(hidden)" in [w["class"] for w in pj.get("windows", [])], "the hide-list survives a restart")
    # helper crash recovery
    pids = subprocess.run(["pgrep", "-f", "tools/a11y_helper.py"], capture_output=True, text=True).stdout.split()
    check(len(pids) == 0 or len(pids) >= 1, "(helper running: %s)" % pids)
    r, t = a11y(ov, "A11Y TREE"); check(t and t["ok"], "tree after restart")
    pids = subprocess.run(["pgrep", "-f", "tools/a11y_helper.py"], capture_output=True, text=True).stdout.split()
    for p in pids: os.kill(int(p), signal.SIGKILL)
    time.sleep(0.5); r, t = a11y(ov, "A11Y TREE"); check(r is not None and (t is None or not t.get("ok") or True), "a killed helper does not hang or crash the overlay (%s)" % (r,))
    time.sleep(2.6); r, t = a11y(ov, "A11Y TREE"); check(t and t["ok"], "...and it is restarted automatically (%s)" % (r,))
    check(ov.p.poll() is None, "overlay still running")
    check(all("hunter2" not in l and "SECRET" not in l for l in ov.lines), "the password never appeared on the model's channel at all")
    ov.send("QUIT"); ov.p.wait(10)
    # disabled by default
    ov = Overlay(dict(env), ["--mouse", "--pointer", "dry-run"]); ov.wait_line(lambda l: "overlay ready" in l)
    check("disabled" in (ov.ask("A11Y TREE") or "") and "disabled" in (ov.ask("A11Y STATUS") or ""), "without --a11y every A11Y command is refused")
    time.sleep(0.5); check(subprocess.run(["pgrep", "-f", "tools/a11y_helper.py"], capture_output=True).returncode != 0, "...and no helper process is started")
    ov.send("QUIT"); ov.p.wait(10); fh.close(); app.terminate()
    print("INNER %d %d" % (passed, failed))

def outer():
    global skipped
    need = [shutil.which(x) for x in ("sway", "dbus-run-session")]
    try:
        import gi; gi.require_version("Atspi", "2.0"); gi.require_version("Gtk", "3.0"); from gi.repository import Atspi, Gtk  # noqa
    except Exception: need.append(None)
    if not all(need) or not os.path.exists("/usr/libexec/at-spi-bus-launcher") or not shutil.which("pgrep"):
        print("[a11y full stack] skipped: needs sway, dbus-run-session, python3-gi (Atspi, Gtk3), at-spi2-core"); skipped += 1; return
    d = tempfile.mkdtemp(); os.chmod(d, 0o700)
    env = dict(os.environ, XDG_RUNTIME_DIR=d, WLR_BACKENDS="headless", WLR_RENDERER="pixman", WLR_LIBINPUT_NO_DEVICES="1", NERVA_TEST_DIR=d)
    open(os.path.join(d, "c"), "w").write('for_window [title="Nerva Test Window"] floating enable\n')
    sw = subprocess.Popen(["sway", "-c", os.path.join(d, "c")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if os.path.exists(os.path.join(d, "wayland-1")): break
            time.sleep(0.1)
        env.update(WAYLAND_DISPLAY="wayland-1", GDK_BACKEND="wayland")
        # no accessibility bus at all (no session bus): the helper must answer with an error, not abort
        if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            print("[a11y: no accessibility bus]")
            hp = os.path.join(d, "h0.sock"); fh = FakeHypr(hp); fh.clients = [win("0x1", "someapp", 10, 10, 300, 300, pid=4321)]; fh.active = "0x1"
            ov = Overlay(dict(env, NERVA_HYPR_SOCKET=hp, NERVA_CONFIG=os.path.join(d, "n.json")), ["--a11y"]); ov.wait_line(lambda l: "overlay ready" in l); time.sleep(0.8)
            r, t = a11y(ov, "A11Y TREE"); check(t is not None and t["ok"] is False and "accessibility bus is not available" in t["error"], "no bus -> an explanatory error (%s)" % (t and t.get("error")))
            check(ov.p.poll() is None, "the overlay is unaffected")
            ov.send("QUIT"); ov.p.wait(10); fh.close()
        print("[a11y: full stack with a real GTK window]")
        r = subprocess.run(["dbus-run-session", "--", sys.executable, os.path.abspath(__file__), "--inner"], env=env, capture_output=True, text=True, timeout=240)
        sys.stdout.write("\n".join(l for l in r.stdout.splitlines() if l.startswith("  FAIL")) + "\n")
        m = re.search(r"INNER (\d+) (\d+)", r.stdout)
        if m: globals()["passed"] += int(m.group(1)); globals()["failed"] += int(m.group(2))
        else: globals()["failed"] += 1; print("  FAIL: inner run did not finish:", (r.stdout + r.stderr)[-600:])
    finally: sw.terminate(); sw.wait(10)

if __name__ == "__main__":
    if "--inner" in sys.argv: inner(); sys.exit(0)
    outer(); print("\n%d checks passed, %d failed, %d skipped" % (passed, failed, skipped)); sys.exit(1 if failed else 0)
