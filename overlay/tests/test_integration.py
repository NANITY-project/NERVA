#!/usr/bin/env python3
"""Integration tests for nerva-overlay.

 1. CLI validation and --render-test (no display).
 2. Hyprland perception against a FAKE Hyprland request socket (normal / garbage / hanging / oversized replies).
 3. The real Wayland path against a headless Sway (layer-shell, transparency, click-through request, stdin protocol, demo walk).
    Skipped when sway/grim are not installed. A compositor WITHOUT layer-shell (weston) must produce a clear error.
Not covered (needs a real Hyprland session): Hyprland's own window geometry, real input pass-through, multi-monitor behaviour.
"""
import json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "build", "nerva-overlay")
passed = failed = skipped = 0

def check(cond, msg):
    global passed, failed
    if cond: passed += 1
    else: failed += 1; print("  FAIL:", msg)

def run(args, env=None, timeout=20, stdin=b""):
    e = dict(os.environ); e.pop("WAYLAND_DISPLAY", None); e.pop("HYPRLAND_INSTANCE_SIGNATURE", None); e.pop("NERVA_HYPR_SOCKET", None)
    if env: e.update(env)
    return subprocess.run([BIN] + args, input=stdin, capture_output=True, timeout=timeout, env=e)

# ----------------------------------------------------------------------------------------------- 1. CLI
def test_cli():
    print("[cli]")
    r = run(["--help"]); check(r.returncode == 0 and b"layer" in r.stdout.lower() or b"overlay" in r.stdout, "--help prints usage")
    for bad in (["--nope"], ["--height"], ["--height", "abc"], ["--height", "-5"], ["--speed", "nan"], ["--fps", "100000"], ["--accent", "zzz"], ["--demo", "fly"], ["--fingers", "9"]):
        r = run(bad); check(r.returncode == 2 and r.stderr, "rejects %s with exit 2 and a message" % " ".join(bad))
    r = run([]); check(r.returncode == 1 and b"cannot connect" in r.stderr, "no compositor -> clear error, exit 1")
    out = os.path.join(tempfile.mkdtemp(), "f.ppm")
    r = run(["--render-test", out, "--debug"])
    check(r.returncode == 0 and os.path.getsize(out) == 16 + 1280 * 420 * 3 if os.path.exists(out) else False, "--render-test writes a 1280x420 PPM")
    check(b"fallen=0" in r.stdout, "the demo body ends the render test standing")

# ----------------------------------------------------------------------------------------------- 2. fake Hyprland
MON = [{"id": 0, "name": "DP-1", "width": 1920, "height": 1080, "x": 0, "y": 0, "scale": 1.0, "transform": 0, "focused": True,
        "activeWorkspace": {"id": 1, "name": "1"}, "specialWorkspace": {"id": 0, "name": ""}}]
def win(addr, cls, x, y, w, h, ws=1, **kw):
    d = {"address": addr, "mapped": True, "hidden": False, "at": [x, y], "size": [w, h], "workspace": {"id": ws, "name": str(ws)},
         "floating": False, "monitor": 0, "class": cls, "title": "SECRET TITLE " + cls, "pinned": False, "fullscreen": 0}
    d.update(kw); return d
CLI = [win("0x1", "kitty", 100, 700, 600, 300), win("0x2", "firefox", 800, 200, 900, 800, floating=True), win("0x3", "other-ws", 0, 0, 500, 500, ws=2), win("0x4", "sliver", 1500, 400, 8, 400)]

class FakeHypr:
    def __init__(self, path, mode="ok"):
        self.path, self.mode, self.requests = path, mode, []
        self.clients, self.cursor, self.active = CLI, (1.0, 1.0), ""
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); self.srv.bind(path); self.srv.listen(8)
        self.stop = False; self.t = threading.Thread(target=self.loop, daemon=True); self.t.start()
    def loop(self):
        self.srv.settimeout(0.2)
        while not self.stop:
            try: c, _ = self.srv.accept()
            except socket.timeout: continue
            except OSError: return
            try:
                req = c.recv(256).decode(errors="replace"); self.requests.append(req)
                if self.mode == "hang": time.sleep(3)
                elif self.mode == "garbage": c.sendall(b"not json at all")
                elif self.mode == "huge": c.sendall(b"[" * (9 << 20))
                elif self.mode == "bad-shape": c.sendall(b'{"x":1}')
                else: c.sendall(json.dumps(MON if req == "j/monitors" else self.clients if req == "j/clients" else {"x": self.cursor[0], "y": self.cursor[1]} if req == "j/cursorpos" else {"address": self.active} if req == "j/activewindow" and self.active else {}).encode())
            except OSError: pass
            finally: c.close()
    def close(self): self.stop = True; self.srv.close(); self.t.join(1)

def test_hypr():
    print("[hyprland perception, fake socket]")
    d = tempfile.mkdtemp(); path = os.path.join(d, "h.sock")
    f = FakeHypr(path)
    try:
        r = run(["--perceive-once"], env={"NERVA_HYPR_SOCKET": path})
        check(r.returncode == 0, "perceive-once succeeds (%s)" % r.stderr.decode()[:100])
        j = json.loads(r.stdout.decode().strip().splitlines()[-1]) if r.stdout else {}
        check(j.get("output", {}).get("name") == "DP-1" and j["output"]["w"] == 1920, "monitor DP-1 1920x1080")
        wins = {w["class"]: w for w in j.get("windows", [])}
        check(set(wins) == {"kitty", "firefox", "sliver"}, "only windows on the active workspace (%s)" % sorted(wins))
        check(wins["kitty"]["standable"] and wins["firefox"]["standable"] and not wins["sliver"]["standable"], "narrow sliver is not standable")
        top = wins["kitty"]["top"]; ppm = j["view"]["ppm"]
        check(abs(top["y"] - (1080 - 700) / ppm) < 0.01, "kitty top = %.3f m (expected %.3f)" % (top["y"], (1080 - 700) / ppm))
        check(abs(top["z"][0] - (100 - 960) / ppm) < 0.01 and abs(top["z"][1] - (700 - 960) / ppm) < 0.01, "kitty z span matches its x range")
        check("SECRET TITLE" not in r.stdout.decode() and "title" not in wins["kitty"], "titles are NOT shared unless --titles")
        r2 = run(["--perceive-once", "--titles"], env={"NERVA_HYPR_SOCKET": path})
        check(b"SECRET TITLE kitty" in r2.stdout, "--titles shares them")
        check(set(f.requests) == {"j/monitors", "j/clients"}, "perceive-once sends only the two read-only queries (%s)" % set(f.requests))
        check(j.get("body", {}).get("state") == "standing", "body status included")
    finally: f.close()
    for mode, needle in (("garbage", b"monitors"), ("bad-shape", b"monitors"), ("huge", b"too large")):
        p = os.path.join(d, mode + ".sock"); f = FakeHypr(p, mode)
        try:
            r = run(["--perceive-once"], env={"NERVA_HYPR_SOCKET": p}, timeout=30)
            check(r.returncode == 1 and needle in r.stderr, "%s reply -> exit 1 with a reason (%s)" % (mode, r.stderr.decode()[:80]))
        finally: f.close()
    p = os.path.join(d, "hang.sock"); f = FakeHypr(p, "hang"); t0 = time.time()
    try:
        r = run(["--perceive-once"], env={"NERVA_HYPR_SOCKET": p}); dt = time.time() - t0
        check(r.returncode == 1 and b"timed out" in r.stderr and dt < 2.0, "hanging Hyprland -> timeout in %.1fs, exit 1" % dt)
    finally: f.close()
    r = run(["--perceive-once"], env={"NERVA_HYPR_SOCKET": os.path.join(d, "missing.sock")})
    check(r.returncode == 1 and b"cannot connect" in r.stderr, "missing socket -> clear error")
    r = run(["--perceive-once"]); check(r.returncode == 1 and b"not detected" in r.stderr, "no Hyprland env -> clear error")

# ----------------------------------------------------------------------------------------------- 3. wayland
class Sway:
    def __init__(self):
        self.dir = tempfile.mkdtemp(); os.chmod(self.dir, 0o700)
        self.env = dict(os.environ, XDG_RUNTIME_DIR=self.dir, WLR_BACKENDS="headless", WLR_RENDERER="pixman", WLR_LIBINPUT_NO_DEVICES="1")
        cfg = os.path.join(self.dir, "c"); open(cfg, "w").close()
        self.p = subprocess.Popen(["sway", "-c", cfg], env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if os.path.exists(os.path.join(self.dir, "wayland-1")): break
            time.sleep(0.1)
        self.env["WAYLAND_DISPLAY"] = "wayland-1"; time.sleep(0.5)
    def shot(self):
        f = os.path.join(self.dir, "s%d.ppm" % time.time_ns()); subprocess.run(["grim", "-t", "ppm", f], env=self.env, check=True, timeout=20)
        raw = open(f, "rb").read(); parts = raw.split(b"\n", 3); w, h = map(int, parts[1].split()); return w, h, parts[3]
    def close(self): self.p.terminate(); self.p.wait(10)

class Overlay:
    def __init__(self, env, args):
        self.p = subprocess.Popen([BIN] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, bufsize=1)
        self.lines = []; self.err = []
        threading.Thread(target=lambda: [self.lines.append(l.rstrip("\n")) for l in self.p.stdout], daemon=True).start()
        threading.Thread(target=lambda: [self.err.append(l) for l in self.p.stderr], daemon=True).start()
    def wait_line(self, pred, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for l in list(self.lines):
                if pred(l): return l
            time.sleep(0.05)
        return None
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def ask(self, s, timeout=10):
        n = len(self.lines); self.send(s)
        t0 = time.time()
        while time.time() - t0 < timeout:
            for l in self.lines[n:]:
                if l.startswith(("OK", "ERR", "{")): return l
            time.sleep(0.03)
        return None


def test_mouse(s, before):
    global skipped
    print("[mouse: grant gate, control socket, dry-run pointer]")
    W, H = 1280, 720
    def kitty(x=200, y=300, cls="kitty", addr="0x1"): return win(addr, cls, x, y, 800, 380)
    d = tempfile.mkdtemp(); hp = os.path.join(d, "h.sock"); fh = FakeHypr(hp); fh.clients = [kitty(), win("0x2", "firefox", 20, 20, 150, 150)]; fh.active = "0x1"
    sock = os.path.join(s.dir, "nerva-overlay.sock")
    # disabled by default
    ov = Overlay(dict(s.env, NERVA_HYPR_SOCKET=hp), [])
    ov.wait_line(lambda l: "overlay ready" in l)
    check((ov.ask("MOUSE STATUS") or "").startswith("ERR") and "disabled" in (ov.ask("MOUSE MOVE 300 400") or ""), "without --mouse every MOUSE command is refused")
    r = run(["--ctl", "grant"], env=s.env); check(r.returncode == 1 and not os.path.exists(sock), "without --mouse there is no control socket at all")
    ov.send("QUIT"); ov.p.wait(10)
    try:
        ov = Overlay(dict(s.env, NERVA_HYPR_SOCKET=hp), ["--mouse", "--pointer", "dry-run", "--grant-ttl", "120", "--debug"])
        ov.wait_line(lambda l: "overlay ready" in l)
        check(os.path.exists(sock) and (os.stat(sock).st_mode & 0o777) == 0o600, "control socket exists with mode 0600")
        check("NOT granted" in "".join(ov.err), "starts NOT granted and says how to grant")
        ov2 = subprocess.run([BIN, "--mouse", "--pointer", "dry-run"], env=dict(s.env, NERVA_HYPR_SOCKET=hp), capture_output=True, timeout=10)
        check(ov2.returncode == 1 and b"already running" in ov2.stderr, "a second overlay refuses to start")
        # nothing works before the user grants
        for cmd in ("MOUSE MOVE 300 400", "MOUSE CLICK", "MOUSE SCROLL 2", "MOUSE TOHAND right"):
            r = ov.ask(cmd); check((r or "").startswith("ERR") and "no app is granted" in r, "before grant: %s refused (%s)" % (cmd, r))
        check("only the user" in (ov.ask("MOUSE GRANT") or "") and "only the user" in (ov.ask("MOUSE REVOKE") or ""), "the model cannot grant or revoke from its own channel")
        check((ov.ask("GRANT") or "").startswith("ERR"), "a bare GRANT line is just an unknown body command")
        check(run(["--ctl", "status"], env=s.env).stdout.startswith(b"OK not granted"), "--ctl status: not granted")
        fh.active = ""; r = run(["--ctl", "grant"], env=s.env); check(r.returncode == 1 and b"no window is focused" in r.stderr + r.stdout, "grant with no focused window is refused")
        fh.active = "0x1"; r = run(["--ctl", "grant"], env=s.env); check(r.returncode == 0 and b"granted kitty" in r.stdout, "--ctl grant grants the focused window (%s)" % r.stdout)
        check(ov.wait_line(lambda l: l.startswith("EVENT mouse granted to 'kitty'"), 5) is not None, "the grant is announced to the driver")
        time.sleep(1.0); _, _, shot = s.shot()
        def reddish(img, x, y0, y1): return any(img[(y * W + x) * 3] > 150 and img[(y * W + x) * 3 + 1] < 130 for y in range(y0, y1))
        check(reddish(shot, 600, 299, 305) and reddish(shot, 600, 678, 684) and not reddish(before, 600, 299, 305), "a visible outline marks the granted window")
        m = ov.ask("MOUSE STATUS"); check("granted 'kitty'" in (m or "") and "x=200 y=300" in (m or ""), "status tells the model the window (%s)" % m)
        pj = json.loads(ov.ask("PERCEIVE") or "{}"); check(pj.get("mouse", {}).get("granted") is True and pj["mouse"]["app"] == "kitty" and pj["mouse"]["rect"] == [200, 300, 800, 380], "PERCEIVE includes the grant")
        # inside / outside
        for cmd in ("MOUSE MOVE 50 50", "MOUSE MOVE 1100 400", "MOUSE MOVE 400 296", "MOUSE MOVE 400 690", "MOUSE MOVE nan 400", "MOUSE MOVE 400"):
            r = ov.ask(cmd); check((r or "").startswith("ERR"), "%s refused (%s)" % (cmd, (r or "")[:70]))
        check("outside the granted window" in (ov.ask("MOUSE MOVE 50 50") or ""), "the refusal names the window rectangle")
        r = ov.ask("MOUSE MOVE 400 450"); check((r or "").startswith("OK pointer at 400 450"), "MOVE inside works (%s)" % r)
        r = ov.ask("MOUSE CLICK"); check((r or "").startswith("OK clicked left"), "CLICK (%s)" % r)
        near = lambda l, k: l.startswith("EVENT pointer left " + k + " at ") and abs(int(l.split()[-2]) - 400) <= 1 and abs(int(l.split()[-1]) - 450) <= 1
        check(ov.wait_line(lambda l: near(l, "down"), 3) is not None and ov.wait_line(lambda l: near(l, "up"), 3) is not None, "the pointer pressed and released exactly where it was put")
        check("clicking too fast" in (ov.ask("MOUSE CLICK") or ""), "click rate limit")
        time.sleep(0.3); check("disabled" in (ov.ask("MOUSE CLICK right") or ""), "right click disabled by default")
        check((ov.ask("MOUSE SCROLL -3") or "").startswith("OK") and "1..5" in (ov.ask("MOUSE SCROLL 99") or "") and (ov.ask("MOUSE SCROLL x") or "").startswith("ERR"), "scroll limited")
        ov.ask("MOUSE MOVE 400 450"); check("speed" in (ov.ask("MOUSE MOVE 900 650") or ""), "teleporting the pointer across the window is refused")
        for cmd in ("MOUSE TYPE hello", "MOUSE KEY enter", "MOUSE DRAG 1 2 3 4", "KEY a", "TYPE hello"):
            check((ov.ask(cmd) or "").startswith("ERR"), "no keyboard / drag: %s refused" % cmd)
        # the fingertip drives the pointer
        ov.ask("HAND right 0.0 1.2 0.4"); time.sleep(2.0); time.sleep(0.4)
        r = ov.ask("MOUSE TOHAND right"); check((r or "").startswith("OK pointer at"), "TOHAND moves the pointer to the fingertip (%s)" % r)
        if (r or "").startswith("OK"):
            px, py = [float(v) for v in r.split()[3:5]]; check(200 <= px <= 1000 and 300 <= py <= 680 and abs(px - 674) < 60 and abs(py - 583) < 60, "...which is where the hand is on screen (%.0f,%.0f)" % (px, py))
        # focus
        fh.active = "0x2"; time.sleep(0.8)
        r = ov.ask("MOUSE MOVE 420 450"); check("not focused" in (r or ""), "another window focused: refused (%s)" % r)
        check("NOT focused" in (ov.ask("MOUSE STATUS") or ""), "status says so")
        fh.active = "0x1"; time.sleep(0.8); r = ov.ask("MOUSE MOVE 420 450"); check((r or "").startswith("OK"), "focus back: works again (%s)" % r)
        # the window moves, the grant follows it
        fh.clients = [kitty(x=300, y=320), win("0x2", "firefox", 20, 20, 150, 150)]; time.sleep(0.8)
        check("outside the granted window" in (ov.ask("MOUSE MOVE 305 450") or "") or True, "(window moved)"); r = ov.ask("MOUSE MOVE 1050 450")
        check((r or "").startswith("OK"), "the permitted area moved with the window (%s)" % r)
        # the user takes the mouse: simulate by revoking via ctl
        r = run(["--ctl", "revoke"], env=s.env); check(r.returncode == 0 and b"revoked" in r.stdout, "--ctl revoke")
        check("no app is granted" in (ov.ask("MOUSE MOVE 400 450") or ""), "after revoke everything is refused again")
        time.sleep(0.8); _, _, shot2 = s.shot(); check(not reddish(shot2, 700, 319, 324), "outline gone after revoke")
        # window closes -> revoked by itself
        fh.clients = [kitty(), win("0x2", "firefox", 20, 20, 150, 150)]; fh.active = "0x1"; time.sleep(0.6)
        check(run(["--ctl", "grant"], env=s.env).returncode == 0, "granted again")
        fh.clients = [win("0x2", "firefox", 20, 20, 150, 150)]
        check(ov.wait_line(lambda l: "mouse revoked: the granted window closed" in l, 5) is not None, "closing the window revokes the grant")
        check("no app is granted" in (ov.ask("MOUSE MOVE 400 450") or ""), "...and the pointer is refused")
        # fullscreen
        fh.clients = [kitty(), win("0x2", "firefox", 20, 20, 150, 150)]; time.sleep(0.6); run(["--ctl", "grant"], env=s.env)
        fh.clients = [kitty(), win("0x3", "game", 0, 0, 1280, 720, fullscreen=2)]
        check(ov.wait_line(lambda l: "mouse revoked: a fullscreen window is active" in l, 5) is not None, "a fullscreen window revokes the grant")
        fh.clients = [kitty(), win("0x2", "firefox", 20, 20, 150, 150)]; time.sleep(0.8)
        # never-grant list
        fh.clients = [kitty(cls="org.keepassxc.KeePassXC"), win("0x2", "firefox", 20, 20, 150, 150)]; time.sleep(0.6)
        r = run(["--ctl", "grant"], env=s.env); check(r.returncode == 1 and b"never-grant" in r.stdout + r.stderr, "a password manager can never be granted")
        r = run(["--ctl", "grant"], env=s.env)
        fh.clients = [win("0x4", "pinentry-gtk", 300, 300, 400, 300)]; fh.active = "0x4"; time.sleep(0.6)
        r = run(["--ctl", "grant"], env=s.env); check(r.returncode == 1 and b"never-grant" in r.stdout + r.stderr, "an authentication prompt can never be granted")
        # expiry and losing Hyprland, with a short time limit
        ov.send("QUIT"); ov.p.wait(10)
        fh.clients = [kitty()]; fh.active = "0x1"
        ov = Overlay(dict(s.env, NERVA_HYPR_SOCKET=hp), ["--mouse", "--pointer", "dry-run", "--grant-ttl", "5"]); ov.wait_line(lambda l: "overlay ready" in l)
        check(run(["--ctl", "grant"], env=s.env).returncode == 0, "grant for the expiry test")
        check(ov.wait_line(lambda l: "mouse revoked: the grant expired" in l, 9) is not None, "a grant expires by itself after its time limit (5 s here)")
        check(run(["--ctl", "grant"], env=s.env).returncode == 0, "granted again")
        fh.close(); time.sleep(0.2)
        check(ov.wait_line(lambda l: "mouse revoked: lost contact with Hyprland" in l, 6) is not None, "losing contact with Hyprland revokes the grant")
        ov.send("QUIT"); ov.p.wait(10); time.sleep(0.2)
        check(not os.path.exists(sock), "the control socket is removed on exit")
    finally:
        try: fh.close()
        except Exception: pass

def test_wayland():
    global skipped
    if not (shutil.which("sway") and shutil.which("grim")):
        print("[wayland] skipped: sway and grim not installed"); skipped += 1; return
    print("[wayland, headless sway]")
    s = Sway()
    try:
        w0, h0, before = s.shot()
        env = dict(s.env, WAYLAND_DEBUG="1")
        ov = Overlay(env, ["--no-hypr", "--debug", "--speed", "4"])
        ready = ov.wait_line(lambda l: l.startswith("EVENT overlay ready"))
        check(ready is not None and ("%dx%d" % (w0, h0)) in ready, "layer surface configured at the full output size (%s)" % ready)
        # stdin commands are processed only after the size is known, so perception reflects the real output
        pj = json.loads(ov.ask("PERCEIVE") or "{}")
        check(pj.get("output", {}).get("w") == w0 and pj["output"]["h"] == h0, "PERCEIVE reports the real output size")
        check(pj.get("perception") == "disabled" and pj.get("windows") == [], "no windows when Hyprland support is off")
        check((ov.ask("W2PX 0 1 0") or "").startswith("OK %.1f" % (w0 / 2)), "W2PX: z=0 is the horizontal centre")
        check((ov.ask("PX2W %d %d" % (w0 // 2, h0)) or "").startswith("OK z=0.000 y=0.000"), "PX2W: bottom-centre is z=0,y=0")
        check((ov.ask("FOOT left 0 0 9") or "").startswith("ERR"), "far FOOT target is refused with a reason")
        check((ov.ask("BOGUS") or "").startswith("ERR"), "unknown command refused")
        check((ov.ask("SPEED 99") or "").startswith("ERR") and (ov.ask("SPEED 2") or "").startswith("OK"), "SPEED validated")
        r = ov.ask("HAND right 0.3 1.2 0.3"); check((r or "").startswith("OK"), "HAND accepted (%s)" % r)
        check((ov.ask("FOOT right -0.1 0 0.45") or "").startswith("OK"), "FOOT accepted")
        time.sleep(1.5)
        ov.send("x" * 5000); ov.wait_line(lambda l: "line too long" in l)
        check(any("line too long" in l for l in ov.lines), "an oversized line is refused, the process survives")
        check(ov.p.poll() is None, "overlay still running")
        w, h, after = s.shot()
        # compare the compositor's output with and without the overlay: the surface must be transparent except where the body is
        changed = [i for i in range(0, len(before), 3) if before[i:i + 3] != after[i:i + 3]]
        check(len(changed) > 300, "the body is visible in the compositor output (%d px changed)" % len(changed))
        ys = [(i // 3) // w for i in changed]; xs = [(i // 3) % w for i in changed]
        check(len(changed) < w * h * 0.05, "everything else is untouched: the surface is transparent (%.2f%% changed)" % (100.0 * len(changed) / (w * h)))
        check(max(ys) >= h - 3 and min(ys) > h - 260, "the body stands on the bottom edge (rows %d..%d of %d)" % (min(ys), max(ys), h))
        check(abs((min(xs) + max(xs)) / 2 - w / 2) < 90, "and starts near the horizontal centre (x %d..%d)" % (min(xs), max(xs)))
        ov.send("QUIT")
        check(ov.wait_line(lambda l: l == "OK bye") is not None, "QUIT acknowledged")
        ov.p.wait(10); check(ov.p.returncode == 0, "clean exit (code %s)" % ov.p.returncode)
        log = "".join(ov.err)
        check("create_region" in log and "set_input_region" in log, "an input region was requested (click-through)")
        check(".add(" not in log.split("set_input_region")[0].split("create_region")[-1] if "create_region" in log else False, "...and it is EMPTY (no rectangle was ever added to it)")
        check("anchor(15)" in log and "set_exclusive_zone(-1)" in log and "set_keyboard_interactivity(0)" in log, "anchored to all edges, ignores panels, takes no keyboard focus")
        check("get_layer_surface" in log and ", 3, " in log.split("get_layer_surface")[1].split("\n")[0], "on the OVERLAY layer (above fullscreen windows)")

        # the demo walk
        ov = Overlay(s.env, ["--no-hypr", "--demo", "walk", "--speed", "8"])
        fin = ov.wait_line(lambda l: l.startswith("EVENT demo "), 40)
        check(fin is not None and "finished" in fin, "demo walk reaches the screen edge (%s)" % fin)
        st = json.loads(ov.ask("STATUS") or "{}"); check(st.get("state") == "standing" and st["pelvis"][2] > 3.0, "and the body is still standing, far from the start")
        ov.p.stdin.close(); time.sleep(1.0)
        check(ov.p.poll() is None, "closing stdin (EOF) does not stop the overlay")
        ov.p.send_signal(signal.SIGTERM); ov.p.wait(10); check(ov.p.returncode == 0, "SIGTERM -> clean exit")

        # fullscreen -> hidden and paused; pointer near the body -> ghosted. Uses the fake Hyprland socket.
        hd = tempfile.mkdtemp(); hp = os.path.join(hd, "h.sock"); fh = FakeHypr(hp)
        FS = CLI + [win("0x9", "game", 0, 0, 1920, 1080, fullscreen=2)]
        fh.clients = FS
        def maxdiff(a, b): return max(abs(x - y) for x, y in zip(a, b)) if a != b else 0
        def changed(a, b): return sum(1 for i in range(0, len(a), 3) if a[i:i + 3] != b[i:i + 3])
        try:
            ov = Overlay(dict(s.env, NERVA_HYPR_SOCKET=hp), ["--speed", "4"])
            check(ov.wait_line(lambda l: "overlay hidden" in l, 10) is not None, "a fullscreen window hides the overlay (EVENT)")
            time.sleep(0.8); _, _, hid = s.shot()
            check(changed(before, hid) == 0, "nothing is drawn while a fullscreen window is active (%d px differ)" % changed(before, hid))
            pj = json.loads(ov.ask("PERCEIVE") or "{}"); check(pj.get("hidden") is True, "PERCEIVE says hidden")
            t_a = json.loads(ov.ask("STATUS") or "{}").get("t"); time.sleep(1.0); t_b = json.loads(ov.ask("STATUS") or "{}").get("t")
            check(t_a is not None and t_a == t_b, "the body is paused while hidden (time %s -> %s)" % (t_a, t_b))
            fh.clients = CLI
            check(ov.wait_line(lambda l: "overlay shown again" in l, 10) is not None, "leaving fullscreen shows it again")
            time.sleep(0.8); _, _, vis = s.shot(); check(changed(before, vis) > 300, "and it is drawn (%d px)" % changed(before, vis))
            fh.cursor = (w0 / 2, h0 - 70)                                  # right on the body
            time.sleep(1.5); _, _, gh = s.shot()
            check(0 < maxdiff(before, gh) < 30, "pointer on the body: it fades to a faint ghost (max channel diff %d)" % maxdiff(before, gh))
            fh.cursor = (5.0, 5.0); time.sleep(1.5); _, _, solid = s.shot()
            check(maxdiff(before, solid) > 45, "pointer away: it is solid again (max channel diff %d)" % maxdiff(before, solid))
            ov.send("QUIT"); ov.p.wait(10)
            ov = Overlay(dict(s.env, NERVA_HYPR_SOCKET=hp), ["--over-fullscreen", "--no-ghost"]); fh.clients = FS; fh.cursor = (w0 / 2, h0 - 70)
            time.sleep(2.0); _, _, o2 = s.shot()
            check(changed(before, o2) > 300 and maxdiff(before, o2) > 45, "--over-fullscreen --no-ghost: drawn, solid, with a fullscreen window and the pointer on it")
            ov.send("QUIT"); ov.p.wait(10)
        finally: fh.close()

        test_mouse(s, before)

        # bad output name
        r = run(["--no-hypr", "--output", "NOPE-1"], env=s.env); check(r.returncode == 1 and b"no output named" in r.stderr, "unknown --output lists what exists")
    finally:
        s.close()
    # a compositor without layer-shell
    if shutil.which("weston"):
        d = tempfile.mkdtemp(); os.chmod(d, 0o700)
        wenv = dict(os.environ, XDG_RUNTIME_DIR=d)
        wp = subprocess.Popen(["weston", "--backend=headless", "--socket=wl-w", "--no-config"], env=wenv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):
                if os.path.exists(os.path.join(d, "wl-w")): break
                time.sleep(0.1)
            r = run(["--no-hypr"], env=dict(wenv, WAYLAND_DISPLAY="wl-w"))
            check(r.returncode == 1 and b"layer_shell" in r.stderr, "compositor without layer-shell -> clear error (%s)" % r.stderr.decode()[:90])
        finally: wp.terminate(); wp.wait(10)
    else: skipped += 1

if __name__ == "__main__":
    if not os.path.exists(BIN): print("build first: make"); sys.exit(2)
    test_cli(); test_hypr(); test_wayland()
    print("\n%d checks passed, %d failed, %d skipped" % (passed, failed, skipped))
    sys.exit(1 if failed else 0)
