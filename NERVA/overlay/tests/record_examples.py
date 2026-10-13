#!/usr/bin/env python3
"""Record REAL training episodes: a scripted teacher drives a real nerva-overlay (real GTK window -> real AT-SPI -> real gate, dry-run
pointer) through tools/nerva_actions.Session, and every observation in the transcript is what the overlay actually answered.
Usage: python3 tests/record_examples.py [OUT.jsonl]      (needs sway, dbus-run-session, python3-gi, at-spi2-core)
Run inside the same environment as tests/test_a11y_full.py. The default output is tools/examples/episodes.jsonl."""
import json, os, re, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import nerva_actions as N
import test_integration as TI
from test_integration import FakeHypr, win, BIN, run

OUT = [a for a in sys.argv[1:] if not a.startswith("--")] or [os.path.join(HERE, "..", "tools", "examples", "episodes.jsonl")]
OUT = OUT[0]

class Ep:
    def __init__(self, sess, eid, task, tags, meta=None):
        self.s, self.d = sess, {"v": N.VERSION, "id": eid, "task": task, "tags": tags, "meta": meta or {}, "start": sess.state(), "turns": []}
        self.last = ""
    def turn(self, text):
        probs = N.lint(text); assert not probs, (self.d["id"], text, probs)
        obs, _, done = self.s.run(text)
        self.d["turns"] += [{"role": "assistant", "content": text}, {"role": "observation", "content": obs}]
        self.last = obs; return obs, done
    def finish(self, text):
        probs = N.lint(text); assert not probs, (self.d["id"], text, probs)
        self.d["turns"].append({"role": "assistant", "content": text}); return self.d

def first_xy(obs, pat=r"@(\d+),(\d+)"):
    m = re.search(pat, obs); assert m, obs; return int(m.group(1)), int(m.group(2))

def inner():
    d = os.environ["NERVA_TEST_DIR"]; cfg = os.path.join(d, "cfg", "perception.json")
    subprocess.Popen(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
    app = subprocess.Popen([sys.executable, os.path.join(HERE, "gtk_testapp.py"), os.path.join(d, "alloc.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(os.path.join(d, "alloc.json")): break
        time.sleep(0.1)
    time.sleep(0.3); al = json.load(open(os.path.join(d, "alloc.json")))
    hp = os.path.join(d, "h.sock"); fh = FakeHypr(hp)
    fh.clients = [win("0x1", "gtk_testapp.py", 200, 150, al["window"][0], al["window"][1], pid=al["pid"], title="Nerva Test Window"), win("0x2", "firefox", 20, 20, 150, 150, pid=999999, title="Page"),
                  win("0x4", "discord", 60, 60, 300, 300, pid=777777, title="chat")]
    fh.active = "0x1"
    env = dict(os.environ, NERVA_HYPR_SOCKET=hp, NERVA_CONFIG=cfg)
    def start(args, grant=False):
        s = N.Session([BIN] + args, env=env); assert s.ready, "overlay did not start"; time.sleep(1.0)
        if grant: assert run(["--ctl", "grant"], env=dict(os.environ)).returncode == 0
        return s
    eps = []
    full = ["--a11y", "--mouse", "--pointer", "dry-run", "--speed", "4"]

    s = start(["--a11y", "--speed", "4"]); time.sleep(1.5)
    e = Ep(s, "reach-001", "Raise your right hand in front of you to about chest height, then tell me which screen pixel the fingertip is on.", ["body", "map"])
    obs, _ = e.turn("<think>Right hand forward and up, then ask where that point is on screen.</think><hand>right 0.30 1.20 0.40</hand><wait>2</wait><w2px>0.30 1.20 0.40</w2px>")
    px, py = [int(float(v)) for v in re.search(r"OK ([\d.-]+) ([\d.-]+)", obs.replace("<r>3 ok ", "OK ")).groups()] if "<r>3 ok" in obs else (0, 0)
    eps.append(e.finish("<say>My right hand is at about pixel %d,%d on the screen.</say><done>reached</done>" % (px, py))); s.close()

    s = start(["--a11y", "--speed", "6"]); time.sleep(1.5)
    e = Ep(s, "walk-001", "Walk forward (to the right) three steps.", ["body"])
    for side, x, z in (("right", -0.10, 0.45), ("left", 0.10, 0.90), ("right", -0.10, 1.35)):
        e.turn("<foot>%s %.2f 0.00 %.2f</foot><wait>2.5</wait>" % (side, x, z))
    e.turn("<status></status>")
    eps.append(e.finish("<say>I took three steps forward.</say><done>walked three steps</done>")); s.close()

    s = start(full, grant=True)
    e = Ep(s, "click-001", "Click the 'Save changes' button.", ["a11y", "pointer", "click"])
    obs, _ = e.turn("<think>Find the button first.</think><find>save changes</find>"); x, y = first_xy(obs)
    e.turn("<click>%d %d</click>" % (x, y))
    eps.append(e.finish("<say>I clicked Save changes.</say><done>clicked</done>"))

    e = Ep(s, "click-002", "Click the search box, then scroll down in the notes box.", ["a11y", "pointer", "click", "scroll", "recovery"], {"recovery": "the first click is deliberately outside the granted window"})
    e.turn("<think>The search box is somewhere at the top left.</think><click>100 100</click>")
    obs, _ = e.turn("<think>That was outside the granted window. Find the real position.</think><find>search box</find>"); x, y = first_xy(obs)
    e.turn("<click>%d %d</click>" % (x, y))
    obs, _ = e.turn("<find>notes</find>"); x, y = first_xy(obs)
    e.turn("<scroll>%d %d -3</scroll>" % (x, y))
    eps.append(e.finish("<say>I clicked the search box and scrolled the notes down.</say><done>both done</done>"))

    e = Ep(s, "read-001", "What does the Notes box say?", ["a11y", "read"])
    obs, _ = e.turn("<read>gtk_testapp</read>")
    m = re.search(r'"(First line[^"]*)"', obs); txt = m.group(1) if m else "(unreadable)"
    eps.append(e.finish("<say>The notes say: %s</say><done></done>" % txt.replace("⏎", "then").replace("<", "").replace(">", "")))

    e = Ep(s, "secret-001", "Log in for me: put the account password into the password box.", ["a11y", "safety"])
    e.turn("<think>Check what the password box is.</think><tree>gtk_testapp 6</tree>")
    eps.append(e.finish("<say>The 'Account password' box is a password field. I never read or type passwords. Please type it yourself.</say><done>the user must type the password</done>"))

    e = Ep(s, "tree-001", "List the controls in this window.", ["a11y", "tree"])
    e.turn("<tree>gtk_testapp 6</tree>")
    eps.append(e.finish("<say>The window has a label, a Save changes button, an Enable feature checkbox, a search box, an account password box and a notes box.</say><done></done>"))

    e = Ep(s, "hidden-001", "Read what is in the Discord window.", ["a11y", "safety", "hidden"])
    run(["--ctl", "hide discord"], env=dict(os.environ)); time.sleep(0.7)
    e.turn("<tree>discord</tree>")
    eps.append(e.finish("<say>I can't read that window: you have hidden it from me.</say><done>not allowed to read discord</done>"))
    run(["--ctl", "unhide discord"], env=dict(os.environ)); s.close()

    s = start(full)                                           # NOT granted
    e = Ep(s, "denied-001", "Click the 'Save changes' button.", ["pointer", "safety", "not-granted"])
    e.turn("<pointer_status></pointer_status>")
    eps.append(e.finish("<say>I can't click yet: the pointer isn't granted to any window. Please focus the app and grant it, then ask me again.</say><done>waiting for the user to grant the pointer</done>"))
    s.close()

    with open(OUT, "w") as f:
        for ep in eps: f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print("INNER recorded %d episodes -> %s" % (len(eps), OUT))
    fh.close(); app.terminate()

def outer():
    d = tempfile.mkdtemp(); os.chmod(d, 0o700)
    env = dict(os.environ, XDG_RUNTIME_DIR=d, WLR_BACKENDS="headless", WLR_RENDERER="pixman", WLR_LIBINPUT_NO_DEVICES="1", NERVA_TEST_DIR=d)
    open(os.path.join(d, "c"), "w").write('for_window [title="Nerva Test Window"] floating enable\n')
    sw = subprocess.Popen(["sway", "-c", os.path.join(d, "c")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if os.path.exists(os.path.join(d, "wayland-1")): break
            time.sleep(0.1)
        env.update(WAYLAND_DISPLAY="wayland-1", GDK_BACKEND="wayland")
        r = subprocess.run(["dbus-run-session", "--", sys.executable, os.path.abspath(__file__), "--inner"] + sys.argv[1:], env=env, capture_output=True, text=True, timeout=300)
        print((r.stdout + r.stderr)[-1500:])
        return r.returncode
    finally: sw.terminate(); sw.wait(10)

if __name__ == "__main__":
    if "--inner" in sys.argv: inner()
    else: sys.exit(outer())
