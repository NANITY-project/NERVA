#!/usr/bin/env python3
"""Tests for tools/nerva_actions.py (the model-facing action language), the example episodes, and docs/ACTIONS.md staying in sync."""
import json, os, re, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import nerva_actions as N
BIN = os.path.join(HERE, "..", "build", "nerva-overlay")
passed = failed = skipped = 0
def check(c, m):
    global passed, failed
    if c: passed += 1
    else: failed += 1; print("  FAIL:", m)

VALID = ["<say>hello</say>", "<wait>1.5</wait>", "<done></done>", "<done>finished</done>", "<think>x</think><stand></stand>", "<hand>left 0.3 1.2 0.4</hand>", "<hand>right rest</hand>",
         "<foot>right -0.10 0 0.45</foot>", "<foot>left 0.1 0 0.9 30</foot>", "<pelvis>0 0.9 0</pelvis>", "<pelvis>auto</pelvis>", "<face>0 1</face>", "<face>auto</face>",
         "<look>0 1.5 2</look>", "<look>auto</look>", "<torso>0 1 0.1</torso>", "<torso>auto</torso>", "<grip>left 0.5</grip>", "<balance>assist</balance>", "<balance>none</balance>",
         "<w2px>0.3 1.2 0.4</w2px>", "<px2w>640 360</px2w>", "<move>400 300</move>", "<tohand>right</tohand>", "<click>838 351</click>", "<click></click>", "<rclick>10 20</rclick>",
         "<scroll>-3</scroll>", "<scroll>400 300 2</scroll>", "<pointer_status></pointer_status>", "<perceive></perceive>", "<status></status>", "<tree></tree>", "<tree>kitty</tree>",
         "<tree>kitty 5</tree>", "<tree>5</tree>", "<read></read>", "<read>granted</read>", "<find>save changes</find>", "<find>save in firefox</find>", "<focused></focused>", "<focused>kitty</focused>", "<apps></apps>"]
INVALID = ["", "   ", "hello", "<say></say>", "<say>" + "x" * 301 + "</say>", "<wait>0</wait>", "<wait>11</wait>", "<wait>x</wait>", "<wait>1 2</wait>", "<hand>middle 1 2 3</hand>",
           "<hand>left 1 2</hand>", "<hand>left a b c</hand>", "<hand>left 1.234 2 3</hand>", "<hand>left 1e9 2 3</hand>", "<foot>left 1 2</foot>", "<foot>left 1 2 3 4 5</foot>", "<pelvis>1 2</pelvis>",
           "<face>1 2 3</face>", "<grip>left 2</grip>", "<grip>left</grip>", "<balance>maybe</balance>", "<stand>now</stand>", "<w2px>1 2</w2px>", "<px2w>1.5 2</px2w>", "<move>1 2 3</move>",
           "<move>1.5 2</move>", "<tohand>both</tohand>", "<click>838</click>", "<click>8.5 3</click>", "<rclick>1</rclick>", "<scroll>0</scroll>", "<scroll>6</scroll>", "<scroll>1 2</scroll>",
           "<scroll>1 2 0</scroll>", "<tree>a b c</tree>", "<tree>kitty 99</tree>", "<read>a b</read>", "<find></find>", "<find>" + "x" * 130 + "</find>", "<apps>now</apps>", "<perceive>x</perceive>",
           "<foo>bar</foo>", "<click>1 2", "</click>", "text <click>1 2</click>", "<click>1 2</click> trailing", "<say><click>1 2</click></say>", "<say>a < b</say>",
           "<grant></grant>", "<revoke></revoke>", "<hide>discord</hide>", "<unhide>x</unhide>", "<key>enter</key>", "<type>hello</type>", "<drag>1 2 3 4</drag>", "<done></done><stand></stand>",
           "<done></done><done></done>", "<think>" + "x" * 401 + "</think><stand></stand>"]

def unit():
    print("[parse + lint]")
    for t in VALID: check(N.lint(t) == [], "valid turn rejected: %s -> %s" % (t, N.lint(t)))
    for t in INVALID: check(N.lint(t) != [], "invalid turn accepted: %r" % t[:60])
    check(N.lint("<stand></stand>" * 8) == [] and N.lint("<stand></stand>" * 9) != [], "max 8 actions per turn")
    check(N.lint("<think>a</think>" * 5 + "<stand></stand>") == [], "<think> does not count towards the action limit")
    check(any("human" in p or "user" in p for p in N.lint("<grant></grant>")), "human-only actions are rejected WITH a reason: %s" % N.lint("<grant></grant>"))
    check(any("keyboard" in p for p in N.lint("<type>hi</type>")), "typing is rejected with a reason")
    acts, pr = N.parse("<say>hi</say><click>1 2</click>"); check(len(acts) == 2 and pr == [] and acts[1].arg == "1 2" and acts[1].index == 1, "parse returns ordered actions")
    check(all(N.lint("<%s></%s>" % (t, t)) is not None for t in N.TAGS), "every tag is known to lint")

    print("[compile to overlay lines]")
    def L(t, ptr=(None, None)): return [(s.line, s.wait_before) for s in N.compile_actions(N.parse(t)[0], ptr)]
    table = {"<hand>left 0.3 1.2 0.4</hand>": ["HAND left 0.3 1.2 0.4"], "<hand>right rest</hand>": ["HAND right rest"], "<foot>right -0.1 0 0.45</foot>": ["FOOT right -0.1 0 0.45"],
             "<pelvis>auto</pelvis>": ["PELVIS auto"], "<face>0 1</face>": ["FACE 0 1"], "<look>0 1 2</look>": ["LOOK 0 1 2"], "<torso>0 1 0</torso>": ["TORSO 0 1 0"], "<grip>left 1</grip>": ["GRIP left 1"],
             "<stand></stand>": ["STAND"], "<balance>none</balance>": ["BALANCE none"], "<w2px>1 2 3</w2px>": ["W2PX 1 2 3"], "<px2w>5 6</px2w>": ["PX2W 5 6"], "<move>400 300</move>": ["MOUSE MOVE 400 300"],
             "<tohand>left</tohand>": ["MOUSE TOHAND left"], "<click>838 351</click>": ["MOUSE MOVE 838 351", "MOUSE CLICK left"], "<click></click>": ["MOUSE CLICK left"],
             "<rclick>1 2</rclick>": ["MOUSE MOVE 1 2", "MOUSE CLICK right"], "<scroll>-3</scroll>": ["MOUSE SCROLL -3"], "<scroll>5 6 2</scroll>": ["MOUSE MOVE 5 6", "MOUSE SCROLL 2"],
             "<pointer_status></pointer_status>": ["MOUSE STATUS"], "<perceive></perceive>": ["PERCEIVE"], "<status></status>": ["STATUS"], "<tree></tree>": ["A11Y TREE"], "<tree>kitty 5</tree>": ["A11Y TREE kitty 5"],
             "<read></read>": ["A11Y TEXT"], "<read>granted</read>": ["A11Y TEXT granted"], "<find>save in firefox</find>": ["A11Y FIND save in firefox"], "<focused></focused>": ["A11Y FOCUSED"], "<apps></apps>": ["A11Y APPS"],
             "<say>hi</say><think>x</think><done></done>": []}
    for t, want in table.items(): check([l for l, w in L(t) if l] == want, "compile %s -> %s (got %s)" % (t, want, L(t)))
    check(set(re.findall(r"<(\w+)>", " ".join(table))) >= set(N.TAGS) - {"wait", "say", "done", "think"} - set(), "the compile table covers every action tag")
    st = N.compile_actions(N.parse("<wait>2</wait>")[0]); check(len(st) == 1 and st[0].kind == "wait" and st[0].wait_before == 2.0, "wait is a pause, not an overlay line")
    s = N.compile_actions(N.parse("<a11yx></a11yx>")[0]); check(s == [], "unknown tags compile to nothing")
    r = L("<click>100 100</click><click>1000 600</click>"); check(r[2][1] > 0.2, "a long pointer jump is preceded by a pause (%s)" % r)
    clicks = [w for l, w in L("<click>100 100</click><click>100 100</click><click></click>") if l.startswith("MOUSE CLICK")]; check(clicks[1] >= 0.3 and clicks[2] >= 0.3, "consecutive clicks are spaced (%s)" % clicks)
    r = L("<move>500 500</move>", ptr=(100, 100)); check(r[0][1] > 0.1, "pacing uses the pointer's known position")
    check([a.kind for a in N.compile_actions(N.parse("<tree></tree><find>x</find><read></read><focused></focused><apps></apps><status></status>")[0])] == ["a11y"] * 5 + ["sync"], "a11y answers are marked asynchronous")

    print("[observation rendering]")
    els = [{"i": i, "p": -1, "role": "push button", "name": "B%d" % i, "rect": [10, 10 + i, 100, 20], "c": [60, 20 + i], "act": ["click"]} for i in range(100)]
    out = N.render_tree({"ok": True, "windows": [{"cls": "kitty", "elements": els, "truncated": False}]}); check(out.count("\n") < 66 and "+40 more" in out and out.startswith("<screen>\nwindow \"kitty\""), "tree is capped at 60 elements: %d lines" % out.count("\n"))
    one = N.render_tree({"ok": True, "windows": [{"cls": "app", "elements": [{"i": 0, "p": -1, "role": "text", "name": 'He said "hi"', "rect": [1, 2, 30, 10], "c": [16, 7], "st": ["focused", "editable"], "text": "typed"}, {"i": 1, "p": -1, "role": "password text", "name": "pw", "rect": [1, 2, 3, 4], "c": [2, 4], "secret": True}]}]})
    check('"He said \'hi\'"' in one and "(focused,editable)" in one and '="typed"' in one and "(secret)" in one, "tree line format: %s" % one.replace("\n", " | "))
    check("no match" in N.render_find({"ok": True, "matches": []}) and "@5,6 [click]" in N.render_find({"ok": True, "matches": [{"role": "push button", "name": "OK", "c": [5, 6], "act": ["click"], "cls": "x"}]}), "find rendering")
    check("nothing" in N.render_focused({"ok": True, "focused": None}) and "<apps>" in N.render_apps({"ok": True, "apps": [{"cls": "a", "readable": True}, {"cls": "b", "readable": False}, {"cls": "c", "error": "hidden by the user"}]}), "focused / apps rendering")
    ap = N.render_apps({"ok": True, "apps": [{"cls": "a", "readable": True}, {"cls": "b", "readable": False}, {"cls": "c", "error": "hidden by the user"}]}); check("a: readable" in ap and "b: not readable" in ap and "c: hidden" in ap, "apps lines")
    check(N.render_a11y("tree", {"ok": False, "error": "boom"}) == "err boom", "an error reply becomes err text")
    p = {"output": {"w": 1280, "h": 720}, "view": {"ppm": 114.29}, "body": {"state": "standing", "pelvis": [0, 0.93, 0]}, "mouse": {"enabled": True, "granted": True, "app": "kitty", "rect": [1, 2, 3, 4], "seconds_left": 100, "focused": False},
         "windows": [{"class": "kitty", "px": [1, 2, 3, 4], "top": {"y": 1.5}}, {"class": "(hidden)", "px": [5, 6, 7, 8]}], "hidden": False}
    rp = N.render_perceive(p); check("pointer: granted" in rp and "NOT focused" in rp and "window (hidden) @5,6 7x8" in rp and "top=1.50m" in rp and "screen_x = 640" in rp, "perceive rendering: %s" % rp.replace("\n", " | "))
    check("not granted" in N.render_perceive({"output": {"w": 1, "h": 1}, "view": {}, "body": {}, "mouse": {"enabled": True, "granted": False}}) and "disabled" in N.render_perceive({"output": {"w": 1, "h": 1}, "view": {}, "body": {}, "mouse": {"enabled": False}}), "pointer states")
    ob = N.render_obs([(1, "ok", "pointer at 4 5"), (2, "err", "nope")], ["foot planted"]); check(ob == "<obs>\n<r>1 ok pointer at 4 5</r>\n<r>2 err nope</r>\n<ev>foot planted</ev>\n</obs>", "obs block format: %r" % ob)
    check("body standing" in N.render_status({"state": "standing", "balance": "assist", "margin_cm": 5.4, "pelvis": [0, 1, 2], "feet": {"left": {"pos": [0, 0, 1], "state": "planted"}, "right": {"pos": [0, 0, 2], "state": "planted"}}, "hands": {}, "steps_queued": 0}), "status rendering")

    print("[tokens, docs]")
    toks = N.special_tokens(); check(len(toks) == len(set(toks)) and all("<%s>" % t in toks and "</%s>" % t in toks for t in N.TAGS) and "<obs>" in toks and "</r>" in toks, "special tokens: unique, every tag open+close, observation tags included (%d)" % len(toks))
    doc = open(os.path.join(HERE, "..", "..", "docs", "ACTIONS.md")).read()
    for t in N.TAGS: check("<%s>" % t in doc, "docs/ACTIONS.md mentions <%s>" % t)
    gen = subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "nerva_actions.py"), "tags"], capture_output=True, text=True).stdout.strip()
    m = re.search(r"<!-- TAGS:BEGIN -->\n```\n(.*?)\n```\n<!-- TAGS:END -->", doc, re.S); check(m and m.group(1).strip() == gen, "the tag table in docs/ACTIONS.md is exactly `nerva_actions.py tags` (regenerate it if you add a tag)")
    check(len(set(re.findall(r"nerva-actions v(\d+)", doc))) == 1 and ("nerva-actions v%d" % N.VERSION) in doc, "docs state the same version as the code (v%d)" % N.VERSION)

def examples():
    print("[example episodes (recorded from the real overlay)]")
    path = os.path.join(HERE, "..", "tools", "examples", "episodes.jsonl")
    eps = [json.loads(l) for l in open(path) if l.strip()]
    check(len(eps) >= 8 and len({e["id"] for e in eps}) == len(eps), "%d episodes, unique ids" % len(eps))
    tags_seen = set()
    for e in eps:
        i = e["id"]; T = e["turns"]
        check(e.get("v") == N.VERSION and e["task"] and "<state>" in e["start"] and e["start"].strip().endswith("</state>"), "%s: header fields" % i)
        check(T and T[0]["role"] == "assistant" and T[-1]["role"] == "assistant" and all(T[k]["role"] == ("assistant" if k % 2 == 0 else "observation") for k in range(len(T))), "%s: roles alternate, start and end with the assistant" % i)
        for k, t in enumerate(T):
            if t["role"] == "assistant":
                pr = N.lint(t["content"]); check(pr == [], "%s turn %d lints clean: %s" % (i, k, pr))
                tags_seen |= {a.tag for a in N.parse(t["content"])[0]}
            else:
                c = t["content"]; check(c.startswith("<obs>") and c.endswith("</obs>"), "%s turn %d is an <obs> block" % (i, k))
                acts = [a for a in N.parse(T[k - 1]["content"])[0] if a.tag != "think"]
                for n in map(int, re.findall(r"<r>(\d+) (?:ok|err)", c)):
                    check(1 <= n <= len(acts) and acts[n - 1].tag not in ("say", "wait", "done"), "%s turn %d: <r>%d points at an action that produces a result" % (i, k, n))
                need = [n for n, a in enumerate(acts, 1) if a.tag not in ("say", "wait", "done")]
                got = [int(x) for x in re.findall(r"<r>(\d+) ", c)]
                check(all(n in got for n in need), "%s turn %d: every executed action has a result (%s vs %s)" % (i, k, need, got))
        last = N.parse(T[-1]["content"])[0]; check(last and last[-1].tag == "done", "%s ends with <done>" % i)
    blob = open(path).read()
    check("hunter2" not in blob and "SECRET" not in blob, "no password anywhere in the examples")
    check(tags_seen >= {"hand", "foot", "w2px", "click", "scroll", "find", "tree", "read", "pointer_status", "status", "say", "done", "wait", "think"}, "the examples cover the main tags (%s)" % sorted(tags_seen))
    check(any("outside the granted window" in t["content"] for e in eps for t in e["turns"]) and any("not granted" in t["content"] for e in eps for t in e["turns"]) and any("hidden by the user" in t["content"] for e in eps for t in e["turns"]) and any(",secret)" in t["content"] for e in eps for t in e["turns"]), "the examples include the refusal / safety cases")

def live():
    global skipped
    if not (shutil.which("sway") and os.path.exists(BIN)):
        print("[live session] skipped: needs sway and a built overlay"); skipped += 1; return
    print("[live session against the real overlay (headless sway)]")
    d = tempfile.mkdtemp(); os.chmod(d, 0o700)
    env = dict(os.environ, XDG_RUNTIME_DIR=d, WLR_BACKENDS="headless", WLR_RENDERER="pixman", WLR_LIBINPUT_NO_DEVICES="1"); env.pop("HYPRLAND_INSTANCE_SIGNATURE", None)
    open(os.path.join(d, "c"), "w").close()
    sw = subprocess.Popen(["sway", "-c", os.path.join(d, "c")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if os.path.exists(os.path.join(d, "wayland-1")): break
            time.sleep(0.1)
        env["WAYLAND_DISPLAY"] = "wayland-1"
        s = N.Session([BIN, "--no-hypr", "--speed", "4"], env=env); check(s.ready, "overlay started")
        st = s.state(); check(st.startswith("<state>") and "pointer: disabled" in st, "state block: %s" % st.replace("\n", " | ")[:100])
        obs, pr, done = s.run("<hand>right 0.30 1.20 0.40</hand><w2px>0.30 1.20 0.40</w2px><say>hi</say><done>ok</done>")
        check(pr == [] and done == "ok" and "<r>1 ok" in obs and re.search(r"<r>2 ok [\d.]+ [\d.]+</r>", obs) and "<r>3" not in obs, "body + map + say/done: %s" % obs.replace("\n", " | ")[:200])
        obs, pr, done = s.run("<click>10 10</click>"); check("<r>1 err" in obs and "disabled" in obs, "pointer commands are refused when the pointer is off: %s" % obs.replace("\n", " | ")[:140])
        obs, pr, done = s.run("<tree></tree>"); check("<r>1 err" in obs and "disabled" in obs, "a11y refused when off")
        obs, pr, done = s.run("<click>1.5 2</click>"); check(pr and "invalid turn" in obs and done is None, "an invalid turn is not executed and says why: %s" % obs.replace("\n", " | ")[:140])
        obs, pr, done = s.run("<grant></grant>"); check(pr and "invalid turn" in obs, "<grant> never reaches the overlay")
        obs, pr, done = s.run("<foot>left 0.10 0 9</foot>"); check("<r>1 err" in obs, "the body's own refusals are passed to the model: %s" % obs.replace("\n", " | ")[:160])
        s.close()
    finally: sw.terminate(); sw.wait(10)

if __name__ == "__main__":
    unit(); examples(); live()
    print("\n%d checks passed, %d failed, %d skipped" % (passed, failed, skipped)); sys.exit(1 if failed else 0)
