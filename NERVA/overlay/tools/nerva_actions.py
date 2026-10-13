#!/usr/bin/env python3
"""nerva-actions v1: the action language a model writes to drive the NERVA overlay, and the observation text it reads back.

  parse(text)    -> actions + problems         (what the model wrote)
  lint(text)     -> list of problems           (use it to FILTER / VALIDATE training data; empty list = valid)
  compile(acts)  -> overlay stdin lines + waits (what the overlay executes)
  render_*       -> compact observation text   (what the model reads next)
  Session        -> runs a turn against a real `nerva-overlay` process and returns the <obs> block (to record real rollouts)

The grammar and every rule are documented in docs/ACTIONS.md. This file is the executable version of that document.
"""
import json
import math
import os
import queue
import re
import subprocess
import threading
import time

VERSION = 1
MAX_ACTIONS_PER_TURN = 8
SIDES = ("left", "right")

# ------------------------------------------------------------------------------------------------------------------- vocabulary
# tag -> (group, one-line usage). The single source of truth for the tag list (docs and the tokenizer list are generated from it).
TAGS = {
    "think":          ("flow",       "free reasoning, never executed (keep it short)"),
    "say":            ("flow",       "text for the user (no effect on the desktop)"),
    "wait":           ("flow",       "seconds 0.1-10"),
    "done":           ("flow",       "end the task; optional reason"),
    "hand":           ("body",       "left|right x y z   |   left|right rest        (metres)"),
    "foot":           ("body",       "left|right x y z [yaw_deg]                    (metres)"),
    "pelvis":         ("body",       "x y z   |   auto"),
    "face":           ("body",       "x z   |   auto"),
    "look":           ("body",       "x y z   |   auto"),
    "torso":          ("body",       "x y z   |   auto"),
    "grip":           ("body",       "left|right 0..1"),
    "stand":          ("body",       "(no arguments)"),
    "balance":        ("body",       "assist|none"),
    "w2px":           ("map",        "x y z   -> screen pixels"),
    "px2w":           ("map",        "px py   -> metres (z, y)"),
    "move":           ("pointer",    "px py"),
    "tohand":         ("pointer",    "left|right   (to where that fingertip is drawn)"),
    "click":          ("pointer",    "px py   (move, then left click)   |   (empty = click where the pointer is)"),
    "rclick":         ("pointer",    "px py   |   (empty)   -- needs --allow-right-click"),
    "scroll":         ("pointer",    "n   |   px py n      (n: -5..5, positive = up)"),
    "pointer_status": ("pointer",    "(no arguments)"),
    "perceive":       ("see",        "(no arguments)  windows, pointer grant, body"),
    "status":         ("see",        "(no arguments)  body state"),
    "tree":           ("see",        "[target] [depth 1-30]   UI elements of a window"),
    "read":           ("see",        "[target]   the text on screen"),
    "find":           ("see",        "words [in target]   elements by name/text, with where to click"),
    "focused":        ("see",        "[target]   the element with keyboard focus"),
    "apps":           ("see",        "(no arguments)  which windows can be read"),
}
# What exists in the overlay but is for HUMANS only. A model that writes these is wrong (and they are rejected, with a reason).
HUMAN_ONLY = {"grant": "only the user can grant the pointer (ask with <say>)", "revoke": "only the user can revoke",
              "hide": "only the user decides what is hidden", "unhide": "only the user decides what is hidden",
              "key": "the pointer has no keyboard", "type": "the pointer has no keyboard", "drag": "dragging is not supported",
              "keypress": "the pointer has no keyboard"}

NUM = r"-?\d+(?:\.\d+)?"
INT = r"-?\d+"


def _num(s):
    return float(s)


# Tokens to register as SPECIAL tokens in the model's tokenizer (one token each; they never appear inside normal text).
OBS_TAGS = ("task", "state", "obs", "r", "ev", "screen", "text", "found", "focused", "apps")


def special_tokens():
    toks = []
    for t in TAGS: toks += ["<%s>" % t, "</%s>" % t]
    for t in OBS_TAGS:
        if "<%s>" % t not in toks: toks += ["<%s>" % t, "</%s>" % t]
    return toks


# ------------------------------------------------------------------------------------------------------------------- parse
class Action:
    def __init__(self, tag, arg, index):
        self.tag, self.arg, self.index = tag, arg.strip(), index

    def __repr__(self):
        return "<%s>%s</%s>" % (self.tag, self.arg, self.tag)


TAG_RE = re.compile(r"<(/?)([a-z0-9_]+)>")


def parse(text):
    """Return (actions, problems). Strict: text outside tags, nesting, unclosed or unknown tags are problems."""
    actions, problems, pos, idx = [], [], 0, 0
    text = text or ""
    while pos < len(text):
        m = TAG_RE.search(text, pos)
        if not m:
            if text[pos:].strip(): problems.append("text outside tags: %r" % text[pos:].strip()[:40])
            break
        if text[pos:m.start()].strip(): problems.append("text outside tags: %r" % text[pos:m.start()].strip()[:40])
        if m.group(1):
            problems.append("closing tag </%s> without an opening tag" % m.group(2)); pos = m.end(); continue
        name = m.group(2); close = "</%s>" % name
        end = text.find(close, m.end())
        if end < 0:
            problems.append("<%s> is never closed" % name); break
        inner = text[m.end():end]
        if "<" in inner or ">" in inner:
            problems.append("<%s> contains another tag or a '<' / '>' character" % name)
        if name in HUMAN_ONLY: problems.append("<%s> is not an action: %s" % (name, HUMAN_ONLY[name]))
        elif name not in TAGS: problems.append("unknown tag <%s>" % name)
        else: actions.append(Action(name, inner, idx)); idx += 1
        pos = end + len(close)
    return actions, problems


# ------------------------------------------------------------------------------------------------------------------- validate
def _check(a):
    """Problems with one action's arguments. (Strict: pixel coordinates are integers; metres have at most 2 decimals.)"""
    t, s = a.tag, a.arg
    w = s.split()
    P = []
    def nums(ws, ints=False):
        for x in ws:
            if not re.fullmatch(INT if ints else NUM, x): P.append("'%s' is not %s" % (x, "an integer" if ints else "a number")); return False
            if abs(float(x)) > 100000: P.append("'%s' is out of range" % x); return False
            if not ints and "." in x and len(x.split(".")[1]) > 2: P.append("'%s' has more than 2 decimals" % x); return False
        return True
    def metres(ws): return nums(ws)
    if t in ("think",): return P if len(s) <= 400 else ["<think> is too long (max 400 characters)"]
    if t == "say":
        if not s: P.append("<say> is empty")
        if len(s) > 300: P.append("<say> is longer than 300 characters")
    elif t == "done":
        if len(s) > 200: P.append("<done> reason is longer than 200 characters")
    elif t == "wait":
        if len(w) != 1 or not re.fullmatch(NUM, w[0]) or not 0.1 <= float(w[0]) <= 10: P.append("<wait> takes one number, 0.1-10 seconds")
    elif t == "hand":
        if not w or w[0] not in SIDES: P.append("<hand> starts with left or right")
        elif w[1:] == ["rest"]: pass
        elif len(w) == 4 and metres(w[1:]): pass
        elif not P: P.append("<hand> takes 'side x y z' or 'side rest'")
    elif t == "foot":
        if not w or w[0] not in SIDES: P.append("<foot> starts with left or right")
        elif len(w) in (4, 5) and metres(w[1:]): pass
        elif not P: P.append("<foot> takes 'side x y z [yaw]'")
    elif t in ("pelvis", "look", "torso"):
        if w == ["auto"]: pass
        elif len(w) == 3 and metres(w): pass
        elif not P: P.append("<%s> takes 'x y z' or 'auto'" % t)
    elif t == "face":
        if w == ["auto"]: pass
        elif len(w) == 2 and metres(w): pass
        elif not P: P.append("<face> takes 'x z' or 'auto'")
    elif t == "grip":
        if len(w) != 2 or w[0] not in SIDES or not re.fullmatch(NUM, w[1]) or not 0 <= float(w[1]) <= 1: P.append("<grip> takes 'side value' with value 0..1")
    elif t == "stand" or t in ("pointer_status", "perceive", "status", "apps"):
        if w: P.append("<%s> takes no arguments" % t)
    elif t == "balance":
        if w not in (["assist"], ["none"]): P.append("<balance> takes assist or none")
    elif t == "w2px":
        if len(w) != 3 or not metres(w): P.append("<w2px> takes 'x y z'")
    elif t == "px2w":
        if len(w) != 2 or not nums(w, True): P.append("<px2w> takes 'px py' (integers)")
    elif t == "move":
        if len(w) != 2 or not nums(w, True): P.append("<move> takes 'px py' (integers)")
    elif t == "tohand":
        if len(w) != 1 or w[0] not in SIDES: P.append("<tohand> takes left or right")
    elif t in ("click", "rclick"):
        if w and len(w) != 2: P.append("<%s> takes 'px py' (integers) or nothing" % t)
        elif w: nums(w, True)
    elif t == "scroll":
        if len(w) == 1: ok = nums(w, True) and 1 <= abs(int(w[0])) <= 5
        elif len(w) == 3: ok = nums(w, True) and 1 <= abs(int(w[2])) <= 5
        else: ok = False
        if not ok and not P: P.append("<scroll> takes 'n' or 'px py n' with n in -5..5, not 0")
    elif t == "tree":
        if len(w) > 2: P.append("<tree> takes [target] [depth]")
        elif w and re.fullmatch(INT, w[-1]) and not 1 <= int(w[-1]) <= 30: P.append("<tree> depth is 1-30")
    elif t in ("read", "focused"):
        if len(w) > 1: P.append("<%s> takes at most one target" % t)
    elif t == "find":
        q = s[:-len(" in " + w[-1])] if len(w) >= 4 and w[-2].lower() == "in" else s
        if not q.strip(): P.append("<find> needs words to look for")
        if len(q) > 120: P.append("<find> query is longer than 120 characters")
    return P


def lint(text, max_actions=MAX_ACTIONS_PER_TURN):
    """All problems with one model turn; [] means the turn is valid nerva-actions v1."""
    actions, problems = parse(text)
    problems = list(problems)
    for a in actions:
        for p in _check(a): problems.append("<%s>: %s" % (a.tag, p))
    if not actions and not problems: problems.append("empty turn (write at least one action, or <done>)")
    exec_n = [a for a in actions if a.tag != "think"]
    if len(exec_n) > max_actions: problems.append("more than %d actions in one turn" % max_actions)
    d = [i for i, a in enumerate(actions) if a.tag == "done"]
    if len(d) > 1: problems.append("more than one <done>")
    if d and d[0] != len(actions) - 1: problems.append("<done> must be the last action of the turn")
    return problems


# ------------------------------------------------------------------------------------------------------------------- compile
class Step:
    """One overlay stdin line. `wait_before` seconds are slept first; `kind` is 'sync' or 'a11y' (answer arrives later)."""
    def __init__(self, line, action, kind="sync", wait_before=0.0):
        self.line, self.action, self.kind, self.wait_before = line, action, kind, wait_before

    def __repr__(self): return "Step(%r, wait=%.2f)" % (self.line, self.wait_before)


PTR_SPEED = 3000.0      # px/s the harness allows itself (the overlay's own limit is 6000)
CLICK_GAP = 0.35        # s between clicks (the overlay's limit is 0.25)


def compile_actions(actions, pointer=(None, None)):
    """Translate parsed actions into overlay lines. Also paces the pointer so the overlay's rate limits are never what fails."""
    steps, (px, py), last_click = [], pointer, -1e9
    clock = 0.0
    def move(x, y, a):
        nonlocal px, py, clock
        gap = 0.0
        if px is not None: gap = max(0.0, math.hypot(x - px, y - py) / PTR_SPEED - 0.0)
        steps.append(Step("MOUSE MOVE %s %s" % (x, y), a, wait_before=round(gap, 2))); px, py = float(x), float(y); clock += gap
    for a in actions:
        t, w, s = a.tag, a.arg.split(), a.arg
        if t in ("think", "say", "done"): continue
        if t == "wait": steps.append(Step("", a, "wait", float(w[0]))); clock += float(w[0]); continue
        if t == "hand": steps.append(Step("HAND " + s, a))
        elif t == "foot": steps.append(Step("FOOT " + s, a))
        elif t in ("pelvis", "face", "look", "torso", "grip", "balance"): steps.append(Step("%s %s" % (t.upper(), s), a))
        elif t == "stand": steps.append(Step("STAND", a))
        elif t == "w2px": steps.append(Step("W2PX " + s, a))
        elif t == "px2w": steps.append(Step("PX2W " + s, a))
        elif t == "move": move(int(w[0]), int(w[1]), a)
        elif t == "tohand": steps.append(Step("MOUSE TOHAND " + s, a)); px = py = None
        elif t in ("click", "rclick"):
            if w: move(int(w[0]), int(w[1]), a)
            gap = max(0.0, CLICK_GAP - (clock - last_click))
            steps.append(Step("MOUSE CLICK " + ("right" if t == "rclick" else "left"), a, wait_before=round(gap, 2))); last_click = clock + gap; clock += gap
        elif t == "scroll":
            if len(w) == 3: move(int(w[0]), int(w[1]), a)
            steps.append(Step("MOUSE SCROLL " + w[-1], a))
        elif t == "pointer_status": steps.append(Step("MOUSE STATUS", a))
        elif t == "perceive": steps.append(Step("PERCEIVE", a))
        elif t == "status": steps.append(Step("STATUS", a))
        elif t == "tree": steps.append(Step("A11Y TREE " + s if s else "A11Y TREE", a, "a11y"))
        elif t == "read": steps.append(Step("A11Y TEXT " + s if s else "A11Y TEXT", a, "a11y"))
        elif t == "find": steps.append(Step("A11Y FIND " + s, a, "a11y"))
        elif t == "focused": steps.append(Step("A11Y FOCUSED " + s if s else "A11Y FOCUSED", a, "a11y"))
        elif t == "apps": steps.append(Step("A11Y APPS", a, "a11y"))
    return steps


# ------------------------------------------------------------------------------------------------------------------- render (observation text)
def _q(s): return '"%s"' % (s or "").replace('"', "'")


def _flags(e):
    out = ""
    if e.get("st") or e.get("secret"): out += " (%s)" % ",".join(list(e.get("st", [])) + (["secret"] if e.get("secret") else []))
    if e.get("act"): out += " [%s]" % ",".join(e["act"])
    if e.get("text"): out += " =" + _q(e["text"])
    return out


def render_tree(resp, max_elements=60):
    out = []
    for w in resp.get("windows", []):
        if w.get("error"): out.append('<screen>\nwindow %s\nerror: %s\n</screen>' % (_q(w.get("cls")), w["error"])); continue
        els = w.get("elements", [])
        lines = ["%d %s %s @%d,%d %dx%d%s" % (e["i"], e["role"], _q(e["name"]), e["c"][0], e["c"][1], e["rect"][2], e["rect"][3], _flags(e)) for e in els[:max_elements]]
        if len(els) > max_elements: lines.append("... +%d more (use <find> or <tree> with a target)" % (len(els) - max_elements))
        if w.get("truncated"): lines.append("(tree cut: too large)")
        out.append('<screen>\nwindow %s\n%s\n</screen>' % (_q(w.get("cls")), "\n".join(lines)))
    return "\n".join(out)


def render_find(resp, max_hits=12):
    ms = resp.get("matches", [])
    if not ms: return "<found>no match</found>"
    lines = ["%s %s @%d,%d%s in %s" % (m["role"], _q(m["name"]), m["c"][0], m["c"][1], (" [%s]" % ",".join(m["act"])) if m.get("act") else "", m.get("cls", "")) for m in ms[:max_hits]]
    if len(ms) > max_hits: lines.append("... +%d more" % (len(ms) - max_hits))
    return "<found>\n%s\n</found>" % "\n".join(lines)


def render_text(resp, max_lines=40):
    out = []
    for w in resp.get("windows", []):
        if w.get("error"): out.append("<text>\nwindow %s\nerror: %s\n</text>" % (_q(w.get("cls")), w["error"])); continue
        items = [i for i in w.get("items", []) if "text" in i][:max_lines]
        out.append("<text>\nwindow %s\n%s\n</text>" % (_q(w.get("cls")), "\n".join(_q(i["text"]) for i in items)))
    return "\n".join(out)


def render_focused(resp):
    f = resp.get("focused")
    if not f: return "<focused>nothing</focused>"
    return "<focused>%s %s @%d,%d%s in %s</focused>" % (f["role"], _q(f["name"]), f["c"][0], f["c"][1], _flags(f), f.get("cls", ""))


def render_apps(resp):
    return "<apps>\n%s\n</apps>" % "\n".join("%s: %s" % (a.get("cls", ""), "readable" if a.get("readable") else ("hidden" if a.get("error") else "not readable")) for a in resp.get("apps", []))


def render_a11y(op, resp):
    if not resp.get("ok"): return "err " + str(resp.get("error", "failed"))
    return {"tree": render_tree, "text": render_text, "find": render_find, "focused": render_focused, "apps": render_apps}[op](resp)


def render_perceive(p):
    o, v, b, m = p.get("output", {}), p.get("view", {}), p.get("body", {}), p.get("mouse", {})
    L = ["screen %sx%s, pixels per metre %s; screen_x = %d + z*ppm, screen_y = %d - y*ppm" % (o.get("w"), o.get("h"), v.get("ppm"), (o.get("w") or 0) // 2, o.get("h") or 0)]
    pel = b.get("pelvis", [0, 0, 0]); L.append("body %s, pelvis (%.2f, %.2f, %.2f)" % (b.get("state", "?"), pel[0], pel[1], pel[2]))
    if p.get("hidden"): L.append("overlay hidden: a fullscreen window is active")
    if not m.get("enabled"): L.append("pointer: disabled")
    elif not m.get("granted"): L.append("pointer: not granted (ask the user to grant a window)")
    else: L.append("pointer: granted to %s @%d,%d %dx%d, %ss left%s" % (_q(m.get("app")), m["rect"][0], m["rect"][1], m["rect"][2], m["rect"][3], m.get("seconds_left"), "" if m.get("focused") else ", NOT focused"))
    for w in p.get("windows", []):
        px = w.get("px", [0, 0, 0, 0]); top = w.get("top")
        L.append("window %s @%d,%d %dx%d%s" % (w.get("class"), px[0], px[1], px[2], px[3], (" top=%.2fm" % top["y"]) if top else ""))
    return "<state>\n%s\n</state>" % "\n".join(L)


def render_status(st):
    f = st.get("feet", {}); pel = st.get("pelvis", [0, 0, 0])
    def foot(side): x = f.get(side, {}); pos = x.get("pos", [0, 0, 0]); return "%s (%.2f, %.2f, %.2f) %s" % (side, pos[0], pos[1], pos[2], x.get("state", "?"))
    h = st.get("hands", {})
    return "body %s, balance %s (margin %.1f cm), pelvis (%.2f, %.2f, %.2f), %s, %s, steps queued %s, hands L (%.2f, %.2f, %.2f) R (%.2f, %.2f, %.2f)" % (
        st.get("state"), st.get("balance"), st.get("margin_cm", 0), pel[0], pel[1], pel[2], foot("left"), foot("right"), st.get("steps_queued", 0),
        *(h.get("left", {}).get("pos", [0, 0, 0]) + h.get("right", {}).get("pos", [0, 0, 0])))


def render_obs(results, events=()):
    """results: [(action_index_1based, 'ok'|'err', text_or_block)]. Returns the <obs> block the model reads."""
    L = ["<obs>"]
    for n, status, text in results:
        L.append("<r>%d %s %s</r>" % (n, status, text) if "\n" not in text and not text.startswith("<") else "<r>%d %s</r>\n%s" % (n, status, text))
    for e in events: L.append("<ev>%s</ev>" % e)
    L.append("</obs>")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------------- run against a real overlay
class Session:
    """Spawn (or wrap) a `nerva-overlay` and run model turns against it. Use it to record REAL rollouts for training data."""
    def __init__(self, cmd, env=None, ready_timeout=15):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self.q = queue.Queue(); self.pointer = (None, None); self.closed = False
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._wait(lambda l: l.startswith("EVENT overlay ready"), ready_timeout)

    def _pump(self):
        for l in self.p.stdout: self.q.put(l.rstrip("\n"))
        self.q.put(None)

    def _wait(self, pred, timeout, keep=None):
        t_end = time.time() + timeout
        while time.time() < t_end:
            try: l = self.q.get(timeout=max(0.01, t_end - time.time()))
            except queue.Empty: return None
            if l is None: return None
            if pred(l): return l
            if keep is not None: keep.append(l)
        return None

    def run(self, model_text, settle=0.3, a11y_timeout=15):
        """Execute one model turn. Returns (obs_block, problems, done_reason_or_None)."""
        problems = lint(model_text)
        actions, _ = parse(model_text)
        if problems: return render_obs([(1, "err", "invalid turn: " + "; ".join(problems[:3]))]), problems, None
        steps = compile_actions(actions, self.pointer)
        numbered = [x for x in actions if x.tag != "think"]
        pos = {id(x): i for i, x in enumerate(numbered, 1)}
        last_step = {}
        for st in steps: last_step[id(st.action)] = st
        results, events, failed = [], [], set()
        def take(noise): events.extend(l[6:] for l in noise if l.startswith("EVENT "))
        for st in steps:
            if st.wait_before: time.sleep(st.wait_before)
            if st.kind == "wait" or id(st.action) in failed: continue
            n, final = pos[id(st.action)], st is last_step[id(st.action)]
            self.p.stdin.write(st.line + "\n"); self.p.stdin.flush()
            noise = []
            line = self._wait(lambda l: l.startswith(("OK", "ERR", "{")), 20, noise); take(noise)
            if line is None: results.append((n, "err", "no reply from the overlay")); break
            if st.line.startswith("MOUSE MOVE") and line.startswith("OK"): self.pointer = tuple(float(v) for v in line.split()[3:5])
            if line.startswith("ERR"):
                failed.add(id(st.action)); results.append((n, "err", line[4:])); continue
            if st.kind == "a11y" and line.startswith("OK a11y"):
                rid = line.split()[2]; noise = []
                got = self._wait(lambda l: l.startswith("A11Y %s " % rid), a11y_timeout, noise); take(noise)
                if got is None: results.append((n, "err", "the accessibility read did not answer")); continue
                body = render_a11y("text" if st.action.tag == "read" else st.action.tag, json.loads(got.split(" ", 2)[2]))
                results.append((n, "err", body[4:]) if body.startswith("err ") else (n, "ok", body)); continue
            if not final: continue                                              # first half of a composite action (the move of a click)
            if line.startswith("{"):
                try: p = json.loads(line)
                except ValueError: results.append((n, "ok", line)); continue
                results.append((n, "ok", render_perceive(p) if "output" in p else render_status(p) if "feet" in p else line))
            else: results.append((n, "ok", line[3:].strip() or "done"))
        time.sleep(settle)
        while True:
            try: l = self.q.get_nowait()
            except queue.Empty: break
            if l and l.startswith("EVENT "): events.append(l[6:])
        done = next((a.arg for a in actions if a.tag == "done"), None)
        return render_obs(results, events[:12]), [], done

    def state(self):
        """The <state> block for the start of an episode."""
        self.p.stdin.write("PERCEIVE\n"); self.p.stdin.flush()
        line = self._wait(lambda l: l.startswith("{"), 10)
        return render_perceive(json.loads(line)) if line else ""

    def close(self):
        if self.closed: return
        self.closed = True
        try: self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(5)
        except Exception: self.p.kill()


# ------------------------------------------------------------------------------------------------------------------- CLI
def _main(argv):
    import sys
    if len(argv) >= 2 and argv[1] == "tags":
        for t, (g, u) in TAGS.items(): print("%-8s %-14s %s" % (g, "<%s>" % t, u))
        return 0
    if len(argv) >= 2 and argv[1] == "tokens":
        print("\n".join(special_tokens())); return 0
    if len(argv) >= 3 and argv[1] == "lint":                               # lint a JSONL of episodes (assistant turns), or a text file with one turn per line
        bad = n = 0
        for path in argv[2:]:
            for ln, line in enumerate(open(path), 1):
                line = line.strip()
                if not line: continue
                turns = []
                try:
                    d = json.loads(line)
                    turns = [t["content"] for t in d.get("turns", []) if t.get("role") == "assistant"] if isinstance(d, dict) else [d]
                except ValueError: turns = [line]
                for t in turns:
                    n += 1; pr = lint(t)
                    if pr: bad += 1; print("%s:%d: %s" % (path, ln, "; ".join(pr)))
        print("%d turns checked, %d invalid" % (n, bad)); return 1 if bad else 0
    print("usage: nerva_actions.py tags | tokens | lint FILE.jsonl ..."); return 2


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
