#!/usr/bin/env python3
"""NERVA accessibility helper: reads what is on screen through AT-SPI (the Linux accessibility bus).

READ-ONLY. It only calls getters. It never performs an action, sets a value or edits text; the only thing that can act on the desktop
is the separately gated pointer in nerva-overlay. Passwords are never read.

Protocol: one JSON request per line on stdin, one JSON line per reply on stdout (the parent runs ONE request at a time).
  {"id":7,"op":"tree|text|find|focused|apps","targets":[{"pid":108,"title":"..","address":"0x1","cls":"kitty","ox":100,"oy":300,"dx":0,"dy":0}],
   "max_nodes":300,"max_depth":16,"query":"save","role":"push button","text_limit":160,"total_text_limit":8000,"hide":["keepass"]}
`ox,oy` = the window's top-left in output pixels (from Hyprland); every returned rectangle is already in output pixels.
"""
import json
import sys
import time

# ----------------------------------------------------------------------------------------------------------------------- policy
PASSWORD_ROLES = {"password text"}
SECRET_WORDS = ("password", "passphrase", "passcode", "secret", "token", "api key", "apikey", "cvv", "cvc", "card number", "otp", "one-time", "recovery key", "private key", "pin code")
SKIP_IF_EMPTY = {"filler", "panel", "section", "redundant object", "unknown", "scroll pane", "viewport", "layered pane", "split pane", "tool bar", "menu bar", "page tab list", "list", "table", "tree", "tree table", "status bar"}
TEXTY_ROLES = {"text", "entry", "password text", "document frame", "document web", "document text", "document spreadsheet", "document presentation", "document email", "terminal", "paragraph", "heading", "label", "static", "caption", "comment", "editbar", "link", "list item", "table cell", "menu item", "push button", "toggle button", "check box", "radio button", "page tab", "combo box"}
FRAME_ROLES = {"frame", "dialog", "window", "alert", "file chooser", "color chooser", "font chooser", "document frame"}
SHOWN_STATES = ("showing",)


def looks_secret(role, name):
    """A text field whose NAME says it holds a secret is treated like a password field."""
    if role in PASSWORD_ROLES:
        return True
    if role in ("text", "entry", "combo box", "document frame", "document web"):
        n = (name or "").lower()
        return any(w in n for w in SECRET_WORDS)
    return False


# ----------------------------------------------------------------------------------------------------------------------- node adapter
class Node:
    """The interface the walkers use. AtspiNode implements it over real AT-SPI; tests use fakes."""
    def role(self): raise NotImplementedError
    def name(self): raise NotImplementedError
    def description(self): return ""
    def states(self): return set()
    def extents(self): return None                       # (x, y, w, h) relative to the toplevel window, or None
    def child_count(self): return 0
    def child(self, i): return None
    def text(self, limit): return None                   # (str, total_chars) or None
    def actions(self): return []


class AtspiNode(Node):
    def __init__(self, atspi, acc):
        self.A, self.acc = atspi, acc

    def role(self):
        try: return (self.acc.get_role_name() or "").lower()
        except Exception: return "unknown"

    def name(self):
        try: return self.acc.get_name() or ""
        except Exception: return ""

    def description(self):
        try: return self.acc.get_description() or ""
        except Exception: return ""

    def states(self):
        out = set()
        try:
            ss = self.acc.get_state_set(); S = self.A.StateType
            for label, st in (("showing", S.SHOWING), ("visible", S.VISIBLE), ("focused", S.FOCUSED), ("checked", S.CHECKED), ("selected", S.SELECTED),
                              ("expanded", S.EXPANDED), ("editable", S.EDITABLE), ("modal", S.MODAL), ("enabled", S.ENABLED), ("sensitive", S.SENSITIVE),
                              ("pressed", S.PRESSED), ("required", S.REQUIRED)):
                if ss.contains(st): out.add(label)
        except Exception:
            pass
        return out

    def extents(self):
        try:
            comp = self.acc.get_component_iface()
            if comp is None: return None
            r = comp.get_extents(self.A.CoordType.WINDOW)
            return (r.x, r.y, r.width, r.height)
        except Exception:
            return None

    def child_count(self):
        try: return self.acc.get_child_count()
        except Exception: return 0

    def child(self, i):
        try:
            c = self.acc.get_child_at_index(i)
            return AtspiNode(self.A, c) if c is not None else None
        except Exception:
            return None

    def text(self, limit):
        try:
            if self.acc.get_text_iface() is None: return None
            n = self.A.Text.get_character_count(self.acc)                    # PyGObject exposes the interface functions statically
            if n <= 0: return ("", 0)
            return (self.A.Text.get_text(self.acc, 0, min(n, limit)) or "", n)
        except Exception:
            return None

    def actions(self):
        try:
            if self.acc.get_action_iface() is None: return []
            return [self.A.Action.get_action_name(self.acc, i) for i in range(self.A.Action.get_n_actions(self.acc))]   # names only: actions are listed, never invoked
        except Exception:
            return []


# ----------------------------------------------------------------------------------------------------------------------- walking
class Budget:
    def __init__(self, max_nodes, deadline_s):
        self.max_nodes, self.left, self.t_end, self.truncated = max_nodes, max_nodes, time.time() + deadline_s, False

    def spend(self):
        if self.left <= 0 or time.time() > self.t_end:
            self.truncated = True
            return False
        self.left -= 1
        return True


def clean(s, limit):
    s = (s or "").replace("\r", " ").replace("\n", " ⏎ ").replace("\t", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def element(node, idx, parent, frame, origin, adj, text_limit, role, name, states, ext):
    """One output element, rectangle already in output pixels. Secrets are never read."""
    fx, fy = frame
    ox, oy = origin
    x, y, w, h = ext
    rx, ry = ox + (x - fx) + adj[0], oy + (y - fy) + adj[1]
    e = {"i": idx, "p": parent, "role": role, "name": clean(name, 120), "rect": [rx, ry, w, h], "c": [rx + w // 2, ry + h // 2]}
    st = sorted(s for s in states if s in ("focused", "checked", "selected", "expanded", "editable", "modal", "pressed", "required"))
    if "enabled" not in states and "sensitive" not in states and states:
        st.append("disabled")
    if st: e["st"] = st
    acts = node.actions()
    if acts: e["act"] = acts[:4]
    if looks_secret(role, name):
        e["secret"] = True                                               # name and position are shown, the content never is
    elif role in TEXTY_ROLES:
        t = node.text(text_limit)
        if t and t[0] and clean(t[0], text_limit) != clean(name, text_limit):
            e["text"] = clean(t[0], text_limit)
            if t[1] > text_limit: e["chars"] = t[1]
    return e


def walk_window(frame_node, origin, adj, max_nodes=300, max_depth=16, text_limit=160, deadline_s=5.0):
    """Flat list of visible elements of one toplevel window, in tree order."""
    fext = frame_node.extents() or (0, 0, 0, 0)
    frame = (fext[0], fext[1]); fw, fh = fext[2], fext[3]
    budget, out = Budget(max_nodes, deadline_s), []

    def visit(node, depth, parent_idx):
        if depth > max_depth: budget.truncated = True; return
        n = node.child_count()
        for i in range(min(n, 2000)):
            c = node.child(i)
            if c is None: continue
            if not budget.spend(): return
            states = c.states()
            if "showing" not in states:                                  # not on screen: skip it and everything below
                continue
            ext = c.extents()
            role, name = c.role(), c.name()
            keep = False
            if ext and ext[2] > 0 and ext[3] > 0:
                inside = ext[0] < frame[0] + fw and ext[1] < frame[1] + fh and ext[0] + ext[2] > frame[0] and ext[1] + ext[3] > frame[1]
                has_content = bool(name) or bool(c.actions()) or role in ("text", "entry", "password text", "document frame", "document web", "terminal", "separator", "image", "icon", "progress bar", "slider", "spin button")
                keep = inside and (role not in SKIP_IF_EMPTY or has_content) and role != "unknown"
            idx = parent_idx
            if keep:
                out.append(element(c, len(out), parent_idx, frame, origin, adj, text_limit, role, name, states, ext))
                idx = len(out) - 1
            visit(c, depth + 1, idx)
            if budget.left <= 0: budget.truncated = True; return

    visit(frame_node, 0, -1)
    return out, budget.truncated, (fext[2], fext[3])


def frames_of(app):
    res = []
    for i in range(min(app.child_count(), 200)):
        c = app.child(i)
        if c is not None and c.role() in FRAME_ROLES and "showing" in c.states():
            res.append(c)
    return res


def pick_frame(frames, title):
    """The AT-SPI window that corresponds to a Hyprland window (same process; several windows -> match the title)."""
    if not frames: return None
    t = (title or "").strip().lower()
    for f in frames:
        if t and f.name().strip().lower() == t: return f
    for f in frames:
        if t and (t in f.name().strip().lower() or f.name().strip().lower() in t) and f.name().strip(): return f
    return frames[0] if len(frames) == 1 else None


def hidden_by_user(target, hide):
    c = (target.get("cls") or "").lower()
    return any(h and h.lower() in c for h in hide or [])


def read_windows(apps, targets, req):
    """apps: list of (pid, app_name, Node). Returns a list of per-target dicts."""
    out = []
    for t in targets:
        d = {"address": t.get("address", ""), "cls": t.get("cls", ""), "title": t.get("title", "")}
        if hidden_by_user(t, req.get("hide")):
            d["error"] = "hidden by the user"; out.append(d); continue
        cands = [n for (pid, nm, n) in apps if pid == t.get("pid")] or [n for (pid, nm, n) in apps if nm and nm.lower() == (t.get("cls") or "").lower()]
        if not cands:
            d["error"] = "no accessibility tree for this application (is accessibility enabled for it? see overlay/README.md)"; out.append(d); continue
        frames = [f for a in cands for f in frames_of(a)]
        fr = pick_frame(frames, t.get("title"))
        if fr is None:
            d["error"] = "cannot tell which accessible window this is (%d candidates)" % len(frames) if frames else "the application exposes no visible window"
            out.append(d); continue
        els, trunc, size = walk_window(fr, (t.get("ox", 0), t.get("oy", 0)), (t.get("dx", 0), t.get("dy", 0)), int(req.get("max_nodes", 300)), int(req.get("max_depth", 16)), int(req.get("text_limit", 160)))
        d.update({"elements": els, "truncated": trunc, "size": list(size)}); out.append(d)
    return out


def op_text(windows, limit):
    """Everything readable in the windows as lines of text, in reading order, size-capped."""
    total, res = 0, []
    for wi, w in enumerate(windows):
        items = []
        for e in w.get("elements", []):
            s = e.get("text") or (e["name"] if e["role"] in ("label", "static", "heading", "paragraph", "link", "push button", "menu item", "page tab", "check box", "radio button", "list item", "table cell", "caption", "toggle button") else "")
            if not s or e.get("secret"): continue
            if total + len(s) > limit: items.append({"note": "text truncated"}); break
            total += len(s); items.append({"role": e["role"], "name": e["name"], "text": s, "rect": e["rect"]})
        res.append({"address": w["address"], "cls": w["cls"], "title": w["title"], "items": items, "error": w.get("error")})
    return res


def op_find(windows, query, role, limit=25):
    q = (query or "").strip().lower()
    hits = []
    for wi, w in enumerate(windows):
        for e in w.get("elements", []):
            if role and e["role"] != role.lower(): continue
            hay = (e["name"] + " " + ("" if e.get("secret") else e.get("text", ""))).lower()
            if q and q in hay:
                hits.append({"window": wi, "address": w["address"], "cls": w["cls"], **{k: e[k] for k in ("role", "name", "rect", "c") if k in e}, **({"act": e["act"]} if "act" in e else {}), **({"st": e["st"]} if "st" in e else {})})
                if len(hits) >= limit: return hits
    return hits


def op_focused(windows):
    for wi, w in enumerate(windows):
        for e in w.get("elements", []):
            if "focused" in e.get("st", []) and e["role"] not in FRAME_ROLES:
                return {"window": wi, "address": w["address"], "cls": w["cls"], **{k: e[k] for k in ("role", "name", "rect", "c", "st") if k in e}, **({"text": e["text"]} if "text" in e and not e.get("secret") else {}), **({"secret": True} if e.get("secret") else {})}
    return None


def handle(req, apps_fn):
    t0 = time.time(); op = req.get("op")
    rid = req.get("id")
    try:
        if op not in ("apps", "tree", "text", "find", "focused"): return {"id": rid, "ok": False, "error": "unknown op"}
        targets = req.get("targets") or []
        if op != "apps" and (not isinstance(targets, list) or len(targets) > 32 or not all(isinstance(t, dict) for t in targets)): return {"id": rid, "ok": False, "error": "bad targets"}
        apps = apps_fn()
        if op == "apps":                                   # which of THESE windows can be read? (never lists applications nobody asked about)
            res = []
            for t in targets:
                d = {"address": t.get("address", ""), "cls": t.get("cls", "")}
                if hidden_by_user(t, req.get("hide")): d["error"] = "hidden by the user"
                else:
                    cands = [(nm, n) for (pid, nm, n) in apps if pid == t.get("pid")] or [(nm, n) for (pid, nm, n) in apps if nm and nm.lower() == (t.get("cls") or "").lower()]
                    d["readable"] = any(frames_of(n) for nm, n in cands)
                    if cands: d["app"] = cands[0][0]
                res.append(d)
            return {"id": rid, "ok": True, "op": op, "apps": res}
        wins = read_windows(apps, targets, req)
        if op == "tree":
            res = {"windows": wins}
        elif op == "text":
            res = {"windows": op_text(wins, int(req.get("total_text_limit", 8000)))}
        elif op == "find":
            res = {"matches": op_find(wins, req.get("query", ""), req.get("role"))}
            errs = [w["error"] for w in wins if w.get("error")]
            if errs: res["errors"] = errs
        elif op == "focused":
            res = {"focused": op_focused(wins)}
        else:
            return {"id": rid, "ok": False, "error": "unknown op"}
        return {"id": rid, "ok": True, "op": op, "ms": int((time.time() - t0) * 1000), **res}
    except Exception as ex:                                                  # never die on a hostile / broken application
        return {"id": rid, "ok": False, "error": str(ex) if isinstance(ex, NoBus) else "%s: %s" % (type(ex).__name__, ex)}


class NoBus(Exception):
    pass


def a11y_bus_available():
    """libatspi ABORTS the whole process when the accessibility bus is missing, so ask the session bus first."""
    try:
        import gi
        from gi.repository import Gio, GLib
        conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        r = conn.call_sync("org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus", "GetAddress", None, GLib.VariantType("(s)"), Gio.DBusCallFlags.NONE, 1500, None)
        return bool(r.unpack()[0])
    except Exception:
        return False


def atspi_apps():
    if not a11y_bus_available():
        raise NoBus("the accessibility bus is not available (is at-spi2-core running in your session?)")
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
    if not getattr(atspi_apps, "ready", False):
        Atspi.init()
        try: Atspi.set_timeout(1000, 1000)                                  # a hung application must not hang us for 25 s
        except Exception: pass
        atspi_apps.ready = True
    d = Atspi.get_desktop(0)
    apps = []
    for i in range(d.get_child_count()):
        a = d.get_child_at_index(i)
        if a is None: continue
        try: apps.append((a.get_process_id(), a.get_name() or "", AtspiNode(Atspi, a)))
        except Exception: pass
    return apps


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict): raise ValueError("request must be an object")
        except Exception as ex:
            print(json.dumps({"id": None, "ok": False, "error": "bad request: %s" % ex}), flush=True); continue
        print(json.dumps(handle(req, atspi_apps), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
