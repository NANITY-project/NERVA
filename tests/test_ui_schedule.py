#!/usr/bin/env python3
"""
tests/test_ui_schedule.py - UI/server.py with a SCHEDULE: generate() chaining several actions, and the /api/plan, /api/lm and /api/generate endpoints.

  python tests/test_ui_schedule.py            needs numpy only (no torch, no NEON, no libnebula, no model weights)

What is real: UI/server.py (generate, parse_segments, Stitcher, the HTTP handler, status), UI/lang.py, and the canonicalize / to_world functions of tools/train_cvqvae.py (taken out of that
file with ast, because the file itself imports torch).
What is a stand-in: the prior (a counter that records the context it is given), the NEON session (decode returns a walk that moves forward for 'walk', a stand for everything else),
the body engine (none: the page then draws a preview skeleton, which is the code path without libnebula).

  1 schedule     one segment per step; the prior is reset once per segment and its first context token is THAT action; decode gets the right action per window; the code grid runs on
                 across segments; the stitcher is not reset (continuous frames; the world position keeps advancing); frame count = 64 + (windows - 1) * 32
  2 refusals     an action the VAE does not know is refused with the list of known ones; a malformed schedule is refused; the single-action path still works, endless included
  3 http         /api/plan and /api/lm need the X-NERVA-UI header; a keyword plan without a model says so; an OpenAI-style endpoint is accepted and shown in /api/status WITHOUT its key;
                 a bad endpoint is refused and the old model kept; /api/generate?schedule=... streams segment events
"""
import ast, http.server, importlib.util, json, os, sys, threading, types, urllib.parse, urllib.request, urllib.error, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "UI"))
FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# ---- the real numeric helpers of train_cvqvae.py, without importing torch
src = open(os.path.join(ROOT, "tools", "train_cvqvae.py")).read(); tree = ast.parse(src)
want_fn, want_var = {"canonicalize", "to_world", "stitch"}, {"NCH", "XZ", "YAWS", "FLAGS"}
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want_fn) or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want_var for t in n.targets))]
shim = types.ModuleType("train_cvqvae_shim"); shim.np = np
exec(compile(ast.Module(body=keep, type_ignores=[]), "train_cvqvae(shim)", "exec"), shim.__dict__)
check(all(hasattr(shim, k) for k in ("canonicalize", "to_world", "YAWS", "FLAGS", "XZ")), "canonicalize / to_world / constants extracted from tools/train_cvqvae.py")

spec = importlib.util.spec_from_file_location("nerva_ui_server", os.path.join(ROOT, "UI", "server.py")); server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
server.ARGS = argparse.Namespace(neon_dir=None, neon_bin=None, no_clone=True, numpy_prior=False, no_engine=True, threads=0, trust_pickle=False, lm="", lm_model="", lm_key_env="")
server.train_cvqvae_mod = lambda: shim
server.get_engine = lambda: None
S = server.S

NAMES = ["walk", "run", "sit", "stand", "pickup", "gesture", "jump"]
K, NCODE, WINDOW, STRIDE = 32, 16, 64, 32


class FakePrior:
    SEP, ACT0, NULL = K, K + 1, K + 1 + len(NAMES)
    n, windows, stride, names, params = NCODE, 4, STRIDE, NAMES, 1
    n_actions, L, d = len(NAMES), [], 1                  # fields status() reads from the real Prior
    K = K

    def __init__(self):
        self.resets, self.first_tokens, self.rng_seeds = 0, [], []

    def reset(self):
        self.resets += 1

    def sample(self, ctx, rng, temperature, top_k, top_p):
        if len(ctx) == 1:
            self.first_tokens.append(ctx[0])               # the context a segment starts from: [action token]
        return int(rng.integers(0, K)), 1.0, 0.5


class FakeSession:
    name = "stand-in NEON"

    def __init__(self):
        self.decoded = []

    def decode(self, codes, action):
        self.decoded.append(action)
        w = np.zeros((WINDOW, 30), np.float32); t = np.arange(WINDOW) / 30.0
        w[:, 1] = 0.95
        if action == "walk":
            w[:, 2] = 1.2 * t                              # 1.2 m/s forward in the window's own frame
        for b in (4, 9):
            w[:, b + 1] = 0.0; w[:, b + 4] = 1.0
        return w

    def close(self):
        pass


def install():
    S.vae = dict(cfg=dict(window=WINDOW, downs=2, num_codes=K, n_actions=len(NAMES)), names=NAMES, dt=1 / 30, gguf="x", file="vae.pt")
    S.prior, S.prior_file = FakePrior(), "prior.pt"
    S.session, S.native_prior = FakeSession(), False


def run(q):
    install(); evs = list(server.generate(q, lambda: False)); return evs, S.prior, S.session


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 0
print("[Stitcher against the reference stitch()]")
rng = np.random.default_rng(5)
rw = []
for i in range(5):                                                                     # random-looking canonical windows: pelvis moves and turns, feet / hands wander, flags 0/1
    w = np.zeros((WINDOW, 30)); t_ = np.arange(WINDOW) / 30.0; yaw = rng.uniform(-1, 1) * np.linspace(0, 1, WINDOW) + 0.2 * np.sin(2 * t_)
    w[:, 3] = yaw; w[:, 1] = 0.95; sp = rng.uniform(0.5, 1.4); w[:, 0] = np.cumsum(sp / 30 * np.sin(yaw)); w[:, 2] = np.cumsum(sp / 30 * np.cos(yaw))
    for b in (4, 9):
        w[:, b] = w[:, 0] + 0.1; w[:, b + 2] = w[:, 2] + 0.2 * np.sin(3 * t_); w[:, b + 3] = yaw; w[:, b + 4] = (np.sin(3 * t_ + b) > 0)
    for b in (14, 18, 22, 26):
        w[:, b:b + 3] = w[:, [0, 1, 2]] + rng.normal(size=3) * 0.1; w[:, b + 3] = rng.random(WINDOW) > 0.5
    rw.append(w.astype(np.float32))
st_ = server.Stitcher(shim, WINDOW, STRIDE); got = np.concatenate([st_.add(w) for w in rw] + [st_.flush()])
ref = shim.stitch(rw, STRIDE)
check(got.shape == ref.shape and np.abs(got - ref).max() < 1e-5, f"incremental Stitcher == stitch() over 5 windows (max difference {np.abs(got - ref).max() if got.shape == ref.shape else 'shape'})")
st_ = server.Stitcher(shim, WINDOW, STRIDE); a1 = np.concatenate([st_.add(w) for w in rw[:3]] + [st_.begin_segment()]); a2 = np.concatenate([st_.add(w) for w in rw[3:]] + [st_.flush()])
r1, r2 = shim.stitch(rw[:3], STRIDE), shim.stitch(rw[3:], STRIDE)
check(len(a1) == len(r1) and np.abs(a1 - r1).max() < 1e-5, "begin_segment(): the first clip comes out exactly as stitch() of its own windows")
check(len(a2) == len(r2), f"the second clip has the length of its own stitch() ({len(a2)} frames)")
last = a1[-1].astype(np.float64); place = shim.to_world(shim.canonicalize(rw[3][None])[0].astype(np.float64), last[0], last[2], last[3])
for y in shim.YAWS:
    place[:, y] += 2 * np.pi * np.round((last[y] - place[0, y]) / (2 * np.pi))
cont = [c for c in range(30) if c not in shim.FLAGS]
check(np.abs(a2[8:32][:, cont] - place[8:32][:, cont]).max() < 1e-5, "after the 8 eased frames the new clip is exactly its own window placed at the old clip's last pose")
al = np.linspace(0, 1, 10)[1:-1][:, None]; want = last[cont] * (1 - al) + place[:8][:, cont] * al
check(np.abs(a2[:8][:, cont] - want).max() < 1e-5, "the 8 eased frames are a linear blend from the held last pose into the window")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1
print("[schedule]")
sched = [dict(action="walk", seconds=4), dict(action="pickup", seconds=3), dict(action="gesture", seconds=2)]
wins = [max(1, round(s["seconds"] / (STRIDE / 30))) for s in sched]
evs, pr, sess = run(dict(schedule=json.dumps(sched), seed="3"))
seg = [e for e in evs if e["type"] == "segment"]
check([e["action"] for e in seg] == ["walk", "pickup", "gesture"] and [e["windows"] for e in seg] == wins, f"segments {[(e['action'], e['windows']) for e in seg]}")
check(pr.resets == 3, f"the prior was reset once per segment ({pr.resets})")
check(pr.first_tokens == [FakePrior.ACT0 + NAMES.index(a) for a in ("walk", "pickup", "gesture")], f"each segment starts from its own action token ({pr.first_tokens})")
exp = [a for a, k in zip(("walk", "pickup", "gesture"), wins) for _ in range(k)]
check(sess.decoded == exp, f"decode got the right action for every window ({len(exp)} windows)")
cw = [e["w"] for e in evs if e["type"] == "code" and e["i"] == 0]
check(cw == list(range(sum(wins))), "the code grid index runs on across segments (no restart at a new action)")
check([e["w0"] for e in seg] == [0, wins[0], wins[0] + wins[1]], "segment events carry the grid row they start at")
start = evs[0]
check(start["type"] == "start" and start["windows"] == sum(wins) and [s["action"] for s in start["schedule"]] == ["walk", "pickup", "gesture"], "the start event announces the whole schedule")
frames = np.concatenate([np.asarray(e["act"], np.float32).reshape(-1, 30) for e in evs if e["type"] == "frames"])
total = sum(wins)
nseg = len(sched); want = nseg * WINDOW + (total - nseg) * STRIDE                    # each clip is its own stitch: 64 + (k-1)*32 frames
check(len(frames) == want, f"frames = sum over clips of 64 + (k-1)*32 = {want} (got {len(frames)}): none lost or doubled at a boundary")
check(evs[-1]["type"] == "done" and evs[-1]["frames"] == len(frames), "done event matches")
z = frames[:, 2]; boundary = WINDOW + (wins[0] - 1) * STRIDE                     # first frame of the second clip
check(bool(np.all(np.diff(z) > -1e-4)) and z[boundary - 1] > 4.0 and abs(z[-1] - z[boundary - 1]) < 1e-3,
      f"the walk's forward progress survives the switch: z {z[boundary - 1]:.2f} m when the walk ends, the next clips start there and never go back (final z {z[-1]:.2f})")
check(np.allclose(frames[boundary, [0, 2, 3]], frames[boundary - 1, [0, 2, 3]], atol=0.05), "the new clip's first frame is at the old clip's last pelvis pose (x, z, yaw)")
step = np.abs(np.diff(frames[:, [0, 2]], axis=0)).max()
check(step < 0.2, f"no jump in pelvis position between frames (largest step {step:.3f} m)")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2
print("[refusals and the single-action path]")
for bad, why in ((json.dumps([dict(action="fly", seconds=2)]), "unknown action"), ("not json", "malformed"), (json.dumps([]), "empty list"), (json.dumps([dict(action="walk")] * 40), "too many steps")):
    install()
    try:
        list(server.generate(dict(schedule=bad), lambda: False)); check(False, f"{why}: refused")
    except RuntimeError as e:
        check(True, f"{why}: refused ({str(e)[:70]})")
install()
try:
    list(server.generate(dict(schedule=json.dumps([dict(action="fly", seconds=2)])), lambda: False))
except RuntimeError as e:
    check("known:" in str(e) and "walk" in str(e), "the refusal lists the actions this VAE knows")
evs, pr, sess = run(dict(action="run", windows="3", seed="1"))
check(not [e for e in evs if e["type"] == "segment"] and sess.decoded == ["run"] * 3 and pr.resets == 1 and evs[-1]["type"] == "done", "single action: no segment events, 3 windows, one reset")
evs, pr, sess = run(dict(action="null", windows="2")); check(sess.decoded == [None, None] and pr.first_tokens == [FakePrior.NULL], "'null' = unconditional (NULL token, decode without an action)")
install(); n = [0]


def stop_after_5():
    n[0] += 1; return n[0] > 40


evs = list(server.generate(dict(action="walk", windows="0"), stop_after_5))
check(any(e["type"] == "window" for e in evs) and evs[-1]["type"] != "done", "endless mode runs until stopped")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3
print("[HTTP]")
install()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); srv.daemon_threads = True; threading.Thread(target=srv.serve_forever, daemon=True).start()
B = f"http://127.0.0.1:{srv.server_port}"


def call(path, body=None, headers=None, method=None):
    req = urllib.request.Request(B + path, None if body is None else json.dumps(body).encode(), headers or {}, method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


H = {"X-NERVA-UI": "1", "Content-Type": "application/json"}
check(call("/api/plan", dict(text="walk"))[0] == 403 and call("/api/lm", dict(kind="none"))[0] == 403, "/api/plan and /api/lm refuse a request without the X-NERVA-UI header (not callable from another website)")
code, b = call("/api/plan", dict(text="walk then sit"), H); j = json.loads(b)
check(j["ok"] and j["source"] == "keyword" and [s["action"] for s in j["schedule"]] == ["walk", "sit"] and "no language model" in j.get("note", ""), "no model loaded: a keyword plan, and the response says so")
code, b = call("/api/plan", dict(text="do a flip"), H); j = json.loads(b)
check(not j["ok"] and "allowed" in j["error"], "no usable action in the text -> ok:false and the allowed actions")


class OAI(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); OAI.seen.append((self.headers.get("Authorization"), body))
        last = body["messages"][-1]["content"]
        out = json.dumps({"choices": [{"message": {"content": "gesture 2\nrun 3" if "wave" in last else "ok"}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


oai = http.server.ThreadingHTTPServer(("127.0.0.1", 0), OAI); threading.Thread(target=oai.serve_forever, daemon=True).start()
code, b = call("/api/lm", dict(kind="http", base=f"http://127.0.0.1:{oai.server_port}/v1", model="m", key="sk-SECRET"), H); j = json.loads(b)
check(j["ok"] and j["status"]["lm"]["kind"] == "http", "an OpenAI-style endpoint is accepted (after a real test request)")
check("sk-SECRET" not in json.dumps(call("/api/status")[1].decode()) and "sk-SECRET" not in b.decode(), "the API key never appears in /api/status or the response")
check(OAI.seen and OAI.seen[0][0] == "Bearer sk-SECRET", "...but it is sent to the endpoint")
code, b = call("/api/plan", dict(text="wave at me"), H); j = json.loads(b)
check(j["ok"] and j["source"] == "model" and [s["action"] for s in j["schedule"]] == ["gesture", "run"], "plan through the model; the VAE's action names were what the model was given")
check(OAI.seen[-1][1]["messages"][0]["content"].count("walk, run, sit, stand, pickup, gesture, jump") == 1, "the model is told this VAE's actions, read at run time")
code, b = call("/api/lm", dict(kind="http", base="ftp://nope"), H); j = json.loads(b)
check(not j["ok"] and S.lm is not None and "http://" in j["error"], "a bad endpoint is refused and the working model is kept")
code, b = call("/api/lm", dict(kind="http", base="http://127.0.0.1:1/v1"), H); j = json.loads(b)
check(not j["ok"] and S.lm is not None, "an unreachable endpoint is refused and the working model is kept")
code, b = call("/api/lm", dict(kind="neon", path="/no/such.gguf"), H); check(not json.loads(b)["ok"], "a missing model file is refused")
code, b = call("/api/lm", dict(kind="bogus"), H); check(not json.loads(b)["ok"], "an unknown kind is refused")
q = "/api/generate?" + urllib.parse.urlencode(dict(schedule=json.dumps(sched), seed="1"))
code, b = call(q); events = [json.loads(l[6:]) for l in b.decode().split("\n\n") if l.startswith("data: ")]
check(code == 200 and [e["action"] for e in events if e["type"] == "segment"] == ["walk", "pickup", "gesture"] and events[-1]["type"] == "done", "/api/generate?schedule=... streams the segments and ends with done")
q = "/api/generate?" + urllib.parse.urlencode(dict(schedule=json.dumps([dict(action="fly", seconds=2)])))
code, b = call(q); events = [json.loads(l[6:]) for l in b.decode().split("\n\n") if l.startswith("data: ")]
check(events and events[-1]["type"] == "error" and "fly" in events[-1]["message"], "a bad schedule over HTTP ends in an error event the page can show")
code, b = call("/api/lm", dict(kind="none"), H); check(json.loads(b)["ok"] and S.lm is None, "kind 'none' removes the model")
srv.shutdown(); oai.shutdown()
print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED"); sys.exit(1)
print("all UI schedule checks passed")
