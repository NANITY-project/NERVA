#!/usr/bin/env python3
"""
NERVA UI server - drag a trained cVQ-VAE (best.pt) and prior (prior_best.pt) into the browser and watch motion being generated live.

    python3 UI/server.py                  # http://127.0.0.1:8770
    python3 UI/server.py --no-open --port 9000
    make ui                               # same, builds libnebula first

What runs where
  VAE      best.pt  -> exported with NEON's own conversion/nanity_export_cvqvae.py -> GGUF -> decoded by NEON
  Prior    prior_best.pt -> exported with conversion/nanity_export_cvqprior.py -> GGUF -> sampled by NEON
           Both run in ONE motion-only NEON process:  neon --motion-model vae.gguf --motion-model prior.gguf --interactive
           (ops cvq_prior_reset / cvq_prior_next with decode:true; no language model needed).
           Fallback when that NEON binary is missing or too old: the prior runs as a small numpy twin of tools/train_cvqvae.py's
           CodePrior inside this server and the VAE decodes through NEON's cvqvae_cli. The header shows which path is active.
  Body     decoded windows are stitched exactly like train_cvqvae.stitch (incremental) and replayed through libnebula
           (the IK engine) frame by frame, so the browser shows the engine's body. No libnebula -> the browser draws a preview skeleton
           from the 30 action channels instead.

Needs: python3, numpy, torch (only to read .pt files), gguf (the exporter), and the NANITY-NEON checkout
(if torch/gguf are missing when you drop a .pt, the page lists the Python environments it found - .venv, conda, ... - or makes a new .venv and installs them, then restarts itself; see torch_env.py)
(found next to this repo, or set NEON_DIR; cloned into UI/.cache if missing, unless --no-clone). A C++ compiler for libnebula / cvqvae_cli.
"""
import argparse, re, base64, ctypes, faulthandler, glob, http.server, importlib.util, json, math, os, shutil, subprocess, sys, threading, time, urllib.parse, webbrowser
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
CACHE = os.path.join(HERE, ".cache")
UPLOADS = os.path.join(CACHE, "uploads")
NCH = 30
FLAGS = [8, 13, 17, 21, 25, 29]
NEON_REPO_URL = "https://github.com/NANITY-project/NANITY-NEON.git"
sys.path.insert(0, HERE)
import torch_env                      # finds / creates a Python environment with torch (UI/torch_env.py)
import lang                           # prompt -> action schedule; the language model is a slot (UI/lang.py)
import goal_plan                      # text -> per-window goal rows: the Goal Translator (UI/goal_plan.py, tools/goal_translator)

ARGS = None
BOOT = time.time()                      # changes on every (re)start: the page uses it to see that the server came back
LOG = []
LOCK = threading.RLock()


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    LOG.append(line)
    del LOG[:-200]


# ================================================================================================================ NEON discovery
def find_neon_dir():
    cands = [ARGS.neon_dir, os.environ.get("NEON_DIR"), os.path.join(os.path.dirname(ROOT), "NANITY-NEON"), os.path.join(os.path.dirname(ROOT), "NEON"),
             os.path.join(os.path.dirname(ROOT), "NEON-Runtime-Beta-Variation"), os.path.expanduser("~/NANITY-NEON"), os.path.expanduser("~/NEON"),
             os.path.join(CACHE, "NANITY-NEON")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "conversion", "nanity_export_cvqvae.py")):
            return os.path.abspath(c)
    if not ARGS.no_clone and shutil.which("git"):
        os.makedirs(CACHE, exist_ok=True)
        dst = os.path.join(CACHE, "NANITY-NEON")
        log(f"NEON checkout not found; cloning {NEON_REPO_URL} into {dst} (use --neon-dir / --no-clone to avoid this)")
        r = subprocess.run(["git", "clone", "--depth", "1", NEON_REPO_URL, dst], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(os.path.join(dst, "conversion", "nanity_export_cvqvae.py")):
            return dst
        log("git clone failed: " + (r.stderr or "").strip()[-300:])
    return None


def cpu_flags():
    try:
        t = open("/proc/cpuinfo").read()
        return set(t.split("flags")[1].split(":")[1].split("\n")[0].split())
    except Exception:
        return set()


def simd_flags():
    f = cpu_flags()
    out = []
    if "avx2" in f and "fma" in f:
        out += ["-mavx2", "-mfma"]
        if "f16c" in f:
            out += ["-mf16c"]
    return out


def find_neon_bin(neon_dir):
    cands = [ARGS.neon_bin, os.environ.get("NEON_BIN"), shutil.which("neon")]
    if neon_dir:
        cands += [os.path.join(neon_dir, p) for p in ("neon", "build/neon", "../neon")]
    cands += [os.path.join(os.path.dirname(ROOT), "neon"), os.path.expanduser("~/.local/bin/neon")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            try:
                r = subprocess.run([c, "--help"], capture_output=True, text=True, timeout=8)
                if "motion-model" in (r.stdout + r.stderr):
                    return os.path.abspath(c)
            except Exception:
                pass
    return None


def build_cli(neon_dir):
    """Compile NEON's standalone cvqvae_cli (small, a few seconds). Returns its path or None."""
    src = os.path.join(neon_dir, "runtime", "motion", "cvqvae_cli.cpp")
    if not os.path.exists(src):
        return None
    out = os.path.join(CACHE, "cvqvae_cli")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return None
    os.makedirs(CACHE, exist_ok=True)
    log("compiling NEON cvqvae_cli ...")
    r = subprocess.run([cxx, "-std=c++20", "-O2", "-pthread", *simd_flags(), src, "-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        log("cvqvae_cli build failed:\n" + r.stderr[-1500:])
        return None
    return out


# ================================================================================================================ decoders (NEON)
class NeonInteractive:
    """neon --motion-model X.gguf --interactive; JSON lines in, {"motion":{...}} + {"done":true} out."""
    name = "NEON --interactive"

    def __init__(self, neon_bin, ggufs, threads=0):
        cmd = [neon_bin]
        for g in ([ggufs] if isinstance(ggufs, str) else ggufs):
            cmd += ["--motion-model", g]
        cmd += ["--interactive"]
        if threads:
            cmd += ["--threads", str(threads)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.ready = threading.Event()
        self.err = []
        threading.Thread(target=self._pump_err, daemon=True).start()
        if not self.ready.wait(30):
            self.close()
            raise RuntimeError("NEON did not report ready within 30 s: " + " | ".join(self.err[-4:]))
        if self.p.poll() is not None:
            raise RuntimeError("NEON exited: " + " | ".join(self.err[-4:]))

    def _pump_err(self):
        for line in self.p.stderr:
            line = line.rstrip()
            self.err.append(line)
            del self.err[:-50]
            if "ready" in line and "Interactive" in line:
                self.ready.set()
        self.ready.set()

    def ask(self, motion):
        self.p.stdin.write(json.dumps({"motion": motion}) + "\n")
        self.p.stdin.flush()
        reply, error = None, None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("NEON closed the pipe: " + " | ".join(self.err[-4:]))
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if "motion" in j:
                reply = j["motion"]
            if "error" in j:
                error = j["error"]
            if j.get("done"):
                break
        if error:
            raise RuntimeError(error)
        return reply

    def decode(self, codes, action):
        r = self.ask({"op": "cvq_decode", "codes": [int(c) for c in codes], "action": action if action is not None else "null"})
        return np.asarray(r["frames"], np.float32)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass


class NeonCli:
    """NEON's cvqvae_cli, one process per window (it mmaps the GGUF, so this is quick)."""
    name = "NEON cvqvae_cli"

    def __init__(self, cli, gguf):
        self.cli, self.gguf = cli, gguf

    def decode(self, codes, action):
        r = subprocess.run([self.cli, self.gguf, "decode", "--codes", ",".join(str(int(c)) for c in codes), "--action", action or "null"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "cvqvae_cli failed")
        rows = [[float(x) for x in ln.split(",")] for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("#")]
        return np.asarray(rows, np.float32)

    def close(self):
        pass


def start_session(ggufs, has_prior, neon_dir):
    """Returns (session, native_prior, native_goal). Prefers one NEON process holding VAE (+ prior); falls back to cvqvae_cli + the numpy prior.
    native_goal: this NEON reports the prior's goal numbers in `info` (the build that takes "goal" in cvq_prior_next). An older build ignores a goal without a word, so a run steered by
    numbers must not go to it: generate() uses the numpy prior then."""
    nb = find_neon_bin(neon_dir)
    if nb:
        for use in ((ggufs, True), (ggufs[:1], False)) if has_prior else ((ggufs, False),):
            files, want_prior = use
            try:
                d = NeonInteractive(nb, files, ARGS.threads)
                probe = d.decode([0] * 4, None)               # a real round trip: proves the cvq ops are wired in this binary
                if not (probe.ndim == 2 and probe.shape[1] == NCH):
                    d.close()
                    continue
                native, native_goal = False, False
                if want_prior and not ARGS.numpy_prior:
                    try:
                        d.ask({"op": "cvq_prior_reset", "action": "null", "seed": 0})
                        native = True
                    except Exception as e:
                        log(f"this NEON has no usable prior op ({e}); the prior will run in numpy")
                    if native:
                        try:
                            native_goal = bool(((d.ask({"op": "info"}) or {}).get("cvqprior") or {}).get("goal"))
                        except Exception:
                            native_goal = False
                log(f"NEON: {nb}  (VAE decode on NEON, prior on {'NEON' if native else 'numpy'}{', goal numbers on NEON' if native_goal else ''})")
                return d, native, native_goal
            except Exception as e:
                log(f"NEON --interactive failed ({e}); " + ("retrying without the prior GGUF" if want_prior else "trying cvqvae_cli"))
    cli = build_cli(neon_dir) if neon_dir else None
    if cli:
        d = NeonCli(cli, ggufs[0])
        d.decode([0] * 4, None)
        log(f"NEON: {d.name} (VAE decode on NEON, prior on numpy). Build a current NEON (see NANITY-NEON README) to run the prior on NEON too.")
        return d, False, False
    raise RuntimeError("no NEON available: build NEON (g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/NEON-3.cpp -o neon) and set NEON_BIN, "
                       "or install a C++ compiler so cvqvae_cli can be built")


# ================================================================================================================ the prior, in numpy
_erf = np.vectorize(math.erf, otypes=[np.float32])


def _ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class _Cache:
    """The KV cache of one branch of the prior (how many tokens it has been fed, the logits after the last one)."""

    def __init__(self, nl, H, maxT, dh):
        self.kc = [np.zeros((H, maxT, dh), np.float32) for _ in range(nl)]
        self.vc = [np.zeros((H, maxT, dh), np.float32) for _ in range(nl)]
        self.t, self.last = 0, None


class Prior:
    """Inference twin of train_cvqvae.CodePrior: token ids  codes 0..K-1 | SEP=K | actions K+1.. | NULL | PAD. KV-cached, one token at a time.

    Goal / style conditioning, exactly as the trainer and NEON's rawllm_cvqprior.hpp do it (a prior trained without them has neither and runs unchanged):
      goal   every window has goal_dim RAW numbers (goal_names: dyaw, dx, dz, speed). They are normalised with the stored mean / std, embedded by goal_mlp (Linear-GELU-Linear) and ADDED
             to the input embedding of every position of that window (position i belongs to window i // (n + 1), the SEP in front of a window's codes included). A window without a goal gets
             the learned goal_null vector. add_window_goal() declares the goal of the next window of the context; reset() forgets them.
      style  one id per sequence (n_styles = NULL), embedded and added to every position.
      cfg    set_cfg_scale(s != 1), with a goal on the current window: logits = uncond + s * (cond - uncond), uncond = the goal nulled (action and style kept) in a second KV cache."""

    def __init__(self, sd, cfg, names, stride, goal_stats=None, goal_names=None, style_names=None):
        self.cfg = dict(cfg)
        self.K, self.n_actions, self.n, self.windows = int(cfg["K"]), int(cfg["n_actions"]), int(cfg["n"]), int(cfg["windows"])
        self.d, self.H, nl = int(cfg["d"]), int(cfg["heads"]), int(cfg["layers"])
        self.SEP, self.ACT0 = self.K, self.K + 1
        self.NULL = self.K + 1 + self.n_actions
        self.names, self.stride = list(names or []), int(stride) if stride else 0
        W = {k: np.ascontiguousarray(v, np.float32) for k, v in sd.items()}
        self.tok, self.pos = W["tok.weight"], W["pos.weight"]
        self.maxT = self.pos.shape[0]
        g = lambda i, k: W[f"blocks.{i}.{k}"]
        self.L = [dict(n1w=g(i, "n1.weight"), n1b=g(i, "n1.bias"), qw=g(i, "qkv.weight"), qb=g(i, "qkv.bias"), pw=g(i, "proj.weight"), pb=g(i, "proj.bias"),
                       n2w=g(i, "n2.weight"), n2b=g(i, "n2.bias"), f0w=g(i, "ff.0.weight"), f0b=g(i, "ff.0.bias"), f2w=g(i, "ff.2.weight"), f2b=g(i, "ff.2.bias")) for i in range(nl)]
        self.lnw, self.lnb, self.head = W["ln.weight"], W["ln.bias"], W["head.weight"]
        self.dh = self.d // self.H
        self.goal_dim, self.n_styles = int(cfg.get("goal_dim", 0) or 0), int(cfg.get("n_styles", 0) or 0)
        self.goal_names, self.goal_stats, self.style_names = [], None, list(style_names or [])
        if self.goal_dim:
            st = goal_stats or {}
            if len(st.get("mean", [])) != self.goal_dim or len(st.get("std", [])) != self.goal_dim:
                raise RuntimeError(f"this prior takes {self.goal_dim} goal numbers per window but the checkpoint has no goal_stats (mean / std) for them: save it again with train_cvqvae.py")
            self.goal_stats = dict(st)
            self.g_mean, self.g_std = np.asarray(st["mean"], np.float64), np.asarray(st["std"], np.float64)
            self.goal_names = list(goal_names or (["dyaw", "dx", "dz", "speed"] if self.goal_dim == 4 else [f"g{i}" for i in range(self.goal_dim)]))
            self.gw0, self.gb0, self.gw2, self.gb2, self.goal_null = W["goal_mlp.0.weight"], W["goal_mlp.0.bias"], W["goal_mlp.2.weight"], W["goal_mlp.2.bias"], W["goal_null"]
        if self.n_styles:
            self.style_emb = W["style_emb.weight"]
        self.params = sum(v.size for v in W.values())
        self.cfg_scale = 1.0
        self.reset()

    def reset(self, style=None):
        """A new sequence: empty caches, no window goals, style id (None / out of range = the NULL style). The guidance scale is kept."""
        self.c = [_Cache(len(self.L), self.H, self.maxT, self.dh), _Cache(len(self.L), self.H, self.maxT, self.dh) if self.goal_dim else None]
        self.win_emb = []                                      # per window slot: the goal embedding, or None = the null goal
        self.style = self.n_styles if (style is None or not 0 <= int(style) < self.n_styles) else int(style)

    def add_window_goal(self, raw):
        """Declare the goal of the NEXT window of the context: call it once per window, after reset() / after the oldest window was dropped, and before that window's SEP and codes are fed.
        raw = goal_dim RAW numbers (goal_names order) or None = this window has no goal."""
        if raw is None:
            self.win_emb.append(None)
            return
        if not self.goal_dim:
            raise RuntimeError("this prior was trained without goal numbers (train again without --no-prior-goal)")
        g = np.asarray(raw, np.float64).reshape(-1)
        if g.shape[0] != self.goal_dim or not np.all(np.isfinite(g)):
            raise RuntimeError(f"a goal is {self.goal_dim} finite numbers ({', '.join(self.goal_names)})")
        h = ((g - self.g_mean) / self.g_std).astype(np.float32) @ self.gw0.T + self.gb0
        h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
        self.win_emb.append(h @ self.gw2.T + self.gb2)

    def set_cfg_scale(self, s):
        s = float(s)
        if not math.isfinite(s) or s < 0.0 or s > 20.0:
            raise RuntimeError("cfg must be in [0, 20] (1 = off)")
        if s != 1.0 and not self.goal_dim:
            raise RuntimeError("guidance (cfg) needs a prior trained with goal numbers")
        self.cfg_scale = s

    def _cfg_on(self):
        return self.cfg_scale != 1.0 and self.goal_dim > 0 and bool(self.win_emb) and self.win_emb[-1] is not None

    def _feed(self, br, tok):
        c = self.c[br]
        t = c.t
        if t >= self.maxT:
            raise RuntimeError("prior context overflow (internal error)")
        x = self.tok[tok] + self.pos[t]
        if self.goal_dim:
            slot = min(t // (self.n + 1), self.windows - 1)
            e = self.win_emb[slot] if (br == 0 and slot < len(self.win_emb)) else None     # branch 1 (the guidance one) sees the null goal everywhere
            x = x + (self.goal_null if e is None else e)
        if self.n_styles:
            x = x + self.style_emb[self.style]
        for i, L in enumerate(self.L):
            h = _ln(x, L["n1w"], L["n1b"])
            qkv = (h @ L["qw"].T + L["qb"]).reshape(3, self.H, self.dh)
            c.kc[i][:, t], c.vc[i][:, t] = qkv[1], qkv[2]
            att = np.einsum("hd,htd->ht", qkv[0], c.kc[i][:, :t + 1]) / math.sqrt(self.dh)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = np.einsum("ht,htd->hd", att, c.vc[i][:, :t + 1]).reshape(self.d)
            x = x + y @ L["pw"].T + L["pb"]
            h = _ln(x, L["n2w"], L["n2b"]) @ L["f0w"].T + L["f0b"]
            h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
            x = x + h @ L["f2w"].T + L["f2b"]
        c.t += 1
        c.last = _ln(x, self.lnw, self.lnb) @ self.head.T

    def feed(self, tok):
        self._feed(0, tok)

    @property
    def t(self):
        return self.c[0].t

    @property
    def last(self):
        return self.c[0].last

    def _upto(self, br, ctx):
        c = self.c[br]
        while c.t < len(ctx):
            self._feed(br, ctx[c.t])
        return c.last

    def logits(self, ctx):
        lc = self._upto(0, ctx)
        if self._cfg_on():
            lu = self._upto(1, ctx)
            return (lu + self.cfg_scale * (lc - lu))[:self.K]
        return lc[:self.K]

    def sample(self, ctx, rng, temperature, top_k, top_p):
        raw = self.logits(ctx).astype(np.float64)
        p0 = _softmax(raw)
        ent = float(-(p0 * np.log(p0 + 1e-12)).sum())
        lg = raw / max(temperature, 1e-4)
        if top_k and top_k < self.K:
            kth = np.partition(lg, -top_k)[-top_k]
            lg = np.where(lg < kth, -np.inf, lg)
        if top_p < 1.0:
            si = np.argsort(-lg)
            pr = _softmax(lg[si])
            rm = (np.cumsum(pr) - pr) > top_p
            keep = np.full(self.K, -np.inf)
            keep[si[~rm]] = lg[si[~rm]]
            lg = keep
        p = _softmax(lg)
        tok = int(rng.choice(self.K, p=p / p.sum()))
        return tok, ent, float(p0[tok])


# ================================================================================================================ stitching (incremental twin of train_cvqvae.stitch)
class Stitcher:
    def __init__(self, tc, window, stride):
        self.tc, self.W, self.ov = tc, window, window - stride
        if self.ov <= 0:
            raise RuntimeError("stride must be smaller than the window (windows must overlap)")
        self.tail, self.base, self.emitted = None, 0, 0
        self.last = None                       # last frame emitted before begin_segment(): where the next clip starts

    def begin_segment(self, ease=8):
        """The next window belongs to a NEW clip (a different action). The prior knows nothing about the old clip, so cross-fading the new window onto the old tail would pull the
        body back to where the overlap began (up to a metre for a walk). Instead: the old clip is emitted to its end (returned), and the next add() puts the new window at the body's
        LAST pose and eases the pose of the first `ease` frames from that pose into the window's own. Limbs may still jump a little; the engine reports what it has to correct."""
        rest = self.flush()
        if self.tail is not None:
            self.last = self.tail[-1].copy()
        self.tail, self.base, self.ease = None, self.emitted, ease
        return rest

    def add(self, win):
        tc, ov = self.tc, self.ov
        w = tc.canonicalize(np.asarray(win, np.float32)[None])[0].astype(np.float64)
        if self.tail is None:
            out = w
            if self.last is not None:                                  # first window of a new clip: start at the pose the old clip ended in
                last = self.last
                out = tc.to_world(w, last[0], last[2], last[3])
                for y in tc.YAWS:
                    out[:, y] += 2 * np.pi * np.round((last[y] - out[0, y]) / (2 * np.pi))
                al = np.linspace(0, 1, self.ease + 2)[1:-1][:, None]
                e = len(al)
                blend = last[None, :] * (1 - al) + out[:e] * al
                blend[:, FLAGS] = np.where(al > 0.5, out[:e][:, FLAGS], last[FLAGS][None, :])
                out = np.concatenate([blend, out[e:]])
                self.last = None
        else:
            out = self.tail
            b = len(out) - ov
            w = tc.to_world(w, out[b, 0], out[b, 2], out[b, 3])
            for y in tc.YAWS:
                w[:, y] += 2 * np.pi * np.round((out[b, y] - w[0, y]) / (2 * np.pi))
            al = np.linspace(0, 1, ov + 2)[1:-1][:, None]
            old, new = out[b:], w[:ov]
            blend = old * (1 - al) + new * al
            blend[:, FLAGS] = np.where(al > 0.5, new[:, FLAGS], old[:, FLAGS])
            out = np.concatenate([out[:b], blend, w[ov:]])
        final = self.base + len(out) - ov
        fresh = out[self.emitted - self.base:final - self.base]
        self.emitted = final
        keep = len(out) - ov
        self.tail, self.base = out[keep:], self.base + keep
        return fresh.astype(np.float32)

    def flush(self):
        if self.tail is None:
            return np.zeros((0, NCH), np.float32)
        fresh = self.tail[self.emitted - self.base:]
        self.emitted = self.base + len(self.tail)
        return fresh.astype(np.float32)


# ================================================================================================================ engine (libnebula)
class Engine:
    def __init__(self):
        sys.path.insert(0, TOOLS)
        import gen_dataset as g
        self.g = g
        try:
            self.lib = g.Lib(None, ROOT, build=True)
        except SystemExit as e:
            raise RuntimeError(str(e))

    def new_body(self):
        return self.g.new_standing(self.lib)


class Replay:
    """Frame-by-frame tracking through the engine; keeps one body across all chunks of a generation."""

    def __init__(self, eng):
        self.eng, self.g = eng, eng.g
        self.b = eng.new_body()
        self.nb = len(self.b.names)
        self.rep = np.zeros(12, np.float32)

    def run(self, acts, dt):
        n = len(acts)
        bones = np.zeros((n, self.nb, 6), np.float32)
        com = np.zeros((n, 3), np.float32)
        margin = np.zeros(n, np.float32)
        bad = np.zeros(n, np.int8)
        L = self.eng.lib.L
        for i in range(n):
            a = np.ascontiguousarray(acts[i], np.float32)
            L.nebula_body_track(self.b.h, self.g.fptr(a), dt, self.g.fptr(self.rep))
            bones[i] = self.b.bones().reshape(self.nb, 10)[:, :6]
            com[i] = self.b.state()[4:7]
            margin[i], bad[i] = self.rep[10], 1 if self.rep[11] > 0.5 else 0
        return bones, com, margin, bad

    def close(self):
        try:
            self.eng.lib.L.nebula_body_track_end(self.b.h)
            self.b.close()
        except Exception:
            pass


def mm16(x):
    return base64.b64encode(np.clip(np.rint(np.asarray(x) * 1000), -32000, 32000).astype(np.int16).tobytes()).decode()


# ================================================================================================================ global state
class State:
    def __init__(self):
        self.neon_dir = None
        self.vae = None        # dict: cfg, names, dt, gguf, decoder, file
        self.prior = None      # Prior
        self.prior_file = None
        self.prior_gguf = None  # exported nanity-cvqprior GGUF (None if export failed)
        self.session = None     # NeonInteractive / NeonCli
        self.native_prior = False
        self.native_goal = False    # the NEON session takes goal numbers in cvq_prior_next (see start_session)
        self.engine = None
        self.engine_err = None
        self.gen_id = 0
        self.tc = None         # train_cvqvae module (canonicalize / to_world / YAWS)
        self.pending = []      # dropped files that could not be read yet (no torch); reloaded after the restart into the chosen environment
        self.env_job = torch_env.Job()
        self.env_scan = []     # last scan of environments: only these interpreters may be used by /api/env/start
        self.env_new_venv = None
        self.lm = None         # lang.NeonLM / lang.HttpLM, or None = keyword fallback only
        self.lm_note = ""      # why the last attempt to load a language model failed (shown in the page)
        self.gt = None         # goal_plan.GoalPlanner (stage C, or a trained translator GGUF); built on first use
        self.gt_file = None    # --goal-translator FILE
        self.gt_note = ""


S = State()


class NeedEnv(RuntimeError):
    """The Python running this server lacks packages the dropped file needs. The page then offers environments to use or create (torch_env.py)."""

    def __init__(self, missing, fname=None):
        self.missing, self.fname = list(missing), fname
        super().__init__("this Python (" + sys.executable + ") has no " + " / ".join(self.missing) + ", which this file needs")


def missing_modules(*mods):
    return [m for m in mods if importlib.util.find_spec(m) is None]


def torch_mod():
    try:
        import torch
        return torch
    except ImportError:
        raise NeedEnv(["torch"])


def train_cvqvae_mod():
    if S.tc is None:
        sys.path.insert(0, TOOLS)
        torch_mod()
        import train_cvqvae
        S.tc = train_cvqvae
    return S.tc


def load_pt(path, trust=False):
    torch = torch_mod()
    if trust:                                            # the user confirmed this file in the UI
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if ARGS.trust_pickle:
            return torch.load(path, map_location="cpu", weights_only=False)
        raise RuntimeError(f"torch.load(weights_only=True) refused {os.path.basename(path)} ({type(e).__name__}). "
                           f"It needs full unpickling, so only trust it if you made it yourself.")


def get_engine():
    if S.engine is None and S.engine_err is None and not ARGS.no_engine:
        try:
            S.engine = Engine()
            log("engine: libnebula loaded")
        except Exception as e:
            S.engine_err = str(e)
            log("engine unavailable (browser will draw a preview skeleton): " + str(e)[:300])
    return S.engine


def gguf_meta(path):
    import gguf
    r = gguf.GGUFReader(path)

    def val(key):
        f = r.fields[key]
        return f.parts[f.data[0]]
    cfg = {k: int(val(f"nanity.cvqvae.{k}")[0]) for k in ("window", "hidden", "emb", "code_dim", "num_codes", "downs", "n_res", "n_actions")}
    names = json.loads(bytes(val("nanity.cvqvae.action_names")).decode()) if "nanity.cvqvae.action_names" in r.fields else []
    dt = float(val("nanity.cvqvae.dt")[0]) if "nanity.cvqvae.dt" in r.fields else 1 / 30
    return cfg, names, dt


def rebuild():
    """(Re)start the NEON session for whatever is loaded now. Both GGUFs live in one process, so any change restarts it."""
    if S.session:
        S.session.close()
    S.session, S.native_prior, S.native_goal = None, False, False
    if not S.vae:
        return
    if S.neon_dir is None:
        S.neon_dir = find_neon_dir()
    ggufs = [S.vae["gguf"]] + ([S.prior_gguf] if S.prior_gguf else [])
    S.session, S.native_prior, S.native_goal = start_session(ggufs, bool(S.prior_gguf), S.neon_dir)


def set_vae(cfg, names, dt, gguf_path, fname):
    S.vae = dict(cfg=cfg, names=list(names), dt=float(dt), gguf=gguf_path, file=fname)
    rebuild()


def load_exporter(name):
    if S.neon_dir is None:
        S.neon_dir = find_neon_dir()
    if not S.neon_dir:
        raise RuntimeError(f"NEON checkout not found (needed for conversion/{name}.py). Clone NANITY-project/NANITY-NEON next to this repo or set NEON_DIR.")
    path = os.path.join(S.neon_dir, "conversion", name + ".py")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is missing: update your NANITY-NEON checkout (git pull).")
    spec = importlib.util.spec_from_file_location(name, path)
    ex = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ex)
    except SystemExit as e:
        raise RuntimeError(str(e))
    return ex


def classify_gguf(path):
    """'vae' (nanity.cvqvae.* keys), 'prior' (a cvqprior export) or 'lm' (anything else: a language model). Needs the gguf package."""
    import gguf
    r = gguf.GGUFReader(path)
    keys = list(r.fields)
    if any(k.startswith("nanity.cvqvae.") for k in keys):
        return "vae"
    if any(k.startswith("nanity.cvqprior.") for k in keys) or any("cvqprior" in k for k in keys):
        return "prior"
    return "lm"


def find_lm_bin(neon_dir):
    """Any NEON binary can run a language model (the motion-model flag is not needed), so this is looser than find_neon_bin."""
    cands = [ARGS.neon_bin, os.environ.get("NEON_BIN"), shutil.which("neon")]
    if neon_dir:
        cands += [os.path.join(neon_dir, p) for p in ("neon", "build/neon", "../neon")]
    cands += [os.path.join(os.path.dirname(ROOT), "neon", "neon"), os.path.join(os.path.dirname(ROOT), "neon"), os.path.expanduser("~/.local/bin/neon")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def set_lm(spec):
    """Install (or clear) the language model. spec: {kind: 'none'|'neon'|'http', path / base / model / key / template}. Raises with a readable reason; the old model is kept on failure."""
    kind = spec.get("kind", "none")
    new = None
    if kind == "neon":
        path = spec.get("path", "")
        if not os.path.isfile(path):
            raise RuntimeError("no such model file")
        if S.neon_dir is None:
            S.neon_dir = find_neon_dir()
        nb = find_lm_bin(S.neon_dir)
        if not nb:
            raise RuntimeError("no NEON binary found: build it (g++ -std=c++20 -O2 -pthread -mavx2 -mfma NEON-3.cpp -o neon) and set NEON_BIN or --neon-bin")
        new = lang.NeonLM(nb, path, spec.get("template", "auto"), ARGS.threads)
        try:                                              # a real round trip, so a model that loads but cannot answer is reported now, not on the first prompt
            new.complete("Reply with the single word: ok", [("user", "ok?")], max_tokens=4, temperature=0.0, timeout=120)
        except Exception:
            new.close()
            raise
    elif kind == "http":
        base = spec.get("base", "").strip()
        if not re.match(r"^https?://", base):
            raise RuntimeError("the endpoint must start with http:// or https://, e.g. http://127.0.0.1:8080/v1")
        new = lang.HttpLM(base, spec.get("model", ""), spec.get("key", ""))
        new.complete("Reply with the single word: ok", [("user", "ok?")], max_tokens=4, temperature=0.0, timeout=60)
    elif kind != "none":
        raise RuntimeError(f"unknown language model kind {kind!r}")
    old, S.lm = S.lm, new
    if old:
        old.close()
    S.lm_note = ""
    log("language model: " + (new.name if new else "none (keyword fallback)"))


def handle_upload(path, fname, trust=False):
    """Classify and load one dropped file. Returns a dict for the UI."""
    with open(path, "rb") as f:
        magic = f.read(4)
    need = missing_modules("gguf") if magic == b"GGUF" else missing_modules("torch", "gguf")
    if need:                                              # remember the file: it is loaded again after the restart into the chosen environment
        if fname not in S.pending:
            S.pending.append(fname)
        raise NeedEnv(need, fname)
    if fname in S.pending:
        S.pending.remove(fname)
    if magic == b"GGUF":
        what = classify_gguf(path)
        if what == "lm":                                  # not one of ours: a language model for the prompt box
            try:
                set_lm(dict(kind="neon", path=path))
            except Exception as e:
                S.lm_note = f"{fname}: {e}"
                info = lang.gguf_info(path)
                if info.get("arch") and info["arch"] != "nanity":
                    raise RuntimeError(lang.convert_hint(path, info))
                raise
            return dict(kind="lm", file=fname, note=S.lm.name)
        if what == "prior":
            raise RuntimeError(f"{fname} is an exported prior. Drop the prior_best.pt it came from instead: the page exports it itself.")
        cfg, names, dt = gguf_meta(path)
        set_vae(cfg, names, dt, path, fname)
        return dict(kind="vae", file=fname, note="GGUF used as is")
    ck = load_pt(path, trust)
    if not isinstance(ck, dict) or "model" not in ck or "cfg" not in ck:
        raise RuntimeError(f"{fname} is not a train_cvqvae.py checkpoint (expected a dict with 'model' and 'cfg').")
    if "norm" in ck:                                     # ---- VAE
        ex = load_exporter("nanity_export_cvqvae")
        os.makedirs(CACHE, exist_ok=True)
        out = os.path.join(CACHE, os.path.splitext(os.path.basename(fname))[0] + ".cvqvae.gguf")
        log(f"exporting {fname} -> {os.path.basename(out)} with NEON's exporter")
        try:
            ex.export_ckpt(ck, out, "f32")
        except SystemExit as e:
            raise RuntimeError(f"NEON exporter refused the checkpoint: {e}")
        names = list(ck.get("action_names", []))
        set_vae(dict(ck["cfg"]), names, float(ck.get("dt", 1 / 30)), out, fname)
        return dict(kind="vae", file=fname, note="exported to GGUF")
    if "K" in ck["cfg"] and "windows" in ck["cfg"]:       # ---- prior
        sd = {k: v.detach().cpu().numpy() for k, v in ck["model"].items()}
        S.prior = Prior(sd, ck["cfg"], ck.get("names", []), ck.get("stride", 0), ck.get("goal_stats"), ck.get("goal_names"), ck.get("style_names"))
        S.prior_file = fname
        log(f"prior loaded: {S.prior.params / 1e6:.2f} M parameters, {S.prior.K} codes, {S.prior.windows} windows of context"
            + (f", goal numbers ({', '.join(S.prior.goal_names)})" if S.prior.goal_dim else ", no goal numbers"))
        note = ""
        try:                                              # the same checkpoint, as a GGUF for NEON's own prior runtime
            ex = load_exporter("nanity_export_cvqprior")
            out = os.path.join(CACHE, os.path.splitext(os.path.basename(fname))[0] + ".cvqprior.gguf")
            log(f"exporting {fname} -> {os.path.basename(out)} with NEON's prior exporter")
            try:
                ex.export_ckpt(ck, out, "f32")
            except SystemExit as e:
                raise RuntimeError(str(e))
            S.prior_gguf = out
            note = "exported to GGUF"
        except Exception as e:
            S.prior_gguf = None
            note = "numpy prior only (NEON export failed: " + str(e)[:120] + ")"
            log(note)
        rebuild()
        return dict(kind="prior", file=fname, note=note)
    raise RuntimeError(f"{fname}: could not tell whether this is a VAE (needs 'norm') or a prior (cfg needs 'K' and 'windows').")


def status():
    v, p = S.vae, S.prior
    warn = []
    names = (v or {}).get("names") or (p.names if p else [])
    if v and p:
        n_lat = v["cfg"]["window"] >> v["cfg"]["downs"]
        if p.K != v["cfg"]["num_codes"]:
            warn.append(f"prior has {p.K} codes, VAE has {v['cfg']['num_codes']}: they were not trained together")
        if p.n != n_lat:
            warn.append(f"prior makes {p.n} codes per window, VAE needs {n_lat}")
        if p.n_actions != v["cfg"]["n_actions"]:
            warn.append(f"prior knows {p.n_actions} actions, VAE knows {v['cfg']['n_actions']}")
    return dict(
        vae=None if not v else dict(file=v["file"], window=v["cfg"]["window"], codes=v["cfg"]["num_codes"], downs=v["cfg"]["downs"], dt=v["dt"], backend=S.session.name if S.session else "—"),
        prior=None if not p else dict(backend="NEON" if S.native_prior else "numpy", file=S.prior_file, params=p.params, K=p.K, n=p.n, windows=p.windows, stride=p.stride, layers=len(p.L), d=p.d,
                                      goal=goal_status(p)),
        torch=not missing_modules("torch"), python=sys.executable, boot=BOOT, pending=list(S.pending),
        lm=dict(name=S.lm.name, kind=S.lm.kind) if S.lm else None, lm_note=S.lm_note,
        goal_translator=dict(source="model" if S.gt_file else "stage_c", file=os.path.basename(S.gt_file) if S.gt_file else None, note=S.gt_note),
        actions=names, neon_dir=S.neon_dir, engine=bool(S.engine), engine_error=S.engine_err, warnings=warn, ready=bool(v and p and not warn), log=LOG[-12:])


def goal_status(p):
    """What the page needs to steer by numbers; None when the prior was trained without goal numbers."""
    if not getattr(p, "goal_dim", 0):
        return None
    st = p.goal_stats or {}
    return dict(names=p.goal_names, turn_max=st.get("turn_max"), speed_cruise=st.get("speed_cruise"), speed_max=st.get("speed_max"),
                backend="NEON" if (S.native_prior and S.native_goal) else "numpy")


# ================================================================================================================ generation (a Python generator of UI events)
MAX_TIMELINE = 500           # windows
MAX_TURN_DEG = 180.0         # degrees of heading change in ONE window (the data turns up to ~40: the page marks what is beyond the prior's own turn_max)
MAX_SPEED = 6.0              # m/s


def parse_timeline(q, tc, p, W, dt):
    """The per-window goals of the request, numbers only (no label, no language): q['timeline'] = a JSON list with one row per window,
         {turn: degrees the heading changes over that window (+ left, - right), speed: m/s  OR  distance: metres of path walked in it}
       (speed wins when both are there; neither = the data's cruise speed; a window is (W - 1) * dt long, so distance = speed * that). Row i is for window i of the run, the last row repeats.
       -> None without a timeline, else [dict(turn, speed, distance, goal)] where goal = the RAW [dyaw, dx, dz, speed] the prior takes, built with train_cvqvae.arc_goal like tools/closed_loop.py does."""
    raw = q.get("timeline")
    if not raw:
        return None
    if not getattr(p, "goal_dim", 0):
        raise RuntimeError("this prior was trained without goal numbers, so a timeline cannot steer it (train again without --no-prior-goal)")
    try:
        rows = json.loads(raw)
        assert isinstance(rows, list) and 0 < len(rows) <= MAX_TIMELINE and all(isinstance(r, dict) for r in rows)
    except Exception:
        raise RuntimeError(f"the timeline must be a JSON list of 1 to {MAX_TIMELINE} windows, each {{turn, speed or distance}}")
    sec = (W - 1) * dt
    cruise = float((p.goal_stats or {}).get("speed_cruise") or 1.0)
    out = []
    for i, r in enumerate(rows):
        if r.get("unspecified"):                                   # the text says nothing about this window: no goal for it (the null goal), no asked-vs-got
            out.append(dict(turn=None, speed=None, distance=None, goal=None))
            continue
        try:
            turn = float(r.get("turn") or 0.0)
            speed = float(r["speed"]) if r.get("speed") is not None else (float(r["distance"]) / sec if r.get("distance") is not None else cruise)
        except (TypeError, ValueError):
            raise RuntimeError(f"timeline window {i + 1}: turn, speed and distance must be numbers")
        if not (math.isfinite(turn) and math.isfinite(speed)) or abs(turn) > MAX_TURN_DEG or not 0.0 <= speed <= MAX_SPEED:
            raise RuntimeError(f"timeline window {i + 1}: turn must be within +-{MAX_TURN_DEG:g} degrees and speed within 0 to {MAX_SPEED:g} m/s (distance 0 to {MAX_SPEED * sec:.1f} m) per window")
        path = speed * sec
        out.append(dict(turn=turn, speed=speed, distance=path, goal=[float(x) for x in tc.arc_goal(math.radians(turn), path, sec)]))
    return out


def parse_segments(q, names, stride, dt):
    """The request's segments [(action name or 'null', windows)]. With q['schedule'] (a JSON list of {action, seconds}) one segment per step; otherwise the single action / window count of the controls."""
    raw = q.get("schedule")
    if not raw:
        return [(q.get("action", "null"), int(q.get("windows", 6)))], False
    try:
        steps = json.loads(raw)
        assert isinstance(steps, list) and 0 < len(steps) <= lang.MAX_STEPS
    except Exception:
        raise RuntimeError(f"the schedule must be a JSON list of 1 to {lang.MAX_STEPS} steps")
    segs = []
    for st in steps:
        act = str(st.get("action", ""))
        if act not in names:
            raise RuntimeError(f"'{act}' is not an action of this VAE (known: {', '.join(names)})")
        sec = min(lang.MAX_SECONDS, max(lang.MIN_SECONDS, float(st.get("seconds", lang.DEFAULT_SECONDS))))
        segs.append((act, lang.to_windows(sec, stride, dt)))
    return segs, True


def generate(q, stop):
    v, p = S.vae, S.prior
    if not (v and p and S.session):
        raise RuntimeError("drop both a VAE (best.pt) and a prior (prior_best.pt) first")
    tc = train_cvqvae_mod()
    cfg = v["cfg"]
    W, n, dt = cfg["window"], p.n, v["dt"]
    stride = p.stride or W // 2
    names = v["names"] or p.names
    temp, top_k, top_p = float(q.get("temperature", 1.0)), int(q.get("top_k", 50)), float(q.get("top_p", 0.95))
    seed = int(q.get("seed", 0))
    segs, scheduled = parse_segments(q, names, stride, dt)
    tl = parse_timeline(q, tc, p, W, dt)                           # per-window goal numbers (turn / speed / distance), or None. Without an action the label is NULL: numbers only
    if tl and not scheduled and "windows" not in q:
        segs = [(segs[0][0], len(tl))]                             # a timeline with no window count runs as many windows as it has rows
    try:
        cfg_scale = float(q.get("cfg", 1.0))
    except ValueError:
        raise RuntimeError("cfg must be a number")
    if not (math.isfinite(cfg_scale) and 0.0 <= cfg_scale <= 20.0):
        raise RuntimeError("cfg must be in [0, 20] (1 = off)")
    if not tl:
        cfg_scale = 1.0                                            # guidance only acts on windows that have a goal
    has_goal = bool(getattr(p, "goal_dim", 0))
    endless = (not scheduled) and segs[0][1] == 0                  # 0 windows = endless, only for the plain single-action mode
    eng = get_engine()
    rp = Replay(eng) if eng else None
    st = Stitcher(tc, W, stride)
    native = S.native_prior
    if tl and native and not S.native_goal:                        # an older NEON build would take the goals and ignore them: the numpy prior does the same job
        log("this NEON build does not report goal numbers for its prior (it would ignore them); the numpy prior runs this one")
        native = False
    sec = (W - 1) * dt                                             # seconds one window lasts: what a goal's speed / distance refer to
    first_act = segs[0][0] if segs[0][0] in names else "null"
    yield dict(type="start", dt=dt, window=W, n=n, stride=stride, action=first_act, bone_names=rp.b.names if rp else None,
               engine=bool(rp), backend=S.session.name, prior_backend="NEON" if native else "numpy", seed=seed,
               windows=0 if endless else sum(k for _, k in segs), schedule=[dict(action=a, windows=k) for a, k in segs] if scheduled else None,
               goals=len(tl) if tl else 0, cfg=cfg_scale, window_seconds=round(sec, 4))
    w, sent, t_start = 0, 0, time.time()                           # w = window index over the whole run (the code grid), sent = frames emitted
    try:
        for si, (act_name, n_windows) in enumerate(segs):
            a = names.index(act_name) if act_name in names else None
            first = p.ACT0 + a if a is not None else p.NULL
            act_arg = act_name if a is not None else "null"
            rng = np.random.default_rng(seed + si)
            # A new action means a new prior context: the action is the first token of it, and the prior was trained on one action per clip. The prior does not know where the
            # body is, so the stitcher ends the old clip and starts the new one at the body's last pose (begin_segment below), instead of cross-fading two unrelated clips.
            if native:
                rq = {"op": "cvq_prior_reset", "action": act_arg, "seed": seed + si}
                if cfg_scale != 1.0:
                    rq["cfg_scale"] = cfg_scale
                with LOCK:
                    S.session.ask(rq)                              # NEON keeps the KV cache / context policy itself
            else:
                p.reset()
                if has_goal:
                    p.set_cfg_scale(cfg_scale)
            ctx, wl, gctx = [first], 0, []                         # wl = windows of THIS segment (context policy); gctx = the goal of every window the context holds (numpy prior)
            if si > 0:                                             # a new clip: finish the old one, start the new one at the body's last pose (see Stitcher.begin_segment)
                rest = st.begin_segment()
                if len(rest):
                    yield emit_frames(rest, sent, dt, rp)
                    sent += len(rest)
            if scheduled:
                yield dict(type="segment", i=si, count=len(segs), action=act_arg, windows=n_windows, w0=w)
            while (endless or wl < n_windows) and not stop():
                row = tl[min(w, len(tl) - 1)] if tl else None      # the timeline row of this window (w counts over the whole run; the last row repeats)
                gw = row["goal"] if row else None
                if native:
                    t0 = time.time()
                    rq = {"op": "cvq_prior_next", "temperature": temp, "top_k": top_k, "top_p": top_p, "decode": True, "normalized": False}
                    if gw is not None:
                        rq["goal"] = gw                            # RAW numbers: NEON normalises them with the prior's own statistics
                    elif row is not None:
                        rq["goal"] = None                          # an unspecified window: explicitly NO goal (null), never the previous one
                    with LOCK:
                        r = S.session.ask(rq)
                    codes = [int(c) for c in r["codes"]]
                    win = np.asarray(r["frames"], np.float32)
                    t1 = t2 = time.time()
                    for i in range(n):                         # NEON returns a whole window at once: replay the codes into the grid
                        yield dict(type="code", w=w, i=i, c=codes[i], ent=round(float(r["entropy"][i]), 3), p=round(float(r["probs"][i]), 4))
                else:
                    if wl >= p.windows:                        # same context policy as train_cvqvae.generate_codes: drop the oldest window, keep the action prefix
                        ctx = [ctx[0]] + ctx[1 + n + 1:]
                        p.reset()
                        if has_goal:                               # the windows move down one slot: declare the goals of the ones that stay again
                            del gctx[0]
                            for g in gctx:
                                p.add_window_goal(g)
                    if has_goal:                                   # this window's goal, before its SEP and codes are fed (None = no goal: the null vector)
                        gctx.append(gw)
                        p.add_window_goal(gw)
                    t0, codes = time.time(), []
                    for i in range(n):
                        if stop():
                            return
                        tok, ent, pt = p.sample(ctx, rng, temp, top_k, top_p)
                        ctx.append(tok)
                        codes.append(tok)
                        yield dict(type="code", w=w, i=i, c=tok, ent=round(ent, 3), p=round(pt, 4))
                    ctx.append(p.SEP)
                    t1 = time.time()
                    with LOCK:
                        win = S.session.decode(codes, act_name if a is not None else None)
                    t2 = time.time()
                win[:, FLAGS] = (win[:, FLAGS] > 0.5).astype(np.float32)
                got = tc.goal_features(tc.canonicalize(win[None]), np.array([len(win)]), dt)[0] if (row and row["goal"] is not None) else None     # what the window really does, measured on the decoded motion itself
                fresh = st.add(win)
                ev = emit_frames(fresh, sent, dt, rp)
                sent += len(fresh)
                t3 = time.time()
                yield dict(type="window", w=w, fused=native, prior_ms=round((t1 - t0) * 1000), decode_ms=round((t2 - t1) * 1000), engine_ms=round((t3 - t2) * 1000), frames=len(fresh))
                if got is not None:                                # asked vs got, per window: turn (degrees, + left), speed (m/s) and the distance walked in the window (m)
                    yield dict(type="goal", w=w, want=dict(turn=round(row["turn"], 1), speed=round(row["speed"], 3), distance=round(row["distance"], 3)),
                               got=dict(turn=round(math.degrees(float(got[0])), 1), speed=round(float(got[3]), 3), distance=round(float(got[3]) * sec, 3)))
                yield ev
                w += 1
                wl += 1
            if stop():
                return
        rest = st.flush()
        if len(rest):
            yield emit_frames(rest, sent, dt, rp)
            sent += len(rest)
        yield dict(type="done", frames=sent, seconds=round(time.time() - t_start, 2))
    finally:
        if rp:
            rp.close()


def emit_frames(acts, start, dt, rp):
    ev = dict(type="frames", start=start, n=len(acts), act=np.round(acts, 4).ravel().tolist(), bones=None)
    if rp and len(acts):
        bones, com, margin, bad = rp.run(acts, dt)
        ev.update(bones=mm16(bones), com=np.round(com, 3).ravel().tolist(), margin=np.round(margin, 3).tolist(), bad=bad.tolist())
    return ev


# ================================================================================================================ HTTP
# ================================================================================================================ python environment (torch)
def env_info():
    """Scan this machine for Python environments. Slow (it imports torch in each one that has it), so the page asks once and shows a spinner."""
    S.env_scan = torch_env.scan(ROOT)
    S.env_new_venv = torch_env.default_new_venv(ROOT)
    need = missing_modules("torch", "gguf")
    envs = []
    for d in S.env_scan:
        cmds = torch_env.plan(d["python"], d["missing"])[0] if d["missing"] and d["can_install"] else []
        envs.append(dict(python=d["python"], label=d["label"], prefix=d["prefix"], version=d["version"], torch=d["torch"], numpy=d["numpy"], gguf=d["gguf"],
                         ready=d["ready"], can_install=d["can_install"], missing=d["missing"], why_not=d.get("why_not", ""),
                         current=os.path.abspath(d["python"]) == os.path.abspath(sys.executable), commands=[" ".join(c) for c in cmds]))
    new_cmds = torch_env.plan(None, list(torch_env.NEEDED), new_venv=S.env_new_venv)[0]
    return dict(python=sys.executable, missing=need, pending=list(S.pending), envs=envs, new_venv=S.env_new_venv,
                new_commands=[" ".join(c) for c in new_cmds], job=S.env_job.snapshot())


def env_restart(python):
    """Job finished: restart this server under `python`; the dropped files are loaded again as soon as it is up."""
    try:
        with open(os.path.join(CACHE, "pending.json"), "w") as f:
            json.dump(list(S.pending), f)
    except OSError as e:
        log(f"could not save the pending file list: {e}")
    log(f"restarting under {python} ...")
    if S.session:
        S.session.close()
    time.sleep(1.5)                                       # let the page read the final job state
    torch_env.relaunch(python, os.path.abspath(__file__), sys.argv[1:])


def env_start(body):
    """Only an interpreter from the last scan, or the server-chosen new-venv path, can be used: a web page cannot name an arbitrary program to run."""
    if body.get("new_venv"):
        if not S.env_new_venv:
            raise RuntimeError("scan first")
        S.env_job.start(new_venv=S.env_new_venv, base_python=sys.executable, on_done=env_restart)
        return
    py = body.get("python", "")
    if py not in {d["python"] for d in S.env_scan}:
        raise RuntimeError("that interpreter was not in the scan")
    S.env_job.start(python=py, on_done=env_restart)


def load_pending():
    """After a restart: read the files that were dropped before it (VAE first, prior last)."""
    p = os.path.join(CACHE, "pending.json")
    if not os.path.exists(p):
        return
    try:
        with open(p) as f:
            names = json.load(f)
        os.remove(p)
    except (OSError, ValueError):
        return
    names.sort(key=lambda n: "prior" in n.lower())
    for n in names:
        path = os.path.join(UPLOADS, os.path.basename(n))
        if not os.path.isfile(path):
            continue
        try:
            with LOCK:
                S.gen_id += 1
                info = handle_upload(path, os.path.basename(n))
            log(f"loaded {n} after the restart ({info['kind']})")
        except Exception as e:
            log(f"could not load {n} after the restart: {e} (drop it again)")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "NERVA-UI/1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/status":
            self._json(status())
        elif u.path == "/api/generate":
            self._sse(q)
        elif u.path == "/api/env":
            self._json(env_info())
        elif u.path == "/api/env/job":
            self._json(S.env_job.snapshot())
        else:
            self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/api/upload":
            fname = os.path.basename(q.get("name", "upload.bin")) or "upload.bin"
            os.makedirs(UPLOADS, exist_ok=True)
            path = os.path.join(UPLOADS, fname)
            left = int(self.headers.get("Content-Length", 0))
            with open(path, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            try:
                with LOCK:
                    S.gen_id += 1                         # stop any running generation: the models are changing
                    info = handle_upload(path, fname, q.get("trust") == "1")
                self._json(dict(ok=True, info=info, status=status()))
            except NeedEnv as e:
                log(f"{fname}: {e}")
                self._json(dict(ok=False, need_env=dict(missing=e.missing, file=fname), error=str(e), status=status()), 200)
            except Exception as e:
                log(f"upload {fname} failed: {e}")
                self._json(dict(ok=False, error=str(e), status=status()), 200)
        elif u.path == "/api/env/start":
            if self.headers.get("X-NERVA-UI") != "1":            # a custom header cannot be sent cross-origin without a preflight, which this server never answers
                self.send_error(403)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
                env_start(body)
                self._json(dict(ok=True))
            except Exception as e:
                self._json(dict(ok=False, error=str(e)))
        elif u.path in ("/api/plan", "/api/lm"):
            if self.headers.get("X-NERVA-UI") != "1":            # same guard as /api/env/start: /api/lm can start a process or make this server call a URL
                self.send_error(403)
                return
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1 << 16)) or b"{}")
                if u.path == "/api/plan":
                    v, pr = S.vae, S.prior
                    names = (v or {}).get("names") or (pr.names if pr else [])
                    with LOCK:
                        res = lang.plan(str(body.get("text", ""))[:2000], names, S.lm)
                    if res["source"] in ("keyword", "none") and S.lm is None:
                        res["note"] = "no language model loaded: only action names written in the text were used"
                    self._json(dict(ok=bool(res["schedule"]), **res))
                else:
                    with LOCK:
                        set_lm(body)
                    self._json(dict(ok=True, status=status()))
            except Exception as e:
                if u.path == "/api/lm":
                    S.lm_note = str(e)
                self._json(dict(ok=False, error=str(e), status=status()))
        elif u.path == "/api/goal_plan":                             # text -> per-window goal rows (Goal Translator); errors are shown to the user, never hidden behind a keyword fallback
            if self.headers.get("X-NERVA-UI") != "1":
                self.send_error(403)
                return
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1 << 16)) or b"{}")
                v, pr = S.vae, S.prior
                if not (v and pr and goal_status(pr)):
                    raise goal_plan.GoalPlanError("drop a VAE and a prior trained with goal numbers first")
                with LOCK:
                    if S.gt is None:
                        S.gt = goal_plan.GoalPlanner(S.gt_file)
                    W = v["cfg"]["window"]
                    res = S.gt.plan(str(body.get("text", "")), pr.goal_stats, W, pr.stride or W // 2, v["dt"])
                self._json(dict(ok=True, **res))
            except goal_plan.GoalPlanError as e:
                S.gt_note = str(e) if "translator" in str(e) else S.gt_note
                self._json(dict(ok=False, error=str(e)))
            except Exception as e:
                self._json(dict(ok=False, error=f"goal translator: {e}"))
        elif u.path == "/api/stop":
            S.gen_id += 1
            self._json(dict(ok=True))
        else:
            self.send_error(404)

    def _sse(self, q):
        S.gen_id += 1
        mine = S.gen_id
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()
        try:
            for ev in generate(q, lambda: S.gen_id != mine):
                send(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"generation error: {e}")
            try:
                send(dict(type="error", message=str(e)))
            except Exception:
                pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--neon-dir", help="NANITY-NEON checkout (default: next to this repo / NEON_DIR / ~/NANITY-NEON)")
    ap.add_argument("--neon-bin", help="a built NEON binary (default: NEON_BIN / PATH / inside the checkout)")
    ap.add_argument("--no-clone", action="store_true", help="never git clone the NEON repo automatically")
    ap.add_argument("--numpy-prior", action="store_true", help="force the numpy prior even if NEON can run it (for comparing the two)")
    ap.add_argument("--no-engine", action="store_true", help="skip libnebula; the browser draws a preview skeleton")
    ap.add_argument("--threads", type=int, default=0, help="threads for NEON (0 = its default)")
    ap.add_argument("--lm", default="", help="language model for the prompt box: a NEON-loadable .gguf path (NEON must be built), or an OpenAI-compatible base URL such as http://127.0.0.1:8080/v1")
    ap.add_argument("--lm-model", default="", help="model name sent to an --lm URL (many servers ignore it)")
    ap.add_argument("--lm-key-env", default="", help="name of an environment variable holding the API key for an --lm URL (never put the key itself on the command line)")
    ap.add_argument("--goal-translator", default="", help="a trained Goal Translator .gguf (tools/goal_translator/export_goal_translator.py) for the Steering box's text field; "
                    "without it the deterministic stage-C bootstrap translates (and the page says so)")
    ap.add_argument("--trust-pickle", action="store_true", help="torch.load(weights_only=False) as a fallback; only for files you made yourself")
    ARGS = ap.parse_args()
    if ARGS.goal_translator:
        if not os.path.isfile(ARGS.goal_translator):
            sys.exit(f"--goal-translator: {ARGS.goal_translator} is not a file")
        S.gt_file = os.path.abspath(ARGS.goal_translator)
    os.makedirs(CACHE, exist_ok=True)
    faulthandler.enable(all_threads=True)                 # a native crash (libnebula, torch ...) prints the Python stack that was running instead of dying silently
    threading.Thread(target=lambda: (get_engine(), None), daemon=True).start()
    threading.Thread(target=load_pending, daemon=True).start()
    if ARGS.lm:
        def _lm():
            try:
                spec = dict(kind="http", base=ARGS.lm, model=ARGS.lm_model, key=os.environ.get(ARGS.lm_key_env, "") if ARGS.lm_key_env else "") if ARGS.lm.startswith(("http://", "https://")) \
                    else dict(kind="neon", path=ARGS.lm)
                with LOCK:
                    set_lm(spec)
            except Exception as e:
                S.lm_note = str(e)
                log(f"language model not loaded: {e}")
        threading.Thread(target=_lm, daemon=True).start()     # build/load the engine while the page loads
    srv = http.server.ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    url = f"http://{ARGS.host}:{ARGS.port}"
    log(f"NERVA UI at {url}   (drop best.pt and prior_best.pt into the page)")
    if not ARGS.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if S.session:
            S.session.close()
        if S.lm:
            S.lm.close()


if __name__ == "__main__":
    main()
