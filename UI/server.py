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
(found next to this repo, or set NEON_DIR; cloned into UI/.cache if missing, unless --no-clone). A C++ compiler for libnebula / cvqvae_cli.
"""
import argparse, base64, ctypes, glob, http.server, importlib.util, json, math, os, shutil, subprocess, sys, threading, time, urllib.parse, webbrowser
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
CACHE = os.path.join(HERE, ".cache")
UPLOADS = os.path.join(CACHE, "uploads")
NCH = 30
FLAGS = [8, 13, 17, 21, 25, 29]
NEON_REPO_URL = "https://github.com/NANITY-project/NANITY-NEON.git"

ARGS = None
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
    """Returns (session, native_prior). Prefers one NEON process holding VAE (+ prior); falls back to cvqvae_cli + the numpy prior."""
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
                native = False
                if want_prior and not ARGS.numpy_prior:
                    try:
                        d.ask({"op": "cvq_prior_reset", "action": "null", "seed": 0})
                        native = True
                    except Exception as e:
                        log(f"this NEON has no usable prior op ({e}); the prior will run in numpy")
                log(f"NEON: {nb}  (VAE decode on NEON, prior on {'NEON' if native else 'numpy'})")
                return d, native
            except Exception as e:
                log(f"NEON --interactive failed ({e}); " + ("retrying without the prior GGUF" if want_prior else "trying cvqvae_cli"))
    cli = build_cli(neon_dir) if neon_dir else None
    if cli:
        d = NeonCli(cli, ggufs[0])
        d.decode([0] * 4, None)
        log(f"NEON: {d.name} (VAE decode on NEON, prior on numpy). Build a current NEON (see NANITY-NEON README) to run the prior on NEON too.")
        return d, False
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


class Prior:
    """Inference twin of train_cvqvae.CodePrior: token ids  codes 0..K-1 | SEP=K | actions K+1.. | NULL | PAD. KV-cached, one token at a time."""

    def __init__(self, sd, cfg, names, stride):
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
        self.params = sum(v.size for v in W.values())
        self.reset()

    def reset(self):
        self.t = 0
        self.kc = [np.zeros((self.H, self.maxT, self.dh), np.float32) for _ in self.L]
        self.vc = [np.zeros((self.H, self.maxT, self.dh), np.float32) for _ in self.L]
        self.last = None

    def feed(self, tok):
        t = self.t
        if t >= self.maxT:
            raise RuntimeError("prior context overflow (internal error)")
        x = self.tok[tok] + self.pos[t]
        for i, L in enumerate(self.L):
            h = _ln(x, L["n1w"], L["n1b"])
            qkv = (h @ L["qw"].T + L["qb"]).reshape(3, self.H, self.dh)
            self.kc[i][:, t], self.vc[i][:, t] = qkv[1], qkv[2]
            att = np.einsum("hd,htd->ht", qkv[0], self.kc[i][:, :t + 1]) / math.sqrt(self.dh)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = np.einsum("ht,htd->hd", att, self.vc[i][:, :t + 1]).reshape(self.d)
            x = x + y @ L["pw"].T + L["pb"]
            h = _ln(x, L["n2w"], L["n2b"]) @ L["f0w"].T + L["f0b"]
            h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
            x = x + h @ L["f2w"].T + L["f2b"]
        self.t += 1
        self.last = _ln(x, self.lnw, self.lnb) @ self.head.T

    def logits(self, ctx):
        while self.t < len(ctx):
            self.feed(ctx[self.t])
        return self.last[:self.K]

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

    def add(self, win):
        tc, ov = self.tc, self.ov
        w = tc.canonicalize(np.asarray(win, np.float32)[None])[0].astype(np.float64)
        if self.tail is None:
            out = w
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
        self.engine = None
        self.engine_err = None
        self.gen_id = 0
        self.tc = None         # train_cvqvae module (canonicalize / to_world / YAWS)


S = State()


def torch_mod():
    try:
        import torch
        return torch
    except ImportError:
        raise RuntimeError("PyTorch is needed to read .pt files: pip install torch (CPU build is enough)")


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
    S.session, S.native_prior = None, False
    if not S.vae:
        return
    if S.neon_dir is None:
        S.neon_dir = find_neon_dir()
    ggufs = [S.vae["gguf"]] + ([S.prior_gguf] if S.prior_gguf else [])
    S.session, S.native_prior = start_session(ggufs, bool(S.prior_gguf), S.neon_dir)


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


def handle_upload(path, fname, trust=False):
    """Classify and load one dropped file. Returns a dict for the UI."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
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
        S.prior = Prior(sd, ck["cfg"], ck.get("names", []), ck.get("stride", 0))
        S.prior_file = fname
        log(f"prior loaded: {S.prior.params / 1e6:.2f} M parameters, {S.prior.K} codes, {S.prior.windows} windows of context")
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
        prior=None if not p else dict(backend="NEON" if S.native_prior else "numpy", file=S.prior_file, params=p.params, K=p.K, n=p.n, windows=p.windows, stride=p.stride, layers=len(p.L), d=p.d),
        actions=names, neon_dir=S.neon_dir, engine=bool(S.engine), engine_error=S.engine_err, warnings=warn, ready=bool(v and p and not warn), log=LOG[-12:])


# ================================================================================================================ generation (a Python generator of UI events)
def generate(q, stop):
    v, p = S.vae, S.prior
    if not (v and p and S.session):
        raise RuntimeError("drop both a VAE (best.pt) and a prior (prior_best.pt) first")
    tc = train_cvqvae_mod()
    cfg = v["cfg"]
    W, n, dt = cfg["window"], p.n, v["dt"]
    stride = p.stride or W // 2
    names = v["names"] or p.names
    act_name = q.get("action", "null")
    a = names.index(act_name) if act_name in names else None
    first = p.ACT0 + a if a is not None else p.NULL
    temp, top_k, top_p = float(q.get("temperature", 1.0)), int(q.get("top_k", 50)), float(q.get("top_p", 0.95))
    seed = int(q.get("seed", 0))
    n_windows = int(q.get("windows", 6))                 # 0 = endless
    rng = np.random.default_rng(seed)
    eng = get_engine()
    rp = Replay(eng) if eng else None
    st = Stitcher(tc, W, stride)
    native = S.native_prior
    act_arg = act_name if a is not None else "null"
    if native:
        with LOCK:
            S.session.ask({"op": "cvq_prior_reset", "action": act_arg, "seed": seed})      # NEON keeps the KV cache / context policy itself
    else:
        p.reset()
    yield dict(type="start", dt=dt, window=W, n=n, stride=stride, action=act_arg, bone_names=rp.b.names if rp else None,
               engine=bool(rp), backend=S.session.name, prior_backend="NEON" if native else "numpy", seed=seed, windows=n_windows)
    ctx, w, sent, t_start = [first], 0, 0, time.time()
    try:
        while (n_windows == 0 or w < n_windows) and not stop():
            if native:
                t0 = time.time()
                with LOCK:
                    r = S.session.ask({"op": "cvq_prior_next", "temperature": temp, "top_k": top_k, "top_p": top_p, "decode": True, "normalized": False})
                codes = [int(c) for c in r["codes"]]
                win = np.asarray(r["frames"], np.float32)
                t1 = t2 = time.time()
                for i in range(n):                         # NEON returns a whole window at once: replay the codes into the grid
                    yield dict(type="code", w=w, i=i, c=codes[i], ent=round(float(r["entropy"][i]), 3), p=round(float(r["probs"][i]), 4))
            else:
                if w >= p.windows:                         # same context policy as train_cvqvae.generate_codes: drop the oldest window, keep the action prefix
                    ctx = [ctx[0]] + ctx[1 + n + 1:]
                    p.reset()
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
            fresh = st.add(win)
            ev = emit_frames(fresh, sent, dt, rp)
            sent += len(fresh)
            t3 = time.time()
            yield dict(type="window", w=w, fused=native, prior_ms=round((t1 - t0) * 1000), decode_ms=round((t2 - t1) * 1000), engine_ms=round((t3 - t2) * 1000), frames=len(fresh))
            yield ev
            w += 1
        if not stop():
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
            except Exception as e:
                log(f"upload {fname} failed: {e}")
                self._json(dict(ok=False, error=str(e), status=status()), 200)
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
    ap.add_argument("--trust-pickle", action="store_true", help="torch.load(weights_only=False) as a fallback; only for files you made yourself")
    ARGS = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    threading.Thread(target=lambda: (get_engine(), None), daemon=True).start()     # build/load the engine while the page loads
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


if __name__ == "__main__":
    main()
