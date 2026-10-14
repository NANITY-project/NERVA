"""NEON integration for Nebula Studio.

NEON (NANITY runtime) is a single-user CLI, not a server. Its interface, read from NEON-3.cpp:

  spawn:   neon --model <file.gguf> --interactive [--threads N]
  ready:   stderr line  "[Mode] Interactive stdin/stdout loop — ready."
  request: one JSON object per stdin line  {"messages":[...],"max_tokens":N,"temperature":T,"top_p":P}
  reply:   stdout lines {"token":"..."} ... then {"done":true}   (or {"error":"..."})
  check:   neon --model <file> --probe  ->  stdout PROBE_OK, or stderr "[CRITICAL ERROR] <reason>", exit 1

This module finds the binary and models, builds NEON from source if only the source exists, and runs one
managed process. One process = one request at a time, so complete() is serialized.
"""
import json
import os
import platform
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque

READY_MARK = "Interactive stdin/stdout loop"
SKIP_DIRS = {".git", "node_modules", "__pycache__", "site-packages", ".venv", "venv", "proc", "sys"}


class NeonError(RuntimeError):
    pass


# --------------------------------------------------------------------------- GGUF header
class GgufError(ValueError):
    pass


_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
           6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 12: "Q4_K", 13: "Q4_K", 15: "Q4_K", 17: "Q5_K", 18: "Q6_K", 32: "BF16"}


def read_gguf_meta(path, max_kv=100000):
    """Read the GGUF metadata header without loading tensors. Scalars and short strings are returned;
    arrays are summarised as '<key>.count'. Raises GgufError on anything malformed."""
    meta = {}
    try:
        f = open(path, "rb")
    except OSError as e:
        raise GgufError(f"cannot open: {e.strerror}") from None
    with f:
        size = os.fstat(f.fileno()).st_size
        def rd(fmt, n):
            b = f.read(n)
            if len(b) != n:
                raise GgufError("truncated file")
            return struct.unpack(fmt, b)[0]
        def string():
            n = rd("<Q", 8)
            if n > 1 << 24 or f.tell() + n > size:
                raise GgufError("implausible string length")
            return f.read(n).decode("utf-8", "replace")
        def skip_value(t):
            if t in _SCALAR:
                f.seek(_SCALAR[t][1], 1)
            elif t == 8:
                n = rd("<Q", 8)
                if f.tell() + n > size:
                    raise GgufError("implausible string length")
                f.seek(n, 1)
            else:
                raise GgufError(f"unsupported value type {t}")
        if f.read(4) != b"GGUF":
            raise GgufError("not a GGUF file (bad magic)")
        version = rd("<I", 4)
        if version not in (2, 3):
            raise GgufError(f"unsupported GGUF version {version}")
        meta["gguf.version"] = version
        rd("<Q", 8)                                    # tensor count
        kv = rd("<Q", 8)
        if kv > max_kv:
            raise GgufError("implausible metadata count")
        for _ in range(kv):
            key = string()
            t = rd("<I", 4)
            if t in _SCALAR:
                fmt, n = _SCALAR[t]
                meta[key] = rd(fmt, n)
            elif t == 8:
                n = rd("<Q", 8)
                if f.tell() + n > size:
                    raise GgufError("implausible string length")
                s = f.read(n)
                meta[key] = s.decode("utf-8", "replace")[:300]
            elif t == 9:
                et, cnt = rd("<I", 4), rd("<Q", 8)
                if et in _SCALAR:
                    f.seek(_SCALAR[et][1] * cnt, 1)
                elif et == 8:
                    for _i in range(cnt):
                        skip_value(8)
                else:
                    raise GgufError(f"unsupported array element type {et}")
                meta[key + ".count"] = cnt
            else:
                raise GgufError(f"unsupported value type {t}")
    return meta


# --------------------------------------------------------------------------- finding things
def _home(*p):
    return os.path.join(os.path.expanduser("~"), *p)


def candidate_dirs(extra=()):
    here = os.getcwd()
    base = [os.environ.get("NEON_DIR", ""), here, os.path.join(here, "neon"), os.path.dirname(here),
            _home("NEON"), _home("neon"), _home("NEON-Runtime-Beta-Variation"), _home("NANITY"), _home("nanity"),
            _home("projects", "NEON-Runtime-Beta-Variation"), _home("src", "NEON-Runtime-Beta-Variation"),
            _home("Downloads", "NEON-Runtime-Beta-Variation"), _home(".local", "bin"), "/usr/local/bin", "/opt/neon"]
    # siblings of the current dir that look like NEON checkouts
    parent = os.path.dirname(here)
    try:
        base += [os.path.join(parent, d) for d in os.listdir(parent) if "neon" in d.lower()]
    except OSError:
        pass
    seen, out = set(), []
    for d in list(extra) + base:
        d = os.path.abspath(os.path.expanduser(d)) if d else ""
        if d and d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def verify_binary(path, timeout=5):
    """True only if `path` is an executable that prints NEON's usage text (so a random 'neon' tool isn't run as one)."""
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False, "not an executable file"
    if "neon" not in os.path.basename(path).lower():
        return False, "file name should contain 'neon'"
    try:
        r = subprocess.run([path], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"could not run it: {e}"
    out = (r.stderr or "") + (r.stdout or "")
    if "--model" in out and "--interactive" in out:
        return True, "ok"
    return False, "does not look like the NEON runtime (unexpected usage output)"


def find_binary(configured=None, extra_dirs=()):
    cands = [configured, os.environ.get("NEON_BIN"), shutil.which("neon")]
    for d in candidate_dirs(extra_dirs):
        for rel in ("neon", os.path.join("build", "neon"), "neon_test"):
            cands.append(os.path.join(d, rel))
    tried = set()
    for c in cands:
        if c and c not in tried:
            tried.add(c)
            ok, _ = verify_binary(c)
            if ok:
                return os.path.abspath(c)
    return None


def find_source(extra_dirs=()):
    for d in candidate_dirs(extra_dirs):
        if os.path.isfile(os.path.join(d, "NEON-3.cpp")):
            return d
    return None


def find_compiler():
    for c in ("g++", "c++", "clang++"):
        p = shutil.which(c)
        if p:
            return p
    return None


def cpu_flags():
    try:
        txt = open("/proc/cpuinfo").read()
        return {x for x in re.findall(r"\bflags\s*:\s*(.*)", txt)[0].split()}
    except (OSError, IndexError):
        return set()


def default_model_dirs(extra=()):
    here = os.getcwd()
    env = [d for d in os.environ.get("NEBULA_MODEL_DIRS", "").split(os.pathsep) if d]
    cands = list(extra) + env + [os.path.join(here, "models"), _home("models"), _home("Models"), _home(".cache", "nanity"),
                           _home("Downloads"), _home("NEON"), _home("NEON-Runtime-Beta-Variation"), here]
    seen, out = set(), []
    for d in cands:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def scan_models(dirs, max_depth=3, max_files=300):
    found, seen = [], set()
    for root in dirs:
        base_depth = root.rstrip(os.sep).count(os.sep)
        for cur, subdirs, files in os.walk(root):
            if cur.count(os.sep) - base_depth >= max_depth:
                subdirs[:] = []
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS and (not d.startswith(".") or d == ".cache")]
            for fn in files:
                if not fn.lower().endswith((".gguf", ".nctr")):
                    continue
                p = os.path.abspath(os.path.join(cur, fn))
                if p in seen:
                    continue
                seen.add(p)
                found.append(describe_model(p))
                if len(found) >= max_files:
                    return found
    found.sort(key=lambda m: (not m["runnable"], m["name"].lower()))
    return found


def describe_model(path):
    info = {"path": path, "file": os.path.basename(path), "dir": os.path.dirname(path), "name": os.path.basename(path),
            "size": 0, "arch": None, "runnable": False, "note": "", "ctx_len": None, "quant": None}
    try:
        info["size"] = os.path.getsize(path)
    except OSError:
        pass
    if path.lower().endswith(".nctr"):
        info["note"] = "NEON can't run .nctr files yet (loader not wired in) — export GGUF instead"
        return info
    try:
        m = read_gguf_meta(path)
    except GgufError as e:
        info["note"] = f"unreadable: {e}"
        return info
    info["arch"] = m.get("general.architecture")
    info["name"] = m.get("general.name") or info["name"]
    info["ctx_len"] = m.get("nanity.context_length")
    info["quant"] = FILE_TYPES.get(m.get("general.file_type"))
    if info["arch"] == "nanity":
        if m.get("nanity.spec_version") not in (None, 1):
            info["note"] = f"spec version {m.get('nanity.spec_version')} (NEON expects 1)"
        else:
            info["runnable"] = True
    else:
        info["note"] = f"architecture '{info['arch']}' — convert it first: python3 nanity_convert.py --detect-only <file>"
    return info


def probe_model(binary, model, timeout=90):
    """Load + validate without generating. Returns {'ok': bool, 'message': str, 'config': {...}}."""
    try:
        r = subprocess.run([binary, "--model", model, "--probe"], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"probe timed out after {timeout}s", "config": {}}
    except OSError as e:
        return {"ok": False, "message": f"could not run NEON: {e}", "config": {}}
    cfg = parse_config(r.stderr)
    if "PROBE_OK" in (r.stdout or ""):
        return {"ok": True, "message": "model loads and passes NANITY spec validation", "config": cfg}
    m = re.search(r"\[CRITICAL ERROR\]\s*(.+)", r.stderr or "")
    return {"ok": False, "message": (m.group(1).strip() if m else (r.stderr or "probe failed").strip()[-300:]), "config": cfg}


def parse_config(stderr_text):
    m = re.search(r"\[Config\]\s*(.*)", stderr_text or "")
    return {k: int(v) if re.fullmatch(r"-?\d+", v) else v for k, v in re.findall(r"(\w+)=([\w.\-+]+)", m.group(1))} if m else {}


# --------------------------------------------------------------------------- building
class Builder:
    def __init__(self):
        self.state, self.log, self.output, self.error = "idle", deque(maxlen=300), None, ""
        self._t = None

    def start(self, source_dir, out_path=None):
        if self.state == "building":
            raise NeonError("a build is already running")
        cxx = find_compiler()
        if not cxx:
            raise NeonError("no C++ compiler found (install g++ or clang++)")
        src = os.path.join(source_dir, "NEON-3.cpp")
        if not os.path.isfile(src):
            raise NeonError("NEON-3.cpp not found in that folder")
        out = out_path or os.path.join(source_dir, "neon")
        flags = cpu_flags()
        cmd = [cxx, "-std=c++20", "-O2", "-pthread"] + (["-mavx2", "-mfma"] if {"avx2", "fma"} <= flags else []) + [src, "-o", out]
        self.state, self.output, self.error = "building", None, ""
        self.log.clear(); self.log.append("$ " + " ".join(cmd))
        def run():
            try:
                p = subprocess.Popen(cmd, cwd=source_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    self.log.append(line.rstrip()[:300])
                rc = p.wait()
                if rc == 0 and os.path.isfile(out):
                    self.state, self.output = "done", out
                    self.log.append("build finished")
                else:
                    self.state, self.error = "failed", f"compiler exited with code {rc}"
            except OSError as e:
                self.state, self.error = "failed", str(e)
        self._t = threading.Thread(target=run, daemon=True, name="neon-build")
        self._t.start()


# --------------------------------------------------------------------------- running
def est_tokens(text):
    return int(len(text) / 3.0) + 1                   # slightly conservative for a greedy-vocab tokenizer


def fit_messages(messages, ctx_len, reserve):
    """Drop the oldest history turns until the prompt (plus room for the reply) fits the context."""
    if not ctx_len:
        return messages
    budget = ctx_len - reserve - 16
    msgs = list(messages)
    cost = lambda ms: sum(est_tokens(m["content"]) + 6 for m in ms)
    while len(msgs) > 2 and cost(msgs) > budget:
        del msgs[1:3]                                   # oldest (user, assistant) pair after the system prompt
    if cost(msgs) > budget:
        raise NeonError(f"prompt (~{cost(msgs)} tokens) does not fit this model's context ({ctx_len}); use a shorter persona or fewer props")
    return msgs


class Process:
    def __init__(self, binary, model, threads=0):
        self.binary, self.model, self.threads = binary, model, threads
        self.state, self.error = "stopped", ""
        self.config, self.stderr = {}, deque(maxlen=60)
        self.requests, self.last_ms, self.last_chars, self.started = 0, 0, 0, 0.0
        self._p = None
        self._out = queue.Queue()
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def start(self):
        if self._p:
            raise NeonError("already running")
        cmd = [self.binary, "--model", self.model, "--interactive"] + (["--threads", str(self.threads)] if self.threads > 0 else [])
        try:
            self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self.state, self.error = "error", f"could not start NEON: {e}"
            raise NeonError(self.error) from None
        self.state, self.error, self.started = "starting", "", time.time()
        threading.Thread(target=self._read_stderr, daemon=True, name="neon-err").start()
        threading.Thread(target=self._read_stdout, daemon=True, name="neon-out").start()

    def _read_stderr(self):
        p = self._p
        try:
            for line in p.stderr:
                line = line.rstrip()
                self.stderr.append(line)
                if line.startswith("[Config]"):
                    self.config = parse_config(line)
                if READY_MARK in line and "ready" in line:
                    self.state = "ready"
                    self._ready.set()
        finally:
            try:
                p.stderr.close()
            except OSError:
                pass
        rc = p.wait()
        if self._p is p and self.state != "stopped":
            crit = next((l for l in reversed(self.stderr) if "[CRITICAL ERROR]" in l or "[Error]" in l), "")
            self.state = "error"
            self.error = (crit.split("]", 1)[-1].strip() if crit else f"NEON exited with code {rc}")
            self._ready.set()
            self._out.put(None)                         # wake any waiting request

    def _read_stdout(self):
        p = self._p
        try:
            for line in p.stdout:
                self._out.put(line)
        finally:
            try:
                p.stdout.close()
            except OSError:
                pass

    def wait_ready(self, timeout=120):
        self._ready.wait(timeout)
        return self.state == "ready"

    def complete(self, messages, max_tokens=160, temperature=0.6, top_p=0.9, timeout=180, queue_timeout=600):
        if not self._lock.acquire(timeout=queue_timeout):
            raise NeonError("NEON is busy (request queue timed out)")
        try:
            if self.state == "starting":
                self.wait_ready(120)
            if self.state != "ready" or self._p is None:
                raise NeonError(f"NEON is not running ({self.state}{': ' + self.error if self.error else ''})")
            msgs = fit_messages(messages, self.config.get("ctx_len"), max_tokens)
            while not self._out.empty():                 # stale lines from a previous failure
                self._out.get_nowait()
            t0 = time.time()
            try:
                self._p.stdin.write(json.dumps({"messages": msgs, "max_tokens": int(max_tokens), "temperature": float(temperature), "top_p": float(top_p)}) + "\n")
                self._p.stdin.flush()
            except (BrokenPipeError, OSError):
                raise NeonError("NEON closed its input (it probably crashed)") from None
            parts = []
            while True:
                try:
                    line = self._out.get(timeout=max(0.1, timeout - (time.time() - t0)))
                except queue.Empty:
                    self.kill("a reply timed out — restart the model")
                    raise NeonError(f"NEON did not answer within {timeout}s") from None
                if line is None:
                    raise NeonError(f"NEON stopped: {self.error or 'process exited'}")
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue                             # stray non-protocol output
                if "token" in msg:
                    parts.append(msg["token"])
                elif msg.get("done"):
                    break
                elif "error" in msg:
                    raise NeonError(f"NEON error: {msg['error']}")
            text = "".join(parts)
            self.requests += 1
            self.last_ms, self.last_chars = int((time.time() - t0) * 1000), len(text)
            return text
        finally:
            self._lock.release()

    def kill(self, reason):
        self.error = reason
        p, self._p = self._p, None
        self.state = "error"
        if p:
            try:
                p.kill()
            except OSError:
                pass
        self._ready.set()
        self._out.put(None)

    def stop(self):
        p, self._p = self._p, None
        self.state = "stopped"
        if p:
            try:
                p.stdin.close()                          # EOF ends NEON's getline loop cleanly
                p.wait(timeout=4)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
        self._ready.set()
        self._out.put(None)


class NeonLLM:
    """Adapter with the same .complete() as the other model clients (grammar is not supported by NEON)."""
    name = "neon"

    def __init__(self, process, max_tokens=160, temperature=0.6):
        self.proc, self.max_tokens, self.temperature = process, max_tokens, temperature

    def complete(self, messages, grammar=None):
        from agent import LLMError
        try:
            return self.proc.complete(messages, self.max_tokens, self.temperature)
        except NeonError as e:
            raise LLMError(str(e)) from None


# --------------------------------------------------------------------------- manager
class Manager:
    def __init__(self, settings_path=None):
        self.settings_path = settings_path or os.environ.get("NEBULA_STUDIO_CONFIG") or _home(".config", "nebula-studio", "settings.json")
        self.settings = {"binary": "", "model_dirs": [], "threads": 0, "last_model": ""}
        try:
            with open(self.settings_path) as f:
                self.settings.update({k: v for k, v in json.load(f).items() if k in self.settings})
        except (OSError, ValueError):
            pass
        self.builder = Builder()
        self.proc = None
        self.models = []
        self.scanned = False
        self._bin_cache = (0.0, None, None)              # (time, key, path): verify_binary executes the file, so don't do it per poll
        self.on_change = None                            # callable(), set by Studio

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, "w") as f:
                json.dump(self.settings, f, indent=1)
        except OSError:
            pass

    def binary(self, force=False):
        key = (self.settings.get("binary"), tuple(self.settings["model_dirs"]), self.builder.output, os.environ.get("NEON_BIN"))
        t, k, path = self._bin_cache
        if not force and k == key and time.time() - t < 15 and (path is None or os.path.isfile(path)):
            return path
        path = find_binary(self.settings.get("binary"), self.settings["model_dirs"])
        if not path and self.builder.output:
            path = self.builder.output if verify_binary(self.builder.output)[0] else None
        self._bin_cache = (time.time(), key, path)
        return path

    def rescan(self):
        self.models = scan_models(default_model_dirs(self.settings["model_dirs"]))
        self.scanned = True
        return self.models

    def add_dirs(self, dirs):
        for d in dirs:
            d = os.path.abspath(os.path.expanduser(str(d)))
            if not os.path.isdir(d):
                raise NeonError(f"not a folder: {d}")
            if d not in self.settings["model_dirs"]:
                self.settings["model_dirs"].append(d)
        self._save()
        return self.rescan()

    def set_binary(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        ok, msg = verify_binary(path)
        if not ok:
            raise NeonError(f"{path}: {msg}")
        self.settings["binary"] = path
        self._save()
        self.binary(force=True)
        return path

    def build(self):
        src = find_source(self.settings["model_dirs"])
        if not src:
            raise NeonError("NEON source (NEON-3.cpp) not found; clone https://github.com/NANITY-project/NEON-Runtime-Beta-Variation next to Studio or set NEON_DIR")
        self.builder.start(src)
        return src

    def probe(self, model):
        b = self.binary()
        if not b:
            raise NeonError("NEON binary not found")
        self._check_model_path(model)
        return probe_model(b, model)

    @staticmethod
    def _check_model_path(model):
        if not isinstance(model, str) or not model.lower().endswith((".gguf", ".nctr")) or not os.path.isfile(model):
            raise NeonError("model must be an existing .gguf file")

    def launch(self, model, threads=None):
        b = self.binary()
        if not b:
            raise NeonError("NEON binary not found — build it or set its path")
        self._check_model_path(model)
        if model.lower().endswith(".nctr"):
            raise NeonError("NEON can't run .nctr files yet")
        self.stop()
        t = int(threads) if threads is not None else int(self.settings.get("threads", 0))
        if not 0 <= t <= 256:
            raise NeonError("threads must be 0 (auto) to 256")
        self.proc = Process(b, os.path.abspath(model), t)
        self.proc.start()
        self.settings.update({"last_model": os.path.abspath(model), "threads": t})
        self._save()
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
        return self.proc

    def _watch(self, proc):
        proc.wait_ready(180)
        if self.on_change:
            self.on_change()

    def stop(self):
        if self.proc:
            self.proc.stop()
            self.proc = None
            if self.on_change:
                self.on_change()

    def llm(self):
        return NeonLLM(self.proc) if self.proc and self.proc.state == "ready" else None

    def status(self):
        if not self.scanned:
            self.rescan()
        if self.builder.state == "done" and self._bin_cache[2] is None:
            self.binary(force=True)
        b = self.binary()
        p = self.proc
        return {
            "binary": b, "source": find_source(self.settings["model_dirs"]), "compiler": bool(find_compiler()),
            "build": {"state": self.builder.state, "log": list(self.builder.log)[-40:], "error": self.builder.error},
            "models": self.models, "settings": self.settings, "platform": platform.system(),
            "process": None if not p else {"state": p.state, "model": p.model, "error": p.error, "config": p.config, "threads": p.threads,
                                           "requests": p.requests, "last_ms": p.last_ms, "last_chars": p.last_chars,
                                           "log": list(p.stderr)[-12:]},
        }
