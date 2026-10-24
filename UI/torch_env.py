"""
torch_env.py - find a Python environment that has PyTorch (or make one), for the NERVA UI.  Standard library only.

The UI needs torch in its own process (to read .pt checkpoints and to import tools/train_cvqvae.py). When a .pt is dropped and the
interpreter running the server has no torch, server.py asks the person which environment to use; this module does the work:

  find_envs()          every Python environment it can find on this machine (current interpreter, $VIRTUAL_ENV, $CONDA_PREFIX, .venv/venv
                       next to this repo, ~/.venvs, ~/.virtualenvs, pipenv, poetry, conda/mamba envs, pyenv versions, python3 on PATH)
  probe(python)        what one interpreter has: version, torch/numpy/gguf versions, pip, venv-or-system, PEP 668 "externally managed"
  scan()               find_envs() + probe() in parallel, de-duplicated, sorted (ready first)
  Job                  runs one setup in the background with a live log: use an environment (installing what it lacks) or create a new venv
  relaunch(python)     restart the server under that interpreter (same arguments)

Nothing is installed or created without an explicit request from the UI. Commands that will run are returned by plan() so the dialog can show them.
Torch is installed from the CPU wheels (download.pytorch.org/whl/cpu) on Linux and Windows; on macOS the default PyPI wheel is already CPU/MPS.
"""
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time

NEEDED = ("torch", "numpy", "gguf")            # torch: read .pt + train_cvqvae; numpy: everything; gguf: NEON exporters
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
MIN_PY = (3, 9)
IS_WIN = os.name == "nt"

_PROBE = r'''
import json, os, sys, sysconfig, importlib.util
d = dict(version="%d.%d.%d" % sys.version_info[:3], prefix=sys.prefix, base=getattr(sys, "base_prefix", sys.prefix))
d["in_venv"] = d["prefix"] != d["base"]
try: d["managed"] = os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))
except Exception: d["managed"] = False
for m in ("numpy", "gguf", "torch"):
    d[m] = None
    if importlib.util.find_spec(m) is not None:
        try:
            mod = __import__(m); d[m] = str(getattr(mod, "__version__", "?"))
        except Exception as e:
            d[m] = None; d[m + "_error"] = type(e).__name__
d["pip"] = importlib.util.find_spec("pip") is not None
print("@@" + json.dumps(d))
'''


def _py_in(prefix):
    for rel in (("Scripts", "python.exe"), ("bin", "python3"), ("bin", "python")):
        p = os.path.join(prefix, *rel)
        if os.path.isfile(p):
            return p
    return None


def _children(base):
    try:
        return [os.path.join(base, n) for n in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, n))]
    except OSError:
        return []


def find_envs(root=None):
    """List of (label, python path). Cheap: only looks at well-known places, never walks the disk."""
    home = os.path.expanduser("~")
    root = os.path.abspath(root or os.getcwd())
    cands = [("this server's Python", sys.executable)]
    for var, label in (("VIRTUAL_ENV", "active venv"), ("CONDA_PREFIX", "active conda env")):
        if os.environ.get(var):
            p = _py_in(os.environ[var])
            if p:
                cands.append((label, p))
    bases = []
    for b in (root, os.path.dirname(root), os.getcwd(), home):
        if b not in bases:
            bases.append(b)
    for b in bases:                                                    # .venv / venv / env directly in the project, its parent, cwd, home
        for n in (".venv", "venv", "env", ".virtualenv", "virtualenv"):
            d = os.path.join(b, n)
            if os.path.isdir(d):
                p = _py_in(d)
                if p:
                    cands.append((d, p))
    for b in (root, os.path.dirname(root)):                            # any sibling dir that looks like a venv (has pyvenv.cfg)
        for d in _children(b):
            if os.path.isfile(os.path.join(d, "pyvenv.cfg")):
                p = _py_in(d)
                if p:
                    cands.append((d, p))
    homes = [".venvs", ".virtualenvs", ".local/share/virtualenvs", ".cache/pypoetry/virtualenvs", "miniconda3/envs", "anaconda3/envs",
             "miniforge3/envs", "mambaforge/envs", "micromamba/envs", ".conda/envs", ".pyenv/versions", ".local/share/uv/python"]
    for h in homes + ["/opt/conda/envs", "/opt/miniconda3/envs", "/usr/local/envs"]:
        for d in _children(h if os.path.isabs(h) else os.path.join(home, h)):
            p = _py_in(d)
            if p:
                cands.append((d, p))
    for n in ("python3", "python"):                                    # whatever is first on PATH
        w = shutil.which(n)
        if w:
            cands.append(("python on PATH", w))
    seen, out = set(), []
    for label, p in cands:
        k = os.path.abspath(p)
        if k not in seen:
            seen.add(k)
            out.append((label, k))
    return out


def probe(python, timeout=120):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    try:
        r = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True, timeout=timeout, env=env)
    except Exception as e:
        return dict(python=python, ok=False, error=f"{type(e).__name__}: {e}")
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("@@")), None)
    if r.returncode or not line:
        return dict(python=python, ok=False, error=(r.stderr.strip().splitlines() or ["did not run"])[-1][:200])
    d = json.loads(line[2:])
    d.update(python=python, ok=True)
    d["supported"] = tuple(int(x) for x in d["version"].split(".")[:2]) >= MIN_PY
    d["missing"] = [m for m in NEEDED if not d.get(m)]
    d["ready"] = not d["missing"] and d["supported"]
    d["can_install"] = bool(d["supported"] and d["pip"] and not (d["managed"] and not d["in_venv"]))
    if not d["supported"]:
        d["why_not"] = f"Python {d['version']} is older than {MIN_PY[0]}.{MIN_PY[1]} (PyTorch needs a newer one)"
    elif d["missing"] and not d["pip"]:
        d["why_not"] = "has no pip"
    elif d["missing"] and d["managed"] and not d["in_venv"]:
        d["why_not"] = "system Python is externally managed (PEP 668): pip would refuse; use or create a venv"
    return d


def scan(root=None, workers=6):
    """Probe every found environment in parallel; de-duplicate by prefix; ready ones first, then installable, then the rest."""
    found = find_envs(root)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda lp: (lp[0], probe(lp[1])), found))
    out, seen = [], set()
    for label, d in res:
        if not d.get("ok"):
            continue
        k = os.path.abspath(d["prefix"])
        if k in seen:
            continue
        seen.add(k)
        d["label"] = label
        out.append(d)
    out.sort(key=lambda d: (not d["ready"], not d["can_install"], d["label"] != "this server's Python"))
    return out


def default_new_venv(root):
    for n in (".venv", ".venv-nerva", ".venv-nerva2"):
        p = os.path.join(root, n)
        if not os.path.exists(p):
            return p
    return os.path.join(root, ".venv-nerva-%d" % int(time.time()))


def _pip(python, *args):
    return [python, "-m", "pip", "install", "--disable-pip-version-check", *args]


def plan(python, missing, new_venv=None, base_python=None):
    """The exact commands a setup would run, as lists of strings (shown to the person before anything happens)."""
    cmds = []
    if new_venv:
        cmds.append([base_python or sys.executable, "-m", "venv", new_venv])
        python = _py_in(new_venv) or os.path.join(new_venv, "Scripts" if IS_WIN else "bin", "python" + (".exe" if IS_WIN else ""))
        missing = list(NEEDED)
    small = [m for m in missing if m != "torch"]
    if small:
        cmds.append(_pip(python, *small))
    if "torch" in missing:
        cmds.append(_pip(python, "torch", *(["--index-url", TORCH_CPU_INDEX] if (sys.platform.startswith("linux") or IS_WIN) else [])))
    return cmds, python


class Job:
    """One background setup with a live log. state: idle | running | done | error."""

    def __init__(self):
        self.state, self.lines, self.python, self.error = "idle", [], None, None
        self.lock = threading.Lock()

    def _log(self, s):
        with self.lock:
            self.lines.append(s.rstrip()[:400])
            del self.lines[:-300]

    def snapshot(self):
        with self.lock:
            return dict(state=self.state, lines=self.lines[-60:], python=self.python, error=self.error)

    def start(self, python=None, new_venv=None, base_python=None, on_done=None):
        """python: an existing interpreter to complete; or new_venv: a directory to create. on_done(python) runs on success (in the job thread)."""
        with self.lock:
            if self.state == "running":
                raise RuntimeError("a setup is already running")
            self.state, self.lines, self.python, self.error = "running", [], None, None
        threading.Thread(target=self._run, args=(python, new_venv, base_python, on_done), daemon=True).start()

    def _run(self, python, new_venv, base_python, on_done):
        try:
            if new_venv:
                if os.path.exists(new_venv) and os.listdir(new_venv):
                    raise RuntimeError(f"{new_venv} already exists and is not empty")
                missing = list(NEEDED)
            else:
                d = probe(python)
                if not d.get("ok"):
                    raise RuntimeError(f"cannot run {python}: {d.get('error')}")
                if not d["supported"]:
                    raise RuntimeError(d["why_not"])
                missing = d["missing"]
                if missing and not d["can_install"]:
                    raise RuntimeError(d.get("why_not") or "cannot install into this environment")
            cmds, py = plan(python, missing, new_venv, base_python)
            for c in cmds:
                self._log("$ " + " ".join(c))
                p = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    self._log(line)
                if p.wait() != 0:
                    hint = ""
                    if "-m venv" in " ".join(c) or c[1:3] == ["-m", "venv"]:
                        hint = " (on Debian/Ubuntu: sudo apt install python3-venv)"
                    raise RuntimeError(f"command failed (exit {p.returncode}): {' '.join(c[:5])} ...{hint}")
            d = probe(py)
            if not d.get("ok") or not d["ready"]:
                raise RuntimeError("setup finished but the environment still lacks: " + ", ".join(d.get("missing") or [d.get("error", "?")]))
            self._log(f"ready: torch {d['torch']}, numpy {d['numpy']}, gguf {d['gguf']}  ({py})")
            with self.lock:
                self.python, self.state = py, "done"
            if on_done:
                on_done(py)
        except Exception as e:
            self._log("ERROR: " + str(e))
            with self.lock:
                self.state, self.error = "error", str(e)


def relaunch(python, script, argv):
    """Restart `script` under `python` with the same arguments (never returns on POSIX)."""
    args = [python, script] + [a for a in argv if a != "--no-open"] + ["--no-open"]
    sys.stdout.flush()
    sys.stderr.flush()
    if IS_WIN:                                                         # no real exec on Windows
        subprocess.Popen(args)
        os._exit(0)
    os.execv(python, args)
