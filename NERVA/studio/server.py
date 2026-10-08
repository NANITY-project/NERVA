#!/usr/bin/env python3
"""Nebula Studio server: static UI + JSON API + SSE state stream. Standard library only.

    python3 studio/server.py [--port 8765] [--lib build/libnebula.so]
                             [--llm-url http://127.0.0.1:8080 --model NAME --grammar] [--no-open]

Binds to 127.0.0.1 only. POSTs must be application/json (or the raw import body) and the
Host header must be local, which blocks cross-site and DNS-rebinding requests to a local tool.
"""
import argparse
import atexit
import json
import math
import mimetypes
import os
import queue
import signal
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent as agent_mod   # noqa: E402
import gltf_import          # noqa: E402
import neon_bridge          # noqa: E402
import sim                  # noqa: E402

MAX_JSON = 1 << 20
MAX_IMPORT = 50 << 20
MAX_MODEL = 30 << 20
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

DEMO = [
    ("table", (0.0, 0.40, 1.6), (1.4, 0.8, 0.7), ["fixed", "surface"]),
    ("cup", (0.25, 0.85, 1.5), (0.08, 0.10, 0.08), []),
    ("teapot", (-0.3, 0.92, 1.6), (0.18, 0.16, 0.18), []),
    ("crate", (2.0, 0.2, -0.5), (0.4, 0.4, 0.4), []),
    ("ball", (-1.6, 0.1, 0.4), (0.2, 0.2, 0.2), []),
]


def seed_demo(w):
    for name, pos, size, tags in DEMO:
        w.add_prop(name, pos, size, tags)
    w.add_character("Alice", "A warm, chatty baker who loves serving tea.", -0.9, 0.0, 0.3)
    w.add_character("Bob", "A shy customer who speaks briefly.", 1.2, 3.2, math.pi + 0.2)


class Studio:
    def __init__(self, lib=None, cfg=None, seed=False, neon_settings=None):
        self.lib = lib
        self.cfg = {"llm_url": "", "model": "", "api_key": "", "use_grammar": False, **(cfg or {})}
        self.clients, self.clients_lock = set(), threading.Lock()
        self.agents = None
        self.neon = neon_bridge.Manager(neon_settings)
        self.neon.on_change = self._refresh_llm
        self._build(seed)

    def mode(self):
        if self.neon.llm():
            return "neon"
        return "model" if self.cfg.get("llm_url") else "mock"

    def llm_label(self):
        m = self.mode()
        if m == "neon":
            return "neon \u00b7 " + os.path.basename(self.neon.proc.model)
        return "model \u00b7 " + (self.cfg.get("model") or self.cfg["llm_url"]) if m == "model" else "mock agent"

    def _refresh_llm(self):
        """The active brain: a running NEON wins, then a configured endpoint, else the built-in mock."""
        if self.agents:
            self.agents.set_llm(self.neon.llm() or agent_mod.make_llm(self.cfg))
        self.world.meta = {"llm": self.llm_label(), "mode": self.mode()}

    def _build(self, seed):
        self.world = sim.World(self.lib)
        self.world.log_listeners.append(self._broadcast_log)
        if seed:
            seed_demo(self.world)
        self.world.build_snapshot()
        self.agents = agent_mod.AgentManager(self.world, self.neon.llm() or agent_mod.make_llm(self.cfg))
        self._refresh_llm()
        self.world.build_snapshot()          # includes meta, so the very first read is already correct
        self.world.start()
        self.agents.start()

    def reset(self, seed=True):
        old_w, old_a = self.world, self.agents
        old_a.close()
        old_w.close()
        self._build(seed)

    def _broadcast_log(self, entry):
        data = json.dumps(entry)
        with self.clients_lock:
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def public_config(self):
        return {**{k: v for k, v in self.cfg.items() if k != "api_key"}, "has_key": bool(self.cfg.get("api_key")),
                "mode": self.mode(), "label": self.llm_label()}

    def state(self):
        w = self.world
        return {"snapshot": w.snapshot, "log": list(w.log)[-200:], "config": self.public_config(),
                "takes": [{"id": t["id"], "name": t["name"], "duration": t["duration"], "frames": len(t["frames"])} for t in w.takes]}

    def close(self):
        self.neon.stop()
        self.agents.close()
        self.world.close()


class ApiError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


def _num(v, name, lo=-1e4, hi=1e4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ApiError(f"{name} must be a number") from None
    if not math.isfinite(f) or not lo <= f <= hi:
        raise ApiError(f"{name} out of range")
    return f


def _vec(v, name, n=3, lo=-1e4, hi=1e4):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ApiError(f"{name} must be a list of {n} numbers")
    return [_num(x, name, lo, hi) for x in v]


def _str(v, name, maxlen=2000, required=True):
    if v is None and not required:
        return ""
    if not isinstance(v, str) or (required and not v.strip()):
        raise ApiError(f"{name} must be a non-empty string" if required else f"{name} must be a string")
    return v[:maxlen]


def make_handler(studio, web_dir, vendor_dir):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "NebulaStudio/0.1"

        def log_message(self, *a):
            pass

        # -- plumbing -----------------------------------------------------------
        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
                else (self.headers.get("Host") or "").split("]")[0] + "]"
            return host in LOCAL_HOSTS

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self):
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ApiError("Content-Type must be application/json", 415)
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_JSON:
                raise ApiError("body too large", 413)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise ApiError("invalid JSON") from None
            if not isinstance(d, dict):
                raise ApiError("JSON body must be an object")
            return d

        def _guard(self, fn):
            if not self._host_ok():
                return self._send(403, {"ok": False, "error": "forbidden host"})
            try:
                fn()
            except ApiError as e:
                self._send(e.code, {"ok": False, "error": str(e)})
            except (sim.ActionError, neon_bridge.NeonError) as e:
                self._send(400, {"ok": False, "error": str(e)})
            except queue.Empty:
                self._send(503, {"ok": False, "error": "simulation busy"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        # -- GET --------------------------------------------------------------------
        def do_GET(self):
            self._guard(self._get)

        def _get(self):
            u = urllib.parse.urlparse(self.path)
            p = u.path
            if p == "/api/state":
                return self._send(200, studio.state())
            if p == "/api/stream":
                return self._stream()
            if p == "/api/neon":
                return self._send(200, studio.neon.status())
            if p.startswith("/api/model/"):
                data = studio.world.models.get(urllib.parse.unquote(p.rsplit("/", 1)[1]))
                if data is None:
                    raise ApiError("no model for that character", 404)
                return self._send(200, data, "model/gltf-binary")
            if p == "/api/takes":
                return self._send(200, studio.state()["takes"])
            if p.startswith("/api/take/"):
                try:
                    t = studio.world.takes[int(p.rsplit("/", 1)[1])]
                except (ValueError, IndexError):
                    raise ApiError("no such take", 404) from None
                extra = {"Content-Disposition": f'attachment; filename="{sim.slug(t["name"])}.json"'} \
                    if "download=1" in u.query else None
                return self._send(200, t, extra=extra)
            if p in ("/", "/index.html"):
                return self._file(os.path.join(web_dir, "index.html"))
            for prefix, root in (("/web/", web_dir), ("/vendor/", vendor_dir)):
                if p.startswith(prefix):
                    rel = urllib.parse.unquote(p[len(prefix):])
                    full = os.path.realpath(os.path.join(root, rel))
                    if not full.startswith(os.path.realpath(root) + os.sep):
                        raise ApiError("not found", 404)
                    return self._file(full)
            raise ApiError("not found", 404)

        def _file(self, path):
            if not os.path.isfile(path):
                raise ApiError("not found", 404)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if path.endswith(".js") or path.endswith(".mjs"):
                ctype = "text/javascript"
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)

        def _stream(self):
            q = queue.Queue(maxsize=300)
            with studio.clients_lock:
                studio.clients.add(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            last_t, last_beat = None, time.time()
            try:
                self.wfile.write(b"retry: 1000\n\n")
                while True:
                    w = studio.world
                    snap_t = w.snapshot.get("t")
                    out = b""
                    if snap_t != last_t:
                        out += b"data: " + w.snapshot_json.encode() + b"\n\n"
                        last_t = snap_t
                    try:
                        while True:
                            out += b"event: log\ndata: " + q.get_nowait().encode() + b"\n\n"
                    except queue.Empty:
                        pass
                    if not out and time.time() - last_beat > 15:
                        out = b": hb\n\n"
                    if out:
                        self.wfile.write(out)
                        self.wfile.flush()
                        last_beat = time.time()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with studio.clients_lock:
                    studio.clients.discard(q)

        # -- POST --------------------------------------------------------------------
        def do_POST(self):
            self._guard(self._post)

        def _post(self):
            u = urllib.parse.urlparse(self.path)
            p, w = u.path, studio.world
            if p == "/api/import":
                return self._import(u)
            if p.startswith("/api/character/") and p.endswith("/model"):
                return self._upload_model(p.split("/")[3], u)
            d = self._json_body()
            if p == "/api/command":
                cid, line = _str(d.get("character"), "character"), _str(d.get("line"), "line", 300)
                lines = [l for l in line.split("\n") if l.strip()][:12]
                for l in lines:
                    w.call(lambda w, c=cid, l=l: w.enqueue(c, l))
                return self._send(200, {"ok": True, "queued": len(lines)})
            if p == "/api/stop":
                w.call(lambda w, c=_str(d.get("character"), "character"): w.stop(c))
                return self._send(200, {"ok": True})
            if p == "/api/direct":
                text, who = _str(d.get("text"), "text", 500), d.get("character") or "all"
                ids = list(w.chars) if who == "all" else [_str(who, "character")]
                for cid in ids:
                    if cid not in w.chars:
                        raise ApiError(f"no character '{cid}'", 404)
                for cid in ids:
                    studio.agents.instruct(cid, text)
                w.call(lambda w: w._log("director", f"Director \u2192 {', '.join(ids)}: {text}"))
                return self._send(200, {"ok": True, "to": ids})
            if p == "/api/character":
                cid = w.call(lambda w: w.add_character(
                    _str(d.get("name"), "name", 40), _str(d.get("persona"), "persona", 600, False),
                    _num(d.get("x", 0), "x", -200, 200), _num(d.get("z", 0), "z", -200, 200), _num(d.get("yaw", 0), "yaw", -20, 20)))
                return self._send(200, {"ok": True, "id": cid})
            if p.startswith("/api/character/"):
                return self._character(p.rsplit("/", 1)[1], d)
            if p == "/api/prop":
                name = w.call(lambda w: w.add_prop(
                    _str(d.get("name"), "name", 40), _vec(d.get("pos"), "pos"), _vec(d.get("size", [0.2, 0.2, 0.2]), "size", 3, 0.01, 50),
                    [str(t)[:20] for t in (d.get("tags") or [])][:8]))
                return self._send(200, {"ok": True, "name": name})
            if p.startswith("/api/prop/"):
                return self._prop(p.rsplit("/", 1)[1], d)
            if p == "/api/record":
                act = d.get("action")
                if act == "start":
                    w.call(lambda w: w.start_recording(_str(d.get("name"), "name", 40, False) or None))
                    return self._send(200, {"ok": True})
                if act == "stop":
                    t = w.call(lambda w: w.stop_recording())
                    return self._send(200, {"ok": True, "id": t["id"], "duration": t["duration"], "frames": len(t["frames"])})
                raise ApiError("action must be start or stop")
            if p == "/api/config":
                for k in ("llm_url", "model", "api_key"):
                    if k in d:
                        studio.cfg[k] = _str(d[k], k, 500, False).strip()
                if "use_grammar" in d:
                    studio.cfg["use_grammar"] = bool(d["use_grammar"])
                if studio.cfg["llm_url"] and not studio.cfg["llm_url"].startswith(("http://", "https://")):
                    raise ApiError("llm_url must start with http:// or https://")
                studio._refresh_llm()            # a running NEON still wins; the label follows
                return self._send(200, {"ok": True, "config": studio.public_config()})
            if p == "/api/reset":
                studio.reset(seed=bool(d.get("demo", False)))
                return self._send(200, {"ok": True})
            if p.startswith("/api/neon/"):
                return self._neon(p.rsplit("/", 1)[1], d)
            raise ApiError("not found", 404)

        def _neon(self, action, d):
            n = studio.neon
            if action == "rescan":
                return self._send(200, {"ok": True, "models": n.rescan()})
            if action == "dirs":
                dirs = d.get("dirs")
                if not isinstance(dirs, list) or not dirs or len(dirs) > 10:
                    raise ApiError("dirs must be a list of 1-10 folders")
                return self._send(200, {"ok": True, "models": n.add_dirs([_str(x, "dir", 500) for x in dirs])})
            if action == "binary":
                return self._send(200, {"ok": True, "binary": n.set_binary(_str(d.get("path"), "path", 500))})
            if action == "build":
                return self._send(200, {"ok": True, "source": n.build()})
            if action == "probe":
                return self._send(200, {"ok": True, "result": n.probe(_str(d.get("model"), "model", 1000))})
            if action == "launch":
                n.launch(_str(d.get("model"), "model", 1000), None if d.get("threads") is None else int(_num(d.get("threads"), "threads", 0, 256)))
                return self._send(200, {"ok": True})
            if action == "stop":
                n.stop()
                studio._refresh_llm()
                return self._send(200, {"ok": True})
            raise ApiError("not found", 404)

        def _upload_model(self, cid, u):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_MODEL:
                raise ApiError("model body missing or larger than 30 MB", 413)
            data = self.rfile.read(n)
            try:
                doc = gltf_import.parse(data)
            except gltf_import.GltfError as e:
                raise ApiError(f"not a readable glTF/GLB: {e}") from None
            if not doc.get("meshes"):
                raise ApiError("that file has no meshes")
            name = (urllib.parse.parse_qs(u.query).get("name") or ["model.glb"])[0][:60]
            ver = studio.world.call(lambda w: w.set_model(cid, data, name, bool(doc.get("skins"))))
            self._send(200, {"ok": True, "version": ver, "skinned": bool(doc.get("skins")), "name": name})

        def _character(self, cid, d):
            w = studio.world
            if cid not in w.chars:
                raise ApiError(f"no character '{cid}'", 404)

            def apply(w):
                c = w.chars[cid]
                if "persona" in d:
                    c.persona = _str(d["persona"], "persona", 600, False)
                if "name" in d:
                    c.name = _str(d["name"], "name", 40)
                if "agent" in d:
                    c.agent_on = bool(d["agent"])
                if "x" in d or "z" in d:
                    c.x, c.z = _num(d.get("x", c.x), "x", -200, 200), _num(d.get("z", c.z), "z", -200, 200)
                if "yaw" in d:
                    c.yaw = _num(d["yaw"], "yaw", -20, 20)
                if d.get("clear_model"):
                    w.set_model(cid, None)
                if d.get("delete"):
                    w.remove_character(cid)
            w.call(apply)
            self._send(200, {"ok": True})

        def _prop(self, name, d):
            w = studio.world
            if name not in w.props:
                raise ApiError(f"no prop '{name}'", 404)

            def apply(w):
                pr = w.props[name]
                if d.get("delete"):
                    for c in w.chars.values():
                        for h in c.hands.values():
                            if h.holding == name:
                                h.holding, h.mode = None, "rest"
                    del w.props[name]
                    return
                if "pos" in d:
                    pr.pos = _vec(d["pos"], "pos")
                    pr.vy = 0.0
                if "size" in d:
                    pr.size = _vec(d["size"], "size", 3, 0.01, 50)
                if "tags" in d:
                    pr.tags = [str(t)[:20] for t in d["tags"]][:8]
            w.call(apply)
            self._send(200, {"ok": True})

        def _import(self, u):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_IMPORT:
                raise ApiError("import body missing or larger than 50 MB", 413)
            data = self.rfile.read(n)
            try:
                props, skipped, bounds = gltf_import.import_bytes(data)
            except gltf_import.GltfError as e:
                raise ApiError(f"glTF import failed: {e}") from None
            except (KeyError, IndexError, TypeError) as e:
                raise ApiError(f"malformed glTF ({type(e).__name__})") from None
            replace = "keep=1" not in u.query

            def apply(w):
                if replace:
                    w.clear_props("mesh")
                return [w.add_prop(p["name"], p["pos"], p["size"], p["tags"], kind="mesh", color="#6b0000") for p in props]
            names = studio.world.call(apply)
            studio.world.call(lambda w: w._log("system", f"Imported environment: {len(names)} props, {len(skipped)} skipped"))
            self._send(200, {"ok": True, "props": names, "skipped": [s[0] for s in skipped], "bounds": bounds})

    return H


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return                       # a browser tab closing mid-request is not an error
        super().handle_error(request, client_address)


def make_server(studio, port=0, host="127.0.0.1"):
    web, vendor = os.path.join(HERE, "web"), os.path.join(HERE, "vendor")
    srv = _Server((host, port), make_handler(studio, web, vendor))
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description="Nebula Studio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lib", default=os.environ.get("NEBULA_LIB"))
    ap.add_argument("--llm-url", default=os.environ.get("NEBULA_LLM_URL", ""))
    ap.add_argument("--model", default=os.environ.get("NEBULA_LLM_MODEL", ""))
    ap.add_argument("--grammar", action="store_true", help="send a GBNF grammar with each request")
    ap.add_argument("--demo", action="store_true", help="start with the demo scene (otherwise the scene is empty; the Guide can load it)")
    ap.add_argument("--neon-bin", default=os.environ.get("NEON_BIN"), help="path to the NEON binary (default: auto-detect)")
    ap.add_argument("--neon-model", help="launch this .gguf in NEON at startup")
    ap.add_argument("--model-dir", action="append", default=[], help="extra folder to scan for models (repeatable)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    studio = Studio(a.lib, {"llm_url": a.llm_url, "model": a.model, "use_grammar": a.grammar}, seed=a.demo)
    if a.neon_bin:
        try: studio.neon.set_binary(a.neon_bin)
        except neon_bridge.NeonError as e: print(f"warning: {e}", file=sys.stderr)
    if a.model_dir:
        try: studio.neon.add_dirs(a.model_dir)
        except neon_bridge.NeonError as e: print(f"warning: {e}", file=sys.stderr)
    if a.neon_model:
        try: studio.neon.launch(a.neon_model)
        except neon_bridge.NeonError as e: print(f"warning: {e}", file=sys.stderr)
    srv = make_server(studio, a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    nb = studio.neon.binary()
    print(f"Nebula Studio on {url}")
    print(f"  NEON runtime: {nb or 'not found (open Model in the UI to build or locate it)'}")
    loading = studio.neon.proc is not None and studio.neon.proc.state == "starting"
    print(f"  brain: {studio.llm_label()}" + ("  (NEON model loading; the top-bar pill switches when it is ready)" if loading else ""))
    atexit.register(studio.close)                                    # NEON is a child process: never leave it behind
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.close()


if __name__ == "__main__":
    main()
