import http.client, http.server, json, math, os, socket, struct, sys, threading, time, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
LIB = os.path.join(os.path.dirname(__file__), "..", "..", "build", "libnebula.so")
os.environ.setdefault("NEBULA_LIB", LIB)
import make_tiny_gguf
import server as srvmod

import neon_bridge
NEON_BIN = os.environ.get("NEON_BIN") or neon_bridge.find_binary() or ""         # real-NEON tests skip when none is found
HAVE_NEON = bool(NEON_BIN) and os.path.isfile(NEON_BIN)


def glb(doc):
    j = json.dumps(doc).encode(); j += b" " * (-len(j) % 4)
    return struct.pack("<4sII", b"glTF", 2, 20 + len(j)) + struct.pack("<II", len(j), 0x4E4F534A) + j


def room_glb():
    nodes = [{"mesh": 0, "name": "Sofa", "translation": [-2, 0.4, 2]}, {"mesh": 0, "name": "Vase", "translation": [0, 1, 0], "scale": [0.1, 0.2, 0.1]},
             {"mesh": 0, "name": "Floor", "scale": [20, 0.1, 20]}]
    return glb({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0, 1, 2]}], "nodes": nodes, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"min": [-1, -0.5, -0.5], "max": [1, 0.5, 0.5]}]})


class StubLLM(http.server.BaseHTTPRequestHandler):
    seen = []
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubLLM.seen.append((self.path, body, self.headers.get("Authorization")))
        out = json.dumps({"choices": [{"message": {"content": "LOOK_AT bob\nSAY Hello from the model"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.mdir = os.path.join(cls.tmp, "models"); os.makedirs(cls.mdir)
        make_tiny_gguf.build(os.path.join(cls.mdir, "good.gguf"), ctx=4096); make_tiny_gguf.build(os.path.join(cls.mdir, "small.gguf"), ctx=512)
        make_tiny_gguf.build(os.path.join(cls.mdir, "llama.gguf"), arch="llama")
        os.environ["NEBULA_MODEL_DIRS"] = cls.mdir
        if HAVE_NEON: os.environ["NEON_BIN"] = NEON_BIN
        cls.studio = srvmod.Studio(LIB, seed=True, neon_settings=os.path.join(cls.tmp, "settings.json"))
        cls.srv = srvmod.make_server(cls.studio, 0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubLLM)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.stub.shutdown(); cls.studio.close()

    def req(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        h = dict(headers or {})
        if raw is None and body is not None:
            raw = json.dumps(body).encode(); h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=raw, headers=h)
        r = c.getresponse(); data = r.read(); c.close()
        try: return r.status, json.loads(data), r
        except ValueError: return r.status, data, r

    def state(self): return self.req("GET", "/api/state")[1]
    def wait(self, cond, limit=60):
        t = time.time()
        while time.time() - t < limit:
            if cond(self.state()["snapshot"]): return True
            time.sleep(0.1)
        return False
    def char(self, snap, cid): return next(c for c in snap["chars"] if c["id"] == cid)

    # ---------------------------------------------------------------- contract
    def test_01_state_and_static(self):
        s = self.state()
        self.assertEqual({c["id"] for c in s["snapshot"]["chars"]}, {"alice", "bob"})
        self.assertGreaterEqual(len(s["snapshot"]["props"]), 5)
        self.assertEqual(s["config"]["mode"], "mock"); self.assertNotIn("api_key", s["config"])
        self.assertEqual(s["snapshot"]["meta"]["llm"], "mock agent")
        code, body, r = self.req("GET", "/")
        self.assertIn(code, (200, 404))                      # index.html is written later in the build; both fine here

    def test_02_command_flow_and_errors(self):
        code, body, _ = self.req("POST", "/api/command", {"character": "alice", "line": "WAVE right"})
        self.assertEqual((code, body["ok"]), (200, True))
        self.assertTrue(self.wait(lambda s: self.char(s, "alice")["busy"] is not None), "alice should start waving")
        self.assertTrue(self.wait(lambda s: self.char(s, "alice")["busy"] is None, 30))
        for payload, frag in (({"character": "alice", "line": "DANCE"}, "unknown command"), ({"character": "ghost", "line": "WAIT 1"}, "no character"),
                              ({"character": "alice", "line": "REACH right nan 1 1"}, "non-finite"), ({"line": "WAIT 1"}, "character"), ({"character": "alice"}, "line")):
            code, body, _ = self.req("POST", "/api/command", payload)
            self.assertEqual(code, 400, payload); self.assertIn(frag, body["error"])

    def test_03_security(self):
        self.assertEqual(self.req("POST", "/api/command", raw=b'{"character":"alice","line":"WAIT 1"}', headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("GET", "/api/state", headers={"Host": "evil.example.com"})[0], 403)
        self.assertEqual(self.req("GET", "/api/state", headers={"Host": "127.0.0.1.evil.com"})[0], 403)
        for p in ("/web/../server.py", "/web/%2e%2e/server.py", "/vendor/../../etc/passwd", "/web//etc/passwd"):
            self.assertEqual(self.req("GET", p)[0], 404, p)
        self.assertEqual(self.req("POST", "/api/command", raw=b"{not json", headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.req("POST", "/api/command", raw=b"[1,2]", headers={"Content-Type": "application/json"})[0], 400)
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall(b"POST /api/import HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 99999999\r\n\r\n")
        self.assertIn(b"413", s.recv(200)); s.close()

    def test_04_sse_stream(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/api/stream"); r = c.getresponse()
        self.assertEqual(r.status, 200); self.assertIn("text/event-stream", r.getheader("Content-Type"))
        self.req("POST", "/api/command", {"character": "bob", "line": "SAY ping from sse test"})
        ts, saw_log, t0 = [], False, time.time()
        while (len(ts) < 6 or not saw_log) and time.time() - t0 < 10:
            line = r.fp.readline().decode()
            if line.startswith("data: ") and not saw_log or line.startswith("data: ") and len(ts) < 6:
                try:
                    d = json.loads(line[6:])
                    if "chars" in d: ts.append(d["t"])
                    elif "ping from sse test" in d.get("text", ""): saw_log = True
                except ValueError: pass
            if line.startswith("event: log"):
                d = json.loads(r.fp.readline().decode()[6:])
                saw_log |= "ping from sse test" in d["text"] or d["kind"] == "cmd"
        c.close()
        self.assertGreaterEqual(len(ts), 6); self.assertEqual(ts, sorted(ts)); self.assertGreater(ts[-1] - ts[0], 0.1)
        self.assertTrue(saw_log, "log events must be streamed")

    # ---------------------------------------------------------------- editing
    def test_05_characters_and_props(self):
        code, b, _ = self.req("POST", "/api/character", {"name": "Cara", "persona": "a guard", "x": 4, "z": 4})
        self.assertEqual((code, b["id"]), (200, "cara"))
        self.assertEqual(self.req("POST", "/api/character/cara", {"persona": "a very tall guard", "agent": True})[0], 200)
        self.assertEqual(self.char(self.state()["snapshot"], "cara")["persona"], "a very tall guard")
        for bad in ({"name": ""}, {"name": "X", "x": "far"}, {"name": "X", "x": 1e9}, {"name": "X", "yaw": float("nan")}):
            self.assertEqual(self.req("POST", "/api/character", raw=json.dumps(bad).encode(), headers={"Content-Type": "application/json"})[0], 400, bad)
        self.assertEqual(self.req("POST", "/api/character/ghost", {"persona": "x"})[0], 404)
        self.assertEqual(self.req("POST", "/api/character/cara", {"delete": True})[0], 200)
        self.assertNotIn("cara", {c["id"] for c in self.state()["snapshot"]["chars"]})
        code, b, _ = self.req("POST", "/api/prop", {"name": "Lamp", "pos": [1, 0.5, 1], "size": [0.2, 1, 0.2], "tags": ["light"]})
        self.assertEqual((code, b["name"]), (200, "lamp"))
        self.assertEqual(self.req("POST", "/api/prop/lamp", {"pos": [2, 0.5, 2]})[0], 200)
        self.assertEqual(next(p for p in self.state()["snapshot"]["props"] if p["name"] == "lamp")["pos"][0], 2)
        self.assertEqual(self.req("POST", "/api/prop", {"name": "bad", "pos": [1, 2]})[0], 400)
        self.assertEqual(self.req("POST", "/api/prop/lamp", {"delete": True})[0], 200)

    def test_06_import_environment(self):
        code, b, _ = self.req("POST", "/api/import", raw=room_glb(), headers={"Content-Type": "model/gltf-binary"})
        self.assertEqual(code, 200, b); self.assertEqual(sorted(b["props"]), ["sofa", "vase"]); self.assertEqual(b["skipped"], ["Floor"])
        names = {p["name"] for p in self.state()["snapshot"]["props"]}
        self.assertTrue({"sofa", "vase", "cup", "table"} <= names, "import keeps hand-made props, adds mesh props")
        self.req("POST", "/api/import", raw=room_glb(), headers={"Content-Type": "model/gltf-binary"})
        self.assertEqual(sum(p["kind"] == "mesh" for p in self.state()["snapshot"]["props"]), 2, "re-import replaces, never duplicates")
        self.assertEqual(self.req("POST", "/api/import", raw=b"not a model at all, sorry", headers={"Content-Type": "application/octet-stream"})[0], 400)

    def test_07_recording_and_takes(self):
        self.assertEqual(self.req("POST", "/api/record", {"action": "stop"})[0], 400)
        self.assertEqual(self.req("POST", "/api/record", {"action": "start", "name": "demo take"})[0], 200)
        self.assertEqual(self.req("POST", "/api/record", {"action": "start"})[0], 400)
        self.req("POST", "/api/command", {"character": "alice", "line": "WAVE left"}); time.sleep(1.5)
        code, b, _ = self.req("POST", "/api/record", {"action": "stop"})
        self.assertEqual(code, 200); self.assertGreater(b["frames"], 30); self.assertAlmostEqual(b["duration"], 1.5, delta=0.5)
        code, take, r = self.req("GET", f"/api/take/{b['id']}?download=1")
        self.assertEqual(code, 200); self.assertIn("demo_take.json", r.getheader("Content-Disposition"))
        self.assertEqual(len(take["frames"][0]["chars"][0]["arms"]["left"]), 3)
        self.assertEqual(self.req("GET", "/api/take/999")[0], 404)

    # ---------------------------------------------------------------- the model path
    def test_08_config_and_real_http_model_client(self):
        self.assertEqual(self.req("POST", "/api/config", {"llm_url": "ftp://nope"})[0], 400)
        url = f"http://127.0.0.1:{self.stub.server_address[1]}"
        code, b, _ = self.req("POST", "/api/config", {"llm_url": url, "model": "nanity-1.7b", "api_key": "sekret", "use_grammar": True})
        self.assertEqual(code, 200); self.assertEqual(b["config"]["mode"], "model"); self.assertTrue(b["config"]["has_key"]); self.assertNotIn("sekret", json.dumps(b))
        StubLLM.seen.clear()
        self.assertEqual(self.req("POST", "/api/direct", {"character": "bob", "text": "greet Alice"})[0], 200)
        t = time.time()
        while not StubLLM.seen and time.time() - t < 15: time.sleep(0.1)
        self.assertTrue(StubLLM.seen, "model endpoint was never called")
        path, body, auth = StubLLM.seen[0]
        self.assertEqual(path, "/v1/chat/completions"); self.assertEqual(body["model"], "nanity-1.7b"); self.assertEqual(auth, "Bearer sekret")
        self.assertIn("root   ::=", body["grammar"]); self.assertIn('"alice"', body["grammar"])
        self.assertIn("greet Alice", body["messages"][-1]["content"]); self.assertEqual(body["messages"][0]["role"], "system")
        t = time.time()
        while time.time() - t < 20 and not any("Hello from the model" in e["text"] for e in self.state()["log"]): time.sleep(0.2)
        self.assertTrue(any("Hello from the model" in e["text"] for e in self.state()["log"]), "model output must reach the world as a SAY")
        self.req("POST", "/api/config", {"llm_url": "", "api_key": "", "use_grammar": False})
        self.assertEqual(self.state()["config"]["mode"], "mock")

    def test_09_director_end_to_end_with_mock_agent(self):
        self.assertEqual(self.req("POST", "/api/direct", {"character": "ghost", "text": "x"})[0], 404)
        self.assertEqual(self.req("POST", "/api/direct", {"text": ""})[0], 400)
        self.assertEqual(self.req("POST", "/api/direct", {"character": "alice", "text": "pick up the cup"})[0], 200)
        self.assertTrue(self.wait(lambda s: next(p for p in s["props"] if p["name"] == "cup")["held"], 60), "alice should pick up the cup")
        self.assertTrue(self.wait(lambda s: self.char(s, "alice")["busy"] is None, 20))
        self.assertEqual(self.char(self.state()["snapshot"], "alice")["holding"]["right"], "cup")

    # ---------------------------------------------------------------- NEON, driven for real
    @unittest.skipUnless(HAVE_NEON, "no real NEON binary")
    def test_10_neon_discovery_launch_and_use(self):
        st = self.req("GET", "/api/neon")[1]
        self.assertEqual(st["binary"], os.path.abspath(NEON_BIN)); self.assertIsNone(st["process"])
        by = {m["file"]: m for m in st["models"]}
        self.assertTrue(by["good.gguf"]["runnable"]); self.assertFalse(by["llama.gguf"]["runnable"]); self.assertIn("nanity_convert", by["llama.gguf"]["note"])
        good, llama = by["good.gguf"]["path"], by["llama.gguf"]["path"]
        # refusals
        for payload, frag in (({"model": "/etc/passwd"}, ".gguf"), ({"model": "nope.gguf"}, ".gguf"), ({}, "model"), ({"model": good, "threads": 9999}, "range")):
            code, b, _ = self.req("POST", "/api/neon/launch", payload); self.assertEqual(code, 400, payload); self.assertIn(frag, b["error"])
        self.assertEqual(self.req("POST", "/api/neon/binary", {"path": "/bin/ls"})[0], 400)
        self.assertEqual(self.req("POST", "/api/neon/dirs", {"dirs": ["/no/such/folder"]})[0], 400)
        self.assertEqual(self.req("POST", "/api/neon/dirs", {"dirs": []})[0], 400)
        code, b, _ = self.req("POST", "/api/neon/probe", {"model": good}); self.assertEqual((code, b["result"]["ok"]), (200, True))
        self.assertFalse(self.req("POST", "/api/neon/probe", {"model": llama})[1]["result"]["ok"])
        # wrong-architecture model: NEON's own message comes back through the API
        self.assertEqual(self.req("POST", "/api/neon/launch", {"model": llama})[0], 200)
        t = time.time()
        while time.time() - t < 20 and (self.req("GET", "/api/neon")[1]["process"] or {}).get("state") != "error": time.sleep(0.2)
        pr = self.req("GET", "/api/neon")[1]["process"]; self.assertEqual(pr["state"], "error"); self.assertIn("nanity", pr["error"])
        self.assertEqual(self.state()["config"]["mode"], "mock", "a failed model must not become the active brain")
        # the good model
        self.addCleanup(lambda: self.req("POST", "/api/neon/stop", {}))
        self.assertEqual(self.req("POST", "/api/neon/launch", {"model": good, "threads": 1})[0], 200)
        t = time.time()
        while time.time() - t < 30 and (self.req("GET", "/api/neon")[1]["process"] or {}).get("state") != "ready": time.sleep(0.2)
        pr = self.req("GET", "/api/neon")[1]["process"]; self.assertEqual(pr["state"], "ready"); self.assertEqual(pr["config"]["ctx_len"], 4096)
        self.assertTrue(self.wait(lambda s: s["meta"]["mode"] == "neon", 5)); self.assertTrue(self.state()["snapshot"]["meta"]["llm"].startswith("neon"))
        self.assertEqual(self.state()["config"]["mode"], "neon")
        # a director prompt now really goes to NEON
        self.assertEqual(self.req("POST", "/api/direct", {"character": "bob", "text": "say hello"})[0], 200)
        t = time.time()
        while time.time() - t < 40 and (self.req("GET", "/api/neon")[1]["process"] or {}).get("requests", 0) < 1: time.sleep(0.2)
        pr = self.req("GET", "/api/neon")[1]["process"]; self.assertGreaterEqual(pr["requests"], 1); self.assertGreater(pr["last_chars"], 0)
        self.assertEqual(self.req("POST", "/api/neon/stop", {})[0], 200)
        self.assertTrue(self.wait(lambda s: s["meta"]["mode"] == "mock", 5)); self.assertIsNone(self.req("GET", "/api/neon")[1]["process"])

    @unittest.skipUnless(HAVE_NEON, "no real NEON binary")
    def test_12_context_too_small_is_explained_to_the_user(self):
        small = next(m["path"] for m in self.req("GET", "/api/neon")[1]["models"] if m["file"] == "small.gguf")
        self.addCleanup(lambda: self.req("POST", "/api/neon/stop", {}))
        self.assertEqual(self.req("POST", "/api/neon/launch", {"model": small, "threads": 1})[0], 200)
        t = time.time()
        while time.time() - t < 30 and (self.req("GET", "/api/neon")[1]["process"] or {}).get("state") != "ready": time.sleep(0.2)
        self.req("POST", "/api/character", {"name": "Zoe", "persona": "x" * 100}); n0 = len(self.state()["log"])
        self.req("POST", "/api/direct", {"character": "zoe", "text": "hello"})
        t = time.time()
        while time.time() - t < 20 and not any("does not fit" in e["text"] for e in self.state()["log"][n0:]): time.sleep(0.2)
        msg = next(e["text"] for e in self.state()["log"] if "does not fit" in e["text"])
        print(f"\n    shown in the transcript: {msg[:150]}")
        self.assertIn("512", msg); self.assertEqual(self.req("GET", "/api/neon")[1]["process"]["state"], "ready", "the model keeps running")
        self.req("POST", "/api/character/zoe", {"delete": True})

    # ---------------------------------------------------------------- character models
    def test_11_character_model_upload(self):
        glb_bytes = room_glb()
        code, b, _ = self.req("POST", "/api/character/alice/model?name=hero.glb", raw=glb_bytes, headers={"Content-Type": "model/gltf-binary"})
        self.assertEqual(code, 200, b); self.assertEqual((b["version"], b["skinned"], b["name"]), (1, False, "hero.glb"))
        c = self.char(self.state()["snapshot"], "alice"); self.assertEqual((c["model"], c["model_name"]), (1, "hero.glb"))
        code, data, r = self.req("GET", "/api/model/alice"); self.assertEqual((code, r.getheader("Content-Type")), (200, "model/gltf-binary")); self.assertEqual(data, glb_bytes)
        rigged = json.loads(glb_bytes[20:20 + struct.unpack_from("<I", glb_bytes, 12)[0]]); rigged["skins"] = [{"joints": [0]}]
        code, b, _ = self.req("POST", "/api/character/alice/model", raw=glb(rigged), headers={"Content-Type": "model/gltf-binary"})
        self.assertEqual((code, b["version"], b["skinned"]), (200, 2, True), "re-upload bumps the version so clients refetch")
        for raw, frag in ((b"definitely not a model", "readable"), (glb({"asset": {"version": "2.0"}}), "no meshes")):
            code, b, _ = self.req("POST", "/api/character/alice/model", raw=raw, headers={"Content-Type": "model/gltf-binary"}); self.assertEqual(code, 400); self.assertIn(frag, b["error"])
        self.assertEqual(self.req("POST", "/api/character/ghost/model", raw=glb_bytes, headers={"Content-Type": "model/gltf-binary"})[0], 400)
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall(b"POST /api/character/alice/model HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 99999999\r\n\r\n"); self.assertIn(b"413", s.recv(200)); s.close()
        self.assertEqual(self.req("GET", "/api/model/bob")[0], 404)
        self.assertEqual(self.req("POST", "/api/character/alice", {"clear_model": True})[0], 200)
        c = self.char(self.state()["snapshot"], "alice"); self.assertEqual(c["model"], 0); self.assertEqual(self.req("GET", "/api/model/alice")[0], 404)

    def test_99_reset(self):
        self.assertEqual(self.req("POST", "/api/reset", {})[0], 200)
        s = self.state()["snapshot"]; self.assertEqual((s["chars"], s["props"]), ([], []), "the default scene is empty")
        self.assertEqual(self.req("POST", "/api/reset", {"demo": True})[0], 200); self.assertEqual(len(self.state()["snapshot"]["chars"]), 2)
        self.assertEqual(self.req("POST", "/api/reset", {"demo": False})[0], 200)
        s = self.state()["snapshot"]; self.assertEqual((s["chars"], s["props"]), ([], []))


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
