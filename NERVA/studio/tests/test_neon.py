import json, os, shutil, signal, stat, subprocess, sys, tempfile, threading, time, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import neon_bridge as nb
import make_tiny_gguf
from agent import LLMError

# Real-NEON tests run when a NEON checkout/binary can be found (env NEON_SRC / NEON_BIN, or auto-detected next to this
# project) and are skipped otherwise.
NEON_SRC = os.environ.get("NEON_SRC") or nb.find_source() or ""
NEON_BIN = os.environ.get("NEON_BIN") or nb.find_binary() or ""
HAVE_REAL = bool(NEON_BIN) and os.path.isfile(NEON_BIN)


def script(path, body):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env python3\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path

USAGE = "import sys\nif len(sys.argv) < 3:\n    sys.stderr.write('Usage: x --model <path.gguf> [--interactive | --probe]\\n'); sys.exit(1)\n"


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.models = os.path.join(cls.d, "models"); os.makedirs(os.path.join(cls.models, "sub", "deeper", "toodeep", "x"))
        make_tiny_gguf.build(os.path.join(cls.models, "good.gguf"))
        make_tiny_gguf.build(os.path.join(cls.models, "sub", "llama.gguf"), arch="llama")
        make_tiny_gguf.build(os.path.join(cls.models, "future.gguf"), spec=2)
        make_tiny_gguf.build(os.path.join(cls.models, "sub", "deeper", "toodeep", "x", "hidden.gguf"))
        put = lambda name, data: (lambda f: (f.write(data), f.close()))(open(os.path.join(cls.models, name), "wb"))
        with open(os.path.join(cls.models, "good.gguf"), "rb") as f:
            good = f.read()
        put("garbage.gguf", os.urandom(300)); put("truncated.gguf", good[:200]); put("weights.nctr", b"NCTR...."); put("notes.txt", b"hi")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)


class GgufAndScan(Fixtures):
    def test_header_reader(self):
        m = nb.read_gguf_meta(os.path.join(self.models, "good.gguf"))
        self.assertEqual((m["general.architecture"], m["nanity.spec_version"], m["nanity.context_length"]), ("nanity", 1, 512))
        self.assertEqual(m["tokenizer.ggml.tokens.count"], 274)
        for bad, frag in (("garbage.gguf", "magic"), ("truncated.gguf", "")):
            with self.assertRaises(nb.GgufError, msg=bad): nb.read_gguf_meta(os.path.join(self.models, bad))
        with self.assertRaises(nb.GgufError): nb.read_gguf_meta(os.path.join(self.models, "nope.gguf"))

    def test_header_reader_survives_hostile_input(self):
        import struct
        evil = [b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40),                                   # absurd kv count
                b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 60),             # absurd string length
                b"GGUF" + struct.pack("<IQQ", 9, 0, 0),                                          # bad version
                b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"k" + struct.pack("<I", 99)]  # bad type
        for i, blob in enumerate(evil):
            p = os.path.join(self.d, f"evil{i}.gguf")
            with open(p, "wb") as f: f.write(blob)
            with self.assertRaises(nb.GgufError, msg=f"evil{i}"): nb.read_gguf_meta(p)

    def test_scan_classifies_models_and_respects_depth(self):
        found = {m["file"]: m for m in nb.scan_models([self.models])}
        self.assertTrue(found["good.gguf"]["runnable"]); self.assertEqual(found["good.gguf"]["ctx_len"], 512)
        self.assertFalse(found["llama.gguf"]["runnable"]); self.assertIn("nanity_convert", found["llama.gguf"]["note"])
        self.assertFalse(found["future.gguf"]["runnable"]); self.assertIn("spec version 2", found["future.gguf"]["note"])
        self.assertIn("unreadable", found["garbage.gguf"]["note"]); self.assertIn("unreadable", found["truncated.gguf"]["note"])
        self.assertIn(".nctr", found["weights.nctr"]["note"]); self.assertFalse(found["weights.nctr"]["runnable"])
        self.assertNotIn("notes.txt", found); self.assertNotIn("hidden.gguf", found, "depth limit")
        order = [m["file"] for m in nb.scan_models([self.models])]
        self.assertEqual(order[0], "good.gguf", "runnable models are listed first")

    def test_default_dirs_honour_env(self):
        os.environ["NEBULA_MODEL_DIRS"] = self.models
        try: self.assertIn(self.models, nb.default_model_dirs())
        finally: del os.environ["NEBULA_MODEL_DIRS"]


class Locating(Fixtures):
    def test_verify_rejects_impostors(self):
        fake_ls = os.path.join(self.d, "notneon_but_ls"); shutil.copy("/bin/ls", fake_ls)
        self.assertFalse(nb.verify_binary(fake_ls)[0])                                           # runs, but isn't NEON
        self.assertFalse(nb.verify_binary(shutil.which("ls"))[0])                                # name check
        self.assertFalse(nb.verify_binary(os.path.join(self.d, "missing_neon"))[0])
        self.assertFalse(nb.verify_binary(self.models)[0])
        slow = script(os.path.join(self.d, "neon_slow"), "import time; time.sleep(30)\n")
        t = time.time(); ok, msg = nb.verify_binary(slow, timeout=1); self.assertFalse(ok); self.assertLess(time.time() - t, 4)
        good = script(os.path.join(self.d, "neon_fake"), USAGE + "sys.stderr.write('--interactive')\n")
        self.assertTrue(nb.verify_binary(good)[0])

    @unittest.skipUnless(HAVE_REAL, "no real NEON binary")
    def test_real_binary_is_recognised_and_found(self):
        self.assertTrue(nb.verify_binary(NEON_BIN)[0])
        os.environ["NEON_BIN"] = NEON_BIN
        try: self.assertEqual(nb.find_binary(), os.path.abspath(NEON_BIN))
        finally: del os.environ["NEON_BIN"]
        self.assertEqual(nb.find_binary(configured="/nonexistent/neon", extra_dirs=[NEON_SRC]), os.path.abspath(NEON_BIN))
        self.assertEqual(nb.find_source([NEON_SRC]), NEON_SRC)

    @unittest.skipUnless(HAVE_REAL, "no real NEON binary")
    def test_probe_real_binary(self):
        ok = nb.probe_model(NEON_BIN, os.path.join(self.models, "good.gguf"))
        self.assertTrue(ok["ok"], ok); self.assertEqual(ok["config"]["ctx_len"], 512); self.assertEqual(ok["config"]["n_layer"], 2)
        for name in ("sub/llama.gguf", "future.gguf", "garbage.gguf", "missing.gguf"):
            r = nb.probe_model(NEON_BIN, os.path.join(self.models, name))
            self.assertFalse(r["ok"], name); self.assertTrue(r["message"], name)
            print(f"\n    probe {name}: {r['message'][:110]}")


@unittest.skipUnless(HAVE_REAL, "no real NEON binary")
class RealProcess(Fixtures):
    def start(self, **kw):
        p = nb.Process(NEON_BIN, os.path.join(self.models, "good.gguf"), **kw); p.start(); self.addCleanup(p.stop)
        self.assertTrue(p.wait_ready(60), p.error or list(p.stderr)); return p

    def test_ready_complete_and_stats(self):
        p = self.start()
        self.assertEqual(p.state, "ready"); self.assertEqual(p.config["ctx_len"], 512)
        text = p.complete([{"role": "system", "content": "You are Bob."}, {"role": "user", "content": "hello"}], max_tokens=10)
        self.assertGreater(len(text), 3); self.assertEqual(p.requests, 1); self.assertGreater(p.last_ms, 0)
        p.complete([{"role": "user", "content": "again"}], max_tokens=4); self.assertEqual(p.requests, 2)

    def test_concurrent_callers_are_serialized(self):
        p = self.start(); out, errs = [], []
        def go(i):
            try: out.append(p.complete([{"role": "user", "content": f"caller {i}"}], max_tokens=6))
            except Exception as e: errs.append(e)
        ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]; [t.start() for t in ts]; [t.join(60) for t in ts]
        self.assertEqual(errs, []); self.assertEqual(len(out), 6); self.assertEqual(p.requests, 6)
        self.assertTrue(all(isinstance(o, str) and o for o in out))

    def test_crash_is_reported_not_hung(self):
        p = self.start(); p.complete([{"role": "user", "content": "x"}], max_tokens=2)
        os.kill(p._p.pid, signal.SIGKILL); time.sleep(0.5)
        self.assertEqual(p.state, "error"); self.assertTrue(p.error)
        t = time.time()
        with self.assertRaises(nb.NeonError): p.complete([{"role": "user", "content": "x"}], max_tokens=2)
        self.assertLess(time.time() - t, 5)

    def test_bad_model_fails_with_neons_own_message(self):
        p = nb.Process(NEON_BIN, os.path.join(self.models, "sub", "llama.gguf")); p.start(); self.addCleanup(p.stop)
        p.wait_ready(30); self.assertEqual(p.state, "error"); self.assertTrue(p.error); print(f"\n    wrong-arch error shown to user: {p.error[:120]}")

    def test_prompt_budget_uses_the_models_context(self):
        p = self.start()
        with self.assertRaises(nb.NeonError) as cm: p.complete([{"role": "user", "content": "x" * 5000}], max_tokens=10)
        self.assertIn("context", str(cm.exception)); self.assertEqual(p.state, "ready", "an oversize prompt must not kill the model")
        self.assertTrue(p.complete([{"role": "user", "content": "ok"}], max_tokens=2) is not None)

    def test_adapter_translates_errors(self):
        p = self.start(); llm = nb.NeonLLM(p, max_tokens=4)
        self.assertTrue(llm.complete([{"role": "user", "content": "hi"}]))
        p.stop()
        with self.assertRaises(LLMError): llm.complete([{"role": "user", "content": "hi"}])

    def test_stop_is_clean_and_restartable(self):
        p = self.start(); pid = p._p.pid; p.stop(); self.assertEqual(p.state, "stopped")
        time.sleep(0.3)
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)


class FakeBinaries(Fixtures):
    def fake(self, name, body):
        p = script(os.path.join(self.d, name), USAGE + body); return p

    def run_fake(self, body, **kw):
        path = self.fake("neon_f%d" % abs(hash(body) % 9999), body)
        pr = nb.Process(path, os.path.join(self.models, "good.gguf")); pr.start(); self.addCleanup(pr.stop); return pr

    def test_hang_times_out_and_marks_error(self):
        pr = self.run_fake("sys.stderr.write('[Mode] Interactive stdin/stdout loop \u2014 ready.\\n'); sys.stderr.flush()\nimport time; sys.stdin.readline(); time.sleep(60)\n")
        self.assertTrue(pr.wait_ready(10))
        t = time.time()
        with self.assertRaises(nb.NeonError) as cm: pr.complete([{"role": "user", "content": "x"}], timeout=1.5)
        self.assertLess(time.time() - t, 6); self.assertIn("did not answer", str(cm.exception)); self.assertEqual(pr.state, "error")

    def test_error_lines_and_noise_are_handled(self):
        body = ("sys.stderr.write('[Mode] Interactive stdin/stdout loop \u2014 ready.\\n'); sys.stderr.flush()\n"
                "for line in sys.stdin:\n"
                "    n = json.loads(line)['max_tokens'] if False else 0\n"
                "    sys.stdout.write('stray log line\\n{\"token\":\"he\"}\\n{not json\\n{\"token\":\"llo\"}\\n{\"done\":true}\\n'); sys.stdout.flush()\n")
        pr = self.run_fake("import json\n" + body); pr.wait_ready(10)
        self.assertEqual(pr.complete([{"role": "user", "content": "x"}]), "hello")
        pr2 = self.run_fake("sys.stderr.write('[Mode] Interactive stdin/stdout loop \u2014 ready.\\n'); sys.stderr.flush()\nfor line in sys.stdin:\n    sys.stdout.write('{\"error\":\"bad JSON\"}\\n'); sys.stdout.flush()\n"); pr2.wait_ready(10)
        with self.assertRaises(nb.NeonError) as cm: pr2.complete([{"role": "user", "content": "x"}])
        self.assertIn("bad JSON", str(cm.exception)); self.assertEqual(pr2.state, "ready")

    def test_start_failures(self):
        pr = nb.Process(os.path.join(self.d, "nope_neon"), "x.gguf")
        with self.assertRaises(nb.NeonError): pr.start()
        self.assertEqual(pr.state, "error")


@unittest.skipUnless(HAVE_REAL, "no real NEON binary")
class ServerLifecycle(Fixtures):
    def test_sigterm_stops_the_server_and_its_neon_child(self):
        import urllib.request
        port = 8780 + os.getpid() % 150
        env = dict(os.environ, NEBULA_STUDIO_CONFIG=os.path.join(self.d, "s.json"), NEON_BIN=NEON_BIN)
        srv = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "server.py"), "--port", str(port), "--no-open",
                                "--neon-model", os.path.join(self.models, "good.gguf")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: srv.poll() is None and srv.kill())
        get = lambda p: json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=3))
        t = time.time(); state = None
        while time.time() - t < 30:
            try: state = (get("/api/neon")["process"] or {}).get("state")
            except OSError: pass
            if state == "ready": break
            time.sleep(0.2)
        self.assertEqual(state, "ready")
        kids = subprocess.run(["pgrep", "-P", str(srv.pid)], capture_output=True, text=True).stdout.split()
        self.assertEqual(len(kids), 1, "exactly one NEON child while running")
        srv.send_signal(signal.SIGTERM)
        self.assertEqual(srv.wait(timeout=10), 0, "SIGTERM is a clean exit")
        time.sleep(0.3)
        self.assertFalse(os.path.exists(f"/proc/{kids[0]}") and "Z" not in open(f"/proc/{kids[0]}/stat").read().split(")")[-1][:3], "NEON child must be gone")
        out = srv.stdout.read()
        self.assertIn("NEON runtime:", out); self.assertIn("brain:", out)
        srv.stdout.close(); srv.stderr.close()


class Trimming(unittest.TestCase):
    def test_fit_messages(self):
        sys_m = {"role": "system", "content": "s" * 100}
        hist = [m for i in range(6) for m in ({"role": "user", "content": "u" * 200}, {"role": "assistant", "content": "a" * 100})]
        msgs = [sys_m] + hist + [{"role": "user", "content": "now"}]
        out = nb.fit_messages(msgs, 700, 100)
        self.assertEqual(out[0], sys_m); self.assertEqual(out[-1]["content"], "now"); self.assertLess(len(out), len(msgs))
        self.assertEqual(nb.fit_messages(msgs, None, 100), msgs); self.assertEqual(nb.fit_messages(msgs, 100000, 100), msgs)
        with self.assertRaises(nb.NeonError): nb.fit_messages([sys_m, {"role": "user", "content": "u" * 5000}], 300, 50)


class ManagerAndBuild(Fixtures):
    def mgr(self):
        return nb.Manager(settings_path=os.path.join(self.d, "settings.json"))

    def test_settings_persist_and_validate(self):
        m = self.mgr(); m.add_dirs([self.models])
        self.assertIn(self.models, self.mgr().settings["model_dirs"], "survives a restart")
        with self.assertRaises(nb.NeonError): m.add_dirs(["/definitely/not/here"])
        with self.assertRaises(nb.NeonError): m.set_binary("/bin/ls")
        with open(os.path.join(self.d, "settings.json"), "w") as f: f.write("{broken")
        self.assertEqual(self.mgr().settings["threads"], 0, "corrupt settings fall back to defaults")

    @unittest.skipUnless(HAVE_REAL, "no real NEON binary")
    def test_manager_launch_flow_with_real_neon(self):
        m = self.mgr(); m.set_binary(NEON_BIN); m.add_dirs([self.models]); events = []; m.on_change = lambda: events.append(m.proc and m.proc.state)
        self.assertTrue(any(x["runnable"] for x in m.models))
        for bad in ("/etc/passwd", "relative.gguf", os.path.join(self.models, "notes.txt"), 123, None):
            with self.assertRaises(nb.NeonError, msg=str(bad)): m.launch(bad)
        with self.assertRaises(nb.NeonError): m.launch(os.path.join(self.models, "weights.nctr"))
        with self.assertRaises(nb.NeonError): m.launch(os.path.join(self.models, "good.gguf"), threads=9999)
        m.launch(os.path.join(self.models, "good.gguf"), threads=2)
        self.assertTrue(m.proc.wait_ready(60)); time.sleep(0.3)
        st = m.status(); self.assertEqual(st["process"]["state"], "ready"); self.assertEqual(st["process"]["threads"], 2); self.assertEqual(st["process"]["config"]["n_embd"], 32)
        self.assertIsNotNone(m.llm()); self.assertIn("ready", events)
        self.assertEqual(self.mgr().settings["last_model"], os.path.join(self.models, "good.gguf"))
        pr = m.probe(os.path.join(self.models, "good.gguf")); self.assertTrue(pr["ok"])
        m.stop(); self.assertIsNone(m.llm()); self.assertIsNone(m.status()["process"])

    @unittest.skipUnless(NEON_SRC and os.path.isfile(os.path.join(NEON_SRC, "NEON-3.cpp")) and nb.find_compiler(), "no NEON source / compiler")
    def test_build_from_source(self):
        work = os.path.join(self.d, "src"); shutil.copytree(NEON_SRC, work, ignore=shutil.ignore_patterns(".git", "neon_test", "neon"))
        m = self.mgr(); m.settings["model_dirs"] = [work]
        self.assertEqual(m.build(), work)
        with self.assertRaises(nb.NeonError): m.build()                       # second build while running
        t = time.time()
        while m.builder.state == "building" and time.time() - t < 400: time.sleep(1)
        print(f"\n    real NEON build: {m.builder.state} in {time.time()-t:.0f}s; command: {m.builder.log[0][:100]}")
        self.assertEqual(m.builder.state, "done", "\n".join(m.builder.log)[-800:])
        self.assertTrue(nb.verify_binary(m.builder.output)[0]); self.assertEqual(m.binary(), m.builder.output)
        r = nb.probe_model(m.builder.output, os.path.join(self.models, "good.gguf")); self.assertTrue(r["ok"], r)

    def test_build_failure_and_missing_pieces(self):
        broken = os.path.join(self.d, "broken"); os.makedirs(broken)
        with open(os.path.join(broken, "NEON-3.cpp"), "w") as f: f.write("int main( { nonsense")
        b = nb.Builder()
        if nb.find_compiler():
            b.start(broken)
            t = time.time()
            while b.state == "building" and time.time() - t < 60: time.sleep(0.2)
            self.assertEqual(b.state, "failed"); self.assertIn("exited", b.error); self.assertTrue(any("error" in l.lower() for l in b.log))
        with self.assertRaises(nb.NeonError): nb.Builder().start(os.path.join(self.d, "empty"))
        m = self.mgr(); m.settings["model_dirs"] = []
        old = nb.find_source; nb.find_source = lambda extra=(): None
        try:
            with self.assertRaises(nb.NeonError) as cm: m.build()
            self.assertIn("NEON-3.cpp", str(cm.exception))
        finally: nb.find_source = old
        oldc = nb.find_compiler; nb.find_compiler = lambda: None
        try:
            with self.assertRaises(nb.NeonError) as cm: nb.Builder().start(broken)
            self.assertIn("compiler", str(cm.exception))
        finally: nb.find_compiler = oldc


if __name__ == "__main__":
    unittest.main(verbosity=2)
