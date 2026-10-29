#!/usr/bin/env python3
"""
tests/test_lang.py - UI/lang.py: the prompt -> action schedule layer. Standard library only (no torch, no engine, no model weights).

  python tests/test_lang.py

  1 parsing      tolerant of case / bullets / punctuation / 'then'; unknown names are dropped, never guessed; seconds clamped; step and total caps
  2 prompts      the few-shot examples only use actions the VAE really has; ChatML has no doubled <|im_start|> when BOS already is one
  3 NEON         NeonLM against a stand-in 'neon' (a script speaking the --interactive JSON-lines protocol): tokens joined, stops at <|im_end|>, refuses a non-nanity GGUF-style arch
  4 http         HttpLM against a local OpenAI-style server: body shape, Authorization header, HTTP errors become readable messages
  5 planner      model -> retry at temperature 0 -> keyword fallback, and the 'source' field tells the truth about which one produced the plan
"""
import json, os, stat, sys, tempfile, threading, http.server
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "UI"))
import lang

FAILS = []
ACTS = ["walk", "run", "sit", "stand", "pickup", "gesture", "jump", "stretch", "get_up"]


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1
print("[parsing]")
P = lambda t: [(s["action"], s["seconds"]) for s in lang.parse_schedule(t, ACTS)]
check(P("walk 4\npickup 3\ngesture 2") == [("walk", 4.0), ("pickup", 3.0), ("gesture", 2.0)], "one step per line")
check(P("1. Walk 4s\n- PICKUP 2.5 seconds") == [("walk", 4.0), ("pickup", 2.5)], "numbering, bullets, case, units")
check(P("walk 3, then run 2") == [("walk", 3.0), ("run", 2.0)], "'then' separates steps")
check(P("fly 3\nwalk 2\nteleport") == [("walk", 2.0)], "unknown names are dropped, not mapped to something close")
check(P("walk") == [("walk", lang.DEFAULT_SECONDS)], "missing seconds -> the default")
check(P("walk 999")[0][1] == lang.MAX_SECONDS and P("walk 0.01")[0][1] == lang.MIN_SECONDS, "seconds are clamped")
check(P("get-up 2") == [("get_up", 2.0)], "'-' and '_' are the same")
check(P("jumping 2") == [("jump", 2.0)] and P("stret 2") == [("stretch", 2.0)], "a clear prefix or a one-letter slip is accepted")
check(P("sat 2") == [], "a different word that merely looks similar ('sat') is not turned into an action")
check(P("") == [] and P("I cannot do that.") == [], "empty / refusal text -> no steps")
check(len(lang.parse_schedule("\n".join(["walk 1"] * 40), ACTS)) == lang.MAX_STEPS, "at most MAX_STEPS steps")
long = lang.parse_schedule("\n".join(["walk 30"] * 12), ACTS)
check(abs(sum(s["seconds"] for s in long) - lang.MAX_TOTAL_SECONDS) < 1.0, f"total length capped at {lang.MAX_TOTAL_SECONDS:.0f} s (got {sum(s['seconds'] for s in long):.1f})")
k = lang.keyword_schedule("please walk over and then pickup the pen, wave, run", ACTS)
check([s["action"] for s in k] == ["walk", "pickup", "run"], "keyword fallback: action names in order of appearance ('wave' is not an action here)")
check(lang.to_windows(6.4, 32, 1 / 30) == 6 and lang.to_windows(0.1, 32, 1 / 30) == 1, "seconds -> windows (stride frames each, at least one)")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2
print("[prompts]")
ex = lang.examples(ACTS)
check(ex and all(a in ACTS for _, rep in ex for a in [l.split()[0] for l in rep.splitlines()]), "few-shot examples use only actions the VAE has")
only = lang.examples(["wiggle", "spin"])
check(len(only) == 1 and only[0][1].startswith("wiggle"), "with none of the usual names, the example is built from the model's own first action")
sysm, turns = lang.build_messages("walk then sit", ACTS)
check(all(a in sysm for a in ACTS) and turns[-1] == ("user", "walk then sit"), "the system prompt lists the allowed actions; the request is the last turn")
c1, c2 = lang.chatml("s", [("user", "hi")], True), lang.chatml("s", [("user", "hi")], False)
check(not c1.startswith("<|im_start|>") and c2.startswith("<|im_start|>system") and c1.endswith("<|im_start|>assistant\n"), "ChatML: BOS that already is <|im_start|> is not doubled; an open assistant turn at the end")
check(lang.plain("s", [("user", "hi")]).endswith("Plan:\n"), "plain template ends in an open plan")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3
print("[NEON stand-in]")
tmp = tempfile.mkdtemp(prefix="lang_test_")
fake = os.path.join(tmp, "neon")
open(fake, "w").write('''#!/usr/bin/env python3
import sys, json, time
sys.stderr.write("[Interactive] ready\\n"); sys.stderr.flush()
for line in sys.stdin:
    req = json.loads(line); p = req["prompt"]
    last = p.split("<|im_start|>user\\n")[-1]                      # only the final request decides the answer (the system prompt and examples are in `p` too)
    # answers by what the prompt asks, and records what it received
    reply = "walk 4\\npickup 3<|im_end|>\\nIGNORED" if last.startswith("pick up the pen") else ("ok<|im_end|>" if last.startswith("ok?") else "nonsense words<|im_end|>")
    open(sys.argv[0] + ".last", "w").write(p)
    for ch in [reply[i:i + 3] for i in range(0, len(reply), 3)]:
        print(json.dumps({"token": ch}), flush=True); time.sleep(0.02)      # slow enough that tokens after <|im_end|> are still in flight when the next request is sent
    print(json.dumps({"done": True}), flush=True)
''')
os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
fake_gguf = os.path.join(tmp, "m.gguf"); open(fake_gguf, "wb").write(b"GGUF")        # gguf_info() cannot read it -> {} -> no architecture check, ChatML assumed
lm = lang.NeonLM(fake, fake_gguf)
check(lm.template == "chatml" and lm.kind == "neon", "no readable metadata -> ChatML assumed")
out = lm.complete("sys", [("user", "pick up the pen")])
check(out.strip().splitlines()[:2] == ["walk 4", "pickup 3"], f"tokens are joined and cut at <|im_end|> (got {out!r})")
last = open(fake + ".last").read()
check(last.startswith("system\n") and last.endswith("<|im_start|>assistant\n"), "the raw ChatML prompt (not NEON's {messages} form) was sent")
res = lang.plan("pick up the pen", ACTS, lm)
check(res["source"] == "model" and [s["action"] for s in res["schedule"]] == ["walk", "pickup"], f"plan() through the stand-in: {res['source']}")
back = [lang.plan("pick up the pen", ACTS, lm)["source"] for _ in range(5)]
check(back == ["model"] * 5, f"five requests in a row: the unread tail of one reply never leaks into the next ({back})")
res = lang.plan("do the thing", ACTS, lm)
check(res["source"] == "none" and not res["schedule"] and "allowed" in res["error"], "a model that answers nonsense and a prompt with no action name -> no plan, and an error that lists the allowed actions")
res = lang.plan("please run", ACTS, lm)
check(res["source"] == "keyword" and res["schedule"][0]["action"] == "run" and res["error"], "nonsense from the model but a clear word in the prompt -> keyword fallback, labelled as such")
lm.close()
try:
    lang.NeonLM("/nonexistent/neon", fake_gguf); check(False, "a missing binary is reported")
except Exception as e:
    check(True, f"a missing binary is reported ({type(e).__name__})")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 4
print("[HTTP endpoint]")
SEEN = {}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.update(path=self.path, auth=self.headers.get("Authorization"), body=body)
        if body.get("model") == "broken":
            self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
        text = "jump 2\nstretch 3" if "jump" in body["messages"][-1]["content"] else "no idea"
        out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/v1"
h = lang.HttpLM(base, "any-model", "sk-test")
res = lang.plan("jump and stretch", ACTS, h)
check(res["source"] == "model" and [s["action"] for s in res["schedule"]] == ["jump", "stretch"], "plan() through an OpenAI-style endpoint")
check(SEEN["path"] == "/v1/chat/completions" and SEEN["auth"] == "Bearer sk-test" and SEEN["body"]["model"] == "any-model" and SEEN["body"]["messages"][0]["role"] == "system", "path, bearer key, model name and message roles")
SEEN.clear(); lang.HttpLM(base).complete("s", [("user", "jump")])
check(SEEN["auth"] is None and "model" not in SEEN["body"], "no key / no model name -> neither is sent")
try:
    lang.HttpLM(base, "broken").complete("s", [("user", "x")]); check(False, "HTTP 500 is an error")
except RuntimeError as e:
    check("500" in str(e) and "boom" in str(e), f"HTTP 500 becomes a readable message ({str(e)[:60]})")
res = lang.plan("go jump", ACTS, lang.HttpLM(base, "broken"))
check(res["source"] == "keyword" and res["schedule"][0]["action"] == "jump" and "500" in res["error"], "a broken model falls back to keywords and says why")
srv.shutdown()
dead = lang.plan("walk", ACTS, lang.HttpLM("http://127.0.0.1:1/v1"))
check(dead["source"] == "keyword" and dead["error"], "an unreachable endpoint also falls back, with the connection error attached")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 5
print("[planner edge cases]")
check(lang.plan("", ACTS)["error"] == "empty request", "empty prompt")
check("load the VAE" in lang.plan("walk", [])["error"], "no actions known (VAE not loaded)")


class Flaky:
    name = "flaky"; calls = []

    def complete(self, system, turns, max_tokens=64, temperature=0.2, timeout=0):
        Flaky.calls.append(temperature)
        return "huh?" if temperature > 0 else "sit 5"


res = lang.plan("get comfortable", ACTS, Flaky())
check(res["source"] == "retry" and Flaky.calls == [0.2, 0.0] and res["schedule"][0]["action"] == "sit", "first try fails, second try at temperature 0 succeeds -> source 'retry'")
print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED"); sys.exit(1)
print("all language-layer checks passed")
