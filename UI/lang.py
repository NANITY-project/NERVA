#!/usr/bin/env python3
"""
lang.py - the language layer of the NERVA UI: a prompt in, an ACTION SCHEDULE out.

    "walk to the table, then pick something up and wave"   ->   [walk 4 s] [pickup 3 s] [gesture 3 s]

The motion prior knows nothing about language. It is conditioned on an action label (one of the names in the VAE, read at run time, never hard-coded here), so
all a language model has to do is pick labels and durations. That is a small job, and any model can do it: the model is a SLOT, not a dependency.

Backends (pick one in the UI, or --lm on the server):
  neon      a GGUF run by a NEON binary:   neon --model X.gguf --interactive      (JSON lines; NEON loads only general.architecture = "nanity", see convert_hint())
  http      any OpenAI-compatible endpoint (llama.cpp server, Ollama, vLLM, a hosted API with a key): POST {base}/chat/completions
  keyword   no model: the names of the actions found in your text, in order. Honest about being a fallback (source = "keyword").

The reply format asked of every model is deliberately tiny: one step per line, "<action> <seconds>". Parsing is tolerant (case, punctuation, a bullet, a near-miss
spelling) but never invents an action: a name that is not in the VAE's list is dropped, and if nothing survives the caller gets an error instead of a silent default.

Standard library only (the gguf package is used for metadata when present).
"""
import difflib, json, os, queue, re, subprocess, threading, time, urllib.error, urllib.request

DEFAULT_SECONDS = 3.0
MIN_SECONDS, MAX_SECONDS, MAX_STEPS, MAX_TOTAL_SECONDS = 0.5, 30.0, 12, 120.0

SYSTEM = ("You turn a request into a short list of body actions for a character.\n"
          "Allowed actions: {actions}.\n"
          "Reply with one step per line, written as: <action> <seconds>\n"
          "Use only the allowed action names, in the order they happen. No other words.\n")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- prompt
def examples(actions):
    """Few-shot (request, reply) pairs built ONLY from actions that exist, so a model is never shown a name the prior does not have."""
    have = set(actions)
    pool = [("go for a walk", [("walk", 5)]),
            ("walk over there and pick something up", [("walk", 4), ("pickup", 3)]),
            ("say hello then sit down", [("gesture", 3), ("sit", 4)]),
            ("jump twice and stretch", [("jump", 2), ("stretch", 4)]),
            ("run, then stop and stand still", [("run", 4), ("stand", 3)]),
            ("dance", [("dance", 8)])]
    out = []
    for req, steps in pool:
        if all(a in have for a, _ in steps):                 # a partial example would teach the model a plan that does not match the request
            out.append((req, "\n".join(f"{a} {s}" for a, s in steps)))
    if not out and actions:                      # an action list with none of the usual names: show the model its own first names
        a = actions[0]
        out.append((f"do {a}", f"{a} {int(DEFAULT_SECONDS)}"))
    return out[:4]


def chatml(system, turns, bos_is_im_start=True):
    """ChatML text ending in an open assistant turn. NEON encodes with add_bos=True; if BOS already IS <|im_start|> (SmolLM2) a second one must not be written."""
    s = "" if bos_is_im_start else "<|im_start|>"
    s += "system\n" + system.rstrip("\n") + "\n<|im_end|>\n"
    for role, text in turns:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def plain(system, turns):
    """For base models / unknown templates: an instruction followed by Request/Plan pairs."""
    s = system.rstrip("\n") + "\n\n"
    for role, text in turns:
        s += ("Request: " if role == "user" else "Plan:\n") + text + "\n\n"
    return s + "Plan:\n"


def build_messages(user_text, actions):
    turns = []
    for req, rep in examples(actions):
        turns += [("user", req), ("assistant", rep)]
    turns.append(("user", user_text.strip()))
    return SYSTEM.format(actions=", ".join(actions)), turns


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- parsing
_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", re.I)


def match_action(word, actions):
    """exact (case-insensitive, '-' and ' ' count as '_') -> unique prefix -> close spelling. None if nothing is close: an unknown name is never guessed into a known one."""
    w = re.sub(r"[^a-z0-9_]+", "_", word.lower()).strip("_")
    if not w:
        return None
    low = {a.lower(): a for a in actions}
    if w in low:
        return low[w]
    pre = [a for k, a in low.items() if k.startswith(w) or (len(k) >= 4 and w.startswith(k))]
    if len(set(pre)) == 1 and len(w) >= 3:
        return pre[0]
    m = difflib.get_close_matches(w, list(low), n=1, cutoff=0.8)
    return low[m[0]] if m else None


def parse_schedule(text, actions, default_seconds=DEFAULT_SECONDS):
    """Model reply -> [{"action", "seconds"}]. Takes every line (or comma / 'then' separated chunk) that starts with an allowed action; seconds default when missing."""
    steps = []
    for line in re.split(r"[\n;]|(?:,\s*then\s*)|(?:\bthen\b)", text or "", flags=re.I):
        line = line.strip().lstrip("-*•0123456789.) \t")
        if not line:
            continue
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", line)
        act = next((match_action(w, actions) for w in words[:2] if match_action(w, actions)), None)
        if not act:
            continue
        n = _NUM.search(line)
        sec = float(n.group(1)) if n else default_seconds
        steps.append(dict(action=act, seconds=min(MAX_SECONDS, max(MIN_SECONDS, sec))))
        if len(steps) >= MAX_STEPS:
            break
    total = sum(s["seconds"] for s in steps)
    if total > MAX_TOTAL_SECONDS:
        k = MAX_TOTAL_SECONDS / total
        for s in steps:
            s["seconds"] = max(MIN_SECONDS, round(s["seconds"] * k, 2))
    return steps


def keyword_schedule(user_text, actions, default_seconds=DEFAULT_SECONDS):
    """No model: the action names (and exact near-spellings) that appear in the text, in order of appearance."""
    steps = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", user_text):
        a = match_action(w, actions) if len(w) >= 3 else None
        if a and (not steps or steps[-1]["action"] != a):
            steps.append(dict(action=a, seconds=default_seconds))
    return steps[:MAX_STEPS]


def to_windows(seconds, stride, dt, first_window_extra=0):
    """How many windows give about `seconds` of motion: every window after the first adds `stride` frames."""
    return max(1, int(round(seconds / (stride * dt))))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- GGUF facts
def gguf_info(path):
    """{'arch', 'bos_is_im_start', 'chat_template'} from the file's metadata, or {} if unreadable / the gguf package is missing."""
    try:
        import gguf
        r = gguf.GGUFReader(path)

        def val(key):
            f = r.fields.get(key)
            if f is None:
                return None
            v = f.parts[f.data[0]]
            return bytes(v).decode("utf-8", "replace") if f.types and f.types[0] == gguf.GGUFValueType.STRING else v
        out = dict(arch=val("general.architecture"), chat_template=val("tokenizer.chat_template"))
        bos, toks = val("tokenizer.ggml.bos_token_id"), r.fields.get("tokenizer.ggml.tokens")
        if bos is not None and toks is not None:
            i = int(bos[0]) if hasattr(bos, "__len__") else int(bos)
            tf = toks
            out["bos_is_im_start"] = bytes(tf.parts[tf.data[i]]).decode("utf-8", "replace") == "<|im_start|>"
        return out
    except Exception:
        return {}


def convert_hint(path, info):
    """What to tell the person when a GGUF is not something NEON can load."""
    return (f"{os.path.basename(path)} declares architecture '{info.get('arch')}', and NEON loads only 'nanity'. Convert it first: "
            f"python3 nanity_convert.py --detect-only {path}   then   python3 nanity_convert.py {path} model-nanity.gguf --quant Q4_0"
            f"   (or run the model in llama.cpp / Ollama and use the http backend).")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- backends
class NeonLM:
    """neon --model X.gguf --interactive. Request: {"prompt": ..., "max_tokens", "temperature", "top_p"}; reply: {"token": ".."} lines then {"done": true}.
    Sends the raw prompt form (NEON's own {"messages"} template is not ChatML). template: chatml | plain | neon | auto."""

    kind = "neon"

    def __init__(self, neon_bin, gguf_path, template="auto", threads=0, ready_timeout=120):
        self.path, self.bin = gguf_path, neon_bin
        self.info = gguf_info(gguf_path)
        arch = self.info.get("arch")
        if arch and arch != "nanity":
            raise RuntimeError(convert_hint(gguf_path, self.info))
        self.template = template if template != "auto" else ("chatml" if ("im_start" in (self.info.get("chat_template") or "") or not self.info) else "plain")
        self.bos_is_im_start = self.info.get("bos_is_im_start", True)
        cmd = [neon_bin, "--model", gguf_path, "--interactive"] + (["--threads", str(threads)] if threads else [])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.err, self.out, self.lock = [], queue.Queue(), threading.Lock()
        threading.Thread(target=self._pump_err, daemon=True).start()
        threading.Thread(target=self._pump_out, daemon=True).start()
        self.name = f"{os.path.basename(gguf_path)} on NEON ({self.template})"
        time.sleep(0.2)
        if self.p.poll() is not None:
            raise RuntimeError("NEON exited: " + " | ".join(self.err[-4:]))

    def _pump_err(self):
        for raw in self.p.stderr:
            self.err.append(raw.decode("utf-8", "replace").rstrip())
            del self.err[:-40]

    def _pump_out(self):
        for raw in self.p.stdout:
            try:
                self.out.put(json.loads(raw.decode("utf-8", "surrogateescape")))
            except ValueError:
                pass
        self.out.put(None)

    def complete(self, system, turns, max_tokens=64, temperature=0.2, timeout=180):
        prompt = plain(system, turns) if self.template == "plain" else chatml(system, turns, self.bos_is_im_start)
        req = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9}
        with self.lock:
            while not self.out.empty():
                self.out.get_nowait()
            if self.p.poll() is not None:
                raise RuntimeError("NEON is not running: " + " | ".join(self.err[-4:]))
            self.p.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            self.p.stdin.flush()
            text, deadline, over = "", time.time() + timeout, False
            while True:
                try:
                    j = self.out.get(timeout=max(0.1, deadline - time.time()))
                except queue.Empty:
                    raise RuntimeError(f"the model gave no answer within {timeout} s")
                if j is None:
                    raise RuntimeError("NEON closed the pipe: " + " | ".join(self.err[-4:]))
                if "error" in j:
                    raise RuntimeError(str(j["error"]))
                if "token" in j and not over:
                    text += j["token"]
                    over = "<|im_end|>" in text or "\n\n" in text.strip("\n") or "Request:" in text      # the plan is over; later tokens are ignored
                if j.get("done"):                                     # ALWAYS read to this request's {"done"}: the protocol has no request ids, so a reply left
                    break                                             # unread would be taken for the start of the next one (NEON runs on to max_tokens or EOS)
        return text.split("<|im_end|>")[0].split("Request:")[0]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass


class HttpLM:
    """Any OpenAI-compatible chat endpoint. base like http://127.0.0.1:8080/v1 ; model may be empty for servers that host one model. key goes in the Authorization header."""

    kind = "http"

    def __init__(self, base, model="", key=""):
        self.base, self.model, self.key = base.rstrip("/"), model, key
        self.name = f"{model or 'endpoint'} at {self.base}"

    def complete(self, system, turns, max_tokens=64, temperature=0.2, timeout=120):
        msgs = [{"role": "system", "content": system}] + [{"role": r, "content": t} for r, t in turns]
        body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        if self.model:
            body["model"] = self.model
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + "/chat/completions", json.dumps(body).encode(), h), timeout=timeout) as r:
                j = json.loads(r.read())
            return j["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.base}: HTTP {e.code} {e.read()[:200].decode('utf-8', 'replace')}")
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            raise RuntimeError(f"{self.base}: {e}")

    def close(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- the planner
def plan(user_text, actions, lm=None, default_seconds=DEFAULT_SECONDS):
    """-> dict(schedule=[{action, seconds}], source='model'|'retry'|'keyword'|'none', raw=<model text>, model=<name>, error=<str or None>).
    With a model: ask at a low temperature, then once at 0 if nothing parsed; if it still fails (or the model is broken) the keyword fallback runs and the source says so."""
    user_text = (user_text or "").strip()
    if not user_text:
        return dict(schedule=[], source="none", raw="", model=getattr(lm, "name", None), error="empty request")
    if not actions:
        return dict(schedule=[], source="none", raw="", model=getattr(lm, "name", None), error="no actions known: load the VAE first")
    raw, err = "", None
    if lm is not None:
        system, turns = build_messages(user_text, actions)
        for source, temp in (("model", 0.2), ("retry", 0.0)):
            try:
                raw = lm.complete(system, turns, temperature=temp)
            except Exception as e:
                err = str(e)
                break
            steps = parse_schedule(raw, actions, default_seconds)
            if steps:
                return dict(schedule=steps, source=source, raw=raw, model=lm.name, error=None)
        err = err or "the model's reply held no allowed action name"
    steps = keyword_schedule(user_text, actions, default_seconds)
    if steps:
        return dict(schedule=steps, source="keyword", raw=raw, model=getattr(lm, "name", None), error=err)
    return dict(schedule=[], source="none", raw=raw, model=getattr(lm, "name", None),
                error=(err + "; " if err else "") + "no action could be read from the request (allowed: " + ", ".join(actions) + ")")


if __name__ == "__main__":                                  # python3 UI/lang.py "walk then wave"  --actions walk,run,gesture,pickup   (keyword fallback; no model)
    import argparse
    ap = argparse.ArgumentParser(description="parse / plan a request without a model (keyword fallback), or with --http BASE [--model M]")
    ap.add_argument("text"); ap.add_argument("--actions", required=True); ap.add_argument("--http", default=""); ap.add_argument("--model", default=""); ap.add_argument("--key", default="")
    a = ap.parse_args()
    print(json.dumps(plan(a.text, a.actions.split(","), HttpLM(a.http, a.model, a.key) if a.http else None), indent=1))
