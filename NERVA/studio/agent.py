"""Agent layer: scene -> prompt -> model -> validated command lines -> world.

Designed for SMALL models (a fine-tuned 1.7B): short fixed-format prompts, a
one-command-per-line protocol, a tolerant parser (bullets, fences, prose are
stripped/ignored), rejection feedback fed into the next prompt so hallucinated
names get corrected, and an optional GBNF grammar that makes malformed output
impossible when the runtime supports grammar-constrained decoding.
"""
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from sim import ActionError, slug

VERBS = ("SAY", "WALK_TO", "LOOK_AT", "REACH", "GRAB", "DROP", "OFFER", "POINT", "WAVE", "WAIT", "RESET")
MAX_LINES = 6

PROTOCOL = """Commands (one per line):
SAY <text>                 speak out loud
WALK_TO <name>             walk to a prop or person
LOOK_AT <name>             turn your eyes to a prop or person
GRAB <left|right> <prop>   pick up a small prop (you walk over and crouch if needed)
OFFER <left|right> <name>  hold out what you carry to a person
DROP <left|right>          put down what you carry
REACH <left|right> <prop>  touch a prop
POINT <left|right> <name>  point at something
WAVE <left|right>          wave
WAIT <seconds>             pause
RESET <left|right>         relax that arm"""

EXAMPLE = """Example:
LOOK_AT bob
GRAB right cup
WALK_TO bob
OFFER right bob
SAY Here you go, Bob."""


class LLMError(RuntimeError):
    pass


# --- prompt ------------------------------------------------------------------
def describe_scene(snap, cid):
    me = next(c for c in snap["chars"] if c["id"] == cid)
    mx, mz = me["pos"][0], me["pos"][2]
    lines = [f"You are at x={mx:.1f} z={mz:.1f}."]
    h = me["holding"]
    lines.append("Hands: left " + (f"holding {h['left']}" if h["left"] else "empty") +
                 ", right " + (f"holding {h['right']}" if h["right"] else "empty") + ".")
    props = []
    for p in sorted(snap["props"], key=lambda p: math.dist((mx, mz), (p["pos"][0], p["pos"][2]))):
        d = math.dist((mx, mz), (p["pos"][0], p["pos"][2]))
        kind = "fixed" if "fixed" in p["tags"] else ("small" if max(p["size"]) <= 0.35 else "large")
        held = " (someone holds it)" if p["held"] else ""
        props.append(f"{p['name']} ({d:.1f} m, {kind}{held})")
    lines.append("Props: " + (", ".join(props) if props else "none") + ".")
    others = []
    for c in snap["chars"]:
        if c["id"] != cid:
            d = math.dist((mx, mz), (c["pos"][0], c["pos"][2]))
            said = f', says "{c["speech"]}"' if c["speech"] else ""
            others.append(f"{c['id']} ({d:.1f} m{said})")
    lines.append("People: " + (", ".join(others) if others else "nobody else") + ".")
    return "\n".join(lines)


def build_messages(snap, cid, events, instruction, history):
    me = next(c for c in snap["chars"] if c["id"] == cid)
    system = (f"You are {me['name']}. {me['persona'] or 'You are a character in a 3D scene.'}\n"
              "You act by writing command lines. Output ONLY command lines, one per line, "
              f"at most {MAX_LINES}. No explanations. Use only names from the scene.\n\n{PROTOCOL}\n\n{EXAMPLE}")
    user = ["SCENE", describe_scene(snap, cid)]
    if events:
        user += ["", "RECENT"] + [f"- {e}" for e in events[-8:]]
    user += ["", "INSTRUCTION", instruction.strip() if instruction else "Continue naturally, in character."]
    msgs = [{"role": "system", "content": system}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "\n".join(user)})
    return msgs


# --- reply parsing ---------------------------------------------------------------
_BULLET = re.compile(r"^\s*(?:[-*>\u2022]|\d+[.)])\s*")


def parse_reply(text, max_lines=MAX_LINES):
    """Model text -> (command_lines, ignored_lines). Never raises."""
    lines, ignored = [], []
    for raw in str(text).replace("\r", "").split("\n"):
        s = _BULLET.sub("", raw).strip().strip("`").strip()
        if not s or s.startswith("```"):
            continue
        verb, _, rest = s.partition(" ")
        if verb.upper() in VERBS:
            lines.append(f"{verb.upper()} {rest}".strip())
        else:
            ignored.append(s[:80])
        if len(lines) >= max_lines:
            break
    return lines, ignored


def build_grammar(prop_ids, char_ids):
    """GBNF grammar for llama.cpp-style grammar-constrained decoding."""
    q = lambda names: " | ".join(f'"{n}"' for n in names) or '"none"'
    targets = list(prop_ids) + list(char_ids)
    return "\n".join([
        'root   ::= line{1,%d}' % MAX_LINES,
        'line   ::= cmd "\\n"',
        'cmd    ::= say | walk | look | grab | drop | offer | reach | point | wave | wait | reset',
        'say    ::= "SAY " text',
        'walk   ::= "WALK_TO " target',
        'look   ::= "LOOK_AT " target',
        'grab   ::= "GRAB " hand " " prop',
        'drop   ::= "DROP " hand',
        'offer  ::= "OFFER " hand " " person',
        'reach  ::= "REACH " hand " " prop',
        'point  ::= "POINT " hand " " target',
        'wave   ::= "WAVE " hand',
        'wait   ::= "WAIT " num',
        'reset  ::= "RESET " hand',
        'hand   ::= "left" | "right"',
        f'prop   ::= {q(prop_ids)}',
        f'person ::= {q(char_ids)}',
        f'target ::= {q(targets)}',
        'num    ::= [0-9] ("." [0-9])?',
        'text   ::= [^\\n"]{1,160}',
    ]) + "\n"


# --- model clients -------------------------------------------------------------------
class OpenAIChat:
    """Any OpenAI-compatible /v1/chat/completions server (llama.cpp, vLLM, a bridge...)."""

    def __init__(self, url, model="", api_key="", timeout=60.0, use_grammar=False):
        self.url, self.model, self.api_key = url.rstrip("/"), model, api_key
        self.timeout, self.use_grammar = timeout, use_grammar

    def complete(self, messages, grammar=None):
        body = {"model": self.model, "messages": messages, "temperature": 0.6, "max_tokens": 160, "stream": False}
        if grammar and self.use_grammar:
            body["grammar"] = grammar
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from None


class MockLLM:
    """Deterministic rule-based stand-in so the studio works with no model attached."""
    name = "mock"

    def complete(self, messages, grammar=None):
        user = messages[-1]["content"]
        instr = user.split("INSTRUCTION\n", 1)[-1].strip().lower()
        props = re.findall(r"(\w+) \([\d.]+ m, (?:small|large|fixed)", user)
        people = re.findall(r"(\w+) \([\d.]+ m(?:,|\))", user.split("People:", 1)[-1].split("\n")[0])
        small = re.findall(r"(\w+) \([\d.]+ m, small", user)
        mention = lambda names: next((n for n in names if n in instr.replace(" ", "_") or n in instr), None)
        prop, person = mention(props), mention(people)
        out = []
        m = re.search(r'(?:say|tell \w+)\s+"?([^"]+)"?$', instr)
        if re.search(r"\b(give|hand|pass|bring)\b", instr) and prop and person:
            out += [f"GRAB right {prop}", f"WALK_TO {person}", f"OFFER right {person}", f"SAY Here you go, {person.title()}."]
        elif re.search(r"\b(pick up|grab|take|get)\b", instr) and prop:
            out += [f"GRAB right {prop}"]
        elif re.search(r"\b(put down|drop|release)\b", instr):
            out += ["DROP right"]
        elif re.search(r"\bwave\b", instr):
            out += [f"LOOK_AT {person}"] * bool(person) + ["WAVE right"]
        elif re.search(r"\bpoint\b", instr) and (prop or person):
            out += [f"POINT right {prop or person}"]
        elif re.search(r"\b(look at|watch|face)\b", instr) and (prop or person):
            out += [f"LOOK_AT {prop or person}"]
        elif re.search(r"\b(go to|walk to|walk over|approach|come)\b", instr) and (prop or person):
            out += [f"WALK_TO {prop or person}"]
        elif m:
            out += [f"SAY {m.group(1).strip()}"]
        elif "hello" in user.lower().split("recent", 1)[-1] and people:
            out += [f"LOOK_AT {people[0]}", f"SAY Hello, {people[0].title()}."]
        elif people:
            out += [f"LOOK_AT {people[0]}", "WAIT 2"]
        return "\n".join(out) or "WAIT 1"


def make_llm(cfg):
    if cfg.get("llm_url"):
        return OpenAIChat(cfg["llm_url"], cfg.get("model", ""), cfg.get("api_key", ""),
                          use_grammar=bool(cfg.get("use_grammar")))
    return MockLLM()


# --- orchestration ---------------------------------------------------------------------
class AgentManager:
    """Per-character think loop. Never blocks the sim: LLM calls run on worker threads."""

    def __init__(self, world, llm=None, interval=8.0):
        self.world, self.llm, self.interval = world, llm or MockLLM(), interval
        self.state = {}               # cid -> dict(thinking, pending, next, history)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    def _st(self, cid):
        return self.state.setdefault(cid, {"thinking": False, "pending": None, "next": 0.0,
                                           "history": deque(maxlen=3)})

    def instruct(self, cid, text):
        with self._lock:
            self._st(cid)["pending"] = text

    def set_llm(self, llm):
        self.llm = llm

    def start(self):
        if self._thread:
            return
        def loop():
            while not self._stop:
                self.tick()
                time.sleep(0.25)
        self._thread = threading.Thread(target=loop, daemon=True, name="nebula-agents")
        self._thread.start()

    def close(self):
        self._stop = True

    def tick(self):
        snap = self.world.snapshot
        now = time.monotonic()
        for c in snap.get("chars", []):
            cid = c["id"]
            with self._lock:
                st = self._st(cid)
                busy = c["busy"] is not None or c["queued"] > 0
                wants = st["pending"] is not None or (c["agent"] and now >= st["next"])
                if st["thinking"] or busy or not wants:
                    continue
                st["thinking"] = True
                instruction, st["pending"] = st["pending"], None
            threading.Thread(target=self._think, args=(cid, instruction), daemon=True).start()

    def _think(self, cid, instruction):
        st = self._st(cid)
        try:
            events = self.world.call(_take_events, cid)
            snap = self.world.snapshot
            if not any(c["id"] == cid for c in snap["chars"]):
                return
            msgs = build_messages(snap, cid, events, instruction, st["history"])
            grammar = build_grammar([p["name"] for p in snap["props"]], [c["id"] for c in snap["chars"]])
            reply = self.llm.complete(msgs, grammar)
            lines, ignored = parse_reply(reply)
            st["history"].append((msgs[-1]["content"], "\n".join(lines) or "WAIT 1"))
            self.world.call(_apply_reply, cid, lines, ignored, instruction)
        except (LLMError, ActionError, KeyError, StopIteration) as e:
            try:
                self.world.call(lambda w, m: w._log("error", m, cid), f"agent error: {e}")
            except Exception:
                pass
        except Exception as e:           # a bug must not kill the worker silently
            try:
                self.world.call(lambda w, m: w._log("error", m, cid), f"agent crashed: {type(e).__name__}: {e}")
            except Exception:
                pass
        finally:
            with self._lock:
                st["thinking"] = False
                st["next"] = time.monotonic() + self.interval


def _take_events(w, cid):
    c = w.chars.get(cid)
    if not c:
        return []
    ev, c.events = list(c.events), []
    return ev


def _apply_reply(w, cid, lines, ignored, instruction):
    c = w.chars.get(cid)
    if not c:
        return
    for ln in lines:
        try:
            w.enqueue(cid, ln, source="agent")
        except ActionError as e:
            c.note(f"{ln} rejected: {e}")
            w._log("error", f"{c.name}: {ln} rejected: {e}", cid)
    for ig in ignored[:2]:
        c.note(f"ignored non-command output: {ig!r}")
    if not lines:
        w._log("error", f"{c.name}: model produced no valid command", cid)
