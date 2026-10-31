#!/usr/bin/env python3
"""
stage_c.py - the bootstrap Goal Translator (spec section 7): the SAME interface as the learned model, no model.

    text  ->  Plan(windows=[Win | UNSPEC ...], ignored=[...])

  1. split     a deterministic dynamic program over general-English words separates run-together text ("Walkrightandthenleft" -> walk right and then left).
               It uses a word-frequency list (the optional `wordfreq` package if installed, otherwise the small embedded list below). The lexicon is plain English,
               not a motion vocabulary; the schema in step 2 is what knows about motion.
  2. schema    a few physical descriptions (how much turn, how fast, how long) are mapped to windows.
  3. never guess   anything it cannot place becomes UNSPEC; text about things the goals cannot express (the wave) is reported in Plan.ignored, never turned into a turn.

Every (text, goals) pair handled is appended to a log (--log / log_pair) - the first real prompts the learned module is evaluated on.
parse_events() is also what the dataset builder's verifier uses (build_dataset.py), so "what the rewrite says" and "what stage C reads" are the same function.

Standard library + numpy. Defaults marked (proposal) in the spec are the constants at the top.
"""
from __future__ import annotations
import argparse, json, math, re, sys, time
from dataclasses import dataclass, field
from typing import List, Optional

import gt_conventions as C

# ---------------------------------------------------------------------------------------------------------------------------------------------- schema constants (proposal)
SIZE_FRAC = {"small": 0.15, "default": 0.45, "large": 0.8}      # per-window turn as a fraction of turn_max when no degrees are given ("walk right", "veer left slightly")
TURN_VERB_DEG = 90.0                                            # "turn left" with no amount
DEG_PER_WINDOW_FRAC = 0.6                                       # a stated total turn is spread over windows of at most this fraction of turn_max
DEFAULT_WINDOWS = {"veer": 2, "straight": 3, "stop": 1, "vague": 2, "turn": 1}
LONG_WINDOWS = 4
SPEED_FRAC = {"slow": 0.5, "walk": 1.0, "fast": 1.4, "run": None}       # of speed_cruise; run = speed_max
TURN_IN_PLACE_SPEED = 0.5                                       # "turn left" with no walking verb: a slow turning gait, as a fraction of speed_cruise

# ---------------------------------------------------------------------------------------------------------------------------------------------- word lists
# Roughly frequency-ordered general English (most common first). Rank -> cost in the splitter. Not exhaustive on purpose: unknown strings are allowed at a high per-character cost.
_COMMON = """the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want
because any these give day most us is are was were been has had did does am being done made going went gone said says left right turn walk move run stop stand wait still slow slowly fast quickly quick keep keeps kept around straight forward ahead backward backwards
slightly little bit lot much more less very really quite sharp sharply hard gently gentle small big large long short while until for second seconds sec secs minute minutes degree degrees deg step steps then next finally later before again once twice
down here there where somewhere anywhere everywhere somehow whatever maybe perhaps idk dunno ok okay please pls plz kinda sorta gonna wanna gotta lemme let lets let's yes yeah nah hey hi hello thanks
u ur ya yo im ill id ive dont cant wont isnt arent wasnt thats whats hes shes theres
wave waving waved hand hands arm arms jump jumping hop sit sitting kneel crouch dance dancing clap point pointing pick grab reach nod bow kick punch throw lift salute head heading face facing look looking
start starting end ending begin beginning continue continuing again over under through across along toward towards away near far close closer further center middle side edge corner circle circles round
uturn quarter half full whole complete completely partly partially barely hardly almost nearly about roughly approximately around abouts
car house door table chair room floor wall wall tree road path street way line walk walking walks walked walker stroll strolling march marching proceed advance travel travelling wander wandering roam roaming explore exploring
jog jogging sprint sprinting dash rush hurry hurrying veer veering curve curving bend bending steer steering swerve swerving pivot pivoting rotate rotating spin spinning bear bearing
character guy man girl body person him her them it thing things stuff something anything nothing everything someone
a b c d e f g h j k l m n o p q r s t v w x y z""".split()

_RANK = {}
for _w in _COMMON:
    _RANK.setdefault(_w, len(_RANK) + 1)
_WORDFREQ = None


def _load_wordfreq():
    global _WORDFREQ
    if _WORDFREQ is None:
        try:
            import wordfreq                                           # optional: pip install wordfreq
            _WORDFREQ = wordfreq
        except Exception:
            _WORDFREQ = False
    return _WORDFREQ


def word_cost(w: str) -> Optional[float]:
    """Lower is better; None = not a word. Single letters other than a / i / u are expensive (they are how garbage gets 'explained')."""
    if w in _RANK:
        c = math.log(_RANK[w] + 25.0)
        if len(w) == 1 and w not in ("a", "i", "u"):
            c += 4.0
        return c
    wf = _load_wordfreq()
    if wf and len(w) > 1:
        z = wf.zipf_frequency(w, "en")
        if z >= 2.5:
            return math.log(25.0 + 1.0e4 * math.exp(-z))         # zipf 2.5..7 -> a similar scale to the embedded ranks
    return None


UNKNOWN_CHAR_COST = 7.0


def split_run(s: str) -> List[str]:
    """Minimum-cost segmentation of one lowercase alphabetic run into words (dynamic program). An unknown stretch costs UNKNOWN_CHAR_COST per character, so known words win."""
    n = len(s)
    if n == 0:
        return []
    best = [(0.0, -1, False)] + [(math.inf, -1, False)] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            piece = s[j:i]
            c = word_cost(piece)
            if c is not None:
                cand = best[j][0] + c
                if cand < best[i][0]:
                    best[i] = (cand, j, True)
        cand = best[i - 1][0] + UNKNOWN_CHAR_COST                    # one unknown character
        if cand < best[i][0]:
            best[i] = (cand, i - 1, False)
    out, i = [], n
    while i > 0:
        _, j, known = best[i]
        out.append((s[j:i], known))
        i = j
    out.reverse()
    merged: List[str] = []                                           # adjacent unknown characters are one unknown token
    prev_unknown = False
    for piece, known in out:
        if not known and prev_unknown:
            merged[-1] += piece
        else:
            merged.append(piece)
        prev_unknown = not known
    return merged


_TOK = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[,;.!?]")


def tokenize(text: str) -> List[str]:
    t = text.lower().replace("°", " deg ").replace("'", "")
    t = re.sub(r"(\d)\s*(?:s|sec)\b", r"\1 seconds", t)
    toks: List[str] = []
    for m in _TOK.finditer(t):
        piece = m.group(0)
        if piece in ",;.!?":
            toks.append(",")
        elif piece[0].isdigit() or piece in _RANK or len(piece) <= 2:
            toks.append(piece)
        else:
            toks += split_run(piece)
    return toks


# ---------------------------------------------------------------------------------------------------------------------------------------------- vocabulary of the SCHEMA
LEFT = {"left", "lefty", "leftward", "leftwards", "lft"}
RIGHT = {"right", "righty", "rightward", "rightwards", "rt"}
STRAIGHT = {"straight", "forward", "forwards", "ahead", "onward", "onwards"}
BACK = {"back", "backward", "backwards", "reverse"}
V_WALK = {"walk", "walking", "walks", "walked", "go", "going", "goes", "head", "heading", "move", "moving", "moves", "step", "stepping", "stroll", "strolling", "march", "marching",
          "proceed", "advance", "travel", "travelling", "keep", "keeps", "continue", "continuing"}
V_RUN = {"run", "running", "runs", "jog", "jogging", "sprint", "sprinting", "dash", "rush", "hurry", "hurrying"}
V_TURN = {"turn", "turning", "turns", "veer", "veering", "curve", "curving", "bend", "bending", "steer", "steering", "swerve", "swerving", "bear", "bearing", "pivot", "pivoting",
          "rotate", "rotating", "spin", "spinning"}
V_STOP = {"stop", "halt", "stand", "standing", "wait", "waiting", "pause", "freeze", "stay", "still", "idle", "rest"}
SLOW = {"slow", "slowly", "carefully", "leisurely", "casually"}
FAST = {"fast", "quickly", "quick", "briskly", "rapidly", "swiftly"}
SMALL = {"slightly", "slight", "little", "bit", "tiny", "gently", "gentle", "lightly", "barely", "subtly", "hardly"}
LARGE = {"sharply", "sharp", "hard", "fully", "completely", "tightly", "abruptly"}
VAGUE = {"idk", "dunno", "whatever", "somehow", "anywhere", "somewhere", "randomly", "random", "wander", "wandering", "roam", "roaming", "explore", "exploring", "maybe", "perhaps",
         "something", "anything", "freely", "around", "somewhat"}
UNSUPPORTED = {"wave", "waving", "waved", "hand", "hands", "arm", "arms", "jump", "jumping", "hop", "sit", "sitting", "kneel", "crouch", "dance", "dancing", "clap", "point",
               "pointing", "pick", "grab", "reach", "nod", "bow", "kick", "punch", "throw", "lift", "salute", "hello", "hi", "hey"}
SEQ = {"then", "next", "after", "afterwards", "afterward", "finally", "later", "lastly", "before", "and", "while"}
SEC_WORDS = {"second", "seconds", "sec", "secs"}
MIN_WORDS = {"minute", "minutes", "min", "mins"}
DEG_WORDS = {"degree", "degrees", "deg"}
LONG_WORDS = {"while", "long", "forever", "continuously", "keep", "keeps", "continue", "continuing"}
BRIEF_WORDS = {"briefly", "quick", "moment", "sec"}
NEUTRAL = {"the", "a", "an", "to", "of", "up", "your", "ur", "u", "you", "it", "its", "in", "on", "at", "for", "by", "with", "me", "my", "please", "pls", "plz", "ok", "okay", "kinda",
           "sorta", "just", "now", "so", "like", "is", "are", "be", "do", "dont", "lets", "let", "lemme", "gonna", "wanna", "can", "could", "would", "will", "should", "i", "we", "he",
           "she", "they", "character", "guy", "body", "person", "yes", "yeah", "thanks", "also", "again", "more", "much", "very", "really", "quite", "way", "side", "direction", "towards",
           "toward", "from", "that", "this", "there", "here", "out", "off", "over", "x", "times", "once", "twice"}


for _grp in (LEFT, RIGHT, STRAIGHT, BACK, V_WALK, V_RUN, V_TURN, V_STOP, SLOW, FAST, SMALL, LARGE, VAGUE, UNSUPPORTED, SEQ, SEC_WORDS, MIN_WORDS, DEG_WORDS, NEUTRAL):
    for _w in sorted(_grp):                                           # every word the schema reads is a known word to the splitter, so it is never cut into pieces
        _RANK.setdefault(_w, len(_RANK) + 1)


@dataclass
class Event:
    kind: str                       # "move" (walk/run, maybe veering) | "turn" (turn verb) | "stop" | "vague"
    sign: int = 0                   # +1 left, -1 right, 0 none (straight)
    deg: Optional[float] = None     # total turn stated, degrees (positive)
    seconds: Optional[float] = None
    size: Optional[str] = None      # "small" | "large" | None
    speed: str = "walk"             # "slow" | "walk" | "fast" | "run"
    raw: str = ""


@dataclass
class Parsed:
    tokens: List[str]
    events: List[Event]
    ignored: List[str]


def _chunks(tokens: List[str]) -> List[List[str]]:
    out, cur = [], []
    for t in tokens:
        if t == ",":
            if cur:
                out.append(cur)
                cur = []
        elif t in SEQ and cur:
            out.append(cur)
            cur = []
        elif t not in SEQ:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def parse_events(text: str) -> Parsed:
    toks = tokenize(text)
    events: List[Event] = []
    ignored: List[str] = []
    for ch in _chunks(toks):
        s = set(ch)
        unsup = [t for t in ch if t in UNSUPPORTED]
        sign = 1 if s & LEFT else -1 if s & RIGHT else 0
        if s & LEFT and s & RIGHT:                                    # "left or right": not a direction
            sign = 0
        deg = seconds = None
        for i, t in enumerate(ch):                                    # numbers: "90 degrees", "3 seconds", "half a minute"
            if t.replace(".", "", 1).isdigit():
                nxt = ch[i + 1] if i + 1 < len(ch) else ""
                v = float(t)
                if nxt in DEG_WORDS:
                    deg = v
                elif nxt in SEC_WORDS:
                    seconds = v
                elif nxt in MIN_WORDS:
                    seconds = v * 60.0
        if "quarter" in s and (s & V_TURN or sign):
            deg = deg or 90.0
        if "half" in s and (s & V_TURN) and not (s & MIN_WORDS):
            deg = deg or 180.0
        if ("uturn" in s) or ("u" in s and (s & V_TURN)) or ("around" in s and (s & V_TURN)):
            deg = deg or 180.0
        has_move = bool(s & (V_WALK | V_RUN)); has_turn = bool(s & V_TURN)
        speed = "run" if s & V_RUN else "fast" if s & FAST else "slow" if s & SLOW else "walk"
        size = "small" if s & SMALL else "large" if s & LARGE else None
        vague = bool(s & VAGUE) and not (sign or s & STRAIGHT) and not (has_turn and deg is not None)
        has_back = bool(s & BACK) and not (sign or s & STRAIGHT)
        if unsup and not (sign or has_move or has_turn or vague or s & V_STOP):
            ignored.append(" ".join(ch))
            continue
        if unsup:
            ignored.append(" ".join(unsup))
        if has_back:
            ignored.append(" ".join(ch) + " (moving backwards is not expressible)")
            continue
        if vague:
            events.append(Event("vague", raw=" ".join(ch)))
        elif sign:
            events.append(Event("turn" if has_turn and not has_move else "move", sign, deg, seconds, size, speed, " ".join(ch)))
        elif has_turn and deg is not None:
            events.append(Event("turn", 0, deg, seconds, size, speed, " ".join(ch)))        # "turn around": a turn with no side -> not expressible without a direction
        elif s & STRAIGHT or (has_move and not vague):
            events.append(Event("move", 0, None, seconds, size, speed, " ".join(ch)))
        elif s & V_STOP:
            events.append(Event("stop", 0, None, seconds, None, "walk", " ".join(ch)))
        elif has_turn:
            events.append(Event("vague", raw=" ".join(ch)))                                # a turn with no direction: do not guess one
        elif events and (size or s & (SLOW | FAST) or seconds is not None):               # modifiers only: they belong to the previous event
            ev = events[-1]
            ev.size = ev.size or size
            ev.seconds = ev.seconds if ev.seconds is not None else seconds
            if s & (SLOW | FAST):
                ev.speed = speed
        elif ch and not (set(ch) <= NEUTRAL):
            ignored.append(" ".join(ch))
    # a direction-only event after a verb event inherits nothing else: it is already a "move" with a sign
    return Parsed(toks, events, ignored)


# ---------------------------------------------------------------------------------------------------------------------------------------------- events -> windows
def _speed(ev: Event, st: C.Stats) -> float:
    if ev.kind == "stop":
        return 0.0
    if ev.speed == "run":
        return st.speed_max
    f = SPEED_FRAC[ev.speed]
    v = st.speed_cruise * f
    if ev.kind == "turn":
        v *= TURN_IN_PLACE_SPEED
    return min(v, st.speed_max)


def _count(ev: Event, default: int, timing: C.Timing) -> int:
    if ev.seconds is not None:
        return int(min(C.MAX_WINDOWS, max(1, round(ev.seconds / timing.step_seconds))))
    return default


def windows_for(ev: Event, st: C.Stats, timing: C.Timing, notes: List[str]) -> List[C.Win]:
    if ev.kind == "vague":
        return [C.UNSPEC_WIN] * DEFAULT_WINDOWS["vague"]
    speed = _speed(ev, st)
    if ev.kind == "stop":
        return [C.spec_win(0.0, 0.0, st)] * _count(ev, DEFAULT_WINDOWS["stop"], timing)
    if ev.sign == 0 and ev.deg is None:
        return [C.spec_win(0.0, speed, st)] * _count(ev, DEFAULT_WINDOWS["straight"], timing)
    if ev.sign == 0:
        notes.append(f"'{ev.raw}': a turn with no side (left / right) is not placed; the character's own left or right is never guessed")
        return [C.UNSPEC_WIN] * DEFAULT_WINDOWS["vague"]                                    # an amount with no side: do not guess a direction
    tmax = st.turn_max
    deg = ev.deg
    if deg is None and ev.kind == "turn":
        deg = TURN_VERB_DEG
    if deg is not None:
        total = math.radians(deg)
        n = _count(ev, 0, timing) or max(1, math.ceil(total / (DEG_PER_WINDOW_FRAC * tmax)))
        per = total / n
        if per > tmax + 1e-9:
            notes.append(f"'{ev.raw}': {deg:.0f} deg in {n} window(s) is more than the prior's turn_max ({math.degrees(tmax):.0f} deg/window): clamped")
            per = tmax
        return [C.spec_win(ev.sign * per, speed, st)] * n
    frac = SIZE_FRAC[ev.size or "default"]
    return [C.spec_win(ev.sign * frac * tmax, speed, st)] * _count(ev, DEFAULT_WINDOWS["veer"], timing)


def translate(text: str, st: Optional[C.Stats] = None, timing: Optional[C.Timing] = None) -> C.Plan:
    st = st or C.Stats()
    timing = timing or C.Timing()
    if not text.strip():
        raise C.PlanError("empty text")
    C.encode_input(text)                                                       # same length rule as the learned model: too long is an error, not a truncation
    parsed = parse_events(text)
    wins: List[C.Win] = []
    notes: List[str] = []
    for ev in parsed.events:
        wins += windows_for(ev, st, timing, notes)
    if len(wins) > C.MAX_WINDOWS:
        raise C.PlanError(f"plan needs {len(wins)} windows; the cap is {C.MAX_WINDOWS}")
    if not wins:
        notes.append("nothing in the text could be placed; no goals were produced")
    return C.Plan(wins, ignored=parsed.ignored, source="stage_c", notes=notes)


def log_pair(path: str, text: str, plan: C.Plan):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.time(), text=text, windows=[None if w.unspecified else [w.turn_bin, w.speed_bin] for w in plan.windows], ignored=plan.ignored,
                                source=plan.source), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("text", nargs="+")
    ap.add_argument("--stats", help="JSON file or string: turn_max, speed_cruise, speed_max (the prior's goal_stats)")
    ap.add_argument("--timing", default="", help="W,stride,dt of the prior, e.g. 64,48,0.0333")
    ap.add_argument("--log", help="append every (text, goals) pair to this JSONL file")
    a = ap.parse_args()
    st = C.Stats()
    if a.stats:
        st = C.Stats.from_dict(json.loads(open(a.stats).read() if not a.stats.lstrip().startswith("{") else a.stats))
    tm = C.Timing(*[t(v) for t, v in zip((int, int, float), a.timing.split(","))]) if a.timing else C.Timing()
    for text in a.text:
        plan = translate(text, st, tm)
        print(f"{text!r}\n  split: {' '.join(tokenize(text))}")
        for i, w in enumerate(plan.windows):
            print(f"  window {i}: {C.describe_window(w, st)}")
        for ig in plan.ignored:
            print(f"  ignored (not expressible by the goals): {ig}")
        for n in plan.notes:
            print(f"  note: {n}")
        if a.log:
            log_pair(a.log, text, plan)


if __name__ == "__main__":
    main()
