#!/usr/bin/env python3
"""
build_dataset.py - the Goal Translator's training data (spec section 8). Per source clip:

  1 MEASURE   the clip's goals are read off its own pelvis path, exactly as the prior's goals are (numpy port of train_cvqvae.goal_features: no labels, the label is exact)
  2 SEGMENT   the per-window goal sequence is cut into runs (turning left / right / straight / stopped / faster) and each run snapped to the model's bins
  3 DESCRIBE  an exact template description of each run: the character's own left/right, order, rough durations
  4 REWRITE   messy human phrasings. Two ways in:
                --rewriter rules    built-in, offline, deterministic: synonym pools + slang + noise (diversity is LIMITED to what is written below: use it to bootstrap and test)
                --rewrites FILE     the output of an external language model run ONCE, offline, on `rewrite_requests.jsonl` (written by --stage requests); see the license note
  5 VERIFY    a rewrite is kept only if a deterministic check agrees with the original on what the goals can express: order + direction of turns, rough size, rough duration
  6 UNSPEC    some descriptions are cut short or made vague on purpose; the windows they no longer describe are labelled UNSPEC (how "idk move around" is learned)

Held-out splits are by SOURCE CLIP (hash of the clip id), never by sentence.

INPUT (one of)
  --clips clips.npz     NERVA tools/gen_dataset.py output: actions [F,30] (all clips concatenated), offsets [N+1], dt. Channels 0, 2, 3 = pelvis x, z, yaw.
  --synthetic N         N random goal sequences without any clip. Teaches the TEXT -> goals mapping only (no motion realism): for smoke tests and a first bootstrap.

LICENSE NOTE (spec section 8): the sentences from step 4 are OUTPUTS of whatever model rewrote them. If you release the dataset, use a rewriter whose license allows
using its outputs for training and redistributing them. The built-in `rules` rewriter is this repository's own code (AGPLv3 with the rest of it). Check the exact terms of
every motion source you actually use (CMU, 100STYLE, ACCAD) before releasing anything derived from them.

  python build_dataset.py --synthetic 3000 --out data/gt --stats stats.json --timing 64,48,0.0333
  python build_dataset.py --clips ../NERVA/data/clips.npz --out data/gt --stats prior_stats.json
  python build_dataset.py --stage requests --clips ... --out data/gt          # then run your LM over data/gt/rewrite_requests.jsonl
  python build_dataset.py --clips ... --out data/gt --rewrites data/gt/rewrites.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import gt_conventions as C
import stage_c as SC

# ------------------------------------------------------------------------------------------------------------------------------------------------ 1. MEASURE
def window_goals_from_pelvis(px, pz, yaw, W: int, stride: int, dt: float) -> np.ndarray:
    """One clip's pelvis path [F] each -> goals [n_windows, 4] (dyaw, dx, dz, speed), windows of W frames every `stride` frames (full windows only).
    Same numbers as train_cvqvae.goal_features(canonicalize(window)): frame 0 at the origin facing +z, x = the character's left."""
    px, pz = np.asarray(px, np.float64), np.asarray(pz, np.float64)
    yaw = np.unwrap(np.asarray(yaw, np.float64))
    F = len(px)
    out = []
    for s in range(0, F - W + 1, stride):
        x, z, y = px[s:s + W], pz[s:s + W], yaw[s:s + W]
        c, sn = math.cos(y[0]), math.sin(y[0])
        dx, dz = x - x[0], z - z[0]
        xc, zc = dx * c - dz * sn, dx * sn + dz * c
        step = np.hypot(np.diff(xc), np.diff(zc))
        out.append([y[-1] - y[0], xc[-1], zc[-1], step.sum() / ((W - 1) * dt)])
    return np.asarray(out, np.float32).reshape(-1, 4)


def goal_stats_from(goals: np.ndarray) -> dict:
    """Same as train_cvqvae.goal_stats (turn_max / speed_cruise / speed_max). Prefer the PRIOR's own stats (--stats): the translator may only ask for what the prior's data contains."""
    mv = goals[goals[:, 3] > 0.15, 3]
    return dict(turn_max=float(max(np.percentile(np.abs(goals[:, 0]), 95), 0.3)), speed_cruise=float(np.median(mv)) if len(mv) else 1.0,
                speed_max=float(np.percentile(mv, 95)) if len(mv) else 1.5)


# ------------------------------------------------------------------------------------------------------------------------------------------------ 2. SEGMENT
DEADBAND = 0.12          # |turn| below this fraction of turn_max counts as "straight"
STAND_SPEED = 0.15       # m/s


def classify(turn: float, speed: float, st: C.Stats) -> Tuple[int, int, int]:
    """-> (sign, magnitude class 0 none / 1 small / 2 medium / 3 large, speed class 0 stand / 1 slow / 2 walk / 3 fast / 4 run)"""
    f = abs(turn) / st.turn_max
    sign = 0 if f < DEADBAND else (1 if turn > 0 else -1)
    mag = 0 if sign == 0 else 1 if f < 0.3 else 2 if f < 0.65 else 3
    if speed < STAND_SPEED and sign == 0:
        sp = 0                                                        # "stand still" only when there is no turn either: turning in place is described as a slow turn
    elif speed < 0.75 * st.speed_cruise:
        sp = 1
    elif speed < 1.25 * st.speed_cruise:
        sp = 2
    elif speed < 0.9 * st.speed_max:
        sp = 3
    else:
        sp = 4
    return sign, mag, sp


@dataclass
class Run:
    n: int                      # windows
    sign: int                   # +1 left, -1 right, 0 straight
    mag: int
    sp: int
    turn: float                 # mean turn per window (rad)
    speed: float                # mean speed (m/s)

    def win(self, st: C.Stats) -> C.Win:
        return C.spec_win(self.turn, self.speed, st)

    @property
    def total_deg(self) -> float:
        return abs(math.degrees(self.turn)) * self.n


def segment(goals: np.ndarray, st: C.Stats, min_run: int = 2) -> List[Run]:
    keys = [classify(float(g[0]), float(g[3]), st) for g in goals]
    turns, speeds = [float(g[0]) for g in goals], [float(g[3]) for g in goals]
    # runs of identical (sign, mag, speed class)
    runs = []                                                           # [key, [indices]]
    for i, k in enumerate(keys):
        if runs and runs[-1][0] == k:
            runs[-1][1].append(i)
        else:
            runs.append([k, [i]])
    # absorb short runs into the longer neighbour (a single flickering window is not a run)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (k, idx) in enumerate(runs):
            if len(idx) < min_run:
                nb = [m for m in (j - 1, j + 1) if 0 <= m < len(runs)]
                m = max(nb, key=lambda m: len(runs[m][1]))
                runs[m][1] = sorted(runs[m][1] + idx)
                del runs[j]
                changed = True
                break
        k = 0
        while k + 1 < len(runs):                                      # re-merge neighbours with the same key
            if runs[k][0] == runs[k + 1][0]:
                runs[k][1] += runs[k + 1][1]
                del runs[k + 1]
            else:
                k += 1
    out = []
    for key, idx in runs:
        t = float(np.mean([turns[i] for i in idx])); v = float(np.mean([speeds[i] for i in idx]))
        sign, mag, sp = classify(t, v, st)                              # classes of the MEAN, so the label and the words agree
        out.append(Run(len(idx), sign, mag, sp, t, v))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------ 3. DESCRIBE
@dataclass
class Desc:
    """What a description says about one run - also what the verifier checks a rewrite against."""
    sign: int
    deg: float                  # total turn of the run (degrees, positive)
    seconds: float
    mag: int
    sp: int
    n: int
    vague: bool = False


def run_seconds(r: Run, timing: C.Timing) -> float:
    return r.n * timing.step_seconds


def _secs(x: float) -> str:
    x = max(1.0, x)
    return str(int(round(x))) if abs(x - round(x)) < 0.25 or x >= 4 else f"{x:.1f}"


SPEED_WORD = {0: "", 1: "slowly", 2: "", 3: "fast", 4: "running"}


def template(r: Run, timing: C.Timing) -> str:
    sec = _secs(run_seconds(r, timing))
    side = "left" if r.sign > 0 else "right"
    spd = {1: " slowly", 3: " fast", 4: ""}.get(r.sp, "")
    verb = "run" if r.sp == 4 else "walk"
    if r.sp == 0:
        return f"stand still for {sec} seconds"
    if r.sign == 0:
        return f"{verb} straight{spd} for {sec} seconds"
    if r.total_deg >= 60 and r.n <= 3:
        return f"turn {side} {int(round(r.total_deg / 5.0) * 5)} degrees{spd} over {sec} seconds"
    size = {1: "slightly ", 2: "", 3: "sharply "}[r.mag]
    return f"{verb} {size}{side}{spd} for {sec} seconds"


def describe(runs: List[Run], timing: C.Timing) -> Tuple[str, List[Desc]]:
    descs = [Desc(r.sign, r.total_deg if r.sign else 0.0, run_seconds(r, timing), r.mag, r.sp, r.n) for r in runs]
    return ", then ".join(template(r, timing) for r in runs), descs


# ------------------------------------------------------------------------------------------------------------------------------------------------ 6. UNSPEC variants
VAGUE_PHRASES = ["idk move around", "move around", "wander around", "whatever", "do something", "just roam", "somehow move around", "idk walk around", "go anywhere", "random stuff"]
UNSUPPORTED_PHRASES = ["wave your hand", "wave ur hand", "jump", "sit down", "dance", "clap", "wave", "raise your arms"]


@dataclass
class Variant:
    kind: str                   # full | cut | vague | cut_vague
    described: List[int]        # run indices the text mentions
    labels: List[Optional[C.Win]]   # per window of the clip
    parts: List[Optional[int]]  # sequence of text parts: run index, or None for a vague phrase / unsupported phrase slot


def make_variants(runs: List[Run], st: C.Stats, rng: random.Random) -> Variant:
    k = len(runs)
    kinds = ["full"] * 5 + (["cut"] * 2 if k > 1 else []) + ["vague"] * 2 + (["cut_vague"] if k > 1 else [])
    kind = rng.choice(kinds)
    parts: List[Optional[int]] = []
    labels: List[Optional[C.Win]] = []
    m = k
    if kind in ("cut", "cut_vague"):
        m = rng.randint(1, k - 1)
    vague_idx = set()
    if kind in ("vague", "cut_vague") and m > 0:
        vague_idx = {rng.randrange(m)}
    described = []
    for j, r in enumerate(runs):
        if j >= m:
            labels += [C.UNSPEC_WIN] * r.n                              # not mentioned at all
        elif j in vague_idx:
            labels += [C.UNSPEC_WIN] * r.n
            parts.append(None)
        else:
            labels += [r.win(st)] * r.n
            parts.append(j)
            described.append(j)
    if kind in ("cut", "cut_vague") and rng.random() < 0.6:
        parts.append(None)                                              # trailing "idk move around": the cut windows are explicitly vague
    return Variant(kind, described, labels, parts)


# ------------------------------------------------------------------------------------------------------------------------------------------------ 4. REWRITE (rules)
def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def rewrite_run_rules(r: Run, timing: C.Timing, rng: random.Random) -> str:
    """A messy phrase for one run, written from the DESCRIPTOR (not from the template string). Every phrase here is one the verifier (stage_c.parse_events) can read back; the
    point of a language-model rewriter is to go beyond this list."""
    sec = run_seconds(r, timing)
    secs = _secs(sec)
    dur = _pick(rng, [f"for {secs} seconds", f"for {secs}s", f"for like {secs} sec", f"for about {secs} seconds", "", "", ""])
    if r.sp == 0:
        return _pick(rng, ["stand still", "stop", "just stand there", "wait", "freeze", "pause"]) + (" " + dur if dur else "")
    spd = {1: _pick(rng, ["slowly", "slow", ""]), 2: "", 3: _pick(rng, ["fast", "quickly", "briskly"]), 4: ""}[r.sp]
    verb = _pick(rng, ["run", "sprint", "jog"]) if r.sp == 4 else _pick(rng, ["walk", "go", "move", "head", "keep walking", "stroll"])
    if r.sign == 0:
        d = _pick(rng, ["straight", "forward", "ahead", "straight ahead"])
        return " ".join(x for x in (verb, d, spd, dur) if x)
    side = "left" if r.sign > 0 else "right"
    if r.total_deg >= 60 and r.n <= 3:
        deg = int(round(r.total_deg / 5.0) * 5)
        v = _pick(rng, ["turn", "turn", "veer", "swing", "hang a"]) if False else _pick(rng, ["turn", "veer"])
        amt = _pick(rng, [f"{deg} degrees", f"{deg} deg", f"{deg}°"])
        base = _pick(rng, [f"{v} {side} {amt}", f"{v} {amt} {side}", f"{v} {side} by {amt}"])
        return " ".join(x for x in (base, spd, dur) if x)
    size = {1: _pick(rng, ["slightly", "a little", "a bit", "gently"]), 2: "", 3: _pick(rng, ["sharply", "hard"])}[r.mag]
    form = rng.randrange(3)
    if form == 0:
        s = f"{verb} {side}"
        return " ".join(x for x in (s, size, spd, dur) if x)
    if form == 1:
        s = f"{verb} {size} {side}".replace("  ", " ")
        return " ".join(x for x in (s, spd, dur) if x)
    return " ".join(x for x in (f"veer {side}", size, spd, dur) if x)


SEQ_JOIN = [" then ", " and then ", ", then ", " and ", ", ", " after that ", " next ", " then "]


def rewrite_rules(runs: List[Run], var: Variant, timing: C.Timing, rng: random.Random) -> str:
    bits = []
    for p in var.parts:
        if p is None:
            bits.append(_pick(rng, VAGUE_PHRASES))
        else:
            bits.append(rewrite_run_rules(runs[p], timing, rng))
    if rng.random() < 0.25:                                           # an unsupported gesture somewhere: its windows are not described by any run, the text is just ignored
        bits.insert(rng.randrange(len(bits) + 1), _pick(rng, UNSUPPORTED_PHRASES))
    s = bits[0]
    for b in bits[1:]:
        s += _pick(rng, SEQ_JOIN) + b
    return s


def template_text(runs: List[Run], var: Variant, timing: C.Timing, rng: random.Random) -> str:
    bits = [_pick(rng, VAGUE_PHRASES) if p is None else template(runs[p], timing) for p in var.parts]
    return ", then ".join(bits)


def add_noise(text: str, rng: random.Random, p_space=0.2, p_typo=0.25) -> str:
    """Dataset-time surface noise (training applies more of it on the fly, train_goal_translator.corrupt)."""
    if rng.random() < p_space:
        text = text.replace(" ", "")
    if rng.random() < p_typo and len(text) > 6:
        i = rng.randrange(1, len(text) - 1)
        op = rng.randrange(3)
        text = text[:i] + text[i + 1:] if op == 0 else text[:i] + text[i] + text[i:] if op == 1 else text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if rng.random() < 0.3:
        text = text.capitalize()
    if rng.random() < 0.15:
        text += rng.choice(["?", "!", "...", " pls"])
    return text


# ------------------------------------------------------------------------------------------------------------------------------------------------ 5. VERIFY
def verify(text: str, runs: List[Run], var: Variant, timing: C.Timing, st: C.Stats) -> Tuple[bool, str]:
    """Deterministic agreement check between a rewrite and the original on the things the goals can express. A rewrite that stage C cannot read back (unknown slang) is REJECTED:
    safe by construction, at the price of diversity (the language-model rewriter should be asked to stay inside plain phrasings for the direction words)."""
    parsed = SC.parse_events(text)
    want = [runs[j] for j in var.described if runs[j].sign != 0]
    got = [e for e in parsed.events if e.sign != 0]
    if [r.sign for r in want] != [e.sign for e in got]:
        return False, f"turn order/direction differs: want {[r.sign for r in want]}, text says {[e.sign for e in got]}"
    for r, e in zip(want, got):
        if e.deg is not None and abs(e.deg - r.total_deg) > max(15.0, 0.35 * r.total_deg):
            return False, f"turn size differs: {r.total_deg:.0f} deg vs text {e.deg:.0f}"
        if e.seconds is not None and abs(e.seconds - run_seconds(r, timing)) > max(1.5, 0.5 * run_seconds(r, timing)):
            return False, f"duration differs: {run_seconds(r, timing):.1f}s vs text {e.seconds:.1f}"
        if e.size == "small" and r.mag == 3:
            return False, "text says slightly, the run turns sharply"
        if e.size == "large" and r.mag == 1:
            return False, "text says sharply, the run turns slightly"
        if e.speed == "run" and r.sp <= 1 or e.speed == "slow" and r.sp >= 3:
            return False, "speed word contradicts the run"
    stops_text = sum(1 for e in parsed.events if e.kind == "stop")
    stops_want = sum(1 for j in var.described if runs[j].sp == 0)
    if stops_text != stops_want:
        return False, f"stand/stop count differs: {stops_want} vs {stops_text}"
    return True, "ok"


# ------------------------------------------------------------------------------------------------------------------------------------------------ sources
def clip_goal_sequences(path: str, timing: C.Timing):
    z = np.load(path, allow_pickle=False)
    A, off = z["actions"], z["offsets"]
    dt = float(z["dt"]) if "dt" in z.files else timing.dt
    if abs(dt - timing.dt) > 1e-6:
        print(f"note: clips.npz says dt={dt:.5f}, --timing says {timing.dt:.5f}: using the clips' dt for measuring", file=sys.stderr)
    for i in range(len(off) - 1):
        a = A[off[i]:off[i + 1]]
        g = window_goals_from_pelvis(a[:, 0], a[:, 2], a[:, 3], timing.W, timing.stride, dt)
        if len(g):
            yield f"clip{i:05d}", g


def synthetic_goal_sequences(n: int, st: C.Stats, rng: random.Random):
    for i in range(n):
        gs = []
        for _ in range(rng.randint(1, 4)):
            nw = rng.randint(2, 6)
            sign = rng.choice([-1, 0, 1, 1, -1])
            f = rng.choice([0.2, 0.45, 0.45, 0.75]) if sign else 0.0
            turn = sign * f * st.turn_max
            speed = 0.0 if rng.random() < 0.12 else rng.choice([0.5, 1.0, 1.0, 1.0, 1.4, 1.75]) * st.speed_cruise
            speed = min(speed, st.speed_max)
            for _ in range(nw):
                gs.append([turn + rng.gauss(0, 0.02), 0.0, 0.0, max(0.0, speed + rng.gauss(0, 0.03))])
        yield f"synth{i:05d}", np.asarray(gs, np.float32)


def split_of(clip_id: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()[:8], 16) % 100
    return "val" if h < 10 else "test" if h < 20 else "train"


def row(clip_id, split, text, labels, kind, variant, extra=None):
    d = dict(clip=clip_id, split=split, text=text, windows=[None if w.unspecified else [w.turn_bin, w.speed_bin] for w in labels], kind=kind, variant=variant)
    d.update(extra or {})
    return d


REWRITE_PROMPT = ("Rewrite the movement instruction below the way a person would really type it to a game character. Give {n} different versions. Rules: keep the ORDER of the moves, "
                  "keep every LEFT and RIGHT exactly (they are the character's own left and right), keep the rough size of each turn and the rough duration. You may use slang, "
                  "different orderings of detail, vagueness, typos, missing spaces and 'idk'. Do not add moves that are not in the instruction. Return JSON: {{\"rewrites\": [\"...\"]}}.\n\n"
                  "Instruction: {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips"); src.add_argument("--synthetic", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior", help="the trained prior (prior_best.pt): takes goal_stats, stride and dt from it (needs torch). Overrides --stats and the stride / dt of --timing")
    ap.add_argument("--vae", help="the trained VAE (best.pt): takes the window length W from it (needs torch). Overrides W of --timing")
    ap.add_argument("--stats", help="the prior's goal_stats as JSON (file or string): turn_max, speed_cruise, speed_max. Default: measured from the clips (warns)")
    ap.add_argument("--timing", default="", help="W,stride,dt of the prior (default 64,48,1/30)")
    ap.add_argument("--stage", choices=["all", "requests"], default="all", help="requests: only write rewrite_requests.jsonl for an external language model")
    ap.add_argument("--rewrites", help="JSONL {id, rewrites:[...]} from an external language model run on the requests")
    ap.add_argument("--rewriter", choices=["rules", "none"], default="rules")
    ap.add_argument("--per-clip", type=int, default=6, help="descriptions (variants) per clip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tm = C.Timing(*[t(v) for t, v in zip((int, int, float), a.timing.split(","))]) if a.timing else C.Timing()
    os.makedirs(a.out, exist_ok=True)

    st = None
    if a.prior or a.vae:
        import torch
        if a.prior:
            pk = torch.load(a.prior, map_location="cpu", weights_only=False)
            if not pk.get("goal_stats"):
                sys.exit(f"{a.prior} was trained without goal numbers (no goal_stats): there is nothing to steer")
            st = C.Stats.from_dict(pk["goal_stats"])
            tm = C.Timing(tm.W, int(pk["stride"]), float(pk.get("dt", tm.dt)))
        if a.vae:
            tm = C.Timing(int(torch.load(a.vae, map_location="cpu", weights_only=False)["cfg"]["window"]), tm.stride, tm.dt)
        print(f"timing from the checkpoints: W {tm.W}, stride {tm.stride}, dt {tm.dt:.5f}" + (f"; goal_stats turn_max {st.turn_max:.3f} rad, cruise {st.speed_cruise:.2f}, max {st.speed_max:.2f} m/s" if st else ""))
    if a.stats and not a.prior:
        st = C.Stats.from_dict(json.loads(open(a.stats).read() if not a.stats.lstrip().startswith("{") else a.stats))
    if a.clips:
        seqs = list(clip_goal_sequences(a.clips, tm))
    else:
        st = st or C.Stats()
        seqs = list(synthetic_goal_sequences(a.synthetic, st, rng))
    if not seqs:
        sys.exit("no clip is long enough for one window")
    if st is None:
        st = C.Stats.from_dict(goal_stats_from(np.concatenate([g for _, g in seqs])))
        print("WARNING: --stats not given: turn_max / speed_cruise / speed_max were measured from these clips. Pass the PRIOR's goal_stats (info.goal in NEON) so the translator "
              "cannot ask for more than the prior's training windows contain.", file=sys.stderr)

    ext: Dict[str, List[str]] = {}
    if a.rewrites:
        for line in open(a.rewrites, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                ext[d["id"]] = [str(x) for x in d.get("rewrites", [])]

    rows, requests, rejected, tried = [], [], 0, 0
    reasons: Dict[str, int] = {}
    for clip_id, goals in seqs:
        runs = segment(goals, st)
        split = split_of(clip_id, a.seed)
        for v in range(a.per_clip):
            var = make_variants(runs, st, rng)
            vid = f"{clip_id}/v{v}"
            if not var.parts:
                continue
            tt = template_text(runs, var, tm, rng)
            rows.append(row(clip_id, split, tt, var.labels, "template", var.kind, dict(id=vid)))           # exact template, kept as is (no verification needed: it IS the spec)
            if a.stage == "requests":
                requests.append(dict(id=vid, template=tt, prompt=REWRITE_PROMPT.format(n=4, t=tt)))
                continue
            cands: List[Tuple[str, str]] = []
            if a.rewriter == "rules":
                cands += [(add_noise(rewrite_rules(runs, var, tm, rng), rng), "rules") for _ in range(3)]
            cands += [(t, "lm") for t in ext.get(vid, [])]
            for text, kind in cands:
                tried += 1
                ok, why = verify(text, runs, var, tm, st)
                if ok:
                    rows.append(row(clip_id, split, text, var.labels, kind, var.kind, dict(id=vid)))
                else:
                    rejected += 1
                    key = why.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1

    if a.stage == "requests":
        with open(os.path.join(a.out, "rewrite_requests.jsonl"), "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(requests)} rewrite requests -> {a.out}/rewrite_requests.jsonl  (run your language model once, offline; see the LICENSE NOTE in --help)")
        return
    counts = {}
    for sp in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == sp]
        counts[sp] = dict(rows=len(part), clips=len({r["clip"] for r in part}))
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(layout_version=C.LAYOUT_VERSION, stats=st.to_dict(), timing=dict(W=tm.W, stride=tm.stride, dt=tm.dt), n_turn_bins=C.N_TURN_BINS, n_speed_bins=C.N_SPEED_BINS,
                turn_positive_is_left=C.TURN_POSITIVE_IS_LEFT, source=("clips:" + os.path.basename(a.clips)) if a.clips else f"synthetic:{a.synthetic}", counts=counts,
                verify=dict(candidates=tried, rejected=rejected, reasons=reasons), seed=a.seed)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v['rows']} rows / {v['clips']} clips" for k, v in counts.items()) + f"   verifier rejected {rejected}/{tried} rewrites {reasons}")


if __name__ == "__main__":
    main()
