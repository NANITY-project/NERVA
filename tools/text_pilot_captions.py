#!/usr/bin/env python3
"""
text_pilot_captions.py - captions for the text-prior pilot (tools/text_prior_pilot.py). numpy only.

A window's measured goal (heading change `dyaw` rad, speed m/s - the numbers goal_features() reads off the pelvis; NO labels) is turned into a short English phrase.
Two vocabularies:
  TRAIN  words the prior is trained with (the phrases are drawn from several synonyms and ~half of them are corrupted: no spaces, no commas, typos, "??" ...)
  HELD   synonyms that never appear in training: the probes for "does a text prior understand a word it has not seen" (test 4)
Convention (same as NERVA): z forward, x = the character's LEFT, yaw grows towards the left: dyaw > 0 = a left turn.

This is a pilot. It checks that the MECHANISM learns text -> motion on the same ground truth the numeric goals use. The real captions come from describe_motion.py (an LM writing
from measured facts); everything here is deliberately small and templated.
"""
import os, random, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_translator"))
import gt_corrupt

SPEED_EDGES = [0.15, 0.8, 1.6, 2.5]                       # m/s: stand | slow | walk | fast walk / jog | run   (the same bands as goal_translator/check_stats.py)
SPEED_CLASSES = ["stand", "slow", "walk", "fast", "run"]
TURN_EDGES_DEG = [8.0, 25.0, 50.0]                        # |heading change| per window: straight | slight | normal | sharp
TURN_CLASSES = ["straight", "slight", "normal", "sharp"]

SPEED_TRAIN = {
    "stand": ["stand still", "stop", "stand", "halt", "stay put", "wait"],
    "slow": ["walk slowly", "slow walk", "walk slow", "walk calmly", "take small slow steps"],
    "walk": ["walk", "walk forward", "walk ahead", "keep walking", "go forward", "walk normally"],
    "fast": ["walk fast", "walk quickly", "walk briskly", "jog", "power walk", "hurry up"],
    "run": ["run", "run fast", "run forward", "start running", "run quickly"],
}
SPEED_HELD = {
    "stand": ["freeze", "hold still", "come to a standstill", "pause", "do nothing"],
    "slow": ["stroll", "amble", "dawdle", "saunter", "take it easy"],
    "walk": ["proceed", "head forward", "move along", "advance", "carry on"],
    "fast": ["hustle", "speed walk", "trot", "stride quickly", "hasten"],
    "run": ["sprint", "dash", "race", "bolt", "run flat out"],
}
TURN_TRAIN = ["turn {s}", "go {s}", "veer {s}", "bear {s}", "turn to the {s}"]
TURN_HELD = ["hang a {s}", "swing {s}", "curve {s}", "head {s}", "steer {s}"]
SIDE = {+1: "left", -1: "right"}
DIR_HELD = {+1: ["anticlockwise", "counterclockwise"], -1: ["clockwise"]}     # direction words that never appear in training (left = anticlockwise seen from above)
DIR_VERBS_HELD = ["turn {d}", "rotate {d}", "spin {d}"]
MAG = {"slight": ["a bit", "slightly", "a little"], "normal": [""], "sharp": ["sharply", "hard", "a lot"]}


def speed_class(v):
    return int(np.searchsorted(SPEED_EDGES, v, side="right"))


def turn_class(dyaw):
    return int(np.searchsorted(TURN_EDGES_DEG, abs(np.degrees(dyaw)), side="right"))


def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def turn_phrase(sign, tc, rng, held_words=False, held_dir=False):
    """sign +1 / -1, tc 1..3 (TURN_CLASSES index) -> 'turn left a bit' ..."""
    if held_dir:
        base = _pick(rng, DIR_VERBS_HELD).format(d=_pick(rng, DIR_HELD[sign]))
    else:
        base = _pick(rng, TURN_HELD if held_words else TURN_TRAIN).format(s=SIDE[sign])
    mag = _pick(rng, MAG[TURN_CLASSES[tc] if TURN_CLASSES[tc] in MAG else "normal"])
    return (base + " " + mag).strip()


def speed_phrase(sc, rng, held_words=False):
    return _pick(rng, (SPEED_HELD if held_words else SPEED_TRAIN)[SPEED_CLASSES[sc]])


def caption(sc, tc, sign, rng, held_speed=False, held_turn=False, held_dir=False):
    """One phrase for a window of speed class `sc`, turn class `tc` (0 = straight) and turn side `sign` (+1 left, -1 right)."""
    sp = speed_phrase(sc, rng, held_speed)
    if tc == 0:
        if sc == 0 or rng.random() < 0.8:
            return sp
        return sp + " straight ahead"
    tp = turn_phrase(sign, tc, rng, held_turn, held_dir)
    if sc == 0:
        return tp if rng.random() < 0.6 else f"{sp} and {tp}"
    r = rng.random()
    if r < 0.15:
        return tp
    return _pick(rng, ["{sp} and {tp}", "{tp} and {sp}", "{sp} then {tp}"]).format(sp=sp, tp=tp)


def keys(goals):
    """goals [N,4] (dyaw, dx, dz, speed) -> int array [N,3] of (speed class, turn class, sign)."""
    g = np.asarray(goals, np.float64)
    sc = np.searchsorted(SPEED_EDGES, g[:, 3], side="right")
    tc = np.searchsorted(TURN_EDGES_DEG, np.abs(np.degrees(g[:, 0])), side="right")
    sg = np.where(g[:, 0] >= 0, 1, -1)
    sg = np.where(tc == 0, 0, sg)
    return np.stack([sc, tc, sg], 1).astype(np.int64)


def build(goals, V=4, pool=96, p_corrupt=0.5, seed=0):
    """-> (strings: list of unique captions, sid [N,V] int32 = which caption each window uses in each of V variants).
    Captions depend only on (speed class, turn class, side), so a pool of `pool` phrases is drawn per key and windows pick from it at random: diverse, but only a few thousand
    distinct strings, which keeps a frozen sentence encoder cheap."""
    rng = random.Random(seed); K = keys(goals); N = len(K)
    uniq, inv = np.unique(K, axis=0, return_inverse=True); inv = inv.reshape(-1)
    strings, index, pools = [], {}, []
    for (sc, tc, sg) in uniq:
        ids = []
        for _ in range(pool):
            c = caption(int(sc), int(tc), int(sg), rng)
            if rng.random() < p_corrupt:
                c = gt_corrupt.corrupt(c, rng)
            if c not in index:
                index[c] = len(strings); strings.append(c)
            ids.append(index[c])
        pools.append(np.array(ids, np.int32))
    pick = np.random.RandomState(seed).randint(0, pool, size=(N, V))
    sid = np.zeros((N, V), np.int32)
    for k, p in enumerate(pools):
        m = inv == k
        sid[m] = p[pick[m]]
    return strings, sid


# ------------------------------------------------------------------------------------------------------------------------------ probes: what to ask, and when the motion matches
SPEED_OK = {"stand": (0.0, 0.35), "slow": (0.15, 1.15), "walk": (0.8, 1.9), "fast": (1.3, 3.0), "run": (2.0, 99.0)}


def probes(seed=0, include_run=True):
    """-> {kind: [dict(text, sign, speed)]}; kind in_vocab | raw | unseen_speed | unseen_turn | unseen_both | unseen_dir. sign +1 left, -1 right, 0 straight."""
    rng = random.Random(seed); out = {k: [] for k in ("in_vocab", "raw", "unseen_speed", "unseen_turn", "unseen_both", "unseen_dir")}
    classes = [c for c in SPEED_CLASSES if include_run or c != "run"]
    for sc, name in enumerate(SPEED_CLASSES):
        if name not in classes:
            continue
        for sign in (0, +1, -1):
            if name == "stand" and sign != 0:
                continue                                                                  # a turn while standing is ambiguous (in place or not): not probed
            tc = 0 if sign == 0 else 2
            for _ in range(2):
                p = dict(sign=sign, speed=name)
                out["in_vocab"].append(dict(p, text=caption(sc, tc, sign, rng)))
                t = caption(sc, tc, sign, rng)
                t = gt_corrupt.corrupt(t, rng, p_typo=0.1, p_unsafe=0.0, p_nospace=0.5, p_case=0.3, p_punct=1.0, p_nocomma=1.0)
                if rng.random() < 0.5:
                    t = t.rstrip("?!.") + "??"
                out["raw"].append(dict(p, text=t))
                out["unseen_speed"].append(dict(p, text=caption(sc, tc, sign, rng, held_speed=True)))
                if sign != 0:
                    out["unseen_turn"].append(dict(p, text=caption(sc, tc, sign, rng, held_turn=True)))
                    out["unseen_both"].append(dict(p, text=caption(sc, tc, sign, rng, held_speed=True, held_turn=True)))
                    out["unseen_dir"].append(dict(p, text=caption(sc, tc, sign, rng, held_dir=True)))
    return out


def window_ok(sign, speed_name, dyaw_deg, speed):
    """per window: (turn matches, speed matches) as bool arrays."""
    d = np.asarray(dyaw_deg, np.float64); v = np.asarray(speed, np.float64)
    t = (d >= 10.0) if sign > 0 else (d <= -10.0) if sign < 0 else (np.abs(d) <= 15.0)
    lo, hi = SPEED_OK[speed_name]
    return t, (v >= lo) & (v < hi)


def summarize(results):
    """results: list of dict(sign, speed, dyaw_deg [n], speed_ms [n]) -> {'turn_ok','speed_ok','both_ok','n'} (shares over windows)."""
    tt, ss, bb, n = 0, 0, 0, 0
    for r in results:
        t, s = window_ok(r["sign"], r["speed"], r["dyaw_deg"], r["speed_ms"])
        tt += int(t.sum()); ss += int(s.sum()); bb += int((t & s).sum()); n += len(t)
    n = max(n, 1)
    return dict(turn_ok=tt / n, speed_ok=ss / n, both_ok=bb / n, n=n)


if __name__ == "__main__":
    rng = random.Random(1)
    for sc, tc, sg in [(0, 0, 0), (2, 0, 0), (2, 2, 1), (3, 3, -1), (1, 1, 1), (4, 0, 0)]:
        print(f"{SPEED_CLASSES[sc]:5s} {TURN_CLASSES[tc]:8s} {sg:+d}: " + " | ".join(caption(sc, tc, sg, rng) for _ in range(4)))
    for kind, ps in probes(0).items():
        print(f"{kind:13s} {len(ps):3d}  e.g. " + " | ".join(repr(p['text']) for p in ps[:3]))
