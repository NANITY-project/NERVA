#!/usr/bin/env python3
"""
describe_motion.py - give every motion clip free-form English captions, with NO predefined motion words anywhere.

  python tools/goal_translator/describe_motion.py --stage inspect  --clips data/clips.npz            # look at the segments + measured facts of a few clips
  python tools/goal_translator/describe_motion.py --stage requests --clips data/clips.npz --out data/captions
  python tools/goal_translator/describe_motion.py --stage run      --out data/captions --model <model>   # optional: calls the Anthropic API (needs ANTHROPIC_API_KEY)
  python tools/goal_translator/describe_motion.py --stage apply    --clips data/clips.npz --out data/captions   # parse responses, filter, write captions.jsonl

The idea
  clip -> SEGMENTS (data-driven: a new segment starts where the measured numbers change, no thresholds named after motions)
       -> FACTS per segment (plain kinematics in the CHARACTER'S OWN frame: how far, how fast, heading change, hand height, gaze, ...)
       -> an LM turns the facts into several phrasings in several registers (first-person narration, command, terse, casual)
       -> filter (a caption whose left/right contradicts the measurements is dropped) -> (text, clip, frame span) pairs.
  The word "walk" is never produced by this script: the LM chooses it (or "stroll", or "head over") from "1.3 m/s, 110 steps/min". The source's own names (CMU action
  names, generator style parameters) are never read.

Frame and sign convention (same as NERVA): z forward, x = the character's LEFT, yaw grows towards the left. Facts are measured in the pelvis frame, so
"left" in a caption always means the character's own left. The LM is told that, and apply() drops captions whose left/right contradicts the numbers.

Honest limits
  * The set of motions is whatever the clips contain. Captions can only be as fine as the measurements: two different motions with the same numbers get the same words.
  * The LM brings its own language habits; ask for variety (the prompt does) and look at the output before training on it (--stage inspect, then a few responses).
  * The left/right filter is a sanity check, not a verifier. Speed words ("brisk"), amounts and hand semantics are not checked.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, random, re, sys, time, urllib.request
import numpy as np

# The examples are about content NOT in the data (a door handle), so they show the register without handing the LM any motion words.
REGISTERS = {
    "narration": 'first person, as if the character describes what it does: "I reach for the handle, then pull it towards me"',
    "command": 'an imperative a user would type to tell the character what to do: "grab the handle and pull"',
    "terse": 'two to six words, keyword style, no full sentence: "pull handle"',
    "casual": 'informal, as typed quickly by a person: lowercase, abbreviations, now and then a typo or a missing word: "grab teh handle n pull"',
}
UNIT_S = 0.5            # segmentation looks at the clip in half-second units
MIN_SEG_S = 1.0         # a segment shorter than this is merged into its neighbour
FLOORS = dict(speed=0.20, yawrate=6.0, hand=0.06, gaze=6.0, pelvis=0.03)     # smallest change worth a new segment (m/s, deg/s, m, deg, m)


# ---------------------------------------------------------------------------------------------------------------------------------------------------- measurement
def load(path):
    z = np.load(path, allow_pickle=True)
    return z["actions"].astype(np.float64), z["offsets"].astype(np.int64), float(z["dt"]), [str(n) for n in z["action_names"]]


def smooth(x, k):
    if k <= 1:
        return x
    pad = np.pad(x, ((k // 2, k - 1 - k // 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(pad, 0)], 0)
    return (cs[k:] - cs[:-k]) / k


def to_body(p, P, yaw):
    """world points [T,3] -> the pelvis frame of every frame: (left, up, forward)."""
    d = p - P; c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([d[:, 0] * c - d[:, 2] * s, d[:, 1], d[:, 0] * s + d[:, 2] * c], 1)


class Clip:
    def __init__(self, A, names, dt):
        ix = {n: i for i, n in enumerate(names)}
        g = lambda *ns: np.stack([A[:, ix[n]] for n in ns], 1)
        self.dt, self.T = dt, len(A)
        self.P = g("pelvis_x", "pelvis_y", "pelvis_z"); self.yaw = np.unwrap(A[:, ix["yaw"]])
        self.contact = g("footL_contact", "footR_contact") > 0.5
        self.hand = {"left": to_body(g("handL_x", "handL_y", "handL_z"), self.P, self.yaw), "right": to_body(g("handR_x", "handR_y", "handR_z"), self.P, self.yaw)}
        self.neck = to_body(g("torso_x", "torso_y", "torso_z"), self.P, self.yaw)
        look = to_body(g("look_x", "look_y", "look_z"), self.P, self.yaw) - self.neck           # gaze direction seen from the neck
        self.gaze_left = np.degrees(np.arctan2(look[:, 0], look[:, 2])); self.gaze_up = np.degrees(np.arctan2(look[:, 1], np.hypot(look[:, 0], look[:, 2])))
        k = max(1, int(round(0.25 / dt)))
        v = np.gradient(self.P[:, [0, 2]], dt, axis=0)
        self.speed = smooth(np.hypot(v[:, 0], v[:, 1]), k)
        self.yawrate = smooth(np.degrees(np.gradient(self.yaw, dt)), k)

    def unit_features(self, a, b):
        sl = slice(a, b)
        return np.array([self.speed[sl].mean(), self.yawrate[sl].mean(), self.hand["left"][sl, 1].mean(), self.hand["right"][sl, 1].mean(), self.gaze_left[sl].mean(),
                         self.gaze_up[sl].mean(), self.P[sl, 1].mean()])


FEATURE_FLOOR = np.array([FLOORS["speed"], FLOORS["yawrate"], FLOORS["hand"], FLOORS["hand"], FLOORS["gaze"], FLOORS["gaze"], FLOORS["pelvis"]])


def unit_bounds(T, dt):
    n = max(1, int(round(T * dt / UNIT_S)))
    return np.linspace(0, T, n + 1).round().astype(int)


def segment(clip, scale):
    """Greedy merge of half-second units: a unit joins the current segment while EVERY feature stays within `scale` of the segment's running mean."""
    b = unit_bounds(clip.T, clip.dt); U = [clip.unit_features(b[i], b[i + 1]) for i in range(len(b) - 1)]
    segs, cur, mean = [], [0], U[0].copy()
    for i in range(1, len(U)):
        if np.all(np.abs(U[i] - mean) <= scale):
            cur.append(i); mean = np.mean([U[j] for j in cur], 0)
        else:
            segs.append(cur); cur, mean = [i], U[i].copy()
    segs.append(cur)
    out = [[int(b[s[0]]), int(b[s[-1] + 1])] for s in segs]
    min_len = int(MIN_SEG_S / clip.dt)
    while len(out) > 1:                                                     # merge too-short segments into the closer neighbour
        j = min(range(len(out)), key=lambda i: out[i][1] - out[i][0])
        if out[j][1] - out[j][0] >= min_len:
            break
        f = lambda s: clip.unit_features(*s)
        nb = [k for k in (j - 1, j + 1) if 0 <= k < len(out)]
        k = min(nb, key=lambda k: float(np.max(np.abs(f(out[k]) - f(out[j])) / FEATURE_FLOOR)))
        lo, hi = min(j, k), max(j, k); out[lo:hi + 1] = [[out[lo][0], out[hi][1]]]
    return out


def corpus_scale(clips, tol):
    """Per-feature tolerance: tol x the corpus spread of the half-second features, never below the floors. Everything is derived from the data."""
    U = []
    for c in clips:
        b = unit_bounds(c.T, c.dt); U += [c.unit_features(b[i], b[i + 1]) for i in range(len(b) - 1)]
    return np.maximum(FEATURE_FLOOR, tol * np.std(np.array(U), 0))


def facts(clip, a, b, ref):
    sl = slice(a, b); dt = clip.dt; dur = (b - a) * dt
    y0 = clip.yaw[a]; d = clip.P[b - 1] - clip.P[a]
    left = d[0] * math.cos(y0) - d[2] * math.sin(y0); fwd = d[0] * math.sin(y0) + d[2] * math.cos(y0)
    path = float(np.hypot(np.diff(clip.P[sl, 0]), np.diff(clip.P[sl, 2])).sum())
    steps = int((np.diff(clip.contact[sl].astype(int), axis=0) == 1).sum())
    out = dict(seconds=round(dur, 1), path_m=round(path, 2), forward_m=round(fwd, 2), leftward_m=round(left, 2),
               heading_change_deg=int(round(math.degrees(clip.yaw[b - 1] - clip.yaw[a]))), mean_speed_mps=round(path / max(dur, 1e-6), 2),
               steps_per_min=int(round(steps / max(dur, 1e-6) * 60)), pelvis_lowered_cm=int(round((ref["pelvis_y"] - clip.P[sl, 1].mean()) * 100)))
    for side in ("left", "right"):
        h = clip.hand[side][sl]; rng = (h.max(0) - h.min(0)) * 100; ax = int(np.argmax(rng)); hz = None
        if rng[ax] >= 8 and len(h) >= 8:          # below ~8 cm it is idle jitter, not a rhythm
            x = h[:, ax] - h[:, ax].mean(); sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), dt); m = (f >= 0.5) & (f <= 6)
            if m.any() and sp[m].max() > 0.2 * sp.sum():
                hz = round(float(f[m][np.argmax(sp[m])]), 1)
        out[f"{side}_hand"] = dict(height_vs_pelvis_cm=int(round(h[:, 1].mean() * 100)), forward_cm=int(round(h[:, 2].mean() * 100)),
                                   movement_range_cm=int(round(rng.max())), rhythm_hz=hz)
    out["gaze"] = dict(toward_left_deg=int(round(clip.gaze_left[sl].mean())), upward_deg=int(round(clip.gaze_up[sl].mean())))
    nk = clip.neck[sl].mean(0)
    out["torso_lean_forward_deg"] = int(round(math.degrees(math.atan2(nk[2], nk[1])))); out["torso_lean_left_deg"] = int(round(math.degrees(math.atan2(nk[0], nk[1]))))
    return out


def reference(clips, all_facts):
    P = np.concatenate([c.P[:, 1] for c in clips]); neck = np.concatenate([c.neck[:, 1] for c in clips])
    mv = np.array([f["mean_speed_mps"] for fs in all_facts for f in fs if f["mean_speed_mps"] > 0.15]); hr = np.array([abs(f["heading_change_deg"]) / max(f["seconds"], 0.1) for fs in all_facts for f in fs])
    q = lambda x, p: round(float(np.percentile(x, p)), 2) if len(x) else None
    return dict(pelvis_y=float(np.percentile(P, 95)), pelvis_height_m=round(float(np.percentile(P, 95)), 2), neck_above_pelvis_cm=int(round(np.median(neck) * 100)),
                moving_speed_mps={"slow_p10": q(mv, 10), "typical_p50": q(mv, 50), "fast_p90": q(mv, 90)}, heading_change_deg_per_s={"typical_p50": q(hr, 50), "sharp_p90": q(hr, 90)})


# ---------------------------------------------------------------------------------------------------------------------------------------------------- prompts
def make_prompt(segs, ref, n_per, registers):
    r = {k: v for k, v in ref.items() if k != "pelvis_y"}
    lines = "\n".join(f"{i + 1}. " + json.dumps(s, separators=(",", ":")) for i, s in enumerate(segs))
    reg = "\n".join(f'  "{k}": {REGISTERS[k]}' for k in registers)
    return (f"A character moved. Below are measurements of consecutive parts of its motion, in time order. Write how different people would describe or ask for this motion.\n\n"
            f"MEASUREMENTS (character's own frame). heading_change_deg, leftward_m and toward_left_deg are POSITIVE towards the character's own LEFT and negative towards its RIGHT; upward_deg is positive when looking up, negative when looking down. "
            f"forward_m is along where it faced at the start of the part. Hand numbers are relative to the pelvis (so 0 cm = hand at pelvis height; the neck is about {ref['neck_above_pelvis_cm']} cm above the pelvis). "
            f"rhythm_hz is how often a repeating hand movement cycles (null = no repeating movement). steps_per_min counts footfalls.\n{lines}\n\n"
            f"REFERENCE for judging amounts (whole dataset): {json.dumps(r, separators=(',', ':'))}\n\n"
            f"RULES\n- Describe only what the numbers support. Left and right are ALWAYS the character's own (say 'my left' / 'to the left'; never mirror them). Positive = left.\n"
            f"- You choose every motion word yourself (walk, stroll, wave, crouch, ...) if and only if the numbers justify it; otherwise describe plainly. Do not invent anything the numbers do not show.\n"
            f"- Keep the parts in order and join them naturally (then, after that, while...). Skip unremarkable details (tiny hand movement, slight lean). Mention a duration or amount only sometimes, rounded the way a person would.\n"
            f"- Never mention the measurement names, numbers with units like 'm/s', 'part 2', or 'segment'.\n- Vary wording a lot between lines.\n\n"
            f"Return ONLY a JSON object with these keys, each a list of {n_per} different strings:\n{reg}\n")


# ---------------------------------------------------------------------------------------------------------------------------------------------------- stages
def split_of(clip_id, seed=0):
    h = int(hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()[:8], 16) % 100          # same rule as build_dataset.py
    return "val" if h < 10 else "test" if h < 20 else "train"


def analyse(a):
    A, off, dt, names = load(a.clips)
    clips = [Clip(A[off[k]:off[k + 1]], names, dt) for k in range(len(off) - 1)]
    scale = corpus_scale(clips, a.tol)
    segs = [segment(c, scale) for c in clips]
    ref0 = dict(pelvis_y=float(np.percentile(np.concatenate([c.P[:, 1] for c in clips]), 95)))
    fs = [[facts(c, s0, s1, ref0) for s0, s1 in sg] for c, sg in zip(clips, segs)]
    return clips, segs, fs, reference(clips, fs) | ref0


def stage_inspect(a):
    clips, segs, fs, ref = analyse(a)
    print("reference:", json.dumps({k: v for k, v in ref.items() if k != "pelvis_y"}))
    print("segments per clip:", np.bincount([len(s) for s in segs]).tolist(), "(index = count)")
    for k in range(min(a.show, len(clips))):
        print(f"\nclip{k:05d}  {clips[k].T} frames")
        for (s0, s1), f in zip(segs[k], fs[k]):
            print(f"  [{s0:4d}:{s1:4d}] {json.dumps(f, separators=(',', ':'))}")


def stage_requests(a):
    clips, segs, fs, ref = analyse(a)
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True); regs = a.registers.split(","); rows = []
    for k, sg in enumerate(segs):
        n = len(sg); spans = {(0, min(n, a.max_segs))}
        while len(spans) < min(a.spans_per_clip, n * (n + 1) // 2 if n < 4 else 99):
            s0 = rng.randrange(n); spans.add((s0, min(n, s0 + rng.randint(1, min(a.max_segs, 4)))))
        for s0, s1 in sorted(spans):
            rows.append(dict(id=f"clip{k:05d}:{s0}-{s1}", clip=f"clip{k:05d}", frames=[sg[s0][0], sg[s1 - 1][1]], n_segments=s1 - s0, facts=fs[k][s0:s1],
                             prompt=make_prompt(fs[k][s0:s1], ref, a.per_register, regs), registers=regs))
    with open(os.path.join(a.out, "requests.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(reference=ref, tol=a.tol, registers=regs, per_register=a.per_register, clips=os.path.abspath(a.clips)), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"wrote {len(rows)} requests for {len(clips)} clips -> {a.out}/requests.jsonl  (feed each `prompt` to any LM; save {{\"id\":..., \"response\":...}} lines to responses.jsonl)")


def stage_run(a):
    key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit("set ANTHROPIC_API_KEY (or run the prompts yourself and write responses.jsonl)")
    req = [json.loads(l) for l in open(os.path.join(a.out, "requests.jsonl"))]; path = os.path.join(a.out, "responses.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set(); todo = [r for r in req if r["id"] not in done]
    def call(r):
        body = json.dumps(dict(model=a.model, max_tokens=1500, temperature=1.0, messages=[dict(role="user", content=r["prompt"])])).encode()
        for t in range(4):
            try:
                rq = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
                d = json.load(urllib.request.urlopen(rq, timeout=120)); return dict(id=r["id"], response="".join(b.get("text", "") for b in d["content"]))
            except Exception as e:
                err = str(e); time.sleep(2 ** t)
        return dict(id=r["id"], error=err)
    with cf.ThreadPoolExecutor(a.workers) as ex, open(path, "a") as f:
        for i, out in enumerate(ex.map(call, todo)):
            f.write(json.dumps(out) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


BODY_SIDE = re.compile(r"\b(left|right)\s+(hand|arm|foot|feet|leg|shoulder|eye|side|knee|elbow|wrist|palm|fist|ear|heel|toe)s?\b|\b(hand|arm|foot|leg|shoulder|eye|knee|elbow)s?\s+(on|to)\s+(my|the)\s+(left|right)\b", re.I)


def side_words(t):
    return set(re.findall(r"\b(left|right)\b", BODY_SIDE.sub(" ", t.lower())))


def measured_sides(fl):
    s = set()
    for f in fl:
        for v, thr, pos in ((f["heading_change_deg"], 25, 1), (f["leftward_m"], 0.4, 1), (f["gaze"]["toward_left_deg"], 20, 1)):
            if abs(v) >= thr:
                s.add("left" if v > 0 else "right")
    return s


def parse(resp):
    m = re.search(r"\{.*\}", resp, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
    return {k: [x.strip() for x in v if isinstance(x, str)] for k, v in d.items() if isinstance(v, list)}


def stage_apply(a):
    req = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.out, "requests.jsonl")))}
    st = dict(requests=len(req), responses=0, unparsed=0, texts=0, bad_length=0, duplicate=0, side_conflict=0, kept=0); seen, rows = set(), []
    for l in open(os.path.join(a.out, "responses.jsonl")):
        r = json.loads(l)
        if r["id"] not in req or "response" not in r:
            continue
        st["responses"] += 1; q = req[r["id"]]; d = parse(r["response"])
        if not d:
            st["unparsed"] += 1; continue
        sides = measured_sides(q["facts"])
        for reg in q["registers"]:
            for t in d.get(reg, []):
                st["texts"] += 1; key = (q["id"], t.lower())
                if not 2 <= len(t) <= 240 or "\n" in t:
                    st["bad_length"] += 1; continue
                if key in seen:
                    st["duplicate"] += 1; continue
                w = side_words(t)
                if sides and w and not (w & sides):                 # the text names only sides the measurements do not show
                    st["side_conflict"] += 1; continue
                seen.add(key); st["kept"] += 1
                rows.append(dict(clip=q["clip"], split=split_of(q["clip"], a.seed), text=t, register=reg, frames=q["frames"], n_segments=q["n_segments"], id=q["id"], source="captioned"))
    with open(os.path.join(a.out, "captions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(st, open(os.path.join(a.out, "apply_report.json"), "w"), indent=1); print(json.dumps(st))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", choices=["inspect", "requests", "run", "apply"], default="inspect")
    ap.add_argument("--clips", default="data/clips.npz"); ap.add_argument("--out", default="data/captions")
    ap.add_argument("--tol", type=float, default=0.5, help="segment tolerance in corpus std units (bigger = fewer, longer segments)")
    ap.add_argument("--spans-per-clip", type=int, default=3); ap.add_argument("--max-segs", type=int, default=6)
    ap.add_argument("--registers", default="narration,command,terse,casual"); ap.add_argument("--per-register", type=int, default=2)
    ap.add_argument("--show", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dict(inspect=stage_inspect, requests=stage_requests, run=stage_run, apply=stage_apply)[a.stage](a)


if __name__ == "__main__":
    main()
