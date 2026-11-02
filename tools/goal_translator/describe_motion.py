#!/usr/bin/env python3
"""
describe_motion.py - give every motion clip free-form English captions, with NO predefined motion words anywhere.

  python tools/goal_translator/describe_motion.py --stage inspect  --clips data/clips.npz            # look at the segments + measured facts of a few clips
  python tools/goal_translator/describe_motion.py --stage requests --clips data/clips.npz --out data/captions
  python tools/goal_translator/describe_motion.py --stage setup                                      # installs ollama if missing, starts it, pulls the model, smoke-tests it
  python tools/goal_translator/describe_motion.py --stage run      --out data/captions               # captions every request with the local model (does the setup itself; resumable)
  python tools/goal_translator/describe_motion.py --stage apply    --out data/captions               # parse, left/right word check -> captions.jsonl
  python tools/goal_translator/describe_motion.py --stage verify   --out data/captions               # back-translate to numbers, compare turn sign + speed -> captions_verified.jsonl

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
import argparse, concurrent.futures as cf, hashlib, json, math, os, platform, random, re, shutil, subprocess, sys, time, urllib.request
import numpy as np

# The examples are about content NOT in the data (a door handle), so they show the register without handing the LM any motion words.
REGISTERS = {
    "narration": 'first person, as if the character describes what it does: "I reach for the handle, then pull it towards me"',
    "command": 'an imperative a user would type to tell the character what to do: "grab the handle and pull"',
    "terse": 'two to six words, keyword style, no full sentence: "pull handle"',
    "casual": 'informal, as typed quickly by a person: lowercase, abbreviations, now and then a typo or a missing word: "grab teh handle n pull"',
}
UNIT_S = 0.5            # segmentation looks at the clip in half-second units
MIN_SEG_S = 1.6         # a segment shorter than this is merged into its neighbour. Set from --timing: ONE goal per prior stride (48 frames = 1.6 s), anything shorter cannot be controlled
DEFAULT_MODEL = "phi4-mini:3.8b-q4_K_M"      # the official ollama tags are latest / 3.8b / 3.8b-q4_K_M / 3.8b-q8_0 / 3.8b-fp16 - there is NO q4_0 (latest == q4_K_M, 2.5 GB)
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


LOCOMOTION_KEYS = ("seconds", "path_m", "forward_m", "leftward_m", "heading_change_deg", "mean_speed_mps", "steps_per_min", "speed_rank_pct", "turn_rate_rank_pct")


def add_ranks(all_facts):
    """Dataset-relative scale, so the LM does not have to guess what 0.74 m/s means here: rank = % of ALL segments of this dataset that are slower / turn less sharply /
    move the arms less. 50 = typical for this data, 95 = among the most. Data-driven; no motion names, no thresholds."""
    flat = [f for fs in all_facts for f in fs]
    sp = np.array([f["mean_speed_mps"] for f in flat]); tr = np.array([abs(f["heading_change_deg"]) / max(f["seconds"], 0.1) for f in flat])
    arm = np.array([max(f["left_hand"]["movement_range_cm"], f["right_hand"]["movement_range_cm"]) for f in flat])
    rk = lambda arr, x: int(round(100 * float((arr < x).mean())))
    for f in flat:
        f["speed_rank_pct"] = rk(sp, f["mean_speed_mps"]); f["turn_rate_rank_pct"] = rk(tr, abs(f["heading_change_deg"]) / max(f["seconds"], 0.1))
        f["arm_movement_rank_pct"] = rk(arm, max(f["left_hand"]["movement_range_cm"], f["right_hand"]["movement_range_cm"]))


def prune(f, mode):
    """mode 'locomotion': only what the CURRENT prior can be conditioned on (heading change, displacement, speed, duration). mode 'all': also hands, gaze, posture - use it once the
    prior is retrained with those inputs (plan tokens), otherwise the captions talk about things the translator cannot turn into anything."""
    return {k: f[k] for k in LOCOMOTION_KEYS} if mode == "locomotion" else f


def reference(clips, all_facts):
    P = np.concatenate([c.P[:, 1] for c in clips]); neck = np.concatenate([c.neck[:, 1] for c in clips])
    mv = np.array([f["mean_speed_mps"] for fs in all_facts for f in fs if f["mean_speed_mps"] > 0.15]); hr = np.array([abs(f["heading_change_deg"]) / max(f["seconds"], 0.1) for fs in all_facts for f in fs])
    q = lambda x, p: round(float(np.percentile(x, p)), 2) if len(x) else None
    return dict(pelvis_y=float(np.percentile(P, 95)), pelvis_height_m=round(float(np.percentile(P, 95)), 2), neck_above_pelvis_cm=int(round(np.median(neck) * 100)),
                moving_speed_mps={"slow_p10": q(mv, 10), "typical_p50": q(mv, 50), "fast_p90": q(mv, 90)}, heading_change_deg_per_s={"typical_p50": q(hr, 50), "sharp_p90": q(hr, 90)})


# ---------------------------------------------------------------------------------------------------------------------------------------------------- prompts
def make_prompt(segs, ref, n_per, registers, mode="all"):
    lines = "\n".join(f"{i + 1}. " + json.dumps(x, separators=(",", ":")) for i, x in enumerate(segs))
    reg = "\n".join(f'  "{k}": {REGISTERS[k]}' for k in registers)
    body = ("hand and gaze numbers are relative to the pelvis (0 cm = pelvis height; the neck is about %d cm above it); upward_deg is positive looking up; rhythm_hz = how often a repeating hand movement cycles (null = none). " % ref["neck_above_pelvis_cm"]) if mode == "all" else ""
    return (f"A character moved. Below are measurements of consecutive parts of its motion, in time order. Write how different people would describe or ask for this motion.\n\n"
            f"MEASUREMENTS (character's own frame). heading_change_deg and leftward_m are POSITIVE towards the character's own LEFT, negative towards its RIGHT. forward_m is along its starting direction. "
            f"{body}The *_rank_pct numbers say how this part compares with ALL parts in this dataset: 50 = typical, 90 = faster/sharper/more vigorous than 90% of them, 5 = nearly the slowest. "
            f"Use the ranks, not the raw units, to choose how fast/slow/sharp/gentle your words are.\n{lines}\n\n"
            f"RULES\n- Describe only what the numbers support. Left and right are ALWAYS the character's own. Positive = left, negative = right.\n"
            f"- Choose every motion word yourself, and only if the numbers justify it; otherwise describe plainly. Invent nothing.\n"
            f"- Keep the parts in time order and join them naturally. Skip unremarkable details. Mention a duration only sometimes, rounded like a person would.\n"
            f"- Never repeat measurement names or raw numbers with units. Vary wording a lot.\n\n"
            f"Return ONLY a JSON object with these keys, each a list of {n_per} different strings:\n{reg}\n")


# ---------------------------------------------------------------------------------------------------------------------------------------------------- stages
def split_of(clip_id, seed=0):
    h = int(hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()[:8], 16) % 100          # same rule as build_dataset.py
    return "val" if h < 10 else "test" if h < 20 else "train"


def analyse(a):
    global MIN_SEG_S
    W, stride, dt_ = (float(x) for x in a.timing.split(","))
    MIN_SEG_S = a.min_seg_s or stride * dt_
    A, off, dt, names = load(a.clips)
    clips = [Clip(A[off[k]:off[k + 1]], names, dt) for k in range(len(off) - 1)]
    scale = corpus_scale(clips, a.tol)
    segs = [segment(c, scale) for c in clips]
    ref0 = dict(pelvis_y=float(np.percentile(np.concatenate([c.P[:, 1] for c in clips]), 95)))
    fs = [[facts(c, s0, s1, ref0) for s0, s1 in sg] for c, sg in zip(clips, segs)]
    add_ranks(fs)
    ref = reference(clips, fs) | ref0
    fs = [[prune(f, a.facts) for f in x] for x in fs]
    return clips, segs, fs, ref


def stage_inspect(a):
    clips, segs, fs, ref = analyse(a)
    print("reference:", json.dumps({k: v for k, v in ref.items() if k != "pelvis_y"}))
    print("segments per clip:", np.bincount([len(s) for s in segs]).tolist(), "(index = count)")
    for k in range(min(a.show, len(clips))):
        print(f"\nclip{k:05d}  {clips[k].T} frames")
        for (s0, s1), f in zip(segs[k], fs[k]):
            print(f"  [{s0:4d}:{s1:4d}] {json.dumps(f, separators=(',', ':'))}")


def sign_gate(a):
    """A mirrored dataset would make every caption consistently wrong and the left/right filter would pass them all, so the one-time sign check runs first."""
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_sign.py"), "--clips", a.clips], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:])
    if r.returncode != 0:
        sys.exit("[describe_motion] check_sign.py failed: the clips look MIRRORED (or channels swapped). Fix the data or gt_conventions.TURN_POSITIVE_IS_LEFT first, or pass --skip-sign-check.")


def stage_requests(a):
    if not a.skip_sign_check:
        sign_gate(a)
    clips, segs, fs, ref = analyse(a)
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True); regs = a.registers.split(","); rows = []
    for k, sg in enumerate(segs):
        n = len(sg); spans = {(0, min(n, a.max_segs))}
        while len(spans) < min(a.spans_per_clip, n * (n + 1) // 2 if n < 4 else 99):
            s0 = rng.randrange(n); spans.add((s0, min(n, s0 + rng.randint(1, min(a.max_segs, 4)))))
        for s0, s1 in sorted(spans):
            rows.append(dict(id=f"clip{k:05d}:{s0}-{s1}", clip=f"clip{k:05d}", frames=[sg[s0][0], sg[s1 - 1][1]], n_segments=s1 - s0, facts=fs[k][s0:s1],
                             prompt=make_prompt(fs[k][s0:s1], ref, a.per_register, regs, a.facts), registers=regs))
    with open(os.path.join(a.out, "requests.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(reference=ref, tol=a.tol, registers=regs, per_register=a.per_register, clips=os.path.abspath(a.clips)), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"wrote {len(rows)} requests for {len(clips)} clips -> {a.out}/requests.jsonl  (feed each `prompt` to any LM; save {{\"id\":..., \"response\":...}} lines to responses.jsonl)")


# ---------------------------------------------------------------------------------------------------------------------------------------------------- local LM (ollama)
def http(host, path, body=None, timeout=600):
    rq = urllib.request.Request(host.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(rq, timeout=timeout))


def server_up(host):
    try:
        http(host, "/api/tags", timeout=3); return True
    except Exception:
        return False


def have_model(host, model):
    names = {m["name"] for m in http(host, "/api/tags", timeout=10).get("models", [])}
    return model in names or (":" not in model and model + ":latest" in names)


def ensure_ollama(a):
    """Everything needed to talk to a local model, done automatically and idempotently: install ollama if missing, start the server, pull the model."""
    if not server_up(a.host):
        exe = shutil.which("ollama")
        if not exe:
            if a.no_install:
                sys.exit("[describe_motion] ollama is not installed (and --no-install was given). Install it from https://ollama.com/download and run again.")
            system = platform.system()
            if system == "Linux":
                print("[describe_motion] installing ollama with the official script: curl -fsSL https://ollama.com/install.sh | sh   (may ask for sudo)", flush=True)
                if subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True).returncode != 0:
                    sys.exit("[describe_motion] the ollama install failed; install it by hand from https://ollama.com/download")
            elif system == "Darwin" and shutil.which("brew"):
                print("[describe_motion] brew install ollama", flush=True)
                if subprocess.run(["brew", "install", "ollama"]).returncode != 0:
                    sys.exit("[describe_motion] brew install ollama failed")
            else:
                sys.exit(f"[describe_motion] cannot install ollama automatically on {system}: download it from https://ollama.com/download, then run again.")
            exe = shutil.which("ollama") or sys.exit("[describe_motion] ollama still not on PATH after install")
        if a.host.rstrip("/").endswith(("localhost:11434", "127.0.0.1:11434")):
            print("[describe_motion] starting `ollama serve` in the background", flush=True)
            subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(60):
            if server_up(a.host):
                break
            time.sleep(0.5)
        else:
            sys.exit(f"[describe_motion] no ollama server answering at {a.host}")
    if not have_model(a.host, a.model):
        print(f"[describe_motion] pulling {a.model} (first time only)", flush=True)
        rq = urllib.request.Request(a.host.rstrip("/") + "/api/pull", json.dumps({"model": a.model, "stream": True}).encode(), {"content-type": "application/json"}); last = 0
        try:
            with urllib.request.urlopen(rq, timeout=3600) as r:
                for line in r:
                    d = json.loads(line)
                    if "error" in d:
                        raise RuntimeError(d["error"])
                    if d.get("total") and time.time() - last > 5:
                        print(f"   {d.get('status', '')} {100 * d.get('completed', 0) / d['total']:.0f}%", flush=True); last = time.time()
        except Exception as e:
            sys.exit(f"[describe_motion] could not pull {a.model}: {e}\n  (official phi4-mini tags: phi4-mini, phi4-mini:3.8b-q4_K_M, 3.8b-q8_0, 3.8b-fp16 - there is no q4_0)")
        if not have_model(a.host, a.model):
            sys.exit(f"[describe_motion] {a.model} is not available after the pull")
    print(f"[describe_motion] local LM ready: {a.model} @ {a.host}")


def chat(a, prompt, schema):
    body = dict(model=a.model, messages=[dict(role="user", content=prompt)], stream=False, format=schema,
                options=dict(temperature=a.temperature, num_ctx=a.num_ctx, num_predict=a.num_predict, seed=a.seed))
    err = ""
    for t in range(3):
        try:
            return http(a.host, "/api/chat", body, timeout=a.timeout)["message"]["content"]
        except Exception as e:
            err = str(e); time.sleep(2 ** t)
    raise RuntimeError(err)


def stage_setup(a):
    ensure_ollama(a)
    t0 = time.time(); out = chat(a, 'Reply with the JSON {"ok": true}', {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    print(f"[describe_motion] smoke test ok in {time.time() - t0:.1f}s: {out.strip()[:60]}")


def caption_schema(regs):
    return {"type": "object", "properties": {k: {"type": "array", "items": {"type": "string"}} for k in regs}, "required": list(regs)}


def run_parallel(a, jobs, fn, path):
    """jobs: [(id, payload)] -> appends {"id", "response"|"error"} lines to `path`, resumable. One worker by default: ollama serves requests one at a time unless OLLAMA_NUM_PARALLEL is set."""
    done = {json.loads(l)["id"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    todo = [j for j in jobs if j[0] not in done]; print(f"[describe_motion] {len(todo)} to do ({len(done)} already done) -> {path}", flush=True)
    def one(j):
        try:
            return dict(id=j[0], response=fn(j[1]))
        except Exception as e:
            return dict(id=j[0], error=str(e)[:300])
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex, open(path, "a") as f:
        for i, out in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(out) + "\n"); f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"   {i}/{len(todo)}  {(time.time() - t0) / i:.1f}s each", flush=True)


def stage_run(a):
    ensure_ollama(a)
    req = [json.loads(l) for l in open(os.path.join(a.out, "requests.jsonl"))]
    run_parallel(a, [(r["id"], r) for r in req], lambda r: chat(a, r["prompt"], caption_schema(r["registers"])), os.path.join(a.out, "responses.jsonl"))


# ---------------------------------------------------------------------------------------------------------------------------------------------------- numeric back-translation check
def estimate_prompt(texts, ref):
    sp = ref["moving_speed_mps"]
    lst = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (f"ESTIMATE. A character is told each instruction below. For each one estimate what the character ends up doing over the whole instruction:\n"
            f"- turn_deg: net heading change in degrees; POSITIVE = towards the character's own left, NEGATIVE = right, 0 = keeps its direction.\n"
            f"- speed_mps: average speed in metres per second; 0 = stands still. In this world slow is about {sp['slow_p10']}, typical {sp['typical_p50']}, fast {sp['fast_p90']}.\n\n"
            f"{lst}\n\nReturn ONLY JSON: {{\"estimates\": [{{\"turn_deg\": number, \"speed_mps\": number}}, ...]}} with exactly {len(texts)} entries, in the same order.")


ESTIMATE_SCHEMA = {"type": "object", "required": ["estimates"], "properties": {"estimates": {"type": "array", "items": {"type": "object", "required": ["turn_deg", "speed_mps"],
                   "properties": {"turn_deg": {"type": "number"}, "speed_mps": {"type": "number"}}}}}}


def stage_verify(a):
    """Second, lexicon-free check: an LM that has NOT seen the measurements turns each caption back into numbers (turn, speed); captions whose numbers disagree with the
    measured ones (opposite turn sign, or speed off by more than --speed-ratio) are dropped. Covers speed, which the left/right word check cannot. Tolerances are hyperparameters."""
    ensure_ollama(a)
    req = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.out, "requests.jsonl")))}
    meta = json.load(open(os.path.join(a.out, "meta.json"))); caps = [json.loads(l) for l in open(os.path.join(a.out, "captions.jsonl"))]
    by = {}
    for c in caps:
        by.setdefault(c["id"], []).append(c)
    path = os.path.join(a.out, "verify_raw.jsonl"); texts = {i: [c["text"] for c in cs] for i, cs in by.items()}
    run_parallel(a, [(i, t) for i, t in texts.items()], lambda t: chat(a, estimate_prompt(t, meta["reference"]), ESTIMATE_SCHEMA), path)
    raw = {}
    for l in open(path):
        r = json.loads(l)
        if "response" in r:
            raw[r["id"]] = r["response"]
    st = dict(captions=len(caps), unverified=0, turn_sign=0, speed_ratio=0, kept=0); out = []
    for i, cs in by.items():
        try:
            est = json.loads(raw[i])["estimates"]; assert len(est) == len(cs)
        except Exception:
            st["unverified"] += len(cs); continue
        fl = req[i]["facts"]; sec = sum(f["seconds"] for f in fl); turn = sum(f["heading_change_deg"] for f in fl); spd = sum(f["path_m"] for f in fl) / max(sec, 1e-6)
        for c, e in zip(cs, est):
            et, es = float(e["turn_deg"]), max(0.0, float(e["speed_mps"]))
            if abs(turn) >= a.turn_min and abs(et) >= a.turn_min and (turn > 0) != (et > 0):
                st["turn_sign"] += 1; continue
            if not (spd < 0.2 and es < 0.2) and max(spd, es) / max(min(spd, es), 0.05) > a.speed_ratio:
                st["speed_ratio"] += 1; continue
            st["kept"] += 1; out.append(dict(c, est_turn_deg=round(et), est_speed_mps=round(es, 2)))
    with open(os.path.join(a.out, "captions_verified.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(st, open(os.path.join(a.out, "verify_report.json"), "w"), indent=1); print(json.dumps(st))
    if st["captions"] and st["kept"] < 0.4 * st["captions"]:
        print("[describe_motion] WARNING: more than 60% dropped. A 3.8B model is a noisy estimator: loosen --speed-ratio / --turn-min, or use a larger --model for this stage.")


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
    ap.add_argument("--stage", choices=["inspect", "requests", "setup", "run", "apply", "verify"], default="inspect")
    ap.add_argument("--clips", default="data/clips.npz"); ap.add_argument("--out", default="data/captions")
    ap.add_argument("--tol", type=float, default=0.5, help="segment tolerance in corpus std units (bigger = fewer, longer segments)")
    ap.add_argument("--spans-per-clip", type=int, default=3); ap.add_argument("--max-segs", type=int, default=6)
    ap.add_argument("--registers", default="narration,command,terse,casual"); ap.add_argument("--per-register", type=int, default=2)
    ap.add_argument("--show", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", choices=["all", "locomotion"], default="all", help="locomotion = only what the current prior accepts (heading, displacement, speed); all = also hands, gaze, posture (needs a retrained prior)")
    ap.add_argument("--timing", default="64,48,0.0333333", help="the prior's W,stride,dt: the minimum segment is one stride (1.6 s)"); ap.add_argument("--min-seg-s", type=float, default=0.0)
    ap.add_argument("--skip-sign-check", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--host", default="http://localhost:11434"); ap.add_argument("--no-install", action="store_true", help="never install/start ollama")
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=600); ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--turn-min", type=float, default=15, help="verify: both turns must be at least this many degrees before their signs are compared")
    ap.add_argument("--speed-ratio", type=float, default=2.0, help="verify: drop a caption whose estimated speed differs from the measured one by more than this factor")
    a = ap.parse_args()
    dict(inspect=stage_inspect, requests=stage_requests, setup=stage_setup, run=stage_run, apply=stage_apply, verify=stage_verify)[a.stage](a)


if __name__ == "__main__":
    main()
