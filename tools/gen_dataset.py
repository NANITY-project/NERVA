#!/usr/bin/env python3
"""
gen_dataset.py - make training clips out of the NERVA/Nebula body engine, and test how far they can be trusted.

  python tools/gen_dataset.py --n 500 --out data            # generate + verify + report
  python tools/gen_dataset.py --n 500 --out data --stress   # also: do bad "generator-like" clips get caught?

What it does
  1. For every clip it samples a style (step time, stride, step width, turning, arm swing, bob, posture, head pitch), plans footsteps, lets the
     engine's dynamic walking controller walk them, and records one ACTION vector per frame (30 floats, the same layout `track` takes):
         pelvis xyz yaw | footL xyz yaw contact | footR xyz yaw contact | handL xyz on | handR xyz on | look xyz on | torso xyz on
     Hands/look/torso are always written out as explicit targets (on=1), so nothing about the clip depends on procedural defaults.
  2. Every clip is replayed from a fresh standing body through the engine's feasibility verdict (preview), so the file says which clips
     the engine itself accepts and how badly the others fail.
  3. Gait statistics are measured from the result (speed, cadence, pelvis bob, foot clearance, jerk) so you can see what you actually made.
  4. --stress corrupts clips the way an imperfect generator would (noise, blur, blending two clips, flipped contacts, over-long strides)
     and reports whether the verdict catches them. Honest limit: the verdict checks FEASIBILITY. Anything feasible-but-unnatural (small
     jitter, robotic timing) passes it; the jerk column is there because the verdict cannot see that.

Output (--out DIR)
  clips.npz    actions [F,30] float32 (all clips concatenated), offsets [N+1], style [N,S], style_names, measured [N,M], measured_names,
               feasible [N], score [N], dt, action_names
  report.json  all the summary numbers      example.csv  the first clip as text, to eyeball or plot

Needs: numpy and a C++ compiler. The engine library is built automatically into build/libnebula.so the first run, and rebuilt
whenever the engine sources change (--lib FILE to use your own, --no-build to never compile).
All clips start from the standing pose at the origin facing +Z (z forward, x = the body's left). Canonicalise per clip, if you need to.
"""
import argparse, ctypes as C, glob, json, math, os, shutil, subprocess, sys, time
import numpy as np

ACTION_NAMES = (["pelvis_x", "pelvis_y", "pelvis_z", "yaw"]
                + [f"foot{s}_{k}" for s in "LR" for k in ("x", "y", "z", "yaw", "contact")]
                + [f"hand{s}_{k}" for s in "LR" for k in ("x", "y", "z", "on")]
                + ["look_x", "look_y", "look_z", "look_on", "torso_x", "torso_y", "torso_z", "torso_on"])
A = len(ACTION_NAMES)  # 30
STYLE_NAMES = ["step_time", "stride", "step_width", "turn_deg", "arm_swing", "twist", "bob", "torso_drop", "torso_lean", "torso_side", "look_down_deg", "look_yaw_deg"]
MEASURED_NAMES = ["speed_mps", "cadence_spm", "bob_cm", "foot_clearance_cm", "min_margin_cm", "jerk_pelvis", "n_steps", "frames"]


# ------------------------------------------------------------------------------------------------------------------ engine binding
def find_repo(start):
    """The engine checkout: the nearest folder above `start` that has capi/nebula_capi.cpp."""
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "capi", "nebula_capi.cpp")):
            return d
        d = os.path.dirname(d)
    return None


def build_lib(repo, force=False):
    """Compile build/libnebula.so from the sources if it is missing or older than any of them. Returns its path."""
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    out = os.path.join(repo, "build", "libnebula" + ext)
    srcs = [os.path.join(repo, "capi", "nebula_capi.cpp"), os.path.join(repo, "capi", "nebula_capi.h")] + glob.glob(os.path.join(repo, "include", "*.hpp"))
    newest = max(os.path.getmtime(x) for x in srcs if os.path.exists(x))
    if not force and os.path.exists(out) and os.path.getmtime(out) >= newest:
        return out
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx:
        sys.exit("libnebula needs building but no C++ compiler was found (install g++ or clang++, or set CXX).")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [cxx, "-std=c++17", "-O3", "-fPIC", "-shared", "-fvisibility=hidden", "-fno-math-errno", "-fno-trapping-math",
           "-I" + os.path.join(repo, "include"), "-I" + os.path.join(repo, "capi"), os.path.join(repo, "capi", "nebula_capi.cpp"), "-o", out]
    print(f"building {os.path.relpath(out)} ({'rebuilding: sources changed' if os.path.exists(out) else 'first run'}) ...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        sys.exit(f"could not run the compiler ({cxx}): {e}")
    if r.returncode != 0:
        sys.exit("building libnebula failed:\n" + "\n".join(r.stderr.strip().splitlines()[-25:]))
    print(f"  built in {time.time() - t0:.0f}s", flush=True)
    return out


class Lib:
    def __init__(self, path=None, repo=None, build=True):
        lib_path = path
        if not lib_path:
            repo = repo or find_repo(os.getcwd()) or find_repo(os.path.dirname(os.path.abspath(__file__)))
            if repo and build:
                lib_path = build_lib(repo)
            else:
                here = os.path.dirname(os.path.abspath(__file__))
                for p in ([os.path.join(repo, "build", "libnebula.so")] if repo else []) + [os.path.join(here, "libnebula.so"), os.path.join(os.getcwd(), "libnebula.so")]:
                    if os.path.exists(p):
                        lib_path = p; break
        if not lib_path or not os.path.exists(lib_path):
            if repo:
                sys.exit("build/libnebula.so does not exist and --no-build was given: run once without --no-build.")
            sys.exit("the engine sources were not found. Run this from inside the repo (or pass --repo DIR), or pass --lib FILE.")
        lib = C.CDLL(os.path.abspath(lib_path))
        F, I, P, S = C.c_float, C.c_int, C.c_void_p, C.c_char_p
        fp = C.POINTER(F)
        sig = {"nebula_abi_version": (I, []), "nebula_body_create": (P, [F, I, I]), "nebula_body_destroy": (None, [P]),
               "nebula_body_command": (I, [P, S, C.c_char_p, I]), "nebula_body_update": (None, [P, F]), "nebula_body_set_param": (I, [P, S, F]),
               "nebula_body_bone_count": (I, [P]), "nebula_body_bone_name": (S, [P, I]), "nebula_body_bones": (I, [P, fp]),
               "nebula_body_get_state": (I, [P, fp]), "nebula_body_track": (I, [P, fp, F, fp]), "nebula_body_replay_clip": (I, [P, fp, I, F, fp, fp, fp]), "nebula_body_state_dim": (I, []), "nebula_body_action_dim": (I, []), "nebula_body_report_dim": (I, []),
               "nebula_body_evaluate_clip": (I, [P, fp, I, F, I, fp, fp, C.c_char_p, I])}
        for k, (res, args) in sig.items():
            try:
                f = getattr(lib, k); f.restype = res; f.argtypes = args
            except AttributeError:
                sys.exit(f"libnebula is missing {k}: it was built from older sources; delete build/libnebula.so and run again")
        if lib.nebula_abi_version() != 3:
            sys.exit("libnebula ABI mismatch (expected 3): delete build/libnebula.so and run again")
        if lib.nebula_body_action_dim() != A:
            sys.exit(f"engine action vector is {lib.nebula_body_action_dim()} floats, this script expects {A}")
        self.L = lib
        self.S = lib.nebula_body_state_dim()


def fptr(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


class Body:
    def __init__(self, lib, scale=1.0):
        self.lib, self.L = lib, lib.L
        self.h = self.L.nebula_body_create(scale, 1, 1)
        n = self.L.nebula_body_bone_count(self.h)
        self.names = [self.L.nebula_body_bone_name(self.h, i).decode() for i in range(n)]
        self.bone = {nm: i for i, nm in enumerate(self.names)}
        self._b = np.zeros(10 * n, np.float32)
        self._s = np.zeros(lib.S, np.float32)

    def close(self):
        if self.h:
            self.L.nebula_body_destroy(self.h); self.h = None

    def cmd(self, line):
        buf = C.create_string_buffer(256)
        ok = self.L.nebula_body_command(self.h, line.encode(), buf, 256)
        return bool(ok), buf.value.decode()

    def param(self, k, v):
        return bool(self.L.nebula_body_set_param(self.h, k.encode(), v))

    def update(self, dt):
        self.L.nebula_body_update(self.h, dt)

    def state(self):
        self.L.nebula_body_get_state(self.h, fptr(self._s)); return self._s

    def bones(self):
        self.L.nebula_body_bones(self.h, fptr(self._b)); return self._b

    def evaluate(self, actions, dt, reports=False):
        """Judge a clip from the standing start on a copy (nothing is committed). Returns the verdict dict (+ reports if asked)."""
        a = np.ascontiguousarray(actions, np.float32); n = len(a)
        buf = C.create_string_buffer(4096); rep = np.zeros(n * 12, np.float32) if reports else None
        self.L.nebula_body_evaluate_clip(self.h, fptr(a), n, dt, 0, fptr(rep) if reports else None, None, buf, 4096)
        v = json.loads(buf.value.decode() or "{}")
        if reports:
            v["reports"] = rep.reshape(n, 12)
        return v


def new_standing(lib):
    b = Body(lib)
    for _ in range(60):
        b.update(1 / 60)
    return b


def qrot(q, v):  # q = (w, x, y, z); rotate v
    w, x, y, z = q; vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


# ------------------------------------------------------------------------------------------------------------------ sampling
STRIDE_RANGE = (0.25, 0.55)   # metres; the engine's reliable range (see the coverage table the script prints: longer strides mostly fail)


def sample_style(rng):
    u = rng.uniform
    turn = 0.0 if rng.random() < 0.4 else float(rng.normal(0, 28))
    return dict(step_time=u(0.7, 1.6), stride=u(*STRIDE_RANGE), step_width=u(0.035, 0.07), turn_deg=float(np.clip(turn, -100, 100)),
                arm_swing=u(0.0, 0.9), twist=u(0.05, 0.30), bob=u(0.955, 0.995),
                torso_drop=max(0.0, rng.normal(0.02, 0.04)), torso_lean=max(0.0, rng.normal(0.03, 0.05)), torso_side=float(rng.normal(0, 0.012)),
                look_down_deg=max(0.0, float(rng.normal(8, 14))), look_yaw_deg=float(rng.normal(0, 6)))


def plan_steps(rng, st):
    """Footsteps as (side, x, z, yaw_deg). Path heading changes steadily by turn_deg over the walk."""
    n = int(rng.integers(4, 11)); side = int(rng.integers(0, 2))  # 0 = left (+x), 1 = right
    L, w, tot = st["stride"], st["step_width"], math.radians(st["turn_deg"])
    cx = cz = 0.0; steps = []
    for i in range(1, n + 1):
        h = tot * i / n
        step_len = (L * 0.5 if i == 1 else L) * float(rng.uniform(0.92, 1.08))
        cx += step_len * math.sin(h); cz += step_len * math.cos(h)
        sgn = 1.0 if side == 0 else -1.0; ww = w + float(rng.normal(0, 0.006))
        steps.append((side, cx + sgn * ww * math.cos(h), cz - sgn * ww * math.sin(h), math.degrees(h) + float(rng.normal(0, 2.5))))
        side = 1 - side
    h = tot; sgn = 1.0 if side == 0 else -1.0  # closing step: the other foot comes level
    steps.append((side, cx + sgn * w * math.cos(h), cz - sgn * w * math.sin(h), math.degrees(h)))
    return steps


# ------------------------------------------------------------------------------------------------------------------ recording
def record_clip(lib, rng, st, fps=30):
    """Walk the planned steps in the engine and record actions. Returns (actions[T,30], measured dict) or (None, reason)."""
    sim_dt = 1 / 60; every = max(1, int(round(60 / fps))); dt = sim_dt * every
    b = new_standing(lib)
    try:
        b.param("step_time_scale", st["step_time"]); b.param("arm_swing_gain", st["arm_swing"]); b.param("twist_gain", st["twist"])
        b.param("leg_len_frac", st["bob"]); b.param("hand_idle_m", 0.0)
        steps = plan_steps(rng, st)
        for _ in range(15):
            b.update(sim_dt)
        frames, accepted = [], 0
        for side, x, z, yaw in steps:
            ok, msg = b.cmd(f"FOOT {'left' if side == 0 else 'right'} {x:.4f} 0 {z:.4f} {yaw:.2f}")
            if not ok:
                break
            accepted += 1
        if accepted < 3:
            return None, "steps refused: " + msg
        settle = 0
        for i in range(60 * 30):
            b.update(sim_dt)
            s = b.state()
            if s[10] > 0.5:
                return None, "fell"
            if i % every == 0:
                frames.append(action_from(b, s, st))
            if s[34] == 0 and s[15] == 0 and s[21] == 0:
                settle += 1
                if settle > 36:
                    break
            else:
                settle = 0
        else:
            return None, "did not finish"
        a = np.array(frames, np.float32)
        return a, dict(dt=dt, steps=accepted)
    finally:
        b.close()


def action_from(b, s, st):
    """One action vector from the body's current state + bones, with the posture/head style applied."""
    bo = b.bones(); yaw = float(s[3]); sy, cy = math.sin(yaw), math.cos(yaw)
    ni, hi = b.bone["neck"], b.bone["head"]
    neck = [float(bo[ni * 10]), float(bo[ni * 10 + 1]), float(bo[ni * 10 + 2])]
    hq = tuple(float(v) for v in bo[hi * 10 + 6:hi * 10 + 10]); hc = [float(s[31]), float(s[32]), float(s[33])]
    # posture: shift the neck base (and with it the head) in the body's heading frame: side, down, forward
    side, drop, lean = st["torso_side"], st["torso_drop"], st["torso_lean"]
    off = (side * cy + lean * sy, -drop, -side * sy + lean * cy)
    neck = [neck[i] + off[i] for i in range(3)]; hc = [hc[i] + off[i] for i in range(3)]
    # gaze: the natural head direction, turned by look_yaw and pitched down by look_down
    d = qrot(hq, (0.0, 0.0, 1.0)); ang = math.atan2(d[0], d[2]) + math.radians(st["look_yaw_deg"])
    pit = math.asin(max(-1.0, min(1.0, d[1]))) - math.radians(st["look_down_deg"])
    dd = (math.cos(pit) * math.sin(ang), math.sin(pit), math.cos(pit) * math.cos(ang)); look = [hc[i] + 2.0 * dd[i] for i in range(3)]
    return [s[0], s[1], s[2], yaw,
            s[11], s[12], s[13], s[14], 1.0 if s[15] == 0 else 0.0,
            s[17], s[18], s[19], s[20], 1.0 if s[21] == 0 else 0.0,
            s[23], s[24], s[25], 1.0, s[26], s[27], s[28], 1.0,
            look[0], look[1], look[2], 1.0, neck[0], neck[1], neck[2], 1.0]


def measure(a, dt, v):
    pel = a[:, [0, 1, 2]]; sp = np.linalg.norm(np.diff(pel[:, [0, 2]], axis=0), axis=1) / dt
    moving = np.where(sp > 0.15)[0]
    if len(moving) < 10:
        return None
    i0, i1 = moving[0], moving[-1] + 1; seg = a[i0:i1]
    c = seg[:, [8, 13]]; lifts = int(((c[1:] < 0.5) & (c[:-1] > 0.5)).sum())            # contact -> swing events = steps
    dur = (i1 - i0) * dt; y = seg[:, 1]; y = y - np.convolve(y, np.ones(int(round(1 / dt)) | 1) / (int(round(1 / dt)) | 1), "same") if len(y) > 2 * int(round(1 / dt)) else y - y.mean()
    n_ = len(y); inner = slice(n_ // 4, n_ - n_ // 4)             # middle half of the walk: start-up and stopping steps are not steady gait
    clear = float(max(seg[:, 5].max(), seg[:, 10].max()))
    jerk = float(np.abs(np.diff(pel, n=3, axis=0)).mean() / dt ** 3) if len(pel) > 4 else 0.0
    rep = v.get("reports")
    return dict(speed_mps=float(sp[moving].mean()), cadence_spm=float(lifts / dur * 60), bob_cm=float(np.ptp(y[inner]) * 100), foot_clearance_cm=clear * 100,
                min_margin_cm=float(rep[:, 10].min() * 100) if rep is not None else 0.0, jerk_pelvis=jerk, n_steps=lifts, frames=len(a))


# ------------------------------------------------------------------------------------------------------------------ stress corruptions
def corruptions(rng, a, other, dt):
    cont = [i for i in range(A) if ACTION_NAMES[i].endswith(("_x", "_y", "_z")) and not ACTION_NAMES[i].startswith(("look", "torso"))]
    out = {}
    for mm in (5, 10, 20):
        c = a.copy(); c[:, cont] += rng.normal(0, mm / 1000, (len(a), len(cont))); out[f"noise {mm} mm"] = c
    k = 7; ker = np.ones(k) / k; c = a.copy()
    for i in cont:
        p = np.pad(a[:, i], (k // 2, k // 2), mode="edge"); c[:, i] = np.convolve(p, ker, "valid")
    out["blur (7-frame average)"] = c
    n = min(len(a), len(other)); out["blend with another clip"] = 0.5 * a[:n] + 0.5 * other[:n]
    c = a.copy(); flip = rng.random(len(a)) < 0.05
    for i in (8, 13):
        c[flip, i] = 1 - c[flip, i]
    out["5% contact flags flipped"] = c
    c = a.copy(); c[:, [4, 6, 9, 11]] = c[:, [4, 6, 9, 11]] * np.array([1.3, 1.3, 1.3, 1.3]); out["feet 30% farther out"] = c
    t = np.arange(0, len(a) - 1, 1 / 1.4); idx = np.floor(t).astype(int); fr = (t - idx)[:, None]
    c = a[idx] * (1 - fr) + a[idx + 1] * fr; c[:, [8, 13]] = a[np.round(t).astype(int), :][:, [8, 13]]; out["slowed x1.4 (control: should pass)"] = c
    return out, dt


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="clips to generate")
    ap.add_argument("--out", default="data"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fps", type=int, default=30, choices=(30, 60))
    ap.add_argument("--stride-range", type=float, nargs=2, metavar=("MIN", "MAX"), help="step length range in metres (default 0.25 0.55)")
    ap.add_argument("--min-realism", type=float, default=0.0, help="drop clips whose realism score (tools/realism.py, 0-100) is below this")
    ap.add_argument("--lib", help="use this libnebula instead of building one"); ap.add_argument("--repo", help="engine checkout (default: found automatically)")
    ap.add_argument("--no-build", action="store_true", help="never compile; use an existing build/libnebula.so")
    ap.add_argument("--stress", action="store_true", help="also test whether bad generator-like clips are caught")
    ap.add_argument("--keep-infeasible", action="store_true", help="keep clips the engine's own verdict rejects (default: drop them, but count them)")
    args = ap.parse_args()
    global STRIDE_RANGE
    if args.stride_range: STRIDE_RANGE = tuple(args.stride_range)
    lib = Lib(args.lib, args.repo, build=not args.no_build); rng = np.random.default_rng(args.seed); os.makedirs(args.out, exist_ok=True)
    import realism, view_clip
    checker = new_standing(lib); t0 = time.time(); realism_scores = []
    clips, styles, meas, feas, score, dropped = [], [], [], [], [], {}
    tries = 0; attempts = []
    while len(clips) < args.n and tries < args.n * 6:
        tries += 1; st = sample_style(rng); a, info = record_clip(lib, rng, st, args.fps)
        if a is None:
            dropped[info.split(":")[0]] = dropped.get(info.split(":")[0], 0) + 1; attempts.append((st, "fell" if info.startswith("fell") else "other")); continue
        v = checker.evaluate(a, info["dt"], reports=True); m = measure(a, info["dt"], v)
        attempts.append((st, "ok" if v["feasible"] else "infeasible"))
        if m is None:
            dropped["barely moved"] = dropped.get("barely moved", 0) + 1; continue
        if not v["feasible"] and not args.keep_infeasible:
            dropped["engine verdict: infeasible"] = dropped.get("engine verdict: infeasible", 0) + 1; continue
        rr = realism.grade_clip(a, info["dt"], realism.angles_from_engine(lib, a, info["dt"]))
        if rr["score"] is None or rr["score"] < args.min_realism:
            dropped["realism below --min-realism"] = dropped.get("realism below --min-realism", 0) + 1; continue
        realism_scores.append(rr)
        clips.append(a); styles.append([st[k] for k in STYLE_NAMES]); meas.append([m[k] for k in MEASURED_NAMES]); feas.append(bool(v["feasible"])); score.append(v["score"])
        if len(clips) % 25 == 0:
            print(f"  {len(clips)}/{args.n} clips ({time.time() - t0:.0f}s)", flush=True)
    if not clips:
        sys.exit("no clips were produced: " + json.dumps(dropped))
    dt = 1 / args.fps; off = np.cumsum([0] + [len(c) for c in clips])
    np.savez_compressed(os.path.join(args.out, "clips.npz"), actions=np.concatenate(clips).astype(np.float32), offsets=off, style=np.array(styles, np.float32),
                        style_names=np.array(STYLE_NAMES), measured=np.array(meas, np.float32), measured_names=np.array(MEASURED_NAMES),
                        feasible=np.array(feas), score=np.array(score, np.float32), realism=np.array([r["score"] for r in realism_scores], np.float32), dt=np.float32(dt), action_names=np.array(ACTION_NAMES))
    with open(os.path.join(args.out, "example.csv"), "w") as f:
        f.write(",".join(ACTION_NAMES) + "\n")
        for r in clips[0]:
            f.write(",".join(f"{x:.5f}" for x in r) + "\n")
    M = np.array(meas); rep = dict(clips=len(clips), tries=tries, dropped=dropped, fps=args.fps, frames=int(off[-1]), seconds=float(off[-1] * dt),
                                   feasible_fraction=float(np.mean(feas)), score_mean=float(np.mean(score)))
    print(f"\n{len(clips)} clips, {off[-1]} frames ({off[-1] * dt:.0f} s of motion) in {time.time() - t0:.0f}s. Dropped while generating: {dropped or 'none'}")
    print("\nWhat the engine's walking actually looks like in this dataset (human: speed 1.0-1.6 m/s at 100-120 steps/min, pelvis bob 3-6 cm):")
    print(f"  {'':22s}{'min':>9s}{'5%':>9s}{'median':>9s}{'95%':>9s}{'max':>9s}")
    for j, nm in enumerate(MEASURED_NAMES[:6]):
        q = np.percentile(M[:, j], [0, 5, 50, 95, 100]); print(f"  {nm:22s}" + "".join(f"{x:9.2f}" for x in q)); rep[nm] = [float(x) for x in q]
    print("\nHow strongly each style knob shows up in the measured motion (correlation, so you know what the knobs really control):")
    S = np.array(styles)
    for sn in ("step_time", "stride", "bob", "torso_lean"):
        j = STYLE_NAMES.index(sn); cs = ", ".join(f"{mn} {np.corrcoef(S[:, j], M[:, k])[0, 1]:+.2f}" for k, mn in enumerate(MEASURED_NAMES[:5]) if M[:, k].std() > 1e-9)
        print(f"  {sn:12s} -> {cs}")
    rs = np.array([r["score"] for r in realism_scores]); rep["realism_mean"] = float(rs.mean())
    print(f"\nRealism (tools/realism.py; real captured walks score 87-100, mean 95): mean {rs.mean():.0f}, 5% {np.percentile(rs, 5):.0f}, median {np.median(rs):.0f}, 95% {np.percentile(rs, 95):.0f}")
    worst = {}
    for r in realism_scores:
        for it in r["items"]:
            if it["score"] is not None:
                worst.setdefault(it["name"], []).append(it["score"])
    weak = sorted(((np.mean(v), k) for k, v in worst.items()))[:5]
    print("  weakest measures (mean 0-1 score): " + ", ".join(f"{k} {m:.2f}" for m, k in weak))
    print("\nWhere the engine succeeds (every attempt, incl. the dropped ones). ok = walked it AND replays as feasible; the rest is where it cannot do what was asked:")
    for key, edges in (("stride", np.linspace(STRIDE_RANGE[0], STRIDE_RANGE[1] + 1e-6, 5)), ("step_time", [0.7, 0.9, 1.1, 1.3, 1.61]), ("step_width", [0.035, 0.05, 0.0701])):
        line = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [o for st_, o in attempts if lo <= st_[key] < hi]
            if sel:
                line.append(f"{lo:.2f}-{hi:.2f}: {100 * sel.count('ok') / len(sel):.0f}% ok, {100 * sel.count('fell') / len(sel):.0f}% fell")
        print(f"  {key:10s} " + " | ".join(line))
    hr = {"speed 1.0-1.6 m/s": (0, 1.0, 1.6), "cadence 95-125 steps/min": (1, 95, 125), "pelvis bob 2-7 cm": (2, 2, 7)}
    print("  share of clips inside human ranges: " + ", ".join(f"{k}: {100 * np.mean((M[:, j] >= lo) & (M[:, j] <= hi)):.0f}%" for k, (j, lo, hi) in hr.items()))
    if args.stress:
        print("\nStress test: could the engine's verdict catch a bad generator? (corrupting your clean clips on purpose)")
        clean_scores = np.array(score); thr = float(np.percentile(clean_scores, 95)) + 1e-4
        sample = list(range(min(len(clips), 60))); res = {}
        cj = []
        for i in sample:
            cs, _ = corruptions(rng, clips[i], clips[(i + 1) % len(clips)], dt)
            for nm, c in cs.items():
                v = checker.evaluate(c, dt); mj = np.abs(np.diff(c[:, :3], n=3, axis=0)).mean() / dt ** 3
                res.setdefault(nm, []).append((not v["feasible"], v["score"], mj))
        base_j = float(np.median(M[:, 5]))
        print(f"  {'corruption':38s}{'flagged':>9s}{'score>clean95%':>16s}{'median score':>14s}{'jerk vs clean':>15s}")
        for nm, r in res.items():
            r = np.array(r, float); print(f"  {nm:38s}{r[:, 0].mean() * 100:8.0f}%{(r[:, 1] > thr).mean() * 100:15.0f}%{np.median(r[:, 1]):14.4f}{np.median(r[:, 2]) / base_j:14.1f}x")
            rep.setdefault("stress", {})[nm] = dict(flagged=float(r[:, 0].mean()), above_clean95=float((r[:, 1] > thr).mean()), median_score=float(np.median(r[:, 1])), jerk_ratio=float(np.median(r[:, 2]) / base_j))
        print("  (flagged = engine said infeasible. jerk vs clean = how much rougher the pelvis path is: this catches jitter the verdict tolerates.)")
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(f"\nwrote {args.out}/clips.npz, report.json, example.csv")
    checker.close()


if __name__ == "__main__":
    main()
