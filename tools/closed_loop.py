#!/usr/bin/env python3
"""
closed_loop.py - the character generates its own movement, window by window, with the ENGINE in the loop.

  python tools/closed_loop.py --run runs/cvqvae --goal 2.0 3.0                          # walk to the point (x=2, z=3) in world metres
  python tools/closed_loop.py --run runs/cvqvae --goal -1 4 --action walk --style Zombie --cfg-scale 2 --k 12
  python tools/closed_loop.py --run runs/cvqvae --goal-mode rate --turn-deg 20 --windows 12   # no target: a steady turn of 20 degrees per window
  python tools/view_clip.py runs/cvqvae/closed_loop.npz --open                          # watch what the body really did

tools/train_cvqvae.py --stage generate samples a whole clip first and plays it afterwards. This script does it the other way round, and asks the engine every time:

  1. a standing Body (the engine, libnebula) is created; the VAE, the prior and the dataset's stride / dt come from the checkpoints
  2. per window: read body.state() -> pelvis x, z, yaw. The GOAL SLOT of the new window is built from where the body is and where it has to go (GOAL SEMANTICS below)
  3. k candidates are sampled at once (batch dimension = k) with sample_window, decoded with the VAE, merged onto the not-yet-played tail of the previous window with merge_window
     and moved from the canonical frame (pelvis at the origin facing +z) to the world frame at the BODY's pose
  4. every candidate is tried on a private copy of the body: body.evaluate(actions, dt, reports=True, states=True) (commit=False), and only its first `stride` frames are JUDGED.
     The rest of the window is fed to the engine as look-ahead and then ignored: the engine's tracking looks ahead to the next landing of every swing foot, so a clip that is cut
     at `stride` frames shows a spurious loss of balance in its last ~0.4 s (measured: engine clips that pass as a whole fail in 3 of 5 stride-sized chunks). Ranking: feasible first,
     then the engine's score over the committed frames (mean correction cost in metres, the engine's own formula; differences below --score-tol count as equal), then progress to the goal
  5. the winner's first `stride` frames are committed: body.evaluate(actions[:stride], dt, commit=True) (the pose after the commit is the same as in a continuous run; only the balance
     margin the engine reports for the last frames is unreliable, so the margin used below is the look-ahead one from step 4). The other `W - stride` frames stay as `pending`
  6. fallen / stuck: the body is snapshotted (clone) before every commit. If it is `fallen` or its (look-ahead) balance margin is below --margin-min afterwards, it is restored and
     the window is sampled again with a fresh seed; when --retries are used up, or when no candidate is feasible, or when the body makes no progress for --stuck-windows windows,
     a stand clip (the body's own pose held for `stride` frames) is played and the history is reset. Too many faults in a row stop the run (status "stuck")

OUTPUT (--out, default the run folder)
  closed_loop.npz     tools/view_clip.py format: the frames the engine really tracked (committed windows and stand fallbacks), world frame, starting at the origin
  closed_loop_log.jsonl   one line per window and attempt: mode, requested goal, remaining turn / distance, all k scores and verdicts, chosen candidate, state afterwards, events
  closed_loop_summary.json   status (arrived / max_windows / stuck), windows, faults, final distance

GOAL SEMANTICS  (the 4 goal numbers of the prior describe what a WINDOW DID: heading change, displacement in its own start frame, path speed)
  arc   (default)  every window asks for the circular arc that ends AT the target: with b = the bearing of the target relative to the pelvis heading, an arc through the target
                   that starts along the heading turns by 2*b (tangent-chord angle) and has path length  dist * (b / sin b).  The turn is clipped to +-goal_stats.turn_max (the 95th
                   percentile of |heading change| of the TRAINING windows: asking for more is out of distribution and the prior will not deliver it, so the body turns over several
                   windows instead) and the path to speed_cruise * window seconds (shorter near the target, so the body slows down instead of overshooting).
  rate             no target: every window asks for --turn-deg degrees at cruise speed. Use it to probe what the prior can do; a turn above turn_max is clipped the same way.
  The previous windows in the context carry what they MEASURED (goal_features of the decoded window), which is how the prior saw them in training.
  The script prints turn_max / speed_cruise / speed_max of the prior at the start and warns when the requested motion cannot be in the data.
"""
import argparse, collections, json, math, os, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import gen_dataset as g
import train_cvqvae as T

A = g.A                                   # 30 action channels
ST_X, ST_Y, ST_Z, ST_YAW, ST_MARGIN, ST_FALLEN = 0, 1, 2, 3, 9, 10


# ================================================================================================================ pure helpers (no engine, no model: tests/test_closed_loop.py)
def wrap_pi(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def plan_window_goal(state, goal_xz, stats, sec, mode="arc", speed=None, turn_deg=0.0, prefer_sign=None):
    """Goal numbers (raw, GOAL_NAMES order) for the NEXT window, from the body state. -> (feat [4] float32, info dict).
    stats: the prior's goal_stats (turn_max, speed_cruise, speed_max). sec: seconds a window lasts, (W - 1) * dt.
    info: dist (m, None without a goal), bearing, remaining_turn (the target's bearing minus the pelvis heading, wrapped to +-pi), turn (what is asked), turn_clipped, path, speed."""
    px, pz, yaw = float(state[ST_X]), float(state[ST_Z]), float(state[ST_YAW])
    info = dict(dist=None, bearing=None, remaining_turn=None)
    if goal_xz is not None:
        dx, dz = goal_xz[0] - px, goal_xz[1] - pz
        dist = math.hypot(dx, dz); bearing = math.atan2(dx, dz)                  # heading convention of the engine: yaw 0 faces +z, heading vector (sin yaw, cos yaw)
        b = float(wrap_pi(bearing - yaw)); info.update(dist=dist, bearing=bearing, remaining_turn=b)
    v = float(stats["speed_cruise"] if speed is None else speed)
    if mode == "arc":
        if goal_xz is None:
            raise ValueError("goal mode 'arc' needs a goal (x, z)")
        b = info["remaining_turn"]
        if prefer_sign is not None and abs(b) > math.pi - 0.3:                  # target almost straight behind: keep turning the way we already did, do not flip
            b = abs(b) * prefer_sign
        want = 2.0 * b
    elif mode == "rate":
        want = math.radians(turn_deg)
    else:
        raise ValueError(f"unknown goal mode {mode!r}")
    tmax = float(stats["turn_max"]); turn = float(np.clip(want, -tmax, tmax)); h = turn / 2.0
    path = v * sec
    if mode == "arc":
        path = min(path, info["dist"] / float(np.sinc(h / math.pi)))             # chord = path * sinc(h/pi)  ->  path that puts the chord on the target; never longer than one cruise window
    info.update(turn=turn, turn_clipped=bool(abs(want) > tmax + 1e-9), path=path, speed=path / sec)
    return T.arc_goal(turn, path, sec).astype(np.float32), info


def anchor_pending(pending, x, z, yaw):
    """Rigidly move the unplayed tail so that its first frame has the pelvis at (x, z) with heading yaw (the body's real pose; the engine corrects what it tracks, so the body is
    never exactly where the clip planned it). Rotates every xz pair and adds the heading change to the yaw channels. [n,30] -> [n,30]"""
    p = np.asarray(pending, np.float64).copy(); d = float(yaw) - p[0, 3]; c, s = math.cos(d), math.sin(d); px, pz = p[0, 0], p[0, 2]
    for xi, zi in T.XZ:
        dx, dz = p[:, xi] - px, p[:, zi] - pz
        p[:, xi] = x + dx * c + dz * s; p[:, zi] = z - dx * s + dz * c
    for y in T.YAWS:
        p[:, y] += d
    return p.astype(np.float32)


def anchor_feet(win, state, fade=8):
    """Planted feet: the engine holds a planted foot where it is and refuses a clip that slides it by more than 5 mm, but the body's sole is never exactly where the clip planned it (and a body
    that has just stood up is somewhere else entirely). The contact run at the start of `win` [n,30] is moved onto the body's real sole (xyz); the shift fades out linearly over the swing that
    follows, so the next landing is where the clip put it. A foot that is not in contact at frame 0 is left alone."""
    p = np.asarray(win, np.float64).copy()
    for c0, s0 in ((4, 11), (9, 17)):
        if p[0, c0 + 4] < 0.5:
            continue
        delta = np.array([state[s0] - p[0, c0], state[s0 + 1] - p[0, c0 + 1], state[s0 + 2] - p[0, c0 + 2]]); n = len(p); run = 1
        while run < n and p[run, c0 + 4] > 0.5:
            run += 1
        nxt = run
        while nxt < n and p[nxt, c0 + 4] < 0.5:
            nxt += 1
        m = (nxt - run) if nxt < n else fade                                     # swing length (to the next landing), or a fixed fade at the end of the window
        w = np.zeros(n); w[:run] = 1.0
        m = min(m, n - run)
        if m > 0:
            w[run:run + m] = 1.0 - (np.arange(m) + 1) / float(m + 1)
        p[:, c0:c0 + 3] += w[:, None] * delta[None]
    return p.astype(np.float32)


def stand_clip(state, n):
    """n frames that hold the body's current pose: pelvis, both soles planted where they are, hands free (on = 0), no look / torso target. The fallback when sampling fails."""
    s = np.asarray(state, np.float64); a = np.zeros((n, A), np.float32)
    a[:, 0:3] = s[0:3]; a[:, 3] = s[3]
    a[:, 4:7] = s[11:14]; a[:, 7] = s[14]; a[:, 8] = 1.0                         # left sole xyz, yaw, contact
    a[:, 9:12] = s[17:20]; a[:, 12] = s[20]; a[:, 13] = 1.0                      # right sole
    a[:, 14:17] = s[23:26]; a[:, 18:21] = s[26:29]                               # wrists (on = 0: not a target)
    a[:, 22:25] = s[31:34]                                                       # look target = where the head is (look_on = 0)
    return a


def rank_key(verdict, progress_err, score_tol):
    """Smaller is better: feasible first, then the engine's score in bins of score_tol metres, then closeness to what was asked."""
    return (0 if verdict.get("feasible") else 1, int(float(verdict.get("score", 1e3)) / max(score_tol, 1e-9)), float(progress_err))


def judge_window(body, merged, stride, dt, fall_margin=-0.03):
    """Try the window `merged` [W,30] from the body's state on a private copy and judge only its first `stride` frames (the rest is look-ahead for the engine's landing prediction).
    The numbers are the engine's own (Body.frame_cost / bad-frame flag, from the per-frame reports), restricted to the committed frames.
    -> dict(feasible, bad_frames, score, worst_frame, margin_end, margin_min, issues, end_state)"""
    v = body.evaluate(np.ascontiguousarray(merged, np.float32), dt, reports=True, states=True)
    r = v["reports"][:stride]; st = v["states"][:stride]
    cost = r[:, [0, 1, 2, 3, 4, 5, 6, 8, 9]].sum(1) + np.where(r[:, 10] < fall_margin, -r[:, 10], 0.0)         # ik::Body::frame_cost
    bad = int((r[:, 11] > 0.5).sum())
    issues = [t for t in v.get("issues", []) if int((re.search(r"frames? (\d+)", t) or [0, 0])[1]) < stride]
    return dict(feasible=bad == 0, bad_frames=bad, score=float(cost.mean()), worst_frame=int(cost.argmax()), margin_end=float(st[-1, ST_MARGIN]), margin_min=float(r[:, 10].min()), issues=issues, end_state=st[-1].copy())


def build_ctx(prior, act_id, hist_codes, B, dev):
    """Token context of the prior: [ACTION] then every earlier window followed by SEP (same layout as generate_codes)."""
    toks = [prior.ACT0 + int(act_id)]
    for c in hist_codes:
        toks += [int(v) for v in c] + [prior.SEP]
    return torch.tensor(toks, dtype=torch.long, device=dev)[None].repeat(B, 1)


# ================================================================================================================ the loop
class ClosedLoop:
    def __init__(self, body, vae, norm, prior, names, stride, dt, action="walk", style=None, goal=None, mode="arc", k=8, temperature=0.9, top_k=40, top_p=1.0, cfg_scale=1.0,
                 speed=None, turn_deg=0.0, arrive=0.3, max_windows=40, retries=3, margin_min=None, stuck_windows=6, stuck_progress=0.15, max_faults=4, score_tol=1e-3,
                 heading_weight=0.2, accept_score=0.0, seed=0, dev="cpu", log=print, fall_margin=-0.03):
        self.body, self.vae, self.norm, self.prior, self.names, self.stride, self.dt, self.dev, self.log = body, vae, norm, prior, names, stride, dt, torch.device(dev), log
        self.W = vae.window; self.sec = (self.W - 1) * dt; self.Lw = prior.windows
        if not (0 < stride < self.W):
            raise ValueError(f"stride {stride} must be between 1 and the window length {self.W} - 1")
        if action in names:
            self.act_id = names.index(action)
        elif action in (None, "", "null"):
            self.act_id = prior.NULL - prior.ACT0
        else:
            raise ValueError(f"unknown action {action!r}; the model knows: {', '.join(names)}")
        self.action = names[self.act_id] if self.act_id < len(names) else "null"
        sn = prior.meta.get("style_names") or []; self.style_name = style or "none"; self.style_id = None
        if style:
            if style not in sn:
                raise ValueError(f"unknown style {style!r}; this prior knows: " + (", ".join(sn) if sn else "none (it was trained without style ids)"))
            self.style_id = sn.index(style)
        self.gstats = prior.meta.get("goal_stats")
        if self.gstats is None:                                                  # a prior without goal conditioning: nothing to steer with, only the ranking sees the goal
            self.log("warning: this prior was trained without goal conditioning: the goal slot is ignored, candidates are only RANKED by progress (retrain without --no-prior-goal)")
            self.gstats = dict(mean=[0, 0, 0, 0], std=[1, 1, 1, 1], turn_max=0.6, speed_cruise=1.0, speed_max=1.5)
        self.goal, self.mode, self.k, self.temperature, self.top_k, self.top_p, self.cfg_scale = goal, mode, k, temperature, top_k, top_p, cfg_scale
        self.speed, self.turn_deg, self.arrive, self.max_windows, self.retries = speed, turn_deg, arrive, max_windows, retries
        self.fall_margin = fall_margin; self.margin_min = fall_margin if margin_min is None else margin_min
        self.stuck_windows, self.stuck_progress, self.max_faults, self.score_tol, self.heading_weight, self.accept_score, self.seed = stuck_windows, stuck_progress, max_faults, score_tol, heading_weight, accept_score, seed
        if mode == "arc" and goal is None:
            raise ValueError("goal mode 'arc' needs --goal X Z")
        self.report_data_limits()

    def report_data_limits(self):
        st = self.gstats; tm = st["turn_max"]
        self.log(f"data limits (goal_stats of the prior): turn_max {tm:.2f} rad = {math.degrees(tm):.0f} deg per window, speed_cruise {st['speed_cruise']:.2f} m/s, speed_max {st.get('speed_max', float('nan')):.2f} m/s"
                 f"; a window lasts {self.sec:.2f} s, the body commits {self.stride * self.dt:.2f} s of it")
        if self.mode == "rate" and abs(math.radians(self.turn_deg)) > tm:
            self.log(f"warning: --turn-deg {self.turn_deg:g} is above turn_max ({math.degrees(tm):.0f} deg/window): the training windows hardly contain it, the ask is clipped to turn_max")
        if self.speed is not None and self.speed > st.get("speed_max", 1e9):
            self.log(f"warning: --speed {self.speed:g} is above the fastest 5% of the training windows ({st['speed_max']:.2f} m/s)")
        if self.mode == "arc" and self.goal is not None:
            self.log(f"goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) in world metres; a target behind the body is reached by turning at most {math.degrees(tm):.0f} deg per window (about {math.ceil(math.pi / tm)} windows for 180 deg)")

    # ---------------------------------------------------------------------------------------------------------- one sampling round
    def _sample(self, state, pending, hist_codes, hist_goals, seed):
        """k candidates for the next window. -> dict(codes [k,n], canon [k,W,30] decoded canonical windows, merged [k,W,30] world frame, info, goal_raw)"""
        feat, info = plan_window_goal(state, self.goal, self.gstats, self.sec, self.mode, self.speed, self.turn_deg, getattr(self, "_sign", None))
        if info.get("remaining_turn") is not None and abs(info["remaining_turn"]) > 0.05:
            self._sign = 1.0 if info["remaining_turn"] > 0 else -1.0
        goal_t = None
        if self.prior.goal_dim:
            slots = np.zeros((self.Lw, T.NG), np.float32); m = len(hist_goals)
            for j, gv in enumerate(hist_goals):
                slots[j] = gv
            slots[m] = T.norm_goal(feat, self.gstats)
            goal_t = torch.from_numpy(slots)[None].repeat(self.k, 1, 1).to(self.dev)
        style_t = None if self.style_id is None else torch.full((self.k,), self.style_id, dtype=torch.long, device=self.dev)
        ctx = build_ctx(self.prior, self.act_id, hist_codes, self.k, self.dev)
        torch.manual_seed(seed)
        codes, _ = T.sample_window(self.prior, ctx, goal_t, style_t, self.temperature, self.top_k, self.top_p, self.cfg_scale)
        acts = torch.full((self.k,), self.act_id, dtype=torch.long, device=self.dev)
        canon = T.to_motion(self.vae.decode_codes(codes, acts), self.norm)                       # [k,W,30], each window canonical-ish (frame 0 near the origin)
        x, z, yaw = float(state[ST_X]), float(state[ST_Z]), float(state[ST_YAW])
        pend = None if pending is None else anchor_pending(pending, x, z, yaw)                    # the plan, moved onto where the body really is
        merged = np.stack([anchor_feet(T.merge_window(pend, canon[i], self.stride, origin=(x, z, yaw)), state) for i in range(self.k)])
        return dict(codes=codes.cpu().numpy(), canon=canon, merged=merged, info=info, goal_raw=feat)

    def _progress_err(self, merged, info):
        """How far a candidate's committed part is from what was asked (smaller is better): distance left to the target plus heading error (arc), or the turn it made vs the turn asked (rate)."""
        e = merged[self.stride - 1]
        if self.mode == "arc":
            dx, dz = self.goal[0] - e[0], self.goal[1] - e[2]
            return math.hypot(dx, dz) + self.heading_weight * abs(float(wrap_pi(math.atan2(dx, dz) - e[3])))
        return abs((float(e[3]) - float(merged[0, 3])) - info["turn"] * (self.stride - 1) / (self.W - 1))

    # ---------------------------------------------------------------------------------------------------------- the run
    def run(self):
        body, stride, dt = self.body, self.stride, self.dt
        chunks, rows = [], []; pending = None; hist_codes = collections.deque(maxlen=self.Lw - 1); hist_goals = collections.deque(maxlen=self.Lw - 1)
        w = 0; faults = 0; attempt = 0; epoch = 0; dists = []; positions = []; status = "max_windows"; n_stand = 0; n_restored = 0; self._sign = None
        t0 = time.time()
        while w < self.max_windows:
            s = body.state().copy()
            dist = None if self.goal is None else math.hypot(self.goal[0] - s[ST_X], self.goal[1] - s[ST_Z])
            if self.mode == "arc" and dist < self.arrive:
                status = "arrived"; break
            if attempt == 0:                                                     # one entry per window, not per retry
                positions.append((float(s[ST_X]), float(s[ST_Z]))); dists.append(dist)
            if len(positions) > self.stuck_windows and attempt == 0:           # stuck: no progress over the last stuck_windows committed windows
                if self.mode == "arc":
                    moved = dists[-self.stuck_windows - 1] - dists[-1]
                else:
                    moved = math.hypot(positions[-1][0] - positions[-self.stuck_windows - 1][0], positions[-1][1] - positions[-self.stuck_windows - 1][1])
                if moved >= self.stuck_progress:
                    faults = 0                                                   # real progress ends a run of faults (a clean window alone does not: a body that stands still is clean)
                else:
                    faults += 1; epoch += 1; self.log(f"window {w}: stuck (moved {moved:.2f} m in {self.stuck_windows} windows, need {self.stuck_progress:.2f}) -> stand clip, fault {faults}/{self.max_faults}")
                    if faults > self.max_faults:
                        status = "stuck"; break
                    n_stand += self._stand(chunks, rows, w, s, "stuck"); pending = None; hist_codes.clear(); hist_goals.clear(); positions.clear(); dists.clear(); w += 1; continue
            seed = self.seed * 1000003 + w * 1009 + attempt * 97 + epoch * 7919
            cand = self._sample(s, pending, list(hist_codes), list(hist_goals), seed)
            snap = body.clone(); verdicts = []
            for i in range(self.k):                                              # every candidate on a private copy: the body itself is not touched (commit = False)
                verdicts.append(judge_window(body, cand["merged"][i], stride, dt, self.fall_margin))
            err = [self._progress_err(cand["merged"][i], cand["info"]) for i in range(self.k)]
            keys = [rank_key(verdicts[i], err[i], self.score_tol) for i in range(self.k)]
            order = sorted(range(self.k), key=lambda i: keys[i]); best = order[0]
            ok = verdicts[best]["feasible"] or verdicts[best]["score"] <= self.accept_score
            row = dict(window=w, attempt=attempt, mode="sample", seed=seed, goal_raw=[round(float(v), 4) for v in cand["goal_raw"]], remaining_turn=cand["info"].get("remaining_turn"),
                       dist=dist, turn_asked=cand["info"]["turn"], turn_clipped=cand["info"]["turn_clipped"], speed_asked=cand["info"]["speed"],
                       scores=[round(float(v["score"]), 5) for v in verdicts], feasible=[bool(v["feasible"]) for v in verdicts], chosen=best if ok else None,
                       chosen_score=float(verdicts[best]["score"]), chosen_margin=float(verdicts[best]["margin_end"]), chosen_issue=(verdicts[best].get("issues") or [None])[0], events=[])
            if not ok:                                                           # nothing usable: resample with a fresh seed, then stand
                snap.close(); row["events"].append("no feasible candidate"); rows.append(row)
                if attempt < self.retries:
                    attempt += 1; self.log(f"window {w}: no feasible candidate (best score {verdicts[best]['score']:.4f}), resampling (attempt {attempt}/{self.retries})"); continue
                faults += 1; epoch += 1; attempt = 0; self.log(f"window {w}: no feasible candidate after {self.retries} retries -> stand clip, fault {faults}/{self.max_faults}")
                if faults > self.max_faults:
                    status = "stuck"; break
                n_stand += self._stand(chunks, rows, w, s, "no feasible candidate"); pending = None; hist_codes.clear(); hist_goals.clear(); positions.clear(); dists.clear(); w += 1; continue
            act = cand["merged"][best][:stride]
            body.evaluate(act, dt, commit=True); s2 = body.state().copy(); v = verdicts[best]
            marg = v["margin_end"]                                               # the body's own margin at the cut is unreliable (no landing look-ahead): use the one judged with look-ahead
            fell = bool(s2[ST_FALLEN] > 0.5) or marg < self.margin_min
            row.update(after=dict(x=float(s2[ST_X]), z=float(s2[ST_Z]), yaw=float(s2[ST_YAW]), margin_m=marg, fallen=bool(s2[ST_FALLEN] > 0.5)))
            if fell:                                                             # undo the commit, sample again with a fresh seed
                body.restore(snap); snap.close(); n_restored += 1; row["events"].append("fell" if s2[ST_FALLEN] > 0.5 else "balance margin below limit"); row["mode"] = "restored"; rows.append(row)
                if attempt < self.retries:
                    attempt += 1; self.log(f"window {w}: body {'fell' if s2[ST_FALLEN] > 0.5 else 'lost balance'} (margin {marg:.3f} m) -> restored, resampling (attempt {attempt}/{self.retries})"); continue
                faults += 1; epoch += 1; attempt = 0; self.log(f"window {w}: still unstable after {self.retries} retries -> stand clip, fault {faults}/{self.max_faults}")
                if faults > self.max_faults:
                    status = "stuck"; break
                n_stand += self._stand(chunks, rows, w, body.state().copy(), "unstable"); pending = None; hist_codes.clear(); hist_goals.clear(); positions.clear(); dists.clear(); w += 1; continue
            snap.close(); rows.append(row); chunks.append(act.copy())
            pending = cand["merged"][best][stride:]; hist_codes.append(cand["codes"][best])
            hist_goals.append(T.norm_goal(T.goal_features(T.canonicalize(cand["canon"][best][None]), np.array([self.W]), dt)[0], self.gstats))   # what the window DID, as in training
            attempt = 0; w += 1
            self.log(f"window {w - 1:3d}  goal turn {math.degrees(cand['info']['turn']):6.1f} deg{' (clipped)' if cand['info']['turn_clipped'] else ''}  "
                     + (f"remaining {math.degrees(cand['info']['remaining_turn']):7.1f} deg  dist {dist:5.2f} m  " if dist is not None else "") + f"chose {best}/{self.k} score {v['score']:.4f}  margin {marg:.3f} m")
        s = body.state(); final = None if self.goal is None else math.hypot(self.goal[0] - s[ST_X], self.goal[1] - s[ST_Z])
        clip = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0, A), np.float32)
        summary = dict(status=status, windows=w, frames=int(len(clip)), seconds=round(len(clip) * dt, 3), faults=faults, stand_clips=n_stand, restores=n_restored, final_distance=final,
                       goal=None if self.goal is None else list(self.goal), mode=self.mode, action=self.action, style=self.style_name, k=self.k, wall_seconds=round(time.time() - t0, 1))
        return clip, rows, summary

    def _stand(self, chunks, rows, w, state, why):
        clip = stand_clip(state, self.stride); v = self.body.evaluate(clip, self.dt, commit=True); chunks.append(clip)       # a held pose has no swing foot: no look-ahead needed
        rows.append(dict(window=w, attempt=0, mode="stand", reason=why, chosen=None, chosen_score=float(v["score"]), feasible=bool(v["feasible"]), events=["stand clip"])); return 1


# ================================================================================================================ main
def write_outputs(out, clip, rows, summary, dt, loop):
    os.makedirs(out, exist_ok=True)
    goal = "" if summary["goal"] is None else f" goal ({summary['goal'][0]:g}, {summary['goal'][1]:g})"
    label = f"closed loop {summary['action']} {summary['mode']}{goal} k={summary['k']}" + ("" if loop.style_id is None else f" style {loop.style_name}") + f" -> {summary['status']}"
    desc = f"{summary['windows']} windows, {summary['seconds']:.1f} s, {summary['stand_clips']} stand clips, {summary['restores']} restores"
    sk = {} if loop.style_id is None else dict(style_labels=[loop.style_name], style_ids=[loop.style_id])
    T.write_clips_npz(os.path.join(out, "closed_loop.npz"), [clip], [summary["action"]], [label], [desc], dt, **sk)
    with open(os.path.join(out, "closed_loop_log.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    json.dump(summary, open(os.path.join(out, "closed_loop_summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/cvqvae", help="folder with best.pt (VAE) and prior_best.pt (written by tools/train_cvqvae.py)"); ap.add_argument("--vae-ckpt", default=""); ap.add_argument("--prior-ckpt", default="")
    ap.add_argument("--out", default="", help="output folder (default: --run)"); ap.add_argument("--device", default="auto")
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Z"), help="target in world metres (the body starts at the origin facing +z, its left is +x)")
    ap.add_argument("--goal-mode", choices=["arc", "rate"], default="arc", help="arc: every window is the arc that ends at the target; rate: a fixed --turn-deg per window, no target needed")
    ap.add_argument("--turn-deg", type=float, default=0.0, help="rate mode: heading change per window (degrees, + = towards the body's left axis x)"); ap.add_argument("--speed", type=float, default=None, help="path speed m/s (default: the data's cruise speed)")
    ap.add_argument("--arrive", type=float, default=0.3, help="stop when the pelvis is this close to the target (m)"); ap.add_argument("--windows", type=int, default=40, help="most windows to play")
    ap.add_argument("--action", default="walk"); ap.add_argument("--style", default="", help="a 100STYLE name the prior knows (e.g. Zombie); empty = none"); ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=8, help="candidates per window"); ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--top-k", type=int, default=40); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3, help="fresh-seed resamples of a window before the stand clip"); ap.add_argument("--margin-min", type=float, default=None, help="balance margin (m, judged with look-ahead) below which the body counts as unstable after a commit (default: --fall-margin)"); ap.add_argument("--fall-margin", type=float, default=-0.03, help="the engine's fall_margin (m), used to cost frames the way the engine does")
    ap.add_argument("--stuck-windows", type=int, default=6); ap.add_argument("--stuck-progress", type=float, default=0.15, help="m of progress (or of travel in rate mode) the body must make per --stuck-windows"); ap.add_argument("--max-faults", type=int, default=4)
    ap.add_argument("--score-tol", type=float, default=1e-3, help="engine scores closer than this (m) rank as equal, progress then decides"); ap.add_argument("--heading-weight", type=float, default=0.2, help="m per rad of heading error in the progress tie-break")
    ap.add_argument("--accept-score", type=float, default=0.0, help="also accept an infeasible winner whose score is at most this (m); 0 = feasible only"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib"); ap.add_argument("--repo"); ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    vp = a.vae_ckpt or os.path.join(a.run, "best.pt"); pp = a.prior_ckpt or os.path.join(a.run, "prior_best.pt")
    for p in (vp, pp):
        if not os.path.exists(p):
            sys.exit(f"{p} not found: train first (python tools/train_cvqvae.py --data data/vae --out {a.run})")
    vae, ck, norm = T.load_ckpt(vp, dev); vae.eval(); prior, pk = T.load_prior(pp, dev)
    if abs(ck["dt"] - pk["dt"]) > 1e-6:
        sys.exit("the VAE and the prior were trained on different dt")
    if a.goal is None and a.goal_mode == "arc":
        sys.exit("--goal X Z is needed in arc mode (or use --goal-mode rate --turn-deg D)")
    lib = g.Lib(a.lib, a.repo, build=not a.no_build); body = g.new_standing(lib)
    try:
        loop = ClosedLoop(body, vae, norm, prior, ck["action_names"], pk["stride"], ck["dt"], a.action, a.style or None, tuple(a.goal) if a.goal else None, a.goal_mode, a.k, a.temperature, a.top_k, a.top_p,
                          a.cfg_scale, a.speed, a.turn_deg, a.arrive, a.windows, a.retries, a.margin_min, a.stuck_windows, a.stuck_progress, a.max_faults, a.score_tol, a.heading_weight, a.accept_score, a.seed, dev, fall_margin=a.fall_margin)
    except ValueError as e:
        sys.exit(str(e))
    clip, rows, summary = loop.run(); out = a.out or a.run; write_outputs(out, clip, rows, summary, ck["dt"], loop)
    print(f"{summary['status']}: {summary['windows']} windows, {summary['seconds']:.1f} s, faults {summary['faults']}, stand clips {summary['stand_clips']}, restores {summary['restores']}"
          + ("" if summary["final_distance"] is None else f", {summary['final_distance']:.2f} m from the target"))
    print(f"wrote {out}/closed_loop.npz (+ closed_loop_log.jsonl, closed_loop_summary.json)  ->  python tools/view_clip.py {out}/closed_loop.npz --open")


if __name__ == "__main__":
    main()
