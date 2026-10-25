#!/usr/bin/env python3
"""
build_vae_dataset.py - start everything and build the training set for the motion VAE, in one command.

  python tools/build_vae_dataset.py                          # ~10 h of motion into data/vae  (make vae-data does the same)
  python tools/build_vae_dataset.py --hours 30               # more, mostly more variants (see the honest note below)
  python tools/build_vae_dataset.py --hours 0.3 --out /tmp/v # a quick smoke run (about a minute)
  python tools/build_vae_dataset.py --no-cc                  # leave out the ACCAD/OSU + 100STYLE sources: engine clips and your stored motions only
  python tools/build_vae_dataset.py --no-engine              # no compiler / engine: memory only
  python tools/view_clip.py data/vae/preview.npz --open      # look at a random sample of what it made

What it does, in order (each step prints what it did and how long it took)
  1. reference motions  the grader's yardstick: the CMU examples (data/real_motions.npz, downloaded the first time) plus data/cc_motions.npz if it exists (tools/get_cc_motions.py: ACCAD/OSU and
                        100STYLE). CMU is the yardstick ONLY and is never a training source (its licence forbids redistributing it, even converted; docs/CREDITS.md). The CC BY sets are both.
  2. memory             opens data/memory (tools/motion_memory.py) and, if it holds none of them yet, seeds it with every ACCAD/OSU and 100STYLE example (kind known, origin accad / 100style)
  3. engine clips       builds the engine, generates --engine-clips walking clips, drops the ones the engine's own verdict rejects, stores the ones that pass the grader gates
  4. split              (exact duplicate clips are dropped first) by SOURCE (ACCAD: by performer; 100STYLE: by style, its one performer acts all of them; engine/user: by clip) into train and validation: a variant never lands in a different split than its source
  5. variants           the training set is filled up to --hours with variants of the training sources (tools/motion_vary.py: mirror, time warp, amplitude, gaze, drift, tremor,
                        heavy-tailed strength, occasional "uncommon" twist). Every variant must pass the grader (gates + fit to its action) and differ from its source and its
                        siblings; variants of ENGINE clips must also pass the engine's feasibility verdict. Names are balanced by count**--balance (0.5: square root).
  6. write              train.jsonl / val.jsonl (variable length clips), train_windows.npz / val_windows.npz (fixed windows for the VAE), stats.json (per-channel mean/std),
                        manifest.json, report.txt, preview.npz, README.txt

Output fields per clip (jsonl): id, action, source (accad/100style/engine/user), role (original/variant), origin_id, ops, strength, uncommon, engine_ok (true / null = not checkable), fit, dt, frames [F][30].
Channel layout (30): see CHANNELS in this file and stats.json. Flags (contact / on) are 0/1: do not normalise them (stats.json lists them as `binary`); yaw channels are angles.

HONEST NOTES (also in report.txt, with the numbers of this run)
  * The engine makes walking (gen_dataset.py) and a scripted set of other actions (gen_actions.py: stand, turn, balance, crouch, pickup, wave/point, stretch, drink, jump), kept only if the
    grader also calls them that action. Jump cannot be replay-checked (engine_ok = null; the engine ran it live without falling). No sit, run, kick, dance or get-up: the engine cannot do those yet.
    Everything else in the set (sit, run, ...) is derived from the real ACCAD/OSU and 100STYLE clips, which the ENGINE REJECTS (the report
    prints the measured share: balance lost, planted feet slide; a retargeting / contact-flag problem of bvh_to_clip.py, not of the grader). They are graded by the
    mocap-based grader, not verified by the engine. engine_ok in each record says which is which. With --no-cc the set is what the engine can vouch for, plus your stored motions: walking from the engine, other actions only if you add them.
  * Variants are the same motion performed differently. 10 h built from 1 h of sources is 1 h of information at ten times the weight. For an 80M-parameter model the report
    prints the number of training values per parameter: expect memorisation unless you use dropout / KL weight / early stopping on the validation split, and treat
    the validation loss (original clips from performers it never saw) as the only honest measure.
  * The graders' limits apply (tools/motion_grader.py, motion_roll.py, motion_vary.py): variants are only as good as "inside the real range of their action".
"""
import argparse, collections, json, os, subprocess, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import motion_features as mf, motion_grader as G, motion_memory as MM, motion_vary as V

CHANNELS = ["pelvis_x", "pelvis_y", "pelvis_z", "pelvis_yaw", "footL_x", "footL_y", "footL_z", "footL_yaw", "footL_contact", "footR_x", "footR_y", "footR_z", "footR_yaw", "footR_contact",
            "handL_x", "handL_y", "handL_z", "handL_on", "handR_x", "handR_y", "handR_z", "handR_on", "look_x", "look_y", "look_z", "look_on", "torso_x", "torso_y", "torso_z", "torso_on"]
BINARY = [i for i, c in enumerate(CHANNELS) if c.endswith("contact") or c.endswith("_on")]
ANGLES = [i for i, c in enumerate(CHANNELS) if c.endswith("yaw")]
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def windows(clip, W, stride):
    """-> (windows [N,W,30], valid frames per window). A clip shorter than W but at least W/2 long becomes one window padded by repeating its last frame (valid < W: mask it); shorter: None."""
    F = len(clip)
    if F >= W:
        w = np.stack([clip[s:s + W] for s in range(0, F - W + 1, stride)]); return w, np.full(len(w), W, np.int16)
    if F >= W // 2:
        return np.concatenate([clip, np.repeat(clip[-1:], W - F, 0)])[None], np.array([F], np.int16)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=10.0, help="training motion to build, in hours (originals + variants); default 10")
    ap.add_argument("--out", default="data/vae"); ap.add_argument("--memory", default=MM.DEFAULT_DIR); ap.add_argument("--real", default=G.REAL_DEFAULT)
    ap.add_argument("--engine-clips", type=int, default=1500, help="walking clips to ask the engine for (duplicates and infeasible ones are dropped)"); ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--engine-skills", type=int, default=150, help="clips PER SKILL to ask tools/gen_actions.py for (stand, turn, balance, crouch, pickup, wave, point, stretch, drink, jump); only clips the engine accepts AND the grader recognises as that action are kept; 0 = walking only")
    ap.add_argument("--cc", default=None, help="the ACCAD/OSU + 100STYLE set to train on and grade against (default: data/cc_motions.npz next to --real, if it exists)")
    ap.add_argument("--no-cc", action="store_true", help="do not seed the memory with the ACCAD/OSU + 100STYLE examples (tools/get_cc_motions.py): engine clips and your stored motions only")
    ap.add_argument("--include-cmu", action="store_true", help=argparse.SUPPRESS)      # removed: CMU is not a training source any more; kept only to explain why
    ap.add_argument("--val", type=float, default=0.08, help="share of the source frames held out for validation (ACCAD performers and 100STYLE styles whole, for all actions at once; engine/user: single clips)")
    ap.add_argument("--window", type=int, default=64); ap.add_argument("--stride", type=int, default=32); ap.add_argument("--no-windows", action="store_true"); ap.add_argument("--no-jsonl", action="store_true")
    ap.add_argument("--balance", type=float, default=0.5, help="a name's share of the set ~ (its number of sources) ** balance; 1 = as in the sources, 0 = every name equal")
    ap.add_argument("--uncommon", type=float, default=0.12); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--try-factor", type=float, default=8.0, help="give up on a name after this many tries per wanted variant")
    a = ap.parse_args()
    if a.include_cmu:
        sys.exit("--include-cmu was removed: CMU motion capture may not be redistributed, even converted, so it is the grader's yardstick only. Train on ACCAD/OSU + 100STYLE (python tools/get_cc_motions.py).")
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

    # 1 ------------------------------------------------------------------------------------------------------------ real data + grader
    zr, RF = G.load_real(a.real, cc=a.cc); prof = G.Profiles(RF, zr["action"]); dt = float(zr["dt"])
    srcs = collections.Counter(str(x) for x in zr["source"]) if "source" in zr.files else collections.Counter({"cmu": len(zr["offsets"]) - 1})
    log(f"reference motions: {len(zr['offsets']) - 1} clips ({', '.join(f'{k} {v}' for k, v in sorted(srcs.items()))}), {len(set(map(str, zr['action'])))} actions, {len(zr['actions']) * dt / 60:.0f} min (grader yardstick; only accad / 100style are training sources)")
    if not any(k in srcs for k in MM.TRAINING_SOURCES) and not a.no_cc:
        log("  no ACCAD/OSU or 100STYLE motion found (data/cc_motions.npz): python tools/get_cc_motions.py builds it; without it the training set is engine clips + stored motions")

    # 2 ------------------------------------------------------------------------------------------------------------ memory
    mem = MM.Memory(a.memory)
    if not a.no_cc and any(k in srcs for k in MM.TRAINING_SOURCES) and not any(e.get("origin") in MM.TRAINING_SOURCES for e in mem.entries):
        r = MM.seed_from_real(mem, zr, RF, prof, rng=np.random.default_rng(a.seed)); log(f"memory seeded with the ACCAD/OSU + 100STYLE examples: {r['stored']} stored, {r['rejected']} rejected by the gates")
    else:
        log(f"memory {a.memory}: {len(mem.entries)} motions already stored")

    # 3 ------------------------------------------------------------------------------------------------------------ engine
    lib = chk = None
    if not a.no_engine:
        try:
            import gen_dataset as gd; lib = gd.Lib(); chk = gd.new_standing(lib)
        except Exception as ex:
            log(f"engine not available ({str(ex)[:120]}): continuing without it (engine_ok stays null)")
    if chk is not None and a.engine_clips > 0:
        edir = os.path.join(a.out, "engine"); r = subprocess.run([sys.executable, os.path.join(HERE, "gen_dataset.py"), "--n", str(a.engine_clips), "--out", edir, "--seed", str(a.seed)], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(os.path.join(edir, "clips.npz")):
            ez = np.load(os.path.join(edir, "clips.npz")); ec = G.split_clips(ez); st = collections.Counter()
            for k, c in enumerate(ec):
                s, _, _ = mem.add(c, float(ez["dt"]), "walk", prof, RF, source=f"engine:{a.seed}#{k}", origin="engine", kind="known", feasible=True); st[s] += 1
            log(f"engine: {len(ec)} feasible clips generated, {st['stored']} stored in memory, {st['reinforced']} near-duplicates merged, {st['rejected']} rejected by the gates")
        else:
            log("engine clip generation failed: " + (r.stderr or r.stdout)[-300:])
    if chk is not None and a.engine_skills > 0:
        sdir = os.path.join(a.out, "engine_skills"); r = subprocess.run([sys.executable, os.path.join(HERE, "gen_actions.py"), "--n", str(a.engine_skills), "--out", sdir, "--seed", str(a.seed), "--real", a.real, "--no-grade"], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(os.path.join(sdir, "clips.npz")):
            sz = np.load(sdir + "/clips.npz", allow_pickle=True); sc_ = G.split_clips(sz); st = collections.Counter(); per = collections.defaultdict(collections.Counter)
            for k, c in enumerate(sc_):
                act = str(sz["action"][k]); rep = str(sz["replay"][k]); g = G.grade_clip(c, float(sz["dt"]), act, prof)
                if not (g["recognised"] and g["score"] is not None and g["score"] >= G.PASS_SCORE):       # the engine accepted it, but a real-motion grader must also call it that action
                    st["not graded as the action"] += 1; per[act]["graded out"] += 1; continue
                s_, _, _ = mem.add(c, float(sz["dt"]), act, prof, RF, source=f"engine_skill:{sz['skill'][k]}:{a.seed}#{k}", origin="engine", kind="known", feasible=None if rep == "live" else True, meta=dict(engine_replay=rep, skill=str(sz["skill"][k])))
                st[s_] += 1; per[act][s_] += 1
            log(f"engine skills: {len(sc_)} engine-accepted clips, {st['stored']} stored in memory, {st['not graded as the action']} dropped because the grader does not call them that action, {st['reinforced']} near-duplicates merged, {st['rejected']} rejected by the gates")
            for act in sorted(per):
                log(f"    {act:9s} stored {per[act]['stored']:4d}, graded out {per[act]['graded out']:4d}, rejected {per[act]['rejected']:3d}")
        else:
            log("engine skill generation failed: " + (r.stderr or r.stdout)[-300:])
    mem.save()
    keep = [i for i, e in enumerate(mem.entries) if e.get("origin") != "cmu" and not (a.no_cc and e.get("origin") in MM.TRAINING_SOURCES)]     # a memory seeded before CMU was removed may still hold origin "cmu": never trained on
    cc_keep = [i for i in keep if mem.entries[i].get("origin") in MM.TRAINING_SOURCES]                    # a mocap set can hold one motion twice under different performer ids (CMU did: 77/139, 91/105, 93/103): keep one copy,
    dup = {cc_keep[j] for _, j in V.duplicate_pairs([mem.clips[i] for i in cc_keep])} if len(cc_keep) > 1 else set()                     # otherwise a "held-out performer" can have an identical twin in training
    if dup:
        keep = [i for i in keep if i not in dup]; log(f"dropped {len(dup)} ACCAD/100STYLE clips that are exact duplicates of another performer's clip (the copy stays out of both splits)")
    if not keep:
        sys.exit("nothing to build from: the memory is empty (add motions to the memory with tools/motion_memory.py add, leave out --no-engine, or build the ACCAD/OSU + 100STYLE set with tools/get_cc_motions.py and leave out --no-cc)")
    name = {i: mem.entries[i]["name"] for i in keep}; origin = {i: mem.entries[i].get("origin", "user") for i in keep}; nfr = {i: len(mem.clips[i]) for i in keep}

    # 4 ------------------------------------------------------------------------------------------------------------ split by source group
    PERF = MM.TRAINING_SOURCES                           # mocap sets are split by group: ACCAD by performer, 100STYLE by style (its one performer acts all 100, so the style is the unit that must not leak)
    group = {i: (str(mem.entries[i].get("performer", i)) if origin[i] in PERF else f"{origin[i]}:{i}") for i in keep}; val = set()
    cmu = [i for i in keep if origin[i] in PERF]
    if cmu:                                              # whole PERFORMERS / STYLES go to validation, for every action at once (a group is never in both splits)
        perfs = collections.OrderedDict()
        for i in cmu:
            perfs.setdefault(group[i], []).append(i)
        pk = list(perfs); tot = sum(nfr[i] for i in cmu); ntot = collections.Counter()
        for i in cmu:
            ntot[name[i]] += nfr[i]
        best = None; rs = np.random.default_rng(a.seed + 17)
        for _ in range(400):                              # random assignments; keep the one where every action has some validation but none loses much of its training data
            chosen, got = [], 0
            for k in rs.permutation(len(pk)):
                if got >= a.val * tot:
                    break
                chosen.append(pk[k]); got += sum(nfr[i] for i in perfs[pk[k]])
            vn = collections.Counter()
            for g in chosen:
                for i in perfs[g]:
                    vn[name[i]] += nfr[i]
            cost = sum((vn[n] == 0) * 1.0 + max(0.0, vn[n] / ntot[n] - 0.25) * 10 for n in ntot)
            if best is None or cost < best[0]:
                best = (cost, chosen)
        for g in best[1]:
            val.update(perfs[g])
    for n in sorted(set(name.values())):                  # engine / user clips: single sources, held out per name
        ids = [i for i in keep if name[i] == n and origin[i] not in PERF]
        if len(ids) < 2:
            continue
        want = a.val * sum(nfr[i] for i in ids); got = 0
        for i in rng.permutation(ids)[:-1]:
            if got >= want:
                break
            val.add(int(i)); got += nfr[int(i)]
    train = [i for i in keep if i not in val]; vlist = [i for i in keep if i in val]
    novl = sorted(set(name.values()) - set(name[i] for i in vlist))
    log(f"split: {len(train)} training sources, {len(vlist)} validation sources ({sum(nfr[i] for i in vlist) * dt / 60:.0f} min), by performer (ACCAD) / style (100STYLE) / by clip otherwise" + (f"; no validation clips for: {', '.join(novl)}" if novl else ""))

    # 5 ------------------------------------------------------------------------------------------------------------ variants
    T = a.hours * 3600 / dt; orig_fr = sum(nfr[i] for i in train); records = []                                        # records: dict(clip, action, source, role, origin_id, ...)
    for i in train:
        records.append(dict(clip=mem.clips[i], action=name[i], source=origin[i], role="original", origin_id=mem.entries[i]["id"], ops=[], strength=None, uncommon=False,
                            engine_ok=True if origin[i] == "engine" and mem.entries[i].get("engine_replay") in (None, "ok", "own_start") else None, fit=mem.entries[i]["best_known_fit"], group=group[i]))
    names = sorted(set(name[i] for i in train)); cnt = {n: sum(1 for i in train if name[i] == n) for n in names}; ofr = {n: sum(nfr[i] for i in train if name[i] == n) for n in names}
    w = {n: cnt[n] ** a.balance for n in names}; sw = sum(w.values()); tgt = {n: T * w[n] / sw for n in names}; quota = {n: max(0.0, tgt[n] - ofr[n]) for n in names}
    need = max(0.0, T - orig_fr); qs = sum(quota.values())
    quota = {n: (q * need / qs if qs > 0 else 0.0) for n, q in quota.items()}
    if need <= 0:
        log(f"the originals alone ({orig_fr * dt / 3600:.1f} h) already reach --hours {a.hours}: no variants made")
    stats = {n: collections.Counter() for n in names}; opstat = collections.Counter(); dists = []; unc_kept = 0; tv = time.time()
    for n in names:
        q = quota[n]
        if q <= 0:
            continue
        src = [i for i in train if name[i] == n]; wt = np.array([0.5 + MM.Memory.effective(mem.entries[i]) for i in src]); wt /= wt.sum(); ctx = {}; sib = {i: collections.deque(maxlen=12) for i in src}
        r_n = np.random.default_rng(a.seed * 1000003 + zlib.crc32(n.encode())); got = 0.0; tries = 0; avg = np.mean([nfr[i] for i in src]); cap = int(a.try_factor * max(20, q / avg))
        while got < q and tries < cap:
            tries += 1; i = src[int(r_n.choice(len(src), p=wt))]
            if i not in ctx:
                ctx[i] = mem._context(i, prof)
            act, fit0, r0 = ctx[i]; b, info = V.make_variant(mem.clips[i], dt, r_n, a.uncommon); ok, why, v, fit, d = V.check_variant(b, dt, act, prof, fit0, r0, sib[i])
            eng = None
            if ok and origin[i] == "engine" and chk is not None and mem.entries[i].get("engine_replay") in (None, "ok"):
                eng = bool(chk.evaluate(b, dt)["feasible"])
                if not eng:
                    ok, why = False, "engine verdict"
            if not ok:
                stats[n][why.split(" (")[0].split(" <")[0].split(" ")[0] if why != "engine verdict" else why] += 1; continue
            sib[i].append(V.clip_rep(b)); got += len(b); stats[n]["kept"] += 1; dists.append(d); unc_kept += info["uncommon"]
            for o in info["ops"]:
                opstat[o] += 1
            records.append(dict(clip=b, action=n, source=origin[i], role="variant", origin_id=mem.entries[i]["id"], ops=info["ops"], strength=info["strength"], uncommon=info["uncommon"],
                                engine_ok=eng, fit=round(float(fit), 1), group=group[i]))
        if got < 0.9 * q:
            log(f"  '{n}': wanted {q * dt / 60:.0f} min of variants, made {got * dt / 60:.0f} (gave up after {tries} tries: the acceptance rate is low for this action)")
    nvar = sum(1 for r in records if r["role"] == "variant"); log(f"variants: {nvar} kept in {time.time() - tv:.0f}s; training set = {len(records)} clips, {sum(len(r['clip']) for r in records) * dt / 3600:.2f} h")

    # 6 ------------------------------------------------------------------------------------------------------------ write
    vrecs = [dict(clip=mem.clips[i], action=name[i], source=origin[i], role="original", origin_id=mem.entries[i]["id"], ops=[], strength=None, uncommon=False,
                  engine_ok=True if origin[i] == "engine" and mem.entries[i].get("engine_replay") in (None, "ok", "own_start") else None, fit=mem.entries[i]["best_known_fit"], group=group[i]) for i in vlist]
    allnames = sorted(set(r["action"] for r in records + vrecs)); aid = {n: k for k, n in enumerate(allnames)}
    out = lambda f: os.path.join(a.out, f)
    if not a.no_jsonl:
        for fn, rs in (("train.jsonl", records), ("val.jsonl", vrecs)):
            with open(out(fn), "w") as f:
                for k, r in enumerate(rs):
                    f.write(json.dumps(dict(id=f"{fn[:-6]}-{k}", action=r["action"], source=r["source"], role=r["role"], origin_id=r["origin_id"], ops=r["ops"], strength=r["strength"],
                                            uncommon=r["uncommon"], engine_ok=r["engine_ok"], fit=r["fit"], dt=round(dt, 6), frames=np.round(r["clip"], 4).tolist())) + "\n")
        log(f"wrote {out('train.jsonl')} ({os.path.getsize(out('train.jsonl')) / 1e6:.0f} MB) and val.jsonl")
    nwin = {}
    if not a.no_windows:
        for fn, rs in (("train_windows.npz", records), ("val_windows.npz", vrecs)):
            Ws, lab, src, rol, gid, vl = [], [], [], [], [], []; skipped = 0
            for r in rs:
                w_ = windows(r["clip"], a.window, a.stride)
                if w_ is None:
                    skipped += 1; continue
                w_, v_ = w_; vl.append(v_); Ws.append(w_.astype(np.float32)); lab += [aid[r["action"]]] * len(w_); src += [r["source"]] * len(w_); rol += [r["role"]] * len(w_); gid += [r["origin_id"]] * len(w_)
            X = np.concatenate(Ws) if Ws else np.zeros((0, a.window, 30), np.float32); nwin[fn] = (len(X), skipped)
            np.savez(out(fn), windows=X, valid=(np.concatenate(vl) if vl else np.zeros(0, np.int16)), action=np.array(lab, np.int16), action_names=np.array(allnames), source=np.array(src), role=np.array(rol), origin_id=np.array(gid, np.int32), dt=np.float32(dt))
        log(f"wrote train_windows.npz ({nwin['train_windows.npz'][0]} windows of {a.window} frames, {nwin['train_windows.npz'][1]} clips shorter than half a window skipped; clips between half and one window are padded: use `valid` to mask) and val_windows.npz ({nwin['val_windows.npz'][0]})")
    allf = np.concatenate([r["clip"] for r in records]).astype(np.float64); mean, std = allf.mean(0), allf.std(0)
    json.dump(dict(channels=CHANNELS, mean=mean.tolist(), std=np.maximum(std, 1e-3).tolist(), binary=BINARY, angles=ANGLES, dt=dt, note="normalise (x-mean)/std on the continuous channels only; leave `binary` as 0/1"), open(out("stats.json"), "w"), indent=1)

    # report ---------------------------------------------------------------------------------------------------------------------------------------------------
    L = []; P = L.append; trf = sum(len(r["clip"]) for r in records); vaf = sum(len(r["clip"]) for r in vrecs)
    P(f"VAE dataset in {a.out}   (built in {time.time() - T0:.0f}s, seed {a.seed})")
    P(f"train: {len(records)} clips, {trf * dt / 3600:.2f} h, {trf} frames   |   validation: {len(vrecs)} original clips, {vaf * dt / 60:.0f} min   |   {len(allnames)} actions")
    P(f"{'action':14s} {'orig':>5s} {'var':>6s} {'hours':>6s}  sources           kept  rejected because")
    for n in allnames:
        rs = [r for r in records if r["action"] == n]; srcs = collections.Counter(r["source"] for r in rs); st = stats.get(n, collections.Counter()); rej = ", ".join(f"{k} {v}" for k, v in st.most_common() if k != "kept")
        P(f"{n:14s} {sum(r['role'] == 'original' for r in rs):5d} {sum(r['role'] == 'variant' for r in rs):6d} {sum(len(r['clip']) for r in rs) * dt / 3600:6.2f}  {','.join(f'{k}:{v}' for k, v in srcs.items()):18s}{st['kept']:5d}  {rej[:80]}")
    if dists:
        P(f"\nvariants: median distance to their source/siblings {np.median(dists):.3f} m (RMS, tools/motion_vary.py), {100 * unc_kept / max(1, nvar):.0f}% 'uncommon'; operations in kept variants: " + ", ".join(f"{k} {100 * v / nvar:.0f}%" for k, v in opstat.most_common()))
    # grade a sample of what was written, per action, with the mocap-based grader
    P("\nGrader check of a random sample (<= 40 clips per action; variants were filtered by this same grader, so for them it only confirms the filter):")
    for n in allnames:
        if n not in prof.p:
            continue
        line = []
        for role in ("original", "variant"):
            rs = [r["clip"] for r in records if r["action"] == n and r["role"] == role]
            if len(rs) >= 5:
                pick = [rs[k] for k in rng.choice(len(rs), min(40, len(rs)), replace=False)]; g = G.grade_dataset(pick, dt, n, prof); line.append(f"{role}s {g['verdict']} (mean {g['mean']:.0f})")
        if line:
            P(f"  {n:14s} " + "   ".join(line))
    eo = [r for r in records if r["source"] == "engine"]; co = [r for r in records if r["source"] in PERF]
    P(f"\nengine verification: {len(eo)} training clips come from the engine (originals and variants all passed its feasibility verdict: {sum(1 for r in eo if r['engine_ok'])} of {len(eo)})" if chk is not None else "\nengine verification: engine not available in this run")
    if co and chk is not None:
        co0 = [r for r in co if r["role"] == "original"]; sample = [co0[k]["clip"] for k in rng.choice(len(co0), min(150, len(co0)), replace=False)]
        feas = sum(bool(chk.evaluate(c, dt)["feasible"]) for c in sample)
        P(f"  {len(co)} training clips derive from real ACCAD/100STYLE clips: {feas} of {len(sample)} sampled originals pass the engine's feasibility verdict -> they are graded by the mocap-based grader only (engine_ok = null)")
    P(f"\nsize for the model: {trf * 30 / 1e6:.1f} M training values in {trf} frames; an 80M-parameter VAE sees {trf * 30 / 80e6:.2f} values per parameter. Independent information is only the sources: "
      f"{sum(nfr[i] for i in train) * dt / 3600:.2f} h of training originals ({len(train)} clips) from {len(set(group[i] for i in train))} source groups; variants are repeats with variation, not new motion.")
    if nwin:
        P(f"windows: {nwin['train_windows.npz'][0]} train / {nwin['val_windows.npz'][0]} val of {a.window} frames ({a.window * dt:.1f} s), stride {a.stride}")
    P("\nfiles: train.jsonl val.jsonl train_windows.npz val_windows.npz stats.json manifest.json report.txt preview.npz README.txt")
    open(out("report.txt"), "w").write("\n".join(L) + "\n"); print("\n" + "\n".join(L))
    json.dump(dict(args=vars(a), seconds=round(time.time() - T0, 1), train_clips=len(records), val_clips=len(vrecs), train_hours=trf * dt / 3600, actions=allnames, memory=a.memory,
                   memory_entries=len(mem.entries), engine_available=chk is not None), open(out("manifest.json"), "w"), indent=1, default=str)
    pick = rng.choice(len(records), min(24, len(records)), replace=False); pc = [records[k] for k in pick]
    np.savez_compressed(out("preview.npz"), actions=np.concatenate([r["clip"] for r in pc]).astype(np.float32), offsets=np.cumsum([0] + [len(r["clip"]) for r in pc]), dt=np.float32(dt),
                        action=np.array([r["action"] for r in pc]), labels=np.array([f"{r['role']} of #{r['origin_id']}" for r in pc]), description=np.array([",".join(r["ops"]) or "original" for r in pc]),
                        style=np.zeros((len(pc), 1), np.float32), style_names=np.array(["none"]))
    open(out("README.txt"), "w").write(f"""Motion VAE dataset (built by tools/build_vae_dataset.py, seed {a.seed})
train.jsonl / val.jsonl   one clip per line: id, action, source (accad/100style/engine/user), role (original/variant), origin_id (memory id of the source), ops, strength, uncommon, engine_ok, fit, dt, frames [F][30]
train_windows.npz / val_windows.npz   windows [N,{a.window},30] float32, valid (frames that are real: shorter clips are padded with their last frame, mask the rest), action (index into action_names), source, role, origin_id, dt
stats.json   per-channel mean/std for the 30 channels, which are binary (leave as 0/1) and which are angles
validation holds ORIGINAL clips only, from sources that have no variant in train; for ACCAD clips the performers (for 100STYLE the styles) are never in train at all (any action).
preview.npz  24 random training clips:  python tools/view_clip.py {a.out}/preview.npz --open
report.txt   the numbers of this run, including what the engine could and could not verify
""")
    log(f"done: look at it with  python tools/view_clip.py {out('preview.npz')} --open")


if __name__ == "__main__":
    main()
