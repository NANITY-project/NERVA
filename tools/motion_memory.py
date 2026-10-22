#!/usr/bin/env python3
"""
motion_memory.py - "muscle memory": a store of motions that are VALID (a body can make them) and USEFUL to keep, with variants made on demand so the same clip is never
replayed the same way. It holds the whole motion collection (the CMU examples and the engine's clips), plus the motions that are not one of the grader's labelled actions
(no "wave", "open the fridge", "scratch head" in the CMU label set).

  python tools/motion_memory.py seed                                                 # fill the memory with the real CMU examples (647 clips, named by action, kind 'known')
  python tools/motion_memory.py add data/clips.npz --clip 4 --name wave_hello        # keep clip 4 under a name (several examples may share a name)
  python tools/motion_memory.py add-segments data/seg.npz                            # keep every segment motion_roll.py called 'novel' (auto-named unlabelled_001 ...)
  python tools/motion_memory.py list                                                 # what is stored, how strong each memory is
  python tools/motion_memory.py recall --name wave_hello                             # all examples of a name, strongest first
  python tools/motion_memory.py recall --like data/clips.npz --clip 9 --k 3          # the 3 stored motions most like that clip (feature distance, milliseconds)
  python tools/motion_memory.py recall --name wave_hello --vary --out data/v.npz     # recall AND perform it differently this time (a fresh variant, never a repeat); counts as a use
  python tools/motion_memory.py vary --name wave_hello --n 12 --out data/v.npz       # 12 graded variants of every example of that name (tools/motion_vary.py says how)
  python tools/motion_memory.py rename unlabelled_001 scratch_head                   # give an auto-named motion a real name
  python tools/motion_memory.py grade data/clips.npz --action wave_hello             # grade a dataset against the real CMU examples AND your stored examples (>= 4)
  python tools/motion_memory.py export data/memory.jsonl                             # {name, kind, origin, strength, dt, frames} per stored motion
  python tools/motion_memory.py prune                                                # lists memories that faded away; --yes deletes them (never automatic)
  python tools/motion_memory.py add ... --engine-check                               # also require the engine's own feasibility verdict

How it behaves like a memory, not a folder
  * ADMISSION: a motion is stored only if it is physically plausible (the grader's jitter / contact gates pass against the limits of the closest known action) and,
    with --engine-check, the engine accepts it. Limit: running/jumping are broadband, so their limit is loose; mild noise (~5-12 mm) on a clip whose nearest action is
    one of those can get in. Gross jitter (cm) is caught. A near-duplicate of something already stored under the same name is not stored twice: it REINFORCES the existing one.
  * KIND: `known` = a labelled example (the CMU seeds, the engine's walks); `novel` = fits no known action well (best fit over all 24 < 60); `variant:<action>` = a version of
    a known action, kept because you named it. ORIGIN says where it came from: cmu, engine, user.
  * STRENGTH: new memories start at 1.0, every use adds 0.5 (max 3.0), and strength halves for every 14 days without use. Recall ranks by similarity divided by
    (0.5 + strength), so what you really use comes first, rarely used things fade to the back and, after ~5 weeks unused, become `prune` candidates. Nothing is deleted without --yes.
  * NEVER THE SAME TWICE: serve() makes a variant with tools/motion_vary.py (smooth speed / amplitude / gaze / drift / tremor / mirror changes, heavy-tailed strength, an
    occasional "uncommon" twist), checks it with the grader, and keeps the seeds of the last 8 it served so the next one differs from those as well as from the original.
  * SPEED: every stored motion keeps its feature vector next to the frames; recall is one numpy distance over that matrix, no engine, no re-measuring.
  * PROMOTION: once a name has >= 4 examples, `grade` (and memory_profiles() in Python) treats it as an action of its own: its profile is built from your examples
    next to the 24 CMU ones, so a dataset can be graded as "wave_hello". Honest limit: that profile is built from the data you chose, not from independent real
    captures, so it says "similar to what you stored", not "a correct wave". Trust it as little as the 4 examples allow.
Storage: DIR/index.json (names, strengths, history) + DIR/clips.npz (frames, offsets, features). Default DIR = data/memory. All clips must share one dt (30 fps).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_features as mf, motion_grader as G, motion_vary as V

DEFAULT_DIR = "data/memory"
HALF_LIFE_DAYS, USE_GAIN, MAX_STRENGTH, FORGET_BELOW = 14.0, 0.5, 3.0, 0.15
DUP_DIST, NOVEL_FIT, MIN_PROMOTE, MIN_SECONDS, KEEP_SEEDS = 0.05, 60.0, 4, 1.0, 8
DAY = 86400.0


class Memory:
    def __init__(self, path=DEFAULT_DIR):
        self.dir = path; self.entries = []; self.clips = []; self.F = np.zeros((0, len(mf.NAMES))); self.dt = None
        ip, cp = os.path.join(path, "index.json"), os.path.join(path, "clips.npz")
        if os.path.exists(ip) and os.path.exists(cp):
            idx = json.load(open(ip)); z = np.load(cp); self.entries = idx["entries"]; self.dt = idx.get("dt"); A, off, self.F = z["actions"], z["offsets"], z["features"]
            self.clips = [A[off[i]:off[i + 1]] for i in range(len(off) - 1)]
            assert len(self.entries) == len(self.clips) == len(self.F), "memory files are out of step with each other"
        self._scale = None

    # ---------------------------------------------------------------------------------------------------- persistence
    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        A = np.concatenate(self.clips).astype(np.float32) if self.clips else np.zeros((0, 30), np.float32); off = np.cumsum([0] + [len(c) for c in self.clips]).astype(np.int64)
        tmp = os.path.join(self.dir, "clips.tmp.npz"); np.savez_compressed(tmp, actions=A, offsets=off, features=self.F); os.replace(tmp, os.path.join(self.dir, "clips.npz"))
        tmp = os.path.join(self.dir, "index.tmp.json"); json.dump(dict(dt=self.dt, entries=self.entries), open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(self.dir, "index.json"))

    def clip(self, i):
        return self.clips[i]

    # ---------------------------------------------------------------------------------------------------- strength
    @staticmethod
    def effective(e, now=None):
        now = time.time() if now is None else now
        return float(e["strength"] * 0.5 ** (max(0.0, now - e["last_used"]) / (HALF_LIFE_DAYS * DAY)))

    def touch(self, i, now=None):
        now = time.time() if now is None else now; e = self.entries[i]
        e["strength"] = min(MAX_STRENGTH, self.effective(e, now) + USE_GAIN); e["last_used"] = now; e["uses"] += 1

    # ---------------------------------------------------------------------------------------------------- distance
    def scale(self, real_F):
        """Per-feature scale = spread of the real examples (10-90%) + the feature's floor: a distance of 1 = 'as different as two real performers'."""
        if self._scale is None:
            q = np.nanpercentile(real_F, [10, 90], axis=0); self._scale = np.nan_to_num(q[1] - q[0]) + mf.FLOOR
        return self._scale

    def distance(self, v, W, scale):
        """Mean capped |difference| in scale units between v and each row of W over the features both have (gates excluded)."""
        use = np.array([n not in mf.GATES for n in mf.NAMES]); D = np.abs(W - v) / scale; D = np.minimum(D, 5.0); D[:, ~use] = np.nan
        with np.errstate(all="ignore"):
            d = np.nanmean(D, axis=1)
        return np.where(np.isfinite(d), d, np.inf)

    # ---------------------------------------------------------------------------------------------------- add
    def add(self, actions, dt, name, prof, real_F, source="", note="", feasible=None, now=None, origin="user", kind=None, meta=None, allow_dup=False):
        """Try to store a clip. Returns (status, message, index): status in stored / reinforced / rejected. kind=None: decided from the grade (novel / variant:<action>)."""
        now = time.time() if now is None else now; a = np.ascontiguousarray(actions, np.float32)
        if self.dt is not None and abs(self.dt - dt) > 1e-3:
            return "rejected", f"frame time {dt:.4f} s differs from the memory's {self.dt:.4f} s", None
        if len(a) * dt < MIN_SECONDS:
            return "rejected", f"only {len(a) * dt:.2f} s long (minimum {MIN_SECONDS:.0f} s)", None
        v = mf.extract(a, dt)
        if np.isfinite(v).sum() < 0.6 * len(v):
            return "rejected", "too few body measures could be computed from this clip", None
        fits = {k: G.grade_features(v, k, prof, use_match=False) for k in prof.p}; fits = {k: f for k, f in fits.items() if f["fit"] is not None}
        ba = max(fits, key=lambda k: fits[k]["fit"]) if fits else None; bf = fits[ba]["fit"] if ba else 0.0
        if ba and fits[ba]["gate"] < 0.99:      # the gates (jitter, planted-foot mismatch) judged against the closest known action's own limits
            bad = [i["name"] for i in fits[ba]["items"] if i["name"] in mf.GATES and i["score"] < 0.99]
            return "rejected", f"not a motion a body can make ({', '.join(bad) or 'gate'} outside what real {ba} motion shows)", None
        if feasible is False:
            return "rejected", "the engine's feasibility verdict rejects it", None
        sc = self.scale(real_F)
        same = [i for i, e in enumerate(self.entries) if e["name"] == name]
        if same and not allow_dup:           # allow_dup: bulk loading of a labelled collection keeps every clip (windows of one trial look alike in the features but are different frames)
            d = self.distance(v, self.F[same], sc); j = int(np.argmin(d))
            if d[j] < DUP_DIST:
                i = same[j]; self.touch(i, now); return "reinforced", f"near-duplicate of '{name}' #{self.entries[i]['id']} (distance {d[j]:.3f}): that memory got stronger instead", i
        kind = kind or ("novel" if bf < NOVEL_FIT else f"variant:{ba}")
        nid = 1 + max([e["id"] for e in self.entries] + [0])
        e = dict(id=nid, name=name, kind=kind, origin=origin, best_known=ba, best_known_fit=round(float(bf), 1), strength=1.0, uses=0, created=now, last_used=now,
                 source=source, note=note, feasible=feasible, frames=len(a), seconds=round(len(a) * dt, 2), seeds=[])
        e.update(meta or {}); self.entries.append(e); self.clips.append(a); self.F = np.vstack([self.F, v[None]]); self.dt = float(dt)
        return "stored", f"#{nid} '{name}' ({kind}; closest known action: {ba} {bf:.0f})", len(self.entries) - 1

    # ---------------------------------------------------------------------------------------------------- recall
    def recall(self, name=None, like=None, k=5, real_F=None, now=None):
        """[(index, distance or None, effective strength)] best first. name: exact (or prefix) name match. like: feature vector of the clip to resemble."""
        now = time.time() if now is None else now
        if not self.entries:
            return []
        if like is not None:
            d = self.distance(like, self.F, self.scale(real_F)); cand = [i for i in range(len(self.entries)) if np.isfinite(d[i])]
            if name:
                cand = [i for i in cand if self.entries[i]["name"].startswith(name)]
            out = [(i, float(d[i]), self.effective(self.entries[i], now)) for i in cand]; out.sort(key=lambda t: t[1] / (0.5 + t[2]))
        else:
            out = [(i, None, self.effective(e, now)) for i, e in enumerate(self.entries) if name is None or e["name"] == name or e["name"].startswith(name)]
            out.sort(key=lambda t: -t[2])
        return out[:k]

    # ---------------------------------------------------------------------------------------------------- serve a variant (never the same twice)
    def _context(self, i, prof):
        """(action to grade as, source fit, source rep) for entry i."""
        e = self.entries[i]; act = e["best_known"] if e["best_known"] in prof.p else None
        if act is None:
            act, _ = V.closest_action(self.F[i], prof)
        fit0 = G.grade_features(self.F[i], act, prof, use_match=False)["fit"]
        return act, fit0, V.clip_rep(self.clips[i])

    def serve(self, i, prof, uncommon=0.12, tries=24, use=True, now=None, seed=None):
        """A graded variant of entry i that differs from the original AND from the last KEEP_SEEDS served. Returns (clip, info) or (None, reason)."""
        e = self.entries[i]; dt = self.dt; a = self.clips[i]; act, fit0, r0 = self._context(i, prof)
        prev = []
        for sd, unc in e.get("seeds", [])[-KEEP_SEEDS:]:                    # regenerate what was served before (variants are deterministic in their seed)
            b, _ = V.make_variant(a, dt, np.random.default_rng(sd), unc); prev.append(V.clip_rep(b))
        base = int(time.time_ns() % (2 ** 31)) if seed is None else int(seed)
        for t in range(tries):
            sd = (base + 7919 * t) % (2 ** 31); b, info = V.make_variant(a, dt, np.random.default_rng(sd), uncommon)
            ok, why, v, fit, d = V.check_variant(b, dt, act, prof, fit0, r0, prev)
            if ok:
                e.setdefault("seeds", []).append([sd, uncommon]); e["seeds"] = e["seeds"][-KEEP_SEEDS:]
                if use:
                    self.touch(i, now)
                info.update(seed=sd, fit=round(fit), distance_m=round(d, 3), tries=t + 1, action=act); return b, info
        return None, f"no acceptable variant in {tries} tries"

    def forgotten(self, below=FORGET_BELOW, now=None):
        return [i for i, e in enumerate(self.entries) if self.effective(e, now) < below]

    def delete(self, idxs):
        drop = set(idxs); keep = [i for i in range(len(self.entries)) if i not in drop]
        self.clips = [self.clips[i] for i in keep]; self.F = self.F[keep]; self.entries = [self.entries[i] for i in keep]


def memory_profiles(mem, real_F, real_action, min_n=MIN_PROMOTE):
    """G.Profiles over the CMU examples plus every memory name with >= min_n examples that is not a CMU action (promotion). Returns (profiles, promoted names)."""
    names = {}
    for i, e in enumerate(mem.entries):
        names.setdefault(e["name"], []).append(i)
    promo = {n: ix for n, ix in names.items() if len(ix) >= min_n and n not in set(real_action)}
    if not promo:
        return G.Profiles(real_F, real_action), []
    for n, ix in promo.items():                    # a promoted name is a relative of the known action(s) its examples resemble, else grading real 'dance' as 'party_move' could never be "recognised"
        bk = [mem.entries[i]["best_known"] for i in ix if mem.entries[i]["kind"].startswith("variant")]
        rel = {b for b in set(bk) if bk.count(b) >= 0.3 * len(ix)}; fam = {n} | rel
        if fam not in G.FAMILIES:
            G.FAMILIES.append(fam)
    F = np.vstack([real_F] + [mem.F[ix] for ix in promo.values()]); act = np.concatenate([np.asarray(real_action, object)] + [np.array([n] * len(ix), object) for n, ix in promo.items()])
    return G.Profiles(F, act), sorted(promo)


def seed_from_real(mem, zr, RF, prof, per_action=0, actions=None, rng=None, log=print):
    """Store the real CMU examples (named by their action, kind 'known', origin 'cmu'). Returns counts."""
    A, off, dt = zr["actions"], zr["offsets"], float(zr["dt"]); act = [str(x) for x in zr["action"]]; perf = zr["performer"]; lab = zr["labels"]
    order = list(range(len(act))); rng = rng or np.random.default_rng(0)
    rng.shuffle(order); count = {}; res = dict(stored=0, reinforced=0, rejected=0)
    for i in order:
        if actions and act[i] not in actions:
            continue
        if per_action and count.get(act[i], 0) >= per_action:
            continue
        st, msg, _ = mem.add(A[off[i]:off[i + 1]], dt, act[i], prof, RF, source=f"cmu:{lab[i]}", origin="cmu", kind="known", allow_dup=True, meta=dict(performer=str(perf[i]), cmu_trial=str(lab[i])))
        res[st] += 1
        if st == "stored":
            count[act[i]] = count.get(act[i], 0) + 1
    return res


# ------------------------------------------------------------------------------------------------------------------ CLI
def _fmt_age(sec):
    d = sec / DAY
    return f"{d:.0f} d" if d >= 1 else f"{sec / 3600:.0f} h" if sec >= 3600 else "now"


def _load_clips(path, ids):
    z = np.load(path, allow_pickle=True); cl = G.split_clips(z); ids = ids if ids else list(range(len(cl)))
    return z, cl, ids


def _write_npz(path, clips, dt, labels, actions, desc):
    np.savez_compressed(path, actions=np.concatenate(clips).astype(np.float32), offsets=np.cumsum([0] + [len(c) for c in clips]), dt=np.float32(dt), action=np.array(actions),
                        labels=np.array(labels), description=np.array(desc), style=np.zeros((len(clips), 1), np.float32), style_names=np.array(["none"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seed", "add", "add-segments", "list", "recall", "vary", "rename", "grade", "export", "prune", "use"])
    ap.add_argument("arg", nargs="*", help="add/add-segments/grade: the clips npz; rename: ID NEWNAME; use: ID; export: OUT.jsonl")
    ap.add_argument("--dir", default=DEFAULT_DIR); ap.add_argument("--real", default=G.REAL_DEFAULT)
    ap.add_argument("--clip", type=int, nargs="+"); ap.add_argument("--name"); ap.add_argument("--note", default=""); ap.add_argument("--like"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--use", action="store_true"); ap.add_argument("--yes", action="store_true"); ap.add_argument("--below", type=float, default=FORGET_BELOW)
    ap.add_argument("--action"); ap.add_argument("--engine-check", action="store_true"); ap.add_argument("--status", default="novel", help="add-segments: which segment status to keep")
    ap.add_argument("--vary", action="store_true", help="recall: also make a fresh variant of the best match"); ap.add_argument("--out", default="data/variants.npz")
    ap.add_argument("--n", type=int, default=8, help="vary: variants per stored example"); ap.add_argument("--uncommon", type=float, default=0.12)
    ap.add_argument("--per-action", type=int, default=0, help="seed: at most this many examples per action (0 = all)"); ap.add_argument("--actions", nargs="*", help="seed: only these actions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); mem = Memory(a.dir)

    def real():
        z, F = G.load_real(a.real); return z, F

    if a.cmd == "seed":
        zr, RF = real(); prof = G.Profiles(RF, zr["action"]); t0 = time.time()
        r = seed_from_real(mem, zr, RF, prof, a.per_action, set(a.actions) if a.actions else None, np.random.default_rng(a.seed)); mem.save()
        by = {}
        for e in mem.entries:
            by[e["name"]] = by.get(e["name"], 0) + 1
        print(f"seeded {r['stored']} CMU examples ({r['reinforced']} duplicates, {r['rejected']} rejected by the gates) in {time.time() - t0:.0f}s; memory now holds {len(mem.entries)} motions, {len(by)} names, {sum(len(c) for c in mem.clips) * mem.dt / 3600:.2f} h")
    elif a.cmd in ("add", "add-segments"):
        if not a.arg:
            ap.error("pass the clips npz")
        z, cl, ids = _load_clips(a.arg[0], a.clip); dt = float(z["dt"]); zr, RF = real(); prof = G.Profiles(RF, zr["action"]); chk = lib = None
        if a.engine_check:
            import gen_dataset as g; lib = g.Lib(); chk = g.new_standing(lib)
        if a.cmd == "add" and not a.name:
            ap.error("add needs --name")
        n_new = n_re = 0; auto = sum(1 for e in mem.entries if e["name"].startswith("unlabelled_"))
        for i in ids:
            if a.cmd == "add-segments":
                if "status" not in z.files:
                    sys.exit("this file has no `status` array: make it with  motion_roll.py CLIPS.npz --cut OUT.npz")
                if str(z["status"][i]) != a.status:
                    continue
                auto += 1; name = f"unlabelled_{auto:03d}"
            else:
                name = a.name
            feas = None
            if chk is not None:
                feas = bool(chk.evaluate(cl[i], dt)["feasible"])
            st, msg, _ = mem.add(cl[i], dt, name, prof, RF, source=f"{os.path.basename(a.arg[0])}#{i}", note=a.note, feasible=feas)
            print(f"  clip {i}: {st}: {msg}"); n_new += st == "stored"; n_re += st == "reinforced"
            if a.cmd == "add-segments" and st != "stored":
                auto -= 1
        mem.save(); print(f"\n{n_new} stored, {n_re} reinforced; memory now holds {len(mem.entries)} motions in {a.dir}")
    elif a.cmd == "list":
        if not mem.entries:
            print("memory is empty (python tools/motion_memory.py seed fills it with the real CMU examples)"); return
        now = time.time(); names = {}
        for e in mem.entries:
            names.setdefault(e["name"], []).append(e)
        if len(mem.entries) > 60:                                   # a big memory: one line per name
            print(f"{'name':20s} {'n':>4s} {'origins':18s} {'kinds':26s} {'sec':>6s} {'strength max':>12s}")
            for n, es in sorted(names.items(), key=lambda kv: -len(kv[1])):
                org = {}; kd = {}
                for e in es:
                    org[e.get("origin", "user")] = org.get(e.get("origin", "user"), 0) + 1; kd[e["kind"].split(":")[0]] = kd.get(e["kind"].split(":")[0], 0) + 1
                print(f"{n:20s} {len(es):4d} {','.join(f'{k}:{v}' for k, v in org.items()):18s} {','.join(f'{k}:{v}' for k, v in kd.items()):26s} {sum(e['seconds'] for e in es):6.0f} {max(Memory.effective(e, now) for e in es):12.2f}")
        else:
            print(f"{'id':>3s}  {'name':20s} {'kind':16s} {'origin':7s} {'strength':>8s} {'uses':>4s} {'last used':>9s} {'sec':>5s}  closest known")
            for e in sorted(mem.entries, key=lambda e: (-Memory.effective(e, now))):
                print(f"{e['id']:3d}  {e['name']:20s} {e['kind']:16s} {e.get('origin', 'user'):7s} {Memory.effective(e, now):8.2f} {e['uses']:4d} {_fmt_age(now - e['last_used']):>9s} {e['seconds']:5.1f}  {e['best_known']} {e['best_known_fit']:.0f}")
        pro = [n for n, es in names.items() if len(es) >= MIN_PROMOTE]
        print(f"\n{len(mem.entries)} motions, {len(names)} names, {sum(len(c) for c in mem.clips) * (mem.dt or 0) / 3600:.2f} h" + (f"; names with >= {MIN_PROMOTE} examples (gradable as actions when not CMU ones): {len(pro)}" if pro else ""))
    elif a.cmd == "recall":
        like = RF = None
        if a.like:
            if not a.clip:
                ap.error("--like needs --clip N")
            z, cl, _ = _load_clips(a.like, a.clip); like = mf.extract(cl[a.clip[0]], float(z["dt"])); _, RF = real()
        t0 = time.perf_counter(); res = mem.recall(a.name, like, a.k, RF); ms = (time.perf_counter() - t0) * 1000
        if not res:
            print("nothing in memory matches"); return
        for i, d, s in res:
            e = mem.entries[i]; print(f"  #{e['id']:<3d} {e['name']:20s} {e['kind']:16s} strength {s:.2f}" + (f"   distance {d:.2f}" if d is not None else "") + f"   {e['seconds']:.1f} s")
        print(f"({ms:.1f} ms)")
        if a.vary:
            zr, RF = real(); prof = G.Profiles(RF, zr["action"]); i = res[0][0]; b, info = mem.serve(i, prof, a.uncommon)
            if b is None:
                print(f"could not make a variant of #{mem.entries[i]['id']}: {info}")
            else:
                _write_npz(a.out, [mem.clip(i), b], mem.dt, ["original", f"variant seed {info['seed']}"], [mem.entries[i]["name"]] * 2, ["the stored motion", ",".join(info["ops"])])
                print(f"variant of #{mem.entries[i]['id']}: ops {','.join(info['ops']) or '-'}, strength {info['strength']}{' UNCOMMON' if info['uncommon'] else ''}, fit {info['fit']}, {info['distance_m']} m from the original/last served; used (strength {mem.entries[i]['strength']:.2f})")
                print(f"wrote {a.out}  (python tools/view_clip.py {a.out} --open)")
            mem.save()
        elif a.use:
            mem.touch(res[0][0]); print(f"used #{mem.entries[res[0][0]]['id']}: strengthened to {mem.entries[res[0][0]]['strength']:.2f}"); mem.save()
    elif a.cmd == "vary":
        if not a.name:
            ap.error("vary needs --name")
        zr, RF = real(); prof = G.Profiles(RF, zr["action"]); ix = [i for i, e in enumerate(mem.entries) if e["name"] == a.name or e["name"].startswith(a.name)]
        if not ix:
            sys.exit("no such name in memory")
        clips, labels, acts, desc = [], [], [], []; rng_seed = a.seed
        for i in ix:
            for k in range(a.n):
                b, info = mem.serve(i, prof, a.uncommon, use=False, seed=rng_seed + 1000 * i + k)
                if b is not None:
                    clips.append(b); labels.append(f"#{mem.entries[i]['id']} v{k}"); acts.append(mem.entries[i]["name"]); desc.append(",".join(info["ops"]) + (" UNCOMMON" if info["uncommon"] else ""))
        mem.save(); _write_npz(a.out, clips, mem.dt, labels, acts, desc)
        print(f"{len(clips)} variants of {len(ix)} stored examples of '{a.name}' -> {a.out}")
    elif a.cmd == "use":
        i = next((j for j, e in enumerate(mem.entries) if str(e["id"]) == a.arg[0] or e["name"] == a.arg[0]), None)
        if i is None:
            sys.exit("no such memory")
        mem.touch(i); mem.save(); print(f"strengthened #{mem.entries[i]['id']} '{mem.entries[i]['name']}' to {mem.entries[i]['strength']:.2f} (uses {mem.entries[i]['uses']})")
    elif a.cmd == "rename":
        if len(a.arg) != 2:
            ap.error("rename ID NEWNAME")
        hit = [e for e in mem.entries if str(e["id"]) == a.arg[0] or e["name"] == a.arg[0]]
        if not hit:
            sys.exit("no such memory")
        for e in hit:
            e["name"] = a.arg[1]
        mem.save(); print(f"renamed {len(hit)} to '{a.arg[1]}'")
    elif a.cmd == "grade":
        if not a.arg or not a.action:
            ap.error("grade FILE.npz --action NAME")
        zr, RF = real(); prof, promo = memory_profiles(mem, RF, zr["action"]); z, cl, ids = _load_clips(a.arg[0], a.clip)
        if a.action not in prof.p:
            sys.exit(f"'{a.action}' is neither a CMU action nor a memory name with >= {MIN_PROMOTE} examples. Promoted from memory: {', '.join(promo) or 'none'}")
        r = G.grade_dataset([cl[i] for i in ids], float(z["dt"]), a.action, prof); G.report(r, verbose=True)
        if a.action in promo:
            print(f"   (profile of '{a.action}' is built from {r['real_n']} examples YOU stored, not from independent captures: it measures similarity to them)")
    elif a.cmd == "export":
        if not a.arg:
            ap.error("export OUT.jsonl")
        now = time.time(); n = 0
        with open(a.arg[0], "w") as f:
            for i, e in enumerate(mem.entries):
                if a.name and e["name"] != a.name:
                    continue
                f.write(json.dumps(dict(id=e["id"], name=e["name"], kind=e["kind"], origin=e.get("origin", "user"), strength=round(Memory.effective(e, now), 3), uses=e["uses"], dt=mem.dt,
                                        frames=np.round(mem.clip(i), 5).tolist())) + "\n"); n += 1
        print(f"wrote {n} motions to {a.arg[0]}")
    elif a.cmd == "prune":
        gone = mem.forgotten(a.below)
        if not gone:
            print(f"nothing has faded below {a.below} yet"); return
        for i in gone:
            e = mem.entries[i]; print(f"  #{e['id']:<3d} {e['name']:20s} strength now {Memory.effective(e):.2f}, last used {_fmt_age(time.time() - e['last_used'])} ago")
        if a.yes:
            mem.delete(gone); mem.save(); print(f"deleted {len(gone)}")
        else:
            print(f"{len(gone)} faded memories (nothing deleted: add --yes)")


if __name__ == "__main__":
    main()
