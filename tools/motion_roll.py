#!/usr/bin/env python3
"""
motion_roll.py - label and grade a clip ROLLING IN TIME: which action is happening when, and is each stretch a good example of it.

  python tools/motion_roll.py data/eng/clips.npz                      # roll every clip, print the segments + a per-action grade
  python tools/motion_roll.py data/clips.npz --clips 0 3 --verbose    # only these clips, with the weak findings per segment
  python tools/motion_roll.py data/clips.npz --cut data/segments.npz  # write every segment as its own labelled clip (motion_grader.py / view_clip.py read it)
  python tools/motion_roll.py data/clips.npz --jsonl data/seg.jsonl   # the same segments as JSONL: {action, status, score, dt, frames}
  python tools/motion_roll.py --selftest                              # glue real sit/stand/walk/run clips together and see if the boundaries are found

Why: motion_grader.py judges a clip as ONE action. A generator (or a person) produces a clip where several things happen one after the other, and one label
per clip is wrong for it. Here a window slides along the clip (default 2 s, every 0.5 s), the grader's classifier gives each window a probability per action,
overlapping windows are averaged per frame, the argmax is the frame's label, and stretches shorter than --min-seg (1.5 s) are absorbed by a neighbour. Each segment
is then graded against the real examples of ITS action (same score as motion_grader.py: 0-100, jitter gate, recognised-as).

Segment status
  ok        score >= 70 and the classifier agrees: a good example of its label
  weak      scored 50-70, or the classifier disagrees: usable, look at the findings
  bad       below 50, or the jitter gate failed: a body cannot move like that
  novel     the body moves plausibly (gate passed) but fits NO known action well (best fit over all actions < 60): this is the "valid, but not in the label set"
            motion. tools/motion_memory.py add-segments takes exactly these.
  short     under 1 s: labelled but too short to grade
Measured with --selftest on glued real clips (sit/stand/walk/run/jump/crouch): 68-74% of frames get the right action or a relative (varies with the clips drawn), and
the 1.5 s minimum merges rather than invents segments (3 true segments came out as ~5). Honest limits: the labels are the CLASSIFIER's, so for a file without its own labels the "recognised" part of the score is circular: the score then says "inside the
real range of that action", not "the right action". Boundaries are only good to about +-1 s (a window sees both sides of a change). Actions that kinematics cannot
tell apart (walk/limp/sneak, sit/get_up, ...) are one family: a label flips inside a family more often than across families. Pass labels (`action` array per clip,
or --action NAME) to also get "label agreement": the share of frames whose rolled label is the declared action or a relative.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_features as mf, motion_grader as G

WIN, HOP, MIN_SEG, MIN_GRADE = 2.0, 0.5, 1.5, 1.0
OK_SCORE, WEAK_SCORE, NOVEL_FIT = 70.0, 50.0, 60.0
MIN_DATASET = 5                      # segments of one action needed for a dataset verdict (variety needs a few)


# ------------------------------------------------------------------------------------------------------------------ rolling
def frame_posteriors(a, dt, prof, win=WIN, hop=HOP):
    """[F, n_actions] probability per frame (average of every window covering the frame) and the action names."""
    F = len(a); n = min(F, int(round(win / dt))); h = max(1, int(round(hop / dt))); names = list(prof.lda.cls)
    starts = list(range(0, F - n + 1, h))
    if starts[-1] + n < F:
        starts.append(F - n)
    P = np.zeros((F, len(names))); C = np.zeros(F)
    for s in starts:
        v = mf.extract(a[s:s + n], dt)
        if not np.isfinite(v).any():
            continue
        pr = prof.lda.proba(v); P[s:s + n] += np.array([pr[c] for c in names]); C[s:s + n] += 1
    if not (C > 0).any():
        return np.full((F, len(names)), 1.0 / len(names)), names
    idx = np.where(C > 0)[0]
    for i in np.where(C == 0)[0]:                        # frames no window could measure: borrow the nearest measured frame
        P[i], C[i] = P[idx[np.argmin(np.abs(idx - i))]], 1.0
    return P / C[:, None], names


def segment(P, names, dt, min_seg=MIN_SEG):
    """Runs of equal argmax label; runs shorter than min_seg are absorbed by the neighbour (previous or next) the run's own probabilities favour. -> [(start, end, label)]"""
    lab = P.argmax(1); runs = []
    s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            runs.append([s, i, int(lab[s])]); s = i
    m = int(round(min_seg / dt))
    while len(runs) > 1:
        lens = [r[1] - r[0] for r in runs]; k = int(np.argmin(lens))
        if lens[k] >= m:
            break
        r = runs[k]; cand = [j for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        j = max(cand, key=lambda j: P[r[0]:r[1], runs[j][2]].mean())
        runs[j][0], runs[j][1] = min(runs[j][0], r[0]), max(runs[j][1], r[1]); del runs[k]
        merged = []                                       # neighbours that now carry the same label join up
        for q in runs:
            if merged and merged[-1][2] == q[2]:
                merged[-1][1] = q[1]
            else:
                merged.append(q)
        runs = merged
    return [(a, b, names[c]) for a, b, c in runs]


def grade_segment(seg, dt, label, prof, P_seg=None):
    """Report card for one segment as an example of its rolled label, plus its best fit over ALL actions (for the novel test)."""
    if len(seg) * dt < MIN_GRADE:
        return dict(status="short", score=None, fit=None, recognised=None, best_fit=None, best_fit_action=None, gate=None, findings=[], runner_up=None)
    v = mf.extract(seg, dt)
    if not np.isfinite(v).any():
        return dict(status="short", score=None, fit=None, recognised=None, best_fit=None, best_fit_action=None, gate=None, findings=[], runner_up=None)
    card = G.grade_clip(seg, dt, label, prof)
    fits = {a: G.grade_features(v, a, prof, use_match=False)["fit"] for a in prof.p}; fits = {a: f for a, f in fits.items() if f is not None}
    bf_a = max(fits, key=fits.get) if fits else None; bf = fits.get(bf_a)
    score, gate = card["score"], card["gate"]
    if score is None:
        status = "short"
    elif gate < 0.99:
        status = "bad"                                       # jitter a body cannot make, whatever the rest says
    elif bf is not None and bf < NOVEL_FIT:
        status = "novel"                                     # plausible body motion that no known action explains
    elif score >= OK_SCORE and card["recognised"]:
        status = "ok"
    elif score >= WEAK_SCORE:
        status = "weak"
    else:
        status = "bad"
    rank = card["ranking"]; ru = next(((a, float(p)) for a, p in rank if a != label), None)
    return dict(status=status, score=score, fit=card["fit"], recognised=bool(card["recognised"]), best_fit=bf, best_fit_action=bf_a, gate=float(gate),
                findings=card["findings"][:3], runner_up=ru)


def roll_clip(a, dt, prof, win=WIN, hop=HOP, min_seg=MIN_SEG):
    """Roll one clip. Returns (segments, frame_labels): segments = list of dicts (start, end, t0, t1, label, prob + the grade_segment fields)."""
    a = np.asarray(a, np.float32); P, names = frame_posteriors(a, dt, prof, win, hop); segs = []
    flab = np.empty(len(a), object)
    for s, e, lab in segment(P, names, dt, min_seg):
        flab[s:e] = lab; g = grade_segment(a[s:e], dt, lab, prof)
        segs.append(dict(start=int(s), end=int(e), t0=round(s * dt, 2), t1=round(e * dt, 2), label=lab, prob=float(P[s:e, names.index(lab)].mean()), **g))
    return segs, flab


# ------------------------------------------------------------------------------------------------------------------ dataset-level
def roll_dataset(clips, dt, prof, declared=None, **kw):
    """Roll every clip. Returns dict(per_clip=[segments], frame_labels=[...], by_action={action: dataset grade or summary}, agreement=...)."""
    per, fl = [], []
    for c in clips:
        s, f = roll_clip(c, dt, prof, **kw); per.append(s); fl.append(f)
    by = {}
    groups = {}
    for ci, segs in enumerate(per):
        for sg in segs:
            groups.setdefault(sg["label"], []).append((ci, sg))
    for act, items in groups.items():
        gradable = [(ci, s) for ci, s in items if s["status"] != "short"]
        secs = sum(s["t1"] - s["t0"] for _, s in items); st = {}
        for _, s in items:
            st[s["status"]] = st.get(s["status"], 0) + 1
        row = dict(segments=len(items), seconds=round(secs, 1), status=st,
                   mean_score=float(np.mean([s["score"] for _, s in gradable])) if gradable else None)
        if len(gradable) >= MIN_DATASET and act in prof.p:
            segs = [clips[ci][s["start"]:s["end"]] for ci, s in gradable]
            r = G.grade_dataset(segs, dt, act, prof); row["dataset"] = {k: v for k, v in r.items() if k != "cards"}
        by[act] = row
    agree = None
    if declared is not None:
        tot = hit = 0
        for f, d in zip(fl, declared):
            fam = G.family(d) if d in prof.p else {d}; tot += len(f); hit += sum(1 for x in f if x in fam)
        agree = hit / max(1, tot)
    return dict(per_clip=per, frame_labels=fl, by_action=by, agreement=agree)


def cut_segments(clips, per, dt, src_ids, out):
    """Write every segment as a labelled clip (npz readable by motion_grader.py, view_clip.py and motion_memory.py add-segments)."""
    acts, labels, status, score, src, start, desc, lab_id = [], [], [], [], [], [], [], []
    for ci, segs in zip(src_ids, per):
        for s in segs:
            c = clips[src_ids.index(ci)][s["start"]:s["end"]]
            acts.append(c); labels.append(s["label"]); status.append(s["status"]); score.append(np.nan if s["score"] is None else s["score"]); src.append(ci)
            start.append(s["start"]); lab_id.append(f"clip{ci}@{s['t0']:.1f}s"); desc.append(f"{s['label']} [{s['status']}] score {'n/a' if s['score'] is None else round(s['score'])}")
    if not acts:
        return 0
    np.savez_compressed(out, actions=np.concatenate(acts).astype(np.float32), offsets=np.cumsum([0] + [len(c) for c in acts]), dt=np.float32(dt),
                        action=np.array(labels), status=np.array(status), score=np.array(score, np.float32), src_clip=np.array(src), start_frame=np.array(start),
                        labels=np.array(lab_id), description=np.array(desc), style=np.zeros((len(acts), 1), np.float32), style_names=np.array(["none"]))
    return len(acts)


def write_jsonl(clips, per, dt, src_ids, out):
    n = 0
    with open(out, "w") as f:
        for ci, segs in zip(src_ids, per):
            for s in segs:
                c = clips[src_ids.index(ci)][s["start"]:s["end"]]
                f.write(json.dumps(dict(action=s["label"], status=s["status"], score=None if s["score"] is None else round(s["score"], 1), src_clip=ci, start_frame=s["start"],
                                        dt=round(dt, 6), frames=np.round(c, 5).tolist())) + "\n"); n += 1
    return n


# ------------------------------------------------------------------------------------------------------------------ reporting
def print_clip(ci, segs, verbose, declared=None):
    tot = segs[-1]["t1"] if segs else 0
    print(f"\nclip {ci}  ({tot:.1f} s)" + (f"  declared: {declared}" if declared else ""))
    for s in segs:
        sc = "  n/a" if s["score"] is None else f"{s['score']:5.0f}"
        extra = ""
        if s["status"] == "novel":
            extra = f"   best known fit: {s['best_fit_action']} {s['best_fit']:.0f}"
        elif s["runner_up"] and s["status"] in ("weak", "bad"):
            extra = f"   also looks like {s['runner_up'][0]} ({100 * s['runner_up'][1]:.0f}%)"
        print(f"  {s['t0']:6.1f}-{s['t1']:5.1f} s  {s['label']:12s} p={s['prob']:.2f}  score {sc}  {s['status']:6s}{extra}")
        if verbose:
            for l in s["findings"][:2]:
                print(f"           - {l[:150]}")


def print_actions(by, agree):
    print("\n== per action (all segments rolled to that label, from every clip)")
    for act, r in sorted(by.items(), key=lambda kv: -kv[1]["seconds"]):
        st = ", ".join(f"{k} {v}" for k, v in sorted(r["status"].items()))
        ms = "n/a" if r["mean_score"] is None else f"{r['mean_score']:.0f}"
        line = f"  {act:12s} {r['segments']:3d} segments {r['seconds']:7.1f} s   mean score {ms:>3s}   ({st})"
        d = r.get("dataset")
        if d and "mean" in d:
            line += f"   dataset: {d['verdict']}" + (f" - {d['problems'][0]}" if d["problems"] else "")
        elif d:
            line += f"   dataset: {d['verdict']}"
        else:
            line += f"   (dataset verdict needs >= {MIN_DATASET} gradable segments)"
        print(line)
    if agree is not None:
        print(f"\n  label agreement with the declared action: {100 * agree:.0f}% of frames (family-aware)")


# ------------------------------------------------------------------------------------------------------------------ selftest
def _glue(parts, dt):
    """Chain clips end to start: every position is shifted so the next clip begins where the previous ended (no teleport at the seam)."""
    pos = [0, 2, 4, 6, 9, 11, 14, 16, 18, 20, 22, 24, 26, 28]; out, labels = [], []
    cur = np.zeros(3)
    for act, c in parts:
        c = c.copy(); d = np.array([cur[0] - c[0, 0], cur[1] - c[0, 2]])
        for j in pos:
            c[:, j] += d[0] if j in (0, 4, 9, 14, 18, 22, 26) else d[1] if j in (2, 6, 11, 16, 20, 24, 28) else 0
        out.append(c); labels += [act] * len(c); cur = np.array([c[-1, 0], 0, c[-1, 2]])
    return np.concatenate(out).astype(np.float32), labels


def selftest(real, seed=0, trials=12):
    prof, z, F = G.get_profiles(real); dt = float(z["dt"]); A, off, act, per = z["actions"], z["offsets"], np.array([str(x) for x in z["action"]]), np.array(z["performer"])
    rng = np.random.default_rng(seed); seqs = [["sit", "stand", "walk"], ["walk", "run", "walk"], ["stand", "walk", "crouch", "stand"], ["walk", "jump", "stand"], ["sit", "get_up", "walk"]]
    acc, bound, rows = [], [], []
    for t in range(trials):
        names = seqs[t % len(seqs)]; parts = []
        for n in names:
            ids = np.where(act == n)[0]
            if len(ids) == 0:
                break
            i = int(rng.choice(ids)); parts.append((n, A[off[i]:off[i + 1]]))
        else:
            a, truth = _glue(parts, dt); segs, fl = roll_clip(a, dt, prof)
            hit = np.mean([p in G.family(tr) or tr in G.family(p) for p, tr in zip(fl, truth)]); acc.append(hit)
            tb = [int(np.sum([len(c) for _, c in parts[:k]])) for k in range(1, len(parts))]; eb = [s["start"] for s in segs[1:]]
            err = [min(abs(b - e) for e in eb) * dt if eb else np.inf for b in tb]; bound += err
            rows.append((names, hit, [s["label"] for s in segs]))
    for names, hit, got in rows:
        print(f"  truth {'>'.join(names):28s} frame accuracy {100 * hit:3.0f}%   rolled: {'>'.join(got)}")
    b = np.array(bound); fin = np.isfinite(b)
    print(f"\nframe accuracy (family-aware): mean {100 * np.mean(acc):.0f}%   boundaries found within 1 s: {100 * np.mean(b[fin] <= 1.0):.0f}% of {len(b)}, within 2 s: {100 * np.mean(b[fin] <= 2.0):.0f}%   (median error {np.median(b[fin]):.2f} s)")
    print("note: the classifier saw these performers' clips in training, so this is optimistic; validate_grader.py has the leave-one-performer-out numbers.")
    return float(np.mean(acc)) >= 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?"); ap.add_argument("--real", default=G.REAL_DEFAULT); ap.add_argument("--clips", type=int, nargs="+")
    ap.add_argument("--action", help="declare every clip to be this action (adds label agreement); default: the file's own `action` array if it has one")
    ap.add_argument("--win", type=float, default=WIN); ap.add_argument("--hop", type=float, default=HOP); ap.add_argument("--min-seg", type=float, default=MIN_SEG)
    ap.add_argument("--verbose", action="store_true"); ap.add_argument("--cut", metavar="OUT.npz"); ap.add_argument("--jsonl", metavar="OUT.jsonl"); ap.add_argument("--json", metavar="OUT.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        G.load_real(a.real); sys.exit(0 if selftest(a.real) else 1)
    if not a.npz:
        ap.error("pass a clips npz (or --selftest)")
    prof, _, _ = G.get_profiles(a.real); z = np.load(a.npz, allow_pickle=True); dt = float(z["dt"]); allc = G.split_clips(z); ids = a.clips or list(range(len(allc)))
    clips = [allc[i] for i in ids]
    decl = [a.action] * len(ids) if a.action else ([str(z["action"][i]) for i in ids] if "action" in z.files else None)
    if decl and any(d not in prof.p for d in decl):
        print("note: declared actions the grader has no profile for are compared by name only: " + ", ".join(sorted({d for d in decl if d not in prof.p})))
    r = roll_dataset(clips, dt, prof, declared=decl, win=a.win, hop=a.hop, min_seg=a.min_seg)
    shown = ids if len(ids) <= 12 else ids[:12]
    for k, ci in enumerate(shown):
        print_clip(ci, r["per_clip"][k], a.verbose, decl[k] if decl else None)
    if len(ids) > len(shown):
        print(f"\n  ... {len(ids) - len(shown)} more clips (--clips to pick; every clip counts in the per-action table)")
    print_actions(r["by_action"], r["agreement"])
    if a.cut:
        print(f"\nwrote {cut_segments(clips, r['per_clip'], dt, ids, a.cut)} segments to {a.cut}")
    if a.jsonl:
        print(f"wrote {write_jsonl(clips, r['per_clip'], dt, ids, a.jsonl)} segments to {a.jsonl}")
    if a.json:
        json.dump(dict(per_clip=r["per_clip"], by_action=r["by_action"], agreement=r["agreement"]), open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
