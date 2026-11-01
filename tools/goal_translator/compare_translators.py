#!/usr/bin/env python3
"""
compare_translators.py - old (words) against new (E5) on the same prompts, BEFORE the old one is removed. Numpy only: no torch, no transformers.

  python tools/goal_translator/compare_translators.py --e5 e5gt.npz --data data/gt
  python tools/goal_translator/compare_translators.py --e5 e5gt.npz --bytes gt.gguf --data data/gt --show 40

Always runs stage C (the word-list parser). Adds the byte-level NANITY model with --bytes and the E5 translator with --e5. Sections:
  1 fixed suite        prompt_suite.json (easy / hard): a regression test, written with stage C in mind
  2 unseen wording     prompt_suite_unseen.json: wording OUTSIDE the stage C lists (unfamiliar verbs, other sentence shapes, clockwise / port / starboard) and text with no
                       motion in it (a refusal or all-unspecified is correct; any invented motion is a failure). THIS is the test of "no word list"
  3 noise              the fixed suite with typos / missing spaces / odd case added (gt_corrupt) at 0x, 1x, 2x the training noise, 5 seeds each
  4 held-out data      --data: plan metrics (direction, order, exact) on a sample of the held-out clips. Stage C's number here is NOT independent: the rows were written from
                       templates and (in the current build_dataset.py) filtered by stage C's own parser
  5 disagreements      the prompts where the translators differ, with what each produced: read these, they tell you more than the totals
  6 speed              median milliseconds per translation
Then the gates. They are PROPOSED thresholds (my judgement, not measured on your data): change them with the flags. Exit code 0 only if every gate passes. Passing the gates means
"reasonable to make E5 the default and keep stage C selectable"; it does not mean "delete stage C": do that after real use in the UI and the closed-loop test.
"""
import argparse, json, os, random, statistics, sys, time

import numpy as np

import gt_conventions as C
import gt_corrupt
import gt_metrics as M
import e5_common as E
import stage_c as SC

HERE = os.path.dirname(os.path.abspath(__file__))


def build_translators(a, st, tm):
    tr = [("stage C", lambda t: SC.translate(t, st, tm))]
    if a.bytes:
        import gt_numpy
        m = gt_numpy.NumpyTranslator.from_gguf(a.bytes)
        tr.append(("byte model", m.translate))
    if a.e5:
        import e5_numpy
        m = e5_numpy.E5NumpyTranslator.from_npz(a.e5)
        if (m.stats.turn_max, m.stats.speed_max) != (st.turn_max, st.speed_max):
            print(f"warning: the E5 file was trained with stats {m.stats.to_dict()} but the data / prior uses {st.to_dict()}: the comparison is not exact")
        tr.append(("E5", m.translate))
    return tr


def run(fn, text):
    """-> ('plan', Plan) or ('refused', message). A refusal is an answer, not a crash: the text was not turned into a made-up plan."""
    try:
        return "plan", fn(text)
    except C.PlanError as e:
        return "refused", str(e)
    except Exception as e:                                       # a translator that crashes on a prompt is a finding, not a reason to stop the comparison
        return "refused", f"{type(e).__name__}: {e}"


def passes(res, case, st):
    kind, v = res
    if kind == "plan" and len(v.windows) == 0:                    # "nothing in the text could be placed": no goals at all is a refusal in everything but name
        kind = "refused"
    if kind == "refused":
        return bool(case.get("refuse_ok", False))
    cls = M.collapse(v, st)
    if "expect" in case and cls != case["expect"].split():
        return False
    return not any(c in cls for c in case.get("forbid", ""))


def shown(res, st):
    if res[0] == "refused":
        return "refused"
    return " ".join(M.collapse(res[1], st)) or "no plan"


def table(title, rows, names):
    print(f"\n{title}")
    print("  " + f"{'':14s}{'n':>4s}  " + "  ".join(f"{n:>10s}" for n in names))
    for label, n, vals in rows:
        print("  " + f"{label:14s}{n:4d}  " + "  ".join(f"{v:10.0%}" for v in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e5", help="e5gt.npz from export_e5_translator.py")
    ap.add_argument("--bytes", help="the byte-level translator .gguf (export_goal_translator.py), optional")
    ap.add_argument("--data", help="folder written by build_dataset.py: the held-out part (val.jsonl) and the prior's stats / timing (meta.json)")
    ap.add_argument("--stats", help="prior_stats.json when --data is not given")
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--noise-seeds", type=int, default=5)
    ap.add_argument("--show", type=int, default=30, help="how many disagreements to print")
    ap.add_argument("--min-easy", type=float, default=0.90, help="gate: E5 pass rate on the easy tier of the fixed suite")
    ap.add_argument("--min-unseen", type=float, default=0.50, help="gate: E5 pass rate on meaning + directions (the tiers where unknown words carry the answer)")
    ap.add_argument("--min-unseen-margin", type=float, default=0.20, help="gate: ... and at least this far above stage C")
    ap.add_argument("--max-false-motion", type=float, default=0.05, help="gate: share of no-motion prompts that E5 turns into a plan with motion")
    ap.add_argument("--min-direction", type=float, default=0.97, help="gate: held-out direction accuracy")
    ap.add_argument("--min-order", type=float, default=0.90, help="gate: held-out order accuracy")
    ap.add_argument("--max-ms", type=float, default=500.0, help="gate: median milliseconds per translation")
    a = ap.parse_args()
    if not (a.e5 or a.bytes):
        sys.exit("give at least --e5 (or --bytes): stage C alone has nothing to be compared with")

    if a.data:
        meta = json.load(open(os.path.join(a.data, "meta.json")))
        st, tm = C.Stats.from_dict(meta["stats"]), C.Timing(**meta["timing"])
    elif a.stats:
        st, tm = C.Stats.from_dict(json.load(open(a.stats))), C.Timing(**C.DEFAULT_WINDOW)
    else:
        sys.exit("give --data (folder from build_dataset.py) or --stats prior_stats.json")

    tr = build_translators(a, st, tm)
    names = [n for n, _ in tr]
    fixed = M.load_suite(os.path.join(HERE, "prompt_suite.json"))
    unseen = M.load_suite(os.path.join(HERE, "prompt_suite_unseen.json"))
    cache = {}                                                    # (translator, text) -> result, so the sections and the disagreement list agree

    def get(name, fn, text):
        k = (name, text)
        if k not in cache:
            cache[k] = run(fn, text)
        return cache[k]

    def rate(name, fn, cases):
        return float(np.mean([passes(get(name, fn, c["text"]), c, st) for c in cases])) if cases else float("nan")

    # ---- 1 + 2 suites
    def tier_rows(suite, tiers):
        rows = []
        for t in tiers:
            cs = [c for c in suite if c["tier"] == t]
            rows.append((t, len(cs), [rate(n, f, cs) for n, f in tr]))
        return rows

    table("1  fixed suite (prompt_suite.json): regression", tier_rows(fixed, ["easy", "hard"]), names)
    table("2  unseen wording (prompt_suite_unseen.json): the test of 'no word list'", tier_rows(unseen, ["meaning", "directions", "verbs", "phrasing", "distractor"]), names)
    core = [c for c in unseen if c["tier"] in ("meaning", "directions")]          # unknown words (or a known word in another sense) carry the answer
    easyish = [c for c in unseen if c["tier"] in ("verbs", "phrasing")]           # a parser that ignores unknown words already passes most of these
    distract = [c for c in unseen if c["tier"] == "distractor"]
    print("     (meaning + directions decide the gate. verbs + phrasing: a parser that ignores unknown words passes most of them, so they only guard against regression."
          " distractor: no plan, or all-unspecified, is the pass.)")

    # ---- 3 noise
    base = dict(p_typo=0.12, p_unsafe=0.04, p_nospace=0.18, p_case=0.3, p_punct=0.15)
    noise = {}
    rows = []
    for mult in (0.0, 1.0, 2.0):
        vals = []
        for n, f in tr:
            ok = tot = 0
            for s in range(1 if mult == 0 else a.noise_seeds):
                for i, c in enumerate(fixed):
                    rng = random.Random(s * 1000 + i)
                    text = c["text"] if mult == 0 else gt_corrupt.corrupt(c["text"], rng, **{k: min(0.9, v * mult) for k, v in base.items()})
                    ok += passes(run(f, text), c, st)
                    tot += 1
            vals.append(ok / tot)
            noise[(n, mult)] = ok / tot
        rows.append((f"noise x{mult:g}", len(fixed) * (1 if mult == 0 else a.noise_seeds), vals))
    table("3  fixed suite with noise added (typos, missing spaces, case, punctuation)", rows, names)

    # ---- 4 held-out data
    val = {}
    if a.data:
        vr = E.load_rows(os.path.join(a.data, "val.jsonl"))
        sub = random.Random(0).sample(vr, min(a.val_n, len(vr)))
        print(f"\n4  held-out clips ({len(sub)} rows from val.jsonl)")
        for n, f in tr:
            scores, refused = [], 0
            for r in sub:
                kind, p = get(n, f, r["text"])
                if kind == "refused":
                    refused += 1
                    continue
                scores.append(M.score_pair(p, M.plan_from_rows(r["windows"]), st, tm))
            agg = M.aggregate(scores) if scores else {}
            val[n] = agg
            print(f"  {n:11s} direction {agg.get('direction_acc', float('nan')):.3f}  order {agg.get('order_acc', float('nan')):.3f}  exact {agg.get('exact', float('nan')):.3f}  "
                  f"duration err {agg.get('dur_err_windows', float('nan')):.2f} windows  refused {refused}/{len(sub)}")
        print("  (stage C's row is not independent evidence, see the top of this file)")

    # ---- 5 disagreements
    ref_n = "E5" if "E5" in names else names[-1]
    ref_f = dict(tr)[ref_n]
    sc_f = dict(tr)["stage C"]
    diffs = []
    for suite_name, suite in (("fixed", fixed), ("unseen", unseen)):
        for c in suite:
            r1, r0 = get(ref_n, ref_f, c["text"]), get("stage C", sc_f, c["text"])
            p1, p0 = passes(r1, c, st), passes(r0, c, st)
            if p1 != p0:
                diffs.append((p1, suite_name, c, r1, r0))
    print(f"\n5  where {ref_n} and stage C differ in pass / fail ({len(diffs)} prompts; {ref_n} wins {sum(d[0] for d in diffs)}, stage C wins {sum(not d[0] for d in diffs)})")
    for p1, sn, c, r1, r0 in sorted(diffs, key=lambda d: d[0])[: a.show]:
        print(f"  {'WIN ' if p1 else 'LOSS'} [{sn}/{c['tier']}] {c['text']!r}  expect {c.get('expect', '-')!r}   {ref_n}: {shown(r1, st)}   stage C: {shown(r0, st)}")

    # ---- 6 speed
    texts = [c["text"] for c in fixed + unseen]
    ms = {}
    for n, f in tr:
        t = []
        for x in texts[:40]:
            t0 = time.perf_counter()
            run(f, x)
            t.append((time.perf_counter() - t0) * 1000)
        ms[n] = statistics.median(t[2:])
    print("\n6  median milliseconds per translation: " + "   ".join(f"{n} {v:.1f}" for n, v in ms.items()))

    # ---- gates
    if "E5" not in names:
        print("\n(no E5 file given: gates skipped)")
        return
    gates = []

    def gate(name, ok, detail):
        gates.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    easy = [c for c in fixed if c["tier"] == "easy"]
    hard = [c for c in fixed if c["tier"] == "hard"]
    f5, fc = dict(tr)["E5"], sc_f
    print("\ngates (proposed thresholds, change them with the flags)")
    e_easy = rate("E5", f5, easy)
    gate("easy tier, no regression", e_easy >= a.min_easy, f"E5 {e_easy:.0%} (needs {a.min_easy:.0%}; stage C {rate('stage C', fc, easy):.0%})")
    e_hard, c_hard = rate("E5", f5, hard), rate("stage C", fc, hard)
    gate("hard tier, at least as good as stage C", e_hard >= c_hard, f"E5 {e_hard:.0%} vs stage C {c_hard:.0%}")
    e_un, c_un = rate("E5", f5, core), rate("stage C", fc, core)
    gate("unseen meaning + directions: no word list means this works", e_un >= a.min_unseen and e_un >= c_un + a.min_unseen_margin,
         f"E5 {e_un:.0%} vs stage C {c_un:.0%} (needs >= {a.min_unseen:.0%} and +{a.min_unseen_margin:.0%} over stage C)")
    e_ez, c_ez = rate("E5", f5, easyish), rate("stage C", fc, easyish)
    gate("unseen verbs + phrasing: no regression", e_ez >= c_ez - 0.10, f"E5 {e_ez:.0%} vs stage C {c_ez:.0%} (may be at most 10 points below)")
    fm = float(np.mean([get("E5", f5, c["text"])[0] == "plan" and bool(set("LRSX") & set(M.collapse(get("E5", f5, c["text"])[1], st))) for c in distract])) if distract else 0.0
    fm_c = float(np.mean([get("stage C", fc, c["text"])[0] == "plan" and bool(set("LRSX") & set(M.collapse(get("stage C", fc, c["text"])[1], st))) for c in distract])) if distract else 0.0
    gate("no invented motion for text without motion", fm <= a.max_false_motion, f"E5 invented motion for {fm:.0%} of {len(distract)} prompts (needs <= {a.max_false_motion:.0%}; stage C {fm_c:.0%})")
    gate("noise: no worse than stage C at 2x noise", noise[("E5", 2.0)] >= noise[("stage C", 2.0)], f"E5 {noise[('E5', 2.0)]:.0%} vs stage C {noise[('stage C', 2.0)]:.0%}")
    if val.get("E5"):
        gate("held-out direction accuracy", val["E5"]["direction_acc"] >= a.min_direction, f"{val['E5']['direction_acc']:.3f} (needs {a.min_direction})")
        gate("held-out order accuracy", val["E5"]["order_acc"] >= a.min_order, f"{val['E5']['order_acc']:.3f} (needs {a.min_order})")
    gate("fast enough for the UI", ms["E5"] <= a.max_ms, f"{ms['E5']:.0f} ms median (needs <= {a.max_ms:.0f})")
    if all(gates):
        print("\nALL GATES PASSED: E5 can be the default; keep stage C selectable until it has been used for real (UI + closed loop).")
    else:
        print(f"\n{gates.count(False)} GATE(S) FAILED: do not retire stage C yet. Read section 5 first: it shows which prompts decided it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
