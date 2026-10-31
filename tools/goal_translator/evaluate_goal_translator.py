#!/usr/bin/env python3
"""
evaluate_goal_translator.py - held-out metrics + the fixed prompt suite, for stage C and (optionally) a trained model, side by side (spec section 9).

  python evaluate_goal_translator.py --data data/gt                          # stage C only (no torch needed)
  python evaluate_goal_translator.py --data data/gt --ckpt runs/gt/gt_best.pt

PROMOTION RULE (spec: "a model is not promoted unless it beats stage C on the suite"). The model is promoted only if ALL hold:
  * suite: strictly more cases passed than stage C, and no easy-tier case that stage C passes is failed
  * held-out rows: direction_acc, order_acc and exact are each >= stage C's
  Anything else prints PROMOTE: no and the reasons. The script never copies or renames files: promotion is your decision.
"""
import argparse, json, os, sys

import numpy as np

import gt_conventions as C
import gt_metrics as M
import stage_c as SC

HERE = os.path.dirname(os.path.abspath(__file__))


def run_rows(translate, rows, st, tm):
    scores, errs = [], 0
    for r in rows:
        try:
            pred = translate(r["text"])
        except C.PlanError:
            errs += 1
            pred = C.Plan([])
        scores.append(M.score_pair(pred, M.plan_from_rows(r["windows"]), st, tm))
    a = M.aggregate(scores)
    a["plan_errors"] = errs
    return a


def run_suite(translate, suite, st):
    res = {}
    for c in suite:
        try:
            plan = translate(c["text"])
            ok = M.suite_check(plan, c, st)
            got = M.collapse(plan, st)
        except C.PlanError as e:
            ok, got = False, [f"error: {e}"]
        res[c["text"]] = (c.get("tier", "easy"), ok, got)
    return res


def fmt(x, d=3):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--n", type=int, default=1000, help="held-out rows scored (greedy decoding is sequential)")
    ap.add_argument("--suite", default=os.path.join(HERE, "prompt_suite.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    st, tm = C.Stats.from_dict(meta["stats"]), C.Timing(**meta["timing"])
    rows = [json.loads(l) for l in open(os.path.join(a.data, f"{a.split}.jsonl"), encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["kind"] != "template"] or rows              # the held-out score is on REWRITTEN text (the exact templates are too easy)
    rows = rows[:a.n]
    suite = M.load_suite(a.suite)

    systems = {"stage C": lambda t: SC.translate(t, st, tm)}
    if a.ckpt:
        import gt_torch as G
        model, cfg, ck = G.load_checkpoint(a.ckpt, a.device)
        if ck["gt"]["layout_version"] != C.LAYOUT_VERSION:
            sys.exit("checkpoint token layout differs from gt_conventions.py")
        systems["model"] = lambda t: G.decode_plan(model, t, a.device)

    held, suites = {}, {}
    for name, fn in systems.items():
        held[name] = run_rows(fn, rows, st, tm)
        suites[name] = run_suite(fn, suite, st)

    keys = ["direction_acc", "order_acc", "exact", "turn_err_bins", "dur_err_windows", "unspec_precision", "unspec_recall", "roundtrip_dyaw", "roundtrip_speed", "plan_errors"]
    print(f"held-out ({a.split}, {len(rows)} rewritten rows, source clips the model never saw)")
    print("  " + "metric".ljust(18) + "".join(n.rjust(10) for n in systems))
    for k in keys:
        print("  " + k.ljust(18) + "".join(fmt(held[n][k]).rjust(10) for n in systems))
    print("\nprompt suite")
    for name in systems:
        for tier in ("easy", "hard"):
            rs = [v for v in suites[name].values() if v[0] == tier]
            print(f"  {name:8s} {tier}: {sum(v[1] for v in rs)}/{len(rs)}")
    if a.verbose or a.ckpt:
        for text in suites["stage C"]:
            row = "  " + ("".join("PASS " if suites[n][text][1] else "FAIL " for n in systems)) + f"[{suites['stage C'][text][0]}] {text}"
            print(row)
            if a.verbose:
                for n in systems:
                    print(f"        {n}: {' '.join(suites[n][text][2])}")

    if a.ckpt:
        reasons = []
        sp = {n: sum(v[1] for v in suites[n].values()) for n in systems}
        if not sp["model"] > sp["stage C"]:
            reasons.append(f"suite: model {sp['model']} is not more than stage C {sp['stage C']}")
        reg = [t for t, v in suites["stage C"].items() if v[0] == "easy" and v[1] and not suites["model"][t][1]]
        if reg:
            reasons.append(f"easy-tier regressions vs stage C: {reg}")
        for k in ("direction_acc", "order_acc", "exact"):
            if not held["model"][k] >= held["stage C"][k]:
                reasons.append(f"held-out {k}: model {fmt(held['model'][k])} < stage C {fmt(held['stage C'][k])}")
        print("\nPROMOTE: " + ("yes" if not reasons else "no"))
        for r in reasons:
            print("  - " + r)


if __name__ == "__main__":
    main()
