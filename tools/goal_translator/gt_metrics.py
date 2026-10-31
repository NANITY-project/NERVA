"""
gt_metrics.py - the evaluation of the Goal Translator (spec section 9). numpy only, no torch: it scores any Plan against the expected one, whichever translator made it.

Metrics (per held-out row, then averaged; held-out = whole SOURCE CLIPS the model never saw):
  direction_acc   of the windows whose true turn is not straight, how often the predicted window turns the same way (the sign)
  order_acc       the collapsed sequence of turn directions (L R L ...) is identical
  turn_err_bins   mean |predicted turn bin - true turn bin| over windows both call specified
  dur_err_windows mean |predicted number of windows - true number|
  unspec_p / _r   window-level precision / recall of the UNSPEC label (windows compared by position; a window past the end of the shorter plan counts as "not UNSPEC")
  roundtrip       mean abs error of the recovered goal numbers [dyaw rad, speed m/s] vs the numbers measured from the clip (the label IS the measured goals, snapped to bins)
  exact           the whole plan is identical
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

import gt_conventions as C

DEAD = 0.12


def window_class(w: C.Win, st: C.Stats) -> str:
    """U unspecified | L / R turning | S straight, moving | X standing"""
    if w.unspecified:
        return "U"
    t = C.bin_to_turn(w.turn_bin, st)
    if abs(t) / st.turn_max >= DEAD:
        return "L" if t > 0 else "R"
    return "S" if C.bin_to_speed(w.speed_bin, st) > 0 else "X"


def collapse(plan: C.Plan, st: C.Stats) -> List[str]:
    out: List[str] = []
    for w in plan.windows:
        c = window_class(w, st)
        if not out or out[-1] != c:
            out.append(c)
    return out


def plan_from_rows(windows) -> C.Plan:
    return C.Plan([C.UNSPEC_WIN if w is None else C.Win(int(w[0]), int(w[1])) for w in windows])


def score_pair(pred: C.Plan, truth: C.Plan, st: C.Stats, tm: C.Timing) -> Dict[str, float]:
    p, t = pred.windows, truth.windows
    n = min(len(p), len(t))
    tcls = [window_class(w, st) for w in t]
    pcls = [window_class(w, st) for w in p]
    turn_idx = [i for i in range(len(t)) if tcls[i] in "LR"]
    r: Dict[str, float] = {}
    r["direction_acc"] = float(np.mean([i < len(p) and pcls[i] == tcls[i] for i in turn_idx])) if turn_idx else float("nan")
    seq = lambda cls: [c for k, c in enumerate(cls) if c in "LR" and (k == 0 or cls[k - 1] != c)]
    r["order_acc"] = float(seq(pcls) == seq(tcls))
    both = [i for i in range(n) if not p[i].unspecified and not t[i].unspecified]
    r["turn_err_bins"] = float(np.mean([abs(p[i].turn_bin - t[i].turn_bin) for i in both])) if both else float("nan")
    r["dur_err_windows"] = float(abs(len(p) - len(t)))
    pu = [i < len(p) and p[i].unspecified for i in range(len(t))]
    tu = [w.unspecified for w in t]
    r["_tp"], r["_fp"] = float(sum(a and b for a, b in zip(pu, tu))), float(sum(a and not b for a, b in zip(pu, tu))) + float(sum(w.unspecified for w in p[len(t):]))
    r["_fn"] = float(sum((not a) and b for a, b in zip(pu, tu)))
    if both:
        gp = C.plan_to_goals(C.Plan([p[i] for i in both]), st, tm)
        gt = C.plan_to_goals(C.Plan([t[i] for i in both]), st, tm)
        r["roundtrip_dyaw"] = float(np.mean([abs(a[0] - b[0]) for a, b in zip(gp, gt)]))
        r["roundtrip_speed"] = float(np.mean([abs(a[3] - b[3]) for a, b in zip(gp, gt)]))
    else:
        r["roundtrip_dyaw"] = r["roundtrip_speed"] = float("nan")
    r["exact"] = float(pred.windows == truth.windows)
    return r


def aggregate(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {"n": float(len(scores))}
    for k in ("direction_acc", "order_acc", "turn_err_bins", "dur_err_windows", "roundtrip_dyaw", "roundtrip_speed", "exact"):
        v = [s[k] for s in scores if not np.isnan(s[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    tp, fp, fn = (sum(s[k] for s in scores) for k in ("_tp", "_fp", "_fn"))
    out["unspec_precision"] = tp / (tp + fp) if tp + fp else float("nan")
    out["unspec_recall"] = tp / (tp + fn) if tp + fn else float("nan")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------ the fixed prompt suite
# Lives in the repo (prompt_suite.json). `expect` = the collapsed window classes in order (see window_class); `forbid` = classes that must NOT appear at all.
def load_suite(path: str) -> List[dict]:
    return json.load(open(path, encoding="utf-8"))["prompts"]


def suite_check(plan: C.Plan, case: dict, st: C.Stats) -> bool:
    cls = collapse(plan, st)
    if "expect" in case and cls != case["expect"].split():
        return False
    return not any(c in cls for c in case.get("forbid", ""))
