#!/usr/bin/env python3
"""
check_stats.py - can this prior actually RUN?  The translator can only ask for what the prior's training windows contain (goal_stats.speed_max = the 95th percentile of the moving windows).
"Run" is asked as exactly that speed, so if the data has no real running, "run" comes out as a brisk walk or a jog and no translator fix can change it.

  python tools/goal_translator/check_stats.py --prior runs/cvqvae/prior_best.pt                      # what the prior was trained with (needs torch)
  python tools/goal_translator/check_stats.py --stats '{"turn_max":0.9,"speed_cruise":1.2,"speed_max":2.1}'     # same, no torch: paste the numbers
  python tools/goal_translator/check_stats.py --clips runs/data/clips.npz --window 64,48,0.0333      # how the speeds of the WINDOWS in your data are spread (W,stride,dt of the prior)
  (--prior / --stats and --clips can be given together)

Reference speeds of a person: walk 1.2-1.5 m/s, jog 2-3, run 3-5. The prior's windows are about 2.1 s long, so these are speeds averaged over a window.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt_conventions as C

BANDS = [(0.15, "stand (<0.15)"), (0.8, "slow walk (0.15-0.8)"), (1.6, "walk (0.8-1.6)"), (2.5, "fast walk / jog (1.6-2.5)"), (3.5, "run (2.5-3.5)"), (99, "sprint (>3.5)")]


def load_prior_stats(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    gs = d.get("goal_stats")
    if not gs:
        sys.exit(f"{path}: no goal_stats in this checkpoint: it was trained without goal numbers (or is not a prior). Keys: {sorted(d.keys())}")
    names = d.get("goal_names")
    return gs, names, d.get("style_names"), d.get("cfg")


def verdict(speed_max):
    if speed_max >= 2.5:
        return f"speed_max {speed_max:.2f} m/s: the data reaches running speeds, so 'run' can look like running."
    if speed_max >= 1.8:
        return f"speed_max {speed_max:.2f} m/s: 'run' = a jog at best. Add running clips (or windows above ~2.5 m/s) to get a real run."
    return f"speed_max {speed_max:.2f} m/s: 'run' is a brisk walk. The data has almost nothing faster; the translator cannot fix this, the dataset has to."


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--prior", help="prior_best.pt (needs torch)")
    ap.add_argument("--stats", help="goal_stats as JSON text or a JSON file: turn_max, speed_cruise, speed_max")
    ap.add_argument("--clips", help="clips.npz of tools/gen_dataset.py")
    ap.add_argument("--window", default="64,48,0.03333", help="W,stride,dt used to cut the clips into windows (the prior's)")
    a = ap.parse_args()
    if not (a.prior or a.stats or a.clips):
        ap.error("give --prior, --stats and/or --clips")
    if a.prior:
        gs, names, styles, cfg = load_prior_stats(a.prior)
        print(f"[prior] goal layout: {len(names) if names else '?'} numbers {names if names else ''}; styles: {len(styles) if styles else 0}")
        print(f"[prior] turn_max {np.degrees(gs['turn_max']):.0f} deg/window  speed_cruise {gs['speed_cruise']:.2f}  speed_max {gs['speed_max']:.2f} m/s")
        print("[prior] " + verdict(gs["speed_max"]))
    elif a.stats:
        gs = json.loads(open(a.stats).read() if not a.stats.lstrip().startswith("{") else a.stats)
        print(f"[stats] turn_max {np.degrees(gs['turn_max']):.0f} deg/window  speed_cruise {gs['speed_cruise']:.2f}  speed_max {gs['speed_max']:.2f} m/s")
        print("[stats] " + verdict(gs["speed_max"]))
    if a.clips:
        import build_dataset as B
        W, stride, dt = [t(v) for t, v in zip((int, int, float), a.window.split(","))]
        tm = C.Timing(W, stride, dt)
        g = [x for _, seq in B.clip_goal_sequences(a.clips, tm) for x in seq]
        if not g:
            sys.exit("no full windows in these clips")
        g = np.asarray(g)
        sp, turn = g[:, 3], np.degrees(np.abs(g[:, 0]))
        print(f"[clips] {len(g)} windows of {W} frames every {stride} (dt {dt:.4f}): speed percentiles 50/90/95/99/max = " + " / ".join(f"{np.percentile(sp, q):.2f}" for q in (50, 90, 95, 99)) + f" / {sp.max():.2f} m/s")
        lo = -1
        for hi, name in BANDS:
            n = int(((sp >= (lo if lo > 0 else 0)) & (sp < hi)).sum()) if lo >= 0 else int((sp < hi).sum())
            print(f"   {name:<28} {n:6d}  {100.0 * n / len(sp):5.1f} %")
            lo = hi
        print(f"[clips] turning: 95th percentile of |turn| per window = {np.percentile(turn, 95):.0f} deg, max {turn.max():.0f} deg")
        print("[clips] " + verdict(float(np.percentile(sp[sp > 0.15], 95)) if (sp > 0.15).any() else 0.0))


if __name__ == "__main__":
    main()
