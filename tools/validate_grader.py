#!/usr/bin/env python3
"""
validate_grader.py - is the realism grader any good? Checks it against REAL captured walks, stylised real walks, your engine's clips and corrupted clips.

  python tools/validate_grader.py                          # uses data/real_walks.npz (downloads it first if missing) and data/clips.npz
  python tools/validate_grader.py --engine data/clips.npz --real data/real_walks.npz

What a good grader shows: real ordinary walks score high (they are the definition of human), your engine / a generator scores lower and the table says
exactly which measures cost it points, jittery or broken clips score low even when a stylised request has allowed the other deviations, and a real
stylised walk (zombie march, old man ...) scores lower on the strict scale but is not punished for what the request legitimately changed (--allow).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realism as R

PRESET = {"zombie march": "zombie", "walk / hobble": "limp", "drunk walk": "drunk", "old man": "tired", "gangly teenager": "drunk", "cat": "robot", "chicken": "robot", "dinosaur": "robot"}


def grade_file(z, angles_from=None, lib=None, idx=None, allow=()):
    A, off, dt = z["actions"], z["offsets"], float(z["dt"]); out = []
    for i in (idx if idx is not None else range(len(off) - 1)):
        a = A[off[i]:off[i + 1]]
        ang = z["angles"][off[i]:off[i + 1]] if "angles" in z.files else R.angles_from_engine(lib, a, dt)
        out.append(R.grade_clip(a, dt, ang, allow))
    return out


def vals(rs, k):
    return [it["value"] for r in rs for it in r["items"] if it["name"] == k and it["value"] is not None]


def scs(rs, k):
    return [it["score"] for r in rs for it in r["items"] if it["name"] == k and it["score"] is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="data/real_walks.npz"); ap.add_argument("--engine", default="data/clips.npz"); ap.add_argument("--n", type=int, default=60, help="engine clips to grade")
    ap.add_argument("--lib"); ap.add_argument("--repo"); ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.real):
        import get_real_walks; sys.argv = ["get_real_walks.py", "--out", a.real]; get_real_walks.main()
    if not os.path.exists(a.engine):
        sys.exit(f"{a.engine} not found: run  python tools/gen_dataset.py --n 300 --out data  first")
    import gen_dataset as g
    lib = g.Lib(a.lib, a.repo, build=not a.no_build); real = np.load(a.real); eng = np.load(a.engine)
    kind = list(real["kind"]); lab = list(real["labels"]); desc = list(real["description"])
    normal = [i for i, k in enumerate(kind) if k == "normal"]; styled = [i for i, k in enumerate(kind) if k == "styled"]
    rn = grade_file(real, idx=normal); rs_ = grade_file(real, idx=styled); rs_allow = [grade_file(real, idx=[i], allow=R.GROUPS.get(PRESET.get(desc[i], "none"), []))[0] for i in styled]
    n = min(a.n, len(eng["offsets"]) - 1); re_ = grade_file(eng, lib=lib, idx=range(n)); re_ = [r for r in re_ if r["score"] is not None]
    f = lambda L: np.array([r["score"] for r in L if r["score"] is not None])
    print("=" * 100); print("1. REAL ordinary walks (should score high)")
    print("   " + "  ".join(f"{lab[i]}:{r['score']:.0f}" for i, r in zip(normal, rn)) + f"   | mean {f(rn).mean():.0f}, min {f(rn).min():.0f}")
    print("\n2. REAL stylised walks: strict score  ->  with the deviations that request legitimately changes allowed")
    for i, r, ra in zip(styled, rs_, rs_allow):
        top = ", ".join(x.split(" =")[0] for x in r["findings"][:3]); p = PRESET.get(desc[i], "none")
        print(f"   {desc[i]:18s} {r['score']:5.0f} -> {ra['score']:5.0f}  (allow={p:7s}) strict findings: {top}")
    print(f"\n3. YOUR ENGINE ({len(re_)} clips from {a.engine}): mean {f(re_).mean():.0f}, median {np.median(f(re_)):.0f}, best {f(re_).max():.0f}, worst {f(re_).min():.0f}   (real walks: {f(rn).mean():.0f})")
    print("   measure              human range        real walks   engine    engine score   -> the weakest first")
    rows = []
    for k, t in R.RULES.items():
        if scs(re_, k) and vals(rn, k):
            rows.append((np.mean(scs(re_, k)), k, t, np.median(vals(rn, k)), np.median(vals(re_, k))))
    for m, k, t, rv, ev in sorted(rows)[:9]:
        print(f"   {k:20s} {t[0]:>7g}-{t[1]:<8g}  {rv:10.3g} {ev:9.3g}   {m:6.2f}      {t[7] if ev < t[0] else t[8]}")
    print("\n4. CORRUPTED engine clips: does realism catch what physics cannot, and vice versa? (physics = the engine's feasibility verdict)")
    rng = np.random.default_rng(0); chk = g.new_standing(lib); A, off, dt = eng["actions"], eng["offsets"], float(eng["dt"]); res = {}
    clean = []
    for i in range(min(40, len(off) - 1)):
        c = A[off[i]:off[i + 1]]; other = A[off[(i + 1) % (len(off) - 1)]:off[(i + 1) % (len(off) - 1) + 1]]
        ang0 = R.angles_from_engine(lib, c, dt); r0 = R.grade_clip(c, dt, ang0)
        if r0["score"] is None: continue
        clean.append(r0["score"]); cs, _ = g.corruptions(rng, c, other, dt)
        for nm, x in cs.items():
            x = x.astype(np.float32); ang = R.angles_from_engine(lib, x, dt); r1 = R.grade_clip(x, dt, ang); r2 = R.grade_clip(x, dt, ang, R.GROUPS["zombie"])
            res.setdefault(nm, []).append((r1["score"] if r1["score"] is not None else np.nan, r2["score"] if r2["score"] is not None else np.nan, not chk.evaluate(x, dt)["feasible"]))
    print(f"   {'corruption':36s}{'realism strict':>16s}{'realism, zombie allowed':>25s}{'physics flags it':>20s}")
    print(f"   {'(the clean clips themselves)':36s}{np.mean(clean):16.0f}")
    for nm, v in res.items():
        v = np.array(v, float); print(f"   {nm:36s}{np.nanmean(v[:, 0]):16.0f}{np.nanmean(v[:, 1]):25.0f}{v[:, 2].mean() * 100:19.0f}%")
    print("=" * 100)
    print("Read it like this: real walks near the top, engine below with named reasons, corruptions low. If real walks score badly on a measure, the range for that")
    print("measure in realism.RULES is wrong; if a stylised real walk is punished for what its style is, add that measure to the preset in realism.GROUPS.")


if __name__ == "__main__":
    main()
