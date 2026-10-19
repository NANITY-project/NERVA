#!/usr/bin/env python3
"""
motion_test.py - ONE command that tests the whole motion-grading toolchain.

  python tools/motion_test.py              # full: real data (downloaded once), grader validation, dataset grading, header export + C++ compile and run   (~2 min)
  python tools/motion_test.py --quick      # small: 8 actions, a few clips each (~20 s after the first download)
  python tools/motion_test.py --engine     # also generate walking clips with the engine and grade them as "walk"   (needs g++; builds libnebula the first time)
  make motion-test                         # the same, from the Makefile

Steps: 1 real data present  2 grader validated on held-out real clips (validate_grader.py's bar)  3 dataset grading behaves: real sit passes as sit, the same clips
FAIL as stand, a mixed file is judged per action  4 headers exported to a temp folder, compiled with g++ -Wall -Wextra, run (src/motion_headers_test.cpp)
5 (--engine) the engine's own clips judged as walk. Exit status 0 only if steps 1-4 pass. Nothing in your repo is modified (headers go to a temp folder;
use  python tools/export_headers.py  to write include/motion/ for real).
"""
import argparse, os, shutil, subprocess, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import motion_features as mf, motion_grader as G

QUICK_ACTIONS = ["sit", "stand", "walk", "run", "jump", "crouch", "get_up", "kick"]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--engine", action="store_true"); ap.add_argument("--real", default=None)
    a = ap.parse_args(); results = []; t00 = time.time()
    def step(name, ok, detail=""):
        results.append((name, ok)); print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)
    real = a.real or os.path.join(ROOT, "data", "real_motions_quick.npz" if a.quick else "real_motions.npz"); os.chdir(ROOT)
    # 1 ---------------------------------------------------------------- real data
    if not os.path.exists(real):
        print(f"downloading real motion data to {real} (once) ...", flush=True)
        r = run([sys.executable, os.path.join(HERE, "get_real_motions.py"), "--out", real] + (["--quick", "--actions"] + QUICK_ACTIONS if a.quick else []))
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:]); step("real motion data", False, "download failed: check your internet connection"); sys.exit(1)
    z, F = G.load_real(real); acts = sorted(set(z["action"])); step("real motion data", len(acts) >= (8 if a.quick else 20), f"{len(z['offsets']) - 1} clips, {len(acts)} actions, {len(set(z['performer']))} performers")
    # 2 ---------------------------------------------------------------- grader validation
    t = time.time(); r = run([sys.executable, os.path.join(HERE, "validate_grader.py"), "--real", real, "--n-corrupt", "30" if a.quick else "60"])
    tail = [l for l in r.stdout.splitlines() if l.startswith("   [")]
    sel = [l for l in tail if "wrong-label" in l or "corruptions" in l]
    if a.quick:           # quick set has 3-6 clips per action: the statistical bar is for the full set, here only the things that must always hold
        ok2 = all("[ok]" in l for l in sel) and bool(sel)
    else:
        ok2 = r.returncode == 0
    step("grader validation (held-out real clips, wrong labels, corruptions)", ok2, f"{time.time() - t:.0f}s")
    for l in (sel if a.quick else tail):          # quick mode: the held-out pass-rate lines need the full set, so they are not shown
        print("      " + l.strip())
    for l in r.stdout.splitlines():
        if l.startswith("   ALL:") or l.startswith("   real 'sit'") or l.startswith("   real 'stand'") or "wrong-label gradings" in l:
            print("      " + l.strip())
    # 3 ---------------------------------------------------------------- dataset grading
    prof = G.Profiles(F, z["action"]); A, off, dt = z["actions"], z["offsets"], float(z["dt"]); act = z["action"]
    clips = lambda n: [A[off[i]:off[i + 1]] for i in range(len(act)) if act[i] == n]
    good = G.grade_dataset(clips("sit"), dt, "sit", prof); wrong = G.grade_dataset(clips("sit"), dt, "stand", prof); other = G.grade_dataset(clips("stand"), dt, "sit", prof)
    step("dataset grading: real 'sit' clips judged as sit", good["verdict"] in ("PASS", "MARGINAL"), f"{good['verdict']}, mean {good['mean']:.0f}, {100 * good['recognised']:.0f}% recognised")
    step("dataset grading: the same clips judged as stand are rejected", wrong["verdict"] == "FAIL", f"{wrong['verdict']}, mean {wrong['mean']:.0f}, {100 * wrong['passing']:.0f}% pass")
    step("dataset grading: real 'stand' clips judged as sit are rejected", other["verdict"] == "FAIL", f"{other['verdict']}, mean {other['mean']:.0f}")
    noisy = [c + np.random.default_rng(0).normal(0, 0.01, c.shape).astype(np.float32) * (np.arange(30) % 4 != 3) for c in clips("walk")[:20]]
    nr = G.grade_dataset(noisy, dt, "walk", prof); step("dataset grading: jittery walking clips are rejected (gate)", nr["verdict"] == "FAIL", f"{nr['verdict']}, mean {nr['mean']:.0f}")
    tmp = tempfile.mkdtemp(prefix="nerva_motion_"); mixed = os.path.join(tmp, "mixed.npz"); sel = [i for i in range(len(act)) if act[i] in ("sit", "stand", "walk")]
    cl = [A[off[i]:off[i + 1]] for i in sel]; np.savez(mixed, actions=np.concatenate(cl), offsets=np.cumsum([0] + [len(c) for c in cl]), dt=z["dt"], action=np.array([act[i] for i in sel]))
    r = run([sys.executable, os.path.join(HERE, "motion_grader.py"), mixed, "--real", real]); n_blocks = sum(1 for l in r.stdout.splitlines() if l.startswith("== "))
    step("dataset grading: a labelled file is judged action by action (CLI)", n_blocks == 3 and r.returncode == 0, f"{n_blocks} actions reported")
    # 4 ---------------------------------------------------------------- headers
    hdr = os.path.join(tmp, "motion"); r = run([sys.executable, os.path.join(HERE, "export_headers.py"), "--out", hdr, "--real", real, "--clips", "3" if a.quick else "6"])
    ok = r.returncode == 0; print("      " + (r.stdout.splitlines()[0] if r.stdout else r.stderr[-300:]))
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if ok and cxx:
        exe = os.path.join(tmp, "motion_headers_test"); r = run([cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-I" + hdr, os.path.join(ROOT, "src", "motion_headers_test.cpp"), "-o", exe])
        if r.returncode != 0:
            print(r.stderr[-1500:]); ok = False
        else:
            r = run([exe]); print("\n".join("      " + l for l in r.stdout.strip().splitlines()[-3:])); ok = r.returncode == 0
        step("headers: exported, compiled (-Wall -Wextra) and decoded by C++", ok)
    else:
        step("headers: exported, compiled and decoded by C++", False, "export failed" if not ok else "no C++ compiler found (install g++/clang++ or set CXX)")
    # 5 ---------------------------------------------------------------- engine (optional)
    if a.engine:
        d = os.path.join(tmp, "engine"); r = run([sys.executable, os.path.join(HERE, "gen_dataset.py"), "--n", "30", "--out", d])
        if r.returncode != 0 or not os.path.exists(os.path.join(d, "clips.npz")):
            print("      engine dataset could not be generated:\n" + (r.stdout + r.stderr)[-800:]); print("[skip] engine walk dataset (not counted)")
        else:
            e = np.load(os.path.join(d, "clips.npz")); ec = [e["actions"][e["offsets"][i]:e["offsets"][i + 1]] for i in range(len(e["offsets"]) - 1)]
            G.report(G.grade_dataset(ec, float(e["dt"]), "walk", prof), verbose=True); print("[info] engine walk dataset graded above (informational: it measures the engine, it is not a pass/fail of the tooling)")
    shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in results if not ok]; print(f"\n{'ALL OK' if not bad else 'FAILED: ' + '; '.join(bad)}   ({len(results)} checks, {time.time() - t00:.0f}s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
