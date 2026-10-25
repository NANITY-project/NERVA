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
    # 3b --------------------------------------------------------------- rolling labeller + muscle memory
    r = run([sys.executable, os.path.join(HERE, "motion_roll.py"), "--selftest", "--real", real]); acc = [l for l in r.stdout.splitlines() if l.startswith("frame accuracy")]
    step("rolling labeller: glued real clips are cut and labelled (frame accuracy >= 60%)", r.returncode == 0, acc[0] if acc else (r.stdout + r.stderr)[-200:])
    import motion_memory as MM
    mdir = os.path.join(tmp, "mem"); mem = MM.Memory(mdir); wv = [i for i in range(len(act)) if act[i] == "walk"][:5]; st = [mem.add(A[off[i]:off[i + 1]], dt, "wave", prof, F)[0] for i in wv]
    st2 = mem.add(A[off[wv[0]]:off[wv[0] + 1]], dt, "wave", prof, F)[0]; mem.save(); back = MM.Memory(mdir)
    hit = back.recall(like=mf.extract(A[off[wv[1]]:off[wv[1] + 1]], dt), k=1, real_F=F)
    step("muscle memory: store, duplicate reinforces, reload, recall by similarity", len(wv) >= 3 and st.count("stored") >= len(wv) - 1 and st2 == "reinforced" and len(back.entries) == st.count("stored") and bool(hit) and back.entries[hit[0][0]]["name"] == "wave", f"{st.count('stored')} stored, duplicate -> {st2}")
    ph = A[off[wv[0]]:off[wv[0] + 1]].copy(); nz = [i for i in range(30) if i % 4 != 3 and i not in (8, 13)]; ph[:, nz] += np.random.default_rng(0).normal(0, 0.04, (len(ph), len(nz))).astype(np.float32)
    step("muscle memory: a grossly jittery clip (4 cm noise) is not admitted", back.add(ph, dt, "junk", prof, F)[0] == "rejected")
    import motion_vary as MV
    c0 = A[off[wv[0]]:off[wv[0] + 1]]; f0 = G.grade_features(mf.extract(c0, dt), "walk", prof, use_match=False)["fit"]; r0 = MV.clip_rep(c0); rv = np.random.default_rng(0); kept, sib = [], []
    for _ in range(30):
        b, _i = MV.make_variant(c0, dt, rv); ok, *_r = MV.check_variant(b, dt, "walk", prof, f0, r0, sib)
        if ok:
            kept.append(b); sib.append(MV.clip_rep(b))
    step("variation: variants of one clip pass the grader and are not copies of it or of each other", len(kept) >= 4 and all(MV.rep_dist(MV.clip_rep(k), [r0]) >= MV.MIN_DIST for k in kept), f"{len(kept)} of 30 kept")
    s1, i1 = back.serve(0, prof); s2, i2 = back.serve(0, prof)
    step("muscle memory: serve() never gives the same variant twice (and records its seeds)", s1 is not None and s2 is not None and MV.rep_dist(MV.clip_rep(s1), [MV.clip_rep(s2)]) > 0 and len(back.entries[0]["seeds"]) == 2, f"seeds {[x[0] for x in back.entries[0]['seeds']]}")
    # CMU is never a training source. To test the pipeline without ACCAD/100STYLE downloads, a copy of the CMU examples stands in for a CC BY set (relabelled source "accad"; never leaves the temp folder)
    z0, _ = G.load_real(real, cc=False); n0 = len(z0["action"]); cc_test = os.path.join(tmp, "cc_motions.npz")
    np.savez(cc_test, actions=z0["actions"], offsets=z0["offsets"], dt=z0["dt"], angles=z0["angles"], labels=z0["labels"], action=z0["action"], performer=np.array(["accad:" + str(p) for p in z0["performer"]]),
             description=z0["description"], kind=z0["kind"], source=np.array(["accad"] * n0), style_label=np.array([""] * n0))
    bd = os.path.join(tmp, "vae"); cmd = [sys.executable, os.path.join(HERE, "build_vae_dataset.py"), "--real", real, "--cc", cc_test, "--hours", "0.12", "--engine-clips", "40", "--no-jsonl", "--out", bd, "--memory", os.path.join(tmp, "vmem")] + ([] if a.engine else ["--no-engine"])
    r = run(cmd); okf = r.returncode == 0 and all(os.path.exists(os.path.join(bd, f)) for f in ("train_windows.npz", "val_windows.npz", "stats.json", "report.txt", "preview.npz"))
    tw = np.load(os.path.join(bd, "train_windows.npz"), allow_pickle=True) if okf else None; vw = np.load(os.path.join(bd, "val_windows.npz"), allow_pickle=True) if okf else None
    step("build_vae_dataset.py: one command builds windows, stats and a report; no source in both splits, none from CMU", okf and len(tw["windows"]) > 0 and np.isfinite(tw["windows"]).all() and not (set(tw["origin_id"].tolist()) & set(vw["origin_id"].tolist())) and "cmu" not in set(map(str, tw["source"])), (r.stdout + r.stderr)[-200:] if not okf else f"{len(tw['windows'])} train windows; sources {sorted(set(map(str, tw['source'])))}")
    # 4 ---------------------------------------------------------------- headers
    hdr = os.path.join(tmp, "motion"); r = run([sys.executable, os.path.join(HERE, "export_headers.py"), "--out", hdr, "--real", real, "--cc", cc_test, "--clips", "3" if a.quick else "6"])
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
