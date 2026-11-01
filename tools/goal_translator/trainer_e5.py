#!/usr/bin/env python3
"""
trainer_e5.py - the whole E5 Goal Translator pipeline in one command.

  python tools/goal_translator/trainer_e5.py                       # data -> train -> export, everything automatic
  python tools/goal_translator/trainer_e5.py --stage data          # only build + merge the dataset
  python tools/goal_translator/trainer_e5.py --synthetic 5000 --epochs 8 --head base
  python tools/goal_translator/trainer_e5.py --clips data/clips.npz --prior runs/cvqvae/prior_best.pt --vae runs/cvqvae/best.pt

Why this exists: build_dataset.py cannot mix sources in one call (--clips and --synthetic are mutually exclusive). So this script builds TWO datasets with the SAME
stats and timing, concatenates them into one folder (train/val/test.jsonl + meta.json) and hands that folder to train_e5_translator.py.

Stages (all run by default, each one skips work that is already done unless --force):
  1 clips     use --clips (default data/clips.npz); if it is missing, generate --gen-clips N with tools/gen_dataset.py (needs a C++ compiler); if that fails,
              continue with synthetic data only and say so
  2 data      real   -> <out>/_real     build_dataset.py --clips      (stats/timing from --prior/--vae if they exist, else measured from the clips)
              synth  -> <out>/_synth    build_dataset.py --synthetic  (stats/timing copied from the real build's meta.json, so both match exactly)
              merge  -> <out>/data      train/val/test.jsonl concatenated, exact duplicates dropped, meta.json rewritten, split integrity checked
  3 train     train_e5_translator.py on <out>/data; resumes from <out>/e5gt_last.pt if it exists
  4 export    export_e5_translator.py -> <out>/e5gt.npz  (numpy-only runtime for UI/goal_plan.py)
  5 compare   (--compare) compare_translators.py against stage C

Honest notes
  * Synthetic rows teach text -> goals only (no motion realism). Real rows carry the realism. Use --synth-train-only to keep every synthetic row in train, so the
    validation numbers (and the "best" checkpoint) are measured on real held-out clips only.
  * If the prior/VAE are missing, goal stats are measured from the clips (build_dataset warns). Pass --prior for the real numbers.
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GEN_DATASET = os.path.join(ROOT, "tools", "gen_dataset.py")
SPLITS = ("train", "val", "test")


def say(msg):
    print(f"[trainer_e5] {msg}", flush=True)


def run(script, args, cwd=None):
    cmd = [sys.executable, os.path.join(HERE, script) if not os.path.isabs(script) else script] + [str(x) for x in args]
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd or HERE)
    if r.returncode != 0:
        sys.exit(f"[trainer_e5] {script} failed (exit {r.returncode})")


def exists(p):
    return bool(p) and os.path.exists(p)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------------------------------------------------------ 1 clips
def ensure_clips(a):
    clips = os.path.abspath(a.clips)
    if exists(clips) and not a.regen_clips:
        say(f"clips: using {clips}")
        return clips
    if a.gen_clips <= 0:
        say("clips: none and --gen-clips 0 -> synthetic data only")
        return None
    out = os.path.dirname(clips) or "."
    say(f"clips: {clips} not found, generating {a.gen_clips} with tools/gen_dataset.py (needs a C++ compiler)")
    try:
        r = subprocess.run([sys.executable, GEN_DATASET, "--n", str(a.gen_clips), "--out", out, "--seed", str(a.seed)], cwd=ROOT)
        if r.returncode == 0 and os.path.exists(os.path.join(out, "clips.npz")):
            gen = os.path.join(out, "clips.npz")
            if os.path.abspath(gen) != clips:
                shutil.copy(gen, clips)
            return clips
        say(f"clips: gen_dataset.py failed (exit {r.returncode})")
    except Exception as e:
        say(f"clips: could not run gen_dataset.py: {e}")
    say("clips: continuing with synthetic data only (no motion realism)")
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------------ 2 data
def build_real(a, clips, d_real):
    args = ["--clips", clips, "--out", d_real, "--seed", a.seed, "--per-clip", a.per_clip, "--rewriter", a.rewriter]
    if exists(a.prior):
        args += ["--prior", a.prior]
    elif a.stats:
        args += ["--stats", a.stats]
    if exists(a.vae):
        args += ["--vae", a.vae]
    if a.timing:
        args += ["--timing", a.timing]
    if a.rewrites:
        args += ["--rewrites", a.rewrites]
    run("build_dataset.py", args)


def build_synth(a, d_synth, meta):
    args = ["--synthetic", a.synthetic, "--out", d_synth, "--seed", a.seed + 1, "--per-clip", a.per_clip, "--rewriter", a.rewriter]
    if meta:                                                                  # same numbers as the real build: the two sets must agree on stats and timing
        t = meta["timing"]
        args += ["--stats", json.dumps(meta["stats"]), "--timing", f'{t["W"]},{t["stride"]},{t["dt"]}']
    else:
        if a.stats:
            args += ["--stats", a.stats]
        if a.timing:
            args += ["--timing", a.timing]
    run("build_dataset.py", args)


def merge(a, parts, d_out):
    """parts: [(label, folder)]. Concatenate the splits, drop exact duplicates, check meta agreement and clip-level split integrity."""
    metas = [(lab, json.load(open(os.path.join(d, "meta.json")))) for lab, d in parts]
    base = metas[0][1]
    for lab, m in metas[1:]:
        for k in ("layout_version", "n_turn_bins", "n_speed_bins", "stats", "timing"):
            if m[k] != base[k]:
                sys.exit(f"[trainer_e5] cannot merge: {lab} differs from {metas[0][0]} in {k!r}:\n  {base[k]}\n  {m[k]}")
    os.makedirs(d_out, exist_ok=True)
    merged = {sp: [] for sp in SPLITS}
    seen, dropped = set(), 0
    for (lab, d), (_, m) in zip(parts, metas):
        for sp in SPLITS:
            for r in read_jsonl(os.path.join(d, f"{sp}.jsonl")):
                tgt = "train" if (a.synth_train_only and lab == "synthetic") else sp
                key = hashlib.sha1(json.dumps([r["text"], r["windows"]], ensure_ascii=False).encode()).hexdigest()
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                r = dict(r, split=tgt, source=lab)
                merged[tgt].append(r)
    ids = {sp: {r["clip"] for r in merged[sp]} for sp in SPLITS}
    for x in SPLITS:
        for y in SPLITS:
            if x < y and ids[x] & ids[y]:
                sys.exit(f"[trainer_e5] split leak: {x} and {y} share {len(ids[x] & ids[y])} source clips")
    counts = {}
    for sp in SPLITS:
        with open(os.path.join(d_out, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in merged[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in merged[sp]:
            by[r["source"]] = by.get(r["source"], 0) + 1
        counts[sp] = dict(rows=len(merged[sp]), clips=len(ids[sp]), by_source=by)
    meta = dict(base)
    meta.update(source="merged:" + "+".join(f"{lab}={m['source']}" for lab, m in metas), counts=counts,
                parts={lab: dict(source=m["source"], counts=m["counts"], verify=m.get("verify")) for lab, m in metas},
                duplicates_dropped=dropped, synth_train_only=bool(a.synth_train_only))
    json.dump(meta, open(os.path.join(d_out, "meta.json"), "w"), indent=1)
    say("merged -> " + d_out + ": " + ", ".join(f"{sp} {v['rows']} rows / {v['clips']} clips {v['by_source']}" for sp, v in counts.items()) + f"  ({dropped} duplicates dropped)")
    if not merged["train"] or not merged["val"]:
        sys.exit("[trainer_e5] empty train or val split after merging: build more clips / a larger --synthetic")


def stage_data(a, out):
    d_real, d_synth, d_data = (os.path.join(out, x) for x in ("_real", "_synth", "data"))
    if exists(os.path.join(d_data, "meta.json")) and not a.force:
        say(f"data: {d_data} already built (use --force to rebuild)")
        return d_data
    parts, meta_real = [], None
    clips = ensure_clips(a)
    if clips:
        build_real(a, clips, d_real)
        meta_real = json.load(open(os.path.join(d_real, "meta.json")))
        parts.append(("real", d_real))
    if a.synthetic > 0:
        build_synth(a, d_synth, meta_real)
        parts.append(("synthetic", d_synth))
    if not parts:
        sys.exit("[trainer_e5] nothing to train on: no clips and --synthetic 0")
    merge(a, parts, d_data)
    return d_data


# ------------------------------------------------------------------------------------------------------------------------------------------------ 3-5
def stage_train(a, out, d_data):
    last = os.path.join(out, "e5gt_last.pt")
    args = ["--data", d_data, "--out", out, "--encoder", a.encoder, "--head", a.head, "--epochs", a.epochs, "--batch", a.batch, "--lr-enc", a.lr_enc, "--lr-head", a.lr_head,
            "--eval-every", a.eval_every, "--device", a.device, "--amp", a.amp, "--seed", a.seed]
    if a.max_steps:
        args += ["--max-steps", a.max_steps]
    if os.path.exists(last) and not a.fresh:
        say(f"train: resuming from {last} (use --fresh to start over)")
        args += ["--resume", last]
    args += a.train_args
    run("train_e5_translator.py", args)


def stage_export(a, out):
    ck = os.path.join(out, "e5gt_best.pt")
    if not os.path.exists(ck):
        sys.exit(f"[trainer_e5] no checkpoint at {ck}: train first")
    run("export_e5_translator.py", [ck, os.path.join(out, "e5gt.npz")] + (["--fp16"] if a.fp16 else []))


def stage_compare(a, out, d_data):
    npz = os.path.join(out, "e5gt.npz")
    if os.path.exists(npz):
        run("compare_translators.py", ["--e5", npz, "--data", d_data])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog="Unknown flags after `--` go to train_e5_translator.py.")
    ap.add_argument("--stage", choices=["all", "data", "train", "export", "compare"], default="all")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "e5gt"))
    ap.add_argument("--force", action="store_true", help="rebuild the dataset even if it exists")
    ap.add_argument("--fresh", action="store_true", help="do not resume from e5gt_last.pt")
    g = ap.add_argument_group("data")
    g.add_argument("--clips", default=os.path.join(ROOT, "data", "clips.npz"))
    g.add_argument("--gen-clips", type=int, default=500, help="clips to generate with gen_dataset.py if --clips is missing (0 = never; synthetic only)")
    g.add_argument("--regen-clips", action="store_true")
    g.add_argument("--synthetic", type=int, default=3000, help="synthetic goal sequences (0 = none)")
    g.add_argument("--synth-train-only", action="store_true", help="every synthetic row goes to train, so val/test are real clips only")
    g.add_argument("--prior", default=os.path.join(ROOT, "runs", "cvqvae", "prior_best.pt"))
    g.add_argument("--vae", default=os.path.join(ROOT, "runs", "cvqvae", "best.pt"))
    g.add_argument("--stats"); g.add_argument("--timing", default=""); g.add_argument("--rewrites"); g.add_argument("--rewriter", choices=["rules", "none"], default="rules")
    g.add_argument("--per-clip", type=int, default=6)
    t = ap.add_argument_group("train")
    t.add_argument("--encoder", default="intfloat/e5-small-v2"); t.add_argument("--head", default="small"); t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--max-steps", type=int, default=0); t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr-enc", type=float, default=3e-5); t.add_argument("--lr-head", type=float, default=3e-4)
    t.add_argument("--eval-every", type=int, default=500); t.add_argument("--device", default="auto"); t.add_argument("--amp", default="auto"); t.add_argument("--seed", type=int, default=0)
    e = ap.add_argument_group("export / compare")
    e.add_argument("--fp16", action="store_true", help="export 2-D matrices as float16 (about half the size)")
    e.add_argument("--compare", action="store_true", help="run compare_translators.py after export")
    a, rest = ap.parse_known_args()
    a.train_args = [x for x in rest if x != "--"]

    for k in ("clips", "prior", "vae", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d_data = os.path.join(a.out, "data")
    if a.stage in ("all", "data"):
        d_data = stage_data(a, a.out)
    if a.stage in ("all", "train"):
        stage_train(a, a.out, d_data)
    if a.stage in ("all", "export"):
        stage_export(a, a.out)
    if a.stage == "compare" or (a.stage == "all" and a.compare):
        stage_compare(a, a.out, d_data)
    say(f"done in {time.time() - t0:.0f}s. Use it: python UI/server.py ... (see UI/README.md) with {os.path.join(a.out, 'e5gt.npz')}")


if __name__ == "__main__":
    main()
