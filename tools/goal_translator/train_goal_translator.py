#!/usr/bin/env python3
"""
train_goal_translator.py - train the NANITY Goal Translator (spec section 4 and 9): a tiny decoder-only transformer in the NANITY architecture that reads BYTES and writes a goal
plan (grammar in gt_conventions.py). The architecture is training/modeling_nanity.py unchanged, so NEON loads the result with the code it already has.

  python build_dataset.py --clips clips.npz --stats prior_stats.json --out data/gt
  python train_goal_translator.py --data data/gt --out runs/gt --preset small
  python train_goal_translator.py --data data/gt --out runs/gt --resume runs/gt/gt_last.pt      # continue (e.g. after the rented GPU went away)
  python evaluate_goal_translator.py --ckpt runs/gt/gt_best.pt --data data/gt                  # held-out metrics + prompt suite, against stage C
  python export_goal_translator.py runs/gt/gt_best.pt goal_translator.gguf

Loss: next-token cross-entropy over the OUTPUT tokens only. Corruption (space removal, typos, case) is applied on the fly. Evaluation every --eval-every steps: validation loss and
constrained-greedy plan metrics on held-out SOURCE CLIPS; the lowest-validation-loss checkpoint is kept as gt_best.pt. CPU is enough for `tiny`; `small` wants a GPU for the
full dataset (a T4 or the MI300X are both fine: --amp auto picks bf16 / fp16 / none).

Honest notes: * a model trained on --synthetic data learns the text -> goals mapping only. * The metrics are worth what the held-out split is worth: clips split by source clip,
not by sentence. * It will not tell you what it ignored (the wave): the grammar has no token for that (spec section 11).
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch

import gt_conventions as C
import gt_metrics as M
import gt_torch as G


def pick_amp(mode, device):
    if mode == "none" or device == "cpu":
        return None
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return torch.bfloat16 if mode == "bf16" else torch.float16


@torch.no_grad()
def validation(model, val_rows, device, st, tm, n_decode, seed=0):
    model.eval()
    ds = G.GTDataset(val_rows, augment=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=G.collate)
    tot, cnt = 0.0, 0
    for ids, lab in dl:
        ids, lab = ids.to(device), lab.to(device)
        n = int((lab[:, 1:] != -100).sum())
        tot += float(G.loss_fn(model, ids, lab)) * n
        cnt += n
    rng = random.Random(seed)
    sub = rng.sample(val_rows, min(n_decode, len(val_rows)))
    scores, failed = [], 0
    for r in sub:
        try:
            pred = G.decode_plan(model, r["text"], device)
        except C.PlanError:
            failed += 1
            pred = C.Plan([])
        scores.append(M.score_pair(pred, M.plan_from_rows(r["windows"]), st, tm))
    agg = M.aggregate(scores)
    agg["plan_errors"] = failed
    return tot / max(cnt, 1), agg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by build_dataset.py (train.jsonl, val.jsonl, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=list(G.PRESETS), default="small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-decode", type=int, default=300, help="held-out rows decoded per evaluation (greedy decoding is sequential)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    if meta["layout_version"] != C.LAYOUT_VERSION or meta["n_turn_bins"] != C.N_TURN_BINS or meta["n_speed_bins"] != C.N_SPEED_BINS:
        sys.exit("the dataset was built with a different token layout than gt_conventions.py: rebuild it")
    st, tm = C.Stats.from_dict(meta["stats"]), C.Timing(**meta["timing"])
    train_rows, val_rows = G.load_rows(os.path.join(a.data, "train.jsonl")), G.load_rows(os.path.join(a.data, "val.jsonl"))
    if not train_rows or not val_rows:
        sys.exit("empty train or val split: too few clips")
    train_clips, val_clips = {r["clip"] for r in train_rows}, {r["clip"] for r in val_rows}
    assert not (train_clips & val_clips), "train and val share a source clip: the held-out split is broken"

    if a.resume:
        model, cfg, ck = G.load_checkpoint(a.resume, device)
        step0 = int(ck["step"])
    else:
        cfg = G.make_config(a.preset)
        model, step0 = G.build_model(cfg).to(device), 0
    gt_meta = dict(layout_version=C.LAYOUT_VERSION, stats=meta["stats"], timing=meta["timing"], n_turn_bins=C.N_TURN_BINS, n_speed_bins=C.N_SPEED_BINS, max_windows=C.MAX_WINDOWS,
                   max_input_bytes=C.MAX_INPUT_BYTES, vocab_size=C.VOCAB_SIZE, turn_positive_is_left=C.TURN_POSITIVE_IS_LEFT, data=meta.get("source"), preset=a.preset)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"device {device}  preset {a.preset}  {nparam / 1e6:.2f} M parameters  train {len(train_rows)} rows / {len(train_clips)} clips  val {len(val_rows)} rows / {len(val_clips)} clips")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=nodecay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.95))
    if a.resume and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    ds = G.GTDataset(train_rows, augment=not a.no_augment, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=G.collate, num_workers=a.workers, drop_last=len(ds) > a.batch)
    total = a.max_steps or a.epochs * len(dl)
    amp = pick_amp(a.amp, device)
    scaler = torch.amp.GradScaler(enabled=(amp == torch.float16))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup)))))

    best = float("inf")
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    step, t0, run_loss, run_n = step0, time.time(), 0.0, 0
    done = False
    while not done:
        for ids, lab in dl:
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            ids, lab = ids.to(device), lab.to(device)
            with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=amp, enabled=amp is not None):
                loss = G.loss_fn(model, ids, lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); run_n += 1; step += 1
            if step % 50 == 0:
                print(f"step {step}/{total}  loss {run_loss / run_n:.4f}  lr {lr_at(step):.2e}  {(time.time() - t0) / (step - step0):.2f}s/step", flush=True)
                run_loss, run_n = 0.0, 0
            if step % a.eval_every == 0 or step >= total:
                vl, agg = validation(model, val_rows, device, st, tm, a.eval_decode, a.seed)
                rec = dict(step=step, val_loss=vl, **agg)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(f"  [val] step {step}: loss {vl:.4f}  direction {agg['direction_acc']:.3f}  order {agg['order_acc']:.3f}  exact {agg['exact']:.3f}  "
                      f"turn_err {agg['turn_err_bins']:.2f} bins  dur_err {agg['dur_err_windows']:.2f} win  unspec P/R {agg['unspec_precision']:.2f}/{agg['unspec_recall']:.2f}  "
                      f"plan_errors {int(agg['plan_errors'])}", flush=True)
                G.save_checkpoint(os.path.join(a.out, "gt_last.pt"), model, cfg, gt_meta, step, rec, opt)
                if vl < best:
                    best = vl
                    G.save_checkpoint(os.path.join(a.out, "gt_best.pt"), model, cfg, gt_meta, step, rec)
                    print(f"  saved gt_best.pt (val loss {vl:.4f})")
            if step >= total:
                done = True
                break
    print(f"done: {step} steps, best val loss {best:.4f}. Next: python evaluate_goal_translator.py --ckpt {a.out}/gt_best.pt --data {a.data}")


if __name__ == "__main__":
    main()
