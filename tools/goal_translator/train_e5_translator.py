#!/usr/bin/env python3
"""
train_e5_translator.py - fine-tune a pretrained text encoder (default intfloat/e5-small-v2, MIT: read the model card of the exact checkpoint you use) plus a small window decoder to
turn free text into a goal plan. No word list anywhere: the encoder brings the language, your data teaches it the motion.

  python build_dataset.py --clips clips.npz --stats prior_stats.json --out data/gt                 # the same dataset the byte model uses
  python train_e5_translator.py --data data/gt --out runs/e5gt                                      # downloads the encoder once (needs the Hugging Face hub)
  python train_e5_translator.py --data data/gt --out runs/e5gt --resume runs/e5gt/e5gt_last.pt      # continue
  python export_e5_translator.py runs/e5gt/e5gt_best.pt e5gt.npz                                    # numpy file for UI/goal_plan.py (no torch at inference)

Training: the encoder is frozen for the first --freeze-steps so the randomly initialised decoder settles without wrecking pretrained weights; then both train, encoder at --lr-enc
(small), decoder at --lr-head. Text is corrupted on the fly (typos, missing spaces, case) with probability --corrupt-p. Held-out = whole source clips (the dataset's split).
Every --eval-every steps: validation loss, plan metrics (gt_metrics) on a held-out sample, and the fixed prompt suite, against stage C. Lowest validation loss is kept as e5gt_best.pt.

Honest notes: * This replaces the 33 turn bins and 16 speed bins with continuous outputs, but the LABELS in the dataset are still the bin centres (build_dataset stores bins), so
precision is bounded by them (about 3 degrees / 0.14 m/s). * The vocabulary is open only on the text side: what the character can DO is still what the prior's goals express
(path and speed). * If the dataset's phrasings were filtered by the stage C parser, the model only saw phrasings that parser understands: build the data with a lexicon-free
verifier (or more rewrites) before judging how well this generalises. * If the held-out numbers are good and the prompt suite is bad, the suite is the better test of messy
real text.
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch

import gt_conventions as C
import gt_metrics as M
import e5_common as E
import e5_torch as L

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_amp(mode, device):
    if mode == "none" or device == "cpu":
        return None
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return torch.bfloat16 if mode == "bf16" else torch.float16


def lr_factor(step, total, warmup, floor=0.1):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def validation(model, wp, val_rows, device, st, tm, amp, pad_id, n_decode, seed=0):
    model.eval()
    dl = torch.utils.data.DataLoader(L.E5Dataset(val_rows, augment=False), batch_size=64, collate_fn=L.make_collate(wp, pad_id))
    tot, cnt = 0.0, 0
    for ids, mask, tg in dl:
        ids, mask, tg = ids.to(device), mask.to(device), {k: v.to(device) for k, v in tg.items()}
        with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
            out = model(ids, mask)
        loss, _ = L.loss_fn(out, tg)
        tot += float(loss) * ids.shape[0]
        cnt += ids.shape[0]
    sub = random.Random(seed).sample(val_rows, min(n_decode, len(val_rows)))
    with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
        plans = L.predict_plans(model, wp, [r["text"] for r in sub], device, st, pad_id=pad_id)
    scores = [M.score_pair(p, M.plan_from_rows(r["windows"]), st, tm) for p, r in zip(plans, sub)]
    return tot / max(cnt, 1), M.aggregate(scores)


@torch.no_grad()
def run_suite(model, wp, device, st, tm, amp, pad_id, verbose=False):
    suite = M.load_suite(os.path.join(HERE, "prompt_suite.json"))
    with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
        plans = L.predict_plans(model, wp, [c["text"] for c in suite], device, st, pad_id=pad_id)
    ok = [M.suite_check(p, c, st) for p, c in zip(plans, suite)]
    if verbose:
        for p, c, k in zip(plans, suite, ok):
            if not k:
                print(f"    FAIL [{c.get('tier', '?')}] {c['text']!r}: expected {c.get('expect', '-')!r}, got {' '.join(M.collapse(p, st))!r}")
    base = None
    try:
        import stage_c as SC
        base = 0
        for c in suite:
            try:
                base += M.suite_check(SC.translate(c["text"], st, tm), c, st)
            except C.PlanError:
                pass
    except Exception:
        pass
    return sum(ok), len(suite), base


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by build_dataset.py (train.jsonl, val.jsonl, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="intfloat/e5-small-v2", help="a BERT-style encoder on the Hugging Face hub (or a local folder): check its license")
    ap.add_argument("--head", choices=list(L.HEAD_PRESETS), default="small")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--freeze-steps", type=int, default=300, help="steps with the encoder frozen at the start")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--w-len", type=float, default=1.0)
    ap.add_argument("--w-spec", type=float, default=1.0)
    ap.add_argument("--w-reg", type=float, default=3.0)
    ap.add_argument("--corrupt-p", type=float, default=0.5)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-decode", type=int, default=400)
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    amp = pick_amp(a.amp, device)
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    if meta["layout_version"] != C.LAYOUT_VERSION or meta["n_turn_bins"] != C.N_TURN_BINS or meta["n_speed_bins"] != C.N_SPEED_BINS:
        sys.exit("the dataset was built with a different token layout than gt_conventions.py: rebuild it")
    st, tm = C.Stats.from_dict(meta["stats"]), C.Timing(**meta["timing"])
    train_rows, val_rows = E.load_rows(os.path.join(a.data, "train.jsonl")), E.load_rows(os.path.join(a.data, "val.jsonl"))
    if not train_rows or not val_rows:
        sys.exit("empty train or val split: too few clips")
    assert not ({r["clip"] for r in train_rows} & {r["clip"] for r in val_rows}), "train and val share a source clip: the held-out split is broken"

    ck = None
    if a.resume:
        model, ck = L.load_checkpoint(a.resume, device)
        vocab, step0, enc_name = ck["vocab"], int(ck["step"]), ck["encoder_name"]
    else:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(a.encoder)
        if not getattr(hf_tok, "do_lower_case", True):
            sys.exit(f"{a.encoder} uses a cased vocabulary; e5_common.WordPiece implements the uncased BERT rules (lowercase, accents stripped): pick an uncased encoder")
        hf = hf_tok.get_vocab()
        vocab = [t for t, _ in sorted(hf.items(), key=lambda kv: kv[1])]
        assert sorted(hf.values()) == list(range(len(vocab))), "the tokenizer's ids are not contiguous"
        model, step0, enc_name = L.E5GoalTranslator.from_pretrained(a.encoder, a.head, a.dropout).to(device), 0, a.encoder
        if len(vocab) != model.encoder.config.vocab_size:
            sys.exit(f"vocab has {len(vocab)} entries but the encoder expects {model.encoder.config.vocab_size}")
    wp = E.WordPiece(vocab)
    pad_id = wp.ids.get("[PAD]", 0)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"device {device}  amp {amp}  params {nparams / 1e6:.1f}M (encoder {sum(p.numel() for p in model.encoder.parameters()) / 1e6:.1f}M)  train rows {len(train_rows)}  val rows {len(val_rows)}")

    def decays(n):
        return not (n.endswith("bias") or "LayerNorm" in n or ".ln" in n or "_ln" in n)
    groups = []
    for kind, mod, lr in (("enc", model.encoder, a.lr_enc), ("head", model.head, a.lr_head)):
        named = list(mod.named_parameters())
        groups.append(dict(params=[p for n, p in named if decays(n)], weight_decay=a.wd, base_lr=lr, kind=kind, lr=lr))
        groups.append(dict(params=[p for n, p in named if not decays(n)], weight_decay=0.0, base_lr=lr, kind=kind, lr=lr))
    opt = torch.optim.AdamW(groups)
    if ck is not None and "optimizer" in ck:
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:
            print("could not restore the optimizer state (continuing with a fresh one):", e)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp == torch.float16)

    ds = L.E5Dataset(train_rows, augment=not a.no_augment, corrupt_p=a.corrupt_p, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=L.make_collate(wp, pad_id), num_workers=a.workers, drop_last=True)
    total = a.max_steps or a.epochs * len(dl)
    frozen = step0 < a.freeze_steps
    model.encoder.requires_grad_(not frozen)
    step, best, t0, run = step0, float("inf"), time.time(), {}
    print(f"{len(dl)} steps per epoch, {total} steps in total; encoder frozen for the first {a.freeze_steps}")

    def evaluate():
        vl, agg = validation(model, wp, val_rows, device, st, tm, amp, pad_id, a.eval_decode)
        ok, n, base = run_suite(model, wp, device, st, tm, amp, pad_id)
        print(f"  [eval @ {step}] val loss {vl:.4f}   " + "  ".join(f"{k} {v:.3f}" for k, v in agg.items()) + f"   prompt suite {ok}/{n}" + (f" (stage C {base}/{n})" if base is not None else ""))
        model.train()
        return vl

    model.train()
    done = False
    while not done:
        for ids, mask, tg in dl:
            if step >= total:
                done = True
                break
            if frozen and step >= a.freeze_steps:
                model.encoder.requires_grad_(True); frozen = False
                print(f"  [step {step}] encoder unfrozen")
            f = lr_factor(step, total, a.warmup)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            ids, mask, tg = ids.to(device), mask.to(device), {k: v.to(device) for k, v in tg.items()}
            with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
                out = model(ids, mask)
            loss, parts = L.loss_fn(out, tg, a.w_len, a.w_spec, a.w_reg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            for k, v in parts.items():
                run[k] = 0.98 * run.get(k, v) + 0.02 * v
            if step % 50 == 0:
                print(f"  step {step}/{total}  loss {float(loss):.4f}  " + " ".join(f"{k} {v:.3f}" for k, v in run.items()) + f"  {(time.time() - t0) / (step - step0):.2f}s/step")
            if step % a.eval_every == 0:
                vl = evaluate()
                extra = dict(optimizer=opt.state_dict(), val_loss=vl)
                L.save_checkpoint(os.path.join(a.out, "e5gt_last.pt"), model, vocab, st, tm, enc_name, step, extra)
                if vl < best:
                    best = vl
                    L.save_checkpoint(os.path.join(a.out, "e5gt_best.pt"), model, vocab, st, tm, enc_name, step, dict(val_loss=vl))
                    print("  new best")
    vl = evaluate()
    L.save_checkpoint(os.path.join(a.out, "e5gt_last.pt"), model, vocab, st, tm, enc_name, step, dict(optimizer=opt.state_dict(), val_loss=vl))
    if vl < best or not os.path.exists(os.path.join(a.out, "e5gt_best.pt")):
        L.save_checkpoint(os.path.join(a.out, "e5gt_best.pt"), model, vocab, st, tm, enc_name, step, dict(val_loss=vl))
        print("  new best")
    print("final prompt suite:")
    ok, n, base = run_suite(model, wp, device, st, tm, amp, pad_id, verbose=True)
    print(f"  {ok}/{n}" + (f" (stage C {base}/{n})" if base is not None else ""))


if __name__ == "__main__":
    main()
