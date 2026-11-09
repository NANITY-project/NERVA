#!/usr/bin/env python3
"""
text_prior_pilot.py - does a prior conditioned on TEXT work at all?   (a pilot: it imports CodePrior & co. from train_cvqvae.py and changes nothing there)

The prior's numeric goal is a per-window vector (goal_mlp output) ADDED to the input embedding of every position of that window. Here that vector comes from a text encoder
instead: the phrase of each window -> encoder -> d numbers -> same place, same learned NULL vector (text dropout = classifier-free guidance). No extra tokens, so the context stays
[NULL] + windows x (n codes + SEP).

  --encoder byte   a small byte-level encoder (2 transformer layers, masked mean pool) trained jointly with the prior: your own, no pretrained weights
  --encoder e5     a FROZEN pretrained sentence encoder (needs `pip install transformers`) + a trained 2-layer projection. Knows synonyms it was never trained on.
  --encoder goal   the baseline: the normal numeric-goal prior (train_cvqvae.train_prior) on the same windows. Upper bound for steering.

Needs, once (the codes of your VAE):
  python tools/train_cvqvae.py --data DATA --out OUT --vae-ckpt OUT/best.pt --stage prior --export-codes --eval-only        # writes OUT/codes_train.npz, codes_val.npz
Then:
  python tools/text_prior_pilot.py --data DATA --codes OUT --vae-ckpt OUT/best.pt --out OUT/pilot --encoder byte
  python tools/text_prior_pilot.py --data DATA --codes OUT --vae-ckpt OUT/best.pt --out OUT/pilot --encoder e5
  python tools/text_prior_pilot.py --data DATA --codes OUT --vae-ckpt OUT/best.pt --out OUT/pilot --encoder goal
  (--ckpt OUT/pilot/text_prior_byte.pt skips training and only runs the tests; --max-windows 20000 --epochs 2 is a quick smoke run)

Captions (text_pilot_captions.py): each window's measured (turn, speed) -> a phrase from a TRAIN vocabulary, half of them corrupted (no spaces / commas, typos, "??").
The four tests:
  1 text is used   val NLL with the true phrase vs a phrase of another sequence vs the null vector. true << shuffled ~ null  = the encoder is used.
  2 steering       generate from phrases of the train vocabulary, decode, measure each window (turn, speed): how often it did what the phrase asked.
  3 raw text       the same intents typed raw: no spaces / commas, "??", typos.
  4 unseen words   synonyms that never appear in training ("sprint", "stroll", "hang a left", "anticlockwise").
Tests 1-3 only show that the mechanism learns the templates. Test 4 is the real question; expect the byte encoder to fail it unless the captions are far more varied.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cvqvae as T
import text_pilot_captions as TC

MAXB = 64                                                       # bytes of one phrase


# ============================================================================================================================ model
class ByteEncoder(nn.Module):
    """phrase bytes [M,L] (0 = pad) -> [M,out]: byte + position embedding, bidirectional transformer layers, masked mean pool, linear."""

    def __init__(self, out, d=128, layers=2, heads=4, L=MAXB):
        super().__init__()
        self.emb = nn.Embedding(256, d); self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, out)

    def forward(self, ids):
        pad = ids == 0
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        x = self.norm(self.enc(x, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).to(x.dtype)
        return self.out((x * keep).sum(1) / keep.sum(1).clamp(min=1))


class TextCodePrior(T.CodePrior):
    """CodePrior whose per-window conditioning vector comes from text. kind 'byte': text = phrase bytes [B,W,L] (long); kind 'vec': text = precomputed sentence vectors [B,W,in_dim]."""

    def __init__(self, K, n_actions, n, windows, d=256, layers=6, heads=4, dropout=0.1, kind="byte", in_dim=0):
        super().__init__(K, n_actions, n, windows, d, layers, heads, dropout, goal_dim=0, n_styles=0)
        self.kind, self.in_dim = kind, in_dim
        self.text_enc = ByteEncoder(d) if kind == "byte" else nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.text_enc.apply(self._init)
        self.text_null = nn.Parameter(torch.zeros(d)); nn.init.normal_(self.text_null, std=0.02)

    def forward(self, tok, text=None, slot_ok=None, drop_text=None):
        """tok [B,T]; text [B,windows,...] (slot j = the j-th window of the context) or None = null for all; slot_ok [B,windows] bool (False = null for that slot);
        drop_text [B] bool (True = null for the whole sequence)."""
        B, Tn = tok.shape; dev = tok.device; W = self.windows
        x = self.tok(tok) + self.pos(torch.arange(Tn, device=dev))[None]
        if text is None:
            g = self.text_null.expand(B, W, -1)
        else:
            g = self.text_enc(text.reshape(B * W, -1)).reshape(B, W, -1) if self.kind == "byte" else self.text_enc(text)
            null = self.text_null.to(g.dtype)
            if slot_ok is not None:
                g = torch.where(slot_ok.unsqueeze(-1), g, null)
            if drop_text is not None:
                g = torch.where(drop_text.reshape(B, 1, 1), null, g)
        x = x + g[:, (torch.arange(Tn, device=dev) // (self.n + 1)).clamp(max=W - 1)]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def text_loss(prior, tok, text, ok, drop, amp, dev):
    inp, tgt = tok[:, :-1], tok[:, 1:]
    with T.amp_ctx(dev, amp):
        lg = prior(inp, text, ok, drop)
    lg = lg.float().reshape(-1, prior.K + 1); t = tgt.reshape(-1); valid = t != prior.PAD
    ce = F.cross_entropy(lg, torch.where(valid, t, torch.full_like(t, -100)), reduction="none", ignore_index=-100)
    code = valid & (t < prior.K)
    return ce[valid].mean(), ce[code].mean().detach(), (lg.argmax(-1) == t)[code].float().mean().detach()


# ============================================================================================================================ text tables
def byte_table(strings, L=MAXB):
    tab = np.zeros((len(strings), L), np.uint8)
    for i, s in enumerate(strings):
        b = s.encode("utf-8", "ignore")[:L] or b" "
        tab[i, :len(b)] = np.frombuffer(b, np.uint8)
    return tab


def e5_table(strings, model_name, dev, bs=256):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("--encoder e5 needs transformers: pip install transformers")
    tk = AutoTokenizer.from_pretrained(model_name); m = AutoModel.from_pretrained(model_name).to(dev).eval(); out = []
    with torch.no_grad():
        for i in range(0, len(strings), bs):
            b = tk(["query: " + s for s in strings[i:i + bs]], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            h = m(**b).last_hidden_state; msk = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(F.normalize((h * msk).sum(1) / msk.sum(1), dim=-1).float().cpu())
    return torch.cat(out).numpy()


def make_slots(chain, Lw):
    """for the sequence that starts at window i: the window index of each slot and whether that slot is still in the same chain."""
    N = len(chain); idx = np.arange(N)[:, None] + np.arange(Lw)[None]; nc = np.minimum(idx, N - 1)
    return nc.astype(np.int64), (idx < N) & (chain[nc] == chain[:, None])


def get_text(D, ix, rand):
    """sequence indices ix [B] -> (text [B,Lw,...], ok [B,Lw]). rand: a random caption variant per window (training); else variant 0 (evaluation)."""
    wi, ok = D["WI"][ix], D["OK"][ix]
    v = torch.randint(0, D["SID"].shape[1], wi.shape, device=wi.device) if rand else torch.zeros_like(wi)
    tab = D["TAB"][D["SID"][wi, v]]
    return (tab.long() if D["kind"] == "byte" else tab), ok


# ============================================================================================================================ train / eval
@torch.no_grad()
def eval_nll(prior, Tv, Dv, a, dev, mode, seed=0):
    """mode true | shuffled (the phrases of another, random sequence) | null."""
    prior.eval(); s = 0.0; nb = 0; bs = 512
    perm = torch.from_numpy(np.random.RandomState(seed).permutation(len(Tv))).to(dev)
    for i in range(0, len(Tv), bs):
        ix = torch.arange(i, min(i + bs, len(Tv)), device=dev); text, ok = get_text(Dv, ix, False)
        if mode == "shuffled":
            text, _ = get_text(Dv, perm[ix], False)
        drop = torch.ones(len(ix), dtype=torch.bool, device=dev) if mode == "null" else None
        s += float(text_loss(prior, Tv[ix], text, ok, drop, a.amp, dev)[1]); nb += 1
    prior.train(); return s / max(nb, 1)


def train_text(a, dev, prior, Tt, Tv, Dt, Dv, path, extra):
    decay = [p for _, p in prior.named_parameters() if p.ndim >= 2]; nod = [p for _, p in prior.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=nod, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.95))
    spe = max(1, len(Tt) // a.batch); total = a.epochs * spe; warm = min(200, total // 10)
    lr_at = lambda s: a.lr * (s / max(1, warm) if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16" and dev.type == "cuda"))
    best, bad, step, t0 = float("inf"), 0, 0, time.time(); log = open(path.replace(".pt", "_metrics.jsonl"), "a"); prior.train()
    for epoch in range(a.epochs):
        perm = torch.randperm(len(Tt), device=dev); run = np.zeros(3); nb = 0
        for bi in range(spe):
            ix = perm[bi * a.batch:(bi + 1) * a.batch]; text, ok = get_text(Dt, ix, True)
            drop = torch.rand(len(ix), device=dev) < a.text_drop
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, nll, acc = text_loss(prior, Tt[ix], text, ok, drop, a.amp, dev)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0); scaler.step(opt); scaler.update(); step += 1
            run += [float(loss), float(nll), float(acc)]; nb += 1
        run /= max(nb, 1); vt = eval_nll(prior, Tv, Dv, a, dev, "true"); vn = eval_nll(prior, Tv, Dv, a, dev, "null"); improved = vt < best
        if improved:
            best, bad = vt, 0
            torch.save(dict(model=prior.state_dict(), cfg=extra["cfg"], val_nll=best, args=vars(a), e5_model=extra.get("e5_model")), path)
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train nll {run[1]:.3f} acc {run[2] * 100:.1f}%  |  val nll {vt:.3f} (text nulled {vn:.3f})  |  {(time.time() - t0) / 60:.1f} min{'  *best*' if improved else ''}", flush=True)
        log.write(json.dumps(dict(epoch=epoch, train_nll=run[1], val_nll=vt, val_null=vn)) + "\n"); log.flush()
        if a.patience and bad >= a.patience:
            print(f"stopping: no validation improvement for {a.patience} epochs"); break
    return best


# ============================================================================================================================ steering: sample, decode, measure
def cond_sample(prior, text, nw, a, dev):
    """text [B,Lw,...] (the same phrase in every slot) -> codes [B,nw,n]. The null prefix token, windows sampled code by code, the oldest window dropped when the context is full."""
    prior.eval(); B, n, Lw = text.shape[0], prior.n, prior.windows; ctx = torch.full((B, 1), prior.NULL, dtype=torch.long, device=dev); out = []
    use_cfg = a.cfg_scale != 1.0; ones = torch.ones(B, dtype=torch.bool, device=dev)
    for w in range(nw):
        if w >= Lw:
            ctx = torch.cat([ctx[:, :1], ctx[:, 1 + n + 1:]], 1)
        for _ in range(n):
            if use_cfg:
                both = prior(torch.cat([ctx, ctx]), torch.cat([text, text]), None, torch.cat([~ones, ones]))[:, -1, :prior.K].float()
                lg = both[B:] + a.cfg_scale * (both[:B] - both[B:])
            else:
                lg = prior(ctx, text)[:, -1, :prior.K].float()
            nxt = torch.multinomial(torch.softmax(T.filter_logits(lg / max(a.temperature, 1e-4), a.top_k, a.top_p), -1), 1); ctx = torch.cat([ctx, nxt], 1)
        out.append(ctx[:, -n:])
        if w < nw - 1:
            ctx = torch.cat([ctx, torch.full_like(ctx[:, :1], prior.SEP)], 1)
    return torch.stack(out, 1)


def measure(vae, norm, canon, codes, n_null, dt, dev):
    """codes [B,nw,n] -> (dyaw_deg [B,nw], speed [B,nw]) read off the decoded windows with the same goal_features the training goals use."""
    B, nw, n = codes.shape
    y = vae.decode_codes(codes.reshape(B * nw, n), torch.full((B * nw,), n_null, dtype=torch.long, device=dev)); win = T.to_motion(y, norm)
    win = win if canon else T.canonicalize(win)
    g = T.goal_features(win, np.full(len(win), win.shape[1], np.int64), dt)
    return np.degrees(g[:, 0]).reshape(B, nw), g[:, 3].reshape(B, nw)


@torch.no_grad()
def run_probes(a, dev, prior, vae, norm, canon, dt, n_null, tab_fn, include_run):
    res = {}; ps = TC.probes(a.seed, include_run)
    for kind, items in ps.items():
        rows = []
        for p in items:
            torch.manual_seed(a.seed)
            vec = tab_fn(p["text"]); text = vec[None, None].expand(a.probe_n, prior.windows, -1).contiguous()
            codes = cond_sample(prior, text, a.probe_windows, a, dev)
            d, v = measure(vae, norm, canon, codes, n_null, dt, dev)
            rows.append(dict(sign=p["sign"], speed=p["speed"], text=p["text"], dyaw_deg=d[:, 1:].reshape(-1) if a.probe_windows > 1 else d.reshape(-1),
                             speed_ms=v[:, 1:].reshape(-1) if a.probe_windows > 1 else v.reshape(-1)))     # the first window has no earlier window to settle into: skipped
        res[kind] = rows
    return res


def report(res, title):
    print(f"\n{title}\n  {'kind':14s} {'turn ok':>8s} {'speed ok':>9s} {'both ok':>8s}   windows")
    for kind, rows in res.items():
        if not rows:
            continue
        s = TC.summarize(rows); print(f"  {kind:14s} {s['turn_ok'] * 100:7.0f}% {s['speed_ok'] * 100:8.0f}% {s['both_ok'] * 100:7.0f}%   {s['n']}")
    return {k: TC.summarize(r) for k, r in res.items() if r}


def show_worst(res, k=4):
    for kind, rows in res.items():
        sc = sorted(rows, key=lambda r: sum(map(lambda x: x.mean(), TC.window_ok(r["sign"], r["speed"], r["dyaw_deg"], r["speed_ms"]))))[:k]
        for r in sc:
            print(f"  worst {kind:13s} {r['text']!r:44s} wanted turn {r['sign']:+d} {r['speed']:5s}  got {np.mean(r['dyaw_deg']):+6.1f} deg/window  {np.mean(r['speed_ms']):.2f} m/s")


# ============================================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the window files (train_windows.npz, val_windows.npz)")
    ap.add_argument("--codes", required=True, help="the folder with codes_train.npz / codes_val.npz (train_cvqvae.py --export-codes)")
    ap.add_argument("--vae-ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["byte", "e5", "goal"], default="byte"); ap.add_argument("--e5-model", default="intfloat/e5-small-v2")
    ap.add_argument("--ckpt", help="a text prior trained by this script: skip training, run the tests"); ap.add_argument("--max-windows", type=int, default=0, help="debug: use only the first N windows")
    ap.add_argument("--variants", type=int, default=4); ap.add_argument("--pool", type=int, default=96); ap.add_argument("--p-corrupt", type=float, default=0.5)
    ap.add_argument("--windows", type=int, default=4); ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--text-drop", type=float, default=0.15, help="share of sequences whose text becomes the NULL vector (also the goal drop of --encoder goal)")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    ap.add_argument("--probe-n", type=int, default=24, help="clips sampled per probe phrase"); ap.add_argument("--probe-windows", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--top-k", type=int, default=40); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=1.0); ap.add_argument("--show-worst", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.amp == "auto":
        a.amp = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if dev.type == "cuda" else "none"
    torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ data
    vae, ck, norm = T.load_ckpt(a.vae_ckpt, dev); vae.eval(); canon = not ck["args"].get("no_canon", False); K = vae.vq.K; n_null = len(ck["action_names"])
    Xt, vt, _, _, dt = T.load_windows(os.path.join(a.data, "train_windows.npz")); Xv, vv, _, _, dtv = T.load_windows(os.path.join(a.data, "val_windows.npz"))
    zt = np.load(os.path.join(a.codes, "codes_train.npz")); zv = np.load(os.path.join(a.codes, "codes_val.npz"))
    codes_t, codes_v = zt["codes"].astype(np.int64), zv["codes"].astype(np.int64); chain_t, chain_v = zt["chain"].astype(np.int64), zv["chain"].astype(np.int64)
    if len(codes_t) != len(Xt) or len(codes_v) != len(Xv):
        sys.exit(f"codes ({len(codes_t)} / {len(codes_v)}) and windows ({len(Xt)} / {len(Xv)}) do not line up: export the codes from the same --data and --vae-ckpt")
    if a.max_windows:
        m = a.max_windows; Xt, vt, codes_t, chain_t = Xt[:m], vt[:m], codes_t[:m], chain_t[:m]; mv = max(m // 8, 1000); Xv, vv, codes_v, chain_v = Xv[:mv], vv[:mv], codes_v[:mv], chain_v[:mv]
    gt = T.goal_features(T.canonicalize(Xt), vt, dt); gv = T.goal_features(T.canonicalize(Xv), vv, dtv); gstats = T.goal_stats(gt)
    n = codes_t.shape[1]; Lw = a.windows; uni_val = T.unigram_ce(codes_t, codes_v, K); W = Xt.shape[1]
    sc = TC.keys(gt)[:, 0]; share = np.bincount(sc, minlength=5) / len(sc)
    print(f"{len(codes_t)} train / {len(codes_v)} val windows, {n} codes each, K {K}, canonical {canon}, device {dev}, amp {a.amp}; unigram baseline {uni_val:.3f} nats")
    print("windows by speed class: " + ", ".join(f"{c} {s * 100:.1f}%" for c, s in zip(TC.SPEED_CLASSES, share)) + f"   (speed_max {gstats['speed_max']:.2f} m/s)")
    include_run = share[4] >= 0.005
    if not include_run:
        print("note: under 0.5% of the windows are running speed, so 'run' probes are skipped (the data can not teach it)")
    at_t, at_v = np.ones(len(codes_t), np.int64), np.ones(len(codes_v), np.int64)             # n_actions = 1: the label prefix is always the NULL token

    # ------------------------------------------------------------------ baseline: numeric goals
    if a.encoder == "goal":
        gdir = os.path.join(a.out, "pilot_goal"); os.makedirs(gdir, exist_ok=True)
        ns = argparse.Namespace(prior_windows=Lw, prior_dim=a.dim, prior_layers=a.layers, prior_heads=a.heads, prior_dropout=a.dropout, prior_wd=a.wd, prior_lr=a.lr, prior_batch=a.batch,
                                prior_epochs=a.epochs, prior_patience=a.patience, prior_cond_drop=0.0, prior_goal_drop=a.text_drop, prior_style_drop=0.0, prior_group_mask_win=0.0,
                                prior_group_mask_seq=0.0, amp=a.amp, out=gdir)
        prior = T.train_prior(ns, dev, codes_t, codes_v, chain_t, chain_v, at_t, at_v, K, 1, ["null"], W // 2, dt, uni_val, gt, gv, gstats, None, None, None)
        res = {"numeric_goal": []}
        with torch.no_grad():
            for sc_i, name in enumerate(TC.SPEED_CLASSES):
                if name == "run" and not include_run:
                    continue
                for sign in (0, 1, -1):
                    if name == "stand" and sign != 0:
                        continue
                    torch.manual_seed(a.seed); spd = {"stand": 0.0, "slow": 0.5, "walk": 1.2, "fast": 2.0, "run": 3.0}[name]
                    goals = T.plan_goals(prior, a.probe_n, a.probe_windows, 30.0 * sign, spd, dt, W, dev)
                    codes = T.generate_codes(prior, torch.full((a.probe_n,), prior.NULL - prior.ACT0, device=dev), a.probe_windows, a.temperature, a.top_k, a.top_p, goals, None, a.cfg_scale)
                    d, v = measure(vae, norm, canon, codes, n_null, dt, dev)
                    res["numeric_goal"].append(dict(sign=sign, speed=name, text=f"goal {30 * sign} deg {spd} m/s", dyaw_deg=d[:, 1:].reshape(-1), speed_ms=v[:, 1:].reshape(-1)))
        report(res, "numeric-goal prior (the upper bound for steering)")
        if a.show_worst:
            show_worst(res)
        return

    # ------------------------------------------------------------------ captions + tables
    strings_t, sid_t = TC.build(gt, a.variants, a.pool, a.p_corrupt, a.seed)
    strings_v, sid_v = TC.build(gv, a.variants, a.pool, a.p_corrupt, a.seed + 1)           # validation captions come from their own pool (same vocabulary, other draws)
    allstr = list(dict.fromkeys(strings_t + strings_v)); idx = {s_: i for i, s_ in enumerate(allstr)}
    sid_t = np.array([idx[s_] for s_ in strings_t], np.int64)[sid_t]; sid_v = np.array([idx[s_] for s_ in strings_v], np.int64)[sid_v]
    print(f"captions: {len(allstr)} distinct phrases; e.g. " + " | ".join(repr(allstr[i]) for i in np.random.RandomState(1).choice(len(allstr), 4, replace=False)))
    ck_in = torch.load(a.ckpt, map_location="cpu", weights_only=False) if a.ckpt else None
    e5_name = (ck_in or {}).get("e5_model") or a.e5_model
    kind = "byte" if a.encoder == "byte" else "vec"
    if ck_in is not None:
        kind = ck_in["cfg"]["kind"]
    if kind == "byte":
        tab_np = byte_table(allstr); in_dim = 0
        tab_fn = lambda s: torch.from_numpy(byte_table([s])[0]).long().to(dev)
    else:
        tab_np = e5_table(allstr, e5_name, dev); in_dim = tab_np.shape[1]
        probe_strings = [p["text"] for items in TC.probes(a.seed, include_run).values() for p in items]; probe_vecs = dict(zip(probe_strings, e5_table(probe_strings, e5_name, dev)))
        tab_fn = lambda s: torch.from_numpy(probe_vecs[s]).float().to(dev)
    tab = torch.from_numpy(tab_np).to(dev)
    Tt = torch.from_numpy(T.make_prior_tokens(codes_t, chain_t, at_t, Lw, K, 1)).to(dev); Tv = torch.from_numpy(T.make_prior_tokens(codes_v, chain_v, at_v, Lw, K, 1)).to(dev)
    wi_t, ok_t = make_slots(chain_t, Lw); wi_v, ok_v = make_slots(chain_v, Lw)
    Dt = dict(WI=torch.from_numpy(wi_t).to(dev), OK=torch.from_numpy(ok_t).to(dev), SID=torch.from_numpy(sid_t).to(dev), TAB=tab, kind=kind)
    Dv = dict(WI=torch.from_numpy(wi_v).to(dev), OK=torch.from_numpy(ok_v).to(dev), SID=torch.from_numpy(sid_v).to(dev), TAB=tab, kind=kind)
    cfg = dict(K=K, n_actions=1, n=n, windows=Lw, d=a.dim, layers=a.layers, heads=a.heads, dropout=a.dropout, kind=kind, in_dim=in_dim)
    if ck_in is not None:
        cfg = ck_in["cfg"]
    prior = TextCodePrior(**cfg).to(dev); npar = sum(p.numel() for p in prior.parameters())
    print(f"text prior ({a.encoder if not ck_in else 'from ' + a.ckpt}): {npar / 1e6:.2f} M parameters, {len(Tt)} train / {len(Tv)} val sequences of up to {Lw} windows")

    # ------------------------------------------------------------------ train
    path = os.path.join(a.out, f"text_prior_{a.encoder}.pt")
    if ck_in is None:
        train_text(a, dev, prior, Tt, Tv, Dt, Dv, path, dict(cfg=cfg, e5_model=e5_name if kind == "vec" else None))
        prior.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    else:
        prior.load_state_dict(ck_in["model"])
    prior.eval()

    # ------------------------------------------------------------------ test 1: is the text used?
    tr, sh, nu = (eval_nll(prior, Tv, Dv, a, dev, m) for m in ("true", "shuffled", "null"))
    print(f"\n1  text is used?   val nll per code: true phrase {tr:.3f} | phrase of another sequence {sh:.3f} | no text {nu:.3f} | unigram {uni_val:.3f}")
    print(f"   -> {'YES: the true phrase is worth ' + format(sh - tr, '.3f') + ' nats over a wrong one' if sh - tr > 0.03 else 'NO (or barely): a wrong phrase is as good as the right one: the encoder is not being used'}")

    # ------------------------------------------------------------------ tests 2-4: steering
    res = run_probes(a, dev, prior, vae, norm, canon, dt, n_null, tab_fn, include_run)
    sm = report(res, "2-4  steering: how often the decoded windows did what the phrase asked (probe phrases are typed once and repeated in every window)")
    print("   2 = in_vocab, 3 = raw, 4 = unseen_speed / unseen_turn / unseen_both / unseen_dir.  Chance is about 33% for turn (left / right / straight) and 20-25% for speed.")
    if a.show_worst:
        show_worst(res)
    json.dump(dict(encoder=a.encoder, nll=dict(true=tr, shuffled=sh, null=nu, unigram=uni_val), steering=sm), open(os.path.join(a.out, f"pilot_{a.encoder}.json"), "w"), indent=1)
    print(f"\nwrote {path} and {os.path.join(a.out, f'pilot_{a.encoder}.json')}")


if __name__ == "__main__":
    main()
