#!/usr/bin/env python3
"""
train_cvqvae.py - PyTorch trainer for a conditional VQ-VAE (cVQ-VAE) over NERVA motion windows, ~20M parameters.

  python tools/build_vae_dataset.py --hours 10                       # 1. make the data (data/vae/train_windows.npz, val_windows.npz, stats.json)
  python tools/train_cvqvae.py --data data/vae --out runs/cvqvae     # 2. train (cuda / ROCm picked up automatically)
  python tools/train_cvqvae.py --params-only                         # print the exact parameter count and exit
  python tools/train_cvqvae.py --data data/vae --epochs 1 --max-steps 30 --out /tmp/cv   # quick smoke run
  python tools/train_cvqvae.py --data data/vae --out runs/cvqvae --eval-only --ckpt runs/cvqvae/best.pt --export-recon 24 --export-codes
  python tools/view_clip.py runs/cvqvae/recon_preview.npz --open     # original / reconstruction pairs, same viewer as the dataset preview

Reads what tools/build_vae_dataset.py writes: windows [N, W, 30] (W = 64 by default), `valid` (real frames per window, the rest is padding), `action`
(index into `action_names`), and stats.json (channel names, which channels are binary flags, which are yaw angles, dt).

MODEL  ("c" = conditional on the ACTION label)
  encoder   Conv1d -> [FiLM residual blocks -> stride-2 conv] x 2          window 64 frames -> 16 latent steps (4x temporal downsampling)
  quantizer one codebook (512 x 256), EMA updates, k-means-style init from data, dead codes restarted from the current batch
  decoder   Conv1d -> [FiLM residual blocks -> nearest x2 + conv] x 2     -> 30 channels per frame
  The action label (plus a learned "null" label) becomes an embedding that FiLM-modulates every residual block of the encoder AND the decoder, so the codes
  only have to carry what the label does not say (style, speed, amplitude, ...). The label is dropped to "null" for --cond-drop of the training windows, so the
  same model also works unconditionally (needed later if you train a prior over the codes, with or without a label).
  --no-cond-enc  gives the label to the decoder only (the encoder always sees "null").
  Default sizes (--hidden 528 --code-dim 256 --n-res 3 --emb 128) come to about 19.9 M parameters; --params-only prints the exact number for the data's action count.

DATA HANDLING (the part that matters most for motion data)
  * Windows are CANONICALISED: every window is moved so frame 0 has the pelvis at x = z = 0 facing +z (all xz channels are rotated by the frame-0 pelvis yaw,
    all yaw channels get that yaw subtracted; same conventions as tools/motion_features.py / motion_vary.py: heading = (sin yaw, cos yaw)). Yaw channels are
    unwrapped along the window first. Without this the model would spend capacity on where in the world the clip happened. Use --no-canon to switch it off.
  * Normalisation statistics are recomputed from the canonicalised TRAIN windows (stats.json describes the raw data, so it is only used for the layout / dt);
    flags (contact / on) stay 0/1 in the targets and are predicted as logits (BCE); the encoder sees them as -1 / +1.
  * Padded windows (valid < W) are masked out of every loss and metric.

LOSS   masked smooth-L1 (or MSE) on the 24 continuous channels  +  velocity loss (frame differences, scaled by their own std)  +  BCE on the 6 flags
       + commitment loss (EMA codebook, so no codebook loss term).
METRICS (validation = performers the model never saw, the only honest number):
       loss terms, pos_rmse_m (RMS error of the 18 position channels in metres), flag_acc, perplexity (effective codes in use), codes_used (share of the codebook).
HONEST NOTE  build_vae_dataset.py itself warns that variants are repeats with variation, not new motion. A 20M-parameter model on a few hours of real
       information can memorise: watch the gap between train and val reconstruction, use --dropout / --patience, and judge by `val`.

Files written to --out: best.pt (lowest val total), last.pt (resumable: --resume), norm.json, metrics.jsonl, and on request recon_preview.npz / codes_{train,val}.npz.
"""
import argparse, collections, contextlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# action-vector layout (30): pelvis xyz yaw | footL xyz yaw contact | footR xyz yaw contact | handL xyz on | handR xyz on | look xyz on | torso xyz on
NCH = 30
XZ = [(0, 2), (4, 6), (9, 11), (14, 16), (18, 20), (22, 24), (26, 28)]
YAWS = [3, 7, 12]
FLAGS = [8, 13, 17, 21, 25, 29]
CONT = [i for i in range(NCH) if i not in FLAGS]
POS = [i for i in CONT if i not in YAWS]


# ============================================================================================================================ data
def canonicalize(W):
    """[N,T,30] -> same shape. Frame 0 of every window: pelvis at x = z = 0, facing +z. Yaw channels unwrapped along time and made relative to the pelvis yaw."""
    W = W.astype(np.float64).copy()
    for y in YAWS:
        W[:, :, y] = np.unwrap(W[:, :, y], axis=1)
    y0 = W[:, 0, 3].copy(); px, pz = W[:, 0, 0].copy(), W[:, 0, 2].copy()
    c, s = np.cos(y0)[:, None], np.sin(y0)[:, None]
    for xi, zi in XZ:
        dx, dz = W[:, :, xi] - px[:, None], W[:, :, zi] - pz[:, None]
        W[:, :, xi] = dx * c - dz * s                      # x' = component along the old "left" axis (cos y0, -sin y0)
        W[:, :, zi] = dx * s + dz * c                      # z' = component along the old heading (sin y0, cos y0)
    for y in YAWS:
        off = W[:, 0, y] - y0                              # keep a foot's frame-0 yaw within +-pi of the pelvis yaw, then stay continuous
        off_w = (off + np.pi) % (2 * np.pi) - np.pi
        W[:, :, y] = W[:, :, y] - y0[:, None] - (off - off_w)[:, None]
    return W.astype(np.float32)


def load_windows(path):
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["action_names"]]
    return z["windows"].astype(np.float32), z["valid"].astype(np.int64), z["action"].astype(np.int64), names, float(z["dt"])


def compute_norm(X, valid):
    """Per-channel mean/std of the continuous channels over valid frames; dstd = std of the frame-to-frame change of the NORMALISED channels. Flags: mean 0, std 1."""
    T = X.shape[1]; m = np.arange(T)[None, :] < valid[:, None]
    v = X[m].astype(np.float64)
    mean = np.zeros(NCH); std = np.ones(NCH)
    mean[CONT] = v[:, CONT].mean(0); std[CONT] = np.maximum(v[:, CONT].std(0), 1e-2)
    Xn = (X - mean.astype(np.float32)) / std.astype(np.float32)
    dm = m[:, 1:] & m[:, :-1]
    d = (Xn[:, 1:] - Xn[:, :-1])[dm].astype(np.float64)
    dstd = np.ones(NCH); dstd[CONT] = np.maximum(d[:, CONT].std(0), 1e-3)
    return dict(mean=mean, std=std, dstd=dstd)


class WindowSet:
    """Normalised windows as tensors, optionally resident on the device; manual batching (no DataLoader needed)."""

    def __init__(self, X, valid, action, norm, device, on_device):
        Xn = (X - norm["mean"].astype(np.float32)) / norm["std"].astype(np.float32)
        self.x = torch.from_numpy(np.ascontiguousarray(Xn)); self.valid = torch.from_numpy(valid); self.action = torch.from_numpy(action)
        if on_device:
            self.x, self.valid, self.action = self.x.to(device), self.valid.to(device), self.action.to(device)
        self.device = device; self.n = len(self.x)

    def batch(self, idx):
        idx = idx.to(self.x.device)
        return (self.x[idx].to(self.device, non_blocking=True), self.valid[idx].to(self.device), self.action[idx].to(self.device))


def prepare_split(path, norm, names_ref, canon, device, on_device_limit_gb):
    X, valid, act, names, dt = load_windows(path)
    if names != names_ref:                                  # same names in the same order is what build_vae_dataset.py writes; remap anyway if not
        mp = {n: i for i, n in enumerate(names_ref)}; act = np.array([mp.get(names[a], len(names_ref)) for a in act], np.int64)
    if canon:
        X = canonicalize(X)
    return X, valid, act, dt


# ============================================================================================================================ model
class FiLMRes(nn.Module):
    """x + conv1x1(drop(act(conv3(act(FiLM(norm(x), cond))))))   FiLM scale/shift start at 0 (identity) so the label is learned in gradually."""

    def __init__(self, h, e, dilation=1, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(8, h, affine=True)
        self.film = nn.Linear(e, 2 * h); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.conv1 = nn.Conv1d(h, h, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(h, h, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, c):
        s, b = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        y = self.norm(x) * (1 + s) + b
        y = self.conv1(F.silu(y))
        return x + self.conv2(self.drop(F.silu(y)))


class Encoder(nn.Module):
    def __init__(self, cin, h, e, d, downs, n_res, dropout):
        super().__init__()
        self.inp = nn.Conv1d(cin, h, 3, padding=1)
        self.stages = nn.ModuleList([nn.ModuleList([nn.ModuleList([FiLMRes(h, e, 3 ** (i % 3), dropout) for i in range(n_res)]), nn.Conv1d(h, h, 4, stride=2, padding=1)])
                                     for _ in range(downs)])
        self.norm = nn.GroupNorm(8, h); self.out = nn.Conv1d(h, d, 3, padding=1)

    def forward(self, x, c):                                # x [B,C,T] -> [B,D,T/2^downs]
        x = self.inp(x)
        for blocks, down in self.stages:
            for b in blocks:
                x = b(x, c)
            x = down(x)
        return self.out(F.silu(self.norm(x)))


class Decoder(nn.Module):
    def __init__(self, cout, h, e, d, ups, n_res, dropout):
        super().__init__()
        self.inp = nn.Conv1d(d, h, 3, padding=1)
        self.stages = nn.ModuleList([nn.ModuleList([nn.ModuleList([FiLMRes(h, e, 3 ** (i % 3), dropout) for i in range(n_res)]), nn.Conv1d(h, h, 3, padding=1)])
                                     for _ in range(ups)])
        self.norm = nn.GroupNorm(8, h); self.out = nn.Conv1d(h, cout, 3, padding=1)

    def forward(self, z, c):                                # z [B,D,N] -> [B,C,N*2^ups]
        x = self.inp(z)
        for blocks, up in self.stages:
            for b in blocks:
                x = b(x, c)
            x = up(F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.norm(x)))


class EMAQuantizer(nn.Module):
    """Vector quantiser with exponential-moving-average codebook (no codebook loss), data init and dead-code restart. Runs in fp32 whatever the autocast mode."""

    def __init__(self, K, D, decay=0.99, restart_thresh=0.5, eps=1e-5):
        super().__init__()
        self.K, self.D, self.decay, self.thresh, self.eps = K, D, decay, restart_thresh, eps
        self.register_buffer("embed", torch.randn(K, D) * 0.1)
        self.register_buffer("cluster_size", torch.ones(K))
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))
        self._ready = False; self._n = 0

    @torch.no_grad()
    def _init_from(self, flat):
        n = flat.shape[0]
        sel = torch.randperm(n, device=flat.device)[:self.K] if n >= self.K else torch.randint(0, n, (self.K,), device=flat.device)
        self.embed.copy_(flat[sel]); self.embed_avg.copy_(self.embed); self.cluster_size.fill_(1.0); self.initted.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat, idx):
        counts = torch.bincount(idx, minlength=self.K).float()
        sums = torch.zeros(self.K, self.D, device=flat.device).index_add_(0, idx, flat)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
        self.embed.copy_(self.embed_avg / cs[:, None])
        self._n += 1
        if self._n % 10:                                    # dead-code check every 10 steps (the bool() below is a GPU sync)
            return
        dead = self.cluster_size < self.thresh
        if bool(dead.any()):
            r = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
            self.embed[dead] = r; self.embed_avg[dead] = r; self.cluster_size[dead] = 1.0

    def forward(self, z):                                   # z [B,N,D] -> (z_q straight-through [B,N,D], idx [B,N], commit loss, perplexity)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float(); B, N, D = z.shape; flat = z.reshape(-1, D)
            if self.training and not self._ready:
                if not bool(self.initted):
                    self._init_from(flat.detach())
                self._ready = True
            with torch.no_grad():
                fd = flat.detach()
                dist = fd.pow(2).sum(1, keepdim=True) - 2 * fd @ self.embed.t() + self.embed.pow(2).sum(1)[None]
                idx = dist.argmin(1)
            q = self.embed[idx]
            if self.training:
                self._ema_update(flat.detach(), idx)
            commit = F.mse_loss(flat, q.detach())
            qst = flat + (q - flat).detach()
            with torch.no_grad():
                p = torch.bincount(idx, minlength=self.K).float(); p = p / p.sum()
                ppl = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        return qst.reshape(B, N, D), idx.reshape(B, N), commit, ppl


class CVQVAE(nn.Module):
    def __init__(self, n_actions, window=64, hidden=528, emb=128, code_dim=256, num_codes=512, downs=2, n_res=3, dropout=0.1, ema_decay=0.99, restart_thresh=0.5,
                 cond_enc=True):
        super().__init__()
        assert hidden % 8 == 0, "hidden must be divisible by 8 (GroupNorm)"
        assert window % (2 ** downs) == 0, f"window {window} must be divisible by 2**downs = {2 ** downs}"
        self.n_actions, self.null, self.window, self.downs, self.cond_enc = n_actions, n_actions, window, downs, cond_enc
        self.act_emb = nn.Embedding(n_actions + 1, emb)             # last row = "null" label
        self.cond_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.enc = Encoder(NCH, hidden, emb, code_dim, downs, n_res, dropout)
        self.vq = EMAQuantizer(num_codes, code_dim, ema_decay, restart_thresh)
        self.dec = Decoder(NCH, hidden, emb, code_dim, downs, n_res, dropout)
        sc, sh = torch.ones(NCH), torch.zeros(NCH); sc[FLAGS] = 2.0; sh[FLAGS] = -1.0     # flags 0/1 -> -1/+1 for the encoder
        self.register_buffer("in_scale", sc); self.register_buffer("in_shift", sh)

    def cond(self, action):
        return self.cond_mlp(self.act_emb(action))

    def encode(self, x, action):                            # x [B,T,30] normalised -> pre-quantisation latents [B,N,D]
        c = self.cond(action if self.cond_enc else torch.full_like(action, self.null))
        return self.enc((x * self.in_scale + self.in_shift).transpose(1, 2), c).transpose(1, 2)

    def decode(self, zq, action):                           # [B,N,D] -> raw output [B,T,30]: continuous channels normalised, flag channels are logits
        return self.dec(zq.transpose(1, 2), self.cond(action)).transpose(1, 2)

    def forward(self, x, action):
        zq, idx, commit, ppl = self.vq(self.encode(x, action))
        return self.decode(zq, action), idx, commit, ppl

    @torch.no_grad()
    def tokens(self, x, action):                            # discrete code indices [B,N]
        return self.vq(self.encode(x, action))[1]

    @torch.no_grad()
    def decode_codes(self, idx, action):                    # code indices [B,N] -> raw output (for a future prior over the codes)
        return self.decode(self.vq.embed[idx], action)


def to_motion(y, norm):
    """raw model output [B,T,30] (tensor) -> physical motion [B,T,30] (numpy): continuous channels de-normalised, flags thresholded to 0/1."""
    y = y.float().cpu().numpy(); out = y * norm["std"].astype(np.float32) + norm["mean"].astype(np.float32)
    out[..., FLAGS] = (y[..., FLAGS] > 0).astype(np.float32)
    return out


# ============================================================================================================================ loss / eval
def compute_loss(model, x, valid, action, norm_t, a, train):
    B, T, _ = x.shape
    mask = (torch.arange(T, device=x.device)[None] < valid[:, None]).float()
    if train and a.cond_drop > 0:
        action_in = torch.where(torch.rand(B, device=x.device) < a.cond_drop, torch.full_like(action, model.null), action)
    else:
        action_in = action
    with amp_ctx(x.device, a.amp):
        y, idx, commit, ppl = model(x, action_in)
    y = y.float(); cont = norm_t["cont"]; flags = norm_t["flags"]
    yc, xc = y[..., cont], x[..., cont]; msum = mask.sum().clamp(min=1)
    err = F.smooth_l1_loss(yc, xc, reduction="none") if a.loss == "huber" else (yc - xc).pow(2)
    rec = (err.mean(-1) * mask).sum() / msum
    vm = mask[:, 1:] * mask[:, :-1]
    vel = ((((yc[:, 1:] - yc[:, :-1]) - (xc[:, 1:] - xc[:, :-1])) / norm_t["dstd_c"]).pow(2).mean(-1) * vm).sum() / vm.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(y[..., flags], x[..., flags], reduction="none").mean(-1) * mask).sum() / msum
    total = rec + a.w_vel * vel + a.w_flag * bce + a.beta * commit
    with torch.no_grad():
        d = (y[..., POS] - x[..., POS]) * norm_t["std_p"]
        sq = (d.pow(2).mean(-1) * mask).sum()                                                   # sum over valid frames of mean squared metres
        acc = (((y[..., flags] > 0) == (x[..., flags] > 0.5)).float().mean(-1) * mask).sum()
    return total, dict(total=total.detach(), rec=rec.detach(), vel=vel.detach(), bce=bce.detach(), commit=commit.detach(), sq=sq, acc=acc, frames=mask.sum(), ppl=ppl), idx


def norm_tensors(norm, device):
    t = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return dict(cont=torch.tensor(CONT, device=device), flags=torch.tensor(FLAGS, device=device), dstd_c=t(norm["dstd"][CONT]), std_p=t(norm["std"][POS]))


@torch.no_grad()
def evaluate(model, ds, norm_t, a, null_label=False, bs=256):
    model.eval(); acc = collections.defaultdict(float); nb = 0; frames = 0.0; used = torch.zeros(model.vq.K, dtype=torch.bool, device=ds.device)
    for i in range(0, ds.n, bs):
        x, v, act = ds.batch(torch.arange(i, min(i + bs, ds.n)))
        if null_label:
            act = torch.full_like(act, model.null)
        _, m, idx = compute_loss(model, x, v, act, norm_t, a, train=False)
        for k in ("total", "rec", "vel", "bce", "commit", "ppl"):
            acc[k] += float(m[k])
        acc["sq"] += float(m["sq"]); acc["acc"] += float(m["acc"]); frames += float(m["frames"]); nb += 1; used[idx.reshape(-1)] = True
    out = {k: acc[k] / max(nb, 1) for k in ("total", "rec", "vel", "bce", "commit", "ppl")}
    out["pos_rmse_m"] = math.sqrt(acc["sq"] / max(frames, 1)); out["flag_acc"] = acc["acc"] / max(frames, 1); out["codes_used"] = float(used.float().mean())
    model.train(); return out


def amp_ctx(device, mode):
    if mode in ("bf16", "fp16"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if mode == "bf16" else torch.float16)
    return contextlib.nullcontext()


# ============================================================================================================================ io helpers
def build_model(cfg):
    return CVQVAE(**cfg)


def count_params(model):
    n = lambda m: sum(p.numel() for p in m.parameters())
    return n(model), dict(encoder=n(model.enc), decoder=n(model.dec), label_embedding=n(model.act_emb) + n(model.cond_mlp), codebook_buffer=model.vq.embed.numel())


def save_ckpt(path, model, cfg, norm, names, dt, a, step, epoch, best, opt=None):
    d = dict(model=model.state_dict(), cfg=cfg, norm={k: np.asarray(v).tolist() for k, v in norm.items()}, action_names=names, dt=dt, step=step, epoch=epoch, best_val=best, args=vars(a))
    if opt is not None:
        d["opt"] = opt.state_dict()
    torch.save(d, path)


def load_ckpt(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    model = build_model(d["cfg"]).to(device); model.load_state_dict(d["model"]); model.vq._ready = True
    return model, d, {k: np.asarray(v) for k, v in d["norm"].items()}


@torch.no_grad()
def export_recon(model, ds, norm, names, dt, path, n_clips, seed, a):
    model.eval(); rng = np.random.default_rng(seed); per = max(1, n_clips // max(1, len(names))); seen = collections.Counter(); pick = []
    labels = ds.action.cpu().numpy()
    for i in rng.permutation(ds.n):
        if seen[int(labels[i])] < per:
            pick.append(int(i)); seen[int(labels[i])] += 1
        if len(pick) >= n_clips:
            break
    x, v, act = ds.batch(torch.tensor(pick))
    with amp_ctx(x.device, a.amp):
        y, _, _, _ = model(x, act)
    orig = x.float().cpu().numpy() * norm["std"].astype(np.float32) + norm["mean"].astype(np.float32); orig[..., FLAGS] = (x[..., FLAGS].float().cpu().numpy() > 0.5)
    rec = to_motion(y, norm); clips, labs, desc, acts = [], [], [], []
    for k, i in enumerate(pick):
        nv = int(v[k]); nm = names[int(act[k])] if int(act[k]) < len(names) else "?"
        for arr, lab in ((orig[k, :nv], "original"), (rec[k, :nv], "reconstruction")):
            clips.append(arr.astype(np.float32)); labs.append(f"{lab} of val window #{i}"); desc.append(lab); acts.append(nm)
    np.savez_compressed(path, actions=np.concatenate(clips), offsets=np.cumsum([0] + [len(c) for c in clips]), dt=np.float32(dt), action=np.array(acts), labels=np.array(labs),
                        description=np.array(desc), style=np.zeros((len(clips), 1), np.float32), style_names=np.array(["none"]))
    print(f"wrote {path}: {len(pick)} val windows as original / reconstruction pairs (canonical frame: frame 0 at the origin facing +z)  ->  python tools/view_clip.py {path} --open")


@torch.no_grad()
def export_codes(model, ds, names, path, bs=256):
    model.eval(); out = []
    for i in range(0, ds.n, bs):
        x, v, act = ds.batch(torch.arange(i, min(i + bs, ds.n))); out.append(model.tokens(x, act).cpu().numpy().astype(np.int16))
    np.savez(path, codes=np.concatenate(out), action=ds.action.cpu().numpy().astype(np.int16), valid=ds.valid.cpu().numpy().astype(np.int16), action_names=np.array(names))
    print(f"wrote {path}: {ds.n} windows x {out[0].shape[1]} codes (codebook size {model.vq.K})")


# ============================================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/vae"); ap.add_argument("--out", default="runs/cvqvae"); ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=500, help="warm-up steps"); ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    ap.add_argument("--min-lr-frac", type=float, default=0.05); ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test); 0 = no limit")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs without a better validation total; 0 = off")
    # model
    ap.add_argument("--hidden", type=int, default=528); ap.add_argument("--emb", type=int, default=128); ap.add_argument("--code-dim", type=int, default=256)
    ap.add_argument("--num-codes", type=int, default=512); ap.add_argument("--downs", type=int, default=2, help="stride-2 stages: 2 -> 4x, 64 frames -> 16 codes")
    ap.add_argument("--n-res", type=int, default=3); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.99); ap.add_argument("--restart-thresh", type=float, default=0.5)
    ap.add_argument("--cond-drop", type=float, default=0.1, help="share of training windows whose label is replaced by 'null'"); ap.add_argument("--no-cond-enc", action="store_true")
    # loss
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber"); ap.add_argument("--beta", type=float, default=0.25, help="commitment weight")
    ap.add_argument("--w-vel", type=float, default=0.2); ap.add_argument("--w-flag", type=float, default=0.5)
    # runtime
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto", help="auto: bf16 on Ampere+/ROCm, fp16 (+loss scaling) on older NVIDIA such as the T4, none on CPU"); ap.add_argument("--compile", action="store_true", help="torch.compile the encoder and decoder")
    ap.add_argument("--no-canon", action="store_true"); ap.add_argument("--data-on-device-gb", type=float, default=3.0, help="keep the windows on the GPU if they take less than this")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=100); ap.add_argument("--resume", default=""); ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--export-recon", type=int, default=0, help="write this many val windows as original/reconstruction pairs (recon_preview.npz)"); ap.add_argument("--export-codes", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    if a.amp == "auto":
        if dev.type == "cuda":                              # bf16 is native on Ampere (sm_80)+ and on AMD ROCm; a T4 (sm_75) emulates it very slowly -> fp16 there
            a.amp = "bf16" if (torch.version.hip or torch.cuda.get_device_capability()[0] >= 8) else "fp16"
        else:
            a.amp = "none"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    # ---- parameter count only (needs no data: 24 actions is the number tools/motion_actions.py defines; the real count is printed again when training starts)
    if a.params_only:
        cfg = dict(n_actions=24, window=64, hidden=a.hidden, emb=a.emb, code_dim=a.code_dim, num_codes=a.num_codes, downs=a.downs, n_res=a.n_res, dropout=a.dropout,
                   ema_decay=a.ema_decay, restart_thresh=a.restart_thresh, cond_enc=not a.no_cond_enc)
        tot, parts = count_params(build_model(cfg)); print(f"{tot / 1e6:.2f} M trainable parameters (assuming 24 actions)  {parts}"); return

    # ---- data
    stats = json.load(open(os.path.join(a.data, "stats.json")))
    assert stats["channels"] and len(stats["channels"]) == NCH, "stats.json does not describe 30 channels"
    if a.eval_only:
        assert a.ckpt, "--eval-only needs --ckpt"
        model, ck, norm = load_ckpt(a.ckpt, dev); names = ck["action_names"]; dt = ck["dt"]; canon = not ck["args"].get("no_canon", False)
    else:
        Xt, vt, at, names, dt = load_windows(os.path.join(a.data, "train_windows.npz")); canon = not a.no_canon
        if canon:
            Xt = canonicalize(Xt)
        norm = compute_norm(Xt, vt)
    Xv, vv, av, dtv = prepare_split(os.path.join(a.data, "val_windows.npz"), norm, names, canon, dev, a.data_on_device_gb)
    assert abs(dtv - dt) < 1e-6, "train and val windows have different dt"
    W = Xv.shape[1]; json.dump({k: np.asarray(v).tolist() for k, v in norm.items()} | dict(dt=dt, canonicalised=canon, channels=stats["channels"]), open(os.path.join(a.out, "norm.json"), "w"), indent=1)
    on_dev = dev.type == "cuda" and (Xv.nbytes + (0 if a.eval_only else Xt.nbytes)) < a.data_on_device_gb * 1e9
    val = WindowSet(Xv, vv, av, norm, dev, on_dev); norm_t = norm_tensors(norm, dev)
    if not a.eval_only:
        train = WindowSet(Xt, vt, at, norm, dev, on_dev)
        print(f"train: {train.n} windows of {W} frames   val: {val.n} windows   actions: {len(names)}   dt {dt:.4f}s   canonical frame: {canon}   device {dev} amp {a.amp}   data on device: {on_dev}")
        cfg = dict(n_actions=len(names), window=W, hidden=a.hidden, emb=a.emb, code_dim=a.code_dim, num_codes=a.num_codes, downs=a.downs, n_res=a.n_res, dropout=a.dropout,
                   ema_decay=a.ema_decay, restart_thresh=a.restart_thresh, cond_enc=not a.no_cond_enc)
        model = build_model(cfg).to(dev)
        tot, parts = count_params(model)
        print(f"model: {tot / 1e6:.2f} M parameters {parts}; {train.n * W * 30 / 1e6:.1f} M training values = {train.n * W * 30 / tot:.2f} values per parameter "
              f"(windows overlap by stride, variants repeat their sources: expect to overfit, trust the validation numbers)")
        if a.compile:
            try:
                model.enc.compile(); model.dec.compile()
            except Exception as ex:
                print(f"torch.compile unavailable ({str(ex)[:100]}): continuing eager")
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.95))
        step = epoch0 = 0; best = float("inf")
        if a.resume:
            ck = torch.load(a.resume, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            step, epoch0, best = ck["step"], ck["epoch"] + 1, ck["best_val"]; model.vq._ready = True; print(f"resumed from {a.resume}: epoch {epoch0}, step {step}, best val {best:.4f}")
        spe = max(1, train.n // a.batch); total_steps = a.epochs * spe
        lr_at = lambda s: a.lr * (s / max(1, a.warmup) if s < a.warmup else a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, total_steps - a.warmup))))
        scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16" and dev.type == "cuda"))     # a no-op unless fp16
        mlog = open(os.path.join(a.out, "metrics.jsonl"), "a"); bad = 0; t0 = time.time(); stop = False
        model.train()
        for epoch in range(epoch0, a.epochs):
            perm = torch.randperm(train.n); run = collections.defaultdict(float); nrun = 0; te = time.time()
            for bi in range(spe):
                x, v, act = train.batch(perm[bi * a.batch:(bi + 1) * a.batch])
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                loss, m, _ = compute_loss(model, x, v, act, norm_t, a, train=True)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); scaler.step(opt); scaler.update(); step += 1
                for k in ("total", "rec", "vel", "bce", "commit", "ppl"):
                    run[k] = run[k] + m[k]                      # stays a device tensor: no GPU sync per step
                nrun += 1
                if step % a.log_every == 0:
                    print(f"  ep {epoch:3d} step {step:7d}  loss {run['total'] / nrun:.4f}  rec {run['rec'] / nrun:.4f}  vel {run['vel'] / nrun:.3f}  flag {run['bce'] / nrun:.3f}  "
                          f"commit {run['commit'] / nrun:.4f}  ppl {run['ppl'] / nrun:5.1f}  gn {float(gn):.2f}  lr {lr_at(step):.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)
                if a.max_steps and step >= a.max_steps:
                    stop = True; break
            ev = evaluate(model, val, norm_t, a); evn = evaluate(model, val, norm_t, a, null_label=True)
            tr = {k: float(run[k]) / max(nrun, 1) for k in run}
            improved = ev["total"] < best
            if improved:
                best = ev["total"]; bad = 0; save_ckpt(os.path.join(a.out, "best.pt"), model, cfg, norm, names, dt, a, step, epoch, best)
            else:
                bad += 1
            save_ckpt(os.path.join(a.out, "last.pt"), model, cfg, norm, names, dt, a, step, epoch, best, opt)
            print(f"epoch {epoch:3d}  train {tr.get('total', 0):.4f} (rec {tr.get('rec', 0):.4f})  |  val {ev['total']:.4f} (rec {ev['rec']:.4f}, pos {ev['pos_rmse_m'] * 100:.1f} cm, "
                  f"flags {ev['flag_acc'] * 100:.1f}%, ppl {ev['ppl']:.0f}, codes used {ev['codes_used'] * 100:.0f}%)  |  val with null label: rec {evn['rec']:.4f}  |  "
                  f"{time.time() - te:.0f}s{'  *best*' if improved else ''}", flush=True)
            mlog.write(json.dumps(dict(epoch=epoch, step=step, train=tr, val=ev, val_null=evn)) + "\n"); mlog.flush()
            if stop or (a.patience and bad >= a.patience):
                print("stopping: " + ("--max-steps reached" if stop else f"no validation improvement for {a.patience} epochs")); break
        model, _, _ = load_ckpt(os.path.join(a.out, "best.pt"), dev) if os.path.exists(os.path.join(a.out, "best.pt")) else (model, None, None)
        print(f"done. best validation total {best:.4f}  ->  {os.path.join(a.out, 'best.pt')}")
    else:
        ev = evaluate(model, val, norm_t, a); evn = evaluate(model, val, norm_t, a, null_label=True)
        print("val: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items())); print("val with null label: " + "  ".join(f"{k} {v:.4f}" for k, v in evn.items()))
        train = None

    if a.export_recon:
        export_recon(model, val, norm, names, dt, os.path.join(a.out, "recon_preview.npz"), a.export_recon, a.seed, a)
    if a.export_codes:
        if train is None:
            Xt, vt, at, _, _ = load_windows(os.path.join(a.data, "train_windows.npz")); Xt = canonicalize(Xt) if canon else Xt; train = WindowSet(Xt, vt, at, norm, dev, False)
        export_codes(model, train, names, os.path.join(a.out, "codes_train.npz")); export_codes(model, val, names, os.path.join(a.out, "codes_val.npz"))


if __name__ == "__main__":
    main()
