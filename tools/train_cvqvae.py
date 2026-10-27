#!/usr/bin/env python3
"""
train_cvqvae.py - ONE script: trains a conditional VQ-VAE (~20M) over NERVA motion windows, tokenises the dataset, trains an autoregressive PRIOR over the codes
(~5M causal transformer) and samples NEW motion from it (windows chained into longer clips, seams cross-faded, optionally graded by tools/motion_grader.py).

  python tools/build_vae_dataset.py --hours 10                                       # 0. the data (data/vae/*_windows.npz, stats.json)
  python tools/train_cvqvae.py --data data/vae --out runs/cvqvae                     # everything: VAE -> codebook check -> tokens -> prior -> samples
  python tools/train_cvqvae.py --data data/vae --out runs/cvqvae --vae-ckpt runs/cvqvae/best.pt     # a VAE you already trained is NOT retrained: tokens -> prior -> samples
  python tools/train_cvqvae.py --out runs/cvqvae --stage generate --temperature 0.9 --top-k 40 --gen-windows 6 --grade   # only sample (needs best.pt + prior_best.pt)
  python tools/train_cvqvae.py --data data/vae --stage vae --out runs/cvqvae         # only the VAE (as before)
  python tools/view_clip.py runs/cvqvae/gen_preview.npz --open                       # look at what it made
  python tools/train_cvqvae.py --params-only                                         # parameter counts of both models, then exit

STAGES (--stage all runs them in order; --vae-ckpt skips the VAE training)
  1 vae       conditional VQ-VAE on windows, validation on unseen performers; best.pt / last.pt                              (details below)
  2 tokens    every train / val window -> 16 codes with the frozen VAE; prints the CODEBOOK HEALTH (codes used, perplexity, top-10 share) and warns on collapse
  3 prior     causal transformer over  [ACTION] c1..c16 [SEP] c1..c16 [SEP] ...  (up to --prior-windows consecutive windows of one clip); the action is a prefix token,
              replaced by NULL for --prior-cond-drop of the sequences, so it also samples unconditionally. prior_best.pt = lowest validation NLL; the report compares it with
              the unigram baseline (a prior that does not beat that has learned nothing about order)
              GOAL    every window also carries 4 numbers measured from the data itself, no labelling: heading change, displacement in the window's canonical frame (dx, dz) and
                      path speed (GOAL_NAMES). A small MLP turns them into a vector that is added to every position of that window; a learned NULL vector replaces them for
                      --prior-goal-drop of the sequences. (--no-prior-goal turns it off; old prior checkpoints without it still load.)
              STYLE   if the window files carry a `style` array (tools/build_vae_dataset.py writes the 100STYLE style of every window) the style id conditions the prior the same way:
                      an embedding added to every position, NULL for --prior-style-drop of the sequences.
              Both nulls give classifier-free guidance: --cfg-scale s samples with  logits = uncond + s * (cond - uncond)  where uncond has the goal nulled (action and style kept).
  4 generate  samples codes (temperature / top-k / top-p), decodes every window with the VAE, chains the windows (stride = the dataset's, found automatically) and
              cross-fades the overlaps -> gen_preview.npz (tools/view_clip.py format), gen_codes.npz; with --grade also scores each clip with the CMU grader (PASS_SCORE) and
              writes gen_passed.npz with the survivors only. --gen-turn-deg / --gen-speed give every window a goal, --gen-style a style, --cfg-scale guides towards them.
              This stage samples EVERYTHING and then plays it; tools/closed_loop.py samples window by window against the engine instead.

CHAINS  build_vae_dataset.py cuts windows with a stride (default 32 of 64 frames), so consecutive windows of one clip share half their frames. The script finds those
        links by comparing the raw frames (no clip ids are stored), which gives the chains the prior is trained on and the stride used to stitch. A window that has no
        neighbour is a chain of one. Every window is canonicalised (frame 0: pelvis at the origin facing +z), so stitching moves each new window to the pose of the
        previous one, blends the overlap linearly (flags: the nearer window wins) and keeps yaw channels continuous. Expect small visible seams: the VAE's own error is ~15 cm.

VAE  ("c" = conditional on the ACTION label)
  encoder   Conv1d -> [FiLM residual blocks -> stride-2 conv] x 2          window 64 frames -> 16 latent steps (4x temporal downsampling)
  quantizer one codebook (512 x 256), EMA updates, k-means-style init from data, dead codes restarted from the current batch
  decoder   Conv1d -> [FiLM residual blocks -> nearest x2 + conv] x 2     -> 30 channels per frame
  The label (plus a learned "null" label) FiLM-modulates every residual block of the encoder AND the decoder; it is dropped to "null" for --cond-drop of the windows.
  --no-cond-enc gives the label to the decoder only. Defaults come to ~19.9 M parameters.
  LOSS  masked smooth-L1 on the 24 continuous channels + velocity loss + BCE on the 6 flags + commitment. Normalisation stats are recomputed from the canonicalised train
  windows; flags stay 0/1; padded windows (valid < W) are masked out of every loss.
  --amp auto: bf16 on Ampere+/ROCm, fp16 with loss scaling on older NVIDIA (T4), none on CPU.

HONEST NOTES  * The prior can only produce what the VAE can decode: reconstruction quality (pos_rmse_m) is the ceiling. * A prior trained on a few hours of real motion plus
variants can memorise; compare train and val NLL and use the lowest-validation checkpoint (it is saved automatically). * Sampled clips are NOT verified by the engine; --grade
only applies the CMU-based grader, as for the variants (tools/closed_loop.py does ask the engine, window by window). * Style from a translator, and retraining on passers, are not part of this script.
* The goal numbers describe what a window DID, so a prior trained on them learns "windows that turn left 30 degrees look like this"; asking for a goal the data never contains (a 180 degree
turn in one second) is out of distribution and the samples will not reach it.
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


def to_world(w, px, pz, y0):
    """Inverse of canonicalize for ONE window [T,30]: rotate by the heading y0 and move to (px, pz)."""
    w = w.astype(np.float64).copy(); c, s = np.cos(y0), np.sin(y0)
    for xi, zi in XZ:
        x, z = w[:, xi].copy(), w[:, zi].copy()
        w[:, xi] = x * c + z * s + px
        w[:, zi] = -x * s + z * c + pz
    for y in YAWS:
        w[:, y] += y0
    return w


def stitch(wins, stride):
    """List of decoded canonical windows [W,30] (flags 0/1; each starts at the origin facing +z) -> ONE clip [W + (n-1)*stride, 30] in a common frame.
    Window i starts at frame i*stride; it is moved to the pose of the clip at that frame, its first W-stride frames are cross-faded with the clip's last W-stride frames."""
    W = wins[0].shape[0]; ov = W - stride
    out = canonicalize(wins[0][None])[0].astype(np.float64)
    for w in wins[1:]:
        w = canonicalize(w[None])[0].astype(np.float64); b = len(out) - ov                  # b: first frame of the overlap in the clip
        w = to_world(w, out[b, 0], out[b, 2], out[b, 3])
        for y in YAWS:                                                                      # same 2*pi branch as the clip, so the blend does not spin
            w[:, y] += 2 * np.pi * np.round((out[b, y] - w[0, y]) / (2 * np.pi))
        if ov > 0:
            al = np.linspace(0, 1, ov + 2)[1:-1][:, None]; old, new = out[b:], w[:ov]
            blend = old * (1 - al) + new * al; blend[:, FLAGS] = np.where(al > 0.5, new[:, FLAGS], old[:, FLAGS])
            out = np.concatenate([out[:b], blend, w[ov:]])
        else:
            out = np.concatenate([out, w])
    return out.astype(np.float32)


def merge_window(pending, w, stride, origin=(0.0, 0.0, 0.0)):
    """ONE step of stitch(), for a loop that commits part of a window and only then decides on the next one (tools/closed_loop.py).
    pending: the not-yet-committed tail of the clip so far, [W - stride, 30] in the world frame, or None for the first window. w: a decoded canonical window [W,30].
    origin: (x, z, yaw) the first window is placed at (the body's pose). Returns the next W frames of the clip [W,30]: the first W - stride cross-fade `pending` with the new window
    (flags: the nearer window wins), the rest are the new window's own. Repeating it reproduces stitch() exactly (tests/test_goal_conditioning.py checks this)."""
    W = w.shape[0]; ov = W - stride
    wc = canonicalize(w[None])[0].astype(np.float64)
    if pending is None:
        return to_world(wc, origin[0], origin[1], origin[2]).astype(np.float32)
    p = np.asarray(pending, np.float64); assert len(p) == ov > 0, f"pending must hold W - stride = {ov} frames, got {len(p)}"
    wc = to_world(wc, p[0, 0], p[0, 2], p[0, 3])
    for y in YAWS:                                                                          # same 2*pi branch as the clip, so the blend does not spin
        wc[:, y] += 2 * np.pi * np.round((p[0, y] - wc[0, y]) / (2 * np.pi))
    al = np.linspace(0, 1, ov + 2)[1:-1][:, None]; new = wc[:ov]
    blend = p * (1 - al) + new * al; blend[:, FLAGS] = np.where(al > 0.5, new[:, FLAGS], p[:, FLAGS])
    return np.concatenate([blend, wc[ov:]]).astype(np.float32)


def find_chains(X, valid, stride=0):
    """RAW windows [N,W,30] -> (chain id [N], position in chain [N], stride). Consecutive windows of one clip overlap by W - stride frames: identical values.
    stride = 0: detect it (the candidate with the most links). No links at all: every window is a chain of one."""
    N, W, _ = X.shape; full = valid >= W
    if N < 2:
        return np.arange(N), np.zeros(N, np.int64), stride or W // 2

    def links(s, hi):
        return np.all(X[:hi, s:] == X[1:hi + 1, :W - s], axis=(1, 2)) & full[:hi] & full[1:hi + 1]
    if not stride:
        cand = sorted({W // 2, W // 4, W // 8, (3 * W) // 4, W // 16} - {0}); sub = min(N - 1, 4000)
        score = {s: int(links(s, sub).sum()) for s in cand}; stride = max(score, key=score.get)
        if score[stride] == 0:
            return np.arange(N), np.zeros(N, np.int64), W // 2
    link = links(stride, N - 1); start = np.r_[True, ~link]; chain = np.cumsum(start) - 1
    first = np.maximum.accumulate(np.where(start, np.arange(N), 0))
    return chain, np.arange(N) - first, stride


def load_windows(path):
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["action_names"]]
    return z["windows"].astype(np.float32), z["valid"].astype(np.int64), z["action"].astype(np.int64), names, float(z["dt"])


def load_styles(path):
    """The 100STYLE style id of every window: (ids [N] int64, names list) or (None, []) when the file has no `style` array (older datasets). names[0] is "none" (no style)."""
    z = np.load(path, allow_pickle=False)
    if "style" not in z.files or "style_names" not in z.files:
        return None, []
    return z["style"].astype(np.int64).reshape(-1), [str(n) for n in z["style_names"]]


def remap_styles(ids, file_names, ref_names):
    """Style ids of one file -> indices into ref_names; a style the reference does not know (a held-out 100STYLE style) becomes the NULL style len(ref_names)."""
    mp = {n: i for i, n in enumerate(ref_names)}
    return np.array([mp.get(file_names[s], len(ref_names)) for s in ids], np.int64)


# ---- goal features: what a window DID, measured from the canonicalised window itself (frame 0 at the origin facing +z) -------------------------------------------------
GOAL_NAMES = ["dyaw", "dx", "dz", "speed"]       # heading change (rad), displacement along the start-left axis (m), displacement along the start heading (m), mean pelvis path speed (m/s)
NG = len(GOAL_NAMES)


def goal_features(Xc, valid, dt):
    """CANONICALISED windows [N,T,30], valid frames [N] -> [N,4] float32 (GOAL_NAMES) over the valid part of each window. No labels needed: it is read off the pelvis channels."""
    N, T, _ = Xc.shape
    X = Xc.astype(np.float64); last = np.clip(np.asarray(valid, np.int64) - 1, 0, T - 1); i = np.arange(N)
    dyaw = X[i, last, 3] - X[:, 0, 3]
    dx = X[i, last, 0] - X[:, 0, 0]; dz = X[i, last, 2] - X[:, 0, 2]
    step = np.hypot(np.diff(X[:, :, 0], axis=1), np.diff(X[:, :, 2], axis=1)); m = np.arange(T - 1)[None, :] < last[:, None]
    speed = (step * m).sum(1) / np.maximum(last * dt, 1e-6)
    return np.stack([dyaw, dx, dz, speed], 1).astype(np.float32)


def arc_goal(turn, path, seconds):
    """The goal features of a walk along a circular arc: heading change `turn` (rad) over `path` metres of pelvis path lasting `seconds`. Arrays broadcast -> [..., 4].
    (Chord direction is half the heading change; chord length = path * sin(turn/2) / (turn/2).)"""
    turn = np.asarray(turn, np.float64); path = np.asarray(path, np.float64); h = turn / 2.0; chord = path * np.sinc(h / np.pi)
    return np.stack(np.broadcast_arrays(turn, chord * np.sin(h), chord * np.cos(h), path / seconds), -1).astype(np.float32)


def goal_stats(g):
    """Statistics of the training windows' goal features, stored with the prior: normalisation (mean/std), how hard a window really turns (turn_max, 95th percentile of |dyaw|),
    and how fast the moving windows travel (speed_cruise = median, speed_max = 95th percentile)."""
    mean, std = g.mean(0).astype(np.float64), np.maximum(g.std(0), 1e-2).astype(np.float64)
    mv = g[g[:, 3] > 0.15, 3]
    return dict(mean=mean.tolist(), std=std.tolist(), turn_max=float(max(np.percentile(np.abs(g[:, 0]), 95), 0.3)),
                speed_cruise=float(np.median(mv)) if len(mv) else 1.0, speed_max=float(np.percentile(mv, 95)) if len(mv) else 1.5)


def norm_goal(g, st):
    return ((np.asarray(g, np.float32) - np.asarray(st["mean"], np.float32)) / np.asarray(st["std"], np.float32)).astype(np.float32)


def remap_actions(act, file_names, ref_names):
    if file_names == ref_names:
        return act
    mp = {n: i for i, n in enumerate(ref_names)}
    return np.array([mp.get(file_names[a], len(ref_names)) for a in act], np.int64)       # unknown name -> the null label


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


# ============================================================================================================================ VAE model
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
    def decode_codes(self, idx, action):                    # code indices [B,N] -> raw output (what the prior samples)
        return self.decode(self.vq.embed[idx], action)


def to_motion(y, norm):
    """raw model output [B,T,30] (tensor) -> physical motion [B,T,30] (numpy): continuous channels de-normalised, flags thresholded to 0/1."""
    y = y.float().cpu().numpy(); out = y * norm["std"].astype(np.float32) + norm["mean"].astype(np.float32)
    out[..., FLAGS] = (y[..., FLAGS] > 0).astype(np.float32)
    return out


# ============================================================================================================================ VAE loss / eval
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


# ============================================================================================================================ VAE io helpers
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


def write_clips_npz(path, clips, action_names, labels, desc, dt, style_labels=None, style_ids=None):
    """tools/view_clip.py format. style / style_names are zeros / ["none"] unless the clips were sampled with a style: then style = [[id]], style_names = ["style_id"], style_label = names."""
    n = len(clips); extra = {} if style_labels is None else dict(style_label=np.array(style_labels))
    style = np.zeros((n, 1), np.float32) if style_ids is None else np.array(style_ids, np.float32).reshape(n, 1)
    np.savez_compressed(path, actions=np.concatenate(clips).astype(np.float32), offsets=np.cumsum([0] + [len(c) for c in clips]), dt=np.float32(dt), action=np.array(action_names),
                        labels=np.array(labels), description=np.array(desc), style=style, style_names=np.array(["none" if style_ids is None else "style_id"]), **extra)


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
    write_clips_npz(path, clips, acts, labs, desc, dt)
    print(f"wrote {path}: {len(pick)} val windows as original / reconstruction pairs (canonical frame: frame 0 at the origin facing +z)  ->  python tools/view_clip.py {path} --open")


# ============================================================================================================================ stage 1: train the VAE
def train_vae(a, dev, model, cfg, train, val, norm, norm_t, names, dt):
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
                print(f"  ep {epoch:3d} step {step:7d}  loss {float(run['total']) / nrun:.4f}  rec {float(run['rec']) / nrun:.4f}  vel {float(run['vel']) / nrun:.3f}  "
                      f"flag {float(run['bce']) / nrun:.3f}  commit {float(run['commit']) / nrun:.4f}  ppl {float(run['ppl']) / nrun:5.1f}  gn {float(gn):.2f}  "
                      f"lr {lr_at(step):.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)
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
    best_path = os.path.join(a.out, "best.pt")
    if os.path.exists(best_path):
        model = load_ckpt(best_path, dev)[0]
    print(f"VAE done. best validation total {best:.4f}  ->  {best_path}")
    return model


# ============================================================================================================================ stage 2: tokens + codebook health
def tokenize(model, ds, a, bs=512):
    model.eval(); out = []
    for i in range(0, ds.n, bs):
        x, v, act = ds.batch(torch.arange(i, min(i + bs, ds.n)))
        with amp_ctx(x.device, a.amp):
            out.append(model.tokens(x, act).cpu())
    return torch.cat(out).numpy().astype(np.int64)


def codebook_health(codes, K, tag, warn=True):
    cnt = np.bincount(codes.reshape(-1), minlength=K).astype(np.float64); p = cnt / cnt.sum(); nz = p[p > 0]
    used = int((cnt > 0).sum()); ent = float(-(nz * np.log(nz)).sum()); ppl = math.exp(ent); top10 = float(np.sort(p)[::-1][:10].sum())
    print(f"codebook health ({tag}): {used}/{K} codes used, perplexity {ppl:.0f}, entropy {ent:.2f} nats ({ent / math.log(2):.2f} bits per code), top-10 codes carry {top10 * 100:.0f}%")
    if warn and (used < 0.25 * K or ppl < 0.1 * K):
        print(f"  WARNING: few codes carry the motion ({used} used, perplexity {ppl:.0f}): a prior over them has little to say. Retrain the VAE (more epochs, "
              f"--restart-thresh higher, smaller --num-codes) before trusting the samples.")
    return ent


def unigram_ce(codes_train, codes_val, K):
    cnt = np.bincount(codes_train.reshape(-1), minlength=K).astype(np.float64) + 0.5; p = cnt / cnt.sum()
    return float(-np.log(p[codes_val.reshape(-1)]).mean())


# ============================================================================================================================ stage 3: the prior over codes
class PriorBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.n2(x)))


class CodePrior(nn.Module):
    """Causal transformer. Token ids: codes 0..K-1 | SEP = K | action tokens K+1 .. K+n_actions | NULL = K+1+n_actions | PAD = K+2+n_actions.
    A sequence is [ACTION] c*n [SEP] c*n [SEP] ... (up to `windows` consecutive windows of one clip, padded at the end). It predicts the next code or SEP.

    goal_dim > 0: every window has a goal vector (the normalised GOAL_NAMES of that window, [B, windows, goal_dim]); an MLP embeds it and the result is ADDED to the
    input embedding of every position that predicts a code (or the SEP) of that window: input position i belongs to window i // (n + 1). A learned null vector replaces it
    (goal=None, or drop_goal[b] True) -- the unconditional branch for classifier-free guidance.
    n_styles > 0: a style id per sequence (0 .. n_styles-1; n_styles = NULL) embedded and added to every position the same way (drop_style[b] True -> NULL)."""

    def __init__(self, K, n_actions, n, windows, d=256, layers=6, heads=4, dropout=0.1, goal_dim=0, n_styles=0):
        super().__init__()
        self.K, self.n_actions, self.n, self.windows, self.goal_dim, self.n_styles = K, n_actions, n, windows, goal_dim, n_styles
        self.SEP, self.ACT0, self.NULL, self.PAD = K, K + 1, K + 1 + n_actions, K + 2 + n_actions
        self.seq_len = windows * (n + 1)                    # 1 + windows*n + (windows-1)
        self.tok = nn.Embedding(K + 3 + n_actions, d); self.pos = nn.Embedding(self.seq_len - 1, d); self.drop = nn.Dropout(dropout)
        if goal_dim:
            self.goal_mlp = nn.Sequential(nn.Linear(goal_dim, d), nn.GELU(), nn.Linear(d, d)); self.goal_null = nn.Parameter(torch.zeros(d))
        if n_styles:
            self.style_emb = nn.Embedding(n_styles + 1, d)  # last row = the NULL style
        self.blocks = nn.ModuleList([PriorBlock(d, heads, dropout) for _ in range(layers)]); self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, K + 1, bias=False)
        self.apply(self._init)
        if goal_dim:
            nn.init.normal_(self.goal_null, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tok, goal=None, style=None, drop_goal=None, drop_style=None):
        """tok [B,T] (T <= seq_len-1) -> logits [B,T,K+1].  goal [B,windows,goal_dim] (window slot j = the j-th window of the context; unused slots may hold zeros) or None = null;
        style [B] long or None = null; drop_goal / drop_style: bool [B], True = use the null for that sequence."""
        B, T = tok.shape; dev = tok.device
        x = self.tok(tok) + self.pos(torch.arange(T, device=dev))[None]
        if self.goal_dim:
            if goal is None:
                g = self.goal_null.expand(B, self.windows, -1)
            else:
                g = self.goal_mlp(goal)
                if drop_goal is not None:
                    g = torch.where(drop_goal.reshape(B, 1, 1), self.goal_null.to(g.dtype), g)
            x = x + g[:, (torch.arange(T, device=dev) // (self.n + 1)).clamp(max=self.windows - 1)]
        if self.n_styles:
            s = style if style is not None else torch.full((B,), self.n_styles, dtype=torch.long, device=dev)
            if drop_style is not None:
                s = torch.where(drop_style, torch.full_like(s, self.n_styles), s)
            x = x + self.style_emb(s)[:, None]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def make_prior_tokens(codes, chain, act, Lw, K, n_actions):
    """codes [N,n], chain id [N], action [N] -> token matrix [N, Lw*(n+1)]: one sequence per window (start), followed by the next windows of the SAME chain."""
    N, n = codes.shape; SEP, ACT0, PAD = K, K + 1, K + 2 + n_actions
    idx = np.arange(N)[:, None] + np.arange(Lw)[None]; nc = np.minimum(idx, N - 1); ok = (idx < N) & (chain[nc] == chain[:, None])
    tok = np.full((N, Lw * (n + 1)), PAD, np.int64); tok[:, 0] = ACT0 + np.minimum(act, n_actions)         # an unknown action (== n_actions) lands on NULL
    for j in range(Lw):
        st = 1 + j * (n + 1); tok[:, st:st + n] = np.where(ok[:, j:j + 1], codes[nc[:, j]], PAD)
        if j < Lw - 1:
            tok[:, st + n] = np.where(ok[:, j + 1], SEP, PAD)
    return tok


def make_prior_goals(gnorm, chain, Lw):
    """normalised window features [N,G] + chain id [N] -> [N, Lw, G]: for the sequence that starts at window i, the features of windows i .. i+Lw-1 of the same chain (zeros after it ends)."""
    N = len(gnorm); idx = np.arange(N)[:, None] + np.arange(Lw)[None]; nc = np.minimum(idx, N - 1); ok = (idx < N) & (chain[nc] == chain[:, None])
    return np.where(ok[..., None], gnorm[nc], 0.0).astype(np.float32)


def prior_loss(prior, tok, amp, dev, goal=None, style=None, drop_goal=None, drop_style=None):
    inp, tgt = tok[:, :-1], tok[:, 1:]
    with amp_ctx(dev, amp):
        lg = prior(inp, goal, style, drop_goal, drop_style)
    lg = lg.float().reshape(-1, prior.K + 1); t = tgt.reshape(-1); valid = t != prior.PAD
    ce = F.cross_entropy(lg, torch.where(valid, t, torch.full_like(t, -100)), reduction="none", ignore_index=-100)
    code = valid & (t < prior.K)
    loss = ce[valid].mean(); nll = ce[code].mean()
    acc = (lg.argmax(-1) == t)[code].float().mean()
    return loss, nll.detach(), acc.detach()


@torch.no_grad()
def eval_prior(prior, T, a, dev, bs=512, goal=None, style=None, no_goal=False):
    """Validation loss. no_goal=True nulls the goal of every sequence (the unconditional branch), to show how much the goal numbers tell the prior."""
    prior.eval(); s = np.zeros(3); nb = 0
    for i in range(0, len(T), bs):
        sl = slice(i, i + bs); dg = torch.ones(len(T[sl]), dtype=torch.bool, device=dev) if (no_goal and goal is not None) else None
        l, nll, acc = prior_loss(prior, T[sl], a.amp, dev, None if goal is None else goal[sl], None if style is None else style[sl], dg)
        s += [float(l), float(nll), float(acc)]; nb += 1
    prior.train(); s /= max(nb, 1); return dict(loss=s[0], nll=s[1], acc=s[2], ppl=math.exp(s[1]))


def train_prior(a, dev, codes_t, codes_v, chain_t, chain_v, at, av, K, n_actions, names, stride, dt, uni_val, gt=None, gv=None, gstats=None, st=None, sv=None, style_names=None):
    """gt / gv: raw goal features of the train / val windows [N,4] (goal_features) or None = no goal conditioning; st / sv: style index per window (remap_styles) or None."""
    n = codes_t.shape[1]; Lw = a.prior_windows
    Tt = torch.from_numpy(make_prior_tokens(codes_t, chain_t, at, Lw, K, n_actions)).to(dev); Tv = torch.from_numpy(make_prior_tokens(codes_v, chain_v, av, Lw, K, n_actions)).to(dev)
    use_goal = gt is not None; n_styles = len(style_names) if (st is not None and style_names and len(style_names) > 1) else 0
    cfg = dict(K=K, n_actions=n_actions, n=n, windows=Lw, d=a.prior_dim, layers=a.prior_layers, heads=a.prior_heads, dropout=a.prior_dropout, goal_dim=NG if use_goal else 0, n_styles=n_styles)
    prior = CodePrior(**cfg).to(dev); npar = sum(p.numel() for p in prior.parameters())
    Gt = Gv = St = Sv = None
    if use_goal:
        Gt = torch.from_numpy(make_prior_goals(norm_goal(gt, gstats), chain_t, Lw)).to(dev); Gv = torch.from_numpy(make_prior_goals(norm_goal(gv, gstats), chain_v, Lw)).to(dev)
    if n_styles:
        St = torch.from_numpy(st.astype(np.int64)).to(dev); Sv = torch.from_numpy(sv.astype(np.int64)).to(dev)
    mult = float(np.mean((Tt[:, 1:] != prior.PAD).sum(1).cpu().numpy()))
    print(f"prior: {npar / 1e6:.2f} M parameters, sequences of up to {Lw} windows x {n} codes ({Tt.shape[1]} tokens), {len(Tt)} train / {len(Tv)} val sequences "
          f"(avg {mult:.0f} real tokens); unigram baseline on val: {uni_val:.3f} nats ({math.exp(uni_val):.0f} effective choices per code)")
    print(f"  conditioning: action prefix (null {a.prior_cond_drop:.0%})" + (f", goal {GOAL_NAMES} per window (null {a.prior_goal_drop:.0%}; turn_max {gstats['turn_max']:.2f} rad, cruise {gstats['speed_cruise']:.2f} m/s)" if use_goal else ", no goal")
          + (f", style: {n_styles} ids incl. 'none' (null {a.prior_style_drop:.0%})" if n_styles else ", no style"))
    decay = [p for nm, p in prior.named_parameters() if p.ndim >= 2]; nod = [p for nm, p in prior.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.prior_wd), dict(params=nod, weight_decay=0.0)], lr=a.prior_lr, betas=(0.9, 0.95))
    spe = max(1, len(Tt) // a.prior_batch); total = a.prior_epochs * spe; warm = min(200, total // 10)
    lr_at = lambda s: a.prior_lr * (s / max(1, warm) if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16" and dev.type == "cuda"))
    best = float("inf"); bad = 0; step = 0; t0 = time.time(); path = os.path.join(a.out, "prior_best.pt"); mlog = open(os.path.join(a.out, "prior_metrics.jsonl"), "a"); prior.train()
    for epoch in range(a.prior_epochs):
        perm = torch.randperm(len(Tt), device=dev); run = np.zeros(3); nb = 0
        for bi in range(spe):
            ix = perm[bi * a.prior_batch:(bi + 1) * a.prior_batch]; tok = Tt[ix].clone()
            if a.prior_cond_drop > 0:
                drop = torch.rand(len(tok), device=dev) < a.prior_cond_drop; tok[drop, 0] = prior.NULL
            goal = Gt[ix] if use_goal else None; style = St[ix] if n_styles else None            # the three nulls are drawn independently, so (action, no goal) and (goal, no action) both occur
            dg = torch.rand(len(tok), device=dev) < a.prior_goal_drop if use_goal else None
            ds = torch.rand(len(tok), device=dev) < a.prior_style_drop if n_styles else None
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, nll, acc = prior_loss(prior, tok, a.amp, dev, goal, style, dg, ds)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0); scaler.step(opt); scaler.update(); step += 1
            run += [float(loss), float(nll), float(acc)]; nb += 1
        run /= max(nb, 1); ev = eval_prior(prior, Tv, a, dev, goal=Gv, style=Sv); improved = ev["nll"] < best
        evn = eval_prior(prior, Tv, a, dev, goal=Gv, style=Sv, no_goal=True) if use_goal else None
        if improved:
            best = ev["nll"]; bad = 0
            torch.save(dict(model=prior.state_dict(), cfg=cfg, names=names, stride=stride, dt=dt, val_nll=best, unigram_val=uni_val, args=vars(a),
                            goal_stats=gstats if use_goal else None, goal_names=GOAL_NAMES if use_goal else None, style_names=list(style_names) if n_styles else None), path)
        else:
            bad += 1
        print(f"prior epoch {epoch:3d}  train nll {run[1]:.3f} acc {run[2] * 100:.1f}%  |  val nll {ev['nll']:.3f} (ppl {ev['ppl']:.0f}) acc {ev['acc'] * 100:.1f}%"
              + (f"  [goal nulled: {evn['nll']:.3f}]" if use_goal else "") + f"  |  {(time.time() - t0) / 60:.1f} min{'  *best*' if improved else ''}", flush=True)
        mlog.write(json.dumps(dict(epoch=epoch, train=dict(loss=run[0], nll=run[1], acc=run[2]), val=ev, val_goal_null=evn)) + "\n"); mlog.flush()
        if a.prior_patience and bad >= a.prior_patience:
            print(f"stopping the prior: no validation improvement for {a.prior_patience} epochs"); break
    print(f"prior done. best val nll {best:.3f} nats vs unigram {uni_val:.3f}  ({'learned order: ' if best < uni_val - 0.05 else 'barely beats the unigram baseline: '}"
          f"{(1 - best / uni_val) * 100:.0f}% fewer nats per code)  ->  {path}")
    return load_prior(path, dev)[0]


def load_prior(path, dev):
    """-> (prior, checkpoint dict). prior.meta = dict(goal_stats, style_names): what the sampler needs to build goals / styles (None for a prior trained without them)."""
    d = torch.load(path, map_location=dev, weights_only=False); prior = CodePrior(**d["cfg"]).to(dev); prior.load_state_dict(d["model"]); prior.eval()
    prior.meta = dict(goal_stats=d.get("goal_stats"), style_names=d.get("style_names")); return prior, d


# ============================================================================================================================ stage 4: sample, decode, chain, grade
def filter_logits(lg, top_k, top_p):
    if top_k and top_k < lg.size(-1):
        kth = torch.topk(lg, top_k, dim=-1).values[..., -1, None]; lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        sl, si = torch.sort(lg, descending=True, dim=-1); pr = torch.softmax(sl, -1); rm = pr.cumsum(-1) - pr > top_p
        lg = torch.full_like(lg, float("-inf")).scatter(-1, si, sl.masked_fill(rm, float("-inf")))
    return lg


@torch.no_grad()
def sample_window(prior, ctx, goal, style, temperature, top_k, top_p, cfg_scale=1.0):
    """Sample the n codes of ONE window, code by code. ctx [B,T] = the tokens so far: [ACTION] then every earlier window followed by SEP (at most prior.windows - 1 of them).
    goal [B, windows, G] normalised, slot j = the j-th window of ctx and the window being sampled is the last used slot (unused slots: zeros), or None = null goal.
    style [B] long or None = null style. cfg_scale s != 1 (needs a goal-conditioned prior): logits = uncond + s * (cond - uncond), uncond = the goal nulled (action and style kept).
    Returns (codes [B,n], ctx with the codes appended)."""
    prior.eval(); B = ctx.shape[0]; use_cfg = cfg_scale != 1.0 and prior.goal_dim > 0 and goal is not None
    if use_cfg:
        drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device), torch.ones(B, dtype=torch.bool, device=ctx.device)])
    for _ in range(prior.n):
        if use_cfg:
            both = prior(torch.cat([ctx, ctx]), torch.cat([goal, goal]), None if style is None else torch.cat([style, style]), drop)[:, -1, :prior.K].float()
            lg = both[B:] + cfg_scale * (both[:B] - both[B:])
        else:
            lg = prior(ctx, goal, style)[:, -1, :prior.K].float()
        nxt = torch.multinomial(torch.softmax(filter_logits(lg / max(temperature, 1e-4), top_k, top_p), -1), 1); ctx = torch.cat([ctx, nxt], 1)
    return ctx[:, -prior.n:], ctx


def goal_context(goals, w, Lw):
    """goals [B, n_windows, G] (normalised, one per window of the clip) -> [B, Lw, G]: the slots of the context while window w is sampled. Slot j = window lo + j, lo = max(0, w - Lw + 1)
    (the context never holds more than Lw windows, the oldest is dropped); unused slots are zeros."""
    lo = max(0, w - Lw + 1); g = goals[:, lo:w + 1]; out = torch.zeros(goals.shape[0], Lw, goals.shape[2], dtype=goals.dtype, device=goals.device)
    out[:, :g.shape[1]] = g
    return out


@torch.no_grad()
def generate_codes(prior, acts, n_windows, temperature, top_k, top_p, goals=None, styles=None, cfg_scale=1.0):
    """acts [B] (long, on device; prior.NULL = unconditional) -> codes [B, n_windows, n]. A window is sampled code by code; SEP is appended by force. Once the context holds
    prior.windows windows the oldest one is dropped (the action prefix stays). goals [B, n_windows, G] (normalised: the plan, one goal per window) / styles [B] / cfg_scale: see
    sample_window; goals=None samples with the null goal."""
    prior.eval(); n, Lw = prior.n, prior.windows; ctx = (prior.ACT0 + acts)[:, None]; out = []
    for w in range(n_windows):
        if w >= Lw:
            ctx = torch.cat([ctx[:, :1], ctx[:, 1 + n + 1:]], 1)
        gctx = goal_context(goals, w, Lw) if (goals is not None and prior.goal_dim) else None
        codes, ctx = sample_window(prior, ctx, gctx, styles, temperature, top_k, top_p, cfg_scale)
        out.append(codes)
        if w < n_windows - 1:
            ctx = torch.cat([ctx, torch.full_like(ctx[:, :1], prior.SEP)], 1)
    return torch.stack(out, 1)


def plan_goals(prior, B, nw, turn_deg, speed, dt, W, dev):
    """The same goal for every window of every clip (an arc of `turn_deg` degrees per window at `speed` m/s; either may be None = 0 deg / the data's cruise speed) -> normalised [B,nw,G]."""
    st = prior.meta["goal_stats"]; sec = (W - 1) * dt
    g = arc_goal(np.deg2rad(turn_deg or 0.0), (st["speed_cruise"] if speed is None else speed) * sec, sec)
    return torch.from_numpy(np.broadcast_to(norm_goal(g, st), (B, nw, NG)).copy()).to(dev)


@torch.no_grad()
def generate(a, dev, vae, norm, prior, names, stride, dt):
    sel = names if a.gen_actions == "all" else [s.strip() for s in a.gen_actions.split(",") if s.strip()]
    bad = [s for s in sel if s not in names]
    if bad:
        sys.exit(f"unknown action(s) {bad}; the model knows: {', '.join(names)}")
    act_ids = [names.index(s) for s in sel for _ in range(a.gen_per_action)] if sel else [prior.NULL - prior.ACT0] * a.gen_per_action
    acts = torch.tensor(act_ids, device=dev); B, nw = len(acts), a.gen_windows; torch.manual_seed(a.gen_seed)
    goals = styles = None; sty_name = "none"
    if a.gen_turn_deg is not None or a.gen_speed is not None:
        if not prior.goal_dim:
            sys.exit("--gen-turn-deg / --gen-speed need a prior trained with goal conditioning (this one was trained without; train again without --no-prior-goal)")
        goals = plan_goals(prior, B, nw, a.gen_turn_deg, a.gen_speed, dt, vae.window, dev)
    if a.gen_style:
        sn = prior.meta.get("style_names") or []
        if a.gen_style not in sn:
            sys.exit(f"unknown style {a.gen_style!r}; this prior knows: " + (", ".join(sn) if sn else "none (the windows had no style ids)"))
        styles = torch.full((B,), sn.index(a.gen_style), dtype=torch.long, device=dev); sty_name = a.gen_style
    if a.cfg_scale != 1.0 and goals is None:
        print("note: --cfg-scale only changes the samples when a goal is given (--gen-turn-deg / --gen-speed)")
    codes = generate_codes(prior, acts, nw, a.temperature, a.top_k, a.top_p, goals, styles, a.cfg_scale)       # [B,nw,n]
    y = vae.decode_codes(codes.reshape(B * nw, -1), acts.repeat_interleave(nw)); win = to_motion(y, norm).reshape(B, nw, -1, NCH)
    clips = [stitch(list(win[b]), stride) for b in range(B)]; an = [names[i] if i < len(names) else "null" for i in act_ids]
    scores = [None] * B
    if a.grade:
        import motion_grader as G
        prof, _, _ = G.get_profiles(a.real)
        for b in range(B):
            if an[b] in prof.p:
                scores[b] = G.grade_clip(clips[b], dt, an[b], prof)["score"]
        print("grader (PASS at %.0f):" % G.PASS_SCORE)
        for n_ in dict.fromkeys(an):
            ss = [scores[b] for b in range(B) if an[b] == n_]; ok = [s for s in ss if s is not None]
            print(f"  {n_:14s} " + (f"{sum(s >= G.PASS_SCORE for s in ok)}/{len(ss)} pass, mean score {np.mean(ok):.0f}" if ok else "not a graded action"))
    sec = len(clips[0]) * dt
    gtxt = (f" goal turn {a.gen_turn_deg or 0:g} deg/window" + (f" speed {a.gen_speed:g}" if a.gen_speed is not None else "") + (f" cfg {a.cfg_scale:g}" if a.cfg_scale != 1.0 else "")) if goals is not None else ""
    labels = [f"generated {an[b]} T={a.temperature} k={a.top_k} p={a.top_p}{gtxt}" + (f" style {sty_name}" if styles is not None else "") + (f" score {scores[b]:.0f}" if scores[b] is not None else "") for b in range(B)]
    desc = [f"{nw} windows, {sec:.1f} s" + (" PASS" if scores[b] is not None and scores[b] >= G.PASS_SCORE else (" FAIL" if scores[b] is not None else "")) for b in range(B)] if a.grade \
        else [f"{nw} windows, {sec:.1f} s"] * B
    sk = dict(style_labels=[sty_name] * B, style_ids=styles.cpu().numpy()) if styles is not None else {}
    write_clips_npz(os.path.join(a.out, "gen_preview.npz"), clips, an, labels, desc, dt, **sk)
    np.savez(os.path.join(a.out, "gen_codes.npz"), codes=codes.cpu().numpy().astype(np.int16), action=np.array(act_ids, np.int16), action_names=np.array(names))
    print(f"wrote {a.out}/gen_preview.npz: {B} clips of {nw} windows ({sec:.1f} s each, seams every {stride * dt:.2f} s)  ->  python tools/view_clip.py {a.out}/gen_preview.npz --open")
    if a.grade:
        keep = [b for b in range(B) if scores[b] is not None and scores[b] >= G.PASS_SCORE]
        if keep:
            write_clips_npz(os.path.join(a.out, "gen_passed.npz"), [clips[b] for b in keep], [an[b] for b in keep], [labels[b] for b in keep], [desc[b] for b in keep], dt,
                            **({"style_labels": [sty_name] * len(keep), "style_ids": styles.cpu().numpy()[keep]} if styles is not None else {}))
            print(f"wrote {a.out}/gen_passed.npz: {len(keep)} of {B} passed the grader (candidates for tools/motion_memory.py; keep real CMU data in the mix)")
        else:
            print("no generated clip passed the grader: lower --temperature / --top-k, or check the codebook health and the prior's val nll above")


# ============================================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/vae"); ap.add_argument("--out", default="runs/cvqvae"); ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", choices=["all", "vae", "prior", "generate"], default="all", help="all: VAE (unless --vae-ckpt) -> tokens -> prior -> samples; vae: only the VAE; prior: tokens -> prior -> samples (needs a VAE); generate: only sample")
    ap.add_argument("--vae-ckpt", default="", help="use this trained VAE instead of training one (default for --stage prior / generate: OUT/best.pt)"); ap.add_argument("--ckpt", default="", help="alias of --vae-ckpt")
    # VAE training
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=500, help="warm-up steps"); ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    ap.add_argument("--min-lr-frac", type=float, default=0.05); ap.add_argument("--max-steps", type=int, default=0, help="stop the VAE after this many steps (smoke test); 0 = no limit")
    ap.add_argument("--patience", type=int, default=0, help="stop the VAE after this many epochs without a better validation total; 0 = off")
    # VAE model
    ap.add_argument("--hidden", type=int, default=528); ap.add_argument("--emb", type=int, default=128); ap.add_argument("--code-dim", type=int, default=256)
    ap.add_argument("--num-codes", type=int, default=512); ap.add_argument("--downs", type=int, default=2, help="stride-2 stages: 2 -> 4x, 64 frames -> 16 codes")
    ap.add_argument("--n-res", type=int, default=3); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.99); ap.add_argument("--restart-thresh", type=float, default=0.5)
    ap.add_argument("--cond-drop", type=float, default=0.1, help="share of training windows whose label is replaced by 'null'"); ap.add_argument("--no-cond-enc", action="store_true")
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber"); ap.add_argument("--beta", type=float, default=0.25, help="commitment weight")
    ap.add_argument("--w-vel", type=float, default=0.2); ap.add_argument("--w-flag", type=float, default=0.5)
    # prior
    ap.add_argument("--prior-windows", type=int, default=4, help="consecutive windows per training sequence (and the context while sampling)")
    ap.add_argument("--prior-dim", type=int, default=256); ap.add_argument("--prior-layers", type=int, default=6); ap.add_argument("--prior-heads", type=int, default=4)
    ap.add_argument("--prior-dropout", type=float, default=0.1); ap.add_argument("--prior-epochs", type=int, default=40); ap.add_argument("--prior-batch", type=int, default=256)
    ap.add_argument("--prior-lr", type=float, default=3e-4); ap.add_argument("--prior-wd", type=float, default=0.01); ap.add_argument("--prior-patience", type=int, default=6)
    ap.add_argument("--prior-cond-drop", type=float, default=0.1, help="share of sequences whose action prefix becomes NULL (unconditional sampling)")
    ap.add_argument("--no-prior-goal", action="store_true", help="train the prior without the per-window goal numbers (heading change, displacement, speed)")
    ap.add_argument("--prior-goal-drop", type=float, default=0.15, help="share of sequences whose goal numbers become the NULL goal (what classifier-free guidance contrasts against)")
    ap.add_argument("--prior-style-drop", type=float, default=0.15, help="share of sequences whose style id becomes NULL (only if the window files have a `style` array)")
    # sampling
    ap.add_argument("--gen-per-action", type=int, default=2); ap.add_argument("--gen-windows", type=int, default=4, help="windows per clip: 64 + (n-1)*stride frames")
    ap.add_argument("--gen-actions", default="all", help="'all' or comma-separated action names"); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--gen-turn-deg", type=float, default=None, help="goal for EVERY window: heading change in degrees per window (+ = left); giving this or --gen-speed switches the goal on")
    ap.add_argument("--gen-speed", type=float, default=None, help="goal for every window: path speed in m/s (default: the cruise speed of the training data); giving this or --gen-turn-deg switches the goal on")
    ap.add_argument("--gen-style", default="", help="a 100STYLE style name the prior knows (see the training log); default: the null style")
    ap.add_argument("--cfg-scale", type=float, default=1.0, help="classifier-free guidance on the goal: 1 = off, 1.5-3 = follow it harder (and sample less diversely)")
    ap.add_argument("--grade", action="store_true", help="score every sample with tools/motion_grader.py (needs data/real_motions.npz, see --real)"); ap.add_argument("--real", default="data/real_motions.npz")
    ap.add_argument("--no-generate", action="store_true", help="skip stage 4")
    # runtime
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto", help="auto: bf16 on Ampere+/ROCm, fp16 (+loss scaling) on older NVIDIA such as the T4, none on CPU")
    ap.add_argument("--no-canon", action="store_true"); ap.add_argument("--data-on-device-gb", type=float, default=3.0, help="keep the windows on the GPU if they take less than this")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=100); ap.add_argument("--resume", default="", help="resume the VAE from a last.pt")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the VAE (--vae-ckpt) and do the exports below, then stop"); ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--export-recon", type=int, default=0, help="write this many val windows as original/reconstruction pairs (recon_preview.npz)"); ap.add_argument("--export-codes", action="store_true")
    a = ap.parse_args()
    if a.ckpt and not a.vae_ckpt:
        a.vae_ckpt = a.ckpt

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

    if a.params_only:
        cfg = dict(n_actions=24, window=64, hidden=a.hidden, emb=a.emb, code_dim=a.code_dim, num_codes=a.num_codes, downs=a.downs, n_res=a.n_res, dropout=a.dropout,
                   ema_decay=a.ema_decay, restart_thresh=a.restart_thresh, cond_enc=not a.no_cond_enc)
        tot, parts = count_params(build_model(cfg)); print(f"VAE:   {tot / 1e6:.2f} M trainable parameters (assuming 24 actions)  {parts}")
        pr = CodePrior(a.num_codes, 24, 64 // 2 ** a.downs, a.prior_windows, a.prior_dim, a.prior_layers, a.prior_heads, goal_dim=0 if a.no_prior_goal else NG)
        print(f"prior: {sum(p.numel() for p in pr.parameters()) / 1e6:.2f} M parameters" + ("" if a.no_prior_goal else " (with the goal MLP; a style embedding adds 256 per style)")); return

    if a.eval_only and not a.vae_ckpt:
        sys.exit("--eval-only needs --vae-ckpt PATH")
    if a.stage in ("prior", "generate") and not a.vae_ckpt:
        a.vae_ckpt = os.path.join(a.out, "best.pt")
        if not os.path.exists(a.vae_ckpt):
            sys.exit(f"--stage {a.stage} needs a trained VAE: pass --vae-ckpt PATH (looked for {a.vae_ckpt})")

    # ---------------------------------------------------------------------------------------------------------------- stage 4 only: no data needed
    if a.stage == "generate":
        vae, ck, norm = load_ckpt(a.vae_ckpt, dev); prior, pk = load_prior(os.path.join(a.out, "prior_best.pt"), dev)
        print(f"VAE {a.vae_ckpt}, prior {a.out}/prior_best.pt (val nll {pk['val_nll']:.3f}); amp {a.amp}")
        generate(a, dev, vae, norm, prior, ck["action_names"], pk["stride"], ck["dt"]); return

    # ---------------------------------------------------------------------------------------------------------------- data
    stats = json.load(open(os.path.join(a.data, "stats.json")))
    assert stats["channels"] and len(stats["channels"]) == NCH, "stats.json does not describe 30 channels"
    model = None; norm = None; ref_names = None
    if a.vae_ckpt:
        model, ck, norm = load_ckpt(a.vae_ckpt, dev); ref_names = ck["action_names"]; canon = not ck["args"].get("no_canon", False); print(f"VAE loaded from {a.vae_ckpt} (not retrained)")
    else:
        canon = not a.no_canon
    Xt_raw, vt, at, names_f, dt = load_windows(os.path.join(a.data, "train_windows.npz"))
    names = ref_names or names_f; at = remap_actions(at, names_f, names)
    Xv_raw, vv, av, names_v, dtv = load_windows(os.path.join(a.data, "val_windows.npz")); av = remap_actions(av, names_v, names)
    assert abs(dtv - dt) < 1e-6, "train and val windows have different dt"
    if model is not None:
        assert abs(ck["dt"] - dt) < 1e-6, "the VAE was trained on a different dt"
    W = Xt_raw.shape[1]
    chain_t, pos_t, stride = find_chains(Xt_raw, vt); chain_v, pos_v, _ = find_chains(Xv_raw, vv, stride)
    nct, ncv = int(chain_t.max()) + 1, int(chain_v.max()) + 1
    print(f"windows: {len(Xt_raw)} train in {nct} chains (avg {len(Xt_raw) / nct:.1f}, longest {int(pos_t.max()) + 1}), {len(Xv_raw)} val in {ncv} chains; stride {stride} of {W} frames; "
          f"{len(names)} actions; dt {dt:.4f}s; canonical frame {canon}; device {dev}, amp {a.amp}")
    Xt = canonicalize(Xt_raw) if canon else Xt_raw; Xv = canonicalize(Xv_raw) if canon else Xv_raw
    if norm is None:
        norm = compute_norm(Xt, vt)
    json.dump({k: np.asarray(v).tolist() for k, v in norm.items()} | dict(dt=dt, canonicalised=canon, channels=stats["channels"], stride=stride), open(os.path.join(a.out, "norm.json"), "w"), indent=1)
    on_dev = dev.type == "cuda" and (Xv.nbytes + Xt.nbytes) < a.data_on_device_gb * 1e9
    train = WindowSet(Xt, vt, at, norm, dev, on_dev); val = WindowSet(Xv, vv, av, norm, dev, on_dev); norm_t = norm_tensors(norm, dev)

    # ---------------------------------------------------------------------------------------------------------------- stage 1: VAE
    if model is None:
        cfg = dict(n_actions=len(names), window=W, hidden=a.hidden, emb=a.emb, code_dim=a.code_dim, num_codes=a.num_codes, downs=a.downs, n_res=a.n_res, dropout=a.dropout,
                   ema_decay=a.ema_decay, restart_thresh=a.restart_thresh, cond_enc=not a.no_cond_enc)
        model = build_model(cfg).to(dev); tot, parts = count_params(model)
        print(f"VAE: {tot / 1e6:.2f} M parameters {parts}; {train.n * W * 30 / 1e6:.1f} M training values = {train.n * W * 30 / tot:.2f} values per parameter "
              f"(windows overlap by stride, variants repeat their sources: expect to overfit, trust the validation numbers)")
        model = train_vae(a, dev, model, cfg, train, val, norm, norm_t, names, dt)
    else:
        ev = evaluate(model, val, norm_t, a); evn = evaluate(model, val, norm_t, a, null_label=True)
        print("VAE val: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items())); print("VAE val with null label: " + "  ".join(f"{k} {v:.4f}" for k, v in evn.items()))
    if a.export_recon:
        export_recon(model, val, norm, names, dt, os.path.join(a.out, "recon_preview.npz"), a.export_recon, a.seed, a)
    if a.stage == "vae" and not a.export_codes:
        return

    # ---------------------------------------------------------------------------------------------------------------- stage 2: tokens + codebook health
    codes_t = tokenize(model, train, a); codes_v = tokenize(model, val, a); K = model.vq.K
    codebook_health(codes_t, K, "train"); codebook_health(codes_v, K, "val", warn=False)
    uni_val = unigram_ce(codes_t, codes_v, K)
    if a.export_codes:
        np.savez(os.path.join(a.out, "codes_train.npz"), codes=codes_t.astype(np.int16), action=at.astype(np.int16), valid=vt.astype(np.int16), chain=chain_t, pos=pos_t, action_names=np.array(names))
        np.savez(os.path.join(a.out, "codes_val.npz"), codes=codes_v.astype(np.int16), action=av.astype(np.int16), valid=vv.astype(np.int16), chain=chain_v, pos=pos_v, action_names=np.array(names))
        print(f"wrote {a.out}/codes_train.npz and codes_val.npz")
    if a.eval_only or a.stage == "vae":
        return

    # ---------------------------------------------------------------------------------------------------------------- stage 3: prior
    gt = gv = gstats = st_ids = sv_ids = style_names = None
    if not a.no_prior_goal:                                                                 # goal numbers are read off the CANONICAL windows (also when the VAE itself was trained with --no-canon)
        gt = goal_features(Xt if canon else canonicalize(Xt_raw), vt, dt); gv = goal_features(Xv if canon else canonicalize(Xv_raw), vv, dt); gstats = goal_stats(gt)
        print("goal features (train windows): " + ", ".join(f"{nm} mean {m:.2f} std {s:.2f}" for nm, m, s in zip(GOAL_NAMES, gstats["mean"], gstats["std"])))
    st_raw, sn_t = load_styles(os.path.join(a.data, "train_windows.npz"))
    if st_raw is not None and len(sn_t) > 1:
        sv_raw, sn_v = load_styles(os.path.join(a.data, "val_windows.npz")); style_names, st_ids = sn_t, st_raw
        sv_ids = remap_styles(sv_raw, sn_v, sn_t) if sv_raw is not None else np.full(len(Xv_raw), len(sn_t), np.int64)       # a style never seen in training -> the NULL style
        print(f"styles: {len(sn_t) - 1} named styles in the train windows ({np.mean(st_raw > 0) * 100:.0f}% of windows carry one); validation windows of unseen styles use the null style")
    prior = train_prior(a, dev, codes_t, codes_v, chain_t, chain_v, at, av, K, len(names), names, stride, dt, uni_val, gt, gv, gstats, st_ids, sv_ids, style_names)

    # ---------------------------------------------------------------------------------------------------------------- stage 4: samples
    if not a.no_generate:
        generate(a, dev, model, norm, prior, names, stride, dt)


if __name__ == "__main__":
    main()
