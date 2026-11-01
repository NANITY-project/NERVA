"""e5_torch.py - everything torch-specific about the E5 Goal Translator (model, loss, dataset, checkpoints, export arrays). Imported by train_e5_translator.py,
export_e5_translator.py and tests/test_e5_translator.py. The numpy twin (e5_numpy.py) mirrors this file's forward pass key for key.

Needs: torch, transformers (BertModel; the weights come from the encoder you fine-tune, e.g. intfloat/e5-small-v2, MIT - read the model card of the exact checkpoint you use).
"""
from __future__ import annotations
import math, random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gt_conventions as C
import gt_corrupt
import e5_common as E

HEAD_PRESETS = {                      # d = model width of the window decoder, ff = its feed-forward width
    "tiny":  dict(d=128, layers=2, heads=4, ff=512),       # ~0.7 M: smoke tests
    "small": dict(d=256, layers=3, heads=4, ff=1024),      # ~3.3 M: default
    "base":  dict(d=384, layers=4, heads=6, ff=1536),      # ~8 M
}


# ------------------------------------------------------------------------------------------------------------------------------------------------ model
class MHA(nn.Module):
    """Multi-head attention with separate q/k/v/o projections (the numpy twin reads exactly these names). key_pad: [B, Lk] bool, True = padding."""
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, key_pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Lq, D = xq.shape
        Lk = xkv.shape[1]
        q = self.q(xq).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if key_pad is not None:
            att = att.masked_fill(key_pad[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(-1))
        return self.o((att @ v).transpose(1, 2).reshape(B, Lq, D))


class Block(nn.Module):
    """Pre-LayerNorm decoder block: self-attention among the window queries, cross-attention to the text, feed-forward (GELU, erf)."""
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa, self.ca = MHA(d, heads), MHA(d, heads)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mem_pad):
        h = self.ln1(x)
        x = x + self.drop(self.sa(h, h))
        x = x + self.drop(self.ca(self.ln2(x), mem, mem_pad))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.ln3(x)))))


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


class Head(nn.Module):
    """W_MAX window queries + 1 length query attend to the encoder's token states. Outputs per window: spec logit, turn (tanh), speed (sigmoid); once: length logits."""
    def __init__(self, hidden: int, d: int, layers: int, heads: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.mem_proj, self.mem_ln = nn.Linear(hidden, d), nn.LayerNorm(d)
        self.q = nn.Embedding(E.W_MAX + 1, d)
        with torch.no_grad():
            self.q.weight.copy_(sinusoidal(E.W_MAX + 1, d))         # ordered start: neighbouring windows begin similar, the model learns the rest
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.out_ln = nn.LayerNorm(d)
        self.win_out, self.len_out = nn.Linear(d, 3), nn.Linear(d, E.W_MAX)

    def forward(self, enc: torch.Tensor, enc_pad: torch.Tensor) -> Dict[str, torch.Tensor]:
        mem = self.mem_ln(self.mem_proj(enc))
        x = self.q.weight.unsqueeze(0).expand(enc.shape[0], -1, -1)
        for blk in self.blocks:
            x = blk(x, mem, enc_pad)
        x = self.out_ln(x)
        win = self.win_out(x[:, :E.W_MAX])
        return dict(len_logits=self.len_out(x[:, E.W_MAX]), spec_logit=win[..., 0], turn=torch.tanh(win[..., 1]), speed=torch.sigmoid(win[..., 2]))


class E5GoalTranslator(nn.Module):
    def __init__(self, encoder: nn.Module, head_cfg: dict, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head_cfg = dict(head_cfg)
        self.head = Head(encoder.config.hidden_size, dropout=dropout, **self.head_cfg)

    @staticmethod
    def from_pretrained(name: str, head: str = "small", dropout: float = 0.1) -> "E5GoalTranslator":
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(name)
        if cfg.model_type != "bert":
            raise SystemExit(f"{name} is a '{cfg.model_type}' model: only BERT-style encoders with an UNCASED WordPiece vocabulary are supported (e5-small, e5-small-v2, e5-base-v2, "
                             f"bge-small-en-v1.5, all-MiniLM-L6-v2 ...). multilingual-e5 is XLM-R with a SentencePiece vocabulary and would need a different tokenizer and runtime.")
        enc = AutoModel.from_pretrained(name, add_pooling_layer=False)
        return E5GoalTranslator(enc, HEAD_PRESETS[head], dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(enc, attention_mask == 0)


# ------------------------------------------------------------------------------------------------------------------------------------------------ loss
def loss_fn(out: Dict[str, torch.Tensor], tg: Dict[str, torch.Tensor], w_len=1.0, w_spec=1.0, w_reg=3.0):
    """len: cross-entropy over the number of windows (light label smoothing). spec: BCE over the windows that exist. turn / speed: smooth-L1 over windows that exist AND are
    specified (an UNSPEC window has no turn or speed to learn). Windows past the end are ignored everywhere (the length decides where the plan stops)."""
    n = tg["n"]
    active = torch.arange(E.W_MAX, device=n.device)[None, :] < n[:, None]
    l_len = F.cross_entropy(out["len_logits"].float(), n - 1, label_smoothing=0.05)
    sl = out["spec_logit"].float()
    l_spec = F.binary_cross_entropy_with_logits(sl[active], tg["spec"][active])
    m = active & (tg["spec"] > 0.5)
    if bool(m.any()):
        l_turn = F.smooth_l1_loss(out["turn"].float()[m], tg["turn"][m], beta=0.05)
        l_speed = F.smooth_l1_loss(out["speed"].float()[m], tg["speed"][m], beta=0.05)
    else:
        l_turn = l_speed = sl.sum() * 0.0
    total = w_len * l_len + w_spec * l_spec + w_reg * (l_turn + l_speed)
    return total, dict(len=float(l_len), spec=float(l_spec), turn=float(l_turn), speed=float(l_speed))


# ------------------------------------------------------------------------------------------------------------------------------------------------ data
class E5Dataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict], augment: bool = True, corrupt_p: float = 0.5, seed: int = 0):
        self.rows, self.augment, self.corrupt_p = rows, augment, corrupt_p
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        if self.augment and self.rng.random() < self.corrupt_p:
            text = gt_corrupt.corrupt(text, self.rng, max_bytes=C.MAX_INPUT_BYTES)         # typos, missing spaces, case: the translator must read what people really type
        return (text,) + E.row_targets(r["windows"])


def pad_batch(id_lists: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids, mask


def make_collate(wp: E.WordPiece, pad_id: int = 0):
    """Training uses OUR tokenizer (e5_common.WordPiece), the one the UI runs, so training and inference cannot disagree on a rare character. Too-long text is truncated here
    (rare, and the encoder has 512 positions); at inference it is rejected."""
    def collate(batch):
        texts, ns, spec, turn, speed = zip(*batch)
        ids, mask = pad_batch([wp.encode(t, E.PREFIX, E.MAX_TOKENS, truncate=True) for t in texts], pad_id)
        tg = dict(n=torch.tensor(ns, dtype=torch.long), spec=torch.from_numpy(np.stack(spec)), turn=torch.from_numpy(np.stack(turn)), speed=torch.from_numpy(np.stack(speed)))
        return ids, mask, tg
    return collate


@torch.no_grad()
def predict_plans(model: E5GoalTranslator, wp: E.WordPiece, texts: Sequence[str], device, st: C.Stats, batch: int = 64, pad_id: int = 0) -> List[C.Plan]:
    model.eval()
    plans: List[C.Plan] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids, mask = pad_batch([wp.encode(t, E.PREFIX, E.MAX_TOKENS, truncate=True) for t in chunk], pad_id)
        o = model(ids.to(device), mask.to(device))
        o = {k: v.float().cpu().numpy() for k, v in o.items()}
        for b in range(len(chunk)):
            plans.append(E.decode_outputs(o["len_logits"][b], o["spec_logit"][b], o["turn"][b], o["speed"][b], st))
    return plans


# ------------------------------------------------------------------------------------------------------------------------------------------------ checkpoints / export
def save_checkpoint(path: str, model: E5GoalTranslator, vocab: List[str], st: C.Stats, tm: C.Timing, encoder_name: str, step: int, extra: Optional[dict] = None):
    ck = dict(model=model.state_dict(), encoder_config=model.encoder.config.to_dict(), head_cfg=model.head_cfg, vocab=vocab, stats=st.to_dict(),
              timing=dict(W=tm.W, stride=tm.stride, dt=tm.dt), encoder_name=encoder_name, e5gt_version=E.E5GT_VERSION, step=step)
    ck.update(extra or {})
    torch.save(ck, path)


def load_checkpoint(path: str, device) -> Tuple[E5GoalTranslator, dict]:
    from transformers import BertConfig, BertModel
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("e5gt_version") != E.E5GT_VERSION:
        raise SystemExit(f"{path}: checkpoint version {ck.get('e5gt_version')} != {E.E5GT_VERSION}")
    enc = BertModel(BertConfig.from_dict(ck["encoder_config"]), add_pooling_layer=False)
    model = E5GoalTranslator(enc, ck["head_cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def export_arrays(model: E5GoalTranslator, vocab: List[str], st: C.Stats, tm: C.Timing, encoder_name: str = "", fp16: bool = False, step: int = 0):
    """-> (weights, meta, vocab) exactly as e5_numpy.E5NumpyTranslator wants them. fp16 stores the 2-D matrices as float16 (about half the file); biases and LayerNorms stay float32."""
    weights = {}
    for k, v in model.state_dict().items():
        if not torch.is_floating_point(v) or "position_ids" in k:
            continue
        a = v.detach().cpu().float().numpy()
        weights[k] = a.astype(np.float16) if (fp16 and a.ndim >= 2) else a
    c = model.encoder.config
    meta = dict(e5gt_version=E.E5GT_VERSION, encoder=dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
                intermediate_size=c.intermediate_size, layer_norm_eps=c.layer_norm_eps), head=dict(model.head_cfg, w_max=E.W_MAX), stats=st.to_dict(),
                timing=dict(W=tm.W, stride=tm.stride, dt=tm.dt), prefix=E.PREFIX, max_tokens=E.MAX_TOKENS, encoder_name=encoder_name, step=step,
                license_note="the encoder is fine-tuned from the checkpoint named in encoder_name: keep that model's license and notice with this file")
    return weights, meta, list(vocab)
