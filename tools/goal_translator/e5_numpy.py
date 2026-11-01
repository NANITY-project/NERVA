"""e5_numpy.py - run the E5 Goal Translator with numpy only (no torch, no transformers): what UI/goal_plan.py loads.

    t = E5NumpyTranslator.from_npz("e5gt.npz")        # written by export_e5_translator.py
    plan = t.translate("walk right and then left")     # gt_conventions.Plan, source "model"; raises gt_conventions.PlanError on text that is too long

The forward pass is a line-by-line mirror of e5_torch.py (BERT encoder with post-LayerNorm, then the window decoder with pre-LayerNorm blocks). tests/test_e5_translator.py checks
the parity against HF's BertModel and the torch module; this file alone can only be tested for shape, determinism and the plan decoding (the numpy-only half of that test file).
"""
from __future__ import annotations
import json
import math
from typing import Dict, List, Optional

import numpy as np

import gt_conventions as C
import e5_common as E


# ------------------------------------------------------------------------------------------------------------------------------------------------ numpy ops
def erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7): numpy has no erf and the runtime must not need scipy."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def lin(x: np.ndarray, W: Dict[str, np.ndarray], name: str) -> np.ndarray:
    return x @ W[name + ".weight"].T + W[name + ".bias"]


def mha(xq: np.ndarray, xkv: np.ndarray, W, p: str, heads: int) -> np.ndarray:
    """Multi-head attention with separate q/k/v/o projections (no mask: one text, no padding)."""
    Lq, D = xq.shape
    Lk = xkv.shape[0]
    dh = D // heads
    q = lin(xq, W, p + ".q").reshape(Lq, heads, dh).transpose(1, 0, 2)
    k = lin(xkv, W, p + ".k").reshape(Lk, heads, dh).transpose(1, 0, 2)
    v = lin(xkv, W, p + ".v").reshape(Lk, heads, dh).transpose(1, 0, 2)
    att = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(dh))
    return lin((att @ v).transpose(1, 0, 2).reshape(Lq, D), W, p + ".o")


# ------------------------------------------------------------------------------------------------------------------------------------------------ model
class E5NumpyTranslator:
    def __init__(self, weights: Dict[str, np.ndarray], meta: dict, vocab: List[str]):
        if meta.get("e5gt_version") != E.E5GT_VERSION:
            raise ValueError(f"E5 translator file version {meta.get('e5gt_version')} != {E.E5GT_VERSION}: export it again")
        self.W = {k: np.asarray(v, np.float32) for k, v in weights.items()}      # float16 files are widened once, here
        self.meta = meta
        self.enc, self.head = meta["encoder"], meta["head"]
        self.tok = E.WordPiece(vocab)
        self.stats = C.Stats.from_dict(meta["stats"])
        self.timing = C.Timing(**meta["timing"])
        self.prefix, self.max_tokens = meta.get("prefix", E.PREFIX), int(meta.get("max_tokens", E.MAX_TOKENS))
        if self.head["w_max"] != E.W_MAX:
            raise ValueError(f"this translator was trained for {self.head['w_max']} windows, e5_common says {E.W_MAX}: export it again")

    @classmethod
    def from_npz(cls, path: str) -> "E5NumpyTranslator":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        vocab = str(z["vocab_txt"]).split("\n")
        weights = {k: z[k] for k in z.files if k not in ("meta_json", "vocab_txt")}
        return cls(weights, meta, vocab)

    def save_npz(self, path: str):
        np.savez(path, meta_json=np.array(json.dumps(self.meta)), vocab_txt=np.array("\n".join(self.tok.vocab)), **self.W)

    # ---- BERT encoder (HF BertModel, hidden_act gelu, post-LayerNorm)
    def encode(self, ids: List[int]) -> np.ndarray:
        W, e = self.W, self.enc
        L = len(ids)
        x = W["encoder.embeddings.word_embeddings.weight"][ids] + W["encoder.embeddings.position_embeddings.weight"][:L] + W["encoder.embeddings.token_type_embeddings.weight"][0]
        x = layernorm(x, W["encoder.embeddings.LayerNorm.weight"], W["encoder.embeddings.LayerNorm.bias"], e["layer_norm_eps"])
        for i in range(e["num_hidden_layers"]):
            p = f"encoder.encoder.layer.{i}."
            H, D = e["num_attention_heads"], e["hidden_size"]
            dh = D // H
            q = lin(x, W, p + "attention.self.query").reshape(L, H, dh).transpose(1, 0, 2)
            k = lin(x, W, p + "attention.self.key").reshape(L, H, dh).transpose(1, 0, 2)
            v = lin(x, W, p + "attention.self.value").reshape(L, H, dh).transpose(1, 0, 2)
            att = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(dh))
            ctx = (att @ v).transpose(1, 0, 2).reshape(L, D)
            x = layernorm(lin(ctx, W, p + "attention.output.dense") + x, W[p + "attention.output.LayerNorm.weight"], W[p + "attention.output.LayerNorm.bias"], e["layer_norm_eps"])
            h = gelu(lin(x, W, p + "intermediate.dense"))
            x = layernorm(lin(h, W, p + "output.dense") + x, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"], e["layer_norm_eps"])
        return x

    # ---- window decoder (pre-LayerNorm blocks: self-attention among the window queries, cross-attention to the text, feed-forward)
    def decode_head(self, mem_in: np.ndarray):
        W, h = self.W, self.head
        mem = layernorm(lin(mem_in, W, "head.mem_proj"), W["head.mem_ln.weight"], W["head.mem_ln.bias"], 1e-5)
        x = W["head.q.weight"]                                              # [W_MAX + 1, d]: W_MAX window queries, then the length query
        for i in range(h["layers"]):
            p = f"head.blocks.{i}."
            h1 = layernorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"], 1e-5)
            x = x + mha(h1, h1, W, p + "sa", h["heads"])
            x = x + mha(layernorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"], 1e-5), mem, W, p + "ca", h["heads"])
            x = x + lin(gelu(lin(layernorm(x, W[p + "ln3.weight"], W[p + "ln3.bias"], 1e-5), W, p + "ff1")), W, p + "ff2")
        x = layernorm(x, W["head.out_ln.weight"], W["head.out_ln.bias"], 1e-5)
        win = lin(x[:E.W_MAX], W, "head.win_out")                           # [W_MAX, 3]
        return dict(len_logits=lin(x[E.W_MAX], W, "head.len_out"), spec_logit=win[:, 0], turn=np.tanh(win[:, 1]), speed=E.sigmoid(win[:, 2]).astype(np.float32))

    def raw(self, text: str) -> dict:
        return self.decode_head(self.encode(self.tok.encode(text, self.prefix, self.max_tokens)))

    def translate(self, text: str) -> C.Plan:
        o = self.raw(text)
        plan = E.decode_outputs(o["len_logits"], o["spec_logit"], o["turn"], o["speed"], self.stats)
        plan.notes.append("plan from the E5 translator: turn and speed are snapped to the plan bins (about 3 degrees and 0.14 m/s)")
        return plan


# ------------------------------------------------------------------------------------------------------------------------------------------------ test helper
def random_translator(seed: int = 0, vocab_size: int = 64, hidden: int = 32, layers: int = 2, heads: int = 4, ff: int = 64, d: int = 32, hlayers: int = 2, hheads: int = 4,
                      hff: int = 64) -> E5NumpyTranslator:
    """A tiny random-weight translator with the real parameter names, for tests that must run without torch / transformers / a download."""
    rng = np.random.default_rng(seed)

    def r(*s, scale=0.1):
        return (rng.standard_normal(s) * scale).astype(np.float32)

    def ln(prefix, n):
        return {prefix + ".weight": np.ones(n, np.float32) + r(n, scale=0.02), prefix + ".bias": r(n, scale=0.02)}

    def li(prefix, o, i):
        return {prefix + ".weight": r(o, i), prefix + ".bias": r(o, scale=0.02)}

    W: Dict[str, np.ndarray] = {}
    W["encoder.embeddings.word_embeddings.weight"] = r(vocab_size, hidden)
    W["encoder.embeddings.position_embeddings.weight"] = r(512, hidden)
    W["encoder.embeddings.token_type_embeddings.weight"] = r(2, hidden)
    W.update(ln("encoder.embeddings.LayerNorm", hidden))
    for i in range(layers):
        p = f"encoder.encoder.layer.{i}."
        for n in ("query", "key", "value"):
            W.update(li(p + "attention.self." + n, hidden, hidden))
        W.update(li(p + "attention.output.dense", hidden, hidden)); W.update(ln(p + "attention.output.LayerNorm", hidden))
        W.update(li(p + "intermediate.dense", ff, hidden)); W.update(li(p + "output.dense", hidden, ff)); W.update(ln(p + "output.LayerNorm", hidden))
    W.update(li("head.mem_proj", d, hidden)); W.update(ln("head.mem_ln", d))
    W["head.q.weight"] = r(E.W_MAX + 1, d, scale=0.5)
    for i in range(hlayers):
        p = f"head.blocks.{i}."
        for n in ("ln1", "ln2", "ln3"):
            W.update(ln(p + n, d))
        for a in ("sa", "ca"):
            for n in "qkvo":
                W.update(li(f"{p}{a}.{n}", d, d))
        W.update(li(p + "ff1", hff, d)); W.update(li(p + "ff2", d, hff))
    W.update(ln("head.out_ln", d)); W.update(li("head.win_out", 3, d)); W.update(li("head.len_out", E.W_MAX, d))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"tok{i}" for i in range(vocab_size - 4)]
    vocab[4:12] = ["query", ":", "walk", "right", "left", "then", "##ing", "turn"]
    meta = dict(e5gt_version=E.E5GT_VERSION, encoder=dict(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=ff, layer_norm_eps=1e-12),
                head=dict(d=d, layers=hlayers, heads=hheads, ff=hff, w_max=E.W_MAX), stats=C.Stats().to_dict(), timing=dict(W=C.DEFAULT_WINDOW["W"], stride=C.DEFAULT_WINDOW["stride"],
                dt=C.DEFAULT_WINDOW["dt"]), prefix=E.PREFIX, max_tokens=E.MAX_TOKENS)
    return E5NumpyTranslator(W, meta, vocab)
