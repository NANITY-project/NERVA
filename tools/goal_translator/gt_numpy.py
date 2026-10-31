"""
gt_numpy.py - the Goal Translator without torch: a numpy forward pass of the NANITY architecture (RMSNorm, adjacent-pair RoPE, GQA, SwiGLU, tied head) + the grammar-masked greedy
decoder, reading the GGUF that export_goal_translator.py writes. ~1-20 M parameters: fast enough on a CPU for a sentence at a time.

It is the reference the UI (NERVA UI/goal_plan.py) uses when NEON is not serving the translator, and what tests/test_goal_translator.py compares against torch and the C++ op.
Needs: numpy, gguf.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional

import numpy as np

import gt_conventions as C


def _field(r, key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    if len(v) == 1:
        x = v[0]
        return x.item() if hasattr(x, "item") else x
    return bytes(v).decode("utf-8")


def _rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyTranslator:
    def __init__(self, tensors: Dict[str, np.ndarray], meta: dict):
        self.t, self.m = tensors, meta
        self.n_layer, self.n_head, self.n_kv, self.hd = meta["n_layer"], meta["n_head"], meta["n_kv"], meta["head_dim"]
        self.eps, self.base, self.scale = meta["eps"], meta["rope_base"], meta["rope_scale"]
        self.rot = meta["rot_dim"]
        self.stats = C.Stats(meta["turn_max"], meta["speed_cruise"], meta["speed_max"])
        self.timing = C.Timing(meta["W"], meta["stride"], meta["dt"])
        self.freq = self.base ** (-np.arange(0, self.rot, 2, dtype=np.float64) / self.rot)

    # ------------------------------------------------------------------------------------------------------------------------------------ loading
    @staticmethod
    def from_gguf(path: str) -> "NumpyTranslator":
        import gguf
        r = gguf.GGUFReader(path)
        if _field(r, "general.architecture") != "nanity" or _field(r, "nanity.goal_translator.layout_version") is None:
            raise C.PlanError(f"{path} is not a Goal Translator GGUF (general.architecture nanity + nanity.goal_translator.* keys)")
        lv = int(_field(r, "nanity.goal_translator.layout_version"))
        if lv != C.LAYOUT_VERSION:
            raise C.PlanError(f"token layout version {lv} in the file, {C.LAYOUT_VERSION} in gt_conventions.py")
        g = lambda k, d=None: _field(r, "nanity.goal_translator." + k, d)
        n_head = int(_field(r, "nanity.attention.head_count"))
        hd = int(_field(r, "nanity.attention.key_length"))
        meta = dict(n_layer=int(_field(r, "nanity.block_count")), n_head=n_head, n_kv=int(_field(r, "nanity.attention.head_count_kv")), head_dim=hd,
                    eps=float(_field(r, "nanity.attention.layer_norm_rms_epsilon", 1e-5)), rope_base=float(_field(r, "nanity.rope.freq_base", 10000.0)),
                    rope_scale=float(_field(r, "nanity.rope.scale_linear", 1.0)), rot_dim=int(_field(r, "nanity.rope.dimension_count", hd)),
                    context=int(_field(r, "nanity.context_length")), turn_max=float(g("turn_max")), speed_cruise=float(g("speed_cruise")), speed_max=float(g("speed_max")),
                    W=int(g("window_W")), stride=int(g("window_stride")), dt=float(g("window_dt")))
        for k, want in (("n_turn_bins", C.N_TURN_BINS), ("n_speed_bins", C.N_SPEED_BINS), ("max_windows", C.MAX_WINDOWS), ("max_input_bytes", C.MAX_INPUT_BYTES), ("vocab_size", C.VOCAB_SIZE),
                        ("win", C.WIN), ("unspec", C.UNSPEC), ("end", C.END), ("turn_base", C.TURN_BASE), ("speed_base", C.SPEED_BASE)):
            if int(g(k)) != want:
                raise C.PlanError(f"{k} = {g(k)} in the file, {want} in gt_conventions.py: the file was made with a different layout")
        ts = {t.name: np.asarray(t.data, np.float32).reshape(tuple(int(x) for x in t.shape[::-1])) if t.data.ndim == 1 else np.asarray(t.data, np.float32) for t in r.tensors}
        return NumpyTranslator(ts, meta)

    # ------------------------------------------------------------------------------------------------------------------------------------ forward
    def _rope(self, x, pos):                         # x [heads, hd]; adjacent pairs (v[2i], v[2i+1]); only the first rot channels
        ang = pos * self.freq / self.scale
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        y = x.copy()
        a, b = x[:, 0:self.rot:2], x[:, 1:self.rot:2]
        y[:, 0:self.rot:2] = a * c - b * s
        y[:, 1:self.rot:2] = a * s + b * c
        return y

    def step(self, tok: int, pos: int, cache: List[dict]) -> np.ndarray:
        t = self.t
        x = t["token_embd.weight"][tok].astype(np.float32)
        grp = self.n_head // self.n_kv
        for i in range(self.n_layer):
            p = f"blk.{i}."
            h = _rmsnorm(x, t[p + "attn_norm.weight"], self.eps)
            q = (t[p + "attn_q.weight"] @ h).reshape(self.n_head, self.hd)
            k = (t[p + "attn_k.weight"] @ h).reshape(self.n_kv, self.hd)
            v = (t[p + "attn_v.weight"] @ h).reshape(self.n_kv, self.hd)
            q, k = self._rope(q, pos), self._rope(k, pos)
            c = cache[i]
            c["k"].append(k); c["v"].append(v)
            K, V = np.stack(c["k"], 0), np.stack(c["v"], 0)                     # [T, n_kv, hd]
            out = np.empty((self.n_head, self.hd), np.float32)
            for hh in range(self.n_head):
                kv = hh // grp
                sc = K[:, kv] @ q[hh] / np.sqrt(self.hd)
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                out[hh] = sc @ V[:, kv]
            x = x + t[p + "attn_output.weight"] @ out.reshape(-1)
            h = _rmsnorm(x, t[p + "ffn_norm.weight"], self.eps)
            x = x + t[p + "ffn_down.weight"] @ (_silu(t[p + "ffn_gate.weight"] @ h) * (t[p + "ffn_up.weight"] @ h))
        x = _rmsnorm(x, t["output_norm.weight"], self.eps)
        head = t["output.weight"] if "output.weight" in t else t["token_embd.weight"]
        return head @ x

    def prefill_logits(self, ids: List[int]) -> np.ndarray:
        """Logits after the last token of ids (no cache reuse): for the equivalence tests."""
        cache = [dict(k=[], v=[]) for _ in range(self.n_layer)]
        lg = None
        for pos, tok in enumerate(ids):
            lg = self.step(tok, pos, cache)
        return lg

    # ------------------------------------------------------------------------------------------------------------------------------------ decode
    def decode(self, text: str) -> C.Plan:
        src = C.encode_input(text)
        if len(src) + 3 * C.MAX_WINDOWS + 2 > self.m["context"]:
            raise C.PlanError("the model's context is too short for the maximum input and plan")
        cache = [dict(k=[], v=[]) for _ in range(self.n_layer)]
        g = C.Grammar()
        out: List[int] = []
        pos, lg, wins = 0, None, 0
        for tok in src:
            lg = self.step(tok, pos, cache); pos += 1
        for _ in range(3 * C.MAX_WINDOWS + 2):
            t = int(np.argmax(lg + g.mask()))
            if t == C.WIN:
                wins += 1
                if wins > C.MAX_WINDOWS:
                    raise C.PlanError(f"the plan reached the {C.MAX_WINDOWS}-window cap without END")
            g.advance(t)
            out.append(t)
            if t == C.END:
                plan = C.tokens_to_plan(out)
                plan.source = "model"
                return plan
            lg = self.step(t, pos, cache); pos += 1
        raise C.PlanError("the output was cut off before END")

    translate = decode
