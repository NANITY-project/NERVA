"""gt_torch.py - everything torch-specific about the Goal Translator: model presets, dataset, constrained decoding, checkpoint IO. Imported by train / evaluate / export."""
from __future__ import annotations
import dataclasses, json, os, random, sys
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def find_neon_training():
    """The NANITY architecture (modeling_nanity.py) lives in the NANITY-NEON checkout: --neon-dir is not needed, it is looked for like the UI does
    (NEON_DIR, next to this repo, ~/NANITY-NEON, UI/.cache/NANITY-NEON)."""
    root = os.path.dirname(os.path.dirname(HERE))
    for c in (os.environ.get("NEON_DIR"), os.path.join(os.path.dirname(root), "NANITY-NEON"), os.path.join(os.path.dirname(root), "NEON"), os.path.expanduser("~/NANITY-NEON"),
              os.path.join(root, "UI", ".cache", "NANITY-NEON")):
        if c and os.path.exists(os.path.join(c, "training", "modeling_nanity.py")):
            return os.path.join(c, "training")
    raise SystemExit("modeling_nanity.py not found: clone https://github.com/NANITY-project/NANITY-NEON next to this repo or set NEON_DIR")


sys.path.insert(0, find_neon_training())
from modeling_nanity import NanityConfig, NanityForCausalLM       # noqa: E402

import gt_conventions as C        # noqa: E402
import gt_corrupt                 # noqa: E402

CONTEXT = 512                     # 1 + 256 input bytes + 1 + (3 * 75 + 1) output tokens = 484

PRESETS = {                       # (d, layers, heads, ff) - heads * head_dim = d; kv heads = heads (the model is too small for GQA to pay)
    "tiny":  dict(d=128, layers=4, heads=4, ff=384),              # ~1 M: smoke tests
    "small": dict(d=256, layers=6, heads=4, ff=768),              # ~5 M: the spec's proposal (same size as the current prior)
    "base":  dict(d=384, layers=8, heads=6, ff=1024),             # ~14 M
}


def make_config(preset: str) -> NanityConfig:
    p = PRESETS[preset]
    return NanityConfig(vocab_size=C.VOCAB_SIZE, embedding_length=p["d"], block_count=p["layers"], attention_head_count=p["heads"], attention_head_count_kv=p["heads"],
                        attention_key_length=p["d"] // p["heads"], feed_forward_length=p["ff"], context_length=CONTEXT, tie_embeddings=True)


def build_model(cfg: NanityConfig) -> NanityForCausalLM:
    return NanityForCausalLM(cfg)


# ------------------------------------------------------------------------------------------------------------------------------------------------ data
def load_rows(path: str, max_bytes=C.MAX_INPUT_BYTES) -> List[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if len(r["text"].encode("utf-8")) <= max_bytes and 1 <= len(r["windows"]) <= C.MAX_WINDOWS:
                rows.append(r)
    return rows


class GTDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict], augment: bool, seed: int = 0):
        self.rows, self.augment, self.seed = rows, augment, seed
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        if self.augment and self._rng.random() < 0.7:
            text = gt_corrupt.corrupt(text, self._rng)
        src = C.encode_input(text)
        plan = C.Plan([C.UNSPEC_WIN if w is None else C.Win(int(w[0]), int(w[1])) for w in r["windows"]])
        out = C.plan_to_tokens(plan)
        return src, out


def collate(batch):
    n = max(len(s) + len(o) for s, o in batch)
    ids = torch.full((len(batch), n), C.PAD, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    for k, (s, o) in enumerate(batch):
        seq = s + o
        ids[k, :len(seq)] = torch.tensor(seq)
        lab[k, len(s):len(seq)] = torch.tensor(o)                  # loss on the OUTPUT tokens only (input bytes are context, not targets)
    return ids, lab


def loss_fn(model, ids, lab):
    logits, _ = model(ids[:, :-1])                                 # right padding + causal attention: pad positions never influence real ones
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), lab[:, 1:].reshape(-1), ignore_index=-100)


# ------------------------------------------------------------------------------------------------------------------------------------------------ constrained greedy decoding
@torch.no_grad()
def decode_plan(model: NanityForCausalLM, text: str, device="cpu") -> C.Plan:
    """Greedy decoding with the output grammar applied as a logit mask, KV cache. A malformed plan cannot be produced; a wrong one can (that is what evaluation measures).
    Reaching the window cap without END is an error (PlanError), not a truncation."""
    model.eval()
    src = C.encode_input(text)
    ids = torch.tensor([src], dtype=torch.long, device=device)
    g = C.Grammar()
    out: List[int] = []
    past = None
    wins = 0
    n_ctx = 0
    for _ in range(3 * C.MAX_WINDOWS + 2):
        n_ctx += ids.shape[1]
        # The explicit all-ones mask is NOT optional: without it a cached single-token step goes through SDPA's is_causal=True, which with 1 query and n keys is aligned to the TOP-LEFT
        # and lets the query see only key 0 (measured: logits off by ~13 vs the full forward). modeling_nanity.NanityAttention needs `is_causal = attn_mask is None and past_kv is None`.
        logits, past = model(ids, attention_mask=torch.ones(1, n_ctx, dtype=torch.long, device=ids.device), past_key_values=past, use_cache=True)
        mask = torch.from_numpy(g.mask()).to(device)
        t = int((logits[0, -1].float() + mask).argmax())
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
        ids = torch.tensor([[t]], dtype=torch.long, device=device)
    raise C.PlanError("the output was cut off before END")


# ------------------------------------------------------------------------------------------------------------------------------------------------ checkpoint
def save_checkpoint(path, model, cfg, meta: dict, step: int, metrics: Optional[dict] = None, opt=None):
    d = dict(model=model.state_dict(), nanity_config=dataclasses.asdict(cfg), gt=meta, step=step, metrics=metrics or {})
    if opt is not None:
        d["opt"] = opt.state_dict()
    torch.save(d, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = NanityConfig(**ck["nanity_config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck
