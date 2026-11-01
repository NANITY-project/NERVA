"""e5_common.py - the torch-free parts of the E5 Goal Translator (stage 1 of the "no word list" plan).

The text side is a fine-tuned E5-small encoder (MIT; check the model card of whichever checkpoint you use). It reads text through its OWN pretrained WordPiece vocabulary,
so there is no list of motion words anywhere: "veer", "mosey" or "swerve" land near the words the encoder already knows. The output side replaces the 33 turn bins and
16 speed bins with CONTINUOUS numbers, one set per window, produced by a small decoder:

    text -> [E5 encoder] -> token states -> [window decoder, W_MAX learned window queries + 1 length query] ->
        per window j:   spec_logit_j   (is the window specified at all, or UNSPEC = no goal)
                        turn_j         in [-1, 1]  (fraction of the prior's turn_max, same sign convention as gt_conventions)
                        speed_j        in [0, 1]   (fraction of the prior's speed_max)
        once:           len_logits     (how many windows, 1..W_MAX, as a classification)

What this file holds (numpy / stdlib only, so the UI and the tests can import it without torch):
  * WordPiece       a BERT-uncased tokenizer (basic tokenizer + greedy WordPiece) that must match HF's BertTokenizer on the same vocab (tests/test_e5_translator.py checks it)
  * row_targets     a dataset row (build_dataset.py) -> the training targets
  * decode_outputs  raw network outputs -> a gt_conventions.Plan (snapped to the existing bins so goal_plan.py and everything downstream stay unchanged)
"""
from __future__ import annotations
import json, math, unicodedata
from typing import List, Optional, Sequence, Tuple

import numpy as np

import gt_conventions as C

E5GT_VERSION = 1
W_MAX = C.MAX_WINDOWS            # 75 windows (about 120 s at the current prior's timing)
MAX_TOKENS = 128                 # WordPiece tokens incl. [CLS] [SEP] and the "query: " prefix. Longer text is rejected, never silently truncated (spec section 3)
PREFIX = "query: "               # E5 was trained with this prefix on short queries; we always add it
SPEC_THRESHOLD = 0.5


# ------------------------------------------------------------------------------------------------------------------------------------------------ tokenizer
def _is_whitespace(ch: str) -> bool:
    if ch in " \t\n\r":
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF) or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))


def _strip_accents(tok: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", tok) if unicodedata.category(c) != "Mn")


def _split_punct(tok: str) -> List[str]:
    out, start_new = [], True
    for ch in tok:
        if _is_punct(ch):
            out.append([ch]); start_new = True
        else:
            if start_new:
                out.append([])
            start_new = False
            out[-1].append(ch)
    return ["".join(x) for x in out]


def basic_tokenize(text: str) -> List[str]:
    """BERT BasicTokenizer with do_lower_case=True (accents stripped), the setting E5 and bert-base-uncased use."""
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or _is_control(ch):
            continue
        cleaned.append(" " if _is_whitespace(ch) else ch)
    text = "".join(cleaned)
    spaced = []
    for ch in text:
        if _is_cjk(ord(ch)):
            spaced += [" ", ch, " "]
        else:
            spaced.append(ch)
    text = unicodedata.normalize("NFC", "".join(spaced))
    toks = []
    for tok in text.split():
        toks += _split_punct(_strip_accents(tok.lower()))
    return " ".join(toks).split()


class WordPiece:
    def __init__(self, vocab: Sequence[str], unk="[UNK]", cls="[CLS]", sep="[SEP]", max_chars=100):
        self.vocab = list(vocab)
        self.ids = {t: i for i, t in enumerate(self.vocab)}
        for t in (unk, cls, sep):
            if t not in self.ids:
                raise ValueError(f"vocab has no {t}")
        self.unk, self.cls, self.sep, self.max_chars = self.ids[unk], self.ids[cls], self.ids[sep], max_chars

    def _word(self, w: str) -> List[int]:
        if len(w) > self.max_chars:
            return [self.unk]
        out, start = [], 0
        while start < len(w):
            end, cur = len(w), None
            while start < end:
                piece = w[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.ids:
                    cur = self.ids[piece]
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            out.append(cur)
            start = end
        return out

    def encode(self, text: str, prefix: str = PREFIX, max_len: int = MAX_TOKENS, truncate: bool = False) -> List[int]:
        """[CLS] <prefix + text as WordPieces> [SEP]. Longer than max_len: raises C.PlanError (inference: rejected, never silently cut), or with truncate=True keeps the
        first max_len - 1 tokens plus [SEP] (training only, where a rare over-long row should not stop an epoch)."""
        ids = [self.cls]
        for w in basic_tokenize(prefix + text):
            ids += self._word(w)
        ids.append(self.sep)
        if len(ids) > max_len:
            if truncate:
                return ids[:max_len - 1] + [self.sep]
            raise C.PlanError(f"the text is too long for the translator ({len(ids)} tokens, the limit is {max_len}): shorten it")
        return ids


# ------------------------------------------------------------------------------------------------------------------------------------------------ data
def load_rows(path: str, max_bytes: int = C.MAX_INPUT_BYTES) -> List[dict]:
    """The rows written by build_dataset.py (same filter as gt_torch.load_rows, without importing torch or the NANITY training code)."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if len(r["text"].encode("utf-8")) <= max_bytes and 1 <= len(r["windows"]) <= W_MAX:
                rows.append(r)
    return rows


def row_targets(windows) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """row['windows'] (None | [turn_bin, speed_bin] per window) -> (n, spec[W_MAX], turn[W_MAX] in [-1, 1], speed[W_MAX] in [0, 1]).
    The values are the bin centres in the prior's own units, so they need no Stats: turn = (bin - mid) / mid, speed = bin / (n_bins - 1)."""
    n = len(windows)
    if not 1 <= n <= W_MAX:
        raise ValueError(f"{n} windows (1..{W_MAX} allowed)")
    spec = np.zeros(W_MAX, np.float32)
    turn = np.zeros(W_MAX, np.float32)
    speed = np.zeros(W_MAX, np.float32)
    for j, w in enumerate(windows):
        if w is None:
            continue
        spec[j] = 1.0
        turn[j] = (w[0] - C.TURN_MID) / C.TURN_MID
        speed[j] = w[1] / (C.N_SPEED_BINS - 1)
    return n, spec, turn, speed


# ------------------------------------------------------------------------------------------------------------------------------------------------ decode
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def decode_outputs(len_logits, spec_logit, turn, speed, st: C.Stats, thr: float = SPEC_THRESHOLD, source: str = "model") -> C.Plan:
    """Raw outputs of ONE text -> a Plan. n = the most likely length; a window is specified when sigmoid(spec_logit) >= thr, then turn / speed are scaled by the prior's
    turn_max / speed_max and snapped to the existing bins (about 3 degrees and 0.14 m/s: finer than the prior can follow), so everything downstream is unchanged."""
    len_logits = np.asarray(len_logits, np.float64).reshape(-1)
    n = int(np.argmax(len_logits[:W_MAX])) + 1
    wins = []
    for j in range(n):
        if float(sigmoid(spec_logit[j])) < thr:
            wins.append(C.UNSPEC_WIN)
        else:
            wins.append(C.spec_win(float(np.clip(turn[j], -1.0, 1.0)) * st.turn_max, float(np.clip(speed[j], 0.0, 1.0)) * st.speed_max, st))
    return C.Plan(wins, source=source)
