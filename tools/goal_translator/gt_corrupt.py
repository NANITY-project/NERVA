"""gt_corrupt.py - on-the-fly surface corruption for training (spec section 9): random space removal, typos, case changes, so one (text, plan) pair is seen in many forms.
Standard library only. Never touches the LABEL, so it must not destroy information the label depends on: the direction words are protected from typos in `safe` mode
(the default, used for the bulk of training); `safe=False` lets typos hit them too (a small fraction of examples, so 'lfet' is learned as left)."""
import random
import re

_PROTECT = {"left", "right"}


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.randrange(4)
    if op == 0:                                   # drop a letter
        return word[:i] + word[i + 1:]
    if op == 1:                                   # double a letter
        return word[:i] + word[i] + word[i:]
    if op == 2:                                   # swap two neighbours
        return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i + 2 <= len(word) else word
    return word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]     # substitute


def corrupt(text: str, rng: random.Random, p_typo=0.12, p_unsafe=0.04, p_nospace=0.18, p_case=0.3, p_punct=0.15, max_bytes=256) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        core = re.sub(r"[^A-Za-z]", "", w).lower()
        if rng.random() < p_typo and core:
            if core not in _PROTECT or rng.random() < p_unsafe / p_typo:
                w = typo(w, rng)
        out.append(w)
    s = " ".join(out)
    if rng.random() < p_nospace:
        s = s.replace(" ", "")
    elif rng.random() < 0.05:
        s = re.sub(r" ", lambda m: " " * rng.randint(1, 3), s)
    r = rng.random()
    if r < p_case / 3:
        s = s.upper()
    elif r < 2 * p_case / 3:
        s = s.capitalize()
    elif r < p_case:
        s = s.lower()
    if rng.random() < p_punct:
        s += rng.choice(["?", "!", "...", " pls", " lol", " thx"])
    b = s.encode("utf-8")
    if len(b) > max_bytes:
        s = b[:max_bytes].decode("utf-8", "ignore")
    return s
