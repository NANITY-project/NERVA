"""
gt_conventions.py - the ONE place the Goal Translator's conventions live (spec section 6).

Used by data generation, training, the stage-C bootstrap, the exporter, and (as a byte-identical copy, UI/gt_conventions.py in the NERVA repo) the UI.
Standard library + numpy only. No torch.

CONVENTIONS
  * Left / right is the CHARACTER's own, never the camera's.
  * turn > 0 is a LEFT turn.  Source of truth in the code that already exists: NERVA tools/train_cvqvae.py canonicalize() makes x the "old left axis" and the heading
    vector (sin yaw, cos yaw), so yaw growing turns the heading toward +x = the character's left; tests/test_goal_conditioning.py asserts "a left turn (+) has dx > 0".
    TURN_POSITIVE_IS_LEFT below is that decision, written once. `check_sign.py` re-derives it from a clip whose direction you know, so a mirrored dataset is caught.
  * turn is radians per window, speed is metres per second.
  * The goal numbers are the prior's: [dyaw, dx, dz, speed] (train_cvqvae.GOAL_NAMES). The translator emits only (turn, speed); dx, dz come from arc_goal().

TIMING (two different seconds - do not mix them up)
  goal_seconds  = (W - 1) * dt   what ONE window's goal was measured over (goal_features: last = valid - 1)  -> used by arc_goal(path = speed * goal_seconds)
  step_seconds  = stride * dt    how far the plan advances per window (the prior emits a new window every `stride` frames)  -> used to turn "3 seconds" into windows
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------------------------ constants
LAYOUT_VERSION = 1                       # written to the model metadata; bump when the token layout changes (an upper-body group is added later, spec section 11)
TURN_POSITIVE_IS_LEFT = True             # the sign convention, decided once (see header)
GOAL_NAMES = ["dyaw", "dx", "dz", "speed"]

# input / output limits (spec section 3 and 5, all "proposal")
MAX_INPUT_BYTES = 256
MAX_WINDOWS = 75                         # about 120 s at 1.6 s per window; matches the UI's length cap
N_TURN_BINS = 33                         # odd: the middle bin is exactly 0 (about 3 degrees per bin at turn_max ~ 0.9 rad)
N_SPEED_BINS = 16                        # bin 0 = stand still
DEFAULT_WINDOW = dict(W=64, stride=48, dt=1.0 / 30.0)     # only used when no prior is given; the real values come from the prior / VAE

# ------------------------------------------------------------------------------------------------------------------------------------------------ token layout
# 0..255 bytes | specials | TURN bins | SPEED bins ; padded to a multiple of 64 for the matmuls
PAD, BOS, SEP, WIN, UNSPEC, END = 256, 257, 258, 259, 260, 261
RESERVED_0, RESERVED_1 = 262, 263                         # kept free for a later token group (upper body, style header)
TURN_BASE = 264
SPEED_BASE = TURN_BASE + N_TURN_BINS                      # 297
N_USED = SPEED_BASE + N_SPEED_BINS                        # 313
VOCAB_SIZE = ((N_USED + 63) // 64) * 64                   # 320
TURN_MID = N_TURN_BINS // 2

SPECIAL_NAMES = {PAD: "<pad>", BOS: "<bos>", SEP: "<sep>", WIN: "<win>", UNSPEC: "<unspec>", END: "<end>", RESERVED_0: "<r0>", RESERVED_1: "<r1>"}


def token_name(t: int) -> str:
    if t < 256:
        return f"<0x{t:02X}>"
    if t in SPECIAL_NAMES:
        return SPECIAL_NAMES[t]
    if TURN_BASE <= t < SPEED_BASE:
        return f"<turn_{t - TURN_BASE}>"
    if SPEED_BASE <= t < N_USED:
        return f"<speed_{t - SPEED_BASE}>"
    return f"<unused_{t}>"


def vocab_strings() -> List[str]:
    """Token strings for the GGUF tokenizer.ggml.tokens array (one per id)."""
    return [token_name(t) for t in range(VOCAB_SIZE)]


# ------------------------------------------------------------------------------------------------------------------------------------------------ prior stats
@dataclass
class Stats:
    """The prior's goal_stats (info.goal in NEON): what the training windows of the prior really contain. The translator can only ask for this much."""
    turn_max: float = 0.9              # rad per window (95th percentile of |dyaw|)
    speed_cruise: float = 1.2          # m/s (median of moving windows)
    speed_max: float = 2.1             # m/s (95th percentile of moving windows)

    @staticmethod
    def from_dict(d: dict) -> "Stats":
        return Stats(float(d["turn_max"]), float(d["speed_cruise"]), float(d["speed_max"]))

    def to_dict(self) -> dict:
        return dict(turn_max=self.turn_max, speed_cruise=self.speed_cruise, speed_max=self.speed_max)


@dataclass
class Timing:
    W: int = DEFAULT_WINDOW["W"]
    stride: int = DEFAULT_WINDOW["stride"]
    dt: float = DEFAULT_WINDOW["dt"]

    @property
    def goal_seconds(self) -> float:
        return (self.W - 1) * self.dt

    @property
    def step_seconds(self) -> float:
        return self.stride * self.dt


# ------------------------------------------------------------------------------------------------------------------------------------------------ bins
def turn_to_bin(turn: float, st: Stats) -> int:
    x = float(np.clip(turn, -st.turn_max, st.turn_max)) / st.turn_max             # -1..1
    return int(np.clip(round(TURN_MID + x * TURN_MID), 0, N_TURN_BINS - 1))


def bin_to_turn(b: int, st: Stats) -> float:
    return (int(b) - TURN_MID) / TURN_MID * st.turn_max


def speed_to_bin(speed: float, st: Stats) -> int:
    x = float(np.clip(speed, 0.0, st.speed_max)) / st.speed_max
    return int(np.clip(round(x * (N_SPEED_BINS - 1)), 0, N_SPEED_BINS - 1))


def bin_to_speed(b: int, st: Stats) -> float:
    return int(b) / (N_SPEED_BINS - 1) * st.speed_max


# ------------------------------------------------------------------------------------------------------------------------------------------------ plan
@dataclass(frozen=True)
class Win:
    """One motion window of the plan. turn_bin / speed_bin both None = UNSPECIFIED (NEON receives no goal for it)."""
    turn_bin: Optional[int] = None
    speed_bin: Optional[int] = None

    @property
    def unspecified(self) -> bool:
        return self.turn_bin is None

    def __post_init__(self):
        if (self.turn_bin is None) != (self.speed_bin is None):
            raise ValueError("a window is either fully specified (turn and speed) or UNSPEC")
        if self.turn_bin is not None and not (0 <= self.turn_bin < N_TURN_BINS and 0 <= self.speed_bin < N_SPEED_BINS):
            raise ValueError(f"bin out of range: {self.turn_bin}, {self.speed_bin}")


UNSPEC_WIN = Win()


def spec_win(turn: float, speed: float, st: Stats) -> Win:
    return Win(turn_to_bin(turn, st), speed_to_bin(speed, st))


@dataclass
class Plan:
    windows: List[Win] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)      # text the goals cannot express (stage C reports it; the learned model cannot, spec section 11)
    source: str = ""                                      # "stage_c" | "model" | ...
    notes: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.windows)


class PlanError(ValueError):
    """The translator's output is invalid or hit the window cap. Shown to the user, never hidden (spec section 10)."""


# ------------------------------------------------------------------------------------------------------------------------------------------------ token <-> plan
def encode_input(text: str) -> List[int]:
    b = text.encode("utf-8")
    if len(b) > MAX_INPUT_BYTES:
        raise PlanError(f"text is {len(b)} bytes; the limit is {MAX_INPUT_BYTES} (longer text is rejected, not truncated)")
    return [BOS] + list(b) + [SEP]


def plan_to_tokens(plan: Plan) -> List[int]:
    out: List[int] = []
    for w in plan.windows:
        out.append(WIN)
        if w.unspecified:
            out.append(UNSPEC)
        else:
            out += [TURN_BASE + w.turn_bin, SPEED_BASE + w.speed_bin]
    out.append(END)
    return out


def tokens_to_plan(tokens: Sequence[int]) -> Plan:
    """Strict parse of an OUTPUT token sequence (after SEP). Raises PlanError on anything the grammar forbids, or on a missing END."""
    g = Grammar()
    wins: List[Win] = []
    cur_turn = None
    for k, t in enumerate(tokens):
        if g.state == "done":
            raise PlanError(f"{len(tokens) - k} token(s) after END")
        if not g.allowed(t):
            raise PlanError(f"token {token_name(t)} is not allowed in state {g.state}")
        g.advance(t)
        if t == WIN:
            pass
        elif t == UNSPEC:
            wins.append(UNSPEC_WIN)
        elif TURN_BASE <= t < SPEED_BASE:
            cur_turn = t - TURN_BASE
        elif SPEED_BASE <= t < N_USED:
            wins.append(Win(cur_turn, t - SPEED_BASE))
        elif t == END:
            if k != len(tokens) - 1:
                raise PlanError(f"{len(tokens) - 1 - k} token(s) after END")
            if len(wins) > MAX_WINDOWS:
                raise PlanError(f"plan has {len(wins)} windows; the cap is {MAX_WINDOWS}")
            return Plan(wins, source="tokens")
    raise PlanError("the plan has no END token (the output was cut off or hit the length cap)")


class Grammar:
    """The output grammar (spec section 5):   WIN (TURN SPEED | UNSPEC) ... END     END only after a window, at least one window.
    States: 'start' (expects WIN), 'win' (TURN or UNSPEC), 'turn' (SPEED), 'after' (WIN or END), 'done'."""
    def __init__(self):
        self.state = "start"

    def allowed_ids(self) -> List[int]:
        s = self.state
        if s == "start":
            return [WIN]
        if s == "win":
            return [UNSPEC] + list(range(TURN_BASE, SPEED_BASE))
        if s == "turn":
            return list(range(SPEED_BASE, N_USED))
        if s == "after":
            return [WIN, END]
        return []

    def allowed(self, t: int) -> bool:
        return t in self.allowed_ids()

    def advance(self, t: int):
        if not self.allowed(t):
            raise PlanError(f"token {token_name(t)} is not allowed in state {self.state}")
        if t == WIN:
            self.state = "win"
        elif t == UNSPEC:
            self.state = "after"
        elif TURN_BASE <= t < SPEED_BASE:
            self.state = "turn"
        elif SPEED_BASE <= t < N_USED:
            self.state = "after"
        elif t == END:
            self.state = "done"

    def mask(self) -> np.ndarray:
        """float32 [VOCAB_SIZE]: 0 for allowed ids, -inf for everything else (added to the logits)."""
        m = np.full(VOCAB_SIZE, -np.inf, np.float32)
        m[self.allowed_ids()] = 0.0
        return m


def grammar_state_after(tokens: Sequence[int]) -> str:
    g = Grammar()
    for t in tokens:
        g.advance(t)
    return g.state


# ------------------------------------------------------------------------------------------------------------------------------------------------ goals
def arc_goal(turn, path, seconds):
    """The goal numbers of a walk along a circular arc: heading change `turn` (rad) over `path` metres of pelvis path lasting `seconds` -> [..., 4] (dyaw, dx, dz, speed).
    Identical formula to NERVA tools/train_cvqvae.arc_goal (tests/test_gt_conventions.py in NERVA compares them when torch is installed)."""
    turn = np.asarray(turn, np.float64)
    path = np.asarray(path, np.float64)
    h = turn / 2.0
    chord = path * np.sinc(h / np.pi)
    return np.stack(np.broadcast_arrays(turn, chord * np.sin(h), chord * np.cos(h), path / seconds), -1).astype(np.float32)


def plan_to_goals(plan: Plan, st: Stats, timing: Timing) -> List[Optional[List[float]]]:
    """Validate (clamp to goal_stats) and derive the four numbers per window. None for an UNSPEC window (NEON then receives no goal).
    The model cannot emit an impossible combination: dx, dz are derived, never emitted."""
    out: List[Optional[List[float]]] = []
    sec = timing.goal_seconds
    for w in plan.windows:
        if w.unspecified:
            out.append(None)
            continue
        turn = float(np.clip(bin_to_turn(w.turn_bin, st), -st.turn_max, st.turn_max))
        speed = float(np.clip(bin_to_speed(w.speed_bin, st), 0.0, st.speed_max))
        out.append([float(v) for v in arc_goal(turn, speed * sec, sec)])
    return out


def describe_window(w: Win, st: Stats) -> str:
    if w.unspecified:
        return "unspecified"
    t = math.degrees(bin_to_turn(w.turn_bin, st))
    side = "left" if t > 0 else "right" if t < 0 else "straight"
    sp = bin_to_speed(w.speed_bin, st)
    return f"{side} {abs(t):.0f} deg, {sp:.2f} m/s" if sp > 0 else f"{side} {abs(t):.0f} deg, standing"
