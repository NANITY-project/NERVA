"""goal_layout.py - what the prior is told about a window: the path goal AND, per engine channel, whether a target is given (numpy only; torch is imported lazily by one helper).

The engine drives some body parts from TARGETS: `handL_on`, `handR_on`, `look_on`, `torso_on` = 1 means "reach / look at / lean to this world point" (IK), 0 means the engine's own
procedural layer takes over (natural arm swing, head following the body, torso following the prior). The prior used to know only the four path numbers of a window. This module adds
the channel targets as conditioning, with an explicit KNOWN flag per group, so that a group can also be left UNSPECIFIED (nothing is said about it) and the prior still works:
that is trained with random masking, the same idea as the partial control of MotionBricks-style keyframe models (the app gives whatever it knows, the model fills in the rest).

Two layouts (the name list stored with the prior decides; old priors keep working):
  "path"  4 numbers   dyaw, dx, dz, speed                                           (what every prior so far was trained with)
  "full"  25 numbers  5 groups of [known, ...]:
            path    known | dyaw dx dz speed
            handL   known | on x y z          handR, look, torso the same
          x y z = the target in the window's CANONICAL frame (frame 0: pelvis at the origin facing +z, x = the body's left, y = absolute height in metres), the mean over the
          frames where the channel is on. A group has three states:
            known=0                  unspecified: nothing is said (all its numbers are 0)
            known=1, on=0            "no target": the engine's procedural layer drives this part   (its x y z are 0)
            known=1, on=1, x y z     the engine drives this part to that point

Normalisation is SCALE ONLY (mean 0, std = rms), so a masked or unspecified entry is exactly 0 both as a raw number and as the number the network sees. That is what lets NEON and
the numpy prior (which compute (raw - mean) / std) take the vector without any change: the engine side only needs a longer goal vector, which they read from the prior file.

How the pieces are used
  training     goal_features_full() measures a window's groups from the data, goal_stats_full() the scale, mask_groups_torch() hides groups at random (per window and per sequence)
  planners     expand_path(arc4, stats) turns the 4 path numbers every planner already makes into the prior's layout (channels unspecified); set_channel() gives a target or says "off"
  closed loop  measured() gives the features of a decoded window, as the prior saw them in training
"""
from __future__ import annotations
import numpy as np

PATH_NAMES = ["dyaw", "dx", "dz", "speed"]
GROUPS = ["path", "handL", "handR", "look", "torso"]
CHANNEL_COLS = {"handL": (14, 15, 16, 17), "handR": (18, 19, 20, 21), "look": (22, 23, 24, 25), "torso": (26, 27, 28, 29)}     # x y z on, columns of the 30-channel action vector
FULL_NAMES = ["path_known"] + PATH_NAMES + [f"{g}_{k}" for g in GROUPS[1:] for k in ("known", "on", "x", "y", "z")]
G_FULL = len(FULL_NAMES)                                                                                                         # 25
GROUP_COLS = {"path": [0, 1, 2, 3, 4]}
for _i, _g in enumerate(GROUPS[1:]):
    GROUP_COLS[_g] = [5 + 5 * _i + k for k in range(5)]                                                                          # known, on, x, y, z
GROUP_MATRIX = np.zeros((len(GROUPS), G_FULL), np.float32)                                                                       # group -> the columns it owns
for _i, _g in enumerate(GROUPS):
    GROUP_MATRIX[_i, GROUP_COLS[_g]] = 1.0
ON_THRESHOLD = 0.5            # a window counts as "channel on" when at least this share of its valid frames has the flag set
MIN_STD = 1e-2


# ------------------------------------------------------------------------------------------------------------------------------------------------ layout
def layout_of(x) -> str:
    """'path' or 'full' from a goal_stats dict, a name list or a goal dimension."""
    if isinstance(x, dict):
        n = len(x["mean"])
    elif isinstance(x, (int, np.integer)):
        n = int(x)
    else:
        n = len(x)
    if n == len(PATH_NAMES):
        return "path"
    if n == G_FULL:
        return "full"
    raise ValueError(f"unknown goal layout with {n} numbers (known: {len(PATH_NAMES)} = path, {G_FULL} = full)")


def names_for_stats(st) -> list:
    return list(PATH_NAMES) if layout_of(st) == "path" else list(FULL_NAMES)


# ------------------------------------------------------------------------------------------------------------------------------------------------ measuring (training data)
def goal_features_full(path4, Xc, valid, on_thr: float = ON_THRESHOLD) -> np.ndarray:
    """path4 [N,4] (train_cvqvae.goal_features), CANONICALISED windows Xc [N,T,30], valid frames [N] -> [N,25] raw features with every group known."""
    path4 = np.asarray(path4, np.float32)
    N, T, C = Xc.shape
    if C != 30 or path4.shape != (N, 4):
        raise ValueError(f"windows [N,T,30] and path goals [N,4] expected, got {Xc.shape} and {path4.shape}")
    valid = np.clip(np.asarray(valid, np.int64), 1, T)
    inside = np.arange(T)[None, :] < valid[:, None]
    out = np.zeros((N, G_FULL), np.float32)
    out[:, GROUP_COLS["path"][0]] = 1.0
    out[:, GROUP_COLS["path"][1:]] = path4
    for g in GROUPS[1:]:
        cx, cy, cz, con = CHANNEL_COLS[g]
        k_known, k_on, k_x, k_y, k_z = GROUP_COLS[g]
        m = (Xc[:, :, con] > 0.5) & inside                                           # frames where the engine is told to drive this part
        is_on = (m.sum(1) / valid) >= on_thr
        w = m.astype(np.float64) * is_on[:, None]
        cnt = np.maximum(w.sum(1), 1.0)
        out[:, k_known] = 1.0
        out[:, k_on] = is_on
        for col, k in ((cx, k_x), (cy, k_y), (cz, k_z)):
            out[:, k] = (Xc[:, :, col].astype(np.float64) * w).sum(1) / cnt
    return out


def goal_stats_full(g, legacy: dict, on_thr: float = ON_THRESHOLD) -> dict:
    """Scale-only statistics for the full layout: mean 0 everywhere, std 1 for the flags, rms for the numbers (the target coordinates over the windows where the group is on).
    legacy = train_cvqvae.goal_stats of the path numbers: turn_max, speed_cruise and speed_max carry over (planners and the closed loop read them)."""
    g = np.asarray(g, np.float64)
    if g.ndim != 2 or g.shape[1] != G_FULL:
        raise ValueError(f"features [N,{G_FULL}] expected, got {g.shape}")
    std = np.ones(G_FULL)
    pc = GROUP_COLS["path"][1:]
    std[pc] = np.maximum(np.sqrt((g[:, pc] ** 2).mean(0)), MIN_STD)
    on_share = {}
    for name in GROUPS[1:]:
        k_known, k_on, kx, ky, kz = GROUP_COLS[name]
        on = g[:, k_on] > 0.5
        on_share[name] = float(on.mean()) if len(on) else 0.0
        for k in (kx, ky, kz):
            std[k] = max(float(np.sqrt((g[on, k] ** 2).mean())), MIN_STD) if on.any() else 1.0
    return dict(mean=[0.0] * G_FULL, std=std.tolist(), turn_max=float(legacy["turn_max"]), speed_cruise=float(legacy["speed_cruise"]), speed_max=float(legacy["speed_max"]),
                layout="full", names=list(FULL_NAMES), on_threshold=float(on_thr), on_share=on_share)


def measured(T, Xc, valid, dt, st) -> np.ndarray:
    """What a DECODED window did, in the prior's layout (one window): the closed loop feeds this back as the context goal, as the prior saw it in training.
    T = the train_cvqvae module (goal_features), Xc = canonicalised windows [1,W,30]."""
    path4 = T.goal_features(Xc, valid, dt)
    return goal_features_full(path4, Xc, valid)[0] if layout_of(st) == "full" else path4[0]


# ------------------------------------------------------------------------------------------------------------------------------------------------ planners
def expand_path(g4, st) -> np.ndarray:
    """[..., 4] path numbers (arc_goal) -> the prior's layout. 'path': unchanged. 'full': the path group known, every channel group unspecified (the engine fills those in)."""
    g4 = np.asarray(g4, np.float32)
    if layout_of(st) == "path":
        return g4
    out = np.zeros(g4.shape[:-1] + (G_FULL,), np.float32)
    out[..., GROUP_COLS["path"][0]] = 1.0
    out[..., GROUP_COLS["path"][1:]] = g4
    return out


def set_channel(vec, st, group: str, on: bool, xyz=None) -> np.ndarray:
    """A copy of `vec` with one channel group given: on=True needs xyz (the point in the window's canonical frame: x left, y up, z forward of where the body stands at the
    start of the window); on=False says 'no target', the engine's procedural layer drives that part. Use unspecify() to say nothing."""
    if layout_of(st) != "full":
        raise ValueError("this prior has no channel targets (train it with --prior-goal-layout full)")
    if group not in CHANNEL_COLS:
        raise ValueError(f"channel group must be one of {list(CHANNEL_COLS)}")
    out = np.array(vec, np.float32, copy=True)
    k_known, k_on, kx, ky, kz = GROUP_COLS[group]
    out[..., k_known] = 1.0
    out[..., k_on] = 1.0 if on else 0.0
    if on:
        if xyz is None or len(xyz) != 3 or not np.all(np.isfinite(xyz)):
            raise ValueError("a target needs three finite numbers (x left, y up, z forward, metres)")
        out[..., [kx, ky, kz]] = np.asarray(xyz, np.float32)
    else:
        out[..., [kx, ky, kz]] = 0.0
    return out


def unspecify(vec, st, group: str) -> np.ndarray:
    """A copy of `vec` with one group left unspecified (known 0, all its numbers 0)."""
    if layout_of(st) != "full":
        raise ValueError("this prior has no groups to leave unspecified")
    out = np.array(vec, np.float32, copy=True)
    out[..., GROUP_COLS[group]] = 0.0
    return out


def describe(st) -> dict:
    """For a UI: the layout, its groups and the states a channel group can be in."""
    if layout_of(st) == "path":
        return dict(layout="path", groups=["path"], names=list(PATH_NAMES))
    return dict(layout="full", groups=list(GROUPS), names=list(FULL_NAMES), channels={g: dict(columns=list(c)) for g, c in CHANNEL_COLS.items()},
                states="known=0 unspecified | known=1,on=0 engine procedural layer | known=1,on=1 target (x left, y up, z forward; metres, canonical window frame)",
                on_share=st.get("on_share") if isinstance(st, dict) else None)


# ------------------------------------------------------------------------------------------------------------------------------------------------ masking (partial control)
def sample_keep(rng: np.random.Generator, B: int, L: int, p_win: float, p_seq: float) -> np.ndarray:
    """[B, L, 5] bool, True = the group stays known. A group disappears for ONE window with probability p_win and for a WHOLE sequence with probability p_seq."""
    drop = rng.random((B, L, len(GROUPS))) < p_win
    drop |= rng.random((B, 1, len(GROUPS))) < p_seq
    return ~drop


def apply_keep(g: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """g [B, L, 25] (normalised), keep [B, L, 5] -> g with every dropped group's columns (known flag included) set to 0."""
    return g * (keep.astype(g.dtype) @ GROUP_MATRIX.astype(g.dtype))


def mask_groups_torch(g, p_win: float, p_seq: float):
    """The training-time version of sample_keep + apply_keep on a torch tensor g [B, L, 25]."""
    import torch
    B, L, _ = g.shape
    drop = torch.rand(B, L, len(GROUPS), device=g.device) < p_win
    drop = drop | (torch.rand(B, 1, len(GROUPS), device=g.device) < p_seq)
    return g * ((~drop).to(g.dtype) @ torch.as_tensor(GROUP_MATRIX, dtype=g.dtype, device=g.device))
