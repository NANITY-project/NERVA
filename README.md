# Nebula IK engine
(NERVA is Nebula... Naming mixed up as its not in release or for use yet.)
> **New:** [`studio/`](studio/README.md) — a local studio where language-model characters (driven by your NEON models) act in a scene through this engine,
> and [`blender/`](blender/) — a Blender add-on that drives a rig with it and bakes keyframes.
> Both use the same shared library (`make lib`) via the C API in [`capi/`](capi/nebula_capi.h).


From-scratch C++17 IK animation engine. Header-only, no dependencies (no glm,
no Eigen). Built to be driven by a model through a one-line text protocol.

```
make            # builds ik_demo (tests), ik_bench, ik_quality
make test       # 37 checks, non-zero exit on failure
make bench      # speed
make sanitize   # tests under ASan + UBSan
NATIVE=0 make   # portable binary instead of -march=native
```

## Layout

| Path | What |
|---|---|
| `include/ik_math.hpp` | `Vec3`, `Quat`, fast `inv_sqrt`, critically-damped `smooth_damp`, cached value noise |
| `include/ik_skeleton.hpp` | `Joint`/`Chain` (fixed capacity, allocation-free) + forward kinematics |
| `include/ik_solver_analytic.hpp` | closed-form 2-bone IK (law of cosines), elbow angle limits |
| `include/ik_solver_fabrik.hpp` | FABRIK for longer chains, with per-joint cone limits |
| `include/ik_engine.hpp` | `IKEngine`: named/ID chains, task-space motion, text protocol |
| `src/main.cpp` | test suite + usage examples |
| `src/bench.cpp`, `src/quality.cpp` | speed benchmark, realism metrics |
| `renderer/` | Vulkan bootstrap (separate; see `renderer/README.md`) |

## How motion works

The hand follows a **minimum-jerk quintic in Cartesian space** (Flash & Hogan
1985) and IK is solved **every frame** against that setpoint. Consequences:

- straight-ish hand paths (plus an optional natural bow, `path_curvature`) with
  a bell-shaped velocity profile, instead of curved joint-space paths;
- retargeting mid-motion keeps position, velocity and acceleration (quintic with
  arbitrary initial conditions) -- no velocity snap, and no second reaction delay;
- movement time follows Fitts' law (`a + b*log2(1 + D/w)`), capped by the
  chain's peak speed (`max_speed_m_per_s`);
- torso lean is a critically-damped spring (it used to be assigned instantly,
  teleporting the hand);
- the elbow pole relaxes back toward its natural direction over time;
- idle hand wander is smooth non-repeating noise; shoulders breathe;
- elbow interior-angle **min and max** limits (exact, via reach distance);
- FABRIK joints take **cone limits** (`set_max_bend`) enforced inside the solver.

All tunables live in `IKEngine::params` (`MotionParams`).

## Text protocol

```
REACH right_arm 0.30 1.20 0.20
POINT right_arm 0.80 1.40 0.20
ORIENT right_arm fx fy fz [ux uy uz]
RESET right_arm
```

`execute_command()` returns `false` for anything malformed -- unknown command or
chain, wrong arg count, non-numeric, trailing junk, **NaN/inf** -- so the caller
can send the model a corrective message. (NaN used to pass straight into the solver.)

## Measured ( x86, g++ 13, `-O3 -march=native`, 1024 chains, best of 7)

Numbers are from a shared 1-core VM, so expect +/-10% run to run.

| | original | now |
|---|---|---|
| `reach_to()` | ~470 ns | ~140 ns |
| idle chains (all settled), per chain-frame | ~136 ns | ~88 ns |
| active chains (constant motion), per chain-frame | ~127 ns | ~145-160 ns |

The active case is slower on purpose: the old engine solved IK once per reach and
slerped joint rotations; this one solves every frame, which is what makes the
paths/constraints/lean correct during the motion. A 20-chain avatar costs a few
microseconds per frame either way.

Realism (`ik_quality`, same arm, same targets):

| metric | original | now |
|---|---|---|
| hand path deviation from a straight line | 29.5% | 4.0% (= configured bow; 0% with `path_curvature = 0`) |
| hand jump inside `reach_to()` (torso lean) | 100 mm | 0 mm |
| velocity jump at mid-motion retarget | 132% of peak | 6% of peak (one frame of normal acceleration) |

## Speed techniques

- `ChainId` handles: no string hashing per call (names only at the API edge / protocol).
- Fixed-capacity chains (`IK_MAX_JOINTS`, default 12): no heap allocation anywhere in
  the frame loop; original `reach_to` copied a `Chain` (strings + vector) per call.
- Hot/cold split: names live in a parallel array.
- 2-bone solver: no `acos`/`cos`/`sin`, no forearm normalisation, cached reach limits.
- SSE reciprocal-sqrt + 1 Newton step (`IK_NO_FAST_RSQRT` for exact; auto-fallback off x86).
- Velocity/acceleration evaluated only on retarget, not per frame.
- Error-bounded solve skipping: settled chains don't re-solve until the setpoint
  drifts more than `solve_epsilon_m` (0.1 mm); forced on the frame a reach completes.
- Cached noise lattice (hashing every frame was costing more than the IK solve).

## API changes from the original

- `Chain`: `joints` is a fixed array + `count` (was `std::vector`); no `name`; `Joint` has no `name`.
  `world_positions()` still returns a vector, plus an allocation-free overload.
- `add_chain` returns a `ChainId` (`kNoChain` on invalid spec: duplicate name, zero-length bone,
  wrong joint count for the solver). Name-based calls still work.
- `RESET` returns the hand to a **rest target** (default: the chain's initial extension;
  `set_rest_target()`), rather than snapping joint rotations to identity.
- `set_end_effector_orientation` duration defaults to "auto from angle".
- `solve_two_bone_ik` keeps its signature (+ optional max elbow angle);
  `solve_two_bone_reach` is the fast path.
