# Full-body layer (`include/ik_body.hpp`) — status and handoff

The model supplies COORDINATES; the body supplies the physics. There are no walk/jump/wave animations in it.
A walk is a sequence of `FOOT` coordinates; reusable skills are built from those by whoever drives the body.

## What exists (tested)
- **Skeleton**: 22 bones (LOD 0), 32 (LOD 1, default), 50 (LOD 2): pelvis, 3 spine, neck, head; per arm clavicle, upper arm,
  forearm, hand + fingers (none / 1 bone per finger / full phalanges); per leg thigh, shin, foot, toe. `scale` scales every length.
  Output per bone: head, tail, orientation (Y along the bone, Z = anatomical front).
- **Feet**: plant on surfaces and never slide while planted, never sink into a surface, knee limits, stride/step-height limits.
  Surfaces: `add_surface(x0,x1,z0,z1,y)` (floor is y=0).
- **Balance**: centre of mass from segment masses, support polygon from planted-foot contact points, extrapolated CoM (Hof 2008).
  `assist` = weight shifts over the support foot before a step and the CoM is kept inside the polygon (capture-point speed limit).
  `none` = no help: lifting a foot with the weight centred, or leaning outside the feet, really falls.
- **Falling** = state `fallen` + a reported cause (the pose is frozen; `STAND` resets it). No collapse animation yet.
- **Arms**: clavicle assists reaches; hands reuse the tested engine (min-jerk, straight-ish paths, retarget without a snap).
- **Head**: neck+head share a turn (limits 75° yaw, +50/−55° tilt, reported when clamped), each within its own fitted range. Torso: see **Torso** below.
- **Feedback**: every refusal says why with numbers ("step too long (1.24 m, max 0.90 m)"); `status_json()` returns state + events.

## Torso (`include/ik_torso.hpp`, `include/ik_torso_prior.hpp`)
Spine (3 bones), neck, head and both clavicles are anatomical joints, each with three coordinates (`tools/torso_math.py` is the Python mirror):
**flex** (the hinge, + = forward), **lat** (side bend, + = toward the body's right), **twist** (about the bone, + = left).
1. **Orientation targets** — what a motion capture has. `Body.set_torso_targets(q)` / `track_clip(..., torso=)` / C `nebula_body_*_torso`: seven quaternions per
   frame (spine1 spine2 spine3 neck head clavicle_L clavicle_R) in the pelvis *heading* frame. `tools/bvh_to_clip.py` saves them as `torso [F,7,4]`, read from the
   BVH rotations of CMU's LowerBack / Spine / Spine1 / Neck / Neck1 / Head / shoulder joints (found by hierarchy, not by name). Orientations only, never positions:
   scaled to the engine's legs the CMU trunk is ~0.35 m above the pelvis against the engine's 0.48 m, so positions do not transfer and rotations do.
   The CMU head joint is not calibrated across performers (walking head tilt 3–35° between people, the neck chord says 4–11°), so the head axis is calibrated per
   file against the neck chord; its motion relative to the neck is untouched.
2. **Joint limits** — per joint hinge, side-bend and twist ranges, fitted to real people (`tools/fit_torso_prior.py`, 0.2/99.8 percentile + margin, anatomical caps,
   lying-down clips excluded). A joint asked past its limit stops there, the error goes up the chain, `torso_limited()` says by how much and an event is raised.
   `torso_limit_scale` widens/narrows them (0 = off).
3. **Pose prior** — a Gaussian over the nine spine coordinates fitted to the same data (natural posture = a gentle S-curve of a few degrees per joint, no net lean;
   covariance = how real people share a bend among the joints). Without orientation targets, asked to put the neck base somewhere, the spine takes the pose closest
   to the natural one in that metric instead of bending every bone equally; the chest turn is shared among the three joints that twist most. `torso_prior=0` is the
   old equal-bend FABRIK chain; `torso_stiffness` scales the covariance. While *tracking*, the neck-base target is read as the trunk's lean direction (the capture
   trunk's length is another skeleton's).
`torso_energy()` scores a pose under the prior (Mahalanobis² per coordinate: ≈1 typical of real people, ≫1 contorted).

**Measured** (`tools/validate_torso.py`, 58 CMU clips from 16 performers the fit never saw, 18 actions):
| | before (as shipped) | prior only | orientation targets |
|---|---|---|---|
| mean orientation error of spine1-3, neck, head vs the capture | 22.7° | 15.4° | 0.08° |
| frames with spine/neck/head inside the joint limits | 14 % | 100 % | 100 % |
| pose typicality (the real clips: 0.88) | 5.0 | 1.05 | 0.89 |
| clavicle motion error | 7.4° | 6.7° | 2.3° |
Without orientation targets the neck and head stay ~18–24° off: nothing says where the person looked. Refit: `python tools/fit_torso_prior.py`; check: `python tools/test_torso.py`,
`make body-test`.

## Commands (text, one per line; case-insensitive)
```
HAND  <left|right> x y z | rest      FOOT <left|right> x y z [yaw_deg]     PELVIS x y z | auto
FACE  x z | auto                     LOOK x y z | auto                      TORSO x y z | auto
GRIP  <left|right> 0..1              STAND                                  BALANCE assist|none
SURFACE x0 x1 z0 z1 y
```
World: Y up, metres; at yaw 0 the body faces +Z and its LEFT is +X. `FOOT` x y z = the point on the sole under the ankle;
y at a surface height = a step, higher = a foot held in the air.

## Measured
- 148 C++ checks (`make body-test`), clean under ASan+UBSan; a 40,000-command fuzz run (NaN/inf/1e6 values) → zero bad states.
- Update cost ≈ 4–7 µs per body-frame (32 bones).
- Walking by coordinates (8 alternating `FOOT` steps, assist): 3.4 m at **0.31 m/s**, min balance margin +0.4 cm, 0 foot slide, no fall.
  Faster settings trade stability: `shift_settle_speed` = ∞ and `step_time_scale` 0.7 → 0.78 m/s but margin dips to −5.9 cm.
  Defaults are the safe ones.

## Python
`python/nebula.py` → `nebula.Body(scale, finger_lod, assist)`: `command(line)->(ok,msg)`, `update`, `run`, `status()`, `bones()`,
`set_param`, `add_surface`, `track_clip(actions, torso=None)`. C API: `nebula_body_*` in `capi/nebula_capi.h` (ABI version 3; the torso functions are additive).

## NOT done yet (next steps, in order)
1. **Python tests** for the body binding (the C++ suite covers the physics; the ctypes layer was only smoke-tested).
2. **Jumping / flight phase** (ballistic CoM, take-off limited by leg strength). Right now both feet off the ground is refused.
3. **Faster, more dynamic walking** (step before the CoM is fully over the foot; swinging arms; pelvis rotation).
4. **Fall/get-up animation**, hand-ground/prop contact, collision with props, partial foot support at platform edges.
5. **Skill layer** (the model's own `walk_to_browser`): parametric skill language, library, promotion of successful sequences,
   repetition penalty / variation, fast-forward headless training loop. Needs a success metric per goal.
6. **Desktop overlay** (separate piece): transparent click-through surface (X11 / wlr-layer-shell; GNOME has no layer-shell as far
   as I know), perception from the accessibility tree / a small vision model, a `/dev/uinput` bridge. Surfaces = window tops.
7. **Studio integration**: replace Studio's scripted verbs with this body (Studio still uses the older arm-only chains + scripted actions).
8. A **retargeting path** from the new bones (named like Mixamo/VRM-friendly) to `studio/web/avatar.js` for `.glb` models.

## Known limits
Torso: the clavicle is a direction, not a shoulder girdle (no scapula); the prior is fitted to CMU's 100-odd performers (mostly adults, mostly Western motion types) and to what the CMU skeleton's
rotations can express; neck/head pitch without orientation targets is not predicted; hand reaches do not yet recruit the trunk (a reach assist is the next step).
Flat-foot contact only (no heel/toe rolling), contact assumed full if the sole origin is over a surface, hands do not collide,
no props/other bodies, fall = flag (no animation), walking is slow by default, one body per `Body` object.
