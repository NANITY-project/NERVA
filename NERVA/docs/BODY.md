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
- **Head**: neck+head share a turn (limits 75° yaw, +50/−55° tilt, reported when clamped). Torso lean via the 3-bone spine.
- **Feedback**: every refusal says why with numbers ("step too long (1.24 m, max 0.90 m)"); `status_json()` returns state + events.

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
`set_param`, `add_surface`. C API: `nebula_body_*` in `capi/nebula_capi.h` (ABI version 2).

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
Flat-foot contact only (no heel/toe rolling), contact assumed full if the sole origin is over a surface, hands do not collide,
no props/other bodies, fall = flag (no animation), walking is slow by default, one body per `Body` object.
