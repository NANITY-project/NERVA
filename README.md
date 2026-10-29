# NERVA

**NERVA is the embodiment layer of [NANITY](https://github.com/NANITY-project): a way for a model to have a body, a pointer and eyes on your desktop.**

It is a from-scratch C++17 stack (no glm, no Eigen) with three parts that work together or on their own:

- **Nebula**, a header-only IK and physics engine for a humanoid body. It is the part that decides what the body can and cannot do.
- **A motion pipeline**: tools to grade motion against real motion capture, build training sets, and train a conditional VQ-VAE and a prior that generate new motion for the body.
- **A desktop overlay**: the body drawn on your Wayland/Hyprland desktop, with an opt-in pointer and accessibility-based screen reading, driven by text actions from a model.

This repository and its code are provided "as is" without any warranty.

> **Status: pre-release.** Built and tested headlessly (fake Hyprland socket, headless Sway, Mesa lavapipe). The real `/dev/uinput` pointer path is not yet verified; see [overlay/README.md](overlay/README.md#known-limits).

## How the pieces fit

```
 text actions                                   generated motion
 ------------                                   ----------------
 NEON (C++ runtime)  --JSON lines-->  nerva_bridge.py  -->  nerva-overlay      cVQ-VAE + prior  -->  Nebula engine  -->  body
 runs the model                       one turn at a time    body, pointer, a11y  (tools/train_cvqvae.py)  (tracks every frame,
                                                                                                          enforces physics)
```

The engine never plays canned animations. A walk is a sequence of foot coordinates, a reach is a hand coordinate, and the body supplies the physics (balance, reach limits, planted feet, falling). Whatever produces motion, a model's text actions or a generated clip, is bounded by the same engine.

The UI has a language layer ([UI/README.md](UI/README.md#prompts-language-layer)): any model, loaded from a `.gguf` or an OpenAI-compatible endpoint, turns a prompt into a schedule of action labels. Not in this repo yet: a translator from intent or emotion to the prior's goal numbers (direction, speed, style), and the loop that retrains the generator on clips the engine and grader pass.

## Layout

| Path | What | Docs |
|---|---|---|
| `include/`, `src/` | **Nebula**: header-only IK engine and full-body layer (feet, balance, arms, head, torso, fingers), plus a demo, benchmark and quality tool | [docs/ENGINE.md](docs/ENGINE.md), [docs/BODY.md](docs/BODY.md) |
| `capi/` | C API that builds `libnebula.so`. Used by the UI, the dataset tools and the Blender add-on | [capi/nebula_capi.h](capi/nebula_capi.h) |
| `tools/` | Motion grader, real-data loaders, dataset builder, **cVQ-VAE + prior trainer**, clip viewer, and the Python wrapper for the C API (`tools/nebula.py`) | [docs/MOTION_GRADER.md](docs/MOTION_GRADER.md) |
| `tests/` | Engine and body tests, the motion-toolchain test, and the motion-header test | |
| `UI/` | Drag-and-drop console: drop a trained cVQ-VAE and prior, watch motion generated live and played through the engine | [UI/README.md](UI/README.md) |
| `overlay/` | Wayland / Hyprland overlay: the body on your desktop, an opt-in AI pointer, accessibility reading. Never touches the keyboard | [overlay/README.md](overlay/README.md) |
| `nerva_bridge.py` | Drives the overlay with a model hosted by NEON, one turn at a time, and records episodes | [docs/ACTIONS.md](docs/ACTIONS.md) |
| `blender/` | Blender add-on: drive a rig with the engine and bake keyframes (`blender/build_addon.sh`) | |
| `renderer/` | Vulkan bootstrap for the embodiment renderer, separate from NEON | [renderer/README.md](renderer/README.md) |
| `docs/` | Engine and body reference, the action language, the motion grader, data credits | [docs/CREDITS.md](docs/CREDITS.md) |

## Quick start

You need a C++17 compiler (`g++` or `clang++`) and `make`. Python 3 with `numpy` covers the tools. `torch` is only needed to train or to load `.pt` files in the UI.

```bash
make                 # builds the demo, benchmark, quality tool, body test and build/libnebula.so
make test            # engine checks
make body-test       # full-body checks
make sanitize        # engine and body tests under ASan + UBSan
make actions-test    # the action-language docs and examples stay in sync with the code
make ui              # builds the library and opens the VAE/prior console at http://127.0.0.1:8770
make overlay         # needs g++, wayland, wayland-protocols
make test-all        # everything above plus the overlay and Blender add-on tests (needs sway, blender, etc.)
```

CMake builds the same engine targets and the shared library: `cmake -S . -B build && cmake --build build`.

## The engine interface

The body is controlled through three fixed-size vectors, so a learned model can drive it directly:

- **Action**, 30 floats per frame: pelvis xyz yaw, left foot xyz yaw contact, right foot xyz yaw contact, left hand xyz on, right hand xyz on, look xyz on, torso xyz on.
- **State**, 35 floats, in world metres and radians. Names come from `nebula_body_state_name`.
- **Report**, 12 floats per frame: how much the engine had to correct (foot slide, sunk feet, unreachable legs, balance and so on).

On top of those the C API offers per-frame tracking (`nebula_body_track`, `track_clip`, `replay_clip`), clip judging from the body's current state (`evaluate_clip`), and snapshots (`clone` / `restore`) for trying candidate motions from the same start. The text protocol (`HAND`, `FOOT`, `JUMP`, `STAND`, `BALANCE`, ...) is documented in [docs/BODY.md](docs/BODY.md).

## Motion pipeline

Real motion capture is used to grade motion, and the model is trained on data the engine has produced and the grader has accepted. Data sources and licences are listed in [docs/CREDITS.md](docs/CREDITS.md); no original motion-capture files are in this repository.

```bash
make motion-test-quick                                 # test the grading toolchain on a small set (downloads data the first time)
make vae-data HOURS=10                                 # build the training set into data/vae
python3 tools/train_cvqvae.py --data data/vae --out runs/cvqvae
python3 tools/view_clip.py runs/cvqvae/gen_preview.npz --open
```

`tools/gen_actions.py` makes engine clips of non-walking actions (stand, turn, balance, crouch, pickup, wave/point, stretch, drink, jump) from scripts of the engine's own commands; `build_vae_dataset.py --engine-skills N` keeps only the ones the engine accepts and the grader calls that action. Jump is verified live, not by replay; there is no sit, run, kick, dance or get-up.

`train_cvqvae.py` trains the VAE, checks the codebook, tokenizes the data, trains the prior and samples new clips. `--stage` runs a single stage and `--vae-ckpt` reuses a VAE you already trained. `--params-only` prints the parameter counts. Generated clips are stitched into longer ones and can be graded with `--grade`. The engine does not verify them in that script; the UI shows the engine's per-frame corrections when you play one.

## Running a model against the overlay

```bash
python3 nerva_bridge.py \
  --neon /path/to/neon --model /path/to/model.gguf \
  --overlay-bin overlay/build/nerva-overlay \
  --fewshot overlay/tools/examples/episodes.jsonl \
  --task "Click the 'Save changes' button." --verbose
```

- `--probe` sends five known prompts and prints the share of garbage characters per case. Healthy output is plain English, about 0% junk. It separates a temperature problem, a prompt-format problem, a BOS problem and a broken `--interactive` path.
- `--mock-session` replaces the overlay with a small stand-in, so you can test the model loop without Wayland.
- `--no-bos-im-start` is for tokenizers whose BOS is not `<|im_start|>` (SmolLM2's is, which is the default).
- The bridge sends a raw ChatML `prompt`, not NEON's `messages` form, because NEON's `messages` template is not ChatML.
- Episodes are written to `rollouts.jsonl` in the training format of [docs/ACTIONS.md](docs/ACTIONS.md#7-training-data-episodes). Lint them with `python3 overlay/tools/nerva_actions.py lint rollouts.jsonl`.
- The pointer is `dry-run` by default (`--session-args "--a11y --mouse --pointer dry-run"`). Real clicks need `--pointer` set to the real backend **and** a window you granted with `nerva-overlay --ctl grant`. The model cannot grant.

## Safety model

Everything the model can do is the action list in [docs/ACTIONS.md](docs/ACTIONS.md). It cannot type, press keys, drag, grant itself access, or read hidden apps, password fields or authentication prompts. The pointer is off by default, only works inside one window a human granted, only while that window is focused, expires after `--grant-ttl`, and pauses when you touch the mouse. Invalid turns are refused before they reach the overlay.

## License

[GNU AGPL-3.0](LICENSE). The Blender add-on's manifest declares `GPL-3.0-or-later` because Blender requires a GPL-compatible SPDX id; AGPL-3.0 code may be combined with GPL-3.0 (AGPL section 13). Motion data used by the tools has its own licences, listed in [docs/CREDITS.md](docs/CREDITS.md).
