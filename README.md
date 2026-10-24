# NERVA

**NERVA is the embodiment layer of [NANITY](https://github.com/NANITY-project): a language model gets a body, a pointer and eyes.**
This repository and its code are provided 'as is' without any warranty.
It is a from-scratch C++17 stack (no glm, no Eigen) where a model writes short text actions and a humanoid carries them out on your desktop,
using its hand as the mouse and reading the screen through accessibility.

> **Status: pre-release.** Built and tested headlessly (fake Hyprland socket, headless Sway, Mesa lavapipe). The real `/dev/uinput` pointer
> path and real Hyprland are **not yet verified**; see [overlay/README.md](overlay/README.md#known-limits). Names: **NERVA** is the whole system;
> **Nebula** is the IK / body engine inside it (`libnebula`, `nebula_*`, "Nebula Studio").

```
 NEON (C++ inference runtime, --interactive)  <--JSON lines-->  nerva_bridge.py  <--Session-->  nerva-overlay
        runs the model (a NECTAR / any nanity GGUF)             one turn at a time               body + pointer + a11y
```

## Pieces

| Path | What | Docs |
|---|---|---|
| `include/`, `src/` | **Nebula** header-only IK engine and full-body layer (22/32/50-bone humanoid, feet, balance, arms, head) | [docs/ENGINE.md](docs/ENGINE.md), [docs/BODY.md](docs/BODY.md) |
| `capi/`, `python/` | C API (`libnebula.so`) and Python wrapper used by the host apps | [capi/nebula_capi.h](capi/nebula_capi.h) |
| `overlay/` | Wayland / Hyprland overlay: the body on your desktop, an opt-in AI pointer, accessibility reading. Never touches the keyboard | [overlay/README.md](overlay/README.md) |
| `docs/ACTIONS.md` | **nerva-actions v1**, the language the model writes (`<click>838 351</click>`, `<find>save</find>`, ...) | [docs/ACTIONS.md](docs/ACTIONS.md) |
| `nerva_bridge.py` | Drives the overlay with a model hosted by NEON, one turn at a time; records episodes | below |
| `studio/` | Local studio: language-model characters act in a 3D scene through the engine | [studio/README.md](studio/README.md) |
| `UI/` | Drag-and-drop console: drop a trained cVQ-VAE + prior (`.pt`), watch motion generated live (VAE on NEON, body through the engine) | [UI/README.md](UI/README.md) |
| `blender/` | Blender add-on: drive a rig with the engine and bake keyframes | `blender/build_addon.sh` |
| `renderer/` | Vulkan bootstrap for the embodiment renderer (separate from NEON) | [renderer/README.md](renderer/README.md) |

## Quick start

```bash
make                 # engine tests, bench, quality, body test, build/libnebula.so
make test            # 37 engine checks
make body-test       # 148 full-body checks
make actions-test    # action-language docs/examples stay in sync with the code
make sanitize        # engine + body tests under ASan + UBSan
make studio          # build the lib and open Nebula Studio at http://127.0.0.1:8765
make ui              # build the lib and open the VAE/prior console at http://127.0.0.1:8770

make overlay         # needs g++, wayland, wayland-protocols
make overlay-test    # unit + integration checks (the Sway / AT-SPI ones also need sway, grim, dbus, python3-gobject, at-spi2-core)
make test-all        # everything above plus Studio (incl. node UI tests) and the Blender add-on (needs `blender`)
```

CMake builds the same engine targets and the shared library: `cmake -S . -B build && cmake --build build`.

## Running a model against the overlay

```bash
python3 nerva_bridge.py \
  --neon /path/to/neon --model /path/to/model.gguf \
  --overlay-bin overlay/build/nerva-overlay \
  --fewshot overlay/tools/examples/episodes.jsonl \
  --task "Click the 'Save changes' button." --verbose
```

- `--probe` sends five known prompts and prints the share of garbage characters per case (separates a temperature problem, a prompt-format problem, a BOS problem and a broken `--interactive` path). Healthy output is plain English, about 0% junk.
- `--mock-session` replaces the overlay with a tiny stand-in, so you can test the model loop with no Wayland.
- `--no-bos-im-start` if your tokenizer's BOS is **not** `<|im_start|>` (SmolLM2's is, which is the default).
- The bridge sends a raw ChatML `prompt`, not NEON's `messages` form: NEON's `messages` template is `<|system|>...<|end|>`, not ChatML.
- Token bytes are joined before decoding, so a UTF-8 character split across two byte-level BPE tokens survives.
- Episodes are written to `rollouts.jsonl` in the training format of [docs/ACTIONS.md](docs/ACTIONS.md#7-training-data-episodes); lint them with `python3 overlay/tools/nerva_actions.py lint rollouts.jsonl`.
- The pointer is `dry-run` by default (`--session-args "--a11y --mouse --pointer dry-run"`). Real clicks need `--pointer` set to the real backend **and** a window granted by you with `nerva-overlay --ctl grant`; the model cannot grant.

## Safety model, in one paragraph

Everything the model can do is the action list in [docs/ACTIONS.md](docs/ACTIONS.md). It cannot type, press keys, drag, grant itself access, or
read hidden apps, password fields, or authentication prompts. The pointer is off by default, only works inside one window a human granted, only while that
window is focused, expires after `--grant-ttl`, and pauses when you touch the mouse. Invalid turns are refused before they reach the overlay.

## License

[GNU AGPL-3.0](LICENSE). `studio/vendor/three/` is three.js (MIT, its own `LICENSE`). The Blender add-on's manifest declares
`GPL-3.0-or-later` because Blender requires a GPL-compatible SPDX id; AGPL-3.0 code may be combined with GPL-3.0 (AGPL section 13).
