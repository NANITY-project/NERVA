# Nebula Studio

A local studio for scenes where every character is driven by a language model: they perceive the scene, decide, and
act through the Nebula IK engine. Direct them with plain text, record takes, import an environment from Blender or
Unreal (via glTF), and give characters real 3D bodies.

```
make lib                        # builds build/libnebula.so (needs g++ and make)
python3 studio/server.py        # opens http://127.0.0.1:8765   (--no-open to skip the browser, --port N to change it)
```

The stage starts **empty**. Open **Guide** (top bar) for a walkthrough and a one-click demo scene (Alice, Bob, a table,
a cup...). Characters start as **blacked-out entities**; select one and choose a `.glb` to give it a body.

## The brain: NEON first

Click **Model**. Studio looks for the NEON runtime on this computer and, if it finds it, lists the models it can run.

- **Finds NEON** by `NEON_BIN`, `neon` on `PATH`, and common locations (next to Studio, `~/NEON*`, `~/.local/bin`...). A
  binary is only accepted if it prints NEON's usage text, so an unrelated `neon` tool is never run.
- **Builds it for you** if only the source (`NEON-3.cpp`) is found: one button runs the README's compile line
  (`-O2 -mavx2 -mfma` when your CPU has them) and streams the compiler output.
- **Lists your models** by scanning `./models`, `~/models`, `~/.cache/nanity`, `~/Downloads`, folders you add, and
  `NEBULA_MODEL_DIRS`. Each `.gguf` header is read: only `general.architecture = nanity` (spec v1) models can be
  selected; others show *why* (e.g. "architecture 'llama': convert it first with nanity_convert.py"). `.nctr` is
  listed but marked unsupported, as NEON itself doesn't load it yet.
- **Check model** runs NEON's own `--probe`; **Launch** starts `neon --model X --interactive`, shows NEON's log, the
  context length, request count and last reply time, and **Stop** ends it.
- Settings (binary path, extra folders, thread count, last model) persist in `~/.config/nebula-studio/settings.json`.

Precedence: a running NEON model, else the external endpoint, else the built-in rule-based agent. The top-bar pill always
says which one is driving. Command line equivalents: `--neon-bin PATH`, `--neon-model FILE.gguf`, `--model-dir DIR`.

How Studio talks to NEON (read from `NEON-3.cpp`): one subprocess, JSON per line on stdin
(`{"messages":[...],"max_tokens":N,"temperature":T,"top_p":P}`), `{"token":...}` lines back, then `{"done":true}`;
ready when stderr prints `[Mode] Interactive stdin/stdout loop — ready.` NEON serves one request at a time, so
characters thinking together are queued. Prompts are budgeted against the model's real context length (oldest history
dropped first; if even the minimum doesn't fit, the transcript says so and names the numbers). NEON takes no grammar, so
that option is ignored for it.

**External endpoint** (second choice; llama.cpp, vLLM...): any OpenAI-compatible `/v1/chat/completions`, with an optional
GBNF grammar listing only names that currently exist.

What makes small models workable: a tolerant parser (bullets, fences, chatty prose are ignored); rejected commands are
explained back in the model's *next* prompt; `GRAB cup` walks over and crouches by itself so the model never plans movement.
Commands: `SAY WALK_TO LOOK_AT GRAB OFFER DROP REACH POINT WAVE WAIT RESET`.

## Character bodies

- **Default**: a blacked-out entity (near-black silhouette, the character's colour only on accents).
- **Any `.glb`** (select a character, **Choose .glb**, up to 30 MB). Studio looks for a humanoid skeleton by bone name
  (Mixamo, VRM/VRoid, Rigify and generic `UpperArm_L`-style names). If found, the arms and legs are posed with two-bone IK using
  *the model's own bone lengths*, so hands still land on their targets; feet stay planted, hips drop when crouching.
  No skeleton? It becomes a static figure carried by the character (the engine's arms stay visible so actions read).
- The body label in the inspector shows `file.glb · rigged|static`. Models face **+Z** (the glTF convention), stand on y=0 and
  are scaled to 1.75 m. Head turning and torso lean are not animated on models yet. Models are held in memory (not saved to disk).

## Using it

| | |
|---|---|
| **Direct bar** | free-text instruction to one character or everyone |
| **Autonomous** | per-character checkbox: they act on their own every few seconds when idle |
| **Command box** | type a command (Tab completes, ↑ recalls) |
| **Viewport** | click select · shift+click ground = walk · alt+click prop = grab · drag = orbit |
| **Import environment** | button or drop a `.glb`: meshes are shown and each object becomes a prop (bounding box). Floors/walls/lights are skipped; Blender custom property `nebula_tags` becomes tags |
| **Rec / Play** | record a take (≤120 s), scrub it, export JSON of every joint position per frame |

## Layout

| File | Role |
|---|---|
| `sim.py` | characters, props, actions, takes (single-threaded, deterministic) |
| `agent.py` | prompts, reply parsing, grammar, external client, per-character think loop |
| `neon_bridge.py` | find/build NEON, read GGUF headers, probe, run + queue requests |
| `gltf_import.py` | props from a glTF/GLB (JSON only) |
| `server.py` | stdlib HTTP + SSE; localhost only; Host-header and JSON-only POST checks |
| `web/` | `app.js` (DOM/network), `viewport.js` (three.js), `avatar.js` (model binding), `logic.js` (pure) |
| `vendor/three/` | three.js r170 (MIT), vendored so the UI works offline (except Google Fonts) |

## Tests (`make studio-test`)

15 simulation · 10 agent · 6 glTF (one against a real Blender export) · 22 NEON bridge · 13 server/API · 10 scene-graph/logic ·
1 failure-note · 10 avatar (a real Blender-exported **skinned** GLB through three's GLTFLoader) · 18 full-stack UI scenarios
(the real server + the real UI code in jsdom, including an XSS attempt).

The NEON tests drive the **real NEON binary** when one can be found (otherwise they skip): they generate a tiny random-weights
NANITY-spec GGUF (`tests/make_tiny_gguf.py`; it produces gibberish but is a valid model), probe it, run it, crash it, hang it,
and build NEON from source.

## Known limits (read these)

- **The UI has never been rendered in a real browser.** The environment it was built in has none. Scene-graph math (every limb
  ends exactly on the engine's joint positions; model bones land on their targets), all DOM/network behaviour and the failure paths
  are tested; how it *looks*, camera feel and WebGL quirks are not. If the viewport can't start, the note now says why.
- **NEON was tested with a random-weights toy model, not a trained NANITY model.** The protocol, discovery, build, probe, context
  budgeting and error handling are real; how well your fine-tuned 1.7B follows this command format is untested. Expect to tune prompts.
- **Locomotion is a slide with procedural legs**: no pathfinding, collision or navmesh; props are boxes.
- Environments give agents bounding boxes, not meshes: no rooms, doors or stairs yet.
- Removing a character leaves its (tiny) engine chains allocated until restart.
- Takes are joint positions only; a Blender importer for them is not written yet.
