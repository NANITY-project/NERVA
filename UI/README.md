# NERVA UI

Drag a trained cVQ-VAE and its prior into the browser and watch NERVA motion being generated, decoded and played through the IK engine in real time.

```bash
pip install torch gguf numpy        # torch only reads the .pt files (CPU build is enough)
make ui                             # builds libnebula, starts http://127.0.0.1:8770
# or: python3 UI/server.py [--port N] [--no-open] [--neon-dir DIR] [--neon-bin FILE] [--no-engine] [--trust-pickle]
```

Then drop `runs/cvqvae/best.pt` and `runs/cvqvae/prior_best.pt` anywhere on the page (together or one by one), pick an action, press **Generate**.
A ready `nanity-cvqvae` `.gguf` can be dropped instead of `best.pt`.

## What runs where

| Piece | Runs on | How |
|---|---|---|
| VAE (`best.pt`) | **NEON** | exported by NEON's own `conversion/nanity_export_cvqvae.py`, decoded with `neon --motion-model X.gguf --interactive` (`cvq_decode`); falls back to NEON's `cvqvae_cli` |
| Prior (`prior_best.pt`) | **NEON**, else numpy | exported with NEON's `nanity_export_cvqprior.py` and sampled by `cvq_prior_reset` / `cvq_prior_next`. Only if that export or op is missing does `server.py` use its numpy twin of `CodePrior` (KV cache). The `Prior` chip in the header shows which one is running |
| Stitching | `server.py` | incremental version of `train_cvqvae.stitch` (same canonicalisation, yaw branch and cross-fade) |
| Body | libnebula | each frame is tracked through the engine (`nebula_body_track`); if libnebula is missing the browser draws a preview skeleton from the 30 action channels instead |

The NEON checkout is looked for next to this repo (`../NANITY-NEON`), in `NEON_DIR`, `~/NANITY-NEON`; if absent it is cloned into `UI/.cache/` (`--no-clone` to prevent that).
A NEON binary is found via `--neon-bin`, `NEON_BIN`, `PATH` or the checkout; without one, `cvqvae_cli` is compiled for you.

## What you see

- **Viewport**: the body (left side crimson, right side white), planted feet (green), centre of mass (pink), pelvis trail, red frame border when the engine had to correct a frame. Drag to orbit, wheel to zoom, space to pause.
- **Code grid** (bottom): one row per 64-frame window, one cell per VAE code, filled the moment the prior samples it. Brightness is the prior's probability for the code it picked. Timings per window: prior / NEON decode / engine.
- **HUD**: seconds of motion buffered ahead of the playhead and generation speed as a multiple of real time. Below 1x the playback will stall and wait for the model.
- **Controls**: a prompt and its schedule, action (or unconditional), temperature, top-k, top-p, seed, number of windows or endless.

## Prompts (language layer)

The prior is conditioned on an **action label**, not on text, so a prompt is turned into a *schedule* of labels first: `walk 4 s → pickup 3 s → gesture 2 s`. The language model's only job is to pick labels and durations from the action list of the VAE you loaded (read at run time, nothing is hard-coded), which is small enough that a small model can do it.

1. Type a request in **Prompt**, press **Plan** (Ctrl+Enter). The steps appear as an editable list (change an action, change the seconds, remove a step); the page says whether the plan came from the model or from the fallback.
2. **Generate** plays the schedule. At each step the prior is reset to that action and the body continues from the pose the last step ended in (`Stitcher.begin_segment`: the old clip is finished, the new one starts at the last pelvis pose and eases its limbs over 8 frames). `clear` goes back to the single action in Sampling.

**The model is a slot.** Any one of:

| How | What |
|---|---|
| Drop a `.gguf` on the page, or `--lm /path/model.gguf` | run by NEON (`neon --model X --interactive`). NEON loads only `general.architecture = "nanity"`; convert other GGUFs with `nanity_convert.py` (the error message gives the commands). The chat template is read from the file: ChatML if it says so, otherwise a plain instruction format |
| `Language model` in the page, or `--lm http://127.0.0.1:8080/v1 [--lm-model M] [--lm-key-env VAR]` | any OpenAI-compatible endpoint: llama.cpp server, Ollama, vLLM, a hosted API. The key is kept in memory only and is never returned by `/api/status` |
| nothing | keyword fallback: the action names found in your text, in order. The page labels it as a fallback |

A model that answers badly gets one retry at temperature 0, then the keyword fallback; the page shows which of the three produced the plan. Names that are not in the VAE's action list are dropped, never guessed.

Limits worth knowing: only the ~24 action labels exist (there is no sit, run, kick, dance or get-up unless your data has them), steering by direction is done with numbers, not words (see **Steering by numbers**), and the prior does not know where the body is, so the first moments of a new step can show the limbs jumping before the engine corrects them (the red frame border).

Tests: `python3 tests/test_lang.py`, `python3 tests/test_ui_schedule.py` and `python3 tests/test_ui_goals.py` (numpy only, no torch / model / NEON needed).

## Steering by numbers (no language)

If the prior was trained with goal numbers (`train_cvqvae.py` without `--no-prior-goal`; `/api/status` then has `prior.goal`), the **Steering** section appears. One row per window of motion (about 2.1 s):

| Column | Meaning |
|---|---|
| turn ° | heading change over the window, **+ left, - right** (the trainer's convention: `arc_goal`) |
| speed m/s, dist m | the same thing two ways, `dist = speed x window seconds`; neither = the data's cruise speed |

The action label stays **NULL** (no label) unless you pick one in Sampling, so the movement is described by the numbers alone. The last row repeats if the run is longer than the list (Endless). **Right, then left** fills a gentle turn right for half the windows and left for the other half (60 % of the prior's own `turn_max`). **Guidance** is classifier-free guidance (1 = off); it only acts on windows that have a goal.

After each window the server measures what the decoded motion really did (`train_cvqvae.goal_features`) and the page lists asked vs got, with a check mark when the window turned the asked way. That is the test for "right, then left" before any text parsing exists. A prior without goals refuses a timeline instead of silently ignoring it.

The numpy prior takes the same goals, style and guidance as NEON's (`Prior.add_window_goal`, `set_cfg_scale`), and `cvq_prior_reset` / `cvq_prior_next` on NEON accept `goal` / `goals`. A NEON build that does not report goal numbers in `info` would ignore them, so a steered run then uses the numpy prior (NEON only decodes).

Tests: `python3 tests/test_ui_goals.py` (numpy only).

## Notes

- Files are checked against each other: a prior with a different codebook size, code count or action list than the VAE is refused with a message.
- `.pt` files are read with `torch.load(weights_only=True)`. If yours needs full unpickling and you made it yourself, add `--trust-pickle`.
- Generated motion is not graded or verified by the engine's grader here; the red border only reflects the engine's per-frame correction report.
- Uploaded files and exported GGUFs live in `UI/.cache/` (git-ignored).

## Steering from text (Goal Translator)

The Steering box has a text field: the sentence becomes one goal row per window (turn in degrees, speed in m/s), and anything the text leaves open becomes an **unspecified** window (no goal; the prior gets `"goal": null`). `--goal-translator FILE.gguf` uses a trained translator (`tools/goal_translator`, see its README); without it the deterministic stage-C bootstrap runs and the page says which one spoke. Errors are shown, never replaced by keywords. Text the goals cannot express (e.g. a wave) is listed as ignored.
