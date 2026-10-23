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
| Prior (`prior_best.pt`) | numpy, in `server.py` | inference twin of `CodePrior` in `tools/train_cvqvae.py` (KV cache). **NEON has no prior runtime yet**, so this part is not on NEON |
| Stitching | `server.py` | incremental version of `train_cvqvae.stitch` (same canonicalisation, yaw branch and cross-fade) |
| Body | libnebula | each frame is tracked through the engine (`nebula_body_track`); if libnebula is missing the browser draws a preview skeleton from the 30 action channels instead |

The NEON checkout is looked for next to this repo (`../NANITY-NEON`), in `NEON_DIR`, `~/NANITY-NEON`; if absent it is cloned into `UI/.cache/` (`--no-clone` to prevent that).
A NEON binary is found via `--neon-bin`, `NEON_BIN`, `PATH` or the checkout; without one, `cvqvae_cli` is compiled for you.

## What you see

- **Viewport**: the body (left side crimson, right side white), planted feet (green), centre of mass (pink), pelvis trail, red frame border when the engine had to correct a frame. Drag to orbit, wheel to zoom, space to pause.
- **Code grid** (bottom): one row per 64-frame window, one cell per VAE code, filled the moment the prior samples it. Brightness is the prior's probability for the code it picked. Timings per window: prior / NEON decode / engine.
- **HUD**: seconds of motion buffered ahead of the playhead and generation speed as a multiple of real time. Below 1x the playback will stall and wait for the model.
- **Controls**: action (or unconditional), temperature, top-k, top-p, seed, number of windows or endless.

## Notes

- Files are checked against each other: a prior with a different codebook size, code count or action list than the VAE is refused with a message.
- `.pt` files are read with `torch.load(weights_only=True)`. If yours needs full unpickling and you made it yourself, add `--trust-pickle`.
- Generated motion is not graded or verified by the engine's grader here; the red border only reflects the engine's per-frame correction report.
- Uploaded files and exported GGUFs live in `UI/.cache/` (git-ignored).
