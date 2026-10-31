# NANITY Goal Translator (spec draft 0.1)

Free text -> one goal per motion window `(turn, speed | unspecified)`, in the numbers the cVQ-VAE prior is conditioned on. A tiny byte-level NANITY model; no action vocabulary.

| file | what |
|---|---|
| `gt_conventions.py` | the ONE place for token layout, bins, grammar, left/right sign, `arc_goal`, plan<->tokens<->goals |
| `stage_c.py` | bootstrap translator, same interface, no model (word splitter + small schema; never guesses; reports what it ignored) |
| `build_dataset.py` | measure -> segment -> describe -> rewrite -> verify -> UNSPEC; split by source clip |
| `gt_corrupt.py`, `gt_torch.py` | on-the-fly corruption; model presets, dataset, grammar-masked decoding, checkpoints |
| `train_goal_translator.py` | training (loss on output tokens only), periodic held-out metrics, `gt_best.pt` / `gt_last.pt`, `--resume` |
| `evaluate_goal_translator.py`, `gt_metrics.py`, `prompt_suite.json` | held-out metrics, fixed prompt suite (easy/hard tiers), promotion rule vs stage C |
| `export_goal_translator.py` | checkpoint -> NANITY GGUF (+ `nanity.goal_translator.*` keys) |
| `gt_numpy.py` | torch-free reference runtime (used by NERVA UI) |
| `runtime/` | C++ runtime (`neon_goal_translator.hpp`, `goal_translator_cli.cpp`), built against a NANITY-NEON checkout |

## Run (from the NERVA repo root; needs a NANITY-NEON checkout next to it or NEON_DIR for `modeling_nanity.py`)
```bash
pip install torch gguf numpy
python tools/gen_dataset.py ...                       # as you already do: gives clips.npz
python tools/goal_translator/build_dataset.py --clips runs/data/clips.npz --prior runs/cvqvae/prior_best.pt --vae runs/cvqvae/best.pt --out data/gt
#   goal_stats, stride, dt come from the prior, W from the VAE. No clips yet? --synthetic 3000 (teaches text->goals only, no motion realism)
python tools/goal_translator/train_goal_translator.py --data data/gt --out runs/gt --preset small      # tiny | small (~5M) | base (~14M)
python tools/goal_translator/evaluate_goal_translator.py --data data/gt --ckpt runs/gt/gt_best.pt      # prints PROMOTE: yes/no vs stage C
python tools/goal_translator/export_goal_translator.py runs/gt/gt_best.pt runs/gt/goal_translator.gguf
python UI/server.py --goal-translator runs/gt/goal_translator.gguf       # without it, stage C translates and the page says so
python tests/test_goal_translator.py && python tests/test_goal_plan.py && python tests/test_ui_goal_timeline.py
# C++ (optional): g++ -std=c++20 -O2 -pthread -mavx2 -mfma -I $NEON/runtime tools/goal_translator/runtime/goal_translator_cli.cpp -o goal_translator ; python tests/test_goal_translator_neon.py --neon $NEON
```

## Things to know
* **Rewrite step (spec 4):** `--rewriter rules` is a built-in, limited-diversity bootstrap. For real diversity run `--stage requests`, put an LM over `rewrite_requests.jsonl` offline, then `--rewrites FILE`. License of that LM follows the sentences (spec section 8).
* **Verifier** rejects rewrites stage C cannot read back (unknown slang): safe, costs diversity.
* **Sign:** turn > 0 = the character's LEFT (matches `tools/train_cvqvae.py`; parity is tested in `tests/test_goal_plan.py`).
* **Timing:** `arc_goal` path = speed * (W-1)*dt; "N seconds" -> windows of stride*dt.
* **Bug in NANITY-NEON `training/modeling_nanity.py` (not changed here):** a cached single-token step uses SDPA `is_causal=True`, which is top-left aligned, so the query sees only key 0 (logits off by ~13). `generate()` with KV cache is affected; training is not. Fix: in `NanityAttention`, `is_causal = attn_mask is None and past_kv is None`. `gt_torch.decode_plan` passes an explicit mask so it is correct either way.
* **C++ vs reference:** logits differ ~0.25-1.3% of scale (int8 V cache explains only a small part; rest not isolated); plans agree except near-ties.
* **Not done:** the C++ translator is not wired into NEON's `NEON-3.cpp` / `neon_multimodal.hpp` (the header has a `handle(json)` ready for it); upper body (the wave) is a later piece; a trained model's quality is untested here (only overfit/pipeline checks, CPU-tiny run still undertrained).
