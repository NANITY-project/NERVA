#!/usr/bin/env python3
"""
export_goal_translator.py - write a trained Goal Translator checkpoint as a NANITY v1 GGUF (general.architecture = "nanity", the same tensor names and metadata as every
other NANITY model, training/train_nanity_fixed.py export_gguf), plus a small group of `nanity.goal_translator.*` keys that carry what the runtime needs to use it:

    layout_version  n_turn_bins  n_speed_bins  max_windows  max_input_bytes  vocab_size
    pad bos sep win unspec end turn_base speed_base        (token ids: the layout version above is what makes an upper-body group addable later without breaking old files)
    turn_max speed_cruise speed_max                        (the prior's goal_stats the translator was trained against: bins are only meaningful with them)
    window W stride dt                                      (the prior's timing)
    turn_positive_is_left

There is NO text tokenizer to speak of: the input is raw bytes, the output tokens are the grammar's. tokenizer.ggml.tokens is written (names like <0x41>, <win>, <turn_16>) so
generic GGUF tools can open the file; do not run it through NEON's text --model path, the plan decoder (grammar mask + goal derivation) is runtime/goal/neon_goal_translator.hpp.

  python export_goal_translator.py runs/gt/gt_best.pt goal_translator.gguf [--dtype f32|f16]
  pip install torch gguf numpy
"""
import argparse, sys

import numpy as np

import gt_conventions as C

try:
    import gguf
    import torch
except ImportError:
    sys.exit("pip install torch gguf numpy")

import gt_torch as G


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("ckpt"); ap.add_argument("dst")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32", help="f16 is lossy (opt-in): re-run evaluate_goal_translator.py on the exported behaviour before trusting it")
    a = ap.parse_args()

    model, cfg, ck = G.load_checkpoint(a.ckpt)
    gt = ck["gt"]
    if gt["layout_version"] != C.LAYOUT_VERSION or gt["vocab_size"] != C.VOCAB_SIZE:
        sys.exit("the checkpoint was trained with a different token layout than gt_conventions.py")
    sd = model.state_dict()
    npdt = np.float32 if a.dtype == "f32" else np.float16
    w = gguf.GGUFWriter(a.dst, "nanity", use_temp_file=True)
    for k, v in cfg.to_gguf_metadata().items():
        if k == "general.architecture":
            continue                                               # GGUFWriter(arch=...) already wrote it
        (w.add_string if isinstance(v, str) else w.add_float32 if isinstance(v, float) else w.add_uint32)(k, v if isinstance(v, (str, float)) else int(v))
    w.add_string("general.name", "NANITY Goal Translator")
    u32 = lambda k, v: w.add_uint32("nanity.goal_translator." + k, int(v))
    f32 = lambda k, v: w.add_float32("nanity.goal_translator." + k, float(v))
    u32("layout_version", C.LAYOUT_VERSION); u32("n_turn_bins", C.N_TURN_BINS); u32("n_speed_bins", C.N_SPEED_BINS)
    u32("max_windows", C.MAX_WINDOWS); u32("max_input_bytes", C.MAX_INPUT_BYTES); u32("vocab_size", C.VOCAB_SIZE)
    for name in ("PAD", "BOS", "SEP", "WIN", "UNSPEC", "END"):
        u32(name.lower(), getattr(C, name))
    u32("turn_base", C.TURN_BASE); u32("speed_base", C.SPEED_BASE); u32("turn_positive_is_left", 1 if C.TURN_POSITIVE_IS_LEFT else 0)
    for k in ("turn_max", "speed_cruise", "speed_max"):
        f32(k, gt["stats"][k])
    u32("window_W", gt["timing"]["W"]); u32("window_stride", gt["timing"]["stride"]); f32("window_dt", gt["timing"]["dt"])
    w.add_string("tokenizer.ggml.model", "goal-bytes")
    w.add_array("tokenizer.ggml.tokens", C.vocab_strings())
    w.add_uint32("tokenizer.ggml.bos_token_id", C.BOS); w.add_uint32("tokenizer.ggml.eos_token_id", C.END)

    def put(name, key):
        w.add_tensor(name, sd[key].detach().float().numpy().astype(npdt if sd[key].ndim >= 2 else np.float32))
    put("token_embd.weight", "token_embd.weight"); put("output_norm.weight", "output_norm.weight")
    for i in range(cfg.n_layer):
        p = f"blk.{i}"
        put(f"{p}.attn_norm.weight", f"{p}.attn_norm.weight")
        for n in ("attn_q", "attn_k", "attn_v", "attn_output"):
            put(f"{p}.{n}.weight", f"{p}.attn.{n}.weight")
        put(f"{p}.ffn_norm.weight", f"{p}.ffn_norm.weight")
        for n in ("ffn_gate", "ffn_up", "ffn_down"):
            put(f"{p}.{n}.weight", f"{p}.ffn.{n}.weight")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    npar = sum(p.numel() for p in model.parameters())
    print(f"wrote {a.dst}: {npar / 1e6:.2f} M parameters ({a.dtype}); layout v{C.LAYOUT_VERSION}, {C.N_TURN_BINS} turn bins, {C.N_SPEED_BINS} speed bins, "
          f"turn_max {gt['stats']['turn_max']:.3f} rad, window {gt['timing']['W']}/{gt['timing']['stride']} @ {gt['timing']['dt']:.4f}s")


if __name__ == "__main__":
    main()
