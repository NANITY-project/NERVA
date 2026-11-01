#!/usr/bin/env python3
"""
export_e5_translator.py - checkpoint (train_e5_translator.py) -> one .npz file that e5_numpy.py / UI/goal_plan.py run with numpy only.

  python export_e5_translator.py runs/e5gt/e5gt_best.pt e5gt.npz            # float32 (about 135 MB for E5-small)
  python export_e5_translator.py runs/e5gt/e5gt_best.pt e5gt.npz --fp16     # 2-D matrices as float16 (about 70 MB); biases and LayerNorms stay float32, widened on load

The file holds the weights, the WordPiece vocabulary, the prior's goal stats and timing, and a license note. It is self-contained: no download at inference.
After writing it, the file is loaded back and one prompt is translated, so a missing or misnamed tensor fails here and not in the UI.
"""
import argparse, json, os, sys

import numpy as np
import torch

import gt_conventions as C
import e5_common as E
import e5_torch as L
import e5_numpy as N


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    model, ck = L.load_checkpoint(a.ckpt, "cpu")
    st, tm = C.Stats.from_dict(ck["stats"]), C.Timing(**ck["timing"])
    weights, meta, vocab = L.export_arrays(model, ck["vocab"], st, tm, ck.get("encoder_name", ""), a.fp16, int(ck.get("step", 0)))
    np.savez(a.out, meta_json=np.array(json.dumps(meta)), vocab_txt=np.array("\n".join(vocab)), **weights)
    mb = os.path.getsize(a.out if a.out.endswith(".npz") else a.out + ".npz") / 1e6
    print(f"wrote {a.out}: {len(weights)} tensors, {mb:.0f} MB, step {meta['step']}, encoder {meta['encoder_name'] or '?'}, head {meta['head']}")

    path = a.out if a.out.endswith(".npz") else a.out + ".npz"
    try:
        t = N.E5NumpyTranslator.from_npz(path)
        plan = t.translate("walk right and then left")
    except Exception as e:
        sys.exit(f"the exported file does not load or run: {type(e).__name__}: {e}")
    print("smoke test: 'walk right and then left' ->", " ".join(f"({w.turn_bin},{w.speed_bin})" if not w.unspecified else "U" for w in plan.windows[:12]), "..." if len(plan) > 12 else "")
    print("keep the license of the encoder named above with this file (see license_note in the metadata).")


if __name__ == "__main__":
    main()
