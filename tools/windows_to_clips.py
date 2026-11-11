#!/usr/bin/env python3
"""
windows_to_clips.py - let tools/goal_translator/describe_motion.py caption the windows the prior is trained on.

describe_motion.py writes free-form captions (text, clip, frame span) for the clips of a clips.npz and reads no labels. The prior trains on WINDOWS (64 frames), and train_windows.npz
does not say which clip or which start frame a window came from. So every window becomes one tiny clip: clip k of the output = window ids[k]. The captions then join back exactly.
The windows are raw clip frames (same 30 channels), so describe_motion measures them as it measures any clip.

  python tools/windows_to_clips.py --data nerva-motion-data/vae --out OUT/wc --every 6          # every 6th train window (a pilot does not need all 21 793), all val windows
  python tools/goal_translator/describe_motion.py --stage requests --clips OUT/wc/window_clips.npz --out OUT/wc/captions --spans-per-clip 1
  python tools/goal_translator/describe_motion.py --stage run     --out OUT/wc/captions        # a local model via ollama, or feed requests.jsonl to any LM -> responses.jsonl
  python tools/goal_translator/describe_motion.py --stage apply   --out OUT/wc/captions        # -> OUT/wc/captions/captions.jsonl  (left/right check included)
  python tools/text_prior_pilot.py ... --captions OUT/wc/captions/captions.jsonl --clip-map OUT/wc/window_map.npz

--spans-per-clip 1 makes one request per window that covers the whole window. Windows that get no caption are trained with the NULL vector.
"""
import argparse, os
import numpy as np

CHANNELS = ["pelvis_x", "pelvis_y", "pelvis_z", "yaw", "footL_x", "footL_y", "footL_z", "footL_yaw", "footL_contact", "footR_x", "footR_y", "footR_z", "footR_yaw", "footR_contact",
            "handL_x", "handL_y", "handL_z", "handL_on", "handR_x", "handR_y", "handR_z", "handR_on", "look_x", "look_y", "look_z", "look_on", "torso_x", "torso_y", "torso_z", "torso_on"]
# the 30 channels of build_vae_dataset.py in the same order; describe_motion.py (and gen_dataset.py) call the pelvis heading "yaw" where build_vae_dataset.py says "pelvis_yaw"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True, help="the folder with train_windows.npz / val_windows.npz (nerva-motion-data/vae)"); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="use every K-th train window"); ap.add_argument("--val-every", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    zt = np.load(os.path.join(a.data, "train_windows.npz"), allow_pickle=False); zv = np.load(os.path.join(a.data, "val_windows.npz"), allow_pickle=False)
    Xt, vt, Xv, vv, dt = zt["windows"], zt["valid"].astype(np.int64), zv["windows"], zv["valid"].astype(np.int64), float(zt["dt"])
    if Xt.shape[2] != len(CHANNELS):
        raise SystemExit(f"{Xt.shape[2]} channels in the windows, {len(CHANNELS)} expected")
    ids = np.concatenate([np.arange(0, len(Xt), a.every), len(Xt) + np.arange(0, len(Xv), a.val_every)]).astype(np.int64)
    frames = [(Xt[i, :vt[i]] if i < len(Xt) else Xv[i - len(Xt), :vv[i - len(Xt)]]) for i in ids]
    off = np.cumsum([0] + [len(f) for f in frames]).astype(np.int64)
    np.savez(os.path.join(a.out, "window_clips.npz"), actions=np.concatenate(frames).astype(np.float32), offsets=off, dt=np.float32(dt), action_names=np.array(CHANNELS))
    np.savez(os.path.join(a.out, "window_map.npz"), ids=ids, n_train=np.int64(len(Xt)), n_val=np.int64(len(Xv)))
    nt = int((ids < len(Xt)).sum())
    print(f"{len(ids)} windows as clips ({nt} train of {len(Xt)}, {len(ids) - nt} val of {len(Xv)}), dt {dt:.4f} -> {a.out}/window_clips.npz, window_map.npz")


if __name__ == "__main__":
    main()
