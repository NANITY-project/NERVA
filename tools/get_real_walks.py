#!/usr/bin/env python3
"""
get_real_walks.py - download a small set of REAL motion-capture walks (CMU Graphics Lab database, BVH mirror on GitHub) and convert them to action
clips, so the realism grader can be checked against real people.

  python tools/get_real_walks.py                 # -> data/real_walks.npz (22 clips, about 5 MB of downloads, cached in data/cmu_cache/)
  python tools/get_real_walks.py --out my.npz --cache /tmp/cmu

Clips: 10 ordinary walks (kind "normal": slow to brisk, several people) and 12 stylised ones (kind "styled": zombie march, hobble, drunk, old man,
gangly teen, graceful lady, strong man, cat, chicken, dinosaur ...). Output keys follow gen_dataset.py plus: angles, labels, kind, description.
The data is from the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), free to use; BVH conversion by B. Hahne (cgspeed), mirrored at
github.com/sfd158/cmu-mocap. Credit them if you publish anything built on it. Needs internet for the first run only.
"""
import argparse, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_to_clip

URL = "https://raw.githubusercontent.com/sfd158/cmu-mocap/HEAD/data/{d}/{f}.bvh"
CLIPS = [  # (file, kind, what it is)
    ("02_01", "normal", "walk"), ("07_01", "normal", "walk"), ("07_04", "normal", "slow walk"), ("07_12", "normal", "brisk walk"), ("08_01", "normal", "walk"),
    ("08_05", "normal", "walk, long stride"), ("16_15", "normal", "walk"), ("16_21", "normal", "brisk walk"), ("35_01", "normal", "walk"), ("137_29", "normal", "normal walk"),
    ("20_08", "styled", "zombie march"), ("21_08", "styled", "zombie march"), ("17_05", "styled", "walk / hobble"), ("137_16", "styled", "drunk walk"),
    ("137_20", "styled", "gangly teenager"), ("137_24", "styled", "graceful lady"), ("137_33", "styled", "old man"), ("137_38", "styled", "sexy lady"),
    ("137_42", "styled", "strong man"), ("137_04", "styled", "cat"), ("137_08", "styled", "chicken"), ("137_12", "styled", "dinosaur"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/real_walks.npz"); ap.add_argument("--cache", default="data/cmu_cache"); ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    acts, angs, labels, kinds, descs, dts = [], [], [], [], [], []
    for f, kind, desc in CLIPS:
        path = os.path.join(a.cache, f + ".bvh")
        if not os.path.exists(path):
            try:
                print(f"  downloading {f}.bvh ...", flush=True); urllib.request.urlretrieve(URL.format(d=f.split("_")[0].zfill(3), f=f), path)
            except Exception as e:
                print(f"  could not download {f}: {e}"); continue
        A, ang, dt = bvh_to_clip.convert(path, a.fps); acts.append(A); angs.append(ang); labels.append(f); kinds.append(kind); descs.append(desc); dts.append(dt)
    if not acts:
        sys.exit("nothing downloaded: check your internet connection, or put the .bvh files in " + a.cache)
    off = np.cumsum([0] + [len(x) for x in acts])
    np.savez_compressed(a.out, actions=np.concatenate(acts), offsets=off, angles=np.concatenate(angs), labels=np.array(labels), kind=np.array(kinds), description=np.array(descs),
                        dt=np.float32(np.mean(dts)), style=np.zeros((len(acts), 1), np.float32), style_names=np.array(["real"]))
    print(f"wrote {a.out}: {len(acts)} clips ({sum(1 for k in kinds if k == 'normal')} normal, {sum(1 for k in kinds if k == 'styled')} styled), {off[-1] * np.mean(dts):.0f} s")


if __name__ == "__main__":
    main()
