# Motion grader: judging datasets per action against real motion capture

`tools/` can now judge "is this a good **sit** dataset?", not only "is this a human walk?".

| file | what it does |
|---|---|
| `tools/motion_actions.py` | the 24 actions (sit, stand, walk, run, jump, crouch, get_up, kick, dance ...) and which CMU trials are examples of each |
| `tools/get_real_motions.py` | downloads the trials, cuts labelled examples (650 clips, 76 performers, ~56 min) -> `data/real_motions.npz` |
| `tools/motion_features.py` | ~40 body measures that exist for any action (height, trunk, legs, feet, speed, hands, rhythm, jitter) |
| `tools/motion_grader.py` | per-clip score, "does it look like THIS action", variety, dataset verdict PASS / MARGINAL / FAIL |
| `tools/validate_grader.py` | validates the grader on held-out real clips (leave-one-performer-out), wrong labels and corrupted clips |
| `tools/export_headers.py` | writes `include/motion/*.hpp`: real examples per action + the grader's reference ranges, plain C++17 |
| `tools/motion_test.py` | one command that tests all of it (`make motion-test`, or `--quick`) |
| `tools/motion_roll.py` | rolling labeller: slides a window along a clip, says which action happens when and grades each segment against ITS action; marks plausible-but-unknown stretches `novel` |
| `tools/motion_memory.py` | muscle memory: keeps valid motions that are not in the label set, with strength that grows with use and fades without it; recall by name or similarity; promotes a name to a gradable action at 4 examples |
| `tools/motion_vary.py` | makes graded variants of a stored clip (mirror, time warp, amplitude, gaze, drift, tremor; heavy-tailed strength + occasional "uncommon" twist) so it is never replayed the same way |
| `tools/build_vae_dataset.py` | one command: seeds the memory with the CMU clips, adds engine clips, fills up to N hours with graded variants, splits by source/performer, writes jsonl + windows + stats + report |
| `tools/vae_eval.py` | judges a model's samples against REAL unseen windows: grader pass rate, nearest-training-window distance (replay check), diversity, engine verdict for samples that start at rest |
| `tools/realism.py` | unchanged: the older walking-only gait grader |

```
python tools/motion_test.py                              # test everything (~1 min, downloads the data the first time)
python tools/motion_grader.py data/clips.npz --action sit    # judge a dataset
python tools/motion_grader.py data/clips.npz                 # file with a per-clip `action` array: every action judged separately
python tools/export_headers.py                               # -> include/motion/  (commit these)
python tools/motion_roll.py data/clips.npz --cut data/seg.npz   # which action when; every segment graded; writes labelled segments
python tools/motion_memory.py add-segments data/seg.npz         # keep the 'novel' segments (valid, but no known label)
python tools/motion_memory.py recall --like data/clips.npz --clip 3   # quick recall; see the file's docstring for every command
python tools/build_vae_dataset.py --hours 10                    # the whole VAE training set in one command (make vae-data); report in data/vae/report.txt
```
Limits: the references are CMU performers (mostly young adults, in a lab); `dribble` has one performer, `turn` two, `carry`/`balance`/`crouch` three to five, so
those profiles are weak. Walk/limp/sneak, jump/hop and drink/gesture/dance/stretch are counted as families: confusing them within a family is not an error of
"recognised as". Blur and time-reversal are physically plausible, the grader does not try to catch them.
Data: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), BVH conversion by B. Hahne (cgspeed), mirror github.com/sfd158/cmu-mocap. Credit them.
