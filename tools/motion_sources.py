#!/usr/bin/env python3
"""
motion_sources.py - which motion datasets NERVA may train on, which it may only look at, and which it cannot use at all. One table, one gate.

  python tools/motion_sources.py              # print the table
  python tools/motion_sources.py --check bnrm # exit status 0 if that source may be trained on and redistributed, 1 if not

Every dataset NERVA ingests has an entry here. tools/build_motion_corpus.py refuses a source that is not in the table and, unless you pass --include-restricted, refuses every
source that is not trainable. Restricted sources that you do include go into a SEPARATE store (corpus_restricted.npz) that tools/build_vae_dataset.py never reads.

Tiers
  open            CC BY, CDLA-Permissive and similar: train, redistribute converted clips, commercial use all fine, with attribution and a note of the changes.
  share-alike     CC BY-SA: same, but converted clips you redistribute must carry the same share-alike licence (CC BY-SA 3.0 or a compatible one). Keep the per-source tag.
  noncommercial   CC BY-NC, Snap Non-Commercial: research use only. NOT trained on by default, never exported into include/.
  research-only   the owner's terms limit use to academic research and forbid redistribution without consent (AIST).
  reference-only  usable as the grader's yardstick on your own machine, never redistributed, even converted (CMU, and anything that is CMU underneath).
  unverified      no licence text could be found on the dataset's own pages: treated like noncommercial until the owners confirm in writing.
  unusable        not motion data NERVA can read, a licence that forbids training, or it is not what it claims to be (see `notes`).

Audited 2026-11-03 from each dataset's own page / licence file (URLs in `url`). Licences change and pages move: re-check before a release. This is engineering bookkeeping, not legal advice.
"""
import argparse, sys
from dataclasses import dataclass

OPEN_TIERS = ("open", "share-alike")


@dataclass(frozen=True)
class Source:
    id: str                 # value of the `source` field in the corpus and in cc_motions.npz
    name: str
    license: str
    tier: str
    url: str
    fmt: str                # what the files are
    converter: str          # "bvh" = tools/bvh_to_clip.py reads it as is; otherwise what is missing
    size: str
    notes: str = ""

    @property
    def trainable(self):
        return self.tier in OPEN_TIERS and self.converter == "bvh"

    @property
    def redistributable(self):
        return self.tier in OPEN_TIERS


SOURCES = {s.id: s for s in [
    Source("accad", "ACCAD / Ohio State University motion capture", "CC BY 3.0", "open", "https://accad.osu.edu/research/motion-lab/system-data", "BVH", "bvh", "~300 files",
           "already in NERVA: tools/get_cc_motions.py"),
    Source("100style", "100STYLE (Mason, Starke, Komura)", "CC BY 4.0", "open", "https://zenodo.org/records/8127870", "BVH", "bvh", "16.2 GB (100STYLE.zip 1.5 GB)",
           "already in NERVA: tools/get_cc_motions.py"),
    Source("hdm05", "HDM05 (MPI Informatik / Bonn)", "CC BY-SA 3.0", "share-alike", "https://resources.mpi-inf.mpg.de/HDM05/index.html", "ASF/AMC and C3D (BVH only via third-party conversions)", "bvh",
           "~3 h", "Trainable once you have BVH files. The site itself ships ASF/AMC + C3D, and NERVA has no ASF/AMC reader (see docs/DATA_SOURCES.md). Share-alike: converted clips you "
           "redistribute stay CC BY-SA. The site also asks you to cite the technical report and send the team a copy of what you publish."),
    Source("cmu", "CMU Graphics Lab Motion Capture Database", "free to use, no resale even converted", "reference-only", "http://mocap.cs.cmu.edu/", "BVH (cgspeed conversion)", "bvh", "~2600 trials",
           "already in NERVA: grader yardstick only (docs/CREDITS.md)"),
    Source("mocapact", "MoCapAct rollouts (Microsoft)", "CDLA-Permissive-2.0 (code MIT)", "reference-only", "https://huggingface.co/datasets/microsoft/mocapact-data", "HDF5 joint-space rollouts of the dm_control CMU humanoid",
           "not implemented: needs forward kinematics of the dm_control humanoid", "50 GB (small) to 600 GB (large)",
           "The licence is permissive, but every rollout tracks a CMU clip (CMU_009_12 ...) and the paper/README say nothing about CMU's no-resale term. Kept reference-only, the same way "
           "NERVA treats CMU itself, until Microsoft or CMU clears it. The rollouts are a simulated humanoid, not captured motion."),
    Source("bnrm", "Bandai Namco Research Motion Dataset 1 + 2", "CC BY-NC 4.0", "noncommercial", "https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset", "BVH", "bvh",
           "36,673 + 384,931 frames", "Non-commercial. Fine for local experiments with --include-restricted; not a training source."),
    Source("snapmogen", "SnapMoGen (Snap Inc.)", "Snap Inc. Non-Commercial License 2025", "noncommercial", "https://huggingface.co/datasets/Ericguo5513/SnapMoGen", "BVH (renamed_bvhs.zip) + feature files", "bvh",
           "16.5 GB, 20,450 clips, 43.7 h", "Research and non-commercial use only; no commercial rights. The Hugging Face card has an empty licence field, the LICENSE file in snap-research/SnapMoGen is the authority."),
    Source("aist", "AIST++ (Google) on the AIST Dance Video Database", "Google's page: CC BY 4.0 for its own assets; AIST terms: academic research only", "research-only",
           "https://google.github.io/aistplusplus_dataset/", "SMPL motion (pickle) + 3D keypoints", "not implemented: SMPL parameters, no BVH", "306 MB motion only (2.3 GB complete)",
           "The motion is reconstructed from AIST's videos and AIST's Terms of Use allow academic research only, no commercial use and no redistribution without written consent."),
    Source("kit", "KIT Whole-Body Human Motion Database / KIT Motion-Language", "no licence stated on the database or dataset pages", "unverified", "https://motion-database.humanoids.kit.edu/",
           "C3D + MMM XML", "not implemented: MMM XML / C3D reader", "3.9 GB per download, 3,911 motions",
           "Only a citation requirement was found. Written confirmation from the KIT H2T lab is needed before any use beyond reading it yourself."),
    Source("finegym", "FineGym", "CC BY-NC 4.0 (annotations)", "unusable", "https://sdolivia.github.io/FineGym/", "YouTube video references + temporal annotations", "no motion data at all",
           "n/a", "Contains no mocap or skeletons: only links to gymnastics videos and labels. Getting motion out of it means running a pose estimator over YouTube videos, which NERVA does not do "
           "and whose video terms are not FineGym's to grant. Also NC."),
    Source("eyesjp", "Eyes, Japan free motion data (mocapdata.com)", "CC BY-SA 2.1 Japan (free packs only; purchased packs: no redistribution)", "share-alike", "https://mocapdata.com/Terms_of_Use.html", "zip packs, motion format not confirmed on the site (BVH expected)",
           "bvh", "4671 free motions in 18 category archives",
           "Free packs only: the terms page gives them CC BY-SA 2.1 JP (attribution, share-alike). Purchased packs may not be redistributed or put on a network: never feed those in. The page does not say commercial use or ML "
           "training is fine or not, and the CC licence is silent on it too. Check one archive is BVH before a big run. Attribution is 'in the manner specified by the author': read the pack's own credit line."),
    Source("motionx", "Motion-X (IDEA Research)", "Motion-X licence: non-commercial, authorisation by form", "noncommercial", "https://github.com/IDEA-Research/Motion-X", "SMPL-X parameters (.npy, 322-dim, 30 fps) + text labels", "not implemented: SMPL-X, no BVH",
           "81.1K clips, 15.6M frames",
           "There is no freely open subset: every subset is requested through a Google Form for 'non-commercial purposes'. The AMASS / GRAB / EgoBody parts are labels only: the motion has to be fetched from their own sites "
           "under their own academic licences. The Motion-X and IDEA400 licence PDFs could not be read here."),
    Source("humaneva", "HumanEva-I / II (Brown, MPI)", "PS:License 1.0 (text not readable here)", "unverified", "https://is.mpg.de/code/humaneva-datasets-i-and-ii", "synchronised video + mocap, mocap file format not stated",
           "not implemented: reader", "small (a few subjects, ~10 actions)",
           "The licence text is behind the MPI download form. Max Planck 'PS' licences are usually non-commercial research licences, but that is not confirmed, so it is treated as restricted. It is also video-first and tiny."),
    Source("mixamo", "Mixamo (Adobe)", "Adobe terms: royalty-free use in projects, but raw animation files may not be distributed and Mixamo content may not be used to train ML models", "unusable", "https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html",
           "FBX (needs Adobe ID, per-clip download)", "not usable",
           "n/a", "Adobe's community FAQ (2022) says Mixamo content 'can't be used in training machine-learning models', and the raw files may not be redistributed. A training corpus is exactly those two things. "
           "The 'royalty-free' wording covers using characters and animations in films and games, not this. Not wired up on purpose, with no workaround. If you want it, ask Adobe in writing."),
    Source("addbio", "AddBiomechanics (Stanford NMBL, FAIR Center)", "CC BY 4.0 (core dataset); the user-generated part has per-uploader credits and no stated licence", "open", "https://addbiomechanics.org/download_data.html",
           "B3D / OpenSim marker + model data", "not implemented: OpenSim / B3D reader", "70+ h, 273 participants (core)",
           "Open licence but biomechanics data: markers and OpenSim models, with gait / sports-lab tasks, so little in the way of styles. Needs a marker->skeleton retarget, not just a file reader. The 'Open Movement Database' name did "
           "not turn up on the FAIR Center's pages (it lists AddBiomechanics, the Marker Enhancer dataset and an OpenCap FSHD set on Zenodo, CC BY 4.0); this entry is AddBiomechanics. Marker Enhancer data: no licence stated, so treat as unverified."),
]}

# the sources build_vae_dataset.py may seed the memory with and train on (motion_memory.TRAINING_SOURCES): open / share-alike with a working converter, no one else
TRAINING_SOURCES = tuple(i for i, s in SOURCES.items() if s.trainable)


def get(source):
    if source not in SOURCES:
        raise KeyError(f"unknown motion source '{source}'. Known: {', '.join(SOURCES)}. Add it to tools/motion_sources.py with its licence before ingesting anything.")
    return SOURCES[source]


def gate(source, include_restricted=False):
    """Return the Source, or raise PermissionError saying why it may not be ingested."""
    s = get(source)
    if s.tier == "unusable":
        raise PermissionError(f"{source}: {s.notes}")
    if s.converter != "bvh":
        raise NotImplementedError(f"{source}: no reader yet ({s.converter}). See docs/DATA_SOURCES.md.")
    if not s.trainable and not include_restricted:
        raise PermissionError(f"{source} is {s.tier} ({s.license}) and is not a training source. Pass --include-restricted to ingest it into the separate restricted store for local use.")
    return s


def table():
    rows = [("id", "tier", "licence", "trainable", "reader")]
    for s in SOURCES.values():
        rows.append((s.id, s.tier, s.license, "yes" if s.trainable else "no", "bvh" if s.converter == "bvh" else "missing"))
    w = [max(len(r[i]) for r in rows) for i in range(5)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        s = get(a.check); print(f"{s.id}: {s.tier} ({s.license}) - trainable: {s.trainable}, redistributable: {s.redistributable}"); sys.exit(0 if s.trainable else 1)
    print(table())
