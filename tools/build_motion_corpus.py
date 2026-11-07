#!/usr/bin/env python3
"""
motion_sources.py - the motion datasets NERVA may train on. One table, one gate. Only sources with an open licence are listed.

  python tools/motion_sources.py              # print the table
  python tools/motion_sources.py --check hdm05 # exit status 0 if listed (trainable), 1 if not

tools/build_motion_corpus.py refuses any source that is not in this table. There is deliberately no switch to override that: datasets whose licence is non-commercial, research-only, forbids
ML training or is unclear are not in the code at all (see docs/DATA_SOURCES.md). To add a dataset, read its licence first, then add an entry with the URL and the exact licence wording.

Tiers
  open         CC BY and similar: train, redistribute converted clips, commercial use fine, with attribution and a note of the changes.
  share-alike  CC BY-SA: same, but converted clips you redistribute must carry the same share-alike licence (CC BY-SA 3.0 or a compatible one). Keep the per-source tag.
               CC BY-SA 3.0 / 2.1 JP are not listed by Creative Commons as compatible with (A)GPL; whether trained weights count as an adaptation is unsettled: see docs/DATA_SOURCES.md.

Audited 2026-11-03 and 2026-11-07 from each dataset's own page (URLs in `url`). Licences change and pages move: re-check before a release. This is engineering bookkeeping, not legal advice.
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
    converter: str          # "bvh" = tools/bvh_to_clip.py reads it as is
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
           "~3 h", "Trainable once you have BVH files. CC BY-SA allows commercial use; the site only says its aim is 'research purposes' (no NC term), which is a slight ambiguity: ask the authors if it matters. The site itself ships ASF/AMC + C3D, and NERVA has no ASF/AMC reader (see docs/DATA_SOURCES.md). Share-alike: converted clips you "
           "redistribute stay CC BY-SA. The site also asks you to cite the technical report and send the team a copy of what you publish."),
    Source("eyesjp", "Eyes, Japan free motion data (mocapdata.com)", "CC BY-SA 2.1 Japan (free packs only; purchased packs: no redistribution)", "share-alike", "https://mocapdata.com/Terms_of_Use.html", "zip packs, motion format not confirmed on the site (BVH expected)",
           "bvh", "4671 free motions in 18 category archives",
           "Free packs only: the terms page names both 'Attribution 2.1 Japan' and 'Attribution-Share Alike 2.1 Japan'; treated as the stricter share-alike. Purchased packs may not be redistributed or put on a network: never feed those in. The page does not say commercial use or ML "
           "training is fine or not, and the CC licence is silent on it too. Check one archive is BVH before a big run. Attribution is 'in the manner specified by the author': read the pack's own credit line."),
]}

# the sources build_vae_dataset.py may seed the memory with and train on (motion_memory.TRAINING_SOURCES)
TRAINING_SOURCES = tuple(i for i, s in SOURCES.items() if s.trainable)


def get(source):
    if source not in SOURCES:
        raise KeyError(f"'{source}' is not an allowed motion source. Allowed: {', '.join(SOURCES)}. Only datasets with an open licence are listed; to add one, check its licence and add an entry in tools/motion_sources.py.")
    return SOURCES[source]


def gate(source):
    """Return the Source, or raise if it is not listed."""
    s = get(source)
    if not s.trainable:
        raise PermissionError(f"{source} is not a training source")
    return s


def table():
    rows = [("id", "tier", "licence")] + [(s.id, s.tier, s.license) for s in SOURCES.values()]
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        try:
            s = get(a.check)
        except KeyError as e:
            print(str(e).strip("\"'")); sys.exit(1)
        print(f"{s.id}: {s.tier} ({s.license})"); sys.exit(0)
    print(table())
