#!/usr/bin/env python3
"""
get_cc_motions.py - build NERVA's CC BY motion set from 100STYLE and ACCAD / Ohio State University motion capture.

  python tools/get_cc_motions.py                              # 100STYLE from Zenodo (large download, cached) + ACCAD from data/cc_cache/accad if you put it there -> data/cc_motions.npz
  python tools/get_cc_motions.py --accad ~/Downloads/accad    # folders and/or .zip files you downloaded from the ACCAD page (below)
  python tools/get_cc_motions.py --style100 100STYLE.zip      # a 100STYLE zip (or the unpacked folder) you already have: no download
  python tools/get_cc_motions.py --list                       # show how every file would be labelled, convert nothing (check this first!)
  python tools/get_cc_motions.py --map my_labels.csv          # your own labelling rules: lines "regex,action" (action "skip" drops the file)

What it makes: data/cc_motions.npz in the same layout as data/real_motions.npz (tools/get_real_motions.py): actions [F,30], offsets, dt, angles, labels, action, performer,
description, features + feature_names, action_names ... plus  source ("100style" / "accad")  and  style_label.  Every clip is a window cut from a mocap take, retargeted to the
NERVA skeleton by tools/bvh_to_clip.py (joints are found by name, so the two datasets' different skeletons need no setup; `bvh_to_clip.py --probe file.bvh` shows how a file is read).
This set has two jobs:
  * model training      tools/build_vae_dataset.py trains on it (originals + graded variants). Both licences (CC BY) allow this and allow redistributing the converted data
                        with credit and a note of the changes: docs/CREDITS.md. So, unlike the CMU data, the example clips exported by tools/export_headers.py can be committed.
  * grader reference    tools/motion_grader.py adds it to the CMU examples when it builds the reference ranges.
Actions: ACCAD clips get the grader's ordinary action names (walk, run, stand, kick ...), by keywords in the file name (ACCAD_RULES below: the file names are descriptive,
e.g. "Walk turn left 90"; check the result with --list). 100STYLE is ONE performer acting 100 styles, so it is NOT mixed into walk/run: it gets its own style profiles,
style_walk (forward and turning walks), style_run and style_idle, and `style_label` holds the style name (Zombie, Drunk ...). For splits and held-out checks the group is the style.

Where the data comes from
  100STYLE   Mason, Starke, Komura. Zenodo record 8127870, CC BY 4.0. Downloaded automatically (API lookup, resumable, md5-checked) unless you pass --style100 / --no-download.
             Files are <Style>_<CONTENT>.bvh at 60 Hz; CONTENT = FW/FR forward walk/run, TW*/TR* turning, ID idle, BW/BR backward, SW/SR sidestep. Default: FW TW1 TW2 FR TR1 TR2 TR3 ID.
             Frame_Cuts.csv (if present and readable) trims every file to the part that is the style; otherwise one second is cut off both ends.
  ACCAD/OSU  Open Motion Project, https://accad.osu.edu/research/motion-lab/system-data, CC BY 3.0. Not downloaded automatically (the page decides what the links are): download the
             BVH zips you want, pass the folder or the zips with --accad, or direct zip links with --accad-url.
Honest limits: written without being able to download either dataset in the environment it was developed in; the skeleton, naming and labelling code was tested on files derived from
other BVH data (tools/test_cc_sources.py). Run --list and `bvh_to_clip.py --probe` on your files first; labelling by file name is a heuristic: fix it with --map.
"""
import argparse, csv, hashlib, io, json, os, re, shutil, sys, tempfile, time, urllib.request, zipfile
from dataclasses import dataclass
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bvh_to_clip, motion_actions as MA, motion_features as mf, get_real_motions as GR

ZENODO_RECORD = "8127870"
OUT_DEFAULT = "data/cc_motions.npz"
CREDIT = {"100style": "100STYLE (Mason, Starke, Komura; CC BY 4.0)", "accad": "ACCAD / The Ohio State University motion capture (Open Motion Project; CC BY 3.0)"}
FLAGS = [8, 13, 17, 21, 25, 29]

# ---- 100STYLE: its own style profiles ----
STYLE_ACTIONS = [MA.Action("style_walk", "locomotion", "walking in one of the 100 styles (100STYLE: forward and turning walks)", secs=6, n_max=600, min_n=6),
                 MA.Action("style_run", "locomotion", "running in one of the 100 styles (100STYLE: forward and turning runs)", secs=4, n_max=600, min_n=6),
                 MA.Action("style_idle", "posture", "standing / idling in one of the 100 styles (100STYLE)", secs=4, n_max=300, min_n=6)]
GR.LOCOMOTION_MIN_SPEED.update({"style_walk": 0.25, "style_run": 1.0})        # same rule as walk_styled / run: a "walk" window must actually travel
STYLE_CONTENT = {"FW": "style_walk", "TW1": "style_walk", "TW2": "style_walk", "BW": "style_walk", "SW": "style_walk", "FR": "style_run", "TR1": "style_run", "TR2": "style_run", "TR3": "style_run",
                 "BR": "style_run", "SR": "style_run", "ID": "style_idle"}
STYLE_DEFAULT_CONTENTS = ["FW", "TW1", "TW2", "FR", "TR1", "TR2", "TR3", "ID"]      # no backward / sidestep: the grader's features assume the body moves the way it faces
STYLE_NAME = re.compile(r"^(?P<style>.+?)_(?P<c>BR|BW|FR|FW|ID|SR|SW|TR[123]|TW[12])$", re.I)

# ---- ACCAD / OSU: ordinary action names, from keywords in the file name. First match wins; text = folder + file name, lower case, separators -> spaces ----
TRANSITION = r"\b(to|into)\b"
ACCAD_RULES = [
    ("sneak", r"sneak|creep|tip ?toe|stealth", ""), ("limp", r"limp|hobbl", ""),
    ("carry", r"carry|carrying|suitcase", ""),
    ("run", r"\b(run|running|jog|jogging|sprint)\b", TRANSITION + r"|walk|stand|stop"),
    ("walk", r"\bwalk(ing)?\b", TRANSITION + r"|run|jog|sneak|limp|carry|stand"),
    ("hop", r"\bhop", ""), ("jump", r"jump|leap", TRANSITION),
    ("kick", r"kick", ""), ("punch", r"punch|jab|uppercut|\bhook\b", ""),
    ("throw", r"throw|toss|pitch", ""),
    ("dance", r"danc|salsa|waltz|ballet|disco|charleston", ""),
    ("stretch", r"stretch|yoga", ""),
    ("gesture", r"\bwave|point|salute|\bbow\b|clap|shrug|beckon|gestur|thumbs", ""),
    ("drink", r"drink|sip|bottle", ""), ("chores", r"sweep|\bmop|wash|wipe|scrub", ""),
    ("pickup", r"pick ?up|scoop|lift", ""), ("crouch", r"crouch|duck|squat|kneel", ""),
    ("get_up", r"get ?up|\blie\b|lying|laying", ""), ("sit", r"\bsit", ""), ("balance", r"balance|one leg|flamingo", ""),
    ("turn", r"turn(ing)? (in place|around|on spot)|pivot|\bspin", r"walk|run|jump"),
    ("stand", r"^(.* )?stand( still)?( \d+)?$|\bidle\b|standing", TRANSITION + r"|walk|run|jump|kick|punch|stance|sway|stretch"),
]


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------------------------ archives
@dataclass
class Member:
    container: str                  # path of the folder or zip
    name: str                       # path of the .bvh inside it (or the file path for a plain folder)
    zipped: bool

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.name))[0]

    @property
    def text(self):                 # what the labelling rules read: folder names + file name
        t = os.path.splitext(self.name)[0].replace("\\", "/").lower(); return re.sub(r"[\s_\-\.\(\)/,]+", " ", t).strip()


def scan(paths):
    """Every .bvh (and Frame_Cuts.csv) under the folders / zips given. Zips inside folders are opened; zips inside zips are not."""
    out = []
    def add_zip(zp):
        with zipfile.ZipFile(zp) as z:
            out.extend(Member(zp, n, True) for n in z.namelist() if n.lower().endswith((".bvh", ".csv")) and "__macosx" not in n.lower())
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    if f.lower().endswith(".zip"):
                        add_zip(fp)
                    elif f.lower().endswith((".bvh", ".csv")):
                        out.append(Member(p, os.path.relpath(fp, p), False))
        elif p.lower().endswith(".zip"):
            add_zip(p)
        elif p.lower().endswith((".bvh", ".csv")):
            out.append(Member(os.path.dirname(os.path.abspath(p)), os.path.basename(p), False))
        else:
            say(f"  ignored (not a folder, .zip or .bvh): {p}")
    return out


def fetch(m, tmp):
    """Local path of a member (zip members are unpacked into tmp)."""
    if not m.zipped:
        return os.path.join(m.container, m.name)
    dst = os.path.join(tmp, re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(m.container) + "__" + m.name))
    if not os.path.exists(dst):
        with zipfile.ZipFile(m.container) as z, open(dst, "wb") as f:
            f.write(z.read(m.name))
    return dst


# ------------------------------------------------------------------------------------------------------------------ 100STYLE
def parse_cuts(text):
    """Frame_Cuts.csv -> {(style key, CONTENT): (first, last)} in the files' own frames. Understands a header with <CONTENT>_START / <CONTENT>_STOP style columns; {} if it does not."""
    key = lambda s: re.sub(r"[^a-z0-9]", "", s.lower()); cuts = {}
    try:
        rows = [r for r in csv.reader(io.StringIO(text)) if r]; head = [h.strip().upper() for h in rows[0]]
        col = next((k for k, h in enumerate(head) if "STYLE" in h or h == "NAME"), 0)
        for code in STYLE_CONTENT:
            tok = lambda h: set(re.split(r"[^A-Z0-9]+", h))
            a = [k for k, h in enumerate(head) if code in tok(h) and tok(h) & {"START", "BEGIN", "FROM", "FIRST"}]
            b = [k for k, h in enumerate(head) if code in tok(h) and tok(h) & {"STOP", "END", "TO", "LAST"}]
            if a and b:
                for r in rows[1:]:
                    try:
                        cuts[(key(r[col]), code)] = (int(float(r[a[0]])), int(float(r[b[0]])))
                    except (ValueError, IndexError):
                        pass
    except Exception:
        return {}
    return cuts


def style100_entries(members, contents, tmp):
    cuts = {}
    for m in members:
        if os.path.basename(m.name).lower().startswith("frame_cuts") and m.name.lower().endswith(".csv"):
            p = fetch(m, tmp); cuts = parse_cuts(open(p, encoding="utf-8", errors="ignore").read())
            say(f"  Frame_Cuts.csv: {len(cuts)} cut ranges understood" if cuts else "  Frame_Cuts.csv found but not understood: trimming one second off both ends instead")
    out = []
    for m in members:
        g = STYLE_NAME.match(m.stem)
        if not m.name.lower().endswith(".bvh") or not g:
            continue
        code = g.group("c").upper(); style = g.group("style")
        if code not in contents:
            continue
        cut = cuts.get((re.sub(r"[^a-z0-9]", "", style.lower()), code))
        out.append(dict(source="100style", member=m, action=STYLE_CONTENT[code], performer=f"100style:{style}", style=style, desc=f"100STYLE {style}, {code}", cut=cut, code=code))
    return out


def download(url, dst, size=None, md5=None):
    """Resumable download (HTTP Range) with a progress line and an optional md5 check."""
    os.makedirs(os.path.dirname(dst), exist_ok=True); part = dst + ".part"
    for attempt in range(5):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0; open(part, "wb").close()
                t0 = time.time(); got = 0
                with open(part, "ab") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b); got += len(b)
                        print(f"\r  {os.path.basename(dst)}: {(have + got) / 1e6:8.0f} MB" + (f" of {size / 1e6:.0f}" if size else "") + f"  {got / 1e6 / max(1e-3, time.time() - t0):5.1f} MB/s", end="", flush=True)
            print(); break
        except Exception as e:
            print(f"\n  download interrupted ({type(e).__name__}: {e}); retrying ({attempt + 1}/5) ..."); time.sleep(3 * (attempt + 1))
    else:
        raise IOError(f"could not download {url}")
    if md5:
        h = hashlib.md5()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != md5:
            os.remove(part); raise IOError(f"md5 mismatch for {dst}: got {h.hexdigest()}, Zenodo says {md5}")
    os.replace(part, dst); return dst


def zenodo_zips(record=ZENODO_RECORD):
    """[(file name, url, size, md5)] of the .zip files of a Zenodo record (API lookup; understands the old and the new response layout)."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        j = json.load(r)
    files = j.get("files") or []
    files = list(files.values()) if isinstance(files, dict) else files; out = []
    for f in files:
        key = f.get("key") or f.get("filename") or f.get("id", ""); links = f.get("links") or {}
        if not key.lower().endswith(".zip"):
            continue
        url = links.get("self") or links.get("download") or f"https://zenodo.org/records/{record}/files/{key}?download=1"
        sum_ = (f.get("checksum") or "").replace("md5:", ""); out.append((key, url, f.get("size"), sum_ if re.fullmatch(r"[0-9a-f]{32}", sum_) else None))
    return out


def fetch_style100(cache):
    zips = zenodo_zips()
    if not zips:
        raise IOError("the Zenodo record lists no .zip file")
    say(f"downloading 100STYLE from Zenodo record {ZENODO_RECORD}: " + ", ".join(f"{k} ({(s or 0) / 1e9:.1f} GB)" for k, _, s, _ in zips) + " (resumable; cached)")
    return [download(u, os.path.join(cache, "100style", k), s, md5) for k, u, s, md5 in zips]


# ------------------------------------------------------------------------------------------------------------------ ACCAD
def load_map(path):
    rules = []
    for row in csv.reader(open(path, encoding="utf-8")):
        if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
            rules.append((row[1].strip(), row[0].strip(), ""))
    return rules


def accad_label(text, extra=()):
    for action, inc, exc in list(extra) + ACCAD_RULES:
        if re.search(inc, text) and not (exc and re.search(exc, text)):
            return None if action in ("skip", "-", "") else action
    return None


def accad_entries(members, extra):
    out = []
    for m in members:
        if not m.name.lower().endswith(".bvh"):
            continue
        act = accad_label(m.text, extra); g = re.search(r"(male|female)\s?(\d+)", m.text + " " + re.sub(r"[^a-z0-9]+", " ", os.path.basename(m.container).lower()))
        perf = (g.group(1).capitalize() + g.group(2)) if g else "unknown"
        out.append(dict(source="accad", member=m, action=act, performer=f"accad:{perf}", style="", desc=f"ACCAD {perf}: {m.stem}", cut=None, code=""))
    return out


# ------------------------------------------------------------------------------------------------------------------ conversion
def resample(A, ang, dt, fps):
    """To exactly `fps` (linear; yaw is already unwrapped; contact / on flags stay 0/1). A file whose rate is not a multiple of fps (100 Hz, ...) needs this."""
    if abs(dt * fps - 1.0) < 0.01:
        return A, ang, dt
    n = int(np.floor((len(A) - 1) * dt * fps)) + 1; t = np.arange(n) / fps; src = np.arange(len(A)) * dt
    A2 = np.stack([np.interp(t, src, A[:, k]) for k in range(A.shape[1])], 1); A2[:, FLAGS] = (A2[:, FLAGS] > 0.5)
    return A2.astype(np.float32), np.stack([np.interp(t, src, ang[:, k]) for k in range(ang.shape[1])], 1).astype(np.float32), 1.0 / fps


def action_obj(name):
    return MA.BY_NAME.get(name) or next(a for a in STYLE_ACTIONS if a.name == name)


def build(entries, a):
    """entries -> clips. Returns (clips, meta, skipped) where clips = [(A, ang, dt)] and meta = dict per clip."""
    clips, meta, skipped = [], [], []; per = {}; tmp = tempfile.mkdtemp(prefix="nerva_cc_"); t0 = time.time()
    by_action = {}
    for e in entries:
        by_action.setdefault(e["action"], []).append(e)
    # accad: spread over performers and respect each action's clip limit; 100style: at most --per-style clips per (style, action)
    todo = []
    for act, es in by_action.items():
        if es[0]["source"] == "accad":
            items = [(f"{e['performer'].split(':')[1]}_{k}", k) for k, e in enumerate(es)]
            todo += [es[k] for _, k in MA.spread_over_subjects(items, len(items))]
        else:
            todo += es
    if a.limit:
        todo = todo[: a.limit]
    for n, e in enumerate(todo):
        act = action_obj(e["action"]); key = (e["source"], e["performer"], e["action"])
        if e["source"] == "100style" and per.get(key, 0) >= a.per_style:
            continue
        if e["source"] == "accad" and per.get((e["source"], e["action"]), 0) >= act.n_max:
            continue
        path = None
        try:
            path = fetch(e["member"], tmp); A, ang, dt = bvh_to_clip.convert(path, a.fps, rows=e["cut"])
            A, ang, dt = resample(A, ang, dt, a.fps)
            if e["source"] == "100style" and not e["cut"]:
                trim = int(round(1.0 / dt))
                if len(A) > 4 * trim:
                    A, ang = A[trim:-trim], ang[trim:-trim]
            wins = GR.windows(A, dt, act, a.per_file)
        except Exception as ex:
            skipped.append((e["member"].name, f"{type(ex).__name__}: {str(ex)[:160]}")); continue
        finally:
            if path and e["member"].zipped and os.path.exists(path):
                os.remove(path)
        for k, (i0, i1) in enumerate(wins):
            if e["source"] == "100style" and per.get(key, 0) >= a.per_style:
                break
            clips.append((GR.canonical(A[i0:i1]), ang[i0:i1], dt)); per[key] = per.get(key, 0) + 1; per[(e["source"], e["action"])] = per.get((e["source"], e["action"]), 0) + 1
            meta.append(dict(label=f"{e['source']}:{e['member'].stem}#{k}", action=e["action"], performer=e["performer"], desc=e["desc"], source=e["source"], style=e["style"]))
        if (n + 1) % 50 == 0:
            say(f"  {n + 1}/{len(todo)} files, {len(clips)} clips, {time.time() - t0:.0f}s")
    shutil.rmtree(tmp, ignore_errors=True)
    return clips, meta, skipped


def save(path, clips, meta):
    dt = float(np.mean([c[2] for c in clips])); off = np.cumsum([0] + [len(c[0]) for c in clips]); feats = np.array([mf.extract(c[0], c[2]) for c in clips], np.float32)
    S = lambda k: np.array([m[k] for m in meta])
    np.savez_compressed(path, actions=np.concatenate([c[0] for c in clips]).astype(np.float32), offsets=off, angles=np.concatenate([c[1] for c in clips]), dt=np.float32(dt),
                        labels=S("label"), action=S("action"), performer=S("performer"), description=S("desc"), features=feats, feature_names=np.array(mf.NAMES),
                        action_names=np.array(sorted({m["action"] for m in meta})), kind=np.array(["real"] * len(clips)), style=np.zeros((len(clips), 1), np.float32), style_names=np.array(["real"]),
                        source=S("source"), style_label=S("style"))
    return dt, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--cache", default="data/cc_cache"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--style100", nargs="*", help="100STYLE zip(s) or folder (default: download from Zenodo)"); ap.add_argument("--no-style100", action="store_true")
    ap.add_argument("--accad", nargs="*", default=None, help="folders / zips / .bvh files of ACCAD data (default: <cache>/accad if it exists)")
    ap.add_argument("--accad-url", nargs="*", default=[], help="direct links to ACCAD BVH zips (from the ACCAD page); downloaded into <cache>/accad")
    ap.add_argument("--no-download", action="store_true"); ap.add_argument("--map", help="CSV of 'regex,action' labelling rules for ACCAD file names (action skip drops)")
    ap.add_argument("--style-contents", nargs="+", default=STYLE_DEFAULT_CONTENTS, help="100STYLE take types to use (default: " + " ".join(STYLE_DEFAULT_CONTENTS) + ")")
    ap.add_argument("--per-file", type=int, default=3, help="max windows cut from one take"); ap.add_argument("--per-style", type=int, default=6, help="max clips per 100STYLE style and action")
    ap.add_argument("--limit", type=int, help="only the first N files (smoke test)"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); entries = []; tmp0 = tempfile.mkdtemp(prefix="nerva_cc_idx_")
    bad = [c for c in a.style_contents if c.upper() not in STYLE_CONTENT]
    if bad:
        sys.exit("unknown 100STYLE take type(s): " + ", ".join(bad) + "; known: " + " ".join(STYLE_CONTENT))
    contents = {c.upper() for c in a.style_contents}
    # 100STYLE
    if not a.no_style100:
        src = a.style100
        if not src:
            cached = [os.path.join(a.cache, "100style", f) for f in os.listdir(os.path.join(a.cache, "100style"))] if os.path.isdir(os.path.join(a.cache, "100style")) else []
            cached = [p for p in cached if p.lower().endswith(".zip")]
            if cached:
                src = cached
            elif a.no_download:
                say("100STYLE: not found in the cache and --no-download given: skipped (use --style100 PATH)")
            else:
                try:
                    src = fetch_style100(a.cache)
                except Exception as ex:
                    say(f"100STYLE: could not download ({type(ex).__name__}: {ex}).\n  Download it from https://zenodo.org/records/{ZENODO_RECORD} and pass --style100 FILE.zip")
        if src:
            say("indexing 100STYLE ..."); entries += style100_entries(scan(src), contents, tmp0)
    # ACCAD
    adir = os.path.join(a.cache, "accad"); os.makedirs(adir, exist_ok=True)
    for u in a.accad_url:
        try:
            download(u, os.path.join(adir, re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(u.split("?")[0]) or "accad.zip")))
        except Exception as ex:
            say(f"ACCAD: could not download {u}: {ex}")
    apaths = a.accad if a.accad else ([adir] if os.listdir(adir) else [])
    if apaths:
        say("indexing ACCAD ..."); extra = load_map(a.map) if a.map else []; entries += accad_entries(scan(apaths), extra)
    else:
        say("ACCAD: no data given (--accad PATH, --accad-url URL, or put the zips in " + adir + "): continuing without it")
    if not entries:
        sys.exit("nothing to do: no 100STYLE and no ACCAD data found")
    lab = [e for e in entries if e["action"]]; unl = [e for e in entries if not e["action"]]
    say(f"\n{'source':9s}{'action':12s}{'files':>6s}{'performers/styles':>19s}")
    for s_ in ("100style", "accad"):
        for act in sorted({e["action"] for e in lab if e["source"] == s_}):
            es = [e for e in lab if e["source"] == s_ and e["action"] == act]; say(f"{s_:9s}{act:12s}{len(es):6d}{len({e['performer'] for e in es}):19d}")
    if unl:
        say(f"\n{len(unl)} ACCAD file(s) match no action (not used). Examples: " + "; ".join(e["member"].stem for e in unl[:8]) + "   -> add rules with --map")
    if a.list:
        for e in lab[:400]:
            say(f"  {e['source']:9s}{e['action']:12s}{e['member'].name}")
        shutil.rmtree(tmp0, ignore_errors=True); return
    clips, meta, skipped = build(lab, a); shutil.rmtree(tmp0, ignore_errors=True)
    for n, why in skipped[:12]:
        say(f"  skipped {n}: {why}")
    if len(skipped) > 12:
        say(f"  ... and {len(skipped) - 12} more skipped files")
    if not clips:
        sys.exit("no clips were made (every file was skipped): see the messages above")
    # an action needs a few clips from a few people to be worth a profile
    drop = [n for n in {m["action"] for m in meta} if sum(1 for m in meta if m["action"] == n) < action_obj(n).min_n]
    if drop:
        say("too few clips, dropped: " + ", ".join(sorted(drop))); keep = [i for i, m in enumerate(meta) if m["action"] not in drop]; clips = [clips[i] for i in keep]; meta = [meta[i] for i in keep]
    if not clips:
        sys.exit("no action has enough clips")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); dt, off = save(a.out, clips, meta)
    say(f"\nwrote {a.out}: {len(clips)} clips, {off[-1] * dt / 60:.1f} minutes, {len({m['performer'] for m in meta})} performers/styles, actions: " + ", ".join(f"{n} {sum(1 for m in meta if m['action'] == n)}" for n in sorted({m['action'] for m in meta})))
    say("credit: " + "; ".join(CREDIT[s_] for s_ in sorted({m["source"] for m in meta})) + "  (docs/CREDITS.md)")


if __name__ == "__main__":
    main()
