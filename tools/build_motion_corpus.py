#!/usr/bin/env python3
"""
build_motion_corpus.py - turn motion-capture collections that are too big to keep (1 TB+ in total) into a small corpus that keeps what is DIFFERENT and drops what is not.

  python tools/build_motion_corpus.py --list-sources                       # the licence table (tools/motion_sources.py)
  python tools/build_motion_corpus.py --source hdm05=~/mocap/hdm05_bvh --hours 20 --out data/corpus
  python tools/build_motion_corpus.py --source bnrm=~/mocap/bandai --source snapmogen=https://host/renamed_bvhs.zip --include-restricted --hours 40 --max-files 3000
  python tools/build_motion_corpus.py --out data/corpus --report            # what is in it, how small it got
  python tools/build_motion_corpus.py --out data/corpus --export-cc data/cc_motions_hdm05.npz        # -> the layout tools/build_vae_dataset.py --cc reads

How it keeps the size down (nothing here needs the whole dataset on disk)
  1. STREAMING   a source is a folder, a .zip, a .bvh, or a URL to a .zip. A URL zip is NOT downloaded: only its directory is read, then each chosen file is fetched with an HTTP Range
                 request (the server must support Range; most CDNs, Hugging Face and Zenodo do). --max-files / --sample cut the file list BEFORE anything is fetched; files are drawn round-robin
                 over performers/folders so a 10 % sample is spread over people, not 10 % of one person. One file is on disk at a time. Sources are interleaved, so no source arrives first and fills the budget.
  2. WINDOWS     each file -> --win second windows (stride win/2), retargeted to the NERVA skeleton by tools/bvh_to_clip.py, started at the origin facing +Z (same as the other loaders).
  3. FILTER      a window with non-finite numbers, or whose jitter / contact-mismatch (the grader's two gates, tools/motion_features.py) are far outside the first windows seen, is dropped.
  4. DIVERSITY   every window is measured with the grader's 42 features, scaled by each feature's own noise floor and weighted so each of the 9 feature groups counts equally. A window
                 closer than TAU to one already kept is NOT stored; it only adds 1 to that kept window's `weight` (how many windows it stands for). Exact duplicates are dropped by hash first.
  5. BUDGET      --hours is a hard cap per store. When it is exceeded TAU grows 15 % and the kept set is thinned (heaviest first) until it fits: the corpus gets coarser, never biased to
                 whoever came first. --max-source-share stops one big source (SnapMoGen is 44 h) from taking more than that share of the budget.
  6. COMPRESSION positions at 1 mm, yaw at 0.5 mrad, joint angles at 0.005 deg, contact/on flags exact; int16, delta coded along time, deflated: about 3.7 MB per hour of motion (measured
                 on real CMU files; float32 would be ~13 MB/h). The worst change this causes in any of the grader's 42 features is 0.19 of that feature's noise floor (on jitter_hf, the most
                 sensitive; 2 mm steps gave 0.47). tests/test_motion_corpus.py checks it stays under 0.5.
  Checkpoints every --checkpoint files; run the same command again to resume (finished files are skipped).

Two stores, never mixed
  corpus.npz              sources whose licence allows training and redistributing (tools/motion_sources.py: open / share-alike)
  corpus_restricted.npz   only with --include-restricted: non-commercial / research-only / unverified sources, for LOCAL experiments. Its clips never influence what the open store keeps, and
                          tools/build_vae_dataset.py never reads it (its source ids are not in motion_memory.TRAINING_SOURCES).

Then scale it up with the engine
  python tools/build_motion_corpus.py --out data/corpus --export-cc data/cc_motions_ext.npz
  python tools/build_vae_dataset.py --cc data/cc_motions_ext.npz --hours 30
  The export is in the same layout as get_cc_motions.py. build_vae_dataset.py keeps the originals and fills up to --hours with graded variants (mirror, time warp, ...) and engine-verified clips, as
  always. Only windows with an action name the grader knows are exported (--keep-unlabeled: all): a source whose file names say nothing (most of them) needs a labelling map, --map rules.csv
  (lines "regex,action", same as get_cc_motions.py --map), or its clips stay in the corpus but out of the training set. Honest limit: no `torso` array is exported (the spine orientation is not stored).

Honest limits
  * Diversity here means "different on the grader's 42 numbers" - speed, posture, rhythm, hand use, turning. It does not know about style, intent or semantics, and a rare-but-real motion that
    the 42 numbers cannot tell apart from a common one is merged with it.
  * Only BVH is read. HDM05 (ASF/AMC), AIST++ (SMPL), KIT (MMM/C3D) and MoCapAct (HDF5) need converters first: docs/DATA_SOURCES.md.
  * Written and tested without access to the big datasets: tests use real CMU BVH files for the conversion path and a local Range server for the URL path. Run --report on your first real
    run and look at the per-source numbers before trusting a 40 h budget.
"""
import argparse, collections, hashlib, io, json, os, random, re, shutil, sys, tempfile, time, urllib.request, zipfile, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bvh_to_clip, motion_features as mf, motion_sources as MS, get_cc_motions as CC, get_real_motions as GR

FPS = 30
CH = 35                                                    # 30 action channels + 5 joint angles (knee L/R, elbow L/R, trunk lean)
FLAGS = [8, 13, 17, 21, 25, 29]; YAW = [3, 7, 12]
SC = np.full(CH, 0.001); SC[YAW] = 0.0005; SC[FLAGS] = 1.0; SC[30:] = 0.005        # quantisation step per channel: m, rad, flag, degrees (0.5 mm / finer overflows int16 deltas)
GATE_IDX = [mf.INDEX[g] for g in mf.GATES]
FLOOR = np.asarray(mf.FLOOR, np.float64)
GROUP_IDX = {g: [j for j, n in enumerate(mf.NAMES) if mf.GROUP[n] == g] for g in mf.GROUP_NAMES}
FEAT_W = np.array([1.0 / np.sqrt(len(GROUP_IDX[mf.GROUP[n]])) for n in mf.NAMES])            # each of the 9 groups weighs the same, whatever its number of features
UNLABELED = "unlabeled"
STORE_FILES = {"open": "corpus.npz", "restricted": "corpus_restricted.npz"}


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------------------------ compression
def pack(A, ang):
    """(actions [F,30], angles [F,5]) -> zlib blob of delta-coded int16, channel-major. None if a step does not fit (corrupt clip)."""
    Q = np.rint(np.concatenate([A, ang], 1).astype(np.float64) / SC).astype(np.int64)
    D = np.diff(Q, axis=0, prepend=np.zeros((1, CH), np.int64))
    if np.abs(D).max() > 32767:
        return None
    return zlib.compress(np.ascontiguousarray(D.T).astype(np.int16).tobytes(), 6)


def unpack(blob, F):
    D = np.frombuffer(zlib.decompress(blob), np.int16).reshape(CH, F).T.astype(np.int32)
    X = np.cumsum(D, axis=0) * SC; A = X[:, :30].astype(np.float32); A[:, FLAGS] = A[:, FLAGS] > 0.5
    return A, X[:, 30:].astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------ the diversity measure
class Scaler:
    """Robust per-feature centre and scale (median, 1.4826 MAD, never below the feature's own noise floor), frozen after the first windows."""
    def __init__(self, med, scale):
        self.med, self.scale = np.asarray(med, np.float64), np.asarray(scale, np.float64)

    @staticmethod
    def fit(F):
        med = np.nanmedian(F, axis=0); mad = 1.4826 * np.nanmedian(np.abs(F - med), axis=0)
        med = np.where(np.isfinite(med), med, 0.0); scale = np.maximum(np.where(np.isfinite(mad), mad, 0.0), FLOOR)
        return Scaler(med, scale)

    def z(self, F):
        return (np.asarray(F, np.float64) - self.med) / self.scale

    def embed(self, F):
        z = np.nan_to_num(self.z(F), nan=0.0, posinf=8.0, neginf=-8.0)
        return (np.clip(z, -8, 8) * FEAT_W).astype(np.float32)

    def noisy(self, F, gate_z):
        z = self.z(F)[:, GATE_IDX]; return np.any(z > gate_z, axis=1)


# ------------------------------------------------------------------------------------------------------------------ a store of kept windows
class Store:
    def __init__(self, name, budget_hours, tau, max_share):
        self.name, self.budget, self.tau, self.max_share = name, budget_hours * 3600.0 / (1.0 / FPS), tau, max_share        # budget in frames
        self.emb = np.zeros((0, len(mf.NAMES)), np.float32); self.feat = np.zeros((0, len(mf.NAMES)), np.float32)
        self.blob, self.frames, self.weight, self.meta = [], [], [], []; self.hashes = set()
        self.n_seen = collections.Counter(); self.fr_seen = collections.Counter(); self.n_kept_ever = collections.Counter()
        self.rej = collections.Counter(); self._tau_s = {}

    def __len__(self):
        return len(self.blob)

    def total_frames(self, source=None):
        return sum(f for f, m in zip(self.frames, self.meta) if source is None or m["source"] == source)

    # --- nearest kept window
    def nearest(self, E):
        if len(self) == 0:
            return np.full(len(E), np.inf), np.full(len(E), -1)
        n2 = (self.emb.astype(np.float64) ** 2).sum(1); best = np.full(len(E), np.inf); arg = np.zeros(len(E), np.int64)
        for s in range(0, len(self), 20000):
            B = self.emb[s:s + 20000].astype(np.float64); d = (E.astype(np.float64) ** 2).sum(1)[:, None] + n2[None, s:s + 20000] - 2.0 * E @ B.T
            k = d.argmin(1); v = d[np.arange(len(E)), k]; upd = v < best; best[upd] = v[upd]; arg[upd] = k[upd] + s
        return np.sqrt(np.maximum(best, 0.0)), arg

    # --- admission
    def offer(self, items):
        """items: dicts with e (embedding), f (features), blob, frames, meta. Near duplicates only add weight to what is already kept."""
        if not items:
            return
        E = np.stack([it["e"] for it in items]); d0, a0 = self.nearest(E); n0 = len(self); new = []                # new: (embedding, item) admitted by this file, row n0 + position
        for i, it in enumerate(items):
            h = hashlib.md5(it["blob"]).digest(); src = it["meta"]["source"]; self.n_seen[src] += 1; self.fr_seen[src] += it["frames"]
            if h in self.hashes:
                self.rej["exact duplicate"] += 1; continue
            self.hashes.add(h); d, a = d0[i], a0[i]
            for k, (ek, _) in enumerate(new):                                      # also against what this same file already added
                dk = float(np.linalg.norm(E[i] - ek))
                if dk < d:
                    d, a = dk, n0 + k
            if a >= 0 and d < self.tau:
                if a < n0:
                    self.weight[a] += 1
                else:
                    new[a - n0][1]["w"] += 1
                self.rej["near duplicate (merged)"] += 1; continue
            it["w"] = 1; new.append((E[i], it)); self.n_kept_ever[src] += 1
        if new:
            self.emb = np.vstack([self.emb, np.stack([e for e, _ in new])]); self.feat = np.vstack([self.feat, np.stack([it["f"] for _, it in new]).astype(np.float32)])
            for _, it in new:
                self.blob.append(it["blob"]); self.frames.append(it["frames"]); self.weight.append(it["w"]); self.meta.append(it["meta"])
        self.enforce_budget()

    def enforce_budget(self):
        """Hysteresis: thinning starts above the budget and goes down to 90 % (85 % of a source's cap), so it does not run again on the next file."""
        if self.total_frames() > self.budget:
            for _ in range(80):
                if self.total_frames() <= 0.9 * self.budget:
                    break
                self.tau *= 1.15; self.thin(range(len(self)), self.tau)
        srcs = {m["source"] for m in self.meta}; cap = self.max_share * self.budget
        if len(srcs) > 1:
            for src in srcs:
                if self.total_frames(src) > cap:
                    t = self._tau_s.get(src, self.tau)
                    for _ in range(80):
                        if self.total_frames(src) <= 0.85 * cap:
                            break
                        t *= 1.15; self.thin([i for i, m in enumerate(self.meta) if m["source"] == src], t)
                    self._tau_s[src] = t

    def thin(self, idx, tau):
        """Greedy, heaviest first: keep a window unless it is within tau of one already kept (from idx); its weight goes to the one that covers it."""
        idx = sorted(idx, key=lambda i: -self.weight[i]); kept = []; K = np.zeros((len(idx), self.emb.shape[1]), np.float32); c = 0; add = collections.Counter()
        for i in idx:
            if c:
                d = ((K[:c] - self.emb[i]) ** 2).sum(1); j = int(d.argmin())
                if d[j] < tau * tau:
                    add[kept[j]] += self.weight[i]; self.rej["thinned (merged)"] += 1; continue
            K[c] = self.emb[i]; kept.append(i); c += 1
        for i, w in add.items():
            self.weight[i] += w
        drop = set(idx) - set(kept); keep = [i for i in range(len(self)) if i not in drop]
        self.emb, self.feat = self.emb[keep], self.feat[keep]
        self.blob = [self.blob[i] for i in keep]; self.frames = [self.frames[i] for i in keep]; self.weight = [self.weight[i] for i in keep]; self.meta = [self.meta[i] for i in keep]

    # --- disk
    def save(self, path):
        off = np.cumsum([0] + [len(b) for b in self.blob]); m = lambda k: np.array([x[k] for x in self.meta])
        tmp = path + ".tmp.npz"
        np.savez(tmp, blob=np.frombuffer(b"".join(self.blob), np.uint8), blob_off=off, frames=np.array(self.frames, np.int32), weight=np.array(self.weight, np.int32), features=self.feat,
                 feature_names=np.array(mf.NAMES), dt=np.float32(1.0 / FPS), tau=np.float64(self.tau), source=m("source"), action=m("action"), performer=m("performer"), label=m("label"),
                 description=m("description"), style_label=m("style"), seen_sources=np.array(sorted(self.n_seen)), seen_windows=np.array([self.n_seen[s] for s in sorted(self.n_seen)]),
                 seen_frames=np.array([self.fr_seen[s] for s in sorted(self.n_seen)]), tau_per_source=np.array(json.dumps(self._tau_s)))
        os.replace(tmp, path)

    def load(self, path, scaler):
        z = np.load(path, allow_pickle=False); off = z["blob_off"]; b = z["blob"].tobytes(); n = len(z["frames"])
        self.blob = [b[off[i]:off[i + 1]] for i in range(n)]; self.frames = [int(x) for x in z["frames"]]; self.weight = [int(x) for x in z["weight"]]; self.feat = z["features"].astype(np.float32)
        self.emb = scaler.embed(self.feat) if n else np.zeros((0, len(mf.NAMES)), np.float32); self.tau = float(z["tau"]); self._tau_s = json.loads(str(z["tau_per_source"]))
        S = lambda k: [str(x) for x in z[k]]
        self.meta = [dict(source=s, action=a, performer=p, label=l, description=d, style=y) for s, a, p, l, d, y in zip(S("source"), S("action"), S("performer"), S("label"), S("description"), S("style_label"))]
        self.hashes = {hashlib.md5(x).digest() for x in self.blob}
        self.n_seen = collections.Counter(dict(zip(S("seen_sources"), map(int, z["seen_windows"])))); self.fr_seen = collections.Counter(dict(zip(S("seen_sources"), map(int, z["seen_frames"]))))

    # --- export in the layout of get_cc_motions.py (what motion_grader.load_real / build_vae_dataset --cc read)
    def export_cc(self, path, keep_unlabeled=False):
        sel = [i for i, m in enumerate(self.meta) if keep_unlabeled or m["action"] != UNLABELED]
        if not sel:
            raise SystemExit("nothing to export: no window has an action name (label the files with --map, or pass --keep-unlabeled)")
        clips = [unpack(self.blob[i], self.frames[i]) for i in sel]; off = np.cumsum([0] + [len(c[0]) for c in clips]); S = lambda k: np.array([self.meta[i][k] for i in sel])
        np.savez_compressed(path, actions=np.concatenate([c[0] for c in clips]), offsets=off, angles=np.concatenate([c[1] for c in clips]), dt=np.float32(1.0 / FPS),
                            labels=np.array([f"{self.meta[i]['source']}:{self.meta[i]['label']}" for i in sel]), action=S("action"), performer=S("performer"), description=S("description"),
                            features=self.feat[sel], feature_names=np.array(mf.NAMES), action_names=np.array(sorted(set(S("action")))), kind=np.array(["real"] * len(sel)),
                            style=np.zeros((len(sel), 1), np.float32), style_names=np.array(["real"]), source=S("source"), style_label=S("style"), weight=np.array([self.weight[i] for i in sel], np.int32))
        return len(sel), len(self) - len(sel)


# ------------------------------------------------------------------------------------------------------------------ reading a source without downloading it
class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with a small block cache: zipfile.ZipFile(HTTPRangeFile(url)) reads a remote zip's directory and single members only."""
    BLOCK = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, collections.OrderedDict()
        req = urllib.request.Request(url, headers=dict(CC.UA, Range="bytes=0-0"))
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise IOError(f"{url}: the server does not support HTTP Range requests, so the zip cannot be read without downloading it (download it and pass the file)")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos

    def _block(self, k):
        if k in self.cache:
            self.cache.move_to_end(k); return self.cache[k]
        a = k * self.BLOCK; b = min(self.size, a + self.BLOCK) - 1
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=dict(CC.UA, Range=f"bytes={a}-{b}")), timeout=60) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        self.cache[k] = data
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n); out = []
        while self.pos < end:
            k, o = divmod(self.pos, self.BLOCK); blk = self._block(k); take = blk[o:o + end - self.pos]; out.append(take); self.pos += len(take)
            if not take:
                break
        return b"".join(out)

    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)


class Shard:
    """One folder / zip / remote zip / .bvh. members() -> [(key, relative name)]; fetch(name, tmpdir) -> local path of that .bvh."""
    def __init__(self, source, spec):
        self.source, self.spec, self.zip, self.remote = source, spec, None, False
        if re.match(r"https?://", spec) and spec.lower().split("?")[0].endswith(".zip"):
            self.zip = zipfile.ZipFile(HTTPRangeFile(spec)); self.remote = True
        elif os.path.isfile(spec) and spec.lower().endswith(".zip"):
            self.zip = zipfile.ZipFile(spec)
        elif os.path.isfile(spec) and spec.lower().endswith(".bvh"):
            self.files = [os.path.basename(spec)]; self.root = os.path.dirname(os.path.abspath(spec))
        elif os.path.isdir(spec):
            self.root = spec; self.files = sorted(os.path.relpath(os.path.join(r, f), spec) for r, _, fs in os.walk(spec) for f in fs if f.lower().endswith(".bvh"))
        else:
            raise FileNotFoundError(f"{source}: '{spec}' is not a folder, .zip, .bvh or an http(s) URL to a .zip")
        if self.zip:
            self.files = sorted(n for n in self.zip.namelist() if n.lower().endswith(".bvh") and "__macosx" not in n.lower())

    def fetch(self, name, tmp):
        dst = os.path.join(tmp, re.sub(r"[^A-Za-z0-9._-]+", "_", name))
        if self.zip:
            with open(dst, "wb") as f:
                f.write(self.zip.read(name))
            return dst
        return os.path.join(self.root, name)


def performer_of(source, name):
    """The group a file belongs to (a person / a style / a folder): files are sampled round-robin over groups, and a held-out split must never cut a group in two."""
    stem = os.path.splitext(os.path.basename(name))[0]; low = name.replace("\\", "/").lower(); g = None
    if source == "hdm05":
        g = re.search(r"hdm_([a-z]{2})_", stem.lower()); g = f"hdm05:{g.group(1)}" if g else None
    elif source == "accad":
        g = re.search(r"(male|female)\s?(\d+)", low); g = f"accad:{g.group(1).capitalize()}{g.group(2)}" if g else None
    elif source == "100style":
        g = CC.STYLE_NAME.match(stem); g = f"100style:{g.group('style')}" if g else None
    if g:
        return g
    parts = low.split("/"); return f"{source}:{parts[-2]}" if len(parts) > 1 else f"{source}:{stem}"          # otherwise the folder is the group; a flat source gets one group per file


def plan(shards, max_files, seed, done):
    """Job list: per source, files drawn round-robin over performer groups (each group shuffled), cut to max_files; then the sources interleaved. Files already done are skipped."""
    rng = random.Random(seed); per_source = []
    for sh in shards:
        groups = collections.defaultdict(list)
        for n in sh.files:
            if f"{sh.source}|{sh.spec}|{n}" not in done:
                groups[performer_of(sh.source, n)].append(n)
        for g in groups.values():
            rng.shuffle(g)
        order = list(groups); rng.shuffle(order); jobs = []
        while any(groups.values()):
            for g in order:
                if groups[g]:
                    jobs.append((sh, groups[g].pop()))
        per_source.append(jobs[:max_files] if max_files else jobs)
    out = []
    for k in range(max((len(j) for j in per_source), default=0)):
        for j in per_source:
            if k < len(j):
                out.append(j[k])
    return out


# ------------------------------------------------------------------------------------------------------------------ windows
def cut(A, ang, win, stride):
    F = len(A); W = int(round(win * FPS)); S = max(1, int(round(stride * FPS))); minlen = FPS
    if F < minlen:
        return []
    if F <= int(W * 1.25):
        return [(0, F)]
    return [(i, i + W) for i in range(0, F - W + 1, S)]


def convert_file(path, win, stride):
    """-> [(actions [F,30] canonical, angles [F,5])] windows of one BVH, or raises."""
    A, ang, dt = bvh_to_clip.convert(path, FPS)
    if abs(dt * FPS - 1.0) > 0.02:
        A, ang, dt = CC.resample(A, ang, dt, FPS)
    return [(GR.canonical(A[i0:i1]), np.asarray(ang[i0:i1], np.float32)) for i0, i1 in cut(A, ang, win, stride)]


# ------------------------------------------------------------------------------------------------------------------ the run
def run(a):
    os.makedirs(a.out, exist_ok=True); spath = os.path.join(a.out, "state.json"); st = json.load(open(spath)) if os.path.exists(spath) else {}
    if st and (st.get("win") != a.win or st.get("fps") != FPS):
        sys.exit(f"{a.out} was built with --win {st.get('win')}: use the same --win or another --out")
    rules = CC.load_map(a.map) if a.map else []; shards = []; tmp = tempfile.mkdtemp(prefix="nerva_corpus_"); cache = os.path.abspath(a.cache)
    for spec in a.source:
        src, _, where = spec.partition("=")
        if not where:
            sys.exit(f"--source expects ID=PATH_OR_URL, got '{spec}'")
        src = src.strip()
        try:
            s = MS.gate(src, a.include_restricted)
        except (PermissionError, NotImplementedError, KeyError) as e:
            sys.exit(str(e).strip("'\""))
        where = os.path.expanduser(where.strip())
        try:
            sh = Shard(src, where)
        except Exception as e:
            sys.exit(f"{src}: {type(e).__name__}: {e}")
        say(f"{src}: {len(sh.files)} BVH files in {where}" + (" (remote zip: read with Range requests, not downloaded)" if sh.remote else "") + f"  [{s.tier}: {s.license}]"); shards.append(sh)
    if not shards:
        sys.exit("no --source given")
    scaler = Scaler(st["scaler"]["med"], st["scaler"]["scale"]) if st.get("scaler") else None
    stores = {}
    for tier in ("open", "restricted"):
        stores[tier] = Store(tier, a.hours, a.tau, a.max_source_share); p = os.path.join(a.out, STORE_FILES[tier])
        if os.path.exists(p):
            if scaler is None:
                sys.exit(f"{p} exists but state.json has no scaler: remove both to start again")
            stores[tier].load(p, scaler); stores[tier].budget = a.hours * 3600.0 * FPS; stores[tier].max_share = a.max_source_share
    done = set(st.get("done", [])); jobs = plan(shards, a.max_files, a.seed, done); counters = collections.Counter(st.get("counters", {})); warm = []
    say(f"{len(jobs)} files to read ({len(done)} already done); budget {a.hours:g} h per store; TAU {stores['open'].tau:.3f}")
    t0 = time.time(); since = 0

    def checkpoint(final=False):
        for tier, s in stores.items():
            if len(s) or os.path.exists(os.path.join(a.out, STORE_FILES[tier])):
                s.save(os.path.join(a.out, STORE_FILES[tier]))
        json.dump(dict(win=a.win, fps=FPS, done=sorted(done), counters=dict(counters), scaler=dict(med=scaler.med.tolist(), scale=scaler.scale.tolist()) if scaler is not None else None), open(spath + ".tmp", "w"))
        os.replace(spath + ".tmp", spath)

    def process(batch):
        """batch: [(store tier, items-without-embedding)]; embeds with the frozen scaler, filters, offers."""
        for tier, items in batch:
            if not items:
                continue
            F = np.stack([it["f"] for it in items]); noisy = scaler.noisy(F, a.gate_z); E = scaler.embed(F); ok = []
            for it, e, bad in zip(items, E, noisy):
                if bad:
                    stores[tier].rej["noisy (jitter / contact gate)"] += 1
                else:
                    it["e"] = e; ok.append(it)
            stores[tier].offer(ok)

    pending = []                                                                  # windows read before the scaler exists (warm-up)
    try:
        for n, (sh, name) in enumerate(jobs):
            key = f"{sh.source}|{sh.spec}|{name}"; s = MS.get(sh.source); tier = "open" if s.trainable else "restricted"
            try:
                path = sh.fetch(name, tmp); wins = convert_file(path, a.win, a.win / 2)
                if sh.zip and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                counters["unreadable file"] += 1; done.add(key)
                if counters["unreadable file"] <= 5:
                    say(f"  skipped {name}: {type(e).__name__}: {str(e)[:120]}")
                continue
            stem = os.path.splitext(os.path.basename(name))[0]; act = CC.accad_label(re.sub(r"[\s_\-\.\(\)/,]+", " ", os.path.splitext(name)[0].lower()).strip(), rules) or UNLABELED
            perf = performer_of(sh.source, name); items = []
            for k, (A, ang) in enumerate(wins):
                blob = pack(A, ang)
                if blob is None or not np.all(np.isfinite(A)):
                    stores[tier].rej["corrupt (non-finite / jump)"] += 1; continue
                items.append(dict(f=mf.extract(A, 1.0 / FPS), blob=blob, frames=len(A), meta=dict(source=sh.source, action=act, performer=perf, label=f"{stem}#{k}", description=f"{sh.source}: {name}", style="")))
            done.add(key); counters["files"] += 1; counters["windows"] += len(items)
            if scaler is None:
                pending.append((tier, items)); warm.extend(it["f"] for it in items)
                if len(warm) >= a.warmup or n == len(jobs) - 1:
                    scaler = Scaler.fit(np.stack(warm)); say(f"  scaler fitted on {len(warm)} windows; filtering and diversity start now"); process(pending); pending = []
            else:
                process([(tier, items)])
            since += 1
            if since >= a.checkpoint:
                checkpoint(); since = 0
            if (n + 1) % 25 == 0 or n == len(jobs) - 1:
                o = stores["open"]; r = stores["restricted"]
                say(f"  [{time.time() - t0:6.0f}s] {n + 1}/{len(jobs)} files, {counters['windows']} windows read | open: {len(o)} kept, {o.total_frames() / FPS / 3600:.2f} h, TAU {o.tau:.2f}"
                    + (f" | restricted: {len(r)} kept, {r.total_frames() / FPS / 3600:.2f} h, TAU {r.tau:.2f}" if len(r) else ""))
    except KeyboardInterrupt:
        say("\ninterrupted: saving a checkpoint, run the same command again to resume")
    finally:
        if scaler is None and warm:
            scaler = Scaler.fit(np.stack(warm)); process(pending)
        if scaler is not None:
            checkpoint(True)
        shutil.rmtree(tmp, ignore_errors=True)
    report(a.out)


def report(out):
    st = json.load(open(os.path.join(out, "state.json"))); say(f"\ncorpus in {out}: {st['counters'].get('files', 0)} files read, {st['counters'].get('windows', 0)} windows, {st['counters'].get('unreadable file', 0)} unreadable files")
    sc = Scaler(st["scaler"]["med"], st["scaler"]["scale"]) if st.get("scaler") else None
    for tier, fn in STORE_FILES.items():
        p = os.path.join(out, fn)
        if not os.path.exists(p):
            continue
        s = Store(tier, 1, 0, 1); s.load(p, sc); size = os.path.getsize(p); hrs = s.total_frames() / FPS / 3600; raw = s.total_frames() * 30 * 4
        say(f"\n[{tier}] {fn}: {len(s)} windows, {hrs:.2f} h, {size / 1e6:.1f} MB on disk ({size / 1e6 / max(hrs, 1e-9):.1f} MB per hour; float32 actions alone would be {raw / 1e6:.0f} MB, x{raw / max(1, sum(map(len, s.blob))):.1f} smaller), final TAU {s.tau:.3f}")
        say(f"  {'source':10s}{'seen h':>8s}{'kept h':>8s}{'kept %':>8s}{'windows':>9s}{'mean weight':>13s}")
        for src in sorted(s.n_seen):
            idx = [i for i, m in enumerate(s.meta) if m["source"] == src]; seen = s.fr_seen[src] / FPS / 3600; kept = sum(s.frames[i] for i in idx) / FPS / 3600
            say(f"  {src:10s}{seen:8.2f}{kept:8.2f}{100 * kept / max(seen, 1e-9):7.1f}%{len(idx):9d}{(np.mean([s.weight[i] for i in idx]) if idx else 0):13.1f}")
        acts = collections.Counter(m["action"] for m in s.meta); say("  actions: " + ", ".join(f"{k} {v}" for k, v in acts.most_common(12)) + ("  (unlabeled windows stay out of --export-cc)" if UNLABELED in acts else ""))
        if s.rej:
            say("  dropped: " + ", ".join(f"{k} {v}" for k, v in s.rej.most_common()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], metavar="ID=PATH_OR_URL", help="a source id from tools/motion_sources.py and a folder, .zip, .bvh or URL of a .zip (repeatable)")
    ap.add_argument("--out", default="data/corpus"); ap.add_argument("--hours", type=float, default=20.0, help="budget per store, hours of motion kept (default 20)")
    ap.add_argument("--max-files", type=int, help="read at most this many files per source (drawn round-robin over performers); everything else stays unread, and for a URL zip is never fetched")
    ap.add_argument("--win", type=float, default=4.0, help="window length in seconds"); ap.add_argument("--tau", type=float, default=0.35, help="starting distance below which a window counts as a duplicate (grows by itself to fit --hours)")
    ap.add_argument("--max-source-share", type=float, default=0.6, help="no source may keep more than this share of the budget (when there are several)"); ap.add_argument("--gate-z", type=float, default=10.0, help="drop windows whose jitter / contact mismatch is this many noise-scales above the first windows")
    ap.add_argument("--warmup", type=int, default=1500, help="windows read before the feature scale is fixed (use several sources in one run so they mix)")
    ap.add_argument("--include-restricted", action="store_true", help="also ingest non-commercial / research-only / unverified sources, into the separate corpus_restricted.npz (local use only)")
    ap.add_argument("--map", help="labelling rules 'regex,action' (action 'skip' drops the file), same as get_cc_motions.py --map"); ap.add_argument("--checkpoint", type=int, default=200, help="save every N files")
    ap.add_argument("--cache", default="data/corpus_cache", help="(unused for URL zips, which are read with Range requests)"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-sources", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--export-cc", metavar="OUT.npz", help="write the open store in the get_cc_motions.py layout"); ap.add_argument("--store", choices=["open", "restricted"], default="open")
    ap.add_argument("--keep-unlabeled", action="store_true", help="with --export-cc: also export windows without an action name")
    a = ap.parse_args()
    if a.list_sources:
        print(MS.table()); return
    if a.report:
        report(a.out); return
    if a.export_cc:
        st = json.load(open(os.path.join(a.out, "state.json"))); s = Store(a.store, 1, 0, 1); s.load(os.path.join(a.out, STORE_FILES[a.store]), Scaler(st["scaler"]["med"], st["scaler"]["scale"]))
        if a.store == "restricted":
            say("!! RESTRICTED store: non-commercial / research-only / unverified sources. For local experiments only; do not train a released model on it or commit it.")
        n, dropped = s.export_cc(a.export_cc, a.keep_unlabeled); say(f"wrote {a.export_cc}: {n} windows" + (f" ({dropped} unlabeled windows left out)" if dropped else "")); return
    run(a)


if __name__ == "__main__":
    main()
