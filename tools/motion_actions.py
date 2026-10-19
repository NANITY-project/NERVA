#!/usr/bin/env python3
"""
motion_actions.py - which actions the grader knows, which REAL motion-capture trials are examples of each, and how a clip is cut out of a trial.

  python tools/motion_actions.py                 # list every action and how many CMU trials match it (downloads the 70 KB trial index once)
  python tools/motion_actions.py sit             # also print the trial ids + descriptions that count as "sit"

Why this exists: a grader that only knows "walking" can only judge walking. Here every action is defined by real examples. An action is
  name        what you pass to the grader (--action sit)
  include     regexes over the CMU trial description that make a trial an example of it (every pattern is anchored on purpose: a trial called
              "walking, squats, stretching, kicking, and punching" is NOT an example of walking, squatting or kicking, it is a mix)
  window      where in a long trial the example is cut from: "energy" the busiest part, "air" around the flight phase, "low" around the lowest
              the pelvis gets (sit, crouch, pick up), "quiet" a still upright stretch (stand)
  secs/n_max  how long each example is and how many of them are kept (examples are spread over as many different performers as possible)
Add your own action in ACTIONS below (or pass --define to get_real_motions.py); everything downstream (profiles, headers, validation) follows.

Data: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), free to use; BVH conversion by B. Hahne (cgspeed), mirrored at
github.com/sfd158/cmu-mocap together with the trial index this file parses. Credit them if you publish anything built on it.
"""
import os, re, sys, urllib.request
from dataclasses import dataclass, field

INDEX_URL = "https://raw.githubusercontent.com/sfd158/cmu-mocap/HEAD/cmu-mocap-index-text.txt"
BVH_URL = "https://raw.githubusercontent.com/sfd158/cmu-mocap/HEAD/data/{d}/{f}.bvh"


@dataclass
class Action:
    name: str
    group: str                      # which header file the examples go to
    about: str
    include: list = field(default_factory=list)
    ids: list = field(default_factory=list)          # trials that count although their description does not say so
    exclude: list = field(default_factory=list)
    window: str = "energy"
    secs: float = 6.0
    n_max: int = 24
    min_n: int = 6                  # fewer examples than this and the action is dropped (a profile from 3 clips is a guess)
    derived: str = ""               # "quiet_stand": examples are also cut from stretches of OTHER trials (CMU has almost no standalone "standing")


# Order matters: the first action that matches a trial wins, so the specific styles (limp, sneak, carry) come before plain walk.
ACTIONS = [
    Action("limp", "locomotion", "walking with a hurt or dragging leg", [r"limp", r"hobbl", r"wounded", r"dragbadleg", r"hurt right leg"], secs=6, n_max=16),
    Action("sneak", "locomotion", "creeping / sneaking / careful steps", [r"^(mickey )?sneak(ing|y)?( walk)?$", r"^sneak sideways$", r"creep", r"stealth", r"^careful ?(walk|creeping)", r"^carefulwalk"], secs=6, n_max=24),
    Action("carry", "locomotion", "walking while carrying a suitcase or object", [r"^carry \d", r"^walk and carry$"], secs=6, n_max=17),
    Action("walk_styled", "locomotion", "walking with a character: zombie, drunk, old man, depressed, cat, dinosaur...",
           [r"(zombie|drunk|depress|clumsy|gangly|elderly|old ?man|graceful lady|sexy lady|strong man|alien|frankenstein|neutral male|bouncy|stiff walk|stretch walk|"
            r"(cat|chicken|dinosaur|mickey|lavish|toocool|cool|sad|excited|attitude|achey)\s?walk|walk with anger|walk jauntily|muscular|walk swinging shoulders|walk with arms out)"],
           secs=6, n_max=40),
    Action("walk", "locomotion", "ordinary walking, slow to brisk, straight and gently turning",
           [r"^(walk|walking|normal ?walk( forward)?|slow ?walk|walk (slow|fast|forward|left|right)|brisk walk|casual ?walk|casualquickwalk|walk ?straight|straight walk|walk/wander|"
            r"confident walk forward|walk, veer (left|right)|walk, 90-degree (left|right) turn|walk forward 90 degree.*|walk forward turn 90.*|walk and turn( \(repeated\))?|"
            r"walk forward, turn around, walk back|walk stop walk|start walk (left|right)|slow walk, stop|follow path.*)$"], secs=6, n_max=140),
    Action("run", "locomotion", "running and jogging", [r"^run(/jog)?(,| |$)(?!.*(jump|duck|dive|over|leap))", r"^careful run"], secs=4, n_max=45),
    Action("hop", "locomotion", "hopping on one foot / hop turns", [r"^(hop on left foot|hop forward|medium hop forward|start hop stop|hopscotch|hop turn.*|hop and turn.*)$"], window="air", secs=4, n_max=20),
    Action("jump", "locomotion", "two-footed jumps (forward, up, turning)",
           [r"^(jump|jumpforward|forward jump|jump ?turn|jumpsmallforward|long jump forward|high jump|\d+ jump|jump up and down.*|jump stop|jump up to grab.*|angle jumps|box jump|"
            r"smalljumpturn|furtherjumpturn|jumptwist|jumping twists|180 jump|360 jumps?)$"], window="air", secs=4, n_max=30),
    Action("turn", "locomotion", "turning on the spot", [r"^(turn in place( \(opposite direction\))?|walk, turn in place \(repeated\)|walk backward, turn in place|turning)$"], secs=5, n_max=12),
    Action("sit", "posture", "sitting down on a chair / stool / the ground, staying seated, getting up again",
           [r"^(sit on (high |step)?stool.*|sit on stepstool.*|sit down.*|sit in chair.*|sitting in chair|(medium|high|low) sit|sitting on ground relaxing)$"], window="low", secs=8, n_max=26),
    Action("stand", "posture", "standing upright and (nearly) still, arms relaxed",
           [r"^(standing( still)?|stand still.*)$"], ids=["77_02", "111_28", "113_21"], window="quiet", secs=4, n_max=40, derived="quiet_stand"),
    Action("crouch", "posture", "crouched / ducking, low body", [r"^(walk (backwards )?(on toes )?crouched|duck under|ducking|duck to avoid flying object|start duck underneath stop)$"], window="low", secs=5, n_max=14),
    Action("pickup", "posture", "bending or squatting to pick something up", [r"^(picking up ball|pick box up, bend from waist|picking box up, bending knees|bend, lift|bend, pick up|pick up|lifting up|bend over, scoop up, rise, lift arm|"
                                                                            r"picking up something|walk forward and pick up object.*|walk up to object, (squat|lean over), pick up.*)$"], window="low", secs=6, n_max=26),
    Action("get_up", "posture", "lying down and getting up from the floor or a chair",
           [r"^(get up from (chair|floor|ground.*)|getting up from chair|lay down and get up|laying down and getting up|getting up from laying down|get up face down|get up laying on side|laying down.*getting up.*)$"],
           window="low", secs=8, n_max=20),
    Action("balance", "posture", "standing balance: one leg, flamingo, balance beam", [r"^(balance on one leg.*|flamingo.*|balance beam|balance object.*)$"], secs=6, n_max=8),
    Action("stretch", "posture", "stretching, yoga, sun salutation", [r"^(stretching|stretch and yawn|stretch - .*|yoga|sun salutation\d*)$"], secs=8, n_max=18),
    Action("kick", "activity", "kicks", [r"^(kick|soccer - kick ball|jump kick|kicking|front kick|front_kicking\d*|left_front_kicking\d*|mawashigeri|punch and kick|punch kick)$"], secs=4, n_max=24),
    Action("punch", "activity", "boxing and punches", [r"^(boxing|punch/strike|walk backwards then attack with a punch)$"], secs=5, n_max=12),
    Action("throw", "activity", "throwing, pitching, tossing, shooting a ball",
           [r"^(baseball pitch|throw|throwing|basketball free throw|basketball shoot|basketball jump shot|basketball lay up|underhand toss|underhand fast pitch|toss and catch|catch and throw( football)?|baseball swing|baseball bunt)$"],
           secs=4, n_max=16),
    Action("gesture", "activity", "waving, signalling, pointing, high-fives", [r"^(wave|wave hello|waving|direct traffic, wave.*|hand signals.*|high five|around the world high five low|basketball signals)$"], secs=6, n_max=28),
    Action("drink", "activity", "drinking and sipping", [r"^(unscrew bottlecap, drink soda.*|drink soda.*|drinking.*|chug.*|sipping.*|hard days drink)$"], secs=6, n_max=14),
    Action("chores", "activity", "sweeping, mopping, washing windows", [r"^(sweep floor|sweeping.*|mop floor|wash windows|washing( a)? window|cleaning window)$"], secs=6, n_max=22),
    Action("dribble", "activity", "dribbling a basketball", [r"^basketball - (forward |backward |sideways |low, fast free style )?dribble.*", r"^basketball - crossover dribble.*"], secs=5, n_max=16),
    Action("dance", "dance", "dancing: ballet, modern, charleston, lindy hop", [r"^dance", r"^charleston", r"^lindyhop", r"^chicken dance$"], secs=8, n_max=40),
]
BY_NAME = {a.name: a for a in ACTIONS}
GROUPS = ["locomotion", "posture", "activity", "dance"]
INTERACTION = re.compile(r"\(\s*(2\s+)?subjects?", re.I)          # two-performer trials: the other person changes what the body does


def fetch_index(cache_dir):
    """The CMU trial list as [(trial_id, description)] and {subject: title}. Cached next to the BVH files."""
    os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, "cmu-mocap-index-text.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(INDEX_URL, path)
    trials, subjects = [], {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        m = re.match(r"^(\d{2,3})_(\d{2,3})\t(.+)$", line)
        if m:
            trials.append((f"{m.group(1)}_{m.group(2)}", re.sub(r"\s+", " ", m.group(3)).strip()))
            continue
        m = re.match(r"^Subject #(\d+) \((.*)\)", line)
        if m:
            subjects[m.group(1).zfill(2)] = m.group(2)
    return trials, subjects


def subject_of(trial_id):
    return trial_id.split("_")[0]


def assign(trials, actions=None):
    """{action name: [(trial_id, description)]}: every trial goes to the FIRST action whose rules match it."""
    actions = actions or ACTIONS; out = {a.name: [] for a in actions}
    for tid, desc in trials:
        d = desc.lower()
        if INTERACTION.search(d):
            continue
        for a in actions:
            if tid in a.ids or (any(re.search(p, d) for p in a.include) and not any(re.search(p, d) for p in a.exclude)):
                out[a.name].append((tid, desc)); break
    return out


def spread_over_subjects(items, n):
    """Pick n trials taking one per performer in turn, so 10 clips come from 10 people rather than one person doing the same thing 10 times."""
    by = {}
    for tid, desc in items:
        by.setdefault(subject_of(tid), []).append((tid, desc))
    out, k = [], 0
    while len(out) < n and any(by.values()):
        for s in sorted(by):
            if by[s] and len(out) < n:
                out.append(by[s].pop(0))
        k += 1
    return out


def define(spec):
    """--define "wave=wave,waving" (name=comma separated regexes): a quick custom action. Anchor the patterns yourself if you need purity."""
    name, _, pats = spec.partition("=")
    return Action(name.strip(), "activity", f"custom: {pats}", [p.strip() for p in pats.split(",") if p.strip()], secs=6, n_max=24)


def main():
    cache = os.path.join("data", "cmu_cache"); trials, subjects = fetch_index(cache); got = assign(trials)
    want = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"{len(trials)} CMU trials ({sum(1 for _, d in trials if INTERACTION.search(d))} two-performer trials are never used)\n")
    print(f"{'action':12s}{'group':12s}{'trials':>7s}{'performers':>12s}{'kept':>6s}   what it is")
    for a in ACTIONS:
        g = got[a.name]; print(f"{a.name:12s}{a.group:12s}{len(g):7d}{len({subject_of(t) for t, _ in g}):12d}{min(len(g), a.n_max):6d}   {a.about}" + ("   [also cut from other trials]" if a.derived else ""))
    if want:
        print(f"\ntrials counted as '{want}':")
        for tid, desc in got.get(want, []):
            print(f"  {tid}  {desc}")


if __name__ == "__main__":
    main()
