#!/usr/bin/env python3
"""
goal_plan.py - text -> a per-window goal TIMELINE: the Goal Translator (tools/goal_translator, spec NANITY_GOAL_TRANSLATOR_SPEC.md) behind the steering rows of the UI.

    "Walkrightandthenleftthenwaveurhandandidkmovearound?"
        ->  [{turn: -23.4, speed: 1.26}, {turn: -23.4, speed: 1.26}, {turn: 23.4, speed: 1.26}, {turn: 23.4, speed: 1.26}, {unspecified: true}, {unspecified: true}]

The rows are what `server.parse_timeline` already takes (turn in degrees, + left / - right, speed in m/s), plus `{"unspecified": true}`: a window the text says nothing about, which
the prior gets NO goal for (the null goal). The server builds the four goal numbers from turn and speed with arc_goal over (W - 1) * dt, like tools/closed_loop.py; the translator never
emits dx, dz.

Two backends, same interface, and they are never mixed silently:
  model     a trained translator GGUF (--goal-translator FILE, tools/goal_translator/export_goal_translator.py), run by the numpy reference runtime. Any failure is an ERROR shown to the user.
  stage_c   no model: the deterministic bootstrap (stage_c.py). Used only when no model file was given; the source is reported as "stage_c" so the page says so.
There is no keyword fallback behind either of them.

The translator lives in this repo (tools/goal_translator, next to the VAE and prior trainers), so the conventions (token layout, left/right sign, bins) have one source of truth.  Standard library + numpy (+ gguf for a model file).
"""
import math, os, sys


class GoalPlanError(RuntimeError):
    pass


def _modules():
    d = os.environ.get("NANITY_GT_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "goal_translator")
    if not os.path.exists(os.path.join(d, "gt_conventions.py")):
        raise GoalPlanError(f"the Goal Translator was not found at {d} (expected tools/goal_translator in this repo, or set NANITY_GT_DIR)")
    if d not in sys.path:
        sys.path.insert(0, d)
    import gt_conventions as C
    import stage_c as SC
    return C, SC


class GoalPlanner:
    def __init__(self, gguf=None):
        self.C, self.SC = _modules()
        self.file, self.model = gguf, None
        if gguf:
            try:
                import gt_numpy
                self.model = gt_numpy.NumpyTranslator.from_gguf(gguf)
            except Exception as e:
                raise GoalPlanError(f"could not load the translator {os.path.basename(gguf)}: {e}")

    def info(self):
        return dict(source="model" if self.model else "stage_c", file=os.path.basename(self.file) if self.file else None)

    def plan(self, text, prior_stats, W, stride, dt):
        """-> dict(rows, source, ignored, notes, windows). prior_stats: the prior's goal_stats (turn_max, speed_cruise, speed_max); W, stride, dt: the prior / VAE timing.
        Raises GoalPlanError with a message for the user (invalid plan, window cap, too long, nothing placed)."""
        C, SC = self.C, self.SC
        try:
            prior = C.Stats.from_dict(prior_stats)
        except Exception:
            raise GoalPlanError("this prior has no goal_stats (turn_max, speed_cruise, speed_max): it was trained without goal numbers")
        timing = C.Timing(int(W), int(stride), float(dt))
        notes = []
        try:
            if self.model:
                plan, dec = self.model.translate(text), self.model.stats
                mt = self.model.timing
                if (mt.W, mt.stride) != (timing.W, timing.stride) or abs(mt.dt - timing.dt) > 1e-6:
                    notes.append(f"the translator was trained for windows of {mt.W} frames every {mt.stride}, this prior uses {timing.W} / {timing.stride}: the plan's pace may be off")
                source = "model"
            else:
                plan, dec, source = SC.translate(text, prior, timing), prior, "stage_c"
        except C.PlanError as e:
            raise GoalPlanError(str(e))
        notes += list(plan.notes)
        rows, clamped_t, clamped_s = [], 0, 0
        for w in plan.windows:
            if w.unspecified:
                rows.append({"unspecified": True})
                continue
            turn, speed = C.bin_to_turn(w.turn_bin, dec), C.bin_to_speed(w.speed_bin, dec)
            t2, s2 = max(-prior.turn_max, min(prior.turn_max, turn)), min(prior.speed_max, speed)      # never ask for more than the prior's own training windows contain
            clamped_t += abs(t2 - turn) > 1e-9
            clamped_s += abs(s2 - speed) > 1e-9
            rows.append(dict(turn=round(math.degrees(t2), 1), speed=round(s2, 3)))
        if clamped_t or clamped_s:
            notes.append(f"{clamped_t + clamped_s} value(s) were clamped to what this prior's data contains (turn_max {math.degrees(prior.turn_max):.0f} deg, speed_max {prior.speed_max:.2f} m/s)")
        if not rows:
            raise GoalPlanError("nothing in the text could be turned into goals" + (f" (not expressible: {'; '.join(plan.ignored)})" if plan.ignored else ""))
        return dict(rows=rows, source=source, ignored=list(plan.ignored), notes=notes, windows=len(rows), step_seconds=round(timing.step_seconds, 3))
