import json, math, os, sys, threading, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NEBULA_LIB", os.path.join(os.path.dirname(__file__), "..", "..", "build", "libnebula.so"))
import sim
from sim import World, ActionError, dist

DT = 1 / 60


def run(w, seconds):
    for _ in range(int(seconds / DT)):
        w.step(DT)


def run_until(w, cond, limit=40.0):
    t = 0.0
    while t < limit:
        w.step(DT); t += DT
        if cond():
            return t
    raise AssertionError(f"condition not met within {limit}s")


def idle(c):
    return c.current is None and not c.queue


class SimTests(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.a = self.w.add_character("Alice", "baker", 0, 0, 0)
        self.alice = self.w.chars[self.a]
        self.w.add_prop("table", (0, 0.4, 1.5), (1.2, 0.8, 0.6), ["fixed", "surface"])
        self.w.add_prop("cup", (0.0, 0.85, 1.5), (0.08, 0.1, 0.08))
        self.w.add_prop("ball", (1.5, 0.1, 0.0), (0.2, 0.2, 0.2))
        run(self.w, 0.5)

    def tearDown(self):
        self.w.close()

    def hand(self, c, side="right"):
        return self.w.hand_pos(c.hands[side])

    def test_rest_pose_is_stable_and_natural(self):
        run(self.w, 2)
        for side in ("left", "right"):
            err = dist(self.hand(self.alice, side), self.alice.rest_point(side))
            self.assertLess(err, 0.01, f"{side} hand drifted {err*1000:.1f} mm from rest")

    def test_reach_point_arrives(self):
        self.w.enqueue(self.a, "REACH right 0.2 1.2 0.35")
        run_until(self.w, lambda: idle(self.alice))
        self.assertLess(dist(self.hand(self.alice), (0.2, 1.2, 0.35)), 0.01)

    def test_walk_to_prop_stops_near_and_faces_it(self):
        self.w.enqueue(self.a, "WALK_TO table")
        run_until(self.w, lambda: idle(self.alice))
        near = self.w.props["table"].nearest_point((self.alice.x, 0.4, self.alice.z))
        self.assertLess(math.hypot(near[0] - self.alice.x, near[2] - self.alice.z), 0.6)
        self.assertLess(abs(sim.ang_diff(self.alice.yaw, 0.0)), 0.3)   # table is along +z

    def test_grab_auto_approaches_and_carries(self):
        self.w.enqueue(self.a, "GRAB right cup")
        run_until(self.w, lambda: idle(self.alice))
        cup = self.w.props["cup"]
        self.assertEqual(cup.held_by, (self.a, "right"))
        self.assertEqual(self.alice.hands["right"].holding, "cup")
        run(self.w, 1.0)
        self.assertLess(dist(tuple(cup.pos), self.hand(self.alice)), 0.01, "cup must follow the hand")
        # carried hand is near the body, not stuck out at the table
        self.assertLess(dist(self.hand(self.alice), self.alice.carry_point("right")), 0.03)
        self.assertTrue(all(abs(v) < 5 for v in self.alice.pos()))

    def test_carry_follows_a_walking_character(self):
        self.w.enqueue(self.a, "GRAB right cup")
        self.w.enqueue(self.a, "WALK_TO 2.0 -1.0")
        run_until(self.w, lambda: idle(self.alice))
        run(self.w, 1.0)
        cup = self.w.props["cup"]
        self.assertLess(dist(tuple(cup.pos), self.hand(self.alice)), 0.01)
        self.assertLess(dist(self.hand(self.alice), self.alice.carry_point("right")), 0.04)

    def test_drop_falls_onto_surface_or_floor(self):
        self.w.enqueue(self.a, "GRAB right cup")
        self.w.enqueue(self.a, "DROP right")
        run_until(self.w, lambda: idle(self.alice))
        run(self.w, 2.0)
        cup, table = self.w.props["cup"], self.w.props["table"]
        self.assertIsNone(cup.held_by)
        rest = table.top() + cup.size[1] / 2 if table.contains_xz(cup.pos[0], cup.pos[2]) else cup.size[1] / 2
        self.assertAlmostEqual(cup.pos[1], rest, places=2)

    def test_floor_prop_makes_the_character_crouch(self):
        self.w.enqueue(self.a, "GRAB left ball")
        peak = 0.0
        def watch():
            nonlocal peak
            peak = max(peak, self.alice.crouch)
            return idle(self.alice)
        run_until(self.w, watch)
        self.assertGreater(peak, 0.35, "should crouch for a floor-level prop")
        self.assertEqual(self.w.props["ball"].held_by, (self.a, "left"))
        run(self.w, 2.0)
        self.assertLess(self.alice.crouch, 0.05, "should stand back up afterwards")

    def test_handover(self):
        b = self.w.add_character("Bob", "", 0.0, 3.0, math.pi)
        bob = self.w.chars[b]
        self.w.enqueue(self.a, "GRAB right cup")
        self.w.enqueue(self.a, "OFFER right bob")
        run_until(self.w, lambda: idle(self.alice))
        self.assertEqual(self.alice.hands["right"].mode, "offer")
        self.w.enqueue(b, "GRAB left cup")
        run_until(self.w, lambda: idle(bob))
        self.assertEqual(self.w.props["cup"].held_by, (b, "left"))
        self.assertIsNone(self.alice.hands["right"].holding)
        self.assertTrue(any("took cup" in e for e in self.alice.events))

    def test_rejections_are_reported_and_sim_survives(self):
        for bad in ("GRAB right unicorn", "GRAB right table", "DROP right", "LOOK_AT nobody", "OFFER right cup"):
            self.w.enqueue(self.a, bad)
        run_until(self.w, lambda: idle(self.alice))
        self.assertGreaterEqual(len(self.alice.events), 5)
        self.assertTrue(any("unknown target" in e for e in self.alice.events))
        self.assertTrue(any("too big or fixed" in e for e in self.alice.events))
        run(self.w, 0.5)
        self.assertTrue(all(math.isfinite(v) for v in self.alice.pos()))

    def test_parser(self):
        for bad in ("", "DANCE", "SAY", "WAIT", "WAIT x", "WAIT nan", "REACH", "REACH middle cup", "REACH right",
                    "REACH right 1 2", "WALK_TO", "WALK_TO 1 2 3", "GRAB right a b", "WAVE", "DROP up"):
            with self.assertRaises(ActionError, msg=bad):
                self.w.enqueue(self.a, bad)
        self.assertEqual(len(self.alice.queue), 0)
        for good in ("say hello there", "WAIT 1", "REACH left 0 1 0.3", "walk_to table", "WALK_TO 1 2", "WAVE right", "POINT left ball"):
            self.w.enqueue(self.a, good)

    def test_say_is_heard_by_others(self):
        b = self.w.add_character("Bob", "", 2, 2)
        self.w.enqueue(self.a, 'SAY "hello Bob"')
        run_until(self.w, lambda: idle(self.alice))
        self.assertTrue(any("hello Bob" in e for e in self.w.chars[b].events))
        self.assertTrue(any(e["kind"] == "say" for e in self.w.log))

    def test_elbows_stay_behind_after_turning_around(self):
        self.w.enqueue(self.a, "WALK_TO 0 -2.0")       # turns to walk toward -z
        run_until(self.w, lambda: idle(self.alice))
        self.w.enqueue(self.a, "REACH right 0.3 1.2 -2.5")
        run_until(self.w, lambda: idle(self.alice))
        sh, el, hd = self.w.eng.world_positions(self.alice.hands["right"].cid)
        mid = tuple((sh[i] + hd[i]) / 2 for i in range(3))
        behind = sum((el[i] - mid[i]) * self.alice.fwd()[i] for i in range(3))
        self.assertLess(behind, 0.0, f"elbow should bend backward relative to facing, got {behind:.3f}")

    def test_recording_take(self):
        self.w.start_recording("t")
        self.w.enqueue(self.a, "WAVE right")
        run(self.w, 2.0)
        take = self.w.stop_recording()
        self.assertAlmostEqual(take["duration"], 2.0, delta=0.1)
        self.assertGreater(len(take["frames"]), 55)
        self.assertEqual(len(take["frames"][0]["chars"][0]["arms"]["right"]), 3)
        json.dumps(take)
        with self.assertRaises(ActionError):
            self.w.stop_recording()

    def test_snapshot_is_valid_json_with_finite_numbers(self):
        self.w.enqueue(self.a, "GRAB right cup")
        run(self.w, 3)
        def finite(x):
            if isinstance(x, float): self.assertTrue(math.isfinite(x))
            elif isinstance(x, dict): [finite(v) for v in x.values()]
            elif isinstance(x, list): [finite(v) for v in x]
        finite(json.loads(self.w.snapshot_json))

    def test_threaded_loop_and_submit(self):
        self.w.start()
        cid = self.w.call(lambda w: w.add_character("Zed", "", 3, 3))
        self.w.call(lambda w: w.enqueue(cid, "WALK_TO 3 5"))
        import time; time.sleep(2.5)
        z = json.loads(self.w.snapshot_json)["chars"]
        zed = next(c for c in z if c["id"] == cid)
        self.assertGreater(zed["pos"][2], 3.5)
        with self.assertRaises(ActionError):
            self.w.call(lambda w: w.enqueue("ghost", "WAIT 1"))
        self.w.close()
        self.w = World()


if __name__ == "__main__":
    unittest.main(verbosity=2)
