import math, os, re, sys, time, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NEBULA_LIB", os.path.join(os.path.dirname(__file__), "..", "..", "build", "libnebula.so"))
import agent
from agent import parse_reply, build_grammar, build_messages, AgentManager, MockLLM, LLMError
from sim import World


class Scripted:
    """LLM stub: returns queued replies, records every prompt it saw."""
    def __init__(self, *replies): self.replies, self.seen = list(replies), []
    def complete(self, messages, grammar=None):
        self.seen.append(messages)
        r = self.replies.pop(0) if self.replies else "WAIT 1"
        if isinstance(r, Exception): raise r
        return r


def wait_for(cond, limit=15.0):
    t = time.time()
    while time.time() - t < limit:
        if cond(): return True
        time.sleep(0.05)
    return False


class ParserTests(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(parse_reply("GRAB right cup\nSAY hi")[0], ["GRAB right cup", "SAY hi"])

    def test_small_model_noise(self):
        text = "Sure! Here is what I'll do:\n```\n1. grab right cup\n- walk_to bob\n* `say Here you go`\n```\nHope that helps"
        lines, ignored = parse_reply(text)
        self.assertEqual(lines, ["GRAB right cup", "WALK_TO bob", "SAY Here you go"])
        self.assertTrue(any("Sure" in i for i in ignored))

    def test_caps_and_garbage(self):
        self.assertEqual(parse_reply("\n".join(["WAIT 1"] * 20))[0], ["WAIT 1"] * agent.MAX_LINES)
        self.assertEqual(parse_reply("")[0], [])
        self.assertEqual(parse_reply("\x00\x01 ??? \n\n")[0], [])
        self.assertEqual(parse_reply(None if False else 12345)[0], [])

    def test_grammar_is_well_formed(self):
        g = build_grammar(["cup", "table"], ["alice", "bob"])
        defined = set(re.findall(r"^(\w+)\s*::=", g, re.M))
        self.assertTrue({"root", "cmd", "hand", "prop", "person", "target", "text", "num"} <= defined)
        self.assertIn('"cup" | "table"', g)
        self.assertIn('"alice" | "bob"', g)
        for verb in agent.VERBS:
            self.assertIn(f'"{verb} "', g)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_scene_and_feedback(self):
        w = World(); a = w.add_character("Alice", "A cheerful baker.", 0, 0); b = w.add_character("Bob", "", 2, 0)
        w.add_prop("cup", (1, 0.9, 0), (0.08, 0.1, 0.08))
        w.step(1 / 60)
        for _ in range(2): w.step(1 / 60)
        w.build_snapshot()
        msgs = build_messages(w.snapshot, a, ["GRAB right unicorn rejected: unknown target 'unicorn'"], "give the cup to bob", [])
        text = msgs[-1]["content"]
        self.assertIn("cup (1.0 m, small)", text); self.assertIn("bob (2.0 m)", text)
        self.assertIn("unicorn", text); self.assertIn("give the cup to bob", text)
        self.assertIn("A cheerful baker.", msgs[0]["content"])
        w.close()


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.w = World(); self.w.start()
        self.a = self.w.call(lambda w: w.add_character("Alice", "", 0, 0))
        self.b = self.w.call(lambda w: w.add_character("Bob", "", 0, 3, math.pi))
        self.w.call(lambda w: w.add_prop("cup", (0.0, 0.1, 1.2), (0.08, 0.1, 0.08)))
        self.mgr = AgentManager(self.w, interval=0.2)

    def tearDown(self):
        self.mgr.close(); self.w.close()

    def test_model_output_becomes_motion(self):
        llm = Scripted("GRAB right cup\nSAY Got it")
        self.mgr.set_llm(llm); self.mgr.instruct(self.a, "pick up the cup")
        self.mgr.tick()
        self.assertTrue(wait_for(lambda: self.w.props["cup"].held_by == (self.a, "right")), "cup should end up in Alice's hand")
        self.assertTrue(any("Got it" in e["text"] for e in self.w.log))

    def test_hallucinated_names_are_rejected_then_fed_back(self):
        llm = Scripted("GRAB right unicorn\nWALK_TO narnia", "WAIT 1")
        self.mgr.set_llm(llm); self.mgr.instruct(self.a, "do something")
        self.mgr.tick()
        self.assertTrue(wait_for(lambda: len(llm.seen) >= 1))
        self.assertTrue(wait_for(lambda: sum(e["kind"] == "error" for e in self.w.log) >= 2))
        self.mgr.instruct(self.a, "try again")
        self.assertTrue(wait_for(lambda: (self.mgr.tick() or True) and len(llm.seen) >= 2))
        second = llm.seen[1][-1]["content"]
        self.assertIn("unicorn", second); self.assertIn("rejected", second)

    def test_garbage_and_failures_do_not_kill_anything(self):
        llm = Scripted("I think the weather is nice today.", LLMError("connection refused"), "WAIT 1")
        self.mgr.set_llm(llm)
        for _ in range(3):
            self.mgr.instruct(self.a, "hello")
            self.mgr.tick()
            self.assertTrue(wait_for(lambda: not self.mgr._st(self.a)["thinking"]))
            time.sleep(0.3)
        self.assertTrue(any("no valid command" in e["text"] for e in self.w.log))
        self.assertTrue(any("agent error" in e["text"] and "connection refused" in e["text"] for e in self.w.log))
        self.assertTrue(math.isfinite(self.w.snapshot["chars"][0]["pos"][0]))

    def test_autonomy_runs_only_when_enabled_and_idle(self):
        llm = Scripted(*["WAIT 1"] * 5); self.mgr.set_llm(llm)
        self.mgr.tick(); time.sleep(0.3)
        self.assertEqual(len(llm.seen), 0, "no autonomy unless switched on")
        self.w.call(lambda w: setattr(w.chars[self.a], "agent_on", True))
        time.sleep(0.2)
        for _ in range(10): self.mgr.tick(); time.sleep(0.15)
        self.assertGreaterEqual(len(llm.seen), 1)

    def test_mock_llm_handles_the_headline_scenario(self):
        self.mgr.set_llm(MockLLM())
        self.mgr.instruct(self.a, "give the cup to bob")
        self.mgr.tick()
        self.assertTrue(wait_for(lambda: self.w.props["cup"].held_by == (self.a, "right")))
        self.assertTrue(wait_for(lambda: self.w.chars[self.a].hands["right"].mode == "offer"
                                  and self.w.chars[self.a].current is None and not self.w.chars[self.a].queue, 40))
        self.assertTrue(any("Here you go" in e["text"] for e in self.w.log))


if __name__ == "__main__":
    unittest.main(verbosity=2)
