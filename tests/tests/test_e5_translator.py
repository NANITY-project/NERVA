#!/usr/bin/env python3
"""
tests/test_e5_translator.py - the E5 Goal Translator.

PART A (numpy only, runs anywhere): tokenizer rules, targets <-> decoding round trip, the numpy runtime (shape, determinism, text sensitivity, .npz save/load) and the whole path
through UI/goal_plan.GoalPlanner.

PART B (needs torch + transformers + the Hugging Face hub for the encoder; skipped otherwise): the parity checks that make the numpy runtime trustworthy:
  B1 our WordPiece == HF's BertTokenizer on the same vocabulary (incl. run-together text, accents, punctuation, emoji)
  B2 numpy BERT == HF BertModel (last hidden state)
  B3 numpy runtime == the torch model (all four outputs) with a random decoder head
  B4 --slow only: a few hundred steps on a tiny synthetic dataset must cut the training loss clearly (the loss, the data path and the optimiser are wired up)

  python tests/test_e5_translator.py [--slow] [--encoder intfloat/e5-small-v2]
"""
import argparse, os, random, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "goal_translator"))
sys.path.insert(0, os.path.join(HERE, "..", "UI"))

import gt_conventions as C      # noqa: E402
import e5_common as E           # noqa: E402
import e5_numpy as N            # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def part_a():
    print("[A] tokenizer")
    check("lowercase, punctuation split, accents stripped", E.basic_tokenize("Walk-RIGHT, then Left! Café") == ["walk", "-", "right", ",", "then", "left", "!", "cafe"])
    check("control characters dropped, whitespace normalised", E.basic_tokenize("a\x00b\t\nc\u00a0d") == ["ab", "c", "d"])
    check("CJK characters become separate tokens", E.basic_tokenize("walk\u8d70go") == ["walk", "\u8d70", "go"])
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "query", ":", "walk", "right", "left", "##right", "##left", "then", "##ing", "turn", "##s"]
    wp = E.WordPiece(vocab)
    ids = wp.encode("walking")
    check("greedy longest-match WordPiece with ## pieces", [vocab[i] for i in ids] == ["[CLS]", "query", ":", "walk", "##ing", "[SEP]"], str([vocab[i] for i in ids]))
    ids = wp.encode("walkright")
    check("run-together text is split by the vocabulary, no word list", [vocab[i] for i in ids][3:-1] == ["walk", "##right"], str([vocab[i] for i in ids]))
    check("an unknown word becomes [UNK] as a whole", vocab[wp.encode("zzz")[3]] == "[UNK]")
    try:
        wp.encode("walk " * 200)
        check("too long text is rejected at inference", False)
    except C.PlanError:
        check("too long text is rejected at inference", True)
    t = wp.encode("walk " * 200, truncate=True)
    check("... and truncated only when asked (training)", len(t) == E.MAX_TOKENS and t[-1] == wp.sep and t[0] == wp.cls)
    check("the prefix is part of every input", [vocab[i] for i in wp.encode("left")][1:3] == ["query", ":"])

    print("[A] targets and decoding")
    rng = random.Random(0)
    bad = 0
    st = C.Stats()
    for _ in range(400):
        n = rng.randint(1, E.W_MAX)
        wins = [None if rng.random() < 0.3 else [rng.randrange(C.N_TURN_BINS), rng.randrange(C.N_SPEED_BINS)] for _ in range(n)]
        nn_, spec, turn, speed = E.row_targets(wins)
        ll = np.full(E.W_MAX, -9.0); ll[n - 1] = 9.0
        plan = E.decode_outputs(ll, np.where(spec > 0.5, 9.0, -9.0), turn, speed, st)
        got = [None if w.unspecified else [w.turn_bin, w.speed_bin] for w in plan.windows]
        bad += got != wins
    check("perfect outputs decode to exactly the training bins (400 random plans)", bad == 0, f"{bad} differ")
    try:
        E.row_targets([])
        check("an empty plan is not a valid row", False)
    except ValueError:
        check("an empty plan is not a valid row", True)
    ll = np.zeros(E.W_MAX); ll[4] = 5.0
    p = E.decode_outputs(ll, np.array([5.0, -5.0, 5.0, -5.0, -5.0] + [0.0] * (E.W_MAX - 5)), np.array([0.5, 0.0, -0.5, 0.0, 0.0] + [0.0] * (E.W_MAX - 5)), np.full(E.W_MAX, 0.5), st)
    check("length = argmax + 1, spec threshold decides UNSPEC", len(p) == 5 and [w.unspecified for w in p.windows] == [False, True, False, True, True])
    check("sign convention: positive turn output -> positive turn bin (same as gt_conventions)", C.bin_to_turn(p.windows[0].turn_bin, st) > 0 > C.bin_to_turn(p.windows[2].turn_bin, st))

    print("[A] numpy runtime")
    t = N.random_translator(0)
    a, b = t.raw("walk right then left"), t.raw("walk right then left")
    check("deterministic", all(np.array_equal(a[k], b[k]) for k in a))
    c = t.raw("turn left")
    check("the output depends on the text", not np.allclose(a["turn"], c["turn"]))
    check("output shapes", a["len_logits"].shape == (E.W_MAX,) and a["spec_logit"].shape == (E.W_MAX,) and a["turn"].shape == (E.W_MAX,) and a["speed"].shape == (E.W_MAX,))
    check("turn in [-1, 1], speed in [0, 1]", float(np.abs(a["turn"]).max()) <= 1.0 and 0.0 <= float(a["speed"].min()) and float(a["speed"].max()) <= 1.0)
    plan = t.translate("walk right then left")
    check("translate() returns a Plan whose windows are valid", isinstance(plan, C.Plan) and 1 <= len(plan) <= E.W_MAX and plan.source == "model")
    try:
        t.translate("walk " * 200)
        check("translate() rejects over-long text with PlanError", False)
    except C.PlanError:
        check("translate() rejects over-long text with PlanError", True)
    path = os.path.join(tempfile.mkdtemp(), "e5gt.npz")
    t.save_npz(path)
    t2 = N.E5NumpyTranslator.from_npz(path)
    a2 = t2.raw("walk right then left")
    check(".npz round trip gives identical outputs", all(np.array_equal(a[k], a2[k]) for k in a))

    print("[A] through UI/goal_plan.GoalPlanner")
    import goal_plan as GP
    gp = GP.GoalPlanner(path)
    info = gp.info()
    check("planner reports the e5 backend", info["source"] == "model" and info["backend"] == "e5", str(info))
    r = gp.plan("walk right then left", dict(turn_max=0.9, speed_cruise=1.2, speed_max=2.1), C.DEFAULT_WINDOW["W"], C.DEFAULT_WINDOW["stride"], C.DEFAULT_WINDOW["dt"])
    ok_rows = all(("unspecified" in x) or (abs(x["turn"]) <= 51.9 and 0 <= x["speed"] <= 2.1) for x in r["rows"])
    check("plan rows are clamped to the prior's data range (degrees, m/s)", r["source"] == "model" and r["windows"] == len(r["rows"]) and ok_rows)
    try:
        GP.GoalPlanner(os.path.join(tempfile.mkdtemp(), "missing.npz"))
        check("a missing translator file is an error, not a silent fallback", False)
    except GP.GoalPlanError:
        check("a missing translator file is an error, not a silent fallback", True)


def part_b(a):
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"[B] skipped: {type(e).__name__} (torch and transformers are needed)")
        return
    import e5_torch as L
    print("[B] loading the encoder", a.encoder)
    try:
        hf = AutoTokenizer.from_pretrained(a.encoder)
        model = L.E5GoalTranslator.from_pretrained(a.encoder, "tiny", dropout=0.0).eval()
    except Exception as e:
        print(f"[B] skipped: could not load {a.encoder} ({type(e).__name__}: {str(e)[:100]})")
        return
    vocab = [t for t, _ in sorted(hf.get_vocab().items(), key=lambda kv: kv[1])]
    wp = E.WordPiece(vocab)
    prompts = ["Walkrightandthenleftthenwaveurhandandidkmovearound?", "walk right and then left", "Turn LEFT, then stop.", "veer to the right a bit and mosey on",
               "café au lait \u00e9\u00e8\u00ea", "go go go!!! \U0001F600", "w a l k  r i g h t", "run straight for 10 seconds then idk", "\u8d70\u8def go", "it's 90\u00b0 -> left;right",
               "spin around and walkbackwards", "x" * 90]
    print("[B1] tokenizer parity with HF")
    bad = []
    for p in prompts:
        mine = wp.encode(p, truncate=True, max_len=512)
        ref = hf(E.PREFIX + p)["input_ids"]
        if mine != ref:
            bad.append((p, [vocab[i] for i in mine][:12], [vocab[i] for i in ref][:12]))
    check(f"our WordPiece == HF on {len(prompts)} prompts", not bad, "; ".join(f"{p!r}: ours {m} vs hf {r}" for p, m, r in bad[:2]))

    print("[B2] numpy BERT == HF BertModel")
    weights, meta, voc = L.export_arrays(model, vocab, C.Stats(), C.Timing(), a.encoder)
    tr = N.E5NumpyTranslator(weights, meta, voc)
    worst = 0.0
    with torch.no_grad():
        for p in prompts[:6]:
            ids = wp.encode(p, truncate=True)
            h_ref = model.encoder(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0].numpy()
            worst = max(worst, float(np.abs(tr.encode(ids) - h_ref).max()))
    check("encoder outputs agree (max abs diff < 2e-3)", worst < 2e-3, f"worst {worst:.2e}")

    print("[B3] numpy runtime == torch model (random decoder head)")
    torch.manual_seed(0)
    with torch.no_grad():
        for prm in model.head.parameters():
            prm.add_(torch.randn_like(prm) * 0.05)                    # a non-trivial head, so the comparison is not about zeros
    weights, meta, voc = L.export_arrays(model, vocab, C.Stats(), C.Timing(), a.encoder)
    tr = N.E5NumpyTranslator(weights, meta, voc)
    worst = dict(len_logits=0.0, spec_logit=0.0, turn=0.0, speed=0.0)
    with torch.no_grad():
        for p in prompts[:8]:
            ids = wp.encode(p, truncate=True)
            o = model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long))
            r = tr.decode_head(tr.encode(ids))
            for k in worst:
                worst[k] = max(worst[k], float(np.abs(o[k][0].numpy() - r[k]).max()))
    check("all four outputs agree (logits < 2e-3, turn/speed < 1e-3)", worst["len_logits"] < 2e-3 and worst["spec_logit"] < 2e-3 and worst["turn"] < 1e-3 and worst["speed"] < 1e-3,
          str({k: f"{v:.1e}" for k, v in worst.items()}))

    if a.slow:
        print("[B4] a few hundred steps on tiny synthetic data must cut the loss")
        tmp = tempfile.mkdtemp()
        stats = os.path.join(tmp, "stats.json")
        open(stats, "w").write('{"turn_max":0.9,"speed_cruise":1.2,"speed_max":2.1}')
        gt = os.path.join(HERE, "..", "tools", "goal_translator")
        subprocess.run([sys.executable, os.path.join(gt, "build_dataset.py"), "--synthetic", "60", "--out", tmp, "--stats", stats, "--timing", "64,48,0.0333"], check=True,
                       stdout=subprocess.DEVNULL)
        rows = E.load_rows(os.path.join(tmp, "train.jsonl"))[:96]
        m2 = L.E5GoalTranslator.from_pretrained(a.encoder, "tiny", dropout=0.0)
        opt = torch.optim.AdamW(m2.parameters(), lr=1e-4)
        dl = torch.utils.data.DataLoader(L.E5Dataset(rows, augment=False), batch_size=16, shuffle=True, collate_fn=L.make_collate(wp))
        first = last = None
        m2.train()
        for ep in range(25):
            tot = 0.0
            for ids, mask, tg in dl:
                loss, _ = L.loss_fn(m2(ids, mask), tg)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            first = tot if first is None else first
            last = tot
        check("training loss falls by more than 40%", last < 0.6 * first, f"{first:.2f} -> {last:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--encoder", default="intfloat/e5-small-v2")
    a = ap.parse_args()
    part_a()
    part_b(a)
    if FAILS:
        print(f"\n{len(FAILS)} check(s) FAILED")
        sys.exit(1)
    print("\nall E5 translator checks passed")


if __name__ == "__main__":
    main()
