"""S3 greedy-equivalence checks (criteria: proj v1.11 §4.5; analysis/judge_greedy.py).

1. The judgment logic on synthetic runs.
2. The recorded judgment of the latest scripts/run_s3.sh matrix (results/summary/s3_greedy_equivalence.json):
   F16 targets and the CPU n_max = 1 control identical except for documented exceptions
   (results/summary/s3_known_exceptions.json), quantized targets diverging only at near-ties.

run: python -m unittest discover -s tests -v   (or: python tests/test_greedy_correctness.py, or pytest)
"""
import importlib.util, json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("judge_greedy", f"{ROOT}/analysis/judge_greedy.py")
jg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jg)


def tok(token, *tops):
    """A tokens.jsonl entry: [token, logprob, [[top token, logprob], ...]]."""
    return [token, tops[0][1] if tops else 0.0, [list(t) for t in tops]]


class JudgeLogic(unittest.TestCase):
    ref = ({"p": "a b c"}, {"p": [tok("a", ("a", -0.1), ("x", -3.0)), tok(" b", (" b", -0.5), (" y", -0.6)),
                                   tok(" c", (" c", -0.2), (" z", -2.0))]})
    tie = ({"p": "a y c"}, {"p": [tok("a", ("a", -0.1)), tok(" y", (" y", -0.5)), tok(" c", (" c", -0.1))]})

    def test_identical(self):
        r = jg.judge_pair(self.ref, ({"p": "a b c"}, self.ref[1]), exact=True)
        self.assertTrue(r["strict_pass"]); self.assertEqual(r["identical"], 1)

    def test_near_tie_allowed_for_quantized_only(self):
        q = jg.judge_pair(self.ref, self.tie, exact=False)
        self.assertTrue(q["pass"]); self.assertEqual(q["near_tie"], 1); self.assertAlmostEqual(q["margin_max"], 0.1)
        self.assertFalse(jg.judge_pair(self.ref, self.tie, exact=True)["pass"])

    def test_known_exception_passes_but_not_strictly(self):
        r = jg.judge_pair(self.ref, self.tie, exact=True, exceptions={"p": 0.2})
        self.assertTrue(r["pass"]); self.assertFalse(r["strict_pass"]); self.assertEqual(len(r["known_exceptions"]), 1)

    def test_known_exception_does_not_mask_a_larger_margin(self):
        self.assertFalse(jg.judge_pair(self.ref, self.tie, exact=True, exceptions={"p": 0.05})["pass"])

    def test_large_margin_fails(self):
        r = jg.judge_pair(self.ref, ({"p": "x"}, {"p": [tok("x", ("x", -0.1))]}), exact=False)
        self.assertFalse(r["pass"]); self.assertEqual(r["large_margin"], 1)

    def test_same_prefix_different_length_is_not_identical(self):
        r = jg.judge_pair(self.ref, ({"p": "a b"}, {"p": self.ref[1]["p"][:2]}), exact=False)
        self.assertFalse(r["pass"]); self.assertEqual(r["unlocated"], 1)

    def test_missing_prompt_fails(self):
        r = jg.judge_pair(self.ref, ({}, {}), exact=False)
        self.assertFalse(r["pass"]); self.assertEqual(r["missing"], 1)

    def test_turns_ending_at_different_places_fail(self):
        # two-turn sample joined the way load_eval_run joins it: the reference ends turn 1 one token earlier
        T = jg.TURN
        ref = ({"p": "a\n<<turn>>\nc"}, {"p": [tok("a", ("a", -0.1), ("b", -2.0)), T, tok("c", ("c", -0.1))]})
        spec = ({"p": "a b\n<<turn>>\nc"}, {"p": [tok("a", ("a", -0.1)), tok(" b", (" b", -0.1)), T,
                                                   tok("c", ("c", -0.1))]})
        r = jg.judge_pair(ref, spec, exact=False)
        self.assertFalse(r["pass"]); self.assertEqual(r["unlocated"], 1)

    def test_criteria(self):
        self.assertTrue(jg.criterion("s3", "F16", 8).startswith("exact"))
        self.assertTrue(jg.criterion("s3cpu", "Q4_K_M", 1).startswith("exact"))
        self.assertTrue(jg.criterion("s3cpu", "Q4_K_M", 8).startswith("near-tie"))
        self.assertTrue(jg.criterion("s3", "Q4_K_M", 1).startswith("near-tie"))


class RecordedMatrix(unittest.TestCase):
    path = f"{ROOT}/results/summary/s3_greedy_equivalence.json"

    @unittest.skipUnless(os.path.exists(path), "run scripts/run_s3.sh first")
    def test_recorded_matrix(self):
        with open(self.path) as f:
            rep = json.load(f)
        self.assertTrue(rep["runs"])
        for r in rep["runs"]:
            with self.subTest(group=r["group"], target=r["target"], drafter=r["drafter"], n_max=r["n_max"]):
                self.assertTrue(r["pass"], r["failures"])
                if r["criterion"].startswith("exact"):
                    self.assertEqual(r["identical"] + len(r["known_exceptions"]), r["prompts"])
                else:
                    self.assertTrue(r["margin_max"] is None or r["margin_max"] < rep["margin_threshold"])


if __name__ == "__main__":
    unittest.main()
