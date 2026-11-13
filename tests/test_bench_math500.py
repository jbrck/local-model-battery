"""The math500 leg on JsonStub: grading stays v1-faithful, counters live in Tally."""
import json
import os
import tempfile
import unittest
import unittest.mock

try:
    from scripts.bench_math500 import _extract_answer, _extract_boxed, _grade_answer
    from scripts.bench_math500 import main as bench_main
    from scripts.bench_math500 import run_math500
    from scripts.lib.harness import LegArgs, Tally
    from tests.stub_json import DieAfter, FakeAsk, completion
except ModuleNotFoundError as exc:
    # The leg grades with sympy, so it cannot be tested without it either. Skip with the
    # reason named rather than failing collection: requirements.txt pins sympy and
    # setup.sh installs it, so this only happens on an interpreter that skipped setup.
    raise unittest.SkipTest(f"the math500 leg needs sympy: {exc}") from exc


def write_items(path: str) -> list[dict]:
    """Three problems with known answers, saved as a bare JSON list."""
    items = [
        {"problem": "What is six times seven?", "answer": "42"},
        {"problem": "What is five plus five?", "answer": "10"},
        {"problem": "Name a famous transcendental constant.", "answer": r"\pi"},
    ]
    with open(path, "w") as handle:
        json.dump(items, handle)
    return items


class BoxExtractionTest(unittest.TestCase):
    def test_the_last_box_is_the_answer(self):
        self.assertEqual(_extract_boxed(r"The value is $\boxed{\frac{12}{7}}$."),
                         r"\frac{12}{7}")
        self.assertEqual(_extract_boxed(r"My working shows $\boxed{\sqrt[3]{27}}$."),
                         r"\sqrt[3]{27}")
        self.assertEqual(_extract_boxed(r"$\boxed{(a + b)^{2}}$"), r"(a + b)^{2}")

    def test_no_box_yields_none(self):
        self.assertIsNone(_extract_boxed("No box here at all."))

    def test_an_unclosed_box_yields_none(self):
        self.assertIsNone(_extract_boxed(r"$\boxed{{not closed$"))

    def test_extract_answer_falls_back_to_the_last_line(self):
        self.assertEqual(_extract_answer("Working through it.\nThe answer is 17."),
                         "17.")
        self.assertEqual(_extract_answer("So result:\n  23"), "23")


class GradeTest(unittest.TestCase):
    def test_a_textually_equal_answer_passes_without_sympy(self):
        self.assertTrue(_grade_answer("42", r"The answer is $\boxed{42}$."))
        self.assertTrue(_grade_answer("the set of all primes", "THE SET OF ALL PRIMES"))

    def test_sympy_equal_but_not_textually_equal_passes(self):
        self.assertTrue(_grade_answer("2 + 2", "4"))
        self.assertTrue(_grade_answer("x^2 - 1", "(x - 1)*(x + 1)"))
        self.assertTrue(_grade_answer("(x - 1)*(x + 1)", "x^2 - 1"))

    def test_a_wrong_answer_fails(self):
        self.assertFalse(_grade_answer(r"\frac{5}{4}", r"\frac{9}{8}"))
        self.assertFalse(_grade_answer("10", r"The answer is $\boxed{\pi}$."))

    def test_non_math_text_is_compared_case_insensitively(self):
        # Neither side parses as math; the fallback string compare decides.
        self.assertTrue(_grade_answer("the set of all primes", "THE SET OF ALL PRIMES"))
        self.assertFalse(
            _grade_answer("Infinite Set of Primes", "Infinite Set of Primes."))

    def test_empty_sides_fail(self):
        self.assertFalse(_grade_answer("42", ""))
        self.assertFalse(_grade_answer("", r"The answer is $\boxed{\pi}$."))


class RunTest(unittest.TestCase):
    """End to end through JsonStub: the counters and the block they produce."""

    def setUp(self) -> None:
        from tests.stub_json import JsonStub

        def routes(body: dict) -> tuple[int, dict]:
            from tests.stub_json import completion

            problem = body["messages"][1]["content"]
            if problem.startswith("What is six times seven"):
                return 200, completion(r"I get $\boxed{42}$.")
            if problem.startswith("What is five plus five"):
                return 200, completion(r"I get $\boxed{e}$.", finish_reason="length",
                                       tokens=9000)
            return 200, completion(r"I get $\boxed{\pi}$.")

        self.stub = JsonStub(routes)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)

    @staticmethod
    def args(**overrides) -> LegArgs:
        base = {"model": "m"}
        base.update(overrides)
        return LegArgs(**base)

    @staticmethod
    def item_path() -> str:
        path = os.path.join(tempfile.mkdtemp(), "items.json")
        write_items(path)
        return path

    def run_once(self) -> dict:
        data = self.item_path()
        # A fresh output dir per run: the leg checkpoints its cases, and one test's
        # leftovers must never be reused as another test's answers.
        args = self.args(endpoint=self.stub.endpoint, data=data,
                         output_dir=tempfile.mkdtemp())
        tally = run_math500(args)
        return tally.to_result("math500", args)

    def test_counters_reflect_the_stubbed_responses(self) -> None:
        block = self.run_once()
        # First and third items are answered correctly out of three asked.
        self.assertEqual(block["items"], 3)
        self.assertEqual(block["scored"], 3)
        # The length stop with real tokens behind it counts as a cap hit.
        self.assertEqual(block["correct"], 2)
        self.assertEqual(block["cap_hits"], 1)
        self.assertEqual(block["runaway"], 0)
        self.assertEqual(block["timeout_hits"], 0)
        self.assertEqual(block["failed"], 0)
        self.assertEqual(block["status"], "pass")
        self.assertAlmostEqual(block["score_pct"], 66.7, delta=0.1)
        self.assertGreater(block["tps"], 0.0)
        # The run made exactly one request per item, all to the stub.
        self.assertEqual(len(self.stub.seen), 3)

    def test_the_block_carries_the_settings_it_ran_under(self) -> None:
        block = self.run_once()
        for key in ("leg", "model", "status", "items", "scored", "correct",
                    "score_pct", "tps", "cap_hits", "runaway", "timeout_hits",
                    "failed", "tokens", "elapsed_s", "cap", "ctx", "thinking",
                    "seed", "temperature"):
            self.assertIn(key, block)
        self.assertEqual(block["leg"], "math500")
        self.assertEqual(block["model"], "m")
        # Optional parts stay absent when there is nothing to report.
        self.assertNotIn("errors", block)
        self.assertNotIn("notes", block)

    def test_main_writes_the_result_file_and_honors_the_flags(self) -> None:
        out_dir = os.path.join(tempfile.mkdtemp(), "results")
        data = self.item_path()
        bench_main(["--model", "m",
                    "--endpoint", self.stub.endpoint,
                    "--api-key", "k",
                    "--data", data,
                    "--output-dir", out_dir,
                    "--cap", "512",
                    "--ctx", "4096",
                    "--thinking", "off",
                    "--seed", "7"])
        with open(os.path.join(out_dir, "math500.result.json")) as handle:
            block = json.load(handle)
            for key in ("leg", "model", "status", "items", "scored", "correct",
                        "score_pct", "tps", "cap_hits"):
                assert key in block
            assert block["leg"] == "math500"
            assert block["cap"] == 512
            assert block["ctx"] == 4096
            assert block["thinking"] == "off"
            assert block["seed"] == 7
            assert block["temperature"] == 0.0
            assert block["status"] == "pass"
            assert block["correct"] == 2
        # The flags reached the wire: cap, thinking switch, and model name.
        sent = self.stub.seen[-1]["body"]
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["temperature"], 0.0)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["model"], "m")

    def test_the_items_file_is_read_from_data(self) -> None:
        data = os.path.join(tempfile.mkdtemp(), "custom.json")
        custom = [{"problem": "Is the sky blue?", "answer": "yes"}]
        with open(data, "w") as handle:
            json.dump(custom, handle)
        args = self.args(endpoint=self.stub.endpoint, data=data,
                         output_dir=tempfile.mkdtemp())
        tally = run_math500(args)
        # The problem text in the request body came from --data, not the repo file.
        sent_problem = self.stub.seen[0]["body"]["messages"][1]["content"]
        self.assertEqual(sent_problem, "Is the sky blue?")
        self.assertEqual(tally.items, 1)


class CheckpointTest(unittest.TestCase):
    """A killed run keeps its finished cases; the resume generates only the rest."""

    ANSWERS = {
        "What is two plus two?": r"$\boxed{4}$",
        "What is three plus three?": r"$\boxed{6}$",
    }

    def setUp(self) -> None:
        self.fake = FakeAsk(self.ANSWERS)
        self.out_dir = tempfile.mkdtemp()
        self.data = os.path.join(tempfile.mkdtemp(), "items.json")
        with open(self.data, "w") as handle:
            json.dump([{"problem": problem, "answer": answer}
                       for problem, answer in self.ANSWERS.items()], handle)

    def args(self) -> LegArgs:
        return LegArgs(model="m", data=self.data, output_dir=self.out_dir)

    def cases_dir(self) -> str:
        return os.path.join(self.out_dir, "cases", "math500")

    def run_killed(self) -> None:
        with unittest.mock.patch("scripts.bench_math500.ask",
                                 DieAfter(self.fake, die_on=1)):
            with self.assertRaises(KeyboardInterrupt):
                run_math500(self.args())

    def run_resumed(self) -> Tally:
        with unittest.mock.patch("scripts.bench_math500.ask", self.fake):
            return run_math500(self.args())

    def test_a_kill_after_the_first_case_leaves_exactly_one_on_disk(self) -> None:
        self.run_killed()
        files = sorted(os.listdir(self.cases_dir()))
        self.assertEqual(files, ["00000.json"])

    def test_a_resume_generates_only_the_missing_case_and_matches_the_full_run(self) -> None:
        full = self.run_resumed()
        self.assertEqual(len(self.fake.calls), 2)

        # Fresh output dir: one case lands, the run dies, then the resume finishes it.
        self.setUp()
        self.run_killed()
        self.fake.calls.clear()
        resumed = self.run_resumed()
        self.assertEqual(self.fake.calls, ["What is three plus three?"])
        self.assertEqual(sorted(os.listdir(self.cases_dir())),
                         ["00000.json", "00001.json"])
        for key in ("items", "scored", "correct", "score_pct", "tps", "cap_hits",
                    "runaway", "timeout_hits", "failed", "tokens", "elapsed_s"):
            self.assertEqual(tally_value(resumed, key), tally_value(full, key), key)

    def test_a_reused_case_is_not_regraded_or_asked_again(self) -> None:
        self.run_killed()
        with open(os.path.join(self.cases_dir(), "00000.json")) as handle:
            saved = json.load(handle)
        saved["correct"] = not saved["correct"]  # a tampered grade must survive replay
        with open(os.path.join(self.cases_dir(), "00000.json"), "w") as handle:
            json.dump(saved, handle)
        self.fake.calls.clear()
        tally = self.run_resumed()
        self.assertEqual(tally.items, 2)
        # Only the second case was asked; the first came back from disk, grade included.
        self.assertEqual(self.fake.calls, ["What is three plus three?"])


def tally_value(tally: Tally, key: str):
    block = tally.to_result("math500", LegArgs(model="m"))
    return block[key]


if __name__ == "__main__":
    unittest.main()
