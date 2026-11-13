"""The humaneval_plus leg on JsonStub: v1 grading, exec timeouts counted."""
import json
import os
import tempfile
import unittest
import unittest.mock

from scripts.bench_humaneval_plus import candidate_source, strip_fences
from scripts.bench_humaneval_plus import main as bench_main
from scripts.bench_humaneval_plus import _replay_item, run_humaneval
from scripts.lib.harness import LegArgs, Tally, case_record
from scripts.lib.model import ChatResponse
from tests.stub_json import DieAfter, FakeAsk

PROMPT = "def add_two(a, b):\n    \"\"\"Return the sum.\"\"\"\n"
TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


def write_items(path: str) -> None:
    """One correct, one wrong and one non-terminating solution to grade."""
    items = [
        {"task_id": "ok/1", "prompt": PROMPT, "entry_point": "add_two",
         "test": TEST, "imports": ""},
        {"task_id": "wrong/1", "prompt": PROMPT, "entry_point": "add_two",
         "test": TEST, "imports": ""},
        {"task_id": "loop/1", "prompt": PROMPT, "entry_point": "add_two",
         "test": TEST, "imports": ""},
    ]
    with open(path, "w") as handle:
        json.dump(items, handle)


class StripFencesTest(unittest.TestCase):
    def test_a_fenced_block_is_stripped(self):
        self.assertEqual(strip_fences("```python\n    return a + b\n```"),
                         "    return a + b")

    def test_bare_code_passes_through(self):
        self.assertEqual(strip_fences("    return a + b"), "    return a + b")

    def test_blank_edges_are_trimmed(self):
        self.assertEqual(strip_fences("\n\n    return 0\n\n"), "    return 0")


class CandidateSourceTest(unittest.TestCase):
    def test_imports_prompt_and_completion_join(self):
        source = candidate_source("import math", PROMPT, "    return math.inf")
        self.assertEqual(source.splitlines()[0], "import math")
        self.assertIn("def add_two", source)

    def test_an_empty_imports_line_is_dropped(self):
        source = candidate_source("", PROMPT, "    return 0")
        self.assertTrue(source.startswith("def add_two"))


class RunTest(unittest.TestCase):
    """End to end through JsonStub: pass, fail and timeout all land in the block."""

    def setUp(self) -> None:
        from tests.stub_json import completion, JsonStub

        answers = [
            "```python\n    return a + b\n```",
            "    return a - b",
            "    while True:\n        pass",
        ]

        def routes(body: dict) -> tuple[int, dict]:
            # Requests arrive in item order; seen already holds this request.
            index = max(0, len(self.stub.seen) - 1)
            return 200, completion(answers[index], tokens=50)

        self.stub = JsonStub(routes)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)

    def run_once(self, **overrides) -> tuple[Tally, dict]:
        data = overrides.pop("data", self.item_path())
        # A fresh output dir per run: the leg checkpoints its cases, and one test's
        # leftovers must never be reused as another test's answers.
        args = LegArgs(model="m", endpoint=self.stub.endpoint, data=data,
                       output_dir=tempfile.mkdtemp(), **overrides)
        tally = run_humaneval(args)
        return tally, tally.to_result("humaneval_plus", args)

    @staticmethod
    def item_path() -> str:
        path = os.path.join(tempfile.mkdtemp(), "items.json")
        write_items(path)
        return path

    def test_pass_fail_and_timeout_are_all_counted(self) -> None:
        tally, block = self.run_once()
        self.assertEqual(block["items"], 3)
        self.assertEqual(block["scored"], 3)
        self.assertEqual(block["correct"], 1)
        self.assertEqual(block["timeout_hits"], 1)
        self.assertEqual(block["cap_hits"], 0)
        self.assertEqual(block["runaway"], 0)
        self.assertEqual(block["failed"], 0)
        self.assertEqual(block["status"], "timeout")
        self.assertAlmostEqual(block["score_pct"], 33.3, delta=0.1)
        self.assertTrue(any("loop/1" in note for note in block["notes"]))

    def test_the_block_carries_the_contract_keys(self) -> None:
        _, block = self.run_once()
        for key in ("leg", "model", "status", "items", "scored", "correct",
                    "score_pct", "tps", "cap_hits", "runaway", "timeout_hits",
                    "failed", "tokens", "elapsed_s", "cap", "ctx", "thinking",
                    "seed", "temperature"):
            self.assertIn(key, block)
        self.assertEqual(block["leg"], "humaneval_plus")

    def test_main_writes_the_result_file_and_sends_the_leg_flags(self) -> None:
        out_dir = os.path.join(tempfile.mkdtemp(), "results")
        bench_main(["--model", "m", "--endpoint", self.stub.endpoint,
                    "--data", self.item_path(), "--output-dir", out_dir,
                    "--cap", "1024", "--thinking", "off", "--seed", "9"])
        with open(os.path.join(out_dir, "humaneval_plus.result.json")) as handle:
            block = json.load(handle)
        self.assertEqual(block["leg"], "humaneval_plus")
        self.assertEqual(block["cap"], 1024)
        self.assertEqual(block["thinking"], "off")
        self.assertEqual(block["seed"], 9)
        sent = self.stub.seen[-1]["body"]
        self.assertEqual(sent["max_tokens"], 1024)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["messages"][1]["content"], PROMPT)


class CheckpointTest(unittest.TestCase):
    """A killed run keeps its finished cases; the resume generates only the rest."""

    TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    PROMPTS = (
        'def add_two(a, b):\n    """Variant one."""\n',
        'def add_two(a, b):\n    """Variant two."""\n',
    )
    ANSWERS = {PROMPTS[0]: "```python\n    return a + b\n```",
               PROMPTS[1]: "```python\n    return a - b\n```"}

    def setUp(self) -> None:
        self.fake = FakeAsk(self.ANSWERS)
        self.tmpdir = tempfile.mkdtemp()
        self.data = os.path.join(self.tmpdir, "items.json")
        with open(self.data, "w") as handle:
            json.dump([{"task_id": f"case/{index}", "prompt": prompt,
                        "entry_point": "add_two", "test": self.TEST, "imports": ""}
                       for index, prompt in enumerate(self.PROMPTS)], handle)
        self.out_dir = os.path.join(self.tmpdir, "results")

    def args(self) -> LegArgs:
        return LegArgs(model="m", data=self.data, output_dir=self.out_dir)

    def cases_dir(self) -> str:
        return os.path.join(self.out_dir, "cases", "humaneval_plus")

    def run_leg_with(self, ask: FakeAsk | DieAfter) -> Tally:
        with unittest.mock.patch("scripts.bench_humaneval_plus.ask", ask):
            return run_humaneval(self.args())

    def test_a_kill_after_the_first_case_leaves_exactly_one_on_disk(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.run_leg_with(DieAfter(self.fake, die_on=1))
        self.assertEqual(sorted(os.listdir(self.cases_dir())), ["00000.json"])

    def test_a_resume_generates_only_the_missing_case_and_matches_the_full_run(self) -> None:
        args = self.args()
        full_block = self.run_leg_with(self.fake).to_result("humaneval_plus", args)
        self.assertEqual(len(self.fake.calls), 2)

        # Fresh directory: one case lands, the run dies, the resume finishes the leg.
        self.setUp()
        with self.assertRaises(KeyboardInterrupt):
            self.run_leg_with(DieAfter(self.fake, die_on=1))
        self.fake.calls.clear()
        args = self.args()
        resumed_block = self.run_leg_with(self.fake).to_result("humaneval_plus", args)
        self.assertEqual(self.fake.calls, [self.PROMPTS[1]])
        self.assertEqual(sorted(os.listdir(self.cases_dir())),
                         ["00000.json", "00001.json"])
        for key in ("items", "scored", "correct", "score_pct", "tps", "cap_hits",
                    "runaway", "timeout_hits", "failed", "tokens", "elapsed_s",
                    "status"):
            self.assertEqual(resumed_block[key], full_block[key], key)


class ReplayItemTest(unittest.TestCase):
    """A checkpointed case is counted again on replay, exactly as it first landed."""

    def saved_case(self, **extra) -> dict:
        """A checkpoint in the shape run_humaneval writes, with any extras applied."""
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                   "usage": {"completion_tokens": 50}}
        response = ChatResponse(status=200, payload=payload, elapsed_s=1.0)
        return case_record("00000", response, correct=False, **extra)

    def test_a_plain_checkpoint_only_counts_itself(self) -> None:
        tally = Tally()
        _replay_item(tally, self.saved_case(), total=1, case_id="loop/1")
        self.assertEqual(tally.items, 1)
        self.assertEqual(tally.scored, 1)
        self.assertEqual(tally.timeout_hits, 0)
        self.assertEqual(tally.notes, [])

    def test_a_checkpointed_exec_timeout_keeps_its_count_and_note(self) -> None:
        saved = self.saved_case(exec_timeout=True,
                                note="loop/1: execution timeout after 10s")
        tally = Tally()
        _replay_item(tally, saved, total=1, case_id="loop/1")
        self.assertEqual(tally.timeout_hits, 1)
        self.assertEqual(tally.notes, ["loop/1: execution timeout after 10s"])

    def test_a_checkpoint_without_a_note_still_counts_the_timeout(self) -> None:
        tally = Tally()
        _replay_item(tally, self.saved_case(exec_timeout=True), total=1, case_id="loop/1")
        self.assertEqual(tally.timeout_hits, 1)
        self.assertEqual(tally.notes, [""])


class TransportFailureTest(unittest.TestCase):
    """A leg keeps counting when the endpoint answers garbage."""

    def setUp(self) -> None:
        from tests.stub_json import JsonStub

        self.stub = JsonStub({}, default=(500, {"error": "boom"}))
        self.stub._thread.start()
        self.addCleanup(self.stub.close)

    def test_a_failed_request_is_counted_not_scored(self) -> None:
        path = RunTest.item_path()
        args = LegArgs(model="m", endpoint=self.stub.endpoint, data=path,
                       output_dir=tempfile.mkdtemp(), limit=1)
        tally = run_humaneval(args)
        block = tally.to_result("humaneval_plus", args)
        self.assertEqual(block["items"], 1)
        self.assertEqual(block["scored"], 0)
        self.assertEqual(block["failed"], 1)
        self.assertEqual(block["correct"], 0)
        self.assertEqual(block["status"], "fail")
        self.assertTrue(block["errors"])


if __name__ == "__main__":
    unittest.main()
