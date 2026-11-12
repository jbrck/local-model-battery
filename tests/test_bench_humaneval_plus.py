"""The humaneval_plus leg on JsonStub: v1 grading, exec timeouts counted."""
import json
import os
import tempfile
import unittest

from scripts.bench_humaneval_plus import candidate_source, strip_fences
from scripts.bench_humaneval_plus import main as bench_main
from scripts.bench_humaneval_plus import run_humaneval
from scripts.lib.harness import LegArgs, Tally

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
        args = LegArgs(model="m", endpoint=self.stub.endpoint, data=data, **overrides)
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
                       limit=1)
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
