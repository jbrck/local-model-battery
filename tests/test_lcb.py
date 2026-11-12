"""The LiveCodeBench adapter: both phases, resume, the guard and the result block."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from scripts.adapters import lcb, lcb_worker
from scripts.adapters.lcb_guard import FailureKind, Guard
from tests.stub_json import JsonStub, completion

PROBLEMS = [{"question_id": "q1", "question_content": "sum two numbers"},
            {"question_id": "q2", "question_content": "sort a list"}]

FAKE_SCORER = """#!/usr/bin/env python3
import json, sys
payload = json.load(open(sys.argv[2]))
detail = {str(i): 1.0 if "PASS_ME" in gen[0] else 0.0
          for i, gen in enumerate(payload["generations"])}
json.dump({"pass@1": sum(detail.values()) / len(detail), "detail": detail},
          open(sys.argv[3], "w"))
"""


def fake_extract(content: str) -> str:
    return content[5:].strip() if content.startswith("CODE:") else ""


def fake_helpers() -> lcb.Helpers:
    return lcb.Helpers(format_prompt=lambda problem: [
        {"role": "user", "content": problem["question_content"]}], extract_code=fake_extract)


class LegTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out")
        self.data = os.path.join(self.tmp.name, "problems.jsonl")
        with open(self.data, "w") as handle:
            for problem in PROBLEMS:
                handle.write(json.dumps(problem) + "\n")
        self.scorer = os.path.join(self.tmp.name, "fake_scorer.py")
        with open(self.scorer, "w") as handle:
            handle.write(FAKE_SCORER)
        os.chmod(self.scorer, 0o755)
        self.stub = JsonStub(lambda body: (200, completion("CODE: PASS_ME")))
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        real = lcb.load_helpers
        lcb.load_helpers = lambda src: fake_helpers()
        self.addCleanup(setattr, lcb, "load_helpers", real)

    def run_leg(self, *extra):
        argv = ["--model", "m", "--endpoint", self.stub.endpoint, "--data", self.data,
                "--output-dir", self.out, "--src", self.tmp.name,
                "--scorer-python", self.scorer, "--slots", "4", *extra]
        return lcb.run(lcb.parse_args(argv))

    def result(self):
        with open(os.path.join(self.out, "lcb.result.json")) as handle:
            return json.load(handle)


class PhasesTest(LegTest):
    def test_both_phases_score_every_problem(self):
        block = self.run_leg()
        self.assertEqual(block["items"], 2)
        self.assertEqual(block["scored"], 2)
        self.assertEqual(block["correct"], 2)
        self.assertEqual(block["status"], "pass")
        self.assertEqual(block["no_code"], 0)
        self.assertEqual(block["ungenerated"], 0)
        self.assertEqual(block["slots"], 4)
        self.assertEqual(block["cap"], lcb.LCB_CAP)
        self.assertNotIn("aborted", block)

    def test_generation_lands_in_the_partial_jsonl(self):
        self.run_leg()
        with open(os.path.join(self.out, "m-lcb.partial.jsonl")) as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual([record["question_id"] for record in records], ["q1", "q2"])
        self.assertTrue(all(record["code"] for record in records))

    def test_the_items_sha_covers_the_run_order(self):
        self.assertEqual(self.run_leg()["items_sha"], lcb.items_sha(PROBLEMS))

    def test_a_transport_failure_fails_the_leg(self):
        self.stub.routes = lambda body: (500, {"error": "boom"})
        block = self.run_leg()
        self.assertEqual(block["failed"], 2)
        self.assertEqual(block["scored"], 0)
        self.assertEqual(block["status"], "fail")

    def test_main_is_the_cli_entry(self):
        argv = ["lcb.py", "--model", "m", "--endpoint", self.stub.endpoint,
                "--data", self.data, "--output-dir", self.out, "--src", self.tmp.name,
                "--scorer-python", self.scorer]
        with mock.patch.object(sys, "argv", argv):
            lcb.main()
        self.assertTrue(os.path.exists(os.path.join(self.out, "lcb.result.json")))


class ResumeTest(LegTest):
    def test_finished_problems_are_not_asked_again(self):
        os.makedirs(self.out)
        partial = os.path.join(self.out, "m-lcb.partial.jsonl")
        with open(partial, "w") as handle:
            handle.write(json.dumps({"question_id": "q2", "code": "PASS_ME"}) + "\n")
        self.run_leg()
        self.assertEqual(len(self.stub.seen), 1)
        block = self.result()
        self.assertEqual(block["generated"], 2)
        self.assertEqual(block["scored"], 2)


class GuardTest(LegTest):
    def test_an_exhausted_budget_stops_and_records_the_abort(self):
        block = self.run_leg("--max-hours", "0")
        self.assertEqual(len(self.stub.seen), 1)
        self.assertTrue(block["aborted"])
        self.assertIn("budget", block["abort_reason"])
        self.assertEqual(block["generated"], 1)
        self.assertEqual(block["ungenerated"], 1)
        with open(os.path.join(self.out, "m-lcb.aborted.json")) as handle:
            aborted = json.load(handle)
        self.assertEqual(aborted["generated"], 1)
        self.assertEqual(aborted["with_code"], 1)

    def test_problems_that_never_emit_code_fail_the_leg(self):
        self.stub.routes = lambda body: (200, completion("plain prose, no code fence"))
        block = self.run_leg()
        self.assertEqual(block["no_code"], 2)
        self.assertEqual(block["scored"], 0)
        self.assertEqual(block["status"], "fail")

    def test_the_stall_guard_fires_after_quiet_minutes(self):
        guard = Guard(max_hours=8.0, stall_min=10, max_failures=5, started_at=0.0)
        self.assertFalse(guard.record(FailureKind.NO_CODE, 60.0).stop)
        decision = guard.record(FailureKind.NO_CODE, 11 * 60.0)
        self.assertTrue(decision.stop)
        self.assertIn("minutes", decision.reason)

    def test_a_failure_streak_stops_the_leg(self):
        guard = Guard(max_hours=8.0, stall_min=10, max_failures=5, started_at=0.0)
        for _ in range(4):
            self.assertFalse(guard.record(FailureKind.ENGINE, 60.0).stop)
        decision = guard.record(FailureKind.ENGINE, 61.0)
        self.assertTrue(decision.stop)
        self.assertIn("engine", decision.reason)

    def test_remaining_seconds_floors_at_zero(self):
        guard = Guard(max_hours=1.0, stall_min=10, max_failures=5, started_at=0.0)
        self.assertEqual(guard.remaining_seconds(600.0), 3000.0)
        self.assertEqual(guard.remaining_seconds(7200.0), 0.0)


class WorkerTest(unittest.TestCase):
    def test_run_metrics_scores_an_aligned_batch(self):
        payload = {"problems": [{"question_id": "q1"}, {"question_id": "q2"}],
                   "generations": [["PASS_ME"], ["nope"]],
                   "workers": 1, "exec_timeout": 6}

        def metrics(samples, generations, num_process_evaluate, timeout, debug):
            self.assertEqual(generations, [["PASS_ME"], ["nope"]])
            self.assertEqual(num_process_evaluate, 1)
            self.assertEqual(timeout, 6)
            return {"pass@1": 0.5, "detail": {"pass@1": {"0": 1.0, "1": 0.0}}}, None, None

        class FakeProblem:
            def __init__(self, raw):
                self.raw = raw

            def get_evaluation_sample(self):
                return self.raw

        result = lcb_worker.run_metrics(payload, metrics, FakeProblem)
        self.assertEqual(result["pass@1"], 0.5)
        self.assertEqual(result["detail"], {"0": 1.0, "1": 0.0})


if __name__ == "__main__":
    unittest.main()
