"""The HermesAgent-20 adapter: command shape, scoring, per-case resume."""
import json
import os
import tempfile
import unittest
from unittest import mock

from scripts.adapters import hermesagent20 as leg

SCENARIOS = [
    {"id": "HA-01", "passed": True, "failure_mode": "passed", "latency_seconds": 10.0, "tokens_completion": 500},
    {"id": "HA-02", "passed": False, "failure_mode": "token_limit", "latency_seconds": 20.0, "tokens_completion": 900},
    {"id": "HA-03", "passed": False, "failure_mode": "timeout", "latency_seconds": 600.0, "tokens_completion": 0},
    {"id": "HA-04", "passed": False, "failure_mode": "http_error", "latency_seconds": 1.0, "tokens_completion": 0},
]

FAKE_RUNNER = """#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
open(os.environ["ARGV_OUT"], "a").write(json.dumps(argv) + "\\n")
save = argv[argv.index("--save-json") + 1]
scenarios = json.load(open(os.environ["FAKE_SCENARIOS"]))
resumed = "--resume" in argv
kept = scenarios if resumed else scenarios[:2]
pack = {"pack_id": "hermesagent-20", "version": "9.9", "scenario_count": len(kept),
        "catalog_scenario_count": len(scenarios), "scenarios": kept}
with open(save + ".partial.jsonl", "w") as journal:
    for scenario in kept:
        journal.write(json.dumps({"journal_version": 1, "pack": pack,
                                  "scenario": scenario}) + "\\n")
json.dump({"runner_version": "9.9", "packs": [pack]}, open(save, "w"))
if not resumed:
    sys.exit(3)
"""


class CommandTest(unittest.TestCase):
    def test_the_command_carries_the_leg_contract(self):
        args = leg.parse_leg_args("t", argv=["--model", "m", "--cap", "4096",
                                             "--timeout", "600"])
        cmd = leg.build_command(args, "/tmp/save.json", None)
        self.assertIn("--enable-sandboxed-packs", cmd)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--timeout-per-case") + 1], "600")
        self.assertIn("--incremental", cmd)
        self.assertIn("--enable-thinking", cmd)

    def test_thinking_off_and_resume_reach_the_command(self):
        args = leg.parse_leg_args("t", argv=["--model", "m", "--thinking", "off"])
        cmd = leg.build_command(args, "/tmp/save.json", "/tmp/save.json.partial.jsonl")
        self.assertIn("--no-thinking", cmd)
        self.assertEqual(cmd[cmd.index("--resume") + 1], "/tmp/save.json.partial.jsonl")


class ScoreTest(unittest.TestCase):
    def test_a_journal_alone_rebuilds_the_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "save.json")
            with open(save + ".partial.jsonl", "w") as handle:
                for scenario in SCENARIOS[:2]:
                    record = {"journal_version": 1,
                              "pack": {"pack_id": leg.PACK_ID, "version": "9.9",
                                       "catalog_scenario_count": 4},
                              "scenario": scenario}
                    handle.write(json.dumps(record) + "\n")
            pack = leg.pack_of(leg.load_run(save) or {})
        self.assertEqual([item["id"] for item in pack["scenarios"]], ["HA-01", "HA-02"])
        self.assertEqual(pack["catalog_scenario_count"], 4)
        self.assertFalse(leg.run_is_complete(pack))

    def test_each_failure_kind_is_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "save.json")
            with open(save, "w") as handle:
                json.dump({"packs": [{"pack_id": leg.PACK_ID, "version": "9.9",
                                      "catalog_scenario_count": 4,
                                      "scenarios": SCENARIOS}]}, handle)
            tally = leg.Tally()
            version = leg.score_pack(leg.pack_of(leg.load_run(save)), tally)
        self.assertEqual(version, "9.9")
        self.assertEqual((tally.items, tally.scored, tally.correct), (4, 2, 1))
        self.assertEqual((tally.runaway, tally.timeout_hits, tally.failed), (1, 1, 1))
        self.assertEqual(tally.status(), "timeout")


class ResumeTest(unittest.TestCase):
    def test_a_stopped_run_resumes_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = os.path.join(tmp, "scenarios.json")
            with open(scenarios, "w") as handle:
                json.dump(SCENARIOS[:3], handle)
            argv_out = os.path.join(tmp, "argv.json")
            runner = os.path.join(tmp, "fake-runner.py")
            with open(runner, "w") as handle:
                handle.write(FAKE_RUNNER)
            os.chmod(runner, 0o755)
            env = {**os.environ, "ARGV_OUT": argv_out, "FAKE_SCENARIOS": scenarios}
            leg.RUNNER = runner
            argv = ["--model", "m", "--output-dir", tmp]
            with mock.patch.dict(os.environ, env, clear=False):
                leg.main(argv)
                leg.main(argv)
            calls = [json.loads(line) for line in open(argv_out)]
            self.assertNotIn("--resume", calls[0])
            self.assertEqual(calls[1][calls[1].index("--resume") + 1],
                             os.path.join(tmp, "hermesagent20-run.json.partial.jsonl"))
            block = json.load(open(os.path.join(tmp, "hermesagent20.result.json")))
        self.assertEqual(block["items"], 3)
        self.assertEqual(block["correct"], 1)
        self.assertEqual(block["runaway"], 1)
        self.assertEqual(block["timeout_hits"], 1)
        self.assertEqual(block["harness_version"], "9.9")


if __name__ == "__main__":
    unittest.main()
