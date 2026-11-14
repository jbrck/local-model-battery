"""The adapter legs: command building, output parsing, and the block they write."""
import json
import os
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from scripts.adapters import ifeval_toolbench as adapter
from scripts.lib.harness import LegArgs, Tally, result_path

REPORT = textwrap.dedent("""
    # IFEval Benchmark - fake
    - **Prompt Accuracy**: **50.0%**
    - **Instruction Accuracy**: **75.0%**
    **Prompt-level Accuracy:** 50.0% (2/4)
    **Instruction-level Accuracy:** 75.0% (3/4)
    **Duration:** 2888.8s
    **Tokens:** 250,061
""")

ENVELOPE = {
    "schema_version": "1", "run_id": "r1", "status": "completed",
    "scores": {
        "final_score": 60, "rating": "ok", "total_tokens": 564,
        "scenario_results": [
            {"scenario_id": "TC-01", "status": "pass", "failure_kind": None},
            {"scenario_id": "TC-02", "status": "pass", "failure_kind": None},
            {"scenario_id": "TC-03", "status": "partial", "failure_kind": "partial"},
            {"scenario_id": "TC-04", "status": "fail", "failure_kind": "budget_exceeded"},
            {"scenario_id": "TC-05", "status": "fail", "failure_kind": "timeout"},
            {"scenario_id": "TC-06", "status": "fail", "failure_kind": "connection_error"},
        ],
    },
}


def args_for(tmp: str, **overrides) -> LegArgs:
    values = {"model": "fake-model", "endpoint": "http://127.0.0.1:9/v1", "output_dir": tmp,
              "cap": 2048}
    values.update(overrides)
    return LegArgs(**values)


def write_cli(tmp: str, script: str) -> str:
    path = os.path.join(tmp, "tool-eval-bench")
    with open(path, "w") as handle:
        handle.write("#!/bin/sh\n" + script)
    os.chmod(path, 0o755)
    return path


class CommandTest(unittest.TestCase):
    def test_toolbench_runs_the_subcommand_with_the_envelope_path(self):
        args = args_for("/tmp/x")
        command = adapter.build_command("toolbench", args, "cli", "/tmp/x/env.json")
        self.assertEqual(command[1:3], ["run", "--model"])
        self.assertIn("--base-url", command)
        self.assertIn("/tmp/x/env.json", command)

    def test_ifeval_selects_the_plugin_mode(self):
        args = args_for("/tmp/x")
        command = adapter.build_command("ifeval", args, "cli", "/tmp/x/env.json")
        self.assertIn("--ifeval-only", command)
        self.assertNotIn("run", command)

    def test_the_cap_reaches_the_payload_and_thinking_off_flips_the_switch(self):
        args = args_for("/tmp/x", cap=4096, thinking="off", seed=7, limit=5)
        command = adapter.build_command("ifeval", args, "cli", "/tmp/x/env.json")
        index = command.index("--backend-kwargs")
        self.assertEqual(json.loads(command[index + 1]), {"max_tokens": 4096})
        self.assertIn("--no-think", command)
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--ifeval-limit") + 1], "5")


class RunLegPathTest(unittest.TestCase):
    """A relative output dir is the historical footgun: the CLI resolves it against its own
    working directory and the report lands where the parser never looks."""

    def test_the_cli_gets_absolute_paths_even_for_a_relative_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("legs")
                args = args_for("legs")
                with mock.patch.object(adapter, "run_cli", return_value=(0, "")) as run:
                    with mock.patch.object(adapter, "parse_toolbench",
                                           return_value=(Tally(), {})) as parse:
                        adapter.run_leg("toolbench", args, "cli")
                command = run.call_args.args[0]
                output_dir = command[command.index("--output-dir") + 1]
                json_path = command[command.index("--json-file") + 1]
                self.assertTrue(os.path.isabs(output_dir))
                self.assertTrue(os.path.isabs(json_path))
                self.assertTrue(json_path.startswith(output_dir + os.sep))
                parse.assert_called_once()
            finally:
                os.chdir(previous)

    def test_find_report_is_depth_tolerant(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = os.path.join(tmp, "runs", "a", "b", "c", "2026", "11")
            os.makedirs(deep)
            path = os.path.join(deep, "report.md")
            with open(path, "w") as handle:
                handle.write("x")
            found = adapter.find_report(tmp, time.time() - 10)
            self.assertEqual(os.path.realpath(found), os.path.realpath(path))


class ToolbenchParseTest(unittest.TestCase):
    def test_statuses_map_to_the_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env.json")
            with open(path, "w") as handle:
                json.dump(ENVELOPE, handle)
            tally, extra = adapter.parse_toolbench(path)
        self.assertEqual(tally.items, 6)
        self.assertEqual((tally.scored, tally.correct), (4, 2))
        self.assertEqual((tally.timeout_hits, tally.failed), (1, 1))
        self.assertEqual(tally.tokens, 564)
        self.assertEqual(extra["final_score"], 60)

    def test_an_error_envelope_is_a_problem_not_a_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env.json")
            with open(path, "w") as handle:
                json.dump({"error": "server unreachable"}, handle)
            with self.assertRaises(ValueError):
                adapter.parse_toolbench(path)


class ReportParseTest(unittest.TestCase):
    def test_the_report_lines_become_counters(self):
        tally, extra = adapter.parse_report(REPORT)
        self.assertEqual((tally.items, tally.scored, tally.correct), (4, 4, 2))
        self.assertEqual(tally.seconds, 2888.8)
        self.assertEqual(tally.tokens, 250061)
        self.assertEqual(extra["instruction_accuracy"], "3/4")

    def test_a_report_without_the_accuracy_line_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter.parse_report("# empty report\n")


class RunTest(unittest.TestCase):
    def run_case(self, leg: str, script: str) -> tuple[dict, dict, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = os.path.join(tmp, "fixture")
            with open(fixture, "w") as handle:
                (handle.write(json.dumps(ENVELOPE)) if leg == "toolbench"
                 else handle.write(REPORT))
            cli = write_cli(tmp, script.format(tmp=tmp, fixture=fixture))
            block = adapter.run_leg(leg, args_for(tmp), cli)
            with open(result_path(leg, args_for(tmp))) as handle:
                saved = json.load(handle)
            with open(os.path.join(tmp, "argv.txt")) as handle:
                argv = handle.read().splitlines()
            return block, saved, argv

    def test_a_passing_toolbench_run_writes_the_block(self):
        script = ("printf '%s\\n' \"$@\" > '{tmp}/argv.txt'\n"
                  "cp '{fixture}' '{tmp}/tool-eval-toolbench.json'\n")
        block, saved, argv = self.run_case("toolbench", script)
        self.assertEqual(block["status"], "timeout")
        self.assertEqual((block["correct"], block["scored"], block["items"]), (2, 4, 6))
        self.assertEqual(saved["leg"], "toolbench")
        self.assertIn("run", argv)
        self.assertIn("--backend-kwargs", argv)

    def test_a_passing_ifeval_run_reads_the_newest_report(self):
        script = ("printf '%s\\n' \"$@\" > '{tmp}/argv.txt'\n"
                  "mkdir -p '{tmp}/runs/2026/11'\n"
                  "cp '{fixture}' '{tmp}/runs/2026/11/report.md'\n")
        block, saved, argv = self.run_case("ifeval", script)
        self.assertEqual(block["status"], "pass")
        self.assertEqual((block["correct"], block["scored"]), (2, 4))
        self.assertEqual(block["instruction_accuracy"], "3/4")
        self.assertEqual(block["tps"], round(250061 / 2888.8, 1))
        self.assertIn("--ifeval-only", argv)
        self.assertNotIn("run", argv)

    def test_a_failed_cli_becomes_a_fail_block_with_the_stderr(self):
        script = ("printf '%s\\n' \"$@\" > '{tmp}/argv.txt'\n"
                  "echo server unreachable >&2\nexit 1\n")
        block, saved, _ = self.run_case("toolbench", script)
        self.assertEqual(block["status"], "fail")
        self.assertIn("server unreachable", " | ".join(block["errors"]))
        self.assertEqual(saved["status"], "fail")

    def test_a_failed_ifeval_run_without_a_report_fails(self):
        script = ("printf '%s\\n' \"$@\" > '{tmp}/argv.txt'\n"
                  "echo download failed >&2\nexit 1\n")
        block, _, _ = self.run_case("ifeval", script)
        self.assertEqual(block["status"], "fail")
        self.assertIn("no report under", block["errors"][0])


class ParseArgsTest(unittest.TestCase):
    def test_the_leg_selector_and_the_cli_override_parse(self):
        leg, args, cli = adapter.parse_args(
            ["--leg", "toolbench", "--model", "m", "--cap", "512", "--thinking", "off",
             "--tool-eval-bench", "/opt/teb"])
        self.assertEqual((leg, cli), ("toolbench", "/opt/teb"))
        self.assertEqual((args.cap, args.thinking), (512, "off"))
        self.assertTrue(args.endpoint.endswith("/v1"))

    def test_the_cli_defaults_to_the_path_lookup(self):
        leg, args, cli = adapter.parse_args(["--leg", "ifeval", "--model", "m"])
        self.assertEqual((leg, cli), ("ifeval", adapter.DEFAULT_CLI))


class MissingCliTest(unittest.TestCase):
    def test_an_absent_binary_is_a_fail_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            block = adapter.run_leg("toolbench", args_for(tmp), os.path.join(tmp, "absent"))
            self.assertEqual(block["status"], "fail")
            self.assertIn("absent", " | ".join(block["errors"]))


class MainTest(unittest.TestCase):
    def test_main_runs_one_leg_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = os.path.join(tmp, "fixture")
            with open(fixture, "w") as handle:
                json.dump(ENVELOPE, handle)
            write_cli(tmp, f"cp '{fixture}' '{tmp}/tool-eval-toolbench.json'\n")
            adapter.main(["--leg", "toolbench", "--model", "fake-model", "--output-dir", tmp,
                          "--tool-eval-bench", os.path.join(tmp, "tool-eval-bench")])
            with open(result_path("toolbench", args_for(tmp))) as handle:
                self.assertEqual(json.load(handle)["items"], 6)


if __name__ == "__main__":
    unittest.main()
