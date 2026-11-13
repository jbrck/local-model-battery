"""The orchestrator's own behaviour: what it refuses to run, and what it does with a leg.

The legs are faked here. A leg is whatever a script prints and writes, so a three-line
script is enough to exercise every path the orchestrator has: a clean pass, a leg that
exits without a block, one that hangs past its budget, and one that already passed.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import bench_writing_generate  # noqa: E402
from scripts import run_battery  # noqa: E402
from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.plan_run import PlannedLeg, Skip  # noqa: E402
from tests import fixtures  # noqa: E402

PASS = ("import json, os, sys\n"
        "out = os.path.join(sys.argv[sys.argv.index('--output-dir') + 1], 'math500.result.json')\n"
        "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "json.dump({'leg': 'math500', 'status': 'pass', 'items': 500, 'scored': 500,\n"
        "           'correct': 421, 'score_pct': 84.2, 'tps': 27.5, 'cap_hits': 0,\n"
        "           'runaway': 0, 'timeout_hits': 0}, open(out, 'w'))\n"
        "print('@@RESULT ' + json.dumps({'leg': 'math500', 'status': 'pass'}))\n")
HANG = "import time\ntime.sleep(30)\n"
CRASH = "import sys\nsys.exit(3)\n"


def write_script(root: Path, relative: str, body: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class FakeRun(unittest.TestCase):
    """A throwaway repo shape: one leg script, its data file, and somewhere to write."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "run"
        self.data = self.root / "data"
        self.data.mkdir(parents=True, exist_ok=True)
        (self.data / "math500.json").write_text("[]")

    def planned(self, leg_id: str = "math500", **changes) -> PlannedLeg:
        leg = replace(fixtures.BY_ID[leg_id], implementation="in_repo",
                      entry="scripts/bench_math500.py", data="data/math500.json")
        planned = PlannedLeg(leg=leg, cap=8192, ctx=131072, thinking="on", budget_s=60,
                             case_timeout_s=900)
        return replace(planned, **changes)

    def writing(self) -> PlannedLeg:
        """The writing leg as the standard plans it: no thinking switch, a 16384 cap."""
        leg = replace(fixtures.BY_ID["writing"], implementation="in_repo",
                      entry="scripts/bench_writing_generate.py",
                      data="data/writing_suite_v1.json")
        return PlannedLeg(leg=leg, cap=16384, ctx=131072, thinking="n/a", budget_s=0,
                          case_timeout_s=900)

    def args(self, **changes) -> argparse.Namespace:
        base = dict(model="arete-27b", profile="profiles/example.toml",
                    model_file="", model_dir="",
                    endpoint="http://127.0.0.1:11434/v1", api_key="not-needed",
                    out=str(self.out), resume=False, no_budget_kill=False)
        base.update(changes)
        return argparse.Namespace(**base)


class JudgeWiringTest(FakeRun):
    """The judge reaches the writing leg's command line, and no other leg's."""

    def test_the_writing_leg_gets_the_judge_flags(self):
        args = self.args(judge_model="nous-glm-5-3-flash",
                         judge_endpoint="http://judge:4000/v1")
        command = run_battery.leg_command(self.writing(), args, self.out, str(self.root),
                                          "arete-27b")
        self.assertEqual(command[command.index("--judge-model") + 1], "nous-glm-5-3-flash")
        self.assertEqual(command[command.index("--judge-endpoint") + 1], "http://judge:4000/v1")

    def test_a_leg_with_no_judge_prereq_never_sees_the_flags(self):
        args = self.args(judge_model="nous-glm-5-3-flash",
                         judge_endpoint="http://judge:4000/v1")
        command = run_battery.leg_command(self.planned(), args, self.out, str(self.root),
                                          "arete-27b")
        self.assertNotIn("--judge-model", command)
        self.assertNotIn("--judge-endpoint", command)

    def test_the_standard_pins_the_judge_when_the_operator_is_silent(self):
        pin = load_toml(run_battery.STANDARD_FILE).get("judge") or {}
        self.assertTrue(pin.get("model"), "the standard must pin a judge model")
        args = self.args(judge_model="", judge_endpoint="")
        run_battery.resolve_judge(args)
        self.assertEqual(args.judge_model, pin["model"])
        self.assertEqual(args.judge_endpoint, args.endpoint)

    def test_the_operator_beats_the_pin(self):
        args = self.args(judge_model="my-judge", judge_endpoint="http://mine/v1")
        run_battery.resolve_judge(args)
        self.assertEqual(args.judge_model, "my-judge")
        self.assertEqual(args.judge_endpoint, "http://mine/v1")


class MissingReasonTest(FakeRun):
    def test_a_leg_that_is_not_port_yet_says_so(self):
        leg = replace(fixtures.BY_ID["toolbench"], implementation="port_pending")
        planned = PlannedLeg(leg=leg, cap=8192, ctx=131072, thinking="on", budget_s=60)
        self.assertIn("not in this repo yet", run_battery.missing_reason(planned, str(self.root)))

    def test_a_leg_whose_script_is_absent_is_not_run(self):
        planned = self.planned()
        self.assertIn("scripts/bench_math500.py is missing",
                      run_battery.missing_reason(planned, str(self.root)))

    def test_a_leg_whose_items_are_absent_is_not_run(self):
        write_script(self.root, "scripts/bench_math500.py", PASS)
        (self.data / "math500.json").unlink()
        self.assertIn("data/math500.json is missing",
                      run_battery.missing_reason(self.planned(), str(self.root)))

    def test_a_leg_with_everything_gets_an_empty_reason(self):
        write_script(self.root, "scripts/bench_math500.py", PASS)
        self.assertEqual("", run_battery.missing_reason(self.planned(), str(self.root)))


class LegCommandTest(FakeRun):
    def test_the_command_carries_every_knob_the_plan_decided(self):
        command = run_battery.leg_command(self.planned(), self.args(), self.out,
                                          str(self.root), "arete-27b")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(self.root / "scripts/bench_math500.py"))
        for flag, value in (("--model", "arete-27b"), ("--cap", "8192"), ("--ctx", "131072"),
                            ("--thinking", "on"), ("--timeout", "900"),
                            ("--data", str(self.root / "data/math500.json")),
                            ("--output-dir", str(self.out / "legs"))):
            self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_a_leg_with_its_own_flags_gets_them_appended(self):
        planned = self.planned(extra_args=("--workers", "1"))
        command = run_battery.leg_command(planned, self.args(), self.out, str(self.root), "m")
        self.assertEqual(command[-2:], ["--workers", "1"])

    def test_a_leg_that_brings_its_own_data_gets_no_data_flag(self):
        planned = self.planned()
        planned = replace(planned, leg=replace(planned.leg, data=""))
        command = run_battery.leg_command(planned, self.args(), self.out, str(self.root), "m")
        self.assertNotIn("--data", command)

    def test_a_leg_with_no_thinking_switch_is_not_handed_one(self):
        command = run_battery.leg_command(self.planned(thinking="n/a"), self.args(),
                                          self.out, str(self.root), "m")
        self.assertNotIn("--thinking", command)

    def test_a_leg_that_switches_thinking_off_is_still_told(self):
        command = run_battery.leg_command(self.planned(thinking="off"), self.args(),
                                          self.out, str(self.root), "m")
        self.assertEqual(command[command.index("--thinking") + 1], "off")

    def test_the_writing_leg_gets_a_command_line_its_own_parser_accepts(self):
        command = run_battery.leg_command(self.writing(), self.args(judge_model="my-judge",
                                                                   judge_endpoint="http://j/v1"),
                                          self.out, str(self.root), "arete-27b")
        options = bench_writing_generate.parse_args(command[2:])
        self.assertEqual(options.args.model, "arete-27b")
        self.assertEqual(options.args.cap, 16384)
        self.assertEqual(options.judge_model, "my-judge")


class ReadBlockTest(FakeRun):
    def _write(self, text: str) -> None:
        (self.out / "legs").mkdir(parents=True, exist_ok=True)
        (self.out / "legs" / "math500.result.json").write_text(text)

    def test_no_block_is_none(self):
        self.assertIsNone(run_battery.read_block(self.out, "math500"))

    def test_a_truncated_block_is_none(self):
        self._write('{"leg": "math500"')
        self.assertIsNone(run_battery.read_block(self.out, "math500"))

    def test_a_written_block_comes_back(self):
        self._write('{"leg": "math500", "status": "pass"}')
        self.assertEqual(run_battery.read_block(self.out, "math500")["status"], "pass")

    def test_done_only_counts_a_pass(self):
        self._write('{"leg": "math500", "status": "fail"}')
        self.assertFalse(run_battery.done(self.out, "math500"))
        self._write('{"leg": "math500", "status": "pass"}')
        self.assertTrue(run_battery.done(self.out, "math500"))


class RunLegTest(FakeRun):
    def test_a_passing_leg_is_reported_from_its_own_block(self):
        write_script(self.root, "scripts/bench_math500.py", PASS)
        block = run_battery.run_leg(self.planned(), self.args(), self.out, str(self.root), "m")
        self.assertEqual(block["status"], "pass")
        self.assertEqual(block["correct"], 421)
        self.assertEqual(block["exit_code"], 0)
        self.assertTrue(Path(block["log"]).is_file())

    def test_a_leg_that_exits_without_writing_a_block_is_a_failure(self):
        write_script(self.root, "scripts/bench_math500.py", CRASH)
        block = run_battery.run_leg(self.planned(), self.args(), self.out, str(self.root), "m")
        self.assertEqual(block["status"], "fail")
        self.assertEqual(block["exit_code"], 3)
        self.assertIn("no result block", block["reason"])

    def test_a_leg_that_overruns_its_budget_is_stopped_and_says_so(self):
        write_script(self.root, "scripts/bench_math500.py", HANG)
        planned = self.planned(budget_s=1)
        block = run_battery.run_leg(planned, self.args(), self.out, str(self.root), "m")
        self.assertEqual(block["status"], "timeout")
        self.assertIn("planned budget", block["reason"])

    def test_a_leg_that_wrote_then_overran_keeps_its_numbers_marked_timeout(self):
        wrote_then_hung = ("import json, os, sys, time\n"
                           "out = os.path.join(sys.argv[sys.argv.index('--output-dir') + 1],"
                           " 'math500.result.json')\n"
                           "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
                           "json.dump({'leg': 'math500', 'status': 'pass', 'correct': 10},"
                           " open(out, 'w'))\n"
                           "time.sleep(30)\n")
        write_script(self.root, "scripts/bench_math500.py", wrote_then_hung)
        planned = self.planned(budget_s=1)
        block = run_battery.run_leg(planned, self.args(), self.out, str(self.root), "m")
        self.assertEqual(block["status"], "timeout")
        self.assertEqual(block["correct"], 10)
        self.assertIn("the numbers here are the ones it had written", block["reason"])

    def test_the_budget_kill_can_be_turned_off(self):
        write_script(self.root, "scripts/bench_math500.py", PASS)
        planned = self.planned(budget_s=0)
        args = self.args(no_budget_kill=True)
        self.assertEqual("pass",
                         run_battery.run_leg(planned, args, self.out, str(self.root), "m")["status"])


class RunAllTest(FakeRun):
    def _plan(self, selected, gate_id: str = "selftest"):
        """A plan whose legs are the fakes in this directory."""
        plan = fixtures.plan(fixtures.FITS, selected=selected)
        legs = []
        for planned in plan.legs:
            if planned.leg.id in ("math500", "selftest"):
                leg = replace(planned.leg, implementation="in_repo",
                              entry=f"scripts/bench_{planned.leg.id}.py",
                              data="data/math500.json")
            else:
                leg = planned.leg
            legs.append(replace(planned, leg=leg, budget_s=2 if leg.id == gate_id else 60))
        return replace(plan, legs=tuple(legs), skips=())

    def test_a_missing_leg_is_recorded_rather_than_scored_zero(self):
        plan = self._plan(("math500",))
        results = run_battery.run_all(plan, self.args(no_budget_kill=True), self.out, str(self.root))
        self.assertEqual(results["math500"]["status"], "not run")
        self.assertIn("is missing", results["math500"]["reason"])

    def test_the_gate_stops_the_run_when_it_fails(self):
        write_script(self.root, "scripts/bench_selftest.py", CRASH)
        write_script(self.root, "scripts/bench_math500.py", PASS)
        plan = self._plan(("selftest", "math500"))
        results = run_battery.run_all(plan, self.args(), self.out, str(self.root))
        self.assertEqual(results["selftest"]["status"], "fail")
        self.assertNotIn("math500", results,
                         "the gate leg failed and the run should have stopped")

    def test_a_leg_that_already_passed_is_left_alone_on_resume(self):
        write_script(self.root, "scripts/bench_math500.py", CRASH)
        (self.out / "legs").mkdir(parents=True, exist_ok=True)
        (self.out / "legs" / "math500.result.json").write_text(
            json.dumps({"leg": "math500", "status": "pass", "correct": 421}))
        plan = self._plan(("math500",))
        results = run_battery.run_all(plan, self.args(resume=True), self.out, str(self.root))
        self.assertEqual(results["math500"]["correct"], 421)


class ReportTest(FakeRun):
    def test_the_summary_names_every_leg_and_the_stamp(self):
        plan = fixtures.plan(fixtures.FITS, selected=fixtures.FREE)
        results = {"math500": {"status": "pass", "correct": 421, "scored": 500, "score_pct": 84.2},
                   "mmlu": {"status": "not run", "reason": "data/mmlu_test.json is missing"}}
        text = run_battery.summarise(plan, results)
        self.assertIn("Stamp:", text)
        self.assertIn("math500", text)
        self.assertIn("not run", text)

    def test_a_failed_leg_makes_the_run_non_zero(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        self.assertEqual(1, run_battery.verdict(plan, {"math500": {"status": "fail"}}, strict=False))
        self.assertEqual(0, run_battery.verdict(plan, {"math500": {"status": "pass"}}, strict=False))

    def test_a_leg_that_never_ran_only_fails_a_strict_run(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        results = {"math500": {"status": "not run", "reason": "missing"}}
        self.assertEqual(0, run_battery.verdict(plan, results, strict=False))
        self.assertEqual(1, run_battery.verdict(plan, results, strict=True))

    def test_skips_are_listed_as_skipped_legs(self):
        # The pair of 3080s has nothing installed, so the adapter legs cannot run there.
        plan = fixtures.plan(fixtures.FITS, host=fixtures.PAIR, selected=("toolbench",))
        block = run_battery._skipped_blocks(plan, str(self.root))
        self.assertTrue(plan.skips, "this host should have skipped toolbench")
        self.assertEqual(block[plan.skips[0].leg]["status"], "not run")


class MainTest(FakeRun):
    def _patched(self, plan, host=None, server=None):
        fixture = (plan, host or fixtures.RICH, server or fixtures.SERVER, fixtures.PILOT)
        return mock.patch.object(run_battery, "probe_from_args", return_value=fixture)

    def test_a_dry_run_plans_and_starts_nothing(self):
        plan = fixtures.plan(fixtures.FITS, selected=fixtures.FREE)
        with self._patched(plan):
            code = run_battery.main(["--profile", "profiles/example.toml", "--dry-run",
                                     "--legs", "math500,mmlu"])
        self.assertEqual(code, 0)
        self.assertFalse(self.out.exists())

    def test_a_real_run_needs_a_directory(self):
        plan = fixtures.plan(fixtures.FITS, selected=fixtures.FREE)
        with self._patched(plan):
            code = run_battery.main(["--profile", "profiles/example.toml", "--skip-preflight"])
        self.assertEqual(code, 2)

    def test_a_run_with_nothing_runnable_still_writes_its_account(self):
        plan = fixtures.plan(fixtures.FITS, selected=fixtures.FREE)
        args = self.args(skip_preflight=True)
        with self._patched(plan), mock.patch.object(run_battery, "missing_reason",
                                                    return_value="script is missing"):
            code = run_battery.main(["--profile", "profiles/example.toml", "--out", str(self.out),
                                     "--skip-preflight", "--legs", "math500,mmlu"])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "MANIFEST.json").is_file())
        self.assertIn("not run", (self.out / "NOTES.md").read_text())
        del args

    def test_a_blocking_preflight_stops_before_any_leg(self):
        plan = fixtures.plan(fixtures.FITS, selected=fixtures.FREE)
        blocked = [SimpleNamespace(name="host:fits", blocks=True, ok=False)]
        with self._patched(plan), mock.patch.object(run_battery.preflight, "run_preflight",
                                                    return_value=blocked), \
                mock.patch.object(run_battery.preflight, "write_preflight"), \
                mock.patch.object(run_battery.preflight, "summarise", return_value="blocked"), \
                mock.patch.object(run_battery.preflight, "blocked", return_value=blocked), \
                mock.patch.object(run_battery, "run_all") as run_all:
            code = run_battery.main(["--profile", "profiles/example.toml", "--out", str(self.out),
                                     "--legs", "math500,mmlu"])
        self.assertEqual(code, 1)
        run_all.assert_not_called()

    def test_skipping_the_gate_says_so_and_still_runs(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        with self._patched(plan), mock.patch.object(run_battery, "missing_reason",
                                                    return_value="script is missing"):
            code = run_battery.main(["--profile", "profiles/example.toml", "--out", str(self.out),
                                     "--skip-preflight", "--legs", "math500"])
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "MANIFEST.json").is_file())


class StatusAndDetachTest(FakeRun):
    """The read-only commands answer and leave the run directory to the run itself."""

    def test_status_reads_the_run_directory_and_leaves_it_alone(self):
        (self.out / "legs").mkdir(parents=True)
        (self.out / "run-plan.json").write_text(
            json.dumps({"legs": [{"id": "math500"}], "skips": []}))
        code = run_battery.main(["--profile", "profiles/example.toml", "--out", str(self.out),
                                 "--status"])
        self.assertEqual(code, 0)
        self.assertFalse((self.out / "MANIFEST.json").exists())

    def test_detach_starts_a_child_and_names_where_to_look(self):
        child = [sys.executable, "scripts/run_battery.py", "--out", str(self.out)]
        with mock.patch.object(run_battery.detach, "launch",
                               return_value=(4242, self.out / "run.log", child)):
            code = run_battery.main(["--profile", "profiles/example.toml", "--out", str(self.out),
                                     "--detach"])
        self.assertEqual(code, 0)


class LoadPlanTest(FakeRun):
    """Reusing a plan keeps its settings; the host is still probed for the manifest."""

    def _write_plan(self, plan) -> Path:
        path = self.out / "run-plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.to_dict()))
        return path

    def test_a_reused_plan_is_not_re_planned(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        path = self._write_plan(plan)
        with mock.patch("scripts.lib.env_probe.probe_host", return_value=fixtures.RICH) as probe:
            loaded, host, server, pilot = run_battery.load_plan(self.args(plan=str(path)))
        self.assertEqual([item.leg.id for item in loaded.legs],
                         [item.leg.id for item in plan.legs])
        self.assertEqual(loaded.legs[0].cap, plan.legs[0].cap)
        self.assertIs(host, fixtures.RICH)
        probe.assert_called_once()
        self.assertIsNone(server)
        self.assertIsNone(pilot)

    def test_a_reused_plan_keeps_its_case_timeout(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        plan = replace(plan, legs=(replace(plan.legs[0], case_timeout_s=1234),))
        loaded, _, _, _ = run_battery.load_plan(self.args(plan=str(self._write_plan(plan))))
        self.assertEqual(loaded.legs[0].case_timeout_s, 1234)


class SkipBlockTest(FakeRun):
    def test_a_skip_and_a_missing_script_both_get_a_block(self):
        plan = fixtures.plan(fixtures.FITS, selected=("math500",))
        plan = replace(plan, skips=(Skip("toolbench", "tool-eval-bench is not installed"),))
        blocks = run_battery._skipped_blocks(plan, str(self.root))
        self.assertEqual(blocks["toolbench"]["reason"], "tool-eval-bench is not installed")
        self.assertIn("is missing", blocks["math500"]["reason"])


class InterpreterTest(FakeRun):
    """A leg that declares an interpreter runs under it; a leg that does not runs under this one."""

    def math500(self, interpreter: str) -> PlannedLeg:
        """The planned leg, carrying the interpreter the plan resolved for it."""
        leg = replace(fixtures.BY_ID["math500"], implementation="in_repo",
                      entry="scripts/bench_math500.py")
        return PlannedLeg(leg=leg, cap=8192, ctx=131072, thinking="n/a", budget_s=0,
                          case_timeout_s=600, interpreter=interpreter)

    def test_a_declared_interpreter_runs_the_leg(self):
        command = run_battery.leg_command(self.math500("/opt/other/bin/python3"), self.args(),
                                          self.out, str(self.root), "arete-27b")
        self.assertEqual(command[0], "/opt/other/bin/python3")

    def test_a_leg_with_no_interpreter_runs_under_this_one(self):
        command = run_battery.leg_command(self.math500(""), self.args(), self.out, str(self.root),
                                          "arete-27b")
        self.assertEqual(command[0], sys.executable)
