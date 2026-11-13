"""The report: a fixture run directory in, a document naming every leg and every deviation out.

The fixture mirrors what run_battery actually leaves behind -- MANIFEST.json, run-plan.json
and legs/*.result.json -- so the report is tested against the real file names and the real
block keys, not the plan's wording.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import report as cli  # noqa: E402
from scripts.lib.report import ReportError, build_report, render_markdown, write_report  # noqa: E402

MANIFEST = {
    "battery_version": "2.0.0",
    "standard_version": "2.0.0",
    "model": "arete-27b",
    "stamp_line": "ON-STANDARD 2.0.0",
    "stamp": {"cap": {"math500": "8192", "lcb": "1024"}},
    "props": {"build_info": "b4567-abc", "model_path": "models/arete-27b-Q4_K_M.gguf"},
}

PLAN = {
    "battery_version": "2.0.0",
    "standard_version": "2.0.0",
    "model": "arete-27b",
    "stamp_line": "ON-STANDARD 2.0.0",
    "footprint": {"known": True, "source": "gguf header",
                  "path": "models/arete-27b-Q4_K_M.gguf", "weights_gb": 16.4,
                  "kv_bytes_per_token": 16384, "kv_type": "q8_0", "notes": [], "missing": []},
    "legs": [
        {"id": "math500", "name": "MATH-500", "cap": 8192, "ctx": 131072, "thinking": "on",
         "budget_s": 7200, "case_timeout_s": 900, "gate": False},
        {"id": "lcb", "name": "LiveCodeBench (release_v6 lite)", "cap": 1024, "ctx": 131072,
         "thinking": "on", "budget_s": 3600, "case_timeout_s": 600, "gate": False},
    ],
    "skips": [],
    "decisions": [
        {"variable": "cap:math500", "kind": "measurement", "chosen": "8192", "standard": "8192",
         "deviation": False, "source": "standard",
         "consequence": "math500 may generate 8192 tokens an answer, as the standard asks"},
        {"variable": "harness_version:math500", "kind": "measurement", "chosen": "math500-v1",
         "standard": "math500-v1", "deviation": False, "source": "standard",
         "consequence": "a harness bump voids the comparison with earlier runs"},
        {"variable": "engine_version", "kind": "capacity", "chosen": "b4567-abc",
         "standard": "n/a", "deviation": False, "source": "engine",
         "consequence": "as reported by the endpoint"},
        {"variable": "seed", "kind": "measurement", "chosen": "42", "standard": "42",
         "deviation": False, "source": "standard",
         "consequence": "every leg inherits it; a different seed is a different run"},
    ],
    "warnings": [],
}

MATH500_BLOCK = {
    "leg": "math500", "model": "arete-27b", "status": "pass",
    "items": 500, "scored": 500, "correct": 401, "score_pct": 80.2,
    "tps": 27.0, "cap_hits": 2, "runaway": 0, "timeout_hits": 0, "failed": 0,
    "tokens": 123456, "elapsed_s": 4572,
    "cap": 8192, "ctx": 131072, "thinking": "on", "seed": 42,
    "items_sha": "1a2b3c4d",
}

LCB_BLOCK = {
    "leg": "lcb", "status": "not run",
    "reason": "scripts/adapters/lcb.py is missing",
}


def write_run(tmp: Path, *, include_plan: bool = True, math500: dict | None = None,
              lcb: dict | None = None) -> Path:
    """A run directory shaped like the ones run_battery leaves behind."""
    run = tmp / "results" / "arete-27b"
    (run / "legs").mkdir(parents=True)
    if include_plan:
        (run / "run-plan.json").write_text(json.dumps(PLAN, indent=2) + "\n")
    (run / "MANIFEST.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    (run / "legs" / "math500.result.json").write_text(
        json.dumps(math500 if math500 is not None else MATH500_BLOCK, indent=2) + "\n")
    (run / "legs" / "lcb.result.json").write_text(
        json.dumps(lcb if lcb is not None else LCB_BLOCK, indent=2) + "\n")
    return run


class ReportBuildTest(unittest.TestCase):
    """What the document says for a run with one scored leg and one skipped leg."""

    def _markdown(self, run: Path) -> str:
        return render_markdown(build_report(run))

    def test_names_both_legs_and_the_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("arete-27b", text)
            self.assertIn("math500", text)
            self.assertIn("lcb", text)
            self.assertIn("pass", text)
            self.assertIn("not run", text)
            self.assertIn("scripts/adapters/lcb.py is missing", text)
            self.assertIn("401/500", text)

    def test_what_was_tested_names_shape_engine_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("gguf header", text)
            self.assertIn("16.4 GB", text)
            self.assertIn("b4567-abc", text)
            self.assertIn("models/arete-27b-Q4_K_M.gguf", text)

    def test_decision_rows_carry_chosen_standard_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("cap:math500", text)
            self.assertIn("standard", text)
            self.assertIn("engine_version", text)

    def test_throughput_and_wall_time_per_leg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("27.0", text)
            self.assertIn("1h 16m 12s", text)

    def test_report_json_matches_the_markdown_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(write_run(Path(tmp)))
            by_id = {row["leg"]: row for row in report["legs"]}
            self.assertEqual(by_id["math500"]["status"], "pass")
            self.assertEqual(by_id["math500"]["cases_scored"], 500)
            self.assertEqual(by_id["lcb"]["status"], "not run")
            self.assertIn("scripts/adapters/lcb.py", by_id["lcb"]["reason"])
            self.assertEqual(report["comparability"]["baseline_recorded"], True)


class ComparabilityTest(unittest.TestCase):
    """The not-comparable section is driven by the recorded values, not by prose."""

    def _markdown(self, run: Path) -> str:
        return render_markdown(build_report(run))

    def test_cap_off_the_plan_is_called_out(self) -> None:
        block = dict(MATH500_BLOCK, cap=1024)
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp), math500=block))
            self.assertIn("not comparable", text)
            self.assertIn("math500", text)
            self.assertIn("1024", text)

    def test_harness_off_the_plan_is_called_out(self) -> None:
        block = dict(MATH500_BLOCK, harness_version="math500-v2")
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp), math500=block))
            self.assertIn("not comparable", text)
            self.assertIn("harness", text)
            self.assertIn("math500-v2", text)

    def test_engine_off_the_manifest_is_called_out(self) -> None:
        plan = json.loads(json.dumps(PLAN))
        for row in plan["decisions"]:
            if row["variable"] == "engine_version":
                row["chosen"] = "b9999-new"
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "run-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            text = self._markdown(run)
            self.assertIn("not comparable", text)
            self.assertIn("engine", text)

    def test_items_hash_is_stated_as_what_other_runs_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("1a2b3c4d", text)

    def test_no_plan_means_no_baseline_and_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp), include_plan=False))
            self.assertIn("no baseline", text)

    def test_no_findings_says_the_run_compares(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp)))
            self.assertIn("compare", text)


class MissingDataTest(unittest.TestCase):
    """A broken run directory gets an explanation, never a traceback."""

    def test_no_manifest_raises_a_one_line_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "results" / "nothing-here"
            empty.mkdir(parents=True)
            with self.assertRaises(ReportError) as caught:
                build_report(empty)
            self.assertEqual(str(caught.exception).count("\n"), 0)
            self.assertIn("MANIFEST.json", str(caught.exception))

    def test_planned_leg_without_a_result_file_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "legs" / "math500.result.json").unlink()
            report = build_report(run)
            text = render_markdown(report)
            self.assertIn("math500", text)
            self.assertIn("no result", text)

    def test_unreadable_result_file_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "legs" / "math500.result.json").write_text("{not json")
            report = build_report(run)
            text = render_markdown(report)
            self.assertIn("math500", text)
            self.assertIn("unreadable", text)


def write_manifest_only(tmp: Path, manifest: dict) -> Path:
    """A run directory holding a manifest and nothing else: no plan, no leg results."""
    run = tmp / "results" / "arete-27b"
    run.mkdir(parents=True)
    (run / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run


def write_plan(run: Path, plan: dict) -> None:
    """Replace the run's plan with a variant."""
    (run / "run-plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def plan_variant(**edits: str) -> dict:
    """A copy of PLAN with one decision row's chosen value and deviation flag rewritten."""
    plan = json.loads(json.dumps(PLAN))
    for row in plan["decisions"]:
        if row["variable"] in edits:
            row["chosen"], row["deviation"] = edits[row["variable"]], True
    return plan


class RunStateTest(unittest.TestCase):
    """run.json is the fallback source: what it supplies, and what its absence is called."""

    def _markdown(self, run: Path) -> str:
        return render_markdown(build_report(run))

    def test_a_leg_block_comes_from_run_json_when_its_result_file_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "legs" / "math500.result.json").unlink()
            (run / "run.json").write_text(
                json.dumps({"results": {"math500": MATH500_BLOCK}}, indent=2) + "\n")
            self.assertIn("401/500", self._markdown(run))

    def test_run_json_that_cannot_be_parsed_is_named_in_the_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "run.json").write_text("{not json")
            report = build_report(run)
            self.assertIn("unreadable", report["sources"]["run_state"])
            self.assertIn("run.json", self._markdown(run))

    def test_run_json_holding_something_other_than_an_object_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "run.json").write_text("[]\n")
            report = build_report(run)
            self.assertIn("is not a JSON object", report["sources"]["run_state"])

    def test_run_json_without_a_results_mapping_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "run.json").write_text(json.dumps({"started": "2026-11-12T00:00:00"}) + "\n")
            report = build_report(run)
            self.assertIn("carries no results mapping", report["sources"]["run_state"])


class PlanVariantTest(unittest.TestCase):
    """Plans that moved a number, dropped a leg, or never named a baseline."""

    def _markdown(self, run: Path) -> str:
        return render_markdown(build_report(run))

    def test_a_planned_deviation_is_a_finding(self) -> None:
        plan = plan_variant(**{"cap:math500": "4096"})
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            write_plan(run, plan)
            text = self._markdown(run)
            self.assertIn("not comparable", text)
            self.assertIn("the plan moved cap:math500 to 4096", text)

    def test_a_deviation_whose_name_carries_no_leg_reports_none(self) -> None:
        plan = plan_variant(seed="7")
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            write_plan(run, plan)
            moved = [f for f in build_report(run)["comparability"]["findings"]
                     if f["detail"].startswith("the plan moved seed")]
            self.assertEqual(len(moved), 1)
            self.assertEqual(moved[0]["leg"], "")
            self.assertEqual(moved[0]["detail"], "the plan moved seed to 7 (standard 42)")

    def test_a_leg_with_a_result_but_no_planned_baseline_is_noted(self) -> None:
        plan = json.loads(json.dumps(PLAN))
        plan["legs"] = [leg for leg in plan["legs"] if leg["id"] != "math500"]
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            write_plan(run, plan)
            self.assertIn("math500 has a result but no planned baseline", self._markdown(run))

    def test_a_plan_that_is_not_an_object_is_named_and_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            (run / "run-plan.json").write_text("[]\n")
            report = build_report(run)
            self.assertIn("is not a JSON object", report["sources"]["plan"])
            self.assertFalse(report["comparability"]["baseline_recorded"])

    def test_no_stamp_anywhere_says_nothing_declares_a_comparison(self) -> None:
        plan = json.loads(json.dumps(PLAN))
        plan.pop("stamp_line")
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            write_plan(run, plan)
            self.assertIn("The run names no stamp", self._markdown(run))

    def test_shape_notes_from_the_plan_are_rendered(self) -> None:
        plan = json.loads(json.dumps(PLAN))
        plan["footprint"]["notes"] = ["KV measured from the GGUF header, not estimated"]
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            write_plan(run, plan)
            self.assertIn("KV measured from the GGUF header", self._markdown(run))

    def test_the_manifest_fit_supplies_the_shape_when_the_plan_has_none(self) -> None:
        manifest = dict(MANIFEST, fit={"weights_gb": 16.4, "kv_bytes_per_token": 16384})
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_manifest_only(Path(tmp), manifest))
            self.assertIn("manifest fit", text)
            self.assertIn("16.4 GB", text)

    def test_a_manifest_with_no_engine_build_says_so(self) -> None:
        manifest = {k: v for k, v in MANIFEST.items() if k != "props"}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("records no engine build",
                          self._markdown(write_manifest_only(Path(tmp), manifest)))

    def test_a_run_with_neither_a_plan_nor_legs_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(write_manifest_only(Path(tmp), MANIFEST))
            self.assertEqual(report["legs"], [])
            self.assertIn("No planned legs and no leg result files", render_markdown(report))

    def test_a_wall_time_under_an_hour_reads_in_minutes(self) -> None:
        block = dict(MATH500_BLOCK, elapsed_s=305)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("5m 5s", self._markdown(write_run(Path(tmp), math500=block)))

    def test_a_harness_matching_the_plan_is_not_a_finding(self) -> None:
        block = dict(MATH500_BLOCK, harness_version="math500-v1")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("Nothing recorded differs",
                          self._markdown(write_run(Path(tmp), math500=block)))

    def test_a_harness_recorded_with_no_planned_baseline_is_a_note(self) -> None:
        plan = json.loads(json.dumps(PLAN))
        plan["decisions"] = [row for row in plan["decisions"]
                             if row["variable"] != "harness_version:math500"]
        block = dict(MATH500_BLOCK, harness_version="math500-v9")
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp), math500=block)
            write_plan(run, plan)
            text = self._markdown(run)
            self.assertIn("records harness math500-v9", text)
            self.assertIn("the plan does not name a baseline", text)

    def test_a_leg_that_records_no_items_hash_says_so(self) -> None:
        block = {key: value for key, value in MATH500_BLOCK.items() if key != "items_sha"}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("math500 records no items hash",
                          self._markdown(write_run(Path(tmp), math500=block)))

    def test_a_seed_off_the_plan_is_a_finding(self) -> None:
        block = dict(MATH500_BLOCK, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            text = self._markdown(write_run(Path(tmp), math500=block))
            self.assertIn("math500 ran with seed 7; the plan pinned 42", text)


class DeterminismTest(unittest.TestCase):
    """Re-running the report changes nothing: every number comes from the run's files."""

    def test_two_builds_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            first = render_markdown(build_report(run))
            second = render_markdown(build_report(run))
            self.assertEqual(first, second)
            self.assertEqual(json.dumps(build_report(run), sort_keys=True),
                             json.dumps(build_report(run), sort_keys=True))


class CliTest(unittest.TestCase):
    """scripts/report.py takes the run directory and writes both files, printing the markdown."""

    def test_writes_both_files_and_prints_the_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp))
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main([str(run)])
            self.assertEqual(code, 0)
            self.assertTrue((run / "report.md").is_file())
            self.assertTrue((run / "report.json").is_file())
            self.assertIn("# arete-27b", out.getvalue())
            self.assertEqual(out.getvalue(),
                             (run / "report.md").read_text())
            self.assertIn("report.md", err.getvalue())

    def test_missing_manifest_exits_nonzero_with_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main([str(empty)])
            self.assertEqual(code, 2)
            self.assertEqual(err.getvalue().strip().count("\n"), 0)
            self.assertIn("MANIFEST.json", err.getvalue())


if __name__ == "__main__":
    unittest.main()
