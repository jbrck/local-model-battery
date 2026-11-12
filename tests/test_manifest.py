"""The run's paper trail: the manifest that has to hold, the notes that end in a stamp."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.manifest import (  # noqa: E402
    MANIFEST_FILE,
    NOTES_FILE,
    PREFLIGHT_FILE,
    RESULTS_FILE,
    ManifestMismatch,
    build_manifest,
    close,
    config_fingerprint,
    env_snapshot,
    harness_digests,
    main,
    notes,
    start,
    start_allowed,
)
from scripts.lib.plan_from_dict import plan_from_dict  # noqa: E402
from scripts.preflight import Check, summarise  # noqa: E402
from tests.fixtures import FITS, FREE, PILOT, RICH, SERVER, SMALL, plan  # noqa: E402

RESULTS = {
    "selftest": {"status": "pass", "items": 2, "tps": 26.4, "cap_hits": 0, "runaway": 0,
                 "timeout_hits": 0},
    "math500": {"status": "pass", "items": 500, "tps": 25.1, "cap_hits": 3, "runaway": 0,
                "timeout_hits": 1},
}


def rich_plan(selected: tuple[str, ...] = FREE):
    """16.4 GB of weights on a 24 GB card: the window moves, so this run is off-standard."""
    return plan(FITS, selected=selected)


def quiet_plan(selected: tuple[str, ...] = FREE):
    """5.2 GB of weights: everything stays where the standard put it."""
    return plan(SMALL, selected=selected)


class EnvSnapshotTest(unittest.TestCase):
    def test_only_the_data_location_names_are_recorded(self):
        snapshot = env_snapshot({"LCB_DATA": "/data/test6.jsonl", "NLTK_DATA": "/data/nltk",
                                 "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin",
                                 "EDITOR": "vim"})
        self.assertEqual(snapshot, {"CUDA_VISIBLE_DEVICES": "0", "LCB_DATA": "/data/test6.jsonl",
                                    "NLTK_DATA": "/data/nltk"})

    def test_a_prefixed_name_that_looks_like_a_credential_is_dropped(self):
        snapshot = env_snapshot({"LCB_TOKEN": "abc", "HF_API_KEY": "abc",
                                 "BATTERY_HOME": "/srv/battery"})
        self.assertEqual(snapshot, {"BATTERY_HOME": "/srv/battery"})

    def test_the_process_environment_is_used_when_none_is_passed(self):
        self.assertIsInstance(env_snapshot(), dict)


class FingerprintTest(unittest.TestCase):
    def test_the_fingerprint_holds_the_legs_and_the_decisions(self):
        fingerprint = config_fingerprint(rich_plan())
        self.assertEqual(sorted(fingerprint["legs"]), ["math500", "mmlu"])
        self.assertEqual(fingerprint["legs"]["math500"]["cap"], 8192)
        self.assertIn("seed", fingerprint["decisions"])

    def test_a_pilot_only_decision_does_not_invalidate_a_resume(self):
        fast = rich_plan(("hermesagent20",))
        slow = plan(FITS, selected=("hermesagent20",), pilot=replace(PILOT, tps=12.0))
        self.assertNotIn("case_timeout", config_fingerprint(fast)["decisions"])
        quick = [item for item in fast.decisions if item.variable == "case_timeout"][0]
        slower = [item for item in slow.decisions if item.variable == "case_timeout"][0]
        self.assertNotEqual(quick.chosen, slower.chosen)
        self.assertEqual(build_manifest(fast, RICH, SERVER)["config_md5"],
                         build_manifest(slow, RICH, SERVER)["config_md5"])

    def test_a_moved_standard_or_profile_changes_the_fingerprint(self):
        base = rich_plan()
        self.assertNotEqual(config_fingerprint(base),
                            config_fingerprint(replace(base, standard_md5="0000")))

    def test_a_narrower_window_changes_the_fingerprint(self):
        wide = plan(FITS)
        narrow = plan({**FITS, "weights_gb": 20.0})
        self.assertNotEqual(build_manifest(wide, RICH, SERVER)["config_md5"],
                            build_manifest(narrow, RICH, SERVER)["config_md5"])


class ManifestTest(unittest.TestCase):
    def test_the_manifest_records_what_this_run_was(self):
        payload = build_manifest(quiet_plan(), RICH, SERVER)
        self.assertEqual(payload["model"], "tiny-8b")
        self.assertEqual(payload["standard_version"], "v2")
        self.assertEqual(payload["battery_version"], "2.0.0")
        self.assertEqual(payload["stamp_line"], "ON-STANDARD v2")
        self.assertEqual(payload["host"]["gpus"][0]["vram_total_mb"], 24576)
        self.assertEqual(payload["props"]["build_info"], "b4567-abc")

    def test_the_endpoint_reports_how_it_was_served(self):
        self.assertIn("arete-27b.gguf", build_manifest(rich_plan(), RICH, SERVER)["serve_cmdline"])

    def test_an_endpoint_that_reports_nothing_says_so(self):
        quiet = replace(SERVER, props={})
        self.assertEqual(build_manifest(rich_plan(), RICH, quiet)["serve_cmdline"],
                         "not reported by the endpoint")

    def test_every_leg_records_its_harness_digest(self):
        digests = harness_digests(rich_plan())
        self.assertEqual(sorted(digests), ["scripts/bench_math500.py", "scripts/bench_mmlu.py"])
        self.assertTrue(all(len(value) in (0, 32) for value in digests.values()))

    def test_a_written_harness_file_gets_a_real_digest(self):
        digests = harness_digests(plan(FITS, selected=("math500",)))
        self.assertEqual(len(digests["scripts/bench_math500.py"]), 32)

    def test_the_deviation_list_travels_with_the_stamp(self):
        payload = build_manifest(plan(FITS), RICH, SERVER)
        self.assertTrue(payload["stamp_line"].startswith("OFF-STANDARD v2: ctx"))
        self.assertIn("ctx", payload["stamp"])

    def test_the_manifest_survives_json(self):
        payload = json.loads(json.dumps(build_manifest(rich_plan(), RICH, SERVER)))
        self.assertEqual(payload["config"]["legs"]["mmlu"]["thinking"], "on")


class StartTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name) / "run"

    def test_start_writes_the_manifest(self):
        path = start(self.out, rich_plan(), RICH, SERVER)
        self.assertEqual(path.name, MANIFEST_FILE)
        self.assertEqual(json.loads(path.read_text())["model"], "arete-27b")

    def test_a_second_start_needs_resume(self):
        start(self.out, rich_plan(), RICH, SERVER)
        with self.assertRaises(ManifestMismatch):
            start(self.out, rich_plan(), RICH, SERVER)
        self.assertTrue(start(self.out, rich_plan(), RICH, SERVER, resume=True).is_file())

    def test_a_resume_after_the_standard_moved_is_refused(self):
        start(self.out, rich_plan(), RICH, SERVER)
        with self.assertRaises(ManifestMismatch) as caught:
            start(self.out, replace(rich_plan(), standard_md5="beef"), RICH, SERVER, resume=True)
        self.assertIn("standard changed", str(caught.exception))

    def test_a_resume_after_the_configuration_moved_is_refused(self):
        start(self.out, rich_plan(), RICH, SERVER)
        moved = plan({**FITS, "weights_gb": 20.0}, selected=FREE)
        with self.assertRaises(ManifestMismatch) as caught:
            start(self.out, moved, RICH, SERVER, resume=True)
        self.assertIn("configuration changed", str(caught.exception))

    def test_a_recorded_manifest_is_reused_when_one_is_passed_in(self):
        payload = build_manifest(rich_plan(), RICH, SERVER)
        payload["note"] = "written by the orchestrator"
        start(self.out, rich_plan(), RICH, SERVER, manifest=payload)
        self.assertEqual(json.loads((self.out / MANIFEST_FILE).read_text())["note"],
                         "written by the orchestrator")


class StartAllowedTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name) / "run"

    def write_gate(self, ok: bool, standard_md5: str) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / PREFLIGHT_FILE).write_text(json.dumps({"ok": ok,
                                                           "standard_md5": standard_md5}))

    def test_a_run_without_a_gate_is_not_allowed(self):
        allowed, why = start_allowed(self.out, rich_plan())
        self.assertFalse(allowed)
        self.assertIn("preflight", why)

    def test_a_blocked_gate_is_not_allowed(self):
        self.write_gate(False, rich_plan().standard_md5)
        self.assertFalse(start_allowed(self.out, rich_plan())[0])

    def test_a_moved_standard_is_not_allowed(self):
        self.write_gate(True, "old-digest")
        allowed, why = start_allowed(self.out, rich_plan())
        self.assertFalse(allowed)
        self.assertIn("standard moved", why)

    def test_a_clean_gate_allows_the_run(self):
        self.write_gate(True, rich_plan().standard_md5)
        self.assertEqual(start_allowed(self.out, rich_plan()), (True, "preflight passed and the "
                                                                      "standard is unchanged"))


class NotesTest(unittest.TestCase):
    def test_the_notes_table_carries_every_leg_and_the_totals(self):
        text = notes(plan(FITS, selected=("selftest", "math500")), RESULTS)
        self.assertIn("| math500 | pass | 500 | 25.1 | 3 | 0 | 1 |", text)
        self.assertIn("| selftest | pass | 2 | 26.4 | 0 | 0 | 0 |", text)
        self.assertIn("Totals: 502 items, 3 cap hits, 0 runaways, 1 timeouts.", text)

    def test_a_leg_that_never_ran_says_so(self):
        self.assertIn("not run", notes(rich_plan(), {}))

    def test_the_notes_end_in_the_comparability_stamp(self):
        text = notes(quiet_plan(), RESULTS)
        self.assertIn("Stamp: `ON-STANDARD v2`", text)
        self.assertIn("these numbers compare with every other ON-STANDARD run", text)

    def test_an_off_standard_run_lists_what_moved_and_what_was_skipped(self):
        text = notes(plan(FITS), RESULTS)
        self.assertIn("OFF-STANDARD v2: ctx", text)
        self.assertIn("Off the standard, deliberately:", text)
        self.assertIn("- lcb:", text)

    def test_warnings_are_carried_into_the_notes(self):
        self.assertIn("- warning: ", notes(plan(FITS), RESULTS))

    def test_a_run_with_no_results_reads_the_results_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            (target / RESULTS_FILE).write_text(json.dumps(RESULTS))
            path = close(target, rich_plan())
            self.assertEqual(path.name, NOTES_FILE)
            self.assertIn("25.1", path.read_text())

    def test_the_notes_are_ascii(self):
        notes(plan(FITS), RESULTS).encode("ascii")


class PlanFromDictTest(unittest.TestCase):
    def test_a_plan_round_trips_through_json(self):
        original = plan(FITS)
        restored = plan_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.stamp_line, original.stamp_line)
        self.assertEqual(restored.render(), original.render())
        self.assertEqual([leg.cap for leg in restored.legs], [leg.cap for leg in original.legs])
        self.assertEqual(restored.stamp, original.stamp)

    def test_prerequisites_survive_the_round_trip(self):
        original = plan(FITS, selected=FREE)
        restored = plan_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual([item.name for item in restored.legs[0].dependencies],
                         [item.name for item in original.legs[0].dependencies])

    def test_a_leg_that_is_not_in_the_catalog_is_refused(self):
        payload = json.loads(json.dumps(plan(FITS).to_dict()))
        payload["legs"][0]["id"] = "imaginary"
        with self.assertRaises(KeyError):
            plan_from_dict(payload, legs=[])

    def test_a_manifest_can_be_rebuilt_from_the_written_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            original = rich_plan()
            (target / "run-plan.json").write_text(json.dumps(original.to_dict()))
            restored = plan_from_dict(json.loads((target / "run-plan.json").read_text()))
            self.assertEqual(build_manifest(restored, RICH, SERVER)["config_md5"],
                             build_manifest(original, RICH, SERVER)["config_md5"])


class CliTest(unittest.TestCase):
    def write_plan(self, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "run-plan.json").write_text(json.dumps(rich_plan().to_dict()))

    def test_close_writes_the_notes(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_plan(Path(folder))
            self.assertEqual(main(["close", "--out", folder]), 0)
            self.assertTrue((Path(folder) / NOTES_FILE).is_file())

    def test_show_prints_the_stamp(self):
        with tempfile.TemporaryDirectory() as folder:
            start(Path(folder), rich_plan(), RICH, SERVER)
            self.assertEqual(main(["show", "--out", folder]), 0)

    def test_close_without_a_plan_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(main(["close", "--out", folder]), 1)

    def test_the_summary_counts_only_unhard_misses_as_warnings(self):
        checks = [Check("a", True, "fine"), Check("b", False, "warn", hard=False)]
        self.assertIn("warn", summarise(checks))


if __name__ == "__main__":
    unittest.main()
