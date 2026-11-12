"""Prerequisite resolution: present, absent, and never silently absent."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import HostFacts, ToolFact  # noqa: E402
from scripts.lib.legs import legs_by_id, load_legs  # noqa: E402
from scripts.lib.prereqs import (  # noqa: E402
    Judge,
    resolve_dependency,
    resolve_prereqs,
    thinking_available,
    unavailable,
    verify_bundled_data,
)
from scripts.lib.server_probe import ServerFacts  # noqa: E402

STANDARD = load_toml("standard.toml")
LEGS = legs_by_id(load_legs(STANDARD))


def host_with(*tool_names: str) -> HostFacts:
    return HostFacts(ram_total_gb=64.0, ram_available_gb=60.0,
                     tools=tuple(ToolFact(name, f"/usr/bin/{name}", "1.0")
                                 for name in tool_names))


class ToolDependencyTest(unittest.TestCase):
    def test_present_tool_reports_its_path_and_version(self):
        dependency = resolve_dependency("docker_sandbox", STANDARD, host_with("docker"))
        self.assertTrue(dependency.available)
        self.assertIn("/usr/bin/docker", dependency.detail)
        self.assertIn("1.0", dependency.detail)

    def test_absent_tool_says_so(self):
        dependency = resolve_dependency("docker_sandbox", STANDARD, host_with())
        self.assertFalse(dependency.available)
        self.assertIn("not on PATH", dependency.detail)

    def test_each_named_tool_maps_to_its_own_check(self):
        host = host_with("benchlocal-cli", "tool-eval-bench")
        self.assertTrue(resolve_dependency("benchlocal_cli", STANDARD, host).available)
        self.assertTrue(resolve_dependency("tool_eval_bench", STANDARD, host).available)
        self.assertFalse(resolve_dependency("docker_sandbox", STANDARD, host).available)


class PathDependencyTest(unittest.TestCase):
    def test_absent_lcb_data_names_the_fetch_command(self):
        dependency = resolve_dependency("lcb_data", STANDARD, host_with())
        self.assertFalse(dependency.available)
        self.assertIn("fetch", dependency.detail)

    def test_env_override_points_at_the_data(self):
        with tempfile.TemporaryDirectory() as folder:
            os.environ["LCB_SRC"] = folder
            try:
                dependency = resolve_dependency("lcb_source", STANDARD, host_with())
            finally:
                del os.environ["LCB_SRC"]
            self.assertTrue(dependency.available)
            self.assertEqual(dependency.detail, folder)

    def test_env_override_that_does_not_exist_is_not_available(self):
        os.environ["LCB_SRC"] = "/nonexistent/lcb"
        try:
            dependency = resolve_dependency("lcb_source", STANDARD, host_with())
        finally:
            del os.environ["LCB_SRC"]
        self.assertFalse(dependency.available)

    def test_a_file_where_a_directory_belongs_is_reported(self):
        with tempfile.NamedTemporaryFile() as handle:
            os.environ["LCB_SRC"] = handle.name
            try:
                dependency = resolve_dependency("lcb_source", STANDARD, host_with())
            finally:
                del os.environ["LCB_SRC"]
        self.assertFalse(dependency.available)
        self.assertIn("not a directory", dependency.detail)

    def test_hash_mismatch_fails_the_dependency(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
            handle.write("not the pinned file\n")
            handle.flush()
            os.environ["LCB_DATA"] = handle.name
            try:
                dependency = resolve_dependency("lcb_data", STANDARD, host_with())
            finally:
                del os.environ["LCB_DATA"]
        self.assertFalse(dependency.available)
        self.assertIn("hash mismatch", dependency.detail)

    def test_a_directory_that_is_missing_is_not_a_passing_check(self):
        dependency = resolve_dependency("lcb_source", STANDARD, host_with())
        self.assertFalse(dependency.available)

    def test_an_entry_not_declared_in_the_standard_is_reported_as_such(self):
        dependency = resolve_dependency("lcb_data", {}, host_with())
        self.assertFalse(dependency.available)
        self.assertIn("not declared in standard.toml", dependency.detail)

    def test_a_directory_where_the_data_file_belongs_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            os.environ["LCB_DATA"] = folder
            try:
                dependency = resolve_dependency("lcb_data", STANDARD, host_with())
            finally:
                del os.environ["LCB_DATA"]
        self.assertFalse(dependency.available)
        self.assertIn("is not a file", dependency.detail)


class NltkDependencyTest(unittest.TestCase):
    def test_nltk_is_reported_either_way_with_a_reason(self):
        dependency = resolve_dependency("nltk_data", STANDARD, host_with())
        self.assertTrue(dependency.detail)
        if not dependency.available:
            self.assertIn("nltk", dependency.detail)

    def setUp(self):
        # The punkt check only runs when the nltk module imports, so the tests that
        # exercise the corpus search stand in for an installed nltk.
        module = ModuleType("nltk")
        module.__spec__ = ModuleSpec("nltk", None)
        self._previous = sys.modules.get("nltk")
        sys.modules["nltk"] = module
        self.addCleanup(self._uninstall_fake_nltk)

    def _uninstall_fake_nltk(self):
        if self._previous is None:
            sys.modules.pop("nltk", None)
        else:
            sys.modules["nltk"] = self._previous

    def test_a_punkt_corpus_under_nltk_data_is_available(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "tokenizers", "punkt").mkdir(parents=True)
            os.environ["NLTK_DATA"] = folder
            try:
                dependency = resolve_dependency("nltk_data", STANDARD, host_with())
            finally:
                del os.environ["NLTK_DATA"]
        self.assertTrue(dependency.available)
        self.assertTrue(dependency.detail.endswith("tokenizers/punkt"))

    def test_nltk_without_a_punkt_corpus_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            os.environ["NLTK_DATA"] = folder
            try:
                dependency = resolve_dependency("nltk_data", STANDARD, host_with())
            finally:
                del os.environ["NLTK_DATA"]
        self.assertFalse(dependency.available)
        self.assertIn("no punkt corpus", dependency.detail)


class JudgeDependencyTest(unittest.TestCase):
    def test_unconfigured_judge_is_unavailable(self):
        dependency = resolve_dependency("judge_endpoint", STANDARD, host_with(), Judge())
        self.assertFalse(dependency.available)
        self.assertIn("cannot be scored", dependency.detail)

    def test_a_model_cannot_judge_itself(self):
        judge = Judge(model="arete-27b", endpoint="http://host/v1", under_test="arete-27b")
        dependency = resolve_dependency("judge_endpoint", STANDARD, host_with(), judge)
        self.assertFalse(dependency.available)
        self.assertIn("model under test", dependency.detail)

    def test_a_separate_judge_is_available(self):
        judge = Judge(model="deckard-40b", endpoint="http://host/v1", under_test="arete-27b")
        dependency = resolve_dependency("judge_endpoint", STANDARD, host_with(), judge)
        self.assertTrue(dependency.available)
        self.assertIn("deckard-40b", dependency.detail)


class UnknownDependencyTest(unittest.TestCase):
    def test_an_unimplemented_check_is_reported_as_unavailable(self):
        dependency = resolve_dependency("made_up_thing", STANDARD, host_with())
        self.assertFalse(dependency.available)
        self.assertIn("no check implemented", dependency.detail)


class LegPrereqTest(unittest.TestCase):
    def test_a_leg_with_no_prereqs_resolves_to_nothing(self):
        self.assertEqual(resolve_prereqs(LEGS["math500"], STANDARD, host_with()), ())

    def test_lcb_resolves_source_and_data(self):
        dependencies = resolve_prereqs(LEGS["lcb"], STANDARD, host_with())
        self.assertEqual([item.name for item in dependencies], ["lcb_source", "lcb_data"])
        self.assertEqual(len(unavailable(dependencies)), 2)

    def test_judge_configuration_reaches_the_writing_leg(self):
        judge = Judge(model="judge", endpoint="http://host/v1", under_test="model")
        dependencies = resolve_prereqs(LEGS["writing"], STANDARD, host_with(), judge)
        self.assertEqual(unavailable(dependencies), ())


class ThinkingAvailabilityTest(unittest.TestCase):
    def test_a_working_switch_is_reported_as_working(self):
        available, why = thinking_available(ServerFacts(reachable=True, thinking_switch=True))
        self.assertTrue(available)
        self.assertIn("changes the output", why)

    def test_a_deaf_endpoint_is_reported_as_deaf(self):
        available, why = thinking_available(ServerFacts(reachable=True, thinking_switch=False))
        self.assertFalse(available)
        self.assertIn("ignores", why)

    def test_an_unmeasured_switch_is_assumed_to_work_and_says_so(self):
        available, why = thinking_available(ServerFacts(reachable=True, thinking_switch=None))
        self.assertTrue(available)
        self.assertIn("assumed", why)


class BundledDataTest(unittest.TestCase):
    def test_every_bundled_dataset_matches_its_pinned_hash(self):
        self.assertEqual(verify_bundled_data(STANDARD), [])

    def test_a_missing_bundled_dataset_is_reported(self):
        broken = {"data": {"bundled": [{"file": "data/does_not_exist.json",
                                       "sha256_16": "0" * 16}]}}
        problems = verify_bundled_data(broken)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing", problems[0])

    def test_a_hash_mismatch_is_reported(self):
        real = [entry for entry in STANDARD["data"]["bundled"]
                if entry["file"].endswith("math500.json")][0]
        broken = {"data": {"bundled": [dict(real, sha256_16="deadbeefdeadbeef")]}}
        problems = verify_bundled_data(broken)
        self.assertEqual(len(problems), 1)
        self.assertIn("hash", problems[0])


if __name__ == "__main__":
    unittest.main()
