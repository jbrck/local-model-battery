"""hermesagent20 must find benchlocal-cli on whatever host the battery lands.

The leg hardcoded one machine's venv path and died with FileNotFoundError before a
single scenario ran. These pin the resolution order so that cannot happen again, and
confirm no machine-local path ships as a last-resort guess.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from scripts.adapters import hermesagent20 as adapter
from scripts.lib.harness import LegArgs


class RunnerResolutionTest(unittest.TestCase):
    def test_an_explicit_bin_wins(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": "/opt/x/benchlocal-cli"}):
            self.assertEqual(adapter.runner(), "/opt/x/benchlocal-cli")

    def test_the_path_wins_when_no_explicit_bin_is_set(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": ""}):
            with mock.patch.object(adapter.shutil, "which",
                                   return_value="/usr/bin/benchlocal-cli"):
                self.assertEqual(adapter.runner(), "/usr/bin/benchlocal-cli")

    def test_nothing_resolved_returns_empty_rather_than_a_machine_local_guess(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": ""}):
            with mock.patch.object(adapter.shutil, "which", return_value=None):
                self.assertEqual(adapter.runner(), "")

    def test_the_command_starts_with_the_resolved_runner(self):
        args = LegArgs(model="m", endpoint="http://e/v1", api_key="k", cap=8192,
                       ctx=131072, thinking="on", temperature=0.0, seed=1, timeout_s=600)
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": "/opt/x/benchlocal-cli"}):
            command = adapter.build_command(args, "/tmp/x-run.json", None)
        self.assertEqual(command[0], "/opt/x/benchlocal-cli")
        self.assertEqual(command[1], "run")


class EndpointRoutingTest(unittest.TestCase):
    """Sandboxed packs cannot see a loopback-bound engine; the bridge address is the route."""

    def test_the_run_endpoint_passes_through_by_default(self):
        with mock.patch.dict(os.environ, {"HERMESAGENT20_ENDPOINT": ""}):
            self.assertEqual(adapter.model_endpoint("http://127.0.0.1:8124/v1"),
                             "http://127.0.0.1:8124/v1")

    def test_an_override_is_used_for_the_command_line(self):
        args = LegArgs(model="m", endpoint="http://127.0.0.1:8124/v1", api_key="k",
                       cap=8192, ctx=131072, thinking="on", temperature=0.0, seed=1,
                       timeout_s=600)
        with mock.patch.dict(os.environ,
                             {"HERMESAGENT20_ENDPOINT": "http://172.17.0.1:8124/v1"}):
            command = adapter.build_command(args, "/tmp/x-run.json", None)
        self.assertEqual(command[command.index("--endpoint") + 1],
                         "http://172.17.0.1:8124/v1")


def unreachable_pack() -> dict:
    detail = "HA-01: model endpoint unreachable from sandbox: no response within 5s"
    return {"pack_id": adapter.PACK_ID, "catalog_scenario_count": 2,
            "scenarios": [{"id": f"HA-0{n}", "failure_mode": "server_error", "detail": detail}
                          for n in (1, 2)]}


class UnreachableEndpointTest(unittest.TestCase):
    """A sandbox that cannot see the engine is a setup problem, never a 0% score."""

    def test_all_infra_failures_that_name_the_endpoint_are_unreachable(self):
        self.assertTrue(adapter.endpoint_unreachable(unreachable_pack()))

    def test_one_real_scenario_means_the_engine_was_reached(self):
        pack = unreachable_pack()
        pack["scenarios"][1] = {"id": "HA-02", "failure_mode": "", "passed": True}
        self.assertFalse(adapter.endpoint_unreachable(pack))

    def test_a_server_error_with_another_cause_is_not_called_unreachable(self):
        pack = unreachable_pack()
        pack["scenarios"][0]["detail"] = "engine returned 500"
        self.assertFalse(adapter.endpoint_unreachable(pack))

    def test_an_empty_pack_is_not_unreachable(self):
        self.assertFalse(adapter.endpoint_unreachable({"scenarios": []}))

    def test_the_stale_run_and_journal_are_set_aside_so_a_rerun_starts_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json = os.path.join(tmp, "x-run.json")
            for path in (save_json, adapter.journal_path(save_json)):
                with open(path, "w") as handle:
                    handle.write("{}")
            adapter.set_stale_run_aside(save_json)
            self.assertFalse(os.path.exists(save_json))
            self.assertFalse(os.path.exists(adapter.journal_path(save_json)))
            self.assertTrue(os.path.exists(save_json + ".unreachable"))

    def test_the_leg_reports_not_run_and_names_the_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_json = os.path.join(tmp, "hermesagent20-run.json")
            with open(save_json, "w") as handle:
                json.dump({"packs": [unreachable_pack()]}, handle)
            argv = ["--model", "m", "--endpoint", "http://127.0.0.1:1/v1",
                    "--output-dir", tmp]

            def rewrite(*_args, **_kwargs):
                with open(save_json, "w") as handle:
                    json.dump({"packs": [unreachable_pack()]}, handle)
                return mock.Mock(returncode=0)

            with mock.patch.dict(os.environ, {"HERMESAGENT20_ENDPOINT": ""}):
                with mock.patch.object(adapter.subprocess, "run", side_effect=rewrite) as run:
                    adapter.main(argv)
            self.assertTrue(run.called)
            result = [name for name in os.listdir(tmp) if name.endswith(".result.json")]
            with open(os.path.join(tmp, result[0])) as handle:
                block = json.load(handle)
            self.assertEqual(block["status"], "not run")
            self.assertEqual(block["scored"], 0)
            self.assertIn("HERMESAGENT20_ENDPOINT", block["reason"])
            self.assertTrue(os.path.exists(save_json + ".unreachable"))
