"""hermesagent20 must find benchlocal-cli on whatever host the battery lands.

The leg hardcoded one machine's venv path and died with FileNotFoundError before a
single scenario ran. These pin the resolution order so that cannot happen again, and
confirm no machine-local path ships as a last-resort guess.
"""
import os
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
