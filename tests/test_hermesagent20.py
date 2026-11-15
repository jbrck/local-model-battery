"""hermesagent20 must find benchlocal-cli on whatever host the battery lands.

The leg hardcoded one machine's venv path and died with FileNotFoundError before a
single scenario ran. These pin the resolution order so that cannot happen again.
"""
import os
import unittest
from unittest import mock

from scripts.adapters import hermesagent20 as adapter
from scripts.lib.harness import LegArgs

FLEET_DEFAULT = "/mnt/models/bench/bench-venv/bin/benchlocal-cli"


class RunnerResolutionTest(unittest.TestCase):
    def test_an_explicit_bin_wins(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": "/opt/x/benchlocal-cli"}):
            self.assertEqual(adapter.runner(), "/opt/x/benchlocal-cli")

    def test_the_path_wins_over_the_fleet_default(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": ""}):
            with mock.patch.object(adapter.shutil, "which",
                                   return_value="/usr/bin/benchlocal-cli"):
                self.assertEqual(adapter.runner(), "/usr/bin/benchlocal-cli")

    def test_the_fleet_default_is_used_when_nothing_else_does(self):
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": ""}):
            with mock.patch.object(adapter.shutil, "which", return_value=None):
                self.assertEqual(adapter.runner(), FLEET_DEFAULT)

    def test_the_command_starts_with_the_resolved_runner(self):
        args = LegArgs(model="m", endpoint="http://e/v1", api_key="k", cap=8192,
                       ctx=131072, thinking="on", temperature=0.0, seed=1, timeout_s=600)
        with mock.patch.dict(os.environ, {"BENCHLOCAL_BIN": "/opt/x/benchlocal-cli"}):
            command = adapter.build_command(args, "/tmp/x-run.json", None)
        self.assertEqual(command[0], "/opt/x/benchlocal-cli")
        self.assertEqual(command[1], "run")
