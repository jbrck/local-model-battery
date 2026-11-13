"""The planning CLI: the flags the skill hands the operator, and what they build.

Host probing, endpoint probing and the pilot are all faked. What is under test is the wiring:
which probe is skipped, which legs get selected, and that the sheet comes out printable.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import plan_run as cli  # noqa: E402
from tests import fixtures  # noqa: E402
from tests.stub_json import JsonStub, completion  # noqa: E402

PROFILE = """id = "cli-test"
weights_gb = 16.4
quant = "Q4_K_M"

[model]
n_layers = 64
n_kv_heads = 8
head_dim = 128
kv_type = "q8_0"
"""


class ProbeWiringTest(unittest.TestCase):
    """The probe path: the gate re-checks what the plan saw, so the probes have to come back."""

    ROUTES = {
        "/props": (200, {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 2,
                         "chat_template": "{{ enable_thinking }}"}),
        "/v1/models": (200, {"data": [{"id": "stub"}]}),
        "/v1/chat/completions": (200, completion("ok", tokens=120)),
    }

    def _args(self, path: Path, endpoint: str, no_probe: bool) -> argparse.Namespace:
        # Built through the real parser: the probe path reads whatever the parser declares, so
        # a hand-assembled Namespace stops matching the CLI the moment a flag is added.
        argv = ["--profile", str(path), "--endpoint", endpoint, "--legs", "math500"]
        if no_probe:
            argv.append("--no-probe")
        return cli.build_parser().parse_args(argv)

    def test_the_probe_path_hands_back_host_endpoint_and_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli-test.toml"
            path.write_text(PROFILE)
            with JsonStub(dict(self.ROUTES)) as stub:
                plan, host, server, pilot = cli.probe_from_args(
                    self._args(path, stub.endpoint, no_probe=False))
        self.assertEqual(plan.model, "cli-test")
        self.assertTrue(server.reachable)
        self.assertEqual(server.ctx, 65536)
        self.assertTrue(pilot.usable)
        self.assertGreater(pilot.generated_tokens, 0)
        self.assertGreater(host.ram_total_gb, 0)

    def test_the_probe_path_stands_down_on_no_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli-test.toml"
            path.write_text(PROFILE)
            plan, _, server, pilot = cli.probe_from_args(
                self._args(path, "http://127.0.0.1:1/v1", no_probe=True))
        self.assertFalse(server.reachable)
        self.assertFalse(pilot.usable)
        self.assertEqual(pilot.tps, 0.0)
        self.assertTrue(plan is not None)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cli-test.toml"
        self.path.write_text(PROFILE)
        patcher = mock.patch.object(cli, "probe_host", return_value=fixtures.RICH)
        self.host = patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, **changes) -> argparse.Namespace:
        base = dict(profile=str(self.path), model_file="", model_dir="",
                    endpoint="http://box:11434/v1", model="served-id",
                    api_key="not-needed", legs="", judge_model="", judge_endpoint="",
                    no_probe=True, json=False)
        base.update(changes)
        return argparse.Namespace(**base)

    def test_a_planning_run_hands_back_every_probe_it_used(self):
        plan, host, server, pilot = cli.probe_from_args(self.args())
        self.assertIs(host, fixtures.RICH)
        self.assertEqual(plan.model, "cli-test")
        self.assertTrue(plan.legs)
        self.assertFalse(server.reachable)
        self.assertTrue(pilot.stalled)

    def test_probing_can_be_turned_off_without_losing_the_plan(self):
        plan, _, server, pilot = cli.probe_from_args(self.args())
        self.assertIn("probing was disabled", server.errors[0])
        self.assertIn("probing was disabled", pilot.errors[0])
        self.assertTrue(plan.stamp_line.startswith(("ON-STANDARD", "OFF-STANDARD")))

    def test_a_live_probe_hands_the_real_endpoint_and_pilot_through(self):
        with mock.patch.object(cli, "probe_server", return_value=fixtures.SERVER) as probe, \
                mock.patch.object(cli, "run_pilot", return_value=fixtures.PILOT):
            _, _, server, pilot = cli.probe_from_args(self.args(no_probe=False))
        self.assertIs(server, fixtures.SERVER)
        self.assertIs(pilot, fixtures.PILOT)
        probe.assert_called_once()

    def test_leg_selection_narrows_the_run(self):
        plan, _, _, _ = cli.probe_from_args(self.args(legs="math500, mmlu"))
        self.assertEqual([planned.leg.id for planned in plan.legs], ["math500", "mmlu"])

    def test_no_selection_means_every_leg_this_box_can_run(self):
        plan, _, _, _ = cli.probe_from_args(self.args())
        self.assertGreater(len(plan.legs) + len(plan.skips), 10)

    def test_a_profile_without_an_id_falls_back_to_the_served_name(self):
        bare = Path(self.tmp.name) / "bare.toml"
        bare.write_text("weights_gb = 8.0\n")
        plan, _, _, _ = cli.probe_from_args(self.args(profile=str(bare), model=""))
        self.assertEqual(plan.model, "bare")

    def test_a_profiles_own_id_labels_the_run_even_when_a_model_is_named(self):
        # --model is what the endpoint serves; the profile's id is the row in the table.
        plan, _, _, _ = cli.probe_from_args(self.args())
        self.assertEqual(plan.model, "cli-test")

    def test_the_judge_flags_reach_the_planner(self):
        marker = object()
        with mock.patch.object(cli, "plan_run", return_value=marker) as planner:
            plan, _, _, _ = cli.probe_from_args(self.args(
                judge_model="judge-x", judge_endpoint="http://judge:11434/v1"))
        judge = planner.call_args.args[7]
        self.assertIs(plan, marker)
        self.assertEqual(judge.model, "judge-x")
        self.assertEqual(judge.endpoint, "http://judge:11434/v1")
        self.assertEqual(judge.under_test, "served-id")

    def test_json_output_is_the_plan_itself(self):
        with mock.patch.object(cli, "probe_from_args",
                               return_value=(fixtures.plan(fixtures.FITS, selected=("math500",)),
                                             fixtures.RICH, fixtures.SERVER, fixtures.PILOT)):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(["--profile", str(self.path), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("stamp", payload)
        self.assertIn("decisions", payload)

    def test_the_sheet_prints_the_decisions_and_the_legs(self):
        with mock.patch.object(cli, "probe_from_args",
                               return_value=(fixtures.plan(fixtures.FITS, selected=("math500",)),
                                             fixtures.RICH, fixtures.SERVER, fixtures.PILOT)):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                cli.main(["--profile", str(self.path)])
        sheet = buffer.getvalue()
        self.assertIn("Legs", sheet)
        self.assertIn("math500", sheet)
        self.assertIn("Set by the standard here", sheet)


if __name__ == "__main__":
    unittest.main()
