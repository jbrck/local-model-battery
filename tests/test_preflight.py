"""The gate between a plan and a run: what blocks, what only warns, what gets recorded."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.env_probe import Gpu, HostFacts  # noqa: E402
from scripts.lib.model_probe import ModelShape  # noqa: E402
from scripts.lib.plan_run import plan_run  # noqa: E402
from scripts.preflight import (  # noqa: E402
    Check,
    PREFLIGHT_FILE,
    blocked,
    check_data,
    check_endpoint,
    check_host,
    check_out_dir,
    check_pilot,
    run_preflight,
    summarise,
    write_preflight,
)
from scripts.lib.prereqs import Dependency  # noqa: E402
from tests.fixtures import (  # noqa: E402
    DEAD_PILOT,
    DEAF,
    FITS,
    FREE,
    PILOT,
    RICH,
    SERVER,
    plan,
    runnable,
)

# A card that is already busy and a host short on RAM: 22 GB of 24 GB taken, 10 GB free to spill
OCCUPIED = replace(RICH, gpus=(Gpu(0, "RTX 3090", 24576, 22528, 350.0),),
                   ram_available_gb=10.0)
# 28 GB of weights on a 24 GB card: the tier that spills weights into RAM.
SPILL = {**FITS, "weights_gb": 28.0}


def named(checks, name: str) -> Check:
    rows = [check for check in checks if check.name == name]
    assert len(rows) == 1, f"expected one check named {name}, found {len(rows)}"
    return rows[0]


def with_broken_dependency(result, leg_id: str = "math500"):
    """The same plan, as if a leg's data disappeared between planning and starting."""
    broken = replace(runnable(result, leg_id),
                     dependencies=(Dependency("lcb_data", False, "no test6.jsonl"),))
    legs = tuple(broken if item.id == leg_id else item for item in result.legs)
    return replace(result, legs=legs)


class DataCheckTest(unittest.TestCase):
    def test_every_planned_leg_reports_its_prerequisites(self):
        result = plan(FITS, selected=FREE)
        checks = check_data(result)
        self.assertEqual([check.name for check in checks], ["data:math500", "data:mmlu"])
        self.assertTrue(all(check.ok for check in checks))
        self.assertEqual(named(checks, "data:math500").detail, "all present")

    def test_a_dependency_that_vanished_blocks_the_leg(self):
        check = named(check_data(with_broken_dependency(plan(FITS))), "data:math500")
        self.assertFalse(check.ok)
        self.assertTrue(check.blocks)
        self.assertIn("no test6.jsonl", check.detail)

    def test_a_skipped_leg_is_not_checked(self):
        self.assertNotIn("data:lcb", [check.name for check in check_data(plan(FITS))])


class EndpointCheckTest(unittest.TestCase):
    def test_a_serving_endpoint_passes(self):
        checks = check_endpoint(plan(FITS), SERVER)
        self.assertTrue(named(checks, "endpoint").ok)
        self.assertTrue(named(checks, "endpoint:ctx").ok)
        self.assertTrue(named(checks, "endpoint:slots").ok)

    def test_an_unreachable_endpoint_blocks(self):
        dead = replace(SERVER, reachable=False, errors=("connection refused",))
        check = named(check_endpoint(plan(FITS), dead), "endpoint")
        self.assertTrue(check.blocks)
        self.assertIn("connection refused", check.detail)

    def test_a_window_narrower_than_the_plan_blocks(self):
        narrow = replace(SERVER, ctx=8192)
        check = named(check_endpoint(plan(FITS), narrow), "endpoint:ctx")
        self.assertTrue(check.blocks)
        self.assertIn("8192", check.detail)

    def test_an_endpoint_that_ignores_thinking_blocks_a_plan_that_switches_it(self):
        switch = [item for item in plan({**FITS, "thinking": "off"}, server=DEAF).decisions
                  if item.variable == "thinking"][0]
        self.assertEqual(switch.chosen, "native")
        checks = check_endpoint(plan({**FITS, "thinking": "off"}), DEAF)
        self.assertTrue(named(checks, "endpoint:thinking").blocks)

    def test_fewer_slots_than_the_standard_only_warns(self):
        check = named(check_endpoint(plan(FITS), DEAF), "endpoint:slots")
        self.assertFalse(check.ok)
        self.assertFalse(check.blocks)
        self.assertIn("against the standard's 4", check.detail)


class HostCheckTest(unittest.TestCase):
    def test_a_free_card_leaves_room(self):
        checks = check_host(plan(FITS), RICH)
        self.assertTrue(named(checks, "host:fits").ok)
        self.assertNotIn("host:ram", [check.name for check in checks])

    def test_a_model_that_does_not_fit_blocks(self):
        fits = plan(FITS, host=OCCUPIED).fit
        self.assertEqual(fits.tier, "T4")
        self.assertTrue(named(check_host(plan(FITS, host=OCCUPIED), OCCUPIED), "host:fits").blocks)

    def test_weights_that_spill_to_ram_are_checked_against_the_ram(self):
        stale = plan(SPILL)
        self.assertEqual(stale.fit.tier, "T2")
        self.assertTrue(named(check_host(stale, RICH), "host:ram").ok)
        tight = replace(RICH, ram_available_gb=18.0)
        ram = named(check_host(stale, tight), "host:ram")
        self.assertFalse(ram.ok)
        self.assertIn("4.0 GB of weights", ram.detail)
        self.assertIn("20 GB", ram.detail)

    def test_other_processes_holding_vram_only_warn(self):
        busy = replace(RICH, compute_apps=({"pid": 42, "name": "llama-server"},))
        check = named(check_host(plan(FITS, host=busy), busy), "host:idle")
        self.assertFalse(check.blocks)
        self.assertIn("1 compute process", check.detail)
        self.assertIn("counted against it", check.detail)

    def test_the_vram_credited_to_the_run_is_spelled_out(self):
        """The operator should see that the tier rests on crediting the loaded model to the run."""
        host = replace(RICH, gpus=(Gpu(0, "RTX 3090", 24576, 19670, 350.0),),
                       compute_apps=({"pid": "1748", "used_memory_mb": 19414.0},))
        shape = ModelShape(arch="qwen35", n_layers=17, n_kv_heads=4, head_dim=256,
                           kv_elements_per_token=4352, weights_bytes=int(4.25 * 1024 ** 3),
                           source="gguf", path="/mnt/models/swift/q4.gguf")
        checks = check_host(plan(FITS, host=host, shape=shape, serving_pid=1748), host)
        credit = named(checks, "host:credit")
        self.assertFalse(credit.blocks)
        self.assertIn("credited to this run", credit.detail)

    def test_a_run_that_credits_nothing_says_nothing_about_credits(self):
        host = replace(RICH, compute_apps=({"pid": "8248", "used_memory_mb": 256.0},))
        keys = {check.name for check in check_host(plan(FITS, host=host), host)}
        self.assertNotIn("host:credit", keys)

    def test_swap_use_warns_about_noisy_timing(self):
        swapped = replace(RICH, swap_used_gb=8.0)
        self.assertIn("swap", named(check_host(plan(FITS, host=swapped), swapped), "host:swap").detail)

    def test_a_host_without_gpu_facts_still_gates(self):
        bare = HostFacts(ram_total_gb=32.0, ram_available_gb=28.0, gpu_error="no nvidia-smi")
        self.assertTrue(all(check.ok for check in check_host(plan(FITS, host=bare), bare)))


class PilotCheckTest(unittest.TestCase):
    def test_a_measured_rate_passes(self):
        self.assertEqual(named(check_pilot(PILOT), "pilot").detail,
                         "27.0 tok/s over 3 requests")

    def test_a_stalled_pilot_blocks_because_every_budget_depends_on_it(self):
        check = named(check_pilot(DEAD_PILOT), "pilot")
        self.assertTrue(check.blocks)
        self.assertIn("HTTP 500", check.detail)

    def test_a_stalled_pilot_without_an_error_string_still_reads(self):
        self.assertIn("all requests failed", named(check_pilot(replace(DEAD_PILOT, errors=())),
                                                   "pilot").detail)


class OutDirCheckTest(unittest.TestCase):
    def test_no_out_dir_means_nothing_to_check(self):
        self.assertEqual(check_out_dir(None, resume=False), [])

    def test_an_existing_run_dir_has_to_be_resumed_on_purpose(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            self.assertTrue(check_out_dir(target, False)[0].ok)
            target.mkdir()
            self.assertTrue(check_out_dir(target, resume=False)[0].blocks)
            self.assertTrue(check_out_dir(target, resume=True)[0].ok)

    def test_the_refusal_names_the_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            target.mkdir()
            self.assertIn("--resume", check_out_dir(target, False)[0].detail)


class GateTest(unittest.TestCase):
    def test_a_clean_host_has_nothing_blocking(self):
        checks = run_preflight(plan(FITS, selected=FREE), RICH, SERVER, PILOT)
        self.assertEqual(blocked(checks), [])
        self.assertIn("ok   endpoint", summarise(checks))

    def test_a_stalled_pilot_and_a_busy_card_are_both_named(self):
        checks = run_preflight(plan(FITS, host=OCCUPIED), OCCUPIED, SERVER, DEAD_PILOT)
        self.assertEqual({check.name for check in blocked(checks)}, {"host:fits", "pilot"})
        self.assertIn("MISS", summarise(checks))

    def test_the_written_gate_carries_the_stamp_and_the_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            result = plan(FITS, selected=FREE)
            path = write_preflight(target, result, run_preflight(result, RICH, SERVER, PILOT))
            self.assertEqual(path.name, PREFLIGHT_FILE)
            payload = json.loads(path.read_text())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["model"], "arete-27b")
            self.assertEqual(payload["standard_md5"], result.standard_md5)
            self.assertEqual(payload["battery_version"], "2.0.0")

    def test_a_blocked_gate_is_recorded_as_not_ok(self):
        with tempfile.TemporaryDirectory() as folder:
            result = plan(FITS)
            checks = run_preflight(result, RICH, SERVER, DEAD_PILOT, Path(folder) / "run")
            payload = json.loads(
                write_preflight(Path(folder) / "run", result, checks).read_text())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["checks"][0]["name"], "out")


class CliTest(unittest.TestCase):
    """The CLI probes the machine it is on, so the assertions are about what it writes."""

    def setUp(self):
        from tests.stub_json import JsonStub, completion
        props = {"model_path": "stub-27b.gguf", "n_ctx": 131072, "total_slots": 4,
                 "build_info": "b4567-abc"}
        self.stub = JsonStub({"/v1/models": (200, {"data": [{"id": "stub-27b"}]}),
                              "/props": (200, props), "/v1/props": (200, props),
                              "/v1/chat/completions": (200, completion(content="42", tokens=90))},
                             delay_s=0.01)
        self.stub._thread.start()
        self.addCleanup(self.stub.close)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.profile = Path(self.folder.name) / "stub-profile.toml"
        self.profile.write_text('id = "stub-27b"\nweights_gb = 5.2\nthinking = "on"\n'
                                '[model]\nn_layers = 32\nn_kv_heads = 8\nhead_dim = 128\n')

    def call(self, out: str, *extra: str):
        from scripts.preflight import main
        return main(["--profile", str(self.profile), "--endpoint", self.stub.url + "/v1",
                     "--model", "stub-27b", "--legs", "selftest,math500", "--out", out, *extra])

    def test_the_cli_writes_the_plan_and_the_gate(self):
        out = Path(self.folder.name) / "run"
        code = self.call(str(out))
        self.assertIn(code, (0, 1))
        self.assertTrue((out / "run-plan.json").is_file())
        payload = json.loads((out / PREFLIGHT_FILE).read_text())
        self.assertEqual(payload["model"], "stub-27b")
        self.assertEqual(len(payload["checks"]), 9)

    def test_a_second_run_into_the_same_directory_is_refused(self):
        out = Path(self.folder.name) / "run"
        self.call(str(out))
        self.assertEqual(self.call(str(out)), 1)
        self.assertEqual(self.call(str(out), "--resume"), 0)

    def test_without_an_out_dir_nothing_is_written(self):
        self.assertIn(self.call(""), (0, 1))


if __name__ == "__main__":
    unittest.main()
