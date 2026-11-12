"""Pilot speed: the number the cap, the case timeout and the session budget come from."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.model import ModelConfig  # noqa: E402
from scripts.lib.pilot import (  # noqa: E402
    CASE_TIMEOUT_FLOOR_S,
    PILOT_ITEMS,
    PilotResult,
    run_pilot,
)
from tests.stub_json import JsonStub, completion  # noqa: E402


def result(tps: float, stalled: bool = False) -> PilotResult:
    return PilotResult(tps=tps, latency_s=1.0, generated_tokens=int(tps * 10), requests=3,
                       failed=0, stalled=stalled)


class BudgetMathTest(unittest.TestCase):
    def test_case_timeout_scales_with_speed_above_the_floor(self):
        # 12 tokens/s at the LCB budget: the 600s floor would clip a slow model, so the
        # timeout has to move with the cap.
        slow = result(tps=12.0)
        self.assertEqual(slow.case_timeout_s(114688), int(1.25 * 114688 / 12))

    def test_fast_host_keeps_the_floor(self):
        self.assertEqual(result(tps=80.0).case_timeout_s(8192), CASE_TIMEOUT_FLOOR_S)

    def test_unusable_speed_still_yields_a_usable_timeout(self):
        self.assertEqual(result(tps=0.0, stalled=True).case_timeout_s(8192),
                         CASE_TIMEOUT_FLOOR_S)

    def test_hours_for_items_scales_with_the_cap(self):
        pilot = result(tps=30.0)
        self.assertGreater(pilot.hours_for(500, 16384), pilot.hours_for(500, 8192))

    def test_unusable_speed_reports_infinite_time_instead_of_a_small_number(self):
        self.assertEqual(result(tps=0.0, stalled=True).hours_for(500, 8192), float("inf"))

    def test_usability_requires_speed_and_a_live_model(self):
        self.assertTrue(result(tps=25.0).usable)
        self.assertFalse(result(tps=25.0, stalled=True).usable)
        self.assertFalse(result(tps=0.0).usable)


class LivePilotTest(unittest.TestCase):
    def test_pilot_measures_tokens_per_second(self):
        routes = {"/v1/chat/completions": (200, completion("ok", tokens=200))}
        with JsonStub(routes, delay_s=0.05) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=10)
            pilot = run_pilot(cfg, items=PILOT_ITEMS, cap=256)
        self.assertEqual(pilot.requests, PILOT_ITEMS)
        self.assertEqual(pilot.failed, 0)
        self.assertGreater(pilot.tps, 0)
        self.assertTrue(pilot.usable)
        self.assertEqual(pilot.generated_tokens, 200 * PILOT_ITEMS)

    def test_every_request_asks_for_the_capshaped_max_tokens(self):
        routes = {"/v1/chat/completions": (200, completion("ok", tokens=10))}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=10)
            run_pilot(cfg, items=2, cap=128)
        self.assertTrue(all(item["body"]["max_tokens"] == 128 for item in stub.seen))

    def test_a_server_that_errors_leaves_the_pilot_unusable(self):
        routes = {"/v1/chat/completions": (500, {"error": "model crashed"})}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=10)
            pilot = run_pilot(cfg, items=PILOT_ITEMS)
        self.assertTrue(pilot.stalled)
        self.assertFalse(pilot.usable)
        self.assertEqual(pilot.failed, PILOT_ITEMS)
        self.assertTrue(pilot.errors)

    def test_model_that_emits_nothing_is_stalled_not_fast(self):
        payload = completion("", tokens=0)
        routes = {"/v1/chat/completions": (200, payload)}
        with JsonStub(routes) as stub:
            cfg = ModelConfig(name="stub", endpoint=stub.endpoint, timeout_s=10)
            pilot = run_pilot(cfg, items=PILOT_ITEMS)
        self.assertTrue(pilot.stalled)
        self.assertEqual(pilot.tps, 0)

    def test_pilot_never_raises_against_a_dead_endpoint(self):
        cfg = ModelConfig(name="stub", endpoint="http://127.0.0.1:9/v1", timeout_s=2)
        pilot = run_pilot(cfg, items=1, cap=32)
        self.assertTrue(pilot.stalled)
        self.assertEqual(pilot.failed, 1)


if __name__ == "__main__":
    unittest.main()
