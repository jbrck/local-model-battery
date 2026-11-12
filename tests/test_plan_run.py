"""The plan: fit, window, caps, decisions, skips and the stamp, on hosts that differ."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.env_probe import Gpu, HostFacts, ToolFact  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.pilot import PilotResult  # noqa: E402
from scripts.lib.plan_run import (  # noqa: E402
    KIND_CAPACITY,
    KIND_DEPENDENCY,
    KIND_MEASUREMENT,
    THINKING_CHOICES,
    budget_seconds,
    model_footprint,
    plan_run,
)
from scripts.lib.prereqs import Judge  # noqa: E402
from scripts.lib.server_probe import ServerFacts  # noqa: E402
from scripts.lib.variables import load_variables  # noqa: E402

STANDARD = load_toml("standard.toml")
LEGS = load_legs(STANDARD)
BY_ID = {leg.id: leg for leg in LEGS}
VARIABLES = load_variables()
TOOLS = (ToolFact("docker", "/usr/bin/docker", "27.0"),
         ToolFact("benchlocal-cli", "/usr/local/bin/benchlocal-cli", "0.4"),
         ToolFact("tool-eval-bench", "/usr/local/bin/tool-eval-bench", "1.2"))

# 3090: one 24 GB card, 64 GB of RAM, every external tool present.
RICH = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 0, 350.0),), ram_total_gb=64.0,
                 ram_available_gb=60.0, cpu_model="i7-8700K", cpu_threads=12, tools=TOOLS)
# 3080 x2: 40 GB of VRAM in total but 20 GB per card, and nothing else installed.
PAIR = HostFacts(gpus=(Gpu(0, "RTX 3080", 20480, 0, 320.0), Gpu(1, "RTX 3080", 20480, 0, 320.0)),
                 ram_total_gb=96.0, ram_available_gb=88.0, tools=())
BAREMETAL = HostFacts(ram_total_gb=32.0, ram_available_gb=28.0, gpu_error="nvidia-smi not found")

SERVER = ServerFacts(reachable=True, served_model="arete-27b", ctx=131072, slots=4,
                     chat_template_mentions_thinking=True, thinking_switch=True,
                     reasoning_channel=True, thinking_on_tokens=1400, thinking_off_tokens=90,
                     tools_with_json_schema_ok=True, props={"build_info": "b4567-abc"})
DEAF = ServerFacts(reachable=True, served_model="arete-27b", ctx=131072, slots=1,
                   chat_template_mentions_thinking=False, thinking_switch=False,
                   reasoning_channel=False, thinking_on_tokens=300, thinking_off_tokens=300,
                   tools_with_json_schema_ok=False, tools_error="grammar and json_schema: 400",
                   props={"build_info": "b11670"})
PILOT = PilotResult(tps=27.0, latency_s=1.2, generated_tokens=810, requests=3, failed=0,
                    stalled=False)
DEAD_PILOT = PilotResult(tps=0.0, latency_s=0.0, generated_tokens=0, requests=3, failed=3,
                         stalled=True, errors=("HTTP 500",))

# 27B dense Q4_K_M: 64 layers, 8 KV heads of 128, q8_0 KV.
FITS = {"id": "arete-27b", "weights_gb": 16.4, "thinking": "on", "weight_quant": "Q4_K_M",
        "model": {"n_layers": 64, "n_kv_heads": 8, "head_dim": 128}}
SMALL = {"id": "tiny-8b", "weights_gb": 5.2, "thinking": "on",
         "model": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128}}
SHAPELESS = {"id": "mystery-30b", "weights_gb": 18.0}
JUDGE = Judge(model="deckard-40b", endpoint="http://127.0.0.1:11435/v1", under_test="arete-27b")
# Two legs with no external prerequisites, for the tests that need a quiet host.
FREE = ("math500", "mmlu")


def plan(profile: dict = FITS, standard: dict | None = None, host: HostFacts = RICH,
         server: ServerFacts = SERVER, pilot: PilotResult = PILOT,
         judge: Judge | None = JUDGE, selected: tuple[str, ...] = ()):
    return plan_run(standard or STANDARD, LEGS, VARIABLES, profile, host, server, pilot,
                    judge, selected)


def runnable(result, leg_id: str):
    """The planned leg, or a failed test when the planner dropped it."""
    leg = result.leg(leg_id)
    assert leg is not None, f"{leg_id} is not in this plan"
    return leg


def decision(result, variable: str):
    """The one decision row for a variable."""
    rows = [item for item in result.decisions if item.variable == variable]
    assert len(rows) == 1, f"expected one row for {variable}, found {len(rows)}"
    return rows[0]


class FitAndWindowTest(unittest.TestCase):
    def test_a_model_that_fits_the_card_keeps_the_standard_window(self):
        result = plan(SMALL)
        self.assertEqual(result.fit.tier, "T1")
        self.assertEqual(runnable(result, "math500").ctx, 131072)
        self.assertEqual(result.deviations, ())
        self.assertEqual(result.stamp["ctx"], "131072")

    def test_a_16gb_model_on_one_24gb_card_cannot_serve_the_standard_window(self):
        result = plan(FITS)
        self.assertEqual(result.fit.tier, "T1")
        self.assertLess(runnable(result, "math500").ctx, 131072)
        self.assertGreater(runnable(result, "math500").ctx, 32768)
        row = decision(result, "ctx")
        self.assertTrue(row.needs_operator)
        self.assertEqual(row.kind, KIND_MEASUREMENT)
        self.assertTrue(row.is_deviation)
        self.assertGreaterEqual(len(row.options), 2)
        self.assertIn("KV", row.options[0].consequence + row.consequence)

    def test_the_window_is_not_offered_below_the_longest_leg(self):
        ctx = runnable(plan(FITS), "math500").ctx
        self.assertGreater(ctx - 16384, 8192)

    def test_a_11gb_occupant_pushes_the_same_model_off_the_card(self):
        busy = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 11264, 350.0),), ram_total_gb=64.0,
                         ram_available_gb=60.0, tools=TOOLS)
        result = plan(FITS, host=busy)
        self.assertEqual(result.fit.tier, "T2")
        self.assertTrue(result.fit.estimate)

    def test_a_big_moe_on_two_20gb_cards_is_offload_not_failure(self):
        moe = {"id": "moe-80b", "weights_gb": 43.0,
               "model": {"n_layers": 48, "n_kv_heads": 8, "head_dim": 128}}
        result = plan(moe, host=PAIR)
        self.assertEqual(result.fit.tier, "T2")
        self.assertEqual(result.fit.verdict, "fits_with_offload")

    def test_no_gpu_facts_leave_placement_to_the_engine(self):
        result = plan(FITS, host=BAREMETAL)
        self.assertEqual(result.fit.tier, "T3")
        self.assertEqual(runnable(result, "math500").ctx, 131072)
        self.assertIn("engine", " ".join(result.warnings).lower())

    def test_a_model_without_a_shape_is_probed_not_guessed(self):
        result = plan(SHAPELESS)
        self.assertTrue(result.fit.estimate)
        self.assertEqual(runnable(result, "math500").ctx, 131072)
        self.assertTrue(any("shape is incomplete" in warning for warning in result.warnings))
        self.assertFalse([item for item in result.decisions
                          if item.variable == "ctx" and item.kind == KIND_MEASUREMENT])
        guess = [item for item in result.decisions
                 if item.variable == "ctx" and item.kind == KIND_CAPACITY][0]
        self.assertTrue(guess.needs_operator)
        self.assertIn("/props", guess.consequence)

    def test_footprint_is_unknown_without_a_shape(self):
        footprint = model_footprint(SHAPELESS, STANDARD)
        self.assertFalse(footprint.known)
        self.assertIn("n_layers", footprint.missing)

    def test_footprint_counts_kv_from_the_shape(self):
        footprint = model_footprint(FITS, STANDARD)
        self.assertTrue(footprint.known)
        self.assertEqual(footprint.kv_bytes_per_token, 131072)


class CapTest(unittest.TestCase):
    def test_a_standard_cap_wider_than_the_window_is_clamped_and_reported(self):
        tight = copy.deepcopy(STANDARD)
        tight["caps"]["default"] = 60000
        result = plan(FITS, standard=tight)
        ctx = runnable(result, "math500").ctx
        self.assertEqual(runnable(result, "math500").cap, ctx - 16384)
        row = decision(result, "cap:math500")
        self.assertTrue(row.needs_operator)
        self.assertEqual(row.standard_value, "60000")

    def test_lcb_keeps_the_widest_cap_the_window_allows(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "test6.jsonl").write_text('{"question_id": 1}\n')
            present = copy.deepcopy(STANDARD)
            for entry in present["data"]["external"]:
                if entry["id"] == "lcb_data":
                    entry["path"] = str(Path(folder, "test6.jsonl"))
                    entry["sha256_16"] = ""
                if entry["id"] == "lcb_source":
                    entry["path"] = folder
            result = plan(FITS, standard=present, selected=("lcb", "math500"))
            lcb = runnable(result, "lcb")
            self.assertEqual(lcb.cap, min(114688, lcb.ctx - 16384))
            self.assertLess(lcb.cap, 114688)

    def test_every_planned_leg_records_its_cap(self):
        result = plan(SMALL, selected=FREE)
        self.assertEqual(result.stamp["cap"], {"math500": "8192", "mmlu": "8192"})
        self.assertEqual(decision(result, "cap:mmlu").standard_value, "8192")

    def test_an_unclamped_cap_is_not_an_operator_question(self):
        result = plan(SMALL, selected=FREE)
        self.assertFalse(decision(result, "cap:math500").needs_operator)

    def test_every_planned_leg_records_its_harness(self):
        result = plan(SMALL, selected=FREE)
        self.assertEqual(result.stamp["harness_version"]["math500"], BY_ID["math500"].harness)


class SkipTest(unittest.TestCase):
    def test_missing_tooling_skips_the_legs_that_need_it(self):
        result = plan(FITS, host=PAIR)
        skipped = {skip.leg for skip in result.skips}
        self.assertLessEqual({"lcb", "hermesagent20", "toolbench", "ifbench"}, skipped)
        self.assertNotIn("math500", skipped)

    def test_the_gate_leg_survives_a_host_with_no_external_data(self):
        self.assertIsNotNone(plan(FITS, host=PAIR).leg("selftest"))

    def test_every_skip_carries_a_reason(self):
        result = plan(FITS, host=PAIR)
        self.assertTrue(all(skip.reason for skip in result.skips))

    def test_a_skip_comes_with_a_dependency_decision(self):
        result = plan(FITS, host=PAIR)
        for skip in result.skips:
            row = decision(result, f"prereq:{skip.leg}")
            self.assertEqual(row.kind, KIND_DEPENDENCY)
            self.assertTrue(row.needs_operator)

    def test_lcb_data_is_skipped_even_when_the_tools_are_there(self):
        self.assertIn("lcb", {skip.leg for skip in plan(FITS).skips})

    def test_a_configured_judge_keeps_the_writing_leg(self):
        result = plan(FITS)
        self.assertEqual(runnable(result, "writing").thinking, "n/a")


class ThinkingTest(unittest.TestCase):
    def test_thinking_off_puts_primary_on_legs_in_the_off_column(self):
        result = plan({**FITS, "thinking": "off"})
        self.assertEqual(runnable(result, "math500").thinking, "off")
        self.assertEqual(runnable(result, "math500").note, "")
        row = decision(result, "thinking")
        self.assertEqual(row.chosen, "off")
        self.assertTrue(row.is_deviation)

    def test_a_deaf_endpoint_forces_the_on_column_and_says_why(self):
        result = plan({**FITS, "thinking": "off"}, server=DEAF)
        self.assertEqual(runnable(result, "math500").thinking, "on")
        self.assertIn("ignores the thinking flag", runnable(result, "math500").note)
        row = decision(result, "thinking")
        self.assertEqual(row.chosen, "native")
        self.assertTrue(row.needs_operator)

    def test_a_required_on_leg_notes_a_deaf_endpoint(self):
        result = plan(FITS, server=DEAF, selected=("reasoning",))
        self.assertEqual(runnable(result, "reasoning").thinking, "on")
        self.assertIn("native-forced", runnable(result, "reasoning").note)

    def test_an_unmeasured_switch_is_not_treated_as_off(self):
        unknown = ServerFacts(reachable=True, served_model="m", ctx=131072)
        result = plan({**FITS, "thinking": "off"}, server=unknown)
        self.assertEqual(runnable(result, "math500").thinking, "off")


class DecisionTest(unittest.TestCase):
    def test_the_standard_knobs_are_stamped_with_their_standard_values(self):
        result = plan(SMALL)
        self.assertEqual(result.stamp["seed"], "42")
        self.assertEqual(result.stamp["temperature"], "0.0")
        self.assertEqual(result.stamp["kv_type"], "q8_0")
        self.assertIn(result.stamp["thinking"], THINKING_CHOICES)

    def test_capacity_rows_are_recorded_but_never_off_standard(self):
        result = plan(SMALL)
        capacity = [item for item in result.decisions if item.kind == KIND_CAPACITY]
        self.assertTrue(any(item.variable == "host_vram" for item in capacity))
        self.assertTrue(any(item.variable == "engine_version" for item in capacity))
        self.assertEqual([item for item in capacity if item.is_deviation], [])
        self.assertIn("24 GB", result.stamp["host_vram"])

    def test_the_engine_version_comes_from_the_endpoint(self):
        self.assertEqual(decision(plan(SMALL), "engine_version").chosen, "b4567-abc")

    def test_a_case_timeout_is_derived_from_the_measured_speed(self):
        slow = PilotResult(tps=12.0, latency_s=4.0, generated_tokens=400, requests=3, failed=0,
                           stalled=False)
        result = plan(FITS, pilot=slow, selected=("hermesagent20",))
        row = decision(result, "case_timeout")
        self.assertEqual(row.chosen, str(int(1.25 * 8192 / 12)))
        self.assertGreater(int(row.chosen), 600)
        self.assertTrue(row.needs_operator)

    def test_a_fast_host_keeps_the_case_timeout_floor(self):
        row = decision(plan(FITS, selected=("hermesagent20",)), "case_timeout")
        self.assertEqual(row.chosen, "600")
        self.assertFalse(row.needs_operator)

    def test_every_measurement_variable_gets_a_value_or_a_warning(self):
        result = plan(FITS, selected=("hermesagent20", "math500"))
        named = {item.variable.partition(":")[0] for item in result.decisions}
        missing = [variable.id for variable in VARIABLES
                   if variable.is_measurement and variable.id not in named
                   and variable.applies_to_leg("math500")]
        self.assertEqual(missing, [])
        self.assertEqual([warning for warning in result.warnings
                          if "has no value" in warning], [])

    def test_the_judge_row_names_the_scoring_model(self):
        row = decision(plan(FITS), "judge_model")
        self.assertEqual(row.chosen, "deckard-40b")
        self.assertFalse(row.needs_operator)

    def test_an_unconfigured_judge_is_reported_and_the_leg_is_asked_about(self):
        result = plan(FITS, judge=None)
        self.assertEqual(decision(result, "judge_model").chosen, "none configured")
        self.assertTrue(decision(result, "prereq:writing").needs_operator)


class BudgetTest(unittest.TestCase):
    def test_budget_follows_the_pilot_and_never_undercuts_the_floor(self):
        leg = BY_ID["math500"]
        self.assertEqual(budget_seconds(leg, 8192, DEAD_PILOT, STANDARD), leg.budget_floor)
        measured = budget_seconds(leg, 8192, PilotResult(1.0, 1.0, 10, 3, 0, False), STANDARD)
        self.assertEqual(measured, int(1.25 * 500 * 8192 / 1.0))

    def test_an_unusable_pilot_produces_a_warning_and_floor_budgets(self):
        result = plan(SMALL, pilot=DEAD_PILOT)
        self.assertEqual(runnable(result, "math500").budget_s, BY_ID["math500"].budget_floor)
        self.assertTrue(any("pilot never measured" in warning for warning in result.warnings))

    def test_measured_speed_stretches_the_budget_past_the_floor(self):
        result = plan(SMALL, pilot=PilotResult(1.0, 1.0, 10, 3, 0, False))
        self.assertGreater(runnable(result, "math500").budget_s, BY_ID["math500"].budget_floor)


class SelectionTest(unittest.TestCase):
    def test_only_the_selected_legs_are_planned(self):
        result = plan(SMALL, selected=("math500", "gpqa_diamond"))
        self.assertEqual([item.id for item in result.legs], ["math500", "gpqa_diamond"])

    def test_no_selection_means_every_leg_the_host_supports(self):
        result = plan(SMALL)
        ids = [item.id for item in result.legs]
        self.assertGreater(len(ids), 8)
        self.assertEqual(ids[0], "selftest")


class OutputTest(unittest.TestCase):
    def test_to_dict_survives_json(self):
        payload = json.loads(json.dumps(plan(FITS).to_dict()))
        self.assertEqual(payload["standard_version"], "v2")
        self.assertEqual([item["id"] for item in payload["legs"]][0], "selftest")
        self.assertIn("needs_operator", payload)

    def test_the_render_names_the_model_the_fit_and_the_pending_decisions(self):
        text = plan(FITS).render()
        self.assertIn("arete-27b", text)
        self.assertIn("fit      T1", text)
        self.assertIn("Decisions that need you", text)
        self.assertIn("Skipped", text)

    def test_the_render_is_quiet_when_nothing_needs_the_operator(self):
        text = plan(SMALL, selected=FREE).render()
        self.assertNotIn("Decisions that need you", text)
        self.assertNotIn("Skipped", text)
        self.assertIn("0 skipped", text)

    def test_the_render_lists_legs_with_caps_and_budgets(self):
        text = plan(SMALL, selected=("math500",)).render()
        self.assertIn("math500", text)
        self.assertIn("cap 8192", text)


if __name__ == "__main__":
    unittest.main()
