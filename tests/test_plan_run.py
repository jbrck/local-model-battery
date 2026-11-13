"""The plan: fit, window, caps, decisions, skips and the stamp, on hosts that differ."""
from __future__ import annotations

import copy
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.env_probe import Gpu, HostFacts  # noqa: E402
from scripts.lib.engine_facts import EngineFacts  # noqa: E402
from scripts.lib.model_probe import ModelShape  # noqa: E402
from scripts.lib.pilot import PilotResult  # noqa: E402
from scripts.lib.plan_from_dict import plan_from_dict  # noqa: E402
from scripts.manifest import build_manifest  # noqa: E402
from scripts.lib.plan_run import (  # noqa: E402
    Decision,
    KIND_CAPACITY,
    KIND_DEPENDENCY,
    KIND_MEASUREMENT,
    Option,
    SOURCE_UNKNOWN,
    THINKING_CHOICES,
    _digit,
    _engine_version,
    _number,
    _pinned,
    budget_seconds,
    model_footprint,
    plan_run,
    resolve_thinking,
    unknown_footprint,
)
from scripts.lib.server_probe import ServerFacts  # noqa: E402
from scripts.lib.variables import Variable  # noqa: E402
from tests.fixtures import (  # noqa: E402
    BAREMETAL,
    BY_ID,
    DEAD_PILOT,
    DEAF,
    ENGINE,
    FITS,
    FREE,
    JUDGE,
    LEGS,
    PAIR,
    PILOT,
    RICH,
    SHAPELESS,
    SMALL,
    STANDARD,
    TOOLS,
    VARIABLES,
    VLLM,
    SERVER,
    decision,
    plan,
    runnable,
)


class ShapeSourceTest(unittest.TestCase):
    """The sheet says whether the model's own numbers made the fit or a profile's did.

    The numbers themselves are the model probe's business. What matters here is that the plan
    carries the source, because a fit built on a typed-in shape and one built on the file read
    exactly the same on the sheet otherwise, and on this box one of them was wrong by 4x.
    """

    READ = ModelShape(arch="qwen35", n_layers=17, n_kv_heads=4, head_dim=256,
                      kv_elements_per_token=4352, kv_fixed_elements=8 * 1024 * 1024,
                      weights_bytes=int(4.25 * 1024 ** 3), source="gguf",
                      path="/mnt/models/llamacpp/swift-qwen3.8-27B/Swift-Qwen3.8-27B-Q4_K_M.gguf")

    def test_a_read_shape_is_named_and_priced(self):
        result = plan(FITS, shape=self.READ)
        self.assertEqual(result.footprint.source, "gguf")
        self.assertEqual(result.footprint.kv_bytes_per_token, 4352)
        self.assertEqual(result.footprint.kv_fixed_bytes, 8 * 1024 * 1024)
        self.assertIn("gguf", result.render())
        self.assertIn("8 MB fixed", result.render())

    def test_a_read_shape_beats_the_profile(self):
        """The profile claims 64 layers of 8x128; the file says 17 of 4x256. The file wins."""
        read = plan(FITS, shape=self.READ)
        typed = plan(FITS)
        self.assertEqual(typed.footprint.kv_bytes_per_token, 131072)
        self.assertEqual(read.footprint.kv_bytes_per_token, 4352)

    def test_a_typed_shape_says_the_numbers_were_typed(self):
        result = plan(FITS)
        self.assertEqual(result.footprint.source, "profile")
        self.assertIn("profile", result.render())
        self.assertTrue(result.footprint.notes)

    def test_a_shape_nobody_can_read_claims_nothing(self):
        result = plan(SHAPELESS, shape=ModelShape(problems=("no header",)))
        self.assertFalse(result.footprint.known)
        self.assertIn("shape unknown", result.render())

    def test_the_source_survives_the_round_trip(self):
        original = plan(FITS, shape=self.READ)
        restored = plan_from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.footprint.source, "gguf")
        self.assertEqual(restored.footprint.notes, original.footprint.notes)
        self.assertEqual(restored.render(), original.render())


class VramCreditTest(unittest.TestCase):
    """A model left loaded while the run is planned must not be counted against itself.

    Planning with the server already up is the normal order of operations, so this decides the
    tier on the sheet: 19.0 GB of a 24 GB card belongs to the model under test or it does not.
    """

    HOST = replace(RICH, gpus=(Gpu(0, "RTX 3090", 24576, 19670, 350.0),),
                   compute_apps=({"gpu_uuid": "GPU-1", "pid": "1748",
                                  "process_name": "llama-server", "used_memory_mb": 19414.0},
                                 {"gpu_uuid": "GPU-1", "pid": "8248",
                                  "process_name": "python3", "used_memory_mb": 256.0}))
    READ = replace(ShapeSourceTest.READ, path="/mnt/models/llamacpp/swift/q4.gguf")

    def test_the_vram_the_endpoint_holds_is_credited_to_the_run(self):
        counted = plan(FITS, host=self.HOST, shape=self.READ)
        credited = plan(FITS, host=self.HOST, shape=self.READ, serving_pid=1748)
        self.assertEqual(counted.fit.tier, "T2")
        self.assertEqual(credited.fit.tier, "T1")
        self.assertEqual(credited.fit.max_ctx, credited.fit.preferred_ctx)

    def test_the_credit_is_stated_on_the_sheet(self):
        credited = plan(FITS, host=self.HOST, shape=self.READ, serving_pid=1748)
        self.assertTrue(any("credited to this run" in warning for warning in credited.warnings))

    def test_a_profile_shape_credits_nothing(self):
        """The profile says nothing about what the loaded server is holding."""
        result = plan(FITS, host=self.HOST, serving_pid=1748)
        self.assertEqual(result.fit.tier, "T2")
        self.assertFalse(any("credited" in warning for warning in result.warnings))

    def test_no_pid_means_no_credit(self):
        result = plan(FITS, host=self.HOST, shape=self.READ)
        self.assertEqual(result.fit.tier, "T2")
        self.assertFalse(any("credited" in warning for warning in result.warnings))


class FootprintFallbackTest(unittest.TestCase):
    """What a plan carries when nothing could read the model: no numbers, and it says so."""

    def test_the_unknown_footprint_names_its_ignorance(self):
        blind = unknown_footprint()
        self.assertFalse(blind.known)
        self.assertEqual(blind.missing, ("the shape was never read",))
        self.assertEqual(blind.source, SOURCE_UNKNOWN)

    def test_the_gate_leg_is_marked_and_the_others_are_not(self):
        self.assertTrue(runnable(plan(SMALL, selected=("selftest", "math500")), "selftest")
                        .is_gate)
        self.assertFalse(runnable(plan(SMALL, selected=("math500",)), "math500").is_gate)


class ConfigCoercionTest(unittest.TestCase):
    """Config numbers arrive as whatever TOML and the operator's hands produced."""

    def test_a_number_passes_through_and_junk_falls_back(self):
        self.assertEqual(_number(8, 1.0), 8.0)
        self.assertEqual(_number(2.5, 1.0), 2.5)
        self.assertEqual(_number(True, 1.0), 1.0)
        self.assertEqual(_number("8", 1.0), 1.0)
        self.assertEqual(_number(None, 1.0), 1.0)

    def test_a_digit_is_whole(self):
        self.assertEqual(_digit("9", 3), 3)
        self.assertEqual(_digit(2.9, 3), 2)


class PinnedValueTest(unittest.TestCase):
    """The standard's answer for a variable, read from the table the planner keeps."""

    def test_a_variable_with_no_standard_path_takes_the_fallback(self):
        self.assertEqual(_pinned(STANDARD, "judge_model"), "")
        self.assertEqual(_pinned(STANDARD, "judge_model", "fallback"), "fallback")

    def test_a_missing_standard_value_reads_not_applicable(self):
        self.assertEqual(_pinned({}, "ctx", "131072"), "n/a")

    def test_a_table_value_is_joined_into_one_row(self):
        self.assertIn("math500 500", _pinned(STANDARD, "items"))


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


class MeasuredCostTest(unittest.TestCase):
    """An open question has to say what each answer costs, not just that a question exists."""

    BUSY = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 11264, 350.0),), ram_total_gb=64.0,
                     ram_available_gb=60.0, tools=TOOLS)

    def test_a_question_the_operator_owns_carries_the_measured_cost(self):
        row = decision(plan(FITS, host=self.BUSY), "thinking")
        self.assertTrue(row.needs_operator)
        self.assertIn("Measured:", row.consequence)
        self.assertIn("28.3", row.consequence)

    def test_a_row_the_standard_owns_carries_no_measured_text(self):
        row = decision(plan(FITS), "temperature")
        self.assertFalse(row.needs_operator)
        self.assertNotIn("Measured:", row.consequence)

    def test_thinking_is_asked_only_when_the_weights_miss_the_card(self):
        self.assertFalse(decision(plan(FITS), "thinking").needs_operator)
        options = decision(plan(FITS, host=self.BUSY), "thinking").options
        self.assertEqual([item.label for item in options], ["keep thinking on", "turn it off"])
        self.assertIn("off-standard", options[1].consequence)

    def test_a_footprint_serialises_to_gigabytes_and_a_reason(self):
        known = model_footprint(FITS, STANDARD).to_dict()
        self.assertEqual(known["weights_gb"], 16.4)
        self.assertEqual(known["kv_type"], "q8_0")
        self.assertTrue(known["known"])
        self.assertEqual(known["missing"], [])
        blind = model_footprint(SHAPELESS, STANDARD).to_dict()
        self.assertFalse(blind["known"])
        self.assertTrue(blind["missing"])

    def test_a_model_that_barely_fits_is_asked_about_its_window(self):
        fat = dict(FITS)
        fat["weights_gb"] = 20.0
        result = plan(fat)
        self.assertEqual(result.fit.tier, "T1")
        self.assertLess(result.fit.max_ctx, 131072)
        row = decision(result, "ctx")
        self.assertTrue(row.needs_operator)
        self.assertIn("lcb", row.consequence)
        self.assertIn("context limit", row.consequence)


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

    def test_an_engine_that_always_reasons_stays_on_whatever_was_asked(self):
        """Pinned regression: a deaf endpoint cannot turn reasoning off, in either column."""
        for requested in ("on", "off"):
            mode, note = resolve_thinking(BY_ID["reasoning"], requested, DEAF)
            self.assertEqual(mode, "on")
            self.assertIn("ignores the thinking flag", note)
            self.assertIn("native-forced", note)

    def test_a_leg_pinned_off_stays_off_and_says_why_when_on_was_asked(self):
        pinned_off = replace(BY_ID["math500"], thinking="required_off")
        self.assertEqual(resolve_thinking(pinned_off, "on", SERVER),
                         ("off", "the harness needs the channel empty"))
        self.assertEqual(resolve_thinking(pinned_off, "off", SERVER), ("off", ""))

    def test_a_leg_that_does_not_reason_at_all_reads_na(self):
        flat = replace(BY_ID["math500"], thinking="n/a")
        self.assertEqual(resolve_thinking(flat, "off", DEAF), ("n/a", ""))


class DecisionRowTest(unittest.TestCase):
    """The row type the sheet is built from: when it reads as a deviation, and how it serialises."""

    def test_only_a_measurement_variable_can_be_off_standard(self):
        moved = Decision("ctx", KIND_MEASUREMENT, "65536", "131072", "the window shrank")
        held = replace(moved, chosen="131072")
        capacity = Decision("host_vram", KIND_CAPACITY, "24 GB", "131072",
                            "capacity is whatever the host is")
        self.assertTrue(moved.is_deviation)
        self.assertFalse(held.is_deviation)
        self.assertFalse(capacity.is_deviation)

    def test_a_standard_value_that_is_not_stated_is_never_a_deviation(self):
        row = Decision("judge_model", KIND_MEASUREMENT, "deckard-40b", "n/a", "scored by")
        self.assertFalse(row.is_deviation)
        self.assertFalse(replace(row, standard_value="").is_deviation)

    def test_to_dict_carries_the_deviation_flag_and_the_option_list(self):
        row = Decision("thinking", KIND_MEASUREMENT, "off", "on", "cheaper and worse",
                       options=(Option("keep thinking on", "stays on-standard", recommended=True),
                                Option("turn it off", "off-standard")),
                       needs_operator=True)
        payload = row.to_dict()
        self.assertEqual(payload["variable"], "thinking")
        self.assertEqual(payload["kind"], KIND_MEASUREMENT)
        self.assertEqual(payload["standard"], "on")
        self.assertTrue(payload["deviation"])
        self.assertTrue(payload["needs_operator"])
        self.assertEqual([item["label"] for item in payload["options"]],
                         ["keep thinking on", "turn it off"])
        self.assertTrue(payload["options"][0]["recommended"])
        self.assertFalse(payload["options"][1]["recommended"])

    def test_the_render_marks_the_recommended_option_and_the_measured_cost(self):
        busy = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 11264, 350.0),), ram_total_gb=64.0,
                         ram_available_gb=60.0, tools=TOOLS)
        sheet = plan(FITS, host=busy).render()
        self.assertIn("* keep thinking on", sheet)
        self.assertIn("Measured:", sheet)


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
        self.assertEqual(decision(plan(SMALL), "engine_version").chosen, "llama.cpp b4567-abc")

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


class EngineStampTest(unittest.TestCase):
    """Every run names its engine, whatever engine it is, and records its window apart."""

    def test_the_props_fallback_names_the_engine_with_the_props_build(self):
        self.assertEqual(_engine_version(ENGINE, SERVER), "llama.cpp b4567-abc")

    def test_an_engine_that_reports_its_own_build_uses_that_build(self):
        self.assertEqual(_engine_version(VLLM, SERVER), "vllm 0.6.6")

    def test_nothing_reporting_build_is_said_plainly(self):
        gone = ServerFacts(reachable=False)
        self.assertEqual(_engine_version(VLLM, gone), "vllm 0.6.6")
        bare = EngineFacts(engine="llama.cpp", supports_kv_flags=True)
        self.assertEqual(_engine_version(bare, gone), "llama.cpp (build not reported)")

    def test_the_plan_stamps_the_engine_it_probed(self):
        result = plan(SMALL)
        self.assertEqual(result.engine, "llama.cpp")
        self.assertEqual(result.engine_version, "llama.cpp b4567-abc")
        self.assertEqual(result.engine_window, 131072)
        self.assertTrue(result.supports_kv_flags)
        self.assertIsNone(result.drafter)

    def test_a_vllm_plan_stamps_vllm_and_no_kv_flags(self):
        result = plan(SMALL, engine=VLLM)
        self.assertEqual(result.engine, "vllm")
        self.assertEqual(result.engine_version, "vllm 0.6.6")
        self.assertFalse(result.supports_kv_flags)

    def test_the_engine_window_is_recorded_apart_from_the_planned_ctx(self):
        """A 16 GB model caps the window below what the engine actually serves."""
        result = plan(FITS)
        self.assertEqual(result.engine_window, 131072)
        self.assertIsNotNone(result.engine_window)
        self.assertLess(runnable(result, "math500").ctx, result.engine_window)

    def test_the_sheet_names_the_engine(self):
        self.assertIn("engine   llama.cpp b4567-abc", plan(SMALL).render())

    def test_the_engine_fields_survive_the_run_plan_round_trip(self):
        result = plan(SMALL)
        restored = plan_from_dict(json.loads(json.dumps(result.to_dict())))
        self.assertEqual(restored.engine, "llama.cpp")
        self.assertEqual(restored.engine_version, "llama.cpp b4567-abc")
        self.assertEqual(restored.engine_window, 131072)
        self.assertTrue(restored.supports_kv_flags)
        self.assertEqual(restored.render(), result.render())

    def test_a_legacy_plan_without_engine_fields_reads_back_as_unknown(self):
        payload = plan(SMALL).to_dict()
        for key in ("engine", "engine_version", "engine_window", "supports_kv_flags",
                    "drafter"):
            payload.pop(key)
        restored = plan_from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored.engine, "unknown")
        self.assertFalse(restored.supports_kv_flags)
        self.assertIsNone(restored.engine_window)

    def test_a_manifest_lists_what_the_engine_said(self):
        manifest = build_manifest(plan(SMALL), RICH, SERVER)
        engine = manifest["engine"]
        self.assertEqual(engine["engine"], "llama.cpp")
        self.assertEqual(engine["engine_version"], "llama.cpp b4567-abc")
        self.assertEqual(engine["engine_window"], 131072)
        self.assertTrue(engine["supports_kv_flags"])
        self.assertIsNone(engine["drafter"])


class EngineKnobTest(unittest.TestCase):
    """A knob the engine does not have is emitted with n/a and a reason, never dropped."""

    def test_the_kv_row_reads_not_applicable_on_vllm(self):
        row = decision(plan(SMALL, engine=VLLM), "kv_type")
        self.assertEqual(row.chosen, "n/a")
        self.assertEqual(row.consequence,
                         "vLLM manages the KV cache itself (paged attention), so this knob "
                         "does not exist here")
        self.assertTrue(row.is_deviation)

    def test_the_kv_row_keeps_the_standard_on_llama_cpp(self):
        self.assertEqual(decision(plan(SMALL), "kv_type").chosen, "q8_0")

    def test_the_flash_attn_row_reads_not_applicable_on_vllm(self):
        row = decision(plan(SMALL, engine=VLLM), "flash_attn")
        self.assertEqual(row.chosen, "n/a")
        self.assertIn("vLLM", row.consequence)
        self.assertIn("flash-attention", row.consequence)
        self.assertTrue(row.is_deviation)

    def test_the_flash_attn_row_keeps_the_standard_on_llama_cpp(self):
        self.assertEqual(decision(plan(SMALL), "flash_attn").chosen, "True")

    def test_a_slots_row_above_one_on_a_batching_engine_names_the_difference(self):
        row = decision(plan(SMALL, engine=VLLM), "slots")
        self.assertEqual(row.chosen, "4")
        self.assertIn("vLLM also batches concurrent requests", row.consequence)
        self.assertIn("slots and batching are different things that both split one KV budget",
                      row.consequence)

    def test_the_slots_row_stays_plain_on_llama_cpp(self):
        row = decision(plan(SMALL), "slots")
        self.assertEqual(row.chosen, "4")
        self.assertIn("more slots means a smaller window", row.consequence)

    def test_an_unknown_engine_reports_na_instead_of_assuming(self):
        row = decision(plan(SMALL, engine=None), "kv_type")
        self.assertEqual(row.chosen, "n/a")
        self.assertIn("no engine answered the probes", row.consequence)

    def test_the_stamp_says_na_where_the_knob_does_not_exist(self):
        result = plan(SMALL, engine=VLLM)
        self.assertEqual(result.stamp["kv_type"], "n/a")
        self.assertEqual(result.stamp["flash_attn"], "n/a")


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


class PlanWarningTest(unittest.TestCase):
    """The warnings block: every reason to stop before spending hours on this run."""

    HOSTLESS = HostFacts(gpus=(Gpu(0, "RTX 3050", 8192, 0, 170.0),), ram_total_gb=8.0,
                         ram_available_gb=6.0)

    def test_a_model_that_cannot_run_here_is_warned_not_planned(self):
        result = plan(FITS, host=self.HOSTLESS)
        self.assertTrue(any("will not run here" in warning for warning in result.warnings))

    def test_a_measurement_variable_no_row_answers_is_warned(self):
        """A planned leg uses the knob, so scoring without a value would not be comparable."""
        extra = Variable(id="extra_knob", variable_class="M", applies_to=("math500",),
                         stamp_key="extra_knob", what_it_changes="how many items are sampled",
                         effect="changes the score's noise", source="standard",
                         decision_rule="keep the standard's value")
        result = plan_run(STANDARD, LEGS, list(VARIABLES) + [extra], FITS, RICH, SERVER,
                          PILOT, JUDGE, engine=ENGINE)
        self.assertIn("measurement variable extra_knob has no value on this plan",
                      result.warnings)

    def test_an_invalid_thinking_request_falls_back_to_the_standard(self):
        """A typo in the profile's thinking switch must not read as an off-standard request."""
        result = plan({**FITS, "thinking": "banana"})
        self.assertEqual(runnable(result, "math500").thinking, "on")


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


class ScriptEntryTest(unittest.TestCase):
    """The file the setup skill actually runs: python3 scripts/plan_run.py must print the sheet."""

    PROFILE = ('id = "sheet-test"\nweights_gb = 5.2\n\n[model]\n'
               'n_layers = 32\nn_kv_heads = 8\nhead_dim = 128\n')

    QUIET_HOST = HostFacts(gpus=(Gpu(0, "RTX 3090", 24576, 0, 350.0),),
                           ram_total_gb=64.0, ram_available_gb=60.0)

    def test_the_script_form_prints_the_decision_sheet_and_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sheet-test.toml")
            path.write_text(self.PROFILE)
            argv = ["scripts/plan_run.py", "--profile", str(path), "--no-probe",
                    "--legs", "math500,mmlu"]
            buffer = io.StringIO()
            with mock.patch("sys.argv", argv), \
                    mock.patch("scripts.lib.env_probe.probe_host",
                               return_value=self.QUIET_HOST):
                with redirect_stdout(buffer), self.assertRaises(SystemExit) as caught:
                    runpy.run_path("scripts/plan_run.py", run_name="__main__")
        self.assertEqual(caught.exception.code, 0)
        sheet = buffer.getvalue()
        self.assertIn("Battery v2 plan -- sheet-test", sheet)
        self.assertIn("math500", sheet)
        self.assertIn("cap 8192", sheet)


class InterpreterSheetTest(unittest.TestCase):
    """The sheet has to say when a leg runs under a different interpreter, and why."""

    def lcb_plan(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        Path(folder.name, "test6.jsonl").write_text('{"question_id": 1}\n')
        present = copy.deepcopy(STANDARD)
        for entry in present["data"]["external"]:
            if entry["id"] == "lcb_data":
                entry["path"] = str(Path(folder.name, "test6.jsonl"))
                entry["sha256_16"] = ""
            if entry["id"] == "lcb_source":
                entry["path"] = folder.name
        return plan(FITS, standard=present, selected=("lcb", "math500"))

    def test_a_leg_that_borrows_an_interpreter_says_so(self):
        row = decision(self.lcb_plan(), "interpreter:lcb")
        self.assertEqual(row.kind, KIND_DEPENDENCY)
        self.assertEqual(row.chosen, row.standard_value)
        self.assertNotEqual(row.chosen, sys.executable)
        self.assertIn("LCB_PYTHON", row.consequence)

    def test_a_leg_that_borrows_nothing_has_no_row(self):
        rows = [row.variable for row in plan(SMALL, selected=FREE).decisions]
        self.assertNotIn("interpreter:math500", rows)
        self.assertNotIn("interpreter:mmlu", rows)


if __name__ == "__main__":
    unittest.main()
