"""Leg catalog tests."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.legs import (  # noqa: E402
    Leg,
    legs_by_id,
    load_legs,
    load_raw_legs,
    resolve_cap,
    validate_legs,
)

EXPECTED_IDS = (
    "selftest", "math500", "humaneval_plus", "gpqa_diamond", "instr_v2", "refusal",
    "reasoning", "ifeval", "ifbench", "toolbench", "mmlu", "hermesagent20", "lcb",
    "writing",
)


def make_leg(**overrides) -> Leg:
    fields = dict(
        id="sample", name="Sample", measures="nothing", scoring="exit code",
        harness="native", implementation="in_repo", entry="scripts/bench_sample.py",
        cap_var="caps.default", ctx_need=8192, thinking="primary_on",
        thinking_effect="none measured", prereqs=(), resume="done_marker",
        comparability_keys=("cap",), known_failures=("it can go wrong",),
        gate=False, items=10, budget_floor=60,
    )
    fields.update(overrides)
    # `**fields` erases the per-parameter types, which the checker cannot follow.
    return Leg(**fields)  # type: ignore[arg-type]


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.standard = load_toml("standard.toml")
        self.legs = load_legs(self.standard)
        self.raw = load_raw_legs()

    def test_catalog_holds_the_fourteen_legs_the_reference_box_runs(self):
        self.assertEqual(tuple(leg.id for leg in self.legs), EXPECTED_IDS)

    def test_item_counts_are_joined_from_the_standard(self):
        by_id = {leg.id: leg for leg in self.legs}
        self.assertEqual(by_id["mmlu"].items, 500)
        self.assertEqual(by_id["lcb"].items, 100)
        self.assertEqual(by_id["writing"].items, 138)

    def test_cap_var_resolves_per_leg(self):
        by_id = {leg.id: leg for leg in self.legs}
        self.assertEqual(resolve_cap(by_id["lcb"], self.standard), 114688)
        self.assertEqual(resolve_cap(by_id["writing"], self.standard), 16384)
        self.assertEqual(resolve_cap(by_id["mmlu"], self.standard), 8192)

    def test_leg_override_beats_the_cap_var(self):
        standard = load_toml("standard.toml")
        standard["caps"]["by_leg"] = {"mmlu": 4096}
        by_id = {leg.id: leg for leg in self.legs}
        self.assertEqual(resolve_cap(by_id["mmlu"], standard), 4096)

    def test_missing_cap_var_raises_with_the_leg_named(self):
        leg = make_leg(cap_var="caps.nonexistent")
        with self.assertRaises(KeyError) as caught:
            resolve_cap(leg, self.standard)
        self.assertIn("caps.nonexistent", str(caught.exception))

    def test_lcb_needs_a_window_larger_than_its_cap(self):
        by_id = {leg.id: leg for leg in self.legs}
        self.assertGreater(by_id["lcb"].ctx_need, 8192)

    def test_legs_that_need_external_harnesses_declare_prerequisites(self):
        by_id = {leg.id: leg for leg in self.legs}
        self.assertEqual(by_id["hermesagent20"].harness, "adapter")
        self.assertIn("docker_sandbox", by_id["hermesagent20"].prereqs)
        self.assertIn("lcb_source", by_id["lcb"].prereqs)
        self.assertIn("tool_eval_bench", by_id["ifeval"].prereqs)

    def test_in_repo_legs_point_at_scripts_that_exist(self):
        for leg in self.legs:
            if leg.implemented:
                self.assertTrue(os.path.isfile(leg.entry), leg.entry)

    def test_shipped_catalog_validates_clean(self):
        self.assertEqual(validate_legs(self.legs, self.raw), [])

    def test_to_dict_lists_every_field_the_manifest_records(self):
        leg = make_leg(prereqs=("lcb_data",), comparability_keys=("cap", "ctx"),
                       known_failures=("flaky exit",), data="data/lcb.jsonl")
        payload = leg.to_dict()
        self.assertEqual(payload["id"], "sample")
        self.assertEqual(payload["measures"], "nothing")
        self.assertEqual(payload["scoring"], "exit code")
        self.assertEqual(payload["implementation"], "in_repo")
        self.assertEqual(payload["prereqs"], ["lcb_data"])
        self.assertEqual(payload["comparability_keys"], ["cap", "ctx"])
        self.assertEqual(payload["known_failures"], ["flaky exit"])
        self.assertEqual(payload["gate"], False)
        self.assertEqual(payload["items"], 10)
        self.assertEqual(payload["budget_floor"], 60)
        self.assertEqual(payload["data"], "data/lcb.jsonl")


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.standard = load_toml("standard.toml")

    def test_unknown_prerequisite_is_reported(self):
        problems = validate_legs([make_leg(prereqs=("teleporter",))])
        self.assertTrue(any("teleporter" in problem for problem in problems))

    def test_unknown_harness_kind_is_reported(self):
        problems = validate_legs([make_leg(harness="vibes")])
        self.assertTrue(any("vibes" in problem for problem in problems))

    def test_zero_items_is_reported(self):
        problems = validate_legs([make_leg(items=0)])
        self.assertTrue(any("item count" in problem for problem in problems))

    def test_missing_known_failures_is_reported(self):
        problems = validate_legs([make_leg(known_failures=())])
        self.assertTrue(any("known failure" in problem for problem in problems))

    def test_duplicate_id_is_reported(self):
        problems = validate_legs([make_leg(), make_leg()])
        self.assertTrue(any("duplicate leg id" in problem for problem in problems))

    def test_missing_raw_field_is_reported(self):
        problems = validate_legs([], [{"id": "half"}])
        self.assertTrue(any("missing field" in problem for problem in problems))

    def test_a_nonpositive_context_window_is_reported(self):
        problems = validate_legs([make_leg(ctx_need=0)])
        self.assertTrue(any("ctx_need must be positive" in problem for problem in problems))

    def test_a_leg_without_an_entry_script_is_reported(self):
        problems = validate_legs([make_leg(entry="")])
        self.assertTrue(any("entry script" in problem for problem in problems))

    def test_a_leg_without_comparability_keys_is_reported(self):
        problems = validate_legs([make_leg(comparability_keys=())])
        self.assertTrue(any("comparability key" in problem for problem in problems))


class InterpreterTest(unittest.TestCase):
    """A leg can borrow another venv's interpreter, which is how the LCB leg runs at all."""

    def test_the_lcb_leg_declares_the_interpreter_its_prompt_helpers_need(self):
        leg = legs_by_id(load_legs(load_toml("standard.toml")))["lcb"]
        self.assertEqual(leg.interpreter, "/mnt/data/benchmark-venv/bin/python3")
        self.assertEqual(leg.to_dict()["interpreter"], leg.interpreter)

    def test_a_leg_without_one_borrows_nothing(self):
        self.assertEqual(make_leg().interpreter, "")

    def test_a_relative_interpreter_is_reported(self):
        problems = validate_legs([make_leg(interpreter="python3")])
        self.assertIn("must be an absolute path", " ".join(problems))

    def test_an_absolute_interpreter_is_accepted(self):
        self.assertEqual(validate_legs([make_leg(interpreter="/usr/bin/python3")]), [])


if __name__ == "__main__":
    unittest.main()
