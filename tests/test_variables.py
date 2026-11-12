"""Variable catalog tests."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.variables import (  # noqa: E402
    ALL_LEGS,
    Variable,
    load_raw_variables,
    load_variables,
    measurement_variables,
    validate_variables,
    variables_by_id,
    variables_for_leg,
)


def make_variable(**overrides) -> Variable:
    fields = dict(
        id="sample", variable_class="M", applies_to=("all",), stamp_key="sample",
        what_it_changes="something", effect="nothing measured yet",
        source="test", decision_rule="take the default",
    )
    fields.update(overrides)
    # `**fields` erases the per-parameter types, which the checker cannot follow.
    return Variable(**fields)  # type: ignore[arg-type]


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.standard = load_toml("standard.toml")
        self.legs = load_legs(self.standard)
        self.variables = load_variables()
        self.by_id = variables_by_id(self.variables)

    def test_shipped_catalog_validates_clean(self):
        self.assertEqual(validate_variables(self.variables, self.legs, load_raw_variables()), [])

    def test_every_class_is_known(self):
        for variable in self.variables:
            self.assertIn(variable.variable_class, ("M", "C", "D"), variable.id)

    def test_the_knobs_that_move_a_score_are_classed_as_measurement(self):
        for knob in ("thinking", "cap", "ctx", "seed", "temperature", "kv_type", "item_set"):
            self.assertTrue(self.by_id[knob].is_measurement, knob)

    def test_speed_only_knobs_are_classed_as_capacity(self):
        for knob in ("power_limit", "offload_placement", "weight_quant"):
            self.assertEqual(self.by_id[knob].variable_class, "C", knob)

    def test_missing_external_things_are_classed_as_dependency(self):
        for knob in ("docker_sandbox", "nltk_data", "lcb_source", "tool_eval_bench"):
            self.assertEqual(self.by_id[knob].variable_class, "D", knob)

    def test_every_variable_explains_itself(self):
        for variable in self.variables:
            self.assertTrue(variable.what_it_changes, variable.id)
            self.assertTrue(variable.effect, variable.id)
            self.assertTrue(variable.decision_rule, variable.id)
            self.assertTrue(variable.source, variable.id)

    def test_effects_carry_measured_numbers_where_we_have_them(self):
        self.assertIn("28.3", self.by_id["thinking"].effect)
        self.assertIn("79 of 108", self.by_id["cap"].effect)
        self.assertIn("15 to 20 percent", self.by_id["power_limit"].effect)

    def test_applies_to_all_covers_every_leg(self):
        for leg in self.legs:
            ids = {variable.id for variable in variables_for_leg(self.variables, leg.id)}
            self.assertIn("cap", ids, leg.id)
            self.assertIn("ctx", ids, leg.id)

    def test_leg_specific_variables_stay_scoped(self):
        self.assertEqual(self.by_id["docker_sandbox"].applies_to, ("hermesagent20",))
        self.assertFalse(self.by_id["nltk_data"].applies_to_leg("mmlu"))
        self.assertTrue(self.by_id["nltk_data"].applies_to_leg("ifbench"))

    def test_measurement_set_is_the_smaller_half(self):
        measurements = [v.id for v in measurement_variables(self.variables)]
        self.assertIn("ctx", measurements)
        self.assertNotIn("power_limit", measurements)
        self.assertLess(len(measurements), len(self.variables))

    def test_all_legs_sentinel_is_used_for_global_knobs(self):
        self.assertEqual(self.by_id["ctx"].applies_to, (ALL_LEGS,))


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.legs = load_legs(load_toml("standard.toml"))

    def test_unknown_variable_class_is_reported(self):
        problems = validate_variables([make_variable(variable_class="Z")], self.legs)
        self.assertTrue(any("class 'Z'" in problem for problem in problems))

    def test_unknown_leg_in_applies_to_is_reported(self):
        problems = validate_variables([make_variable(applies_to=("ghost_leg",))], self.legs)
        self.assertTrue(any("ghost_leg" in problem for problem in problems))

    def test_variable_without_an_effect_is_reported(self):
        problems = validate_variables([make_variable(effect="")], self.legs)
        self.assertTrue(any("measured effect" in problem for problem in problems))

    def test_variable_without_a_decision_rule_is_reported(self):
        problems = validate_variables([make_variable(decision_rule="")], self.legs)
        self.assertTrue(any("decision rule" in problem for problem in problems))

    def test_leg_missing_a_core_knob_is_reported(self):
        from scripts.lib.legs import Leg
        legs = [Leg(id="orphan", name="Orphan", measures="m", scoring="s", harness="native",
                    implementation="in_repo", entry="x.py", cap_var="caps.default",
                    ctx_need=8192, thinking="primary_on", thinking_effect="e", prereqs=(),
                    resume="none", comparability_keys=("cap",), known_failures=("f",),
                    items=1, budget_floor=1)]
        problems = validate_variables([make_variable()], legs)
        self.assertTrue(any("not named by any variable" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
