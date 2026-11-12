"""The catalog consistency check: quiet on the shipped catalogs, loud on broken ones.

The catalog paths are injectable, so every rule is driven here against deliberately
broken catalogs written into a throwaway directory as well as against the real ones.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_catalogs import collect_problems, main  # noqa: E402
from tests import fixtures  # noqa: E402

LEG_TEMPLATE = """\
[[leg]]
id = "{leg_id}"
name = "Tiny leg"
measures = "single-turn knowledge"
scoring = "accuracy"
harness = "native"
implementation = "in_repo"
entry = "scripts/{leg_id}.py"
cap_var = "{cap_var}"
ctx_need = 8192
thinking = "primary_on"
thinking_effect = "none"
prereqs = []
resume = "none"
comparability_keys = ["cap", "ctx", "thinking"]
known_failures = ["endpoint unreachable"]

"""

VARIABLE_TEMPLATE = """\
[[variable]]
id = "{var_id}"
class = "M"
applies_to = "{applies_to}"
stamp_key = "{var_id}"
what_it_changes = "the knob"
effect = "none measured"
source = "the pilot"
decision_rule = "use the standard value"

"""


def write_catalog(directory: str, legs: str, variables: str, standard: str) -> tuple:
    paths = (os.path.join(directory, "standard.toml"),
             os.path.join(directory, "legs.toml"),
             os.path.join(directory, "variables.toml"))
    for path, text in zip(paths, (standard, legs, variables)):
        Path(path).write_text(text)
    return paths


def catalog_for(directory: str, leg_ids: list[str], cap_var: str = "caps.default",
                extra_variables: str = "") -> tuple:
    """A small consistent catalog: the three core variables plus one leg per id."""
    legs = "".join(LEG_TEMPLATE.format(leg_id=leg_id, cap_var=cap_var)
                   for leg_id in leg_ids)
    variables = "".join(VARIABLE_TEMPLATE.format(var_id=var_id, applies_to="all")
                        for var_id in ("cap", "ctx", "thinking")) + extra_variables
    items = "".join(f"{leg_id} = 10\n" for leg_id in sorted(set(leg_ids)))
    standard = (f'standard_version = "test-9.9"\n[caps]\ndefault = 4096\n'
                f"[budget.items]\n{items}")
    return write_catalog(directory, legs, variables, standard)


class CollectProblemsTest(unittest.TestCase):
    def test_the_real_catalogs_have_no_problems(self):
        problems, summary = collect_problems()
        self.assertEqual(problems, [])
        self.assertEqual(summary["legs"], len(fixtures.LEGS))
        self.assertEqual(summary["implemented"] + summary["port_pending"],
                         summary["legs"])
        self.assertEqual(summary["variables"], len(fixtures.VARIABLES))
        self.assertEqual(sorted(summary["caps"]),
                         sorted(leg.id for leg in fixtures.LEGS))

    def test_a_duplicate_leg_id_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems, _ = collect_problems(catalog_for(tmp, ["tiny", "tiny"]))
        self.assertIn("duplicate leg id 'tiny'", problems)

    def test_a_leg_field_left_out_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = catalog_for(tmp, ["tiny"])
            legs_text = Path(paths[1]).read_text().replace('scoring = "accuracy"\n', "")
            write_catalog(tmp, legs_text, Path(paths[2]).read_text(),
                          Path(paths[0]).read_text())
            problems, _ = collect_problems(paths)
        self.assertIn("leg tiny: missing field 'scoring'", problems)

    def test_a_cap_var_missing_from_the_standard_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems, summary = collect_problems(catalog_for(tmp, ["tiny"],
                                                             cap_var="caps.vip"))
        self.assertIn("leg tiny: cap_var 'caps.vip' is not in the standard", problems)
        self.assertNotIn("tiny", summary["caps"])

    def test_a_variable_pointing_at_an_unknown_leg_is_named(self):
        extra = VARIABLE_TEMPLATE.format(var_id="temp", applies_to="no_such_leg")
        with tempfile.TemporaryDirectory() as tmp:
            problems, _ = collect_problems(catalog_for(tmp, ["tiny"],
                                                       extra_variables=extra))
        self.assertIn("temp: applies_to 'no_such_leg' is not a known leg", problems)

    def test_a_variable_field_left_out_is_named(self):
        extra = VARIABLE_TEMPLATE.format(var_id="temp", applies_to="all").replace(
            'effect = "none measured"\n', "")
        with tempfile.TemporaryDirectory() as tmp:
            problems, _ = collect_problems(catalog_for(tmp, ["tiny"],
                                                       extra_variables=extra))
        self.assertIn("variable temp: missing field 'effect'", problems)


class MainTest(unittest.TestCase):
    def test_a_clean_catalog_exits_zero_and_prints_the_summary(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["problems"], [])
        self.assertEqual(payload["legs"], len(fixtures.LEGS))

    def test_a_clean_catalog_in_text_mode_says_so(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("catalogs OK", out.getvalue())
        self.assertIn("cap", out.getvalue())

    def test_a_broken_catalog_exits_one_and_names_the_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = catalog_for(tmp, ["tiny", "tiny"])
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main([], paths=broken)
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM: duplicate leg id 'tiny'", err.getvalue())

    def test_a_broken_catalog_in_json_mode_carries_the_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = catalog_for(tmp, ["tiny"], cap_var="caps.vip")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--json"], paths=broken)
        self.assertEqual(code, 1)
        payload = json.loads(out.getvalue())
        self.assertTrue(any("caps.vip" in problem for problem in payload["problems"]))


if __name__ == "__main__":
    unittest.main()
