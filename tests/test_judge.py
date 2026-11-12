"""The judge pin: which model scores the writing, and where that answer comes from.

The judge model is a comparability key, so it belongs in the standard. The endpoint stays
the operator's, because the judge lives wherever their proxy is.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.judge import resolve_judge_defaults  # noqa: E402

PINNED = {"judge": {"model": "nous-glm-5-3-flash", "endpoint": ""}}


class JudgeDefaultsTest(unittest.TestCase):
    def test_the_standard_pins_the_model_when_the_operator_is_silent(self):
        model, endpoint = resolve_judge_defaults("", "", "http://run:4000/v1", PINNED)
        self.assertEqual(model, "nous-glm-5-3-flash")
        self.assertEqual(endpoint, "http://run:4000/v1")

    def test_the_operator_beats_the_pin(self):
        resolved = resolve_judge_defaults("mine", "http://judge/v1", "http://run/v1", PINNED)
        self.assertEqual(resolved, ("mine", "http://judge/v1"))

    def test_a_pinned_judge_endpoint_wins_over_the_run_endpoint(self):
        pin = {"judge": {"model": "j", "endpoint": "http://judge/v1"}}
        self.assertEqual(resolve_judge_defaults("", "", "http://run/v1", pin)[1],
                         "http://judge/v1")

    def test_a_standard_with_no_judge_section_leaves_the_judge_unset(self):
        model, endpoint = resolve_judge_defaults("", "", "http://run/v1", {})
        self.assertEqual(model, "")
        self.assertEqual(endpoint, "http://run/v1")

    def test_the_run_endpoint_is_not_used_when_the_operator_named_a_judge(self):
        resolved = resolve_judge_defaults("", "http://judge/v1", "http://run/v1", PINNED)
        self.assertEqual(resolved, ("nous-glm-5-3-flash", "http://judge/v1"))


if __name__ == "__main__":
    unittest.main()
