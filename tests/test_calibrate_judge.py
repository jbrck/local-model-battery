"""Judge calibration: four answers of known quality, one pass, a verdict.

The scoring function is injected, so the verdict logic runs against fixed records with no
network at all. The real scoring path (judge.score_answer) is exercised by test_judge; the
one subprocess test here proves the CLI reports a judge that errors instead of passing it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calibrate_judge import (  # noqa: E402
    CALIBRATION_ANSWERS,
    CALIBRATION_STAMP,
    CALIBRATION_TASK,
    DISCRIMINATION_SPREAD,
    render_report,
    run_calibration,
    verdict_exit_code,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fixed_scorer(scores: dict[str, float], errors: dict[str, str] | None = None):
    """A score function with the shape of judge.score_answer's return value."""
    errors = errors or {}

    def score(_label: str, answer: str) -> dict:
        for known_label, known_answer in CALIBRATION_ANSWERS:
            if answer == known_answer:
                label = known_label
                break
        else:
            raise AssertionError(f"calibration saw an unknown answer: {answer!r}")
        if label in errors:
            return {"scores": [], "mean": None, "empty": False,
                    "judge_error": errors[label]}
        return {"scores": [{"criterion": "answer", "score": scores[label],
                            "justification": "fixed"}],
                "mean": scores[label], "empty": not answer.strip(), "judge_error": None}

    return score


class CalibrationVerdict(unittest.TestCase):
    """The verdict on a discriminating spread, a flat one, and an inverted one."""

    def test_every_calibration_answer_is_scored(self) -> None:
        record = run_calibration(fixed_scorer({"good": 9, "bad": 2, "truncated": 4,
                                               "empty": 1}))
        answers = record[CALIBRATION_STAMP]["answers"]
        self.assertEqual(sorted(answers), ["bad", "empty", "good", "truncated"])
        self.assertTrue(answers["empty"]["empty"])

    def test_discriminating_spread_passes(self) -> None:
        record = run_calibration(fixed_scorer({"good": 9, "bad": 2, "truncated": 4,
                                               "empty": 1}))
        stamp = record[CALIBRATION_STAMP]
        self.assertEqual(stamp["verdict"], "discriminates")
        self.assertEqual(verdict_exit_code(record), 0)

    def test_flat_spread_fails_loudly(self) -> None:
        record = run_calibration(fixed_scorer({"good": 5, "bad": 5, "truncated": 5,
                                               "empty": 5}))
        stamp = record[CALIBRATION_STAMP]
        self.assertEqual(stamp["verdict"], "does not discriminate")
        self.assertIn("spread", stamp["reason"])
        self.assertEqual(verdict_exit_code(record), 1)

    def test_inverted_ordering_fails_even_with_a_wide_spread(self) -> None:
        record = run_calibration(fixed_scorer({"good": 2, "bad": 9, "truncated": 4,
                                               "empty": 1}))
        stamp = record[CALIBRATION_STAMP]
        self.assertEqual(stamp["verdict"], "does not discriminate")
        self.assertIn("inverted", stamp["reason"])

    def test_threshold_is_three_points(self) -> None:
        self.assertEqual(DISCRIMINATION_SPREAD, 3.0)


class JudgeErrors(unittest.TestCase):
    """A judge that errors is reported, never silently passed."""

    def test_errored_judge_fails_the_verdict_and_names_the_error(self) -> None:
        record = run_calibration(fixed_scorer(
            {"good": 9, "bad": 2, "truncated": 4, "empty": 1},
            errors={"good": "HTTP 500 upstream exploded"}))
        stamp = record[CALIBRATION_STAMP]
        self.assertEqual(stamp["verdict"], "does not discriminate")
        self.assertIn("HTTP 500 upstream exploded", stamp["reason"])
        self.assertIn("judge error", render_report(record))
        self.assertEqual(verdict_exit_code(record), 1)

    def test_report_prints_one_line_per_answer(self) -> None:
        record = run_calibration(fixed_scorer({"good": 9, "bad": 2, "truncated": 4,
                                               "empty": 1}))
        report = render_report(record)
        for label in ("good", "bad", "truncated", "empty"):
            self.assertIn(label, report)
        self.assertIn("Spread:", report)
        self.assertIn("Verdict:", report)


class CalibrationTask(unittest.TestCase):
    """The fixed task and answers: quality known by construction."""

    def test_task_has_a_rubric_the_scoring_path_can_use(self) -> None:
        self.assertEqual(CALIBRATION_TASK["prompt"], CALIBRATION_TASK["prompt"].strip())
        rubric = CALIBRATION_TASK["rubric"]
        self.assertEqual(len(rubric), 1)
        self.assertTrue(rubric[0]["criterion"])
        self.assertIn("2", rubric[0]["anchors"])

    def test_four_answers_in_a_fixed_order(self) -> None:
        self.assertEqual([label for label, _answer in CALIBRATION_ANSWERS],
                         ["good", "bad", "truncated", "empty"])


class CommandLine(unittest.TestCase):
    """The real CLI against a dead default endpoint: the error path, reported loudly."""

    def test_unreachable_judge_is_reported_not_passed(self) -> None:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "calibrate_judge.py"),
             "--endpoint", "http://127.0.0.1:1/v1"],
            capture_output=True, text=True, timeout=120, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not discriminate", result.stdout)
        self.assertIn("judge error", result.stdout)


if __name__ == "__main__":
    unittest.main()
