"""Judge calibration: four answers of known quality, one pass, a verdict.

The scoring function is injected, so the verdict logic runs against fixed records with no
network at all. The real scoring path (judge.score_answer) is exercised by test_judge; the
one subprocess test here proves the CLI reports a judge that errors instead of passing it.
"""
from __future__ import annotations

import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calibrate_judge import (  # noqa: E402
    CALIBRATION_ANSWERS,
    CALIBRATION_STAMP,
    CALIBRATION_TASK,
    DISCRIMINATION_SPREAD,
    build_parser,
    judge_from_args,
    main,
    render_report,
    run_calibration,
    verdict_exit_code,
)
from scripts.lib.model import ModelConfig  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fixed_scorer(scores: dict[str, float], errors: dict[str, str] | None = None,
                 unscored: set[str] | None = None):
    """A run_calibration score function: keyed by the answer text, via its label."""
    errors = errors or {}
    unscored = unscored or set()

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
        if label in unscored:
            return {"scores": [], "mean": None, "empty": not answer.strip(),
                    "judge_error": None}
        return {"scores": [{"criterion": "answer", "score": scores[label],
                            "justification": "fixed"}],
                "mean": scores[label], "empty": not answer.strip(), "judge_error": None}

    return score


def score_answer_stub(scores: dict[str, float], errors: dict[str, str] | None = None):
    """A judge.score_answer stand-in: same records, keyed by the answer text."""
    inner = fixed_scorer(scores, errors)
    return lambda _judge, _task, answer: inner("", answer)


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


class JudgeResolution(unittest.TestCase):
    """The judge is the standard's pin unless the operator named one, per flag."""

    def args(self, judge_model="", judge_endpoint="", api_key="not-needed"):
        argv = ["--endpoint", "http://run:11434/v1",
                "--judge-model", judge_model,
                "--judge-endpoint", judge_endpoint,
                "--api-key", api_key]
        return build_parser().parse_args(argv)

    def test_the_pin_names_the_judge_when_the_operator_names_none(self):
        args = self.args()
        standard = {"judge": {"model": "deckard-40b", "endpoint": ""}}
        with mock.patch("scripts.calibrate_judge.load_toml", return_value=standard), \
                mock.patch("scripts.calibrate_judge.setup_judge",
                           return_value=ModelConfig(name="judge")) as setup:
            judge = judge_from_args(args)
        setup.assert_called_once_with(model="deckard-40b", endpoint="http://run:11434/v1",
                                      api_key="not-needed")
        self.assertEqual(judge.name, "judge")

    def test_the_operator_flags_override_the_pin(self):
        args = self.args(judge_model="j2", judge_endpoint="http://judge:9/v1",
                         api_key="k2")
        standard = {"judge": {"model": "deckard-40b", "endpoint": "http://pin/v1"}}
        with mock.patch("scripts.calibrate_judge.load_toml", return_value=standard), \
                mock.patch("scripts.calibrate_judge.setup_judge",
                           return_value=ModelConfig(name="j2")) as setup:
            judge_from_args(args)
        setup.assert_called_once_with(model="j2", endpoint="http://judge:9/v1", api_key="k2")

    def test_flag_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.endpoint, "")
        self.assertEqual(args.judge_model, "")
        self.assertEqual(args.judge_endpoint, "")
        self.assertEqual(args.api_key, "not-needed")
        self.assertFalse(args.json)
        self.assertEqual(args.out, "")


class Main(unittest.TestCase):
    """The CLI prints the report or the JSON, writes --out, and exits on the verdict."""

    def run_main(self, argv, scores=None, errors=None):
        if scores is None:
            scores = {"good": 9.0, "bad": 2.0, "truncated": 4.0, "empty": 1.0}
        with mock.patch("scripts.calibrate_judge.score_answer",
                        side_effect=score_answer_stub(scores, errors)):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(argv)
        return code, buffer.getvalue()

    def test_the_report_form_prints_and_exits_zero_when_it_discriminates(self):
        code, out = self.run_main(["--endpoint", "http://127.0.0.1:1/v1"])
        self.assertEqual(code, 0)
        self.assertIn("Calibration answers", out)
        self.assertIn("Verdict: discriminates", out)
        self.assertNotIn("{", out)

    def test_the_json_form_prints_the_record(self):
        code, out = self.run_main(["--json"])
        self.assertEqual(code, 0)
        record = json.loads(out)
        self.assertIn(CALIBRATION_STAMP, record)

    def test_a_flat_judge_exits_one(self):
        code, out = self.run_main([], scores={"good": 5, "bad": 5, "truncated": 5,
                                              "empty": 5})
        self.assertEqual(code, 1)
        self.assertIn("does not discriminate", out)

    def test_the_record_is_written_to_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibration.json")
            code, _ = self.run_main(["--out", path])
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        self.assertEqual(record[CALIBRATION_STAMP]["verdict"], "discriminates")

    def test_the_script_form_exits_on_the_verdict(self):
        """python3 scripts/calibrate_judge.py must run: the guard is a real entry point."""
        argv = ["scripts/calibrate_judge.py", "--endpoint", "http://127.0.0.1:1/v1"]
        stub = score_answer_stub({}, errors={"good": "refused", "bad": "refused",
                                             "truncated": "refused", "empty": "refused"})
        with mock.patch("sys.argv", argv), \
                mock.patch("scripts.lib.judge.score_answer", side_effect=stub):
            buffer = io.StringIO()
            with redirect_stdout(buffer), self.assertRaises(SystemExit) as caught:
                runpy.run_path("scripts/calibrate_judge.py", run_name="__main__")
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("judge error", buffer.getvalue())


class FewerThanTwoScored(unittest.TestCase):
    def test_a_judge_that_scores_almost_nothing_is_not_discrimination(self):
        record = run_calibration(fixed_scorer(
            {"good": 9.0, "bad": 2.0, "truncated": 4.0, "empty": 1.0},
            unscored={"bad", "truncated", "empty"}))
        stamp = record[CALIBRATION_STAMP]
        self.assertEqual(stamp["verdict"], "does not discriminate")
        self.assertEqual(stamp["reason"], "fewer than two answers were scored")
        self.assertEqual(stamp["spread"], 0.0)

    def test_a_judge_that_scores_nothing_is_not_discrimination_either(self):
        record = run_calibration(fixed_scorer(
            {"good": 9.0, "bad": 2.0, "truncated": 4.0, "empty": 1.0},
            unscored={"good", "bad", "truncated", "empty"}))
        self.assertEqual(record[CALIBRATION_STAMP]["verdict"], "does not discriminate")


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
