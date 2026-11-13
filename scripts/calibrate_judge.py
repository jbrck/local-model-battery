#!/usr/bin/env python3
"""Calibrate a judge: score four answers whose quality is known by construction.

A run scored by a judge that cannot tell a good answer from a bad one is not comparable to
anything, including itself. Before a run is set up, this scores one good, one bad, one
truncated and one empty answer through the same scoring path the writing leg uses, prints
the score per answer, the spread, and a verdict: `discriminates` or `does not
discriminate`. A judge that errors is reported, never silently passed.

One pass, serial. The JSON record is what the run stamps as `judge_calibration`.

    python3 scripts/calibrate_judge.py [--endpoint URL] [--judge-model ID]
                                       [--judge-endpoint URL] [--api-key K] [--json]

Exit code 0 means the judge discriminates; 1 means it does not, or errored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib.judge import resolve_judge_defaults, score_answer, setup_judge  # noqa: E402
from scripts.lib.model import DEFAULT_ENDPOINT  # noqa: E402

STANDARD_FILE = "standard.toml"
CALIBRATION_STAMP = "judge_calibration"

# A judge that cannot open this much distance between its highest and lowest calibration
# score cannot order models, so its scores are noise.
DISCRIMINATION_SPREAD = 3.0

# The calibration task is fixed so two judges are comparable against each other. The
# rubric has one criterion whose anchors bracket exactly what the four answers do.
CALIBRATION_TASK = {
    "prompt": "Name the largest planet in the solar system, and say in one sentence "
              "why it is the largest.",
    "rubric": [
        {
            "criterion": "answer",
            "description": "names the largest planet and gives a correct reason",
            "anchors": {
                "2": "no answer, or a wrong planet or object",
                "5": "names the largest planet with no reason, or an incomplete one",
                "8": "names the largest planet and gives a correct reason",
            },
        },
    ],
}

GOOD_ANSWER = ("Jupiter is the largest planet in the solar system. It is the largest "
               "because it holds more mass than all the other planets combined.")
BAD_ANSWER = "The largest planet is the Moon, which shines brightly in the night sky."
TRUNCATED_ANSWER = ("Jupiter is the largest planet in the solar system. It is the "
                    "largest because it holds")
EMPTY_ANSWER = ""

# (label, answer, quality known by construction). The labels are the record's keys.
CALIBRATION_ANSWERS = (
    ("good", GOOD_ANSWER),
    ("bad", BAD_ANSWER),
    ("truncated", TRUNCATED_ANSWER),
    ("empty", EMPTY_ANSWER),
)

ScoreFn = Callable[[str, str], dict]


def run_calibration(score: ScoreFn) -> dict:
    """Score every calibration answer once, serially, and judge the spread.

    `score(label, answer)` returns the record `judge.score_answer` produces. Any judge
    error is carried into the record verbatim and fails the verdict; it is never swallowed.
    """
    answers: dict[str, dict] = {}
    for label, answer in CALIBRATION_ANSWERS:
        answers[label] = score(label, answer)
    scored = [record["mean"] for record in answers.values()
              if record.get("judge_error") is None and record.get("mean") is not None]
    errors = {label: record["judge_error"] for label, record in answers.items()
              if record.get("judge_error")}
    spread = round(max(scored) - min(scored), 3) if len(scored) >= 2 else 0.0
    verdict, reason = _verdict(answers, scored, errors, spread)
    return {
        CALIBRATION_STAMP: {
            "answers": answers,
            "spread": spread,
            "verdict": verdict,
            "reason": reason,
        },
    }


def _verdict(answers: dict, scored: list, errors: dict, spread: float) -> tuple[str, str]:
    """(verdict, reason). An errored or flat or inverted judge does not discriminate."""
    if errors:
        named = "; ".join(f"{label}: {error}" for label, error in errors.items())
        return "does not discriminate", f"the judge errored: {named}"
    if len(scored) < 2:
        return "does not discriminate", "fewer than two answers were scored"
    if spread < DISCRIMINATION_SPREAD:
        return ("does not discriminate",
                f"the spread is {spread}, under the {DISCRIMINATION_SPREAD} the verdict needs")
    if answers["good"]["mean"] <= answers["bad"]["mean"]:
        return ("does not discriminate",
                "the good answer did not outscore the bad one, so the ordering is inverted")
    return "discriminates", f"the spread is {spread} across the four answers"


def verdict_exit_code(record: dict) -> int:
    """0 only when the judge discriminates; a run setup fails loudly otherwise."""
    stamp = record[CALIBRATION_STAMP]
    return 0 if stamp["verdict"] == "discriminates" else 1


def render_report(record: dict) -> str:
    """The human-readable report: one line per answer, the spread, the verdict."""
    stamp = record[CALIBRATION_STAMP]
    lines = ["Calibration answers, scored blind:"]
    for label, result in stamp["answers"].items():
        if result.get("judge_error"):
            lines.append(f"  {label}: judge error -- {result['judge_error']}")
        else:
            lines.append(f"  {label}: {result['mean']}")
    lines.append(f"Spread: {stamp['spread']}")
    lines.append(f"Verdict: {stamp['verdict']} ({stamp['reason']})")
    return "\n".join(lines)


def judge_from_args(args: argparse.Namespace):
    """The judge the calibration scores with: the pin, unless the operator named one."""
    standard = load_toml(STANDARD_FILE)
    run_endpoint = args.endpoint or DEFAULT_ENDPOINT
    model, endpoint = resolve_judge_defaults(args.judge_model, args.judge_endpoint,
                                             run_endpoint, standard)
    return setup_judge(model=model or "", endpoint=endpoint or run_endpoint,
                       api_key=args.api_key)


def build_parser() -> argparse.ArgumentParser:
    """The calibration flags. The judge is named, never inherited silently."""
    parser = argparse.ArgumentParser(
        description="Prove a judge scores: four answers of known quality, one pass.")
    parser.add_argument("--endpoint", default="",
                        help="base URL the judge is served from when --judge-endpoint is unset")
    parser.add_argument("--judge-model", default="",
                        help="the judge to calibrate; empty means the standard's pin")
    parser.add_argument("--judge-endpoint", default="",
                        help="the judge's base URL when it differs from --endpoint")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    parser.add_argument("--out", default="", help="also write the record to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    judge = judge_from_args(args)
    record = run_calibration(lambda _label, answer: score_answer(judge, CALIBRATION_TASK,
                                                                 answer))
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print(render_report(record))
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return verdict_exit_code(record)


if __name__ == "__main__":
    raise SystemExit(main())
