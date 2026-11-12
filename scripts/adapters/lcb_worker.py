"""Scorer subprocess for the LiveCodeBench leg: the official codegen metrics.

Runs under the scorer interpreter with cwd set to the LiveCodeBench checkout,
which is where lcb_runner expects to find its few-shot prompt files.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable


def run_metrics(payload: dict, codegen_metrics: Callable, make_problem: Callable) -> dict:
    """Score one aligned (problems, generations) batch and return the pass detail."""
    problems = [make_problem(problem) for problem in payload["problems"]]
    samples = [problem.get_evaluation_sample() for problem in problems]
    metrics, _results, _metadata = codegen_metrics(
        samples, payload["generations"],
        num_process_evaluate=payload["workers"], timeout=payload["exec_timeout"], debug=False)
    return {"pass@1": metrics.get("pass@1"),
            "detail": metrics.get("detail", {}).get("pass@1", {})}


def main(argv: list[str]) -> None:  # pragma: no cover - runs in the scorer subprocess
    """Read the score input, run the metrics, write the score output."""
    _worker, input_path, output_path = argv
    checkout = os.getcwd()
    if checkout not in sys.path:
        sys.path.insert(0, checkout)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    def make_problem(raw: dict) -> object:
        return CodeGenerationProblem(**raw)

    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    with open(input_path) as handle:
        payload = json.load(handle)
    result = run_metrics(payload, codegen_metrics, make_problem)
    with open(output_path, "w") as handle:
        json.dump(result, handle)


if __name__ == "__main__":  # pragma: no cover - subprocess entry
    main(sys.argv[1:])
