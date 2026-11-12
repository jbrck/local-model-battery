"""Phase 2 scoring for the LiveCodeBench leg: official lcb_runner codegen metrics."""
from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.adapters.lcb import LcbArgs

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_worker.py")


def scorable_pairs(problems: list[dict],
                   done: dict[str, dict]) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """The (question id, code) pairs the scorer can judge, plus the raw problem lookup."""
    by_qid = {str(problem["question_id"]): problem for problem in problems}
    pairs = [(qid, done[qid]["code"]) for qid in sorted(done)
             if done[qid].get("code") and qid in by_qid]
    return pairs, by_qid


def passes_from_detail(detail: dict, scorable: list[tuple[str, str]]) -> dict[str, bool]:
    """The scorer's per-index pass@1 detail, mapped back onto question ids."""
    return {qid: float(detail.get(str(index)) or 0.0) > 0.5
            for index, (qid, _code) in enumerate(scorable)}


def score(problems: list[dict], done: dict[str, dict],
          args: "LcbArgs") -> tuple[dict[str, bool], str | None]:
    """Run the official codegen metrics in the scorer interpreter."""
    scorable, by_qid = scorable_pairs(problems, done)
    if not scorable:
        return {}, "no problem produced extractable code"
    input_path = os.path.join(args.output_dir, "lcb-score-input.json")
    output_path = os.path.join(args.output_dir, "lcb-score-output.json")
    payload = {"problems": [by_qid[qid] for qid, _ in scorable],
               "generations": [[code] for _, code in scorable],
               "workers": args.workers, "exec_timeout": args.exec_timeout}
    with open(input_path, "w") as handle:
        json.dump(payload, handle)
    result = subprocess.run([args.scorer_python, WORKER_SCRIPT, input_path, output_path],
                            cwd=args.src, env={**os.environ, "PYTHONPATH": args.src})
    if result.returncode != 0:
        return {}, f"scorer exited {result.returncode}"
    with open(output_path) as handle:
        detail = json.load(handle).get("detail", {})
    return passes_from_detail(detail, scorable), None
