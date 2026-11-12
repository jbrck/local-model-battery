#!/usr/bin/env python3
"""LiveCodeBench leg (release_v6 lite): generate code per problem, then score with lcb_runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.adapters.lcb_guard import FailureKind, Guard  # noqa: E402
from scripts.lib import harness  # noqa: E402
from scripts.lib.model import ChatResponse, ModelConfig  # noqa: E402

LEG_ID = "lcb"
DEFAULT_DATA = "/mnt/data/lcb-data/test6.jsonl"
DEFAULT_SRC = "/mnt/data/lcb-src/LiveCodeBench"
DEFAULT_SCORER_PYTHON = "/mnt/data/benchmark-venv/bin/python3"
LCB_CAP = 114688
DEFAULT_MAX_HOURS = 8.0
DEFAULT_STALL_MIN = 10
DEFAULT_MAX_FAILURES = 5
DEFAULT_WORKERS = 4
DEFAULT_EXEC_TIMEOUT = 6
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_worker.py")


@dataclass
class Helpers:
    """The two lcb_runner entry points the generation phase needs."""

    format_prompt: Callable[[dict], list[dict]]
    extract_code: Callable[[str], str]


@dataclass
class LcbArgs(harness.LegArgs):
    """The shared leg flags plus everything this leg needs to find its tooling."""

    src: str = DEFAULT_SRC
    scorer_python: str = DEFAULT_SCORER_PYTHON
    max_hours: float = DEFAULT_MAX_HOURS
    stall_min: int = DEFAULT_STALL_MIN
    max_consecutive_failures: int = DEFAULT_MAX_FAILURES
    workers: int = DEFAULT_WORKERS
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT
    slots: int = 0
    no_guard: bool = False


def parse_args(argv: list[str] | None = None) -> LcbArgs:
    """The shared leg CLI with the standard's cap for this leg as the default."""
    parser = argparse.ArgumentParser(description="LiveCodeBench (release_v6 lite) leg")
    harness.add_leg_args(parser, default_data=os.environ.get("LCB_DATA") or DEFAULT_DATA)
    parser.set_defaults(cap=LCB_CAP)
    parser.add_argument("--src", default=os.environ.get("LCB_SRC") or DEFAULT_SRC,
                        help="LiveCodeBench checkout carrying lcb_runner")
    parser.add_argument("--scorer-python", default=DEFAULT_SCORER_PYTHON,
                        help="interpreter with the scorer's dependencies")
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--stall-min", type=int, default=DEFAULT_STALL_MIN)
    parser.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT)
    parser.add_argument("--slots", type=int, default=0,
                        help="server slots the leg ran against, for the comparability stamp")
    parser.add_argument("--no-guard", action="store_true")
    return LcbArgs(**vars(parser.parse_args(argv)))


def load_helpers(src: str) -> Helpers:  # pragma: no cover - needs the checkout named by --src
    """Import the prompt and extraction helpers from the checkout named by --src.

    Inline imports: the checkout path is a runtime flag, so lcb_runner cannot be
    imported at the top of the file.
    """
    sys.path.insert(0, src)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.utils.extraction_utils import extract_code

    def format_prompt(problem: dict) -> list[dict]:
        return format_prompt_generation(CodeGenerationProblem(**problem), LMStyle.OpenAIChat)

    def extract(content: str) -> str:
        return extract_code(content, LMStyle.OpenAIChat) or ""

    return Helpers(format_prompt=format_prompt, extract_code=extract)


def load_problems(path: str, limit: int, seed: int) -> list[dict]:
    """The dataset's problems in the legacy order: sort, seeded shuffle, limit, sort."""
    with open(path) as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    raw.sort(key=lambda problem: str(problem["question_id"]))
    random.Random(seed).shuffle(raw)
    if limit:
        raw = raw[:limit]
    raw.sort(key=lambda problem: str(problem["question_id"]))
    return raw


def items_sha(problems: list[dict]) -> str:
    """The comparability stamp over exactly these problems, in run order."""
    ids = [str(problem["question_id"]) for problem in problems]
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()


def partial_path(args: LcbArgs) -> str:
    """The resumable generation log for this model and output directory."""
    return os.path.join(args.output_dir, f"{args.model}-lcb.partial.jsonl")


def load_done(path: str) -> dict[str, dict]:
    """Every generation already in the partial jsonl, keyed by question id."""
    if not os.path.exists(path):
        return {}
    done: dict[str, dict] = {}
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(record.get("question_id"))] = record
    return done


def append_record(path: str, record: dict) -> None:
    """Append one generation, creating the output directory on the way."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def write_aborted(args: LcbArgs, reason: str, done: dict[str, dict], elapsed_s: float) -> str:
    """Persist why generation stopped, so a truncated leg is never read as a full one."""
    path = partial_path(args)[: -len(".partial.jsonl")] + ".aborted.json"
    payload = {"aborted": True, "reason": reason, "model": args.model,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(elapsed_s, 1),
               "budget_hours": args.max_hours, "generated": len(done),
               "with_code": sum(1 for record in done.values() if record.get("code"))}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    return path


def classify_response(qid: str, response: ChatResponse, helpers: Helpers,
                      tally: harness.Tally) -> tuple[FailureKind, dict]:
    """Account for one response and shape it into a partial record."""
    record = {"question_id": qid, "code": "", "error": None,
              "completion_tokens": response.completion_tokens}
    if not response.ok:
        record["error"] = response.error or f"HTTP {response.status}"
        tally.record(response)
        return FailureKind.ENGINE, record
    record["code"] = helpers.extract_code(response.content)
    tally.record(response, correct=None)
    if record["code"]:
        return FailureKind.NONE, record
    record["error"] = "no code extracted"
    tally.failed += 1
    tally.errors.append(f"{qid}: no code extracted")
    return FailureKind.NO_CODE, record


def generate(problems: list[dict], helpers: Helpers, cfg: ModelConfig, args: LcbArgs,
             tally: harness.Tally) -> tuple[dict[str, dict], str | None]:
    """Phase 1: one request per problem, appended to the resumable partial jsonl."""
    done = load_done(partial_path(args))
    guard = None if args.no_guard else Guard(args.max_hours, args.stall_min,
                                             args.max_consecutive_failures)
    started = time.monotonic()
    for problem in problems:
        qid = str(problem["question_id"])
        if qid in done:
            continue
        response = harness.ask(cfg, helpers.format_prompt(problem), args)
        kind, record = classify_response(qid, response, helpers, tally)
        done[qid] = record
        append_record(partial_path(args), record)
        if guard:
            decision = guard.record(kind, time.monotonic())
            if decision.stop:
                write_aborted(args, decision.reason, done, time.monotonic() - started)
                return done, decision.reason
    return done, None


def score(problems: list[dict], done: dict[str, dict],
          args: LcbArgs) -> tuple[dict[str, bool], str | None]:
    """Phase 2: run the official codegen metrics in the scorer interpreter."""
    by_qid = {str(problem["question_id"]): problem for problem in problems}
    scorable = [(qid, done[qid]["code"]) for qid in sorted(done)
                if done[qid].get("code") and qid in by_qid]
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
    passed = {}
    for index, (qid, _code) in enumerate(scorable):
        value = detail.get(str(index), detail.get(index, 0.0))
        passed[qid] = float(value or 0.0) > 0.5
    return passed, None


def run(args: LcbArgs) -> dict:
    """Run both phases and write the result block."""
    problems = load_problems(args.data, args.limit, args.seed)
    tally = harness.Tally()
    extra: dict = {"slots": args.slots, "items_sha": items_sha(problems)}
    if not problems:
        tally.notes.append(f"no problems in {args.data}")
    else:
        done, stop = generate(problems, load_helpers(args.src), harness.leg_config(args),
                              args, tally)
        passed, score_error = score(problems, done, args)
        for problem in problems:
            qid = str(problem["question_id"])
            if qid in passed:
                tally.scored += 1
                tally.correct += int(passed[qid])
        if score_error:
            tally.errors.append(score_error)
        extra["generated"] = len(done)
        extra["ungenerated"] = len(problems) - len(done)
        extra["no_code"] = sum(1 for record in done.values() if not record.get("code"))
        extra["pass_at_1"] = round(tally.accuracy, 4)
        if stop:
            extra["aborted"] = True
            extra["abort_reason"] = stop
    return harness.finish(LEG_ID, args, tally, extra=extra)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
