#!/usr/bin/env python3
"""Writing Suite v1 leg: generate every answer, score only when a judge exists.

Generation is unconditional and writes every sample plus the result block.
Scoring uses the blind rubric judge in scripts.lib.judge, driven by
--judge-model/--judge-endpoint or WRITING_JUDGE_MODEL/WRITING_JUDGE_ENDPOINT.
With no judge the leg reports itself not scored, names what is missing, exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.harness import (  # noqa: E402
    LegArgs,
    Tally,
    add_leg_args,
    ask,
    finish,
    leg_config,
    thinking_body,
)
from scripts.lib.judge import score_answer, setup_judge  # noqa: E402
from scripts.lib.model import ChatResponse, ModelConfig, build_body  # noqa: E402

LEG_ID = "writing"
DEFAULT_DATA = "data/writing_suite_v1.json"
WRITING_CAP = 16384
WRITING_TEMPERATURE = 0.7
SOURCE_HEADER = "\n\nSOURCE TEXT:\n"
JUDGE_MODEL_ENV = "WRITING_JUDGE_MODEL"
JUDGE_ENDPOINT_ENV = "WRITING_JUDGE_ENDPOINT"


def build_user_message(task: dict) -> str:
    """Compose the single user message for one task, verbatim task text first."""
    prompt = task["prompt"]
    source = (task.get("source_text") or "").strip()
    if source and source not in prompt:
        return f"{prompt}{SOURCE_HEADER}{source}"
    return prompt


def safe_model_dir(model: str) -> str:
    return model.replace("/", "--").replace(":", "--").replace(" ", "-")


def sample_path(output_dir: str, model: str, task_id: str) -> str:
    return os.path.join(output_dir, safe_model_dir(model), f"{task_id}.json")


def write_sample(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)


def load_suite(path: str) -> tuple[str, list[dict]]:
    """The suite file carries suite metadata plus its task list."""
    with open(path) as handle:
        suite = json.load(handle)
    return str(suite.get("suite_version", "unknown")), list(suite["tasks"])


def select_tasks(tasks: list[dict], categories: list[str] | None,
                 limit: int) -> list[dict]:
    if categories:
        wanted = set(categories)
        tasks = [task for task in tasks if task["category"] in wanted]
    return tasks[:limit] if limit else tasks


def generate_one(cfg: ModelConfig, args: LegArgs,
                 task: dict, suite_version: str) -> tuple[dict, ChatResponse]:
    """One request per task, exactly as v1 sent it: no system role, no thinking switch."""
    messages = [{"role": "user", "content": build_user_message(task)}]
    response = ask(cfg, messages, args)
    text = response.answer(allow_reasoning_fallback=False)
    record = {
        "task_id": task["task_id"],
        "category": task["category"],
        "suite_version": suite_version,
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(response.elapsed_s, 2),
        "request": build_body(cfg, messages, args.cap, args.temperature,
                              thinking_body(args.thinking)),
        "response": text,
        "empty": text is None,
        "error": response.error,
    }
    return record, response


def run_generation(args: LegArgs, suite_version: str, tasks: list[dict],
                   skip_existing: bool) -> tuple[Tally, dict[str, dict]]:
    """Ask every task, write every sample. Returns the counters and the samples."""
    cfg = leg_config(args)
    tally = Tally()
    samples: dict[str, dict] = {}
    for index, task in enumerate(tasks, start=1):
        path = sample_path(args.output_dir, args.model, task["task_id"])
        if skip_existing and os.path.exists(path):
            with open(path) as handle:
                samples[task["task_id"]] = json.load(handle)
            tally.record(ChatResponse(status=200, payload={}))
            print(f"  [{index}/{len(tasks)}] {task['task_id']:8s} skip (exists)", flush=True)
            continue
        record, response = generate_one(cfg, args, task, suite_version)
        write_sample(path, record)
        samples[task["task_id"]] = record
        tally.record(response)
        size = "EMPTY" if record["empty"] else f"{len(record['response'])} chars"
        print(f"  [{index}/{len(tasks)}] {task['task_id']:8s} {size} "
              f"({record['elapsed_s']}s)", flush=True)
    return tally, samples


def run_scoring(args: LegArgs, tasks: list[dict], samples: dict[str, dict],
                judge_model: str, judge_endpoint: str, judge_api_key: str) -> dict:
    """Score every sample with the blind rubric judge and fold the score into the sample."""
    judge = setup_judge(judge_model, judge_endpoint, judge_api_key)
    results: dict[str, dict] = {}
    for task in tasks:
        record = samples[task["task_id"]]
        outcome = score_answer(judge, task, record.get("response") or "")
        record["judgement"] = outcome
        write_sample(sample_path(args.output_dir, args.model, task["task_id"]), record)
        results[task["task_id"]] = outcome
        mark = "judge error" if outcome["judge_error"] else f"mean {outcome['mean']}"
        print(f"  scored {task['task_id']:8s} {mark}", flush=True)
    return scoring_extra(results, judge_model, judge_endpoint)


def scoring_extra(results: dict[str, dict], judge_model: str, judge_endpoint: str) -> dict:
    scored = {tid: item for tid, item in results.items() if item["judge_error"] is None}
    means = [item["mean"] for item in scored.values()]
    return {
        "judge_model": judge_model,
        "judge_endpoint": judge_endpoint,
        "judge_scored": len(scored),
        "judge_errors": len(results) - len(scored),
        "mean_score": round(sum(means) / len(means), 3) if means else 0.0,
        "scores": {tid: item["mean"] for tid, item in results.items()},
    }


def not_scored(model: str, judge_model: str, judge_endpoint: str) -> dict:
    """The block that says the run happened but carries no score, and why."""
    if judge_model and judge_model == model:
        reason = f"the judge is the model under test ({model}); scoring would not be blind"
    elif judge_model and not judge_endpoint:
        reason = f"judge {judge_model} has no endpoint: pass --judge-endpoint or set {JUDGE_ENDPOINT_ENV}"
    elif judge_endpoint and not judge_model:
        reason = f"a judge endpoint was given but no model: pass --judge-model or set {JUDGE_MODEL_ENV}"
    else:
        reason = ("no judge configured: pass --judge-model and --judge-endpoint "
                  f"or set {JUDGE_MODEL_ENV} and {JUDGE_ENDPOINT_ENV}")
    return {"status": "not scored", "reason": reason}


@dataclass
class Options:
    """The leg args plus the flags only this leg knows."""

    args: LegArgs
    categories: list[str]
    skip_existing: bool
    judge_model: str
    judge_endpoint: str
    judge_api_key: str


def parse_args(argv: list[str] | None) -> Options:
    """The shared leg flags with this leg's standard cap and temperature, plus judge flags."""
    parser = argparse.ArgumentParser(description="Writing Suite v1 generation leg")
    add_leg_args(parser, DEFAULT_DATA)
    parser.set_defaults(cap=WRITING_CAP, temperature=WRITING_TEMPERATURE)
    parser.add_argument("--category", action="append", default=None,
                        help="only run these categories (repeatable)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="keep answers already on disk (resume a stopped run)")
    parser.add_argument("--judge-model", default="",
                        help=f"scoring judge; falls back to {JUDGE_MODEL_ENV}")
    parser.add_argument("--judge-endpoint", default="",
                        help=f"judge base URL; falls back to {JUDGE_ENDPOINT_ENV}")
    parser.add_argument("--judge-api-key", default="not-needed")
    namespace = parser.parse_args(argv)
    fields = {name: vars(namespace)[name] for name in LegArgs.__dataclass_fields__}
    return Options(
        args=LegArgs(**fields),
        categories=list(namespace.category or []),
        skip_existing=namespace.skip_existing,
        judge_model=namespace.judge_model or os.environ.get(JUDGE_MODEL_ENV, ""),
        judge_endpoint=namespace.judge_endpoint or os.environ.get(JUDGE_ENDPOINT_ENV, ""),
        judge_api_key=namespace.judge_api_key,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    args = options.args
    suite_version, tasks = load_suite(args.data)
    selected = select_tasks(tasks, options.categories, args.limit)
    if not selected:
        print("No tasks selected. Check --category/--limit.", file=sys.stderr)
        return 1
    print(f"writing [{args.model}] cap {args.cap} temperature {args.temperature}", flush=True)
    tally, samples = run_generation(args, suite_version, selected, options.skip_existing)
    has_judge = options.judge_model and options.judge_endpoint
    if has_judge and options.judge_model != args.model:
        extra = run_scoring(args, selected, samples, options.judge_model,
                            options.judge_endpoint, options.judge_api_key)
    else:
        extra = not_scored(args.model, options.judge_model, options.judge_endpoint)
        if tally.status() != "pass":
            extra = {"reason": extra["reason"]}
        print(f"not scored: {extra['reason']}", file=sys.stderr)
    finish(LEG_ID, args, tally, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
