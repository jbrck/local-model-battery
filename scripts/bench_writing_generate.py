"""Phase B: generate answers to a writing suite. Thin pipe, no harness.

Reads every task in the suite, sends exactly one user message, saves the raw
answer plus provenance. Never cleans, scores, or retries. One model per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.model import ModelConfig, chat_request, verify_endpoint  # noqa: E402
from scripts.lib.reporting import elapsed, start_timer  # noqa: E402

DEFAULT_SUITE = "data/writing_suite_v1.json"
DEFAULT_ANSWERS_DIR = "answers"
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 16384

# Tasks whose prompt refers to a source document must carry it in the same
# message; the prompt alone is not self-contained for these.
SOURCE_HEADER = "\n\nSOURCE TEXT:\n"


def build_user_message(task: dict) -> str:
    """Compose the single user message for one task, verbatim task text first."""
    prompt = task["prompt"]
    source = (task.get("source_text") or "").strip()
    if source and source not in prompt:
        return f"{prompt}{SOURCE_HEADER}{source}"
    return prompt


def safe_model_dir(model: str) -> str:
    """Filesystem-safe directory name for a model alias."""
    return model.replace("/", "--").replace(":", "--").replace(" ", "-")


def answer_path(answers_dir: str, model: str, task_id: str) -> str:
    return os.path.join(answers_dir, safe_model_dir(model), f"{task_id}.json")


def generate_answer(cfg: ModelConfig, task: dict, model: str, suite_version: str,
                    max_tokens: int, temperature: float) -> dict:
    """Send one task and return the answer record, including provenance."""
    message = build_user_message(task)
    request_body = {
        "model": cfg.name,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    started = time.time()
    text = chat_request(
        cfg,
        [{"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=temperature,
        allow_reasoning_fallback=False,
    )
    return {
        "task_id": task["task_id"],
        "category": task["category"],
        "suite_version": suite_version,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 2),
        "request": request_body,
        "response": text,
        "empty": text is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate writing suite answers")
    parser.add_argument("--model", required=True,
                        help="Model alias passed to the endpoint (label in the answer record)")
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--answers-dir", default=DEFAULT_ANSWERS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N tasks")
    parser.add_argument("--category", action="append", default=None,
                        help="Only run these categories (repeatable)")
    parser.add_argument("--max-tokens", type=int, default=GENERATION_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=GENERATION_TEMPERATURE)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Keep answers already on disk (resume a stopped run)")
    args = parser.parse_args()

    with open(args.suite) as handle:
        suite = json.load(handle)
    suite_version = str(suite.get("suite_version", "unknown"))
    tasks = suite["tasks"]
    if args.category:
        wanted = set(args.category)
        tasks = [task for task in tasks if task["category"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No tasks selected. Check --category/--limit.", file=sys.stderr)
        return 1

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key,
                      token_limit=args.max_tokens, temperature=args.temperature)
    if not verify_endpoint(args.endpoint, args.model, args.api_key):
        print(f"ERROR: cannot reach {args.endpoint} or model '{args.model}' not listed",
              file=sys.stderr)
        return 1

    timer = start_timer()
    written = 0
    empty = 0
    for index, task in enumerate(tasks, start=1):
        path = answer_path(args.answers_dir, args.model, task["task_id"])
        if args.skip_existing and os.path.exists(path):
            print(f"  [{index}/{len(tasks)}] {task['task_id']:8s} skip (exists)")
            continue
        record = generate_answer(cfg, task, args.model, suite_version,
                                 args.max_tokens, args.temperature)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(record, handle, indent=2)
        written += 1
        empty += int(record["empty"])
        status = "EMPTY" if record["empty"] else f"{len(record['response'])} chars"
        print(f"  [{index}/{len(tasks)}] {task['task_id']:8s} {status} "
              f"({record['elapsed_s']}s)")
        sys.stdout.flush()

    print(f"\n{written} answers written, {empty} empty "
          f"({elapsed(timer)}). Suite v{suite_version}, model {args.model}.")
    print("Scoring is a separate, blind step: scripts/build_scoring_corpus.py → "
          "scripts/score_writing.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
