#!/usr/bin/env python3
"""HumanEval+: code generation benchmark, self-contained test runner.

Reads problems from data/humaneval_plus.json, asks the model to complete each
function, then runs the bundled test suite for that problem in a subprocess.
No external evaluation harness required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.code_exec import run_python_tests  # noqa: E402
from scripts.lib.model import ModelConfig, chat_request  # noqa: E402
from scripts.lib.reporting import elapsed, report_stdout, save_result, start_timer  # noqa: E402

SYSTEM = """You are an expert Python developer. Write a solution for the given problem.
Output ONLY the code, no explanation, no markdown formatting, no imports.
The solution must be a standalone Python function."""

CODE_FENCE = "```"


def strip_fences(text: str) -> str:
    """Remove a markdown code fence, preserving the code's own indentation."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().startswith(CODE_FENCE):
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().startswith(CODE_FENCE):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def run_humaneval(cfg: ModelConfig, data_path: str, limit: int | None = None) -> dict:
    """Run HumanEval+ over the data file. Returns pass@1 plus per-task detail."""
    with open(data_path) as handle:
        problems = json.load(handle)
    if limit:
        problems = problems[:limit]

    total = len(problems)
    passed = 0
    results = []

    for index, item in enumerate(problems, start=1):
        started = time.time()
        response = chat_request(cfg, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["prompt"]},
        ], max_tokens=1024, temperature=0.0)
        elapsed_s = time.time() - started

        if response is None:
            print(f"  [{index}/{total}] SKIP (no response)")
            results.append({"task_id": item["task_id"], "passed": False,
                            "error": "no response", "time_s": round(elapsed_s, 1)})
            continue

        imports = item.get("imports") or ""
        completion = strip_fences(response)
        source = "\n".join(part for part in
                           (imports, item["prompt"], completion) if part.strip())
        ok, detail = run_python_tests(source, item["entry_point"], item["test"])
        passed += int(ok)
        results.append({"task_id": item["task_id"], "passed": ok,
                        "error": None if ok else detail, "time_s": round(elapsed_s, 1)})
        print(f"  [{index}/{total}] {'PASS' if ok else 'FAIL'} ({elapsed_s:.1f}s)"
              + ("" if ok else f" {detail}"))
        sys.stdout.flush()

    pass_at_1 = passed / total if total else 0.0
    return {"pass_at_1": pass_at_1, "pass_at_1_pct": round(pass_at_1 * 100, 1),
            "n_passed": passed, "n_total": total, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="HumanEval+ benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--data", default="data/humaneval_plus.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = ModelConfig(name=args.model, endpoint=args.endpoint, api_key=args.api_key)
    timer = start_timer()
    print(f"HumanEval+ [{args.model}]", flush=True)
    results = run_humaneval(cfg, args.data, args.limit)
    results["_elapsed"] = int(time.time() - timer)
    path = save_result("humaneval_plus", args.model, results, args.output_dir)
    report_stdout(results, f"HumanEval+ [{args.model}] done")
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
