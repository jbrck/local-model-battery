#!/usr/bin/env python3
"""HumanEval+: code generation benchmark with evalplus test harness.

Reads problems from data/humaneval_plus.json, generates solutions, and
runs official evalplus test suite against each one.
"""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.lib.model import ModelConfig, chat_request
from scripts.lib.reporting import start_timer, elapsed, save_result, report_stdout

SYSTEM = """You are an expert Python developer. Write a solution for the given problem.
Output ONLY the code, no explanation, no markdown formatting, no imports.
The solution must be a standalone Python function."""


def run_humaneval(cfg: ModelConfig, data_path: str, limit: int | None = None) -> dict:
    with open(data_path) as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]

    from evalplus.data import get_human_eval_plus
    from evalplus.evaluate import evaluate  # type: ignore

    total = len(problems)
    passed = 0
    results = []

    for i, item in enumerate(problems):
        task_id = item["task_id"]
        prompt = item["prompt"]
        entry_point = item["entry_point"]
        test = item["test"]
        imports = item.get("imports", "")

        start_t = time.time()
        response = chat_request(cfg, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ], max_tokens=1024, temperature=0.0)
        elapsed_t = time.time() - start_t

        if response is None:
            print(f"  [{i+1}/{total}] SKIP (no response)")
            continue

        # Build the solution module
        solution = f"{imports}\n{response}" if imports else response
        # evalplus evaluate runs the test suite with a sandbox timeout
        try:
            result = evaluate(solution, entry_point, test)
            p = result.get("passed", False)
        except Exception:
            p = False
        if p:
            passed += 1
        results.append({"task_id": task_id, "passed": p, "time_s": round(elapsed_t, 1)})
        print(f"  [{i+1}/{total}] {'✓' if p else '✗'} ({elapsed_t:.1f}s)", flush=True)

    pass_at_1 = passed / total if total > 0 else 0.0
    return {"pass_at_1": pass_at_1, "pass_at_1_pct": round(pass_at_1 * 100, 1),
            "n_passed": passed, "n_total": total, "results": results}


def main():
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
    results["elapsed"] = elapsed(timer)
    path = save_result("humaneval_plus", args.model, results, args.output_dir)
    report_stdout(results, f"HumanEval+ [{args.model}] done")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()