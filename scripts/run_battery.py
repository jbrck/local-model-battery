#!/usr/bin/env python3
"""Local Model Battery orchestrator.

Runs the mechanically scored benchmarks in order, then optionally generates
writing-suite answers. Writing answers are scored blind in a separate step.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.model import verify_endpoint  # noqa: E402
from scripts.lib.reporting import elapsed, start_timer  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MECHANICAL_BENCHES = ("math500", "humaneval_plus", "gpqa_diamond", "instr_v2")
HEADLINE_KEYS = ("accuracy_pct", "pass_at_1_pct", "score_pct", "score", "n_passed")


def run_bench(script: str, model: str, endpoint: str, output_dir: str,
              extra: list[str] | None = None) -> dict:
    """Run one benchmark script as a subprocess and report its exit status."""
    command = [sys.executable, script, "--model", model, "--endpoint", endpoint,
               "--output-dir", output_dir]
    if extra:
        command.extend(extra)

    print(f"\n{'=' * 50}\nRunning: {script}\n{'=' * 50}")
    sys.stdout.flush()
    result = subprocess.run(command, check=False)
    return {"script": script, "exit_code": result.returncode,
            "completed": result.returncode == 0}


def find_latest_result(bench_name: str, model: str, output_dir: str) -> dict | None:
    """Find the newest saved result for a benchmark."""
    safe_model = model.replace("/", "--").replace(" ", "-")
    pattern = os.path.join(output_dir, f"{bench_name}-{safe_model}-*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as handle:
        return json.load(handle)


def headline_of(result: dict) -> dict:
    """Pull the headline metrics out of a benchmark result."""
    return {key: result[key] for key in HEADLINE_KEYS if key in result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local model battery")
    parser.add_argument("--model", required=True, help="Model alias on the endpoint")
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip", action="append", default=None,
                        help="Skip a benchmark (repeatable): "
                             + ", ".join(MECHANICAL_BENCHES))
    parser.add_argument("--writing", action="store_true",
                        help="Also generate writing-suite answers (generation only)")
    parser.add_argument("--writing-suite", default="data/writing_suite_v1.json")
    parser.add_argument("--answers-dir", default="answers")
    parser.add_argument("--smoke-only", action="store_true", help="Smoke test only")
    parser.add_argument("--smoke-limit", type=int, default=10,
                        help="Problems per benchmark in smoke mode (default: 10)")
    args = parser.parse_args()

    skipped = set(args.skip or [])
    unknown = skipped - set(MECHANICAL_BENCHES)
    if unknown:
        print(f"ERROR: unknown --skip value(s): {sorted(unknown)}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    timer = start_timer()

    print(f"\n{'=' * 60}")
    print(f"Local Model Battery — {args.model}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Output:   {args.output_dir}")
    print(f"{'=' * 60}")

    if not verify_endpoint(args.endpoint, args.model, args.api_key):
        print(f"ERROR: cannot reach {args.endpoint} or model '{args.model}' not listed",
              file=sys.stderr)
        return 1
    print("Endpoint reachable, model listed")

    benches = [name for name in MECHANICAL_BENCHES if name not in skipped]

    if args.smoke_only:
        print("\n--- SMOKE TESTS ---")
        for name in benches:
            script = os.path.join(SCRIPTS_DIR, f"bench_{name}.py")
            extra = ["--limit", str(args.smoke_limit)] if name != "instr_v2" else None
            result = run_bench(script, args.model, args.endpoint, args.output_dir, extra)
            if not result["completed"]:
                print(f"  WARN: {name} smoke failed (exit {result['exit_code']}), continuing")
        print(f"\nSmoke-only mode done ({elapsed(timer)}).")
        return 0

    print(f"\n\n{'=' * 60}\nFULL BATTERY\n{'=' * 60}")
    outcomes: dict[str, dict] = {}
    for name in benches:
        script = os.path.join(SCRIPTS_DIR, f"bench_{name}.py")
        outcomes[name] = run_bench(script, args.model, args.endpoint, args.output_dir)
        if not outcomes[name]["completed"]:
            print(f"  ERROR: {name} failed (exit {outcomes[name]['exit_code']})")

    if args.writing:
        script = os.path.join(SCRIPTS_DIR, "bench_writing_generate.py")
        outcomes["writing"] = run_bench(
            script, args.model, args.endpoint, args.output_dir,
            ["--suite", args.writing_suite, "--answers-dir", args.answers_dir],
        )
        if not outcomes["writing"]["completed"]:
            print("  ERROR: writing generation failed")

    print(f"\n\n{'=' * 60}\nRESULTS SUMMARY\n{'=' * 60}")
    summary = {"model": args.model, "endpoint": args.endpoint, "results": {}}
    for name in benches:
        result = find_latest_result(name, args.model, args.output_dir)
        if result:
            summary["results"][name] = headline_of(result)
            print(f"  {name:20s}  {summary['results'][name]}")
        else:
            print(f"  {name:20s}  NOT FOUND")

    if args.writing:
        summary["writing"] = {
            "answers_dir": args.answers_dir,
            "scored": False,
            "next": "scripts/build_scoring_corpus.py then scripts/score_writing.py",
        }
        print("  writing             answers generated; scoring is a separate blind step")

    summary_path = os.path.join(
        args.output_dir, f"summary-{args.model}-{int(time.time())}.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Total elapsed: {elapsed(timer)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
