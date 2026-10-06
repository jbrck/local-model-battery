#!/usr/bin/env python3
"""Local Model Battery orchestrator.

Usage:
  python scripts/run_battery.py --model MyModel --endpoint http://localhost:11434/v1

Runs: smoke test → MATH-500 → HumanEval+ → GPQA-Diamond → Instr v2 → Prose ELO
Reports progress and aggregates results.
"""
import argparse, json, os, subprocess, sys, time


def run_bench(script: str, model: str, endpoint: str, output_dir: str, extra: list[str] | None = None) -> dict:
    """Run a benchmark script and return its JSON result path."""
    cmd = [sys.executable, script,
           "--model", model,
           "--endpoint", endpoint,
           "--output-dir", output_dir]
    if extra:
        cmd.extend(extra)

    print(f"\n{'=' * 50}")
    print(f"Running: {script}")
    print(f"{'=' * 50}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=False)
    return {"script": script, "exit_code": result.returncode, "completed": result.returncode == 0}


def find_latest_result(bench_name: str, model: str, output_dir: str) -> dict | None:
    """Find the latest saved result for a benchmark."""
    import glob
    safe = model.replace("/", "--").replace(" ", "-")
    pattern = os.path.join(output_dir, f"{bench_name}-{safe}-*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run the full local model battery")
    parser.add_argument("--model", required=True, help="Model label (for logging)")
    parser.add_argument("--endpoint", default="http://localhost:11434/v1",
                        help="OpenAI-compatible inference endpoint")
    parser.add_argument("--judge-endpoint", default=None,
                        help="Judge model endpoint (defaults to inference endpoint)")
    parser.add_argument("--judge-model", default="hermes-4-70b",
                        help="Judge model for prose ELO")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-prose", action="store_true", help="Skip prose ELO (no judge)")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke test only")
    parser.add_argument("--smoke-limit", type=int, default=10,
                        help="Problems per test for smoke (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    judge_ep = args.judge_endpoint or args.endpoint
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(base_dir, "scripts")

    print(f"\n{'=' * 60}")
    print(f"Local Model Battery — {args.model}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Output:   {args.output_dir}")
    print(f"{'=' * 60}")

    # 1. Verify endpoint is live
    from scripts.lib.model import verify_endpoint
    if not verify_endpoint(args.endpoint, args.model):
        print(f"ERROR: Cannot reach {args.endpoint} or model '{args.model}' not loaded")
        sys.exit(1)
    print("✓ Endpoint reachable, model loaded")

    # 2. Smoke tests (small versions of each)
    print("\n--- SMOKE TESTS ---")
    smoke_results = {}
    for bench in [("math500", ["--limit", str(args.smoke_limit)]),
                  ("gpqa_diamond", ["--limit", str(args.smoke_limit)]),
                  ("instr_v2", None)]:
        name, extra = bench
        script = os.path.join(scripts_dir, f"bench_{name}.py")
        r = run_bench(script, args.model, args.endpoint, args.output_dir, extra)
        smoke_results[name] = r
        if not r["completed"]:
            print(f"  WARN: {name} smoke failed, continuing...")

    if args.smoke_only:
        print("\nSmoke-only mode. Results saved.")
        return

    # 3. Full battery
    print(f"\n\n{'=' * 60}")
    print("FULL BATTERY")
    print(f"{'=' * 60}")
    full_results = {}

    benches = [
        ("math500", None),
        ("humaneval_plus", None),
        ("gpqa_diamond", None),
        ("instr_v2", None),
    ]
    if not args.skip_prose:
        benches.append(("prose_elo", ["--judge-endpoint", judge_ep, "--judge-model", args.judge_model]))

    for name, extra in benches:
        script = os.path.join(scripts_dir, f"bench_{name}.py")
        r = run_bench(script, args.model, args.endpoint, args.output_dir, extra)
        full_results[name] = r
        if not r["completed"]:
            print(f"  ERROR: {name} failed. Continuing with remaining tests.")

    # 4. Summary
    print(f"\n\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    summary = {"model": args.model, "endpoint": args.endpoint, "results": {}}
    for name in [b[0] for b in benches]:
        result = find_latest_result(name, args.model, args.output_dir)
        if result:
            # Extract headline metrics
            headline = {}
            for key in ["accuracy_pct", "pass_at_1_pct", "score_pct", "elo", "score", "n_passed"]:
                if key in result:
                    headline[key] = result[key]
            summary["results"][name] = headline
            print(f"  {name:20s}  {headline}")
        else:
            print(f"  {name:20s}  NOT FOUND")

    summary_path = os.path.join(args.output_dir, f"summary-{args.model}-{int(time.time())}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()