"""The ifeval and toolbench legs: drive tool-eval-bench, read what it wrote.

The CLI talks to the endpoint itself, so this adapter's job is command
construction plus output parsing. The tool-call run writes a versioned JSON
envelope to --json-file. The IFEval plugin path ignores --json-file and writes
a markdown report under <output-dir>/runs/YYYY/MM (a known harness quirk), so
that leg reads the newest report instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.harness import LegArgs, Tally, add_leg_args, finish

LEGS = ("ifeval", "toolbench")
DEFAULT_CLI = "tool-eval-bench"
INFRA_FAILURES = frozenset({"connection_error", "server_error", "model_crash"})
PROMPT_ACCURACY = re.compile(r"\*\*Prompt-level Accuracy:\*\* [\d.]+% \((\d+)/(\d+)\)")
INSTRUCTION_ACCURACY = re.compile(r"\*\*Instruction-level Accuracy:\*\* [\d.]+% \((\d+)/(\d+)\)")
DURATION = re.compile(r"\*\*Duration:\*\* ([\d.]+)s")
TOKENS = re.compile(r"\*\*Tokens:\*\* ([\d,]+)")


def parse_args(argv: list[str] | None = None) -> tuple[str, LegArgs, str]:
    """The shared leg flags plus the leg selector and the CLI override."""
    parser = argparse.ArgumentParser(description="ifeval and toolbench adapter leg")
    parser.add_argument("--leg", choices=LEGS, required=True)
    parser.add_argument("--tool-eval-bench", default=DEFAULT_CLI, dest="tool_eval_bench",
                        help="path to the tool-eval-bench executable")
    add_leg_args(parser)
    ns = parser.parse_args(argv)
    args = LegArgs(model=ns.model, endpoint=ns.endpoint, api_key=ns.api_key, data=ns.data,
                   output_dir=ns.output_dir, cap=ns.cap, ctx=ns.ctx, thinking=ns.thinking,
                   seed=ns.seed, temperature=ns.temperature, timeout_s=ns.timeout_s,
                   limit=ns.limit, tag=ns.tag)
    return ns.leg, args, ns.tool_eval_bench


def build_command(leg: str, args: LegArgs, cli: str, json_path: str) -> list[str]:
    """The exact CLI invocation for one leg."""
    command = [cli, "run"] if leg == "toolbench" else [cli, "--ifeval-only"]
    command += ["--model", args.model, "--base-url", args.endpoint,
                "--api-key", args.api_key, "--temperature", str(args.temperature),
                "--timeout", str(args.timeout_s), "--output-dir", args.output_dir,
                "--json-file", json_path, "--label", args.tag or leg]
    if args.thinking == "off":
        command.append("--no-think")
    command += ["--backend-kwargs", json.dumps({"max_tokens": args.cap})]
    if args.seed:
        command += ["--seed", str(args.seed)]
    if leg == "ifeval" and args.limit:
        command += ["--ifeval-limit", str(args.limit)]
    return command


def run_cli(command: list[str], cwd: str) -> tuple[int, str]:
    """Run the CLI to completion and return (exit code, stderr tail)."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    try:
        done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        return done.returncode, done.stderr[-2000:]
    except FileNotFoundError as exc:
        return 127, str(exc)


def parse_toolbench(path: str) -> tuple[Tally, dict]:
    """The scenario counts from the run envelope."""
    envelope = json.loads(Path(path).read_text())
    if "scores" not in envelope:
        raise ValueError(f"the run did not complete: {envelope.get('error', envelope)}")
    tally = Tally()
    for result in envelope["scores"].get("scenario_results", []):
        kind = result.get("failure_kind")
        if kind == "timeout":
            tally.timeout_hits += 1
        elif kind in INFRA_FAILURES or kind == "evaluator_error":
            tally.failed += 1
        else:
            tally.scored += 1
            tally.correct += int(result.get("status") == "pass")
        tally.items += 1
    scores = envelope["scores"]
    tally.tokens = int(scores.get("total_tokens") or 0)
    extra = {"final_score": scores.get("final_score"), "run_id": envelope.get("run_id"),
             "rating": scores.get("rating")}
    return tally, extra


def find_report(output_dir: str, started: float) -> str | None:
    """The newest report this run wrote, under runs/ or the output dir itself.

    Tool-eval-bench nests reports under <output-dir>/runs/YYYY/MM, but it resolves a
    relative --output-dir against its own anchor, which historically doubled the tree and
    hid the report from a fixed-depth glob. An absolute output dir fixes the layout; this
    glob covers both candidate roots anyway, so a completed run never reads as a failure
    because the report sat somewhere the parser did not expect.
    """
    root = Path(output_dir)
    report_files = [path for path in root.glob("**/*.md")
                    if path.stat().st_mtime >= started]
    if not report_files:
        return None
    return str(max(report_files, key=lambda path: path.stat().st_mtime))


def parse_report(text: str) -> tuple[Tally, dict]:
    """The counters from an IFEval markdown report."""
    prompt = PROMPT_ACCURACY.search(text)
    if prompt is None:
        raise ValueError("the report has no prompt accuracy line")
    tally = Tally(items=int(prompt.group(2)), scored=int(prompt.group(2)),
                  correct=int(prompt.group(1)))
    extra: dict = {}
    instruction = INSTRUCTION_ACCURACY.search(text)
    if instruction:
        extra["instruction_accuracy"] = f"{instruction.group(1)}/{instruction.group(2)}"
    duration = DURATION.search(text)
    if duration:
        tally.seconds = float(duration.group(1))
    tokens = TOKENS.search(text)
    if tokens:
        tally.tokens = int(tokens.group(1).replace(",", ""))
    return tally, extra


def run_leg(leg: str, args: LegArgs, cli: str) -> dict:
    """Run the CLI for one leg and write the result block.

    The CLI resolves relative --output-dir and --json-file against ITS working directory,
    which is this script's own legs directory, not the repo root. Paths stay relative for
    one layer and the report lands deeper than the parser looks: that is exactly how a run
    completes and still reads as failed. Everything the CLI touches must be absolute.
    """
    output_dir = os.path.abspath(args.output_dir)
    json_path = os.path.join(output_dir, f"tool-eval-{leg}.json")
    os.makedirs(output_dir, exist_ok=True)
    command = build_command(leg, args, cli, json_path)
    command[command.index("--output-dir") + 1] = output_dir
    print(" ".join(command), flush=True)
    started = time.time()
    code, tail = run_cli(command, output_dir)
    try:
        if leg == "toolbench":
            tally, extra = parse_toolbench(json_path)
        else:
            report = find_report(args.output_dir, started)
            if report is None:
                raise ValueError(f"no report under {args.output_dir}/runs (exit {code})")
            tally, extra = parse_report(Path(report).read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        tally = Tally(items=1, failed=1, errors=[str(exc)])
        if tail:
            tally.errors.append(tail.strip().splitlines()[-1])
        return finish(leg, args, tally, extra={"command": " ".join(command)})
    if code != 0:
        tally.notes.append(f"tool-eval-bench exited {code}")
    if tail:
        tally.notes.append(tail.strip().splitlines()[-1])
    return finish(leg, args, tally, extra)


def main(argv: list[str] | None = None) -> None:
    leg, args, cli = parse_args(argv)
    print(f"{leg} [{args.model}] cap {args.cap} thinking {args.thinking}", flush=True)
    run_leg(leg, args, cli)


if __name__ == "__main__":
    main()
