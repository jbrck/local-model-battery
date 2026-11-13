#!/usr/bin/env python3
"""Run the battery on one model: probe, plan, gate, run every leg, write the manifest.

One command, one run directory. The directory ends up holding:

    PREFLIGHT.json   what the gate checked and what it refused to pass
    run-plan.json    the plan the run was held to
    MANIFEST.json    the standard, the profile, the host and the serve, all fingerprinted
    legs/*.result.json   one block per leg, with the counters the standard compares
    logs/*.log       the leg's own stdout, kept so a weird score can be read back
    run.log          the orchestrator's own output when started with --detach
    run.pid          the detached run's pid, read back by --status
    NOTES.md         the run's account of itself, ending in the comparability stamp

    python3 scripts/run_battery.py --model arete-27b \
        --profile profiles/arete-27b.toml --out runs/arete-27b

Legs run one at a time, and one request is in flight at a time inside a leg. The box serves one
model, so anything overlapping fights for the same GPU and the tokens-per-second figure stops
meaning anything. That is a rule, not an implementation detail: no leg fan-out, no threads, no
asyncio. It is held by construction -- there is no pool to schedule into. Each leg gets the cap,
window, thinking setting and case timeout the plan decided, and a wall-clock limit off the
pilot's measured speed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.lib.config import load_toml  # noqa: E402
from scripts.lib import detach  # noqa: E402
from scripts.lib.harness import RESULT_SUFFIX  # noqa: E402
from scripts.lib.judge import resolve_judge_defaults  # noqa: E402
from scripts.lib.legs import load_legs  # noqa: E402
from scripts.lib.plan_run import RunPlan  # noqa: E402
from scripts.plan_run import probe_from_args  # noqa: E402
from scripts.lib.plan_from_dict import plan_from_dict  # noqa: E402
from scripts import manifest  # noqa: E402
from scripts import preflight  # noqa: E402

STANDARD_FILE = "standard.toml"
LEGS_DIR = "legs"
LOG_DIR = "logs"
PENDING_STATUS = "not run"


def start_timer() -> float:
    """Wall-clock start, in seconds."""
    return time.time()


def elapsed(start: float) -> str:
    """Elapsed wall time as a short human string."""
    seconds = int(time.time() - start)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


FAILED_STATUSES = ("fail", "timeout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="served model name (defaults to profile id)")
    parser.add_argument("--profile", default="",
                        help="model profile TOML; only for what the probes cannot read")
    parser.add_argument("--model-file", default="",
                        help="the GGUF or model directory to read the shape from")
    parser.add_argument("--model-dir", default="",
                        help="a HF model directory when the weights are safetensors")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--out", default="",
                        help="run directory (created if missing; not needed with --dry-run)")
    parser.add_argument("--legs", default="", help="comma-separated subset of leg ids")
    parser.add_argument("--judge-model", default="", help="scoring model for the writing leg")
    parser.add_argument("--judge-endpoint", default="", help="judge base URL")
    parser.add_argument("--plan", default="", help="reuse a run-plan.json instead of probing")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="run without the gate (the manifest records that it was skipped)")
    parser.add_argument("--no-probe", action="store_true", help="plan without touching the box")
    parser.add_argument("--resume", action="store_true",
                        help="continue a run directory: legs that passed are left alone")
    parser.add_argument("--no-budget-kill", action="store_true",
                        help="do not stop a leg that overruns its planned budget")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when a leg is missing or was never planned")
    parser.add_argument("--detach", action="store_true",
                        help="start the run in its own session and exit; "
                             "output appends to run.log, the pid goes to run.pid")
    parser.add_argument("--status", action="store_true",
                        help="read a run directory: whether the pid is alive, "
                             "and each leg's state")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def load_plan(args: argparse.Namespace) -> tuple[RunPlan, Any, Any, Any]:
    """Either reuse the plan in the run directory, or probe the box and build one.

    A reused plan is taken as written: the legs and the settings do not get re-planned. The
    host is still probed, because the manifest has to record the box the run actually happened on.
    """
    if not args.plan:
        return probe_from_args(args)
    payload = json.loads(Path(args.plan).read_text())
    plan = plan_from_dict(payload, load_legs(load_toml(STANDARD_FILE)))
    print(f"reusing the plan in {args.plan}: leg settings are taken as written")
    from scripts.lib.env_probe import probe_host
    return plan, probe_host(), None, None


def _entry_path(plan_root: str, planned) -> Path:
    return Path(plan_root) / planned.leg.entry


def missing_reason(planned, root: str) -> str:
    """Why this leg cannot run here, or an empty string when it can."""
    if not planned.leg.implemented:
        return f"the entry script is not in this repo yet ({planned.leg.entry})"
    entry = _entry_path(root, planned)
    if not entry.is_file():
        return f"{planned.leg.entry} is missing"
    if planned.leg.data and not (Path(root) / planned.leg.data).is_file():
        return f"{planned.leg.data} is missing"
    return ""


def _skipped_blocks(plan: RunPlan, root: str) -> dict:
    """A block for every leg that will not run, so the run directory explains itself."""
    blocks = {}
    for skip in plan.skips:
        blocks[skip.leg] = {"leg": skip.leg, "status": PENDING_STATUS, "reason": skip.reason}
    for planned in plan.legs:
        reason = missing_reason(planned, root)
        if reason:
            blocks[planned.leg.id] = {"leg": planned.leg.id, "status": PENDING_STATUS,
                                      "reason": reason}
    return blocks


def leg_command(planned, args: argparse.Namespace, out_dir: Path, root: str,
                model: str) -> list[str]:
    """The exact command line one leg runs under: every knob the plan decided shows up here."""
    entry = _entry_path(root, planned)
    command = [planned.interpreter or sys.executable, str(entry),
               "--model", model,
               "--endpoint", args.endpoint, "--api-key", args.api_key,
               "--output-dir", str(out_dir / LEGS_DIR),
               "--cap", str(planned.cap), "--ctx", str(planned.ctx),
               "--thinking", planned.thinking,
               "--timeout", str(planned.case_timeout_s)]
    if planned.leg.data:
        command += ["--data", str(Path(root) / planned.leg.data)]
    if "judge_endpoint" in planned.leg.prereqs and args.judge_model:
        command += ["--judge-model", args.judge_model, "--judge-endpoint", args.judge_endpoint]
    command += list(planned.extra_args)
    return command


def read_block(out_dir: Path, leg_id: str) -> dict | None:
    """The leg's own result block, which is the only thing the orchestrator trusts."""
    path = out_dir / LEGS_DIR / f"{leg_id}{RESULT_SUFFIX}"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _run_command(command: list[str], log, limit: int | None) -> tuple[int, bool]:
    """Run one leg under its budget; returns (exit code, stopped at the budget)."""
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                check=False, timeout=limit)
        return result.returncode, False
    except subprocess.TimeoutExpired:
        return -1, True


def _final_block(planned, out_dir: Path, log_path: Path, code: int, killed: bool) -> dict:
    """The leg's own block if it wrote one, otherwise an honest failure block."""
    block = read_block(out_dir, planned.leg.id)
    if block is None:
        block = {"leg": planned.leg.id, "status": "timeout" if killed else "fail",
                 "reason": (f"stopped at the planned budget of {planned.budget_s / 3600:.1f}h"
                            if killed else f"exit {code}, no result block in {log_path}")}
    elif killed:
        block["status"] = "timeout"
        block["reason"] = (f"stopped at the planned budget of {planned.budget_s / 3600:.1f}h; "
                           f"the numbers here are the ones it had written")
    block["exit_code"] = code
    block["log"] = str(log_path)
    block["_live"] = True
    return block


def run_leg(planned, args: argparse.Namespace, out_dir: Path, root: str, model: str) -> dict:
    """Run one leg and report what happened, including a leg that had to be stopped."""
    log_dir = out_dir / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{planned.leg.id}.log"
    command = leg_command(planned, args, out_dir, root, model)
    limit = None if args.no_budget_kill else planned.budget_s or None
    print(f"\n{'=' * 60}\n{planned.leg.id}: cap {planned.cap}, ctx {planned.ctx}, "
          f"thinking {planned.thinking}, budget {planned.budget_s / 3600:.1f}h\n{'=' * 60}",
          flush=True)
    with open(log_path, "w") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        code, killed = _run_command(command, log, limit)
    return _final_block(planned, out_dir, log_path, code, killed)


def run_all(plan: RunPlan, args: argparse.Namespace, out_dir: Path, root: str) -> dict:
    """Run every planned leg in catalog order, keeping whatever each one wrote."""
    results = _skipped_blocks(plan, root)
    for planned in plan.legs:
        leg_id = planned.leg.id
        if missing_reason(planned, root):
            continue
        if args.resume and done(out_dir, leg_id):
            print(f"\n{leg_id}: already passed in this directory, left alone")
            results[leg_id] = read_block(out_dir, leg_id)
            continue
        results[leg_id] = run_leg(planned, args, out_dir, root, plan.model)
        # The gate leg exists to stop the run before it burns hours.
        if planned.leg.gate and results[leg_id].get("status") in FAILED_STATUSES:
            print(f"\n{leg_id} is the gate and it failed: stopping here", file=sys.stderr)
            break
    return results


def done(out_dir: Path, leg_id: str) -> bool:
    block = read_block(out_dir, leg_id)
    return bool(block) and block.get("status") == "pass"


def _summary_row(leg_id: str, block: dict) -> str:
    if block.get("status") == PENDING_STATUS:
        return f"  {leg_id:16s} not run          {block.get('reason', '')}"
    return (f"  {leg_id:16s} {block.get('status', '?'):8s} "
            f"{block.get('correct', 0)}/{block.get('scored', 0)}"
            f"  {block.get('score_pct', 0)}%  {block.get('tps', 0)} tok/s"
            f"  cap hits {block.get('cap_hits', 0)}  runaway {block.get('runaway', 0)}"
            f"  timeouts {block.get('timeout_hits', 0)}")


def summarise(plan: RunPlan, results: dict) -> str:
    lines = [f"\n{'=' * 60}"]
    lines.append(f"{plan.model} on the portable battery {plan.standard_version}")
    lines.append(f"Stamp: {plan.stamp_line}")
    lines.append("=" * 60)
    for planned in plan.legs:
        lines.append(_summary_row(planned.leg.id, results.get(planned.leg.id, {})))
    return "\n".join(lines)


def _failed_legs(results: dict) -> list[str]:
    """Legs that failed or were stopped at their budget."""
    return [leg_id for leg_id, block in results.items()
            if block.get("status") in FAILED_STATUSES]


def _never_ran(plan: RunPlan, results: dict) -> list[str]:
    """Planned legs whose block still says they did not run."""
    return [planned.leg.id for planned in plan.legs
            if results.get(planned.leg.id, {}).get("status") == PENDING_STATUS]


def verdict(plan: RunPlan, results: dict, strict: bool) -> int:
    """Zero when the run is usable, one when something planned did not happen."""
    broken = _failed_legs(results)
    absent = _never_ran(plan, results)
    if broken:
        print("\nfailed legs: " + ", ".join(sorted(broken)), file=sys.stderr)
    if absent:
        print("\nlegs that did not run: " + ", ".join(sorted(absent)), file=sys.stderr)
    return 1 if broken or (strict and absent) else 0


def gate(plan: RunPlan, args: argparse.Namespace, probes) -> list:
    """Run the preflight gate and write it into the run directory, or record the bypass."""
    out_dir = Path(args.out)
    if args.skip_preflight:
        print("\npreflight skipped on request: these numbers carry no gate", file=sys.stderr)
        return []
    host, server, pilot = probes
    if host is None:
        print("\npreflight needs live probes: drop --plan or drop --skip-preflight",
              file=sys.stderr)
        return []
    checks = preflight.run_preflight(plan, host, server, pilot, out_dir, args.resume)
    preflight.write_preflight(out_dir, plan, checks)
    print("\n" + preflight.summarise(checks))
    return checks


def resolve_judge(args: argparse.Namespace) -> None:
    """Fill the judge from the standard pin, so every run scores with the same model."""
    args.judge_model, args.judge_endpoint = resolve_judge_defaults(
        args.judge_model, args.judge_endpoint, args.endpoint, load_toml(STANDARD_FILE))


def detach_run(args: argparse.Namespace) -> int:
    """Start the orchestrator again in its own session, then tell the operator where to look."""
    if not args.out:
        print("\n--out is required for --detach", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser()
    pid, log_path, child = detach.launch(sys.argv, out_dir, os.path.abspath(sys.argv[0]))
    print(f"run directory: {out_dir}")
    print(f"pid: {pid}")
    print(f"log: tail -f {log_path}")
    print(f"resume: {' '.join(detach.resume_argv(child))}")
    return 0


def status_report(args: argparse.Namespace) -> int:
    """What the run directory says, plainly: the pid's liveness and each leg's state."""
    if not args.out:
        print("\n--out is required for --status", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser()
    state = detach.run_state(out_dir)
    print(f"run directory: {out_dir}")
    if state["pid"] is not None:
        print(f"pid: {state['pid']}")
    print(state["detail"])
    legs = detach.leg_states(out_dir)
    if not legs:
        print(f"no {detach.PLAN_FILE} in {out_dir}: no planned legs to report")
    for leg_id, block in legs.items():
        line = f"  {leg_id:16s} {block['status']}"
        if block["reason"]:
            line += f"  ({block['reason']})"
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_judge(args)
    if args.status:
        return status_report(args)
    if args.detach:
        return detach_run(args)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timer = start_timer()
    plan, host, server, pilot = load_plan(args)
    print(plan.render())
    if args.dry_run:
        print("\ndry run: nothing was started")
        return 0
    if not args.out:
        print("\n--out is required for a real run", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = gate(plan, args, (host, server, pilot))
    misses = preflight.blocked(checks)
    if misses:
        print(f"\npreflight blocked: {', '.join(check.name for check in misses)}",
              file=sys.stderr)
        return 1

    manifest.start(out_dir, plan, host, server, resume=args.resume)
    (out_dir / "run-plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    results = run_all(plan, args, out_dir, root)
    notes_path = manifest.close(out_dir, plan, results)
    print(summarise(plan, results))
    print(f"\nnotes: {notes_path}")
    print(f"elapsed: {elapsed(timer)}")
    if args.json:
        print(json.dumps({"stamp": plan.stamp_line, "results": results}, indent=2))
    return verdict(plan, results, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
