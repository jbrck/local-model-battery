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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

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
STALL_STATUS = "stalled"

# How often the orchestrator looks at a running leg, and how long a leg may go without
# checkpointing a single case before it is taken down. A leg that has answered nothing after
# fifteen minutes is not slow, it is broken, and the whole point of a battery is that it
# finds that out before it has burned the night.
POLL_SECONDS = 5.0
VERIFY_AFTER_S = 900.0
STOP_GRACE_S = 10.0
# Legs exempt from the progress check. The check stops a leg that has written nothing
# inside the window, and for most legs that is right: a half hour of silence is a broken
# leg. Not for these. LCB generates one answer at a time at up to 114k context and can
# sit on a single case for half an hour with nothing written. The tool-eval
# harnesses (ifeval, ifbench) only checkpoint at leg end, and toolbench follows the same
# harness. hermesagent20 runs a small number of long agent tasks that print little in
# between. For all of them silence is health; their budgets are the runaway bound.
VERIFY_EXEMPT = frozenset({"lcb", "ifeval", "ifbench", "toolbench", "hermesagent20"})


class ProgressWatch(NamedTuple):
    """A leg's progress check: what it has written, and when to look.

    Progress means either a new case file or a growing leg log. The second half matters
    for legs that log per task but only checkpoint at the end: a quiet log is a stuck
    leg, but a growing one is alive even when no case has landed yet.
    """

    count: Callable[[], int]
    after_s: float
    log: Path
    start_size: int

    def has_progress(self, start_count: int) -> bool:
        if self.count() > start_count:
            return True
        try:
            return self.log.stat().st_size > self.start_size
        except OSError:
            return False


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


FAILED_STATUSES = ("fail", "timeout", "stalled")

# The values a leg's --thinking flag accepts. The standard also records n/a, for a leg
# whose generation takes no thinking switch at all (the writing leg), so that value must
# never reach a command line: argparse in those legs would reject the whole invocation and
# the leg would be recorded as missing rather than run.
THINKING_SWITCHES = ("on", "off")


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
    parser.add_argument("--verify-after", type=float, default=VERIFY_AFTER_S,
                        help="stop a leg that has checkpointed no case within this many "
                             "seconds (default 900; 0 disables the check)")
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
               "--timeout", str(planned.case_timeout_s)]
    if planned.thinking in THINKING_SWITCHES:
        command += ["--thinking", planned.thinking]
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


def _run_command(command: list[str], log, limit: int | None,
                 watch: ProgressWatch | None = None) -> tuple[int, bool, str]:
    """Run one leg under its budget and its progress check.

    Returns (exit code, stopped at the budget, stall reason). The leg runs in its own
    process group so stopping it stops whatever it started: a killed orchestrator must not
    leave a leg behind holding the GPU.
    """
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    _register_child(process)
    started = time.monotonic()
    cases_at_start = watch.count() if watch else 0
    try:
        while True:
            code = process.poll()
            if code is not None:
                return code, False, ""
            waited = time.monotonic() - started
            disarm, stall = _watch_verdict(watch, cases_at_start, waited)
            if disarm:
                watch = None
            if stall:
                _stop(process)
                return -2, False, stall
            if limit is not None and waited >= limit:
                _stop(process)
                return -1, True, ""
            time.sleep(POLL_SECONDS)
    finally:
        _register_child(None)


def _watch_verdict(watch: ProgressWatch | None, cases_at_start: int,
                   waited: float) -> tuple[bool, str]:
    """What the progress check says after the window closed: either progress (disarm the
    watch), a stall reason to enforce, or nothing while the window is still open."""
    if watch is None or waited < watch.after_s:
        return False, ""
    if watch.has_progress(cases_at_start):
        return True, ""
    return False, f"no case written in the first {watch.after_s / 60:.0f} min"


def _stop(process) -> None:
    """Stop the leg and everything it started, so nothing is left holding the GPU."""
    if process.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


_current_child: subprocess.Popen | None = None


def _register_child(process: subprocess.Popen | None) -> None:
    """Remember the running leg, so a signal to the orchestrator can take it down too."""
    global _current_child
    _current_child = process


def _stop_on_signal(signum, _frame) -> None:
    """An interrupted orchestrator leaves no leg running: it stops the one it started."""
    if _current_child is not None:
        _stop(_current_child)
    raise SystemExit(128 + signum)


def case_count(out_dir: Path, leg_id: str) -> int:
    """How many cases the leg has checkpointed so far for this run."""
    cases = out_dir / LEGS_DIR / "cases" / leg_id
    if not cases.is_dir():
        return 0
    return sum(1 for _ in cases.glob("*.json"))


def last_error(log_path: Path) -> str:
    """The exception line a leg died on, if it died on one. A leg that fails silently is
    the thing this whole runner exists to avoid."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return ""
    if "Traceback (most recent call last)" not in text:
        return ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def _record(out_dir: Path, block: dict, wrote_own: bool) -> None:
    """Put a failed leg on disk. A leg that dies must not read as 'not run' next morning.

    The leg's own block wins when it wrote one and finished: it holds the counters. This
    writes only when the leg wrote nothing, or when the orchestrator stopped it and the
    status it recorded is the one an operator needs to see.
    """
    if wrote_own and block.get("status") not in (STALL_STATUS, "timeout"):
        return
    path = out_dir / LEGS_DIR / f"{block['leg']}{RESULT_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(block, indent=2) + "\n")


def _final_block(planned, out_dir: Path, log_path: Path, code: int, killed: bool,
                 stall: str = "") -> dict:
    """The leg's own block if it wrote one, otherwise an honest failure block."""
    block = read_block(out_dir, planned.leg.id)
    wrote_own = block is not None
    if block is None:
        status = "timeout" if killed else (STALL_STATUS if stall else "fail")
        reason = (f"stopped at the planned budget of {planned.budget_s / 3600:.1f}h" if killed
                  else stall or f"exit {code}, no result block in {log_path}")
        block = {"leg": planned.leg.id, "status": status, "reason": reason}
    elif killed:
        block["status"] = "timeout"
        block["reason"] = (f"stopped at the planned budget of {planned.budget_s / 3600:.1f}h; "
                           f"the numbers here are the ones it had written")
    block["exit_code"] = code
    block["log"] = str(log_path)
    error = last_error(log_path)
    if error:
        block["last_error"] = error
    block["_live"] = True
    _record(out_dir, block, wrote_own)
    return block


def _progress_watch(planned, args: argparse.Namespace, out_dir: Path,
                    log_path: Path) -> ProgressWatch | None:
    """The progress check for a leg, if it has one. Exempt legs (VERIFY_EXEMPT, e.g. LCB
    and the tool-eval harnesses) only checkpoint on a finished generation or run long
    silent generations: for them silence is health and the check would be a false alarm."""
    after_s = float(getattr(args, "verify_after", VERIFY_AFTER_S) or 0)
    if not after_s or planned.leg.id in VERIFY_EXEMPT:
        return None
    start_size = log_path.stat().st_size if log_path.exists() else 0
    return ProgressWatch(lambda: case_count(out_dir, planned.leg.id), after_s,
                         log_path, start_size)


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
        watch = _progress_watch(planned, args, out_dir, log_path)
        code, killed, stall = _run_command(command, log, limit, watch)
    if stall:
        print(f"\n{planned.leg.id}: stopped — {stall}", file=sys.stderr, flush=True)
    return _final_block(planned, out_dir, log_path, code, killed, stall)


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
        block = results[leg_id]
        _report_failure(leg_id, block)
        # The gate leg exists to stop the run before it burns hours.
        if planned.leg.gate and block.get("status") in FAILED_STATUSES:
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


def _report_failure(leg_id: str, block: dict) -> None:
    """Shout about a dirty leg where the run's own output can be seen."""
    if block.get("status") not in FAILED_STATUSES:
        return
    print(f"\n{leg_id}: {block.get('status')} — {block.get('reason', '')}",
          file=sys.stderr, flush=True)
    if block.get("last_error"):
        print(f"{leg_id}: {block['last_error']}", file=sys.stderr, flush=True)


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
        return 0
    for line in _status_lines(legs):
        print(line)
    return _failed_exit(legs)


def _failed_exit(legs: dict) -> int:
    """A run with a dirty leg is not a finished run: status says so and exits 1."""
    failed = [leg_id for leg_id, block in legs.items()
              if block["status"] in FAILED_STATUSES]
    if not failed:
        return 0
    print(f"\n{len(failed)} leg(s) did not finish clean: {', '.join(sorted(failed))}")
    return 1


def _status_lines(legs: dict) -> list[str]:
    """One readable line per planned leg, with whatever the failure left behind."""
    lines = []
    for leg_id, block in legs.items():
        line = f"  {leg_id:16s} {block['status']}"
        if block["reason"]:
            line += f"  ({block['reason']})"
        lines.append(line)
        if block.get("last_error"):
            lines.append(f"      error: {block['last_error']}")
    return lines


def _early_exit(args: argparse.Namespace) -> int | None:
    """The read-only commands that answer and leave the run directory alone."""
    if args.status:
        return status_report(args)
    if args.detach:
        return detach_run(args)
    return None


def _run_for_real(plan: RunPlan, args: argparse.Namespace, probes, timer: float,
                  root: str) -> int:
    """Everything after the plan is rendered: gate, manifest, every leg, the verdict."""
    if not args.out:
        print("\n--out is required for a real run", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop_on_signal)
    host, server, pilot = probes
    checks = gate(plan, args, probes)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_judge(args)
    early = _early_exit(args)
    if early is not None:
        return early
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timer = start_timer()
    plan, host, server, pilot = load_plan(args)
    print(plan.render())
    if args.dry_run:
        print("\ndry run: nothing was started")
        return 0
    return _run_for_real(plan, args, (host, server, pilot), timer, root)


if __name__ == "__main__":
    raise SystemExit(main())
